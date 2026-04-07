from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import aiohttp


def _slice_audio_bytes(full_audio_path: str, t0: float, t1: float) -> tuple[float, float, bytes]:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(t0),
        "-to", str(t1),
        "-i", full_audio_path,
        "-f", "wav",
        "-ac", "1",
        "-ar", "16000",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return (t0, t1, b"")
    return (t0, t1, proc.stdout)

class ASRClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        num_preproc_workers: int = 16,
        upload_concurrency: int = 32,
        chunk_concurrency: int = 2,
        enable_audio_cache: bool = True,
        enable_result_cache: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_preproc_workers = num_preproc_workers
        self.upload_concurrency = upload_concurrency
        self.chunk_concurrency = chunk_concurrency
        self.enable_audio_cache = enable_audio_cache

        self._session: aiohttp.ClientSession | None = None
        self._audio_cache: dict[str, str] = {}
        self._preproc_pool = ProcessPoolExecutor(max_workers=num_preproc_workers)

        self._request_latencies_s: list[float] = []
        self._chunk_latencies_s: list[float] = []

        self._audio_cache_hits = 0
        self._audio_cache_misses = 0
        self._full_audio_extract_time_s = 0.0
        
        self._temp_full_audio_files: set[str] = set()
        self.enable_result_cache = enable_result_cache
        self._result_cache: dict[tuple, dict[str, Any]] = {}
        self._result_cache_hits = 0
        self._result_cache_misses = 0
        self._model_calls = 0

        self._timing = {
        "audio_extract_s": 0.0,
        "request_chunk_expand_s": 0.0,
        "audio_slice_s": 0.0,
        "result_cache_lookup_s": 0.0,
        "result_cache_hit_return_s": 0.0,
        "model_inference_s": 0.0,
        "group_results_s": 0.0,
    }

    def _result_cache_key(self, video_path: str, t0: float, t1: float) -> tuple:
        return (
            video_path,
            round(t0, 3),
            round(t1, 3),
            self.model,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=600)
            connector = aiohttp.TCPConnector(limit=0)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

        for audio_path in self._temp_full_audio_files:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        self._temp_full_audio_files.clear()
        self._audio_cache.clear()

        self._preproc_pool.shutdown(wait=True, cancel_futures=True)
        

    async def _ensure_full_audio(self, video_path: str) -> str:
        if self.enable_audio_cache:
            cached = self._audio_cache.get(video_path)
            if cached and os.path.exists(cached):
                self._audio_cache_hits += 1
                return cached

        self._audio_cache_misses += 1
        t0 = time.time()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        self._temp_full_audio_files.add(audio_path)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            audio_path,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await proc.wait()
        if rc != 0:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            self._temp_full_audio_files.discard(audio_path)
            raise RuntimeError(f"ffmpeg full-audio extraction failed for {video_path}")
        

        t1 = time.time()
        dt = t1 - t0
        self._full_audio_extract_time_s += dt
        self._timing["audio_extract_s"] += dt

        if self.enable_audio_cache:
            self._audio_cache[video_path] = audio_path

        return audio_path
    
    def _split_window_into_request_chunks(
        self,
        window: dict,
        request_chunk_len_s: float,
    ) -> list[dict]:
        parent_t0 = float(window["t0"])
        parent_t1 = float(window["t1"])

        out = []
        cur = parent_t0
        while cur < parent_t1:
            nxt = min(parent_t1, cur + request_chunk_len_s)
            out.append({
                "parent_t0": parent_t0,
                "parent_t1": parent_t1,
                "t0": cur,
                "t1": nxt,
            })
            cur = nxt
        return out

    def _expand_windows_into_request_chunks(
        self,
        windows: list[dict],
        request_chunk_len_s: float,
    ) -> list[dict]:
        out = []
        for w in windows:
            out.extend(self._split_window_into_request_chunks(w, request_chunk_len_s))
        return out

    async def _upload_one(self, *, t0: float, t1: float, audio_bytes: bytes) -> dict[str, Any]:
        self._model_calls += 1
        if not audio_bytes:
            return {
                "t0": t0,
                "t1": t1,
                "transcript": "",
                "segments": [],
            }

        url = f"{self.base_url}/v1/audio/transcriptions"
        session = await self._get_session()

        data = aiohttp.FormData()
        data.add_field("model", self.model)
        data.add_field("temperature", "0")
        data.add_field("response_format", "json")
        data.add_field(
            "file",
            audio_bytes,
            filename=f"clip_{t0:.3f}_{t1:.3f}.wav",
            content_type="audio/wav",
        )

        t_req0 = time.time()
        async with session.post(url, data=data) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        t_req1 = time.time()

        latency_s = t_req1 - t_req0
        self._request_latencies_s.append(latency_s)
        #measure model inference.
        self._timing["model_inference_s"] += latency_s

        text = payload.get("text", "").strip()

        print(
            f"[ASRClient] upload [{t0:.2f}, {t1:.2f}] "
            f"request={latency_s:.3f}s "
            f"bytes={len(audio_bytes)} "
            f"text={repr(text[:80])}"
        )

        return {
            "t0": t0,
            "t1": t1,
            "transcript": text,
            "segments": [
                {
                    "start": t0,
                    "end": t1,
                    "text": text,
                }
            ] if text else [],
        }
    
    async def _transcribe_one_chunk(
        self,
        *,
        video_path: str,
        windows: list[dict],
) -> tuple[list[dict[str, Any]], float]:
        if not windows:
            return [], 0.0

        t0_all = time.time()
        full_audio_path = await self._ensure_full_audio(video_path)
        loop = asyncio.get_running_loop()

        queue_maxsize = max(32, self.upload_concurrency * 2)
        low_watermark = max(8, self.upload_concurrency // 2)
        high_watermark = max(low_watermark + 1, int(queue_maxsize * 0.75))

        queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        results: list[dict[str, Any] | None] = [None] * len(windows)

        # ---------------------------------------------------------
        # EARLY RESULT-CACHE LOOKUP:
        # fill results immediately for cache hits, and only send
        # misses through preprocessing + upload.
        # ---------------------------------------------------------
        miss_items: list[tuple[int, dict]] = []

        for idx, w in enumerate(windows):
            t0 = float(w["t0"])
            t1 = float(w["t1"])
            parent_t0 = float(w.get("parent_t0", w["t0"]))
            parent_t1 = float(w.get("parent_t1", w["t1"]))

            t_lookup0 = time.time()
            cache_key = self._result_cache_key(video_path, t0, t1)
            cache_hit = self.enable_result_cache and cache_key in self._result_cache
            t_lookup1 = time.time()
            self._timing["result_cache_lookup_s"] += (t_lookup1 - t_lookup0)

            if cache_hit:
                self._result_cache_hits += 1

                t_hit0 = time.time()
                cached = dict(self._result_cache[cache_key])
                t_hit1 = time.time()
                self._timing["result_cache_hit_return_s"] += (t_hit1 - t_hit0)

                cached["parent_t0"] = parent_t0
                cached["parent_t1"] = parent_t1
                results[idx] = cached
            else:
                self._result_cache_misses += 1
                miss_items.append((
                    idx,
                    {
                        "parent_t0": parent_t0,
                        "parent_t1": parent_t1,
                        "t0": t0,
                        "t1": t1,
                    },
                ))

        # If everything was a cache hit, return immediately.
        if not miss_items:
            t1_all = time.time()
            chunk_latency_s = t1_all - t0_all
            self._chunk_latencies_s.append(chunk_latency_s)

            final_results = [r for r in results if r is not None]

            print(
                f"[ASRClient] chunk summary windows={len(windows)} "
                f"all_cache_hits=True "
                f"chunk_latency={chunk_latency_s:.3f}s"
            )
            return final_results, chunk_latency_s

        next_submit_idx = 0
        inflight_preproc: set[asyncio.Task] = set()
        upload_sem = asyncio.Semaphore(self.upload_concurrency)

        async def preprocess_one(miss_pos: int, item_w: dict) -> None:
            idx, w = miss_items[miss_pos]
            t0 = float(w["t0"])
            t1 = float(w["t1"])
            parent_t0 = float(w["parent_t0"])
            parent_t1 = float(w["parent_t1"])

            t_slice0 = time.time()
            item = await loop.run_in_executor(
                self._preproc_pool,
                _slice_audio_bytes,
                full_audio_path,
                t0,
                t1,
            )
            t_slice1 = time.time()
            self._timing["audio_slice_s"] += (t_slice1 - t_slice0)

            _t0, _t1, audio_bytes = item

            await queue.put((
                idx,
                {
                    "parent_t0": parent_t0,
                    "parent_t1": parent_t1,
                    "t0": t0,
                    "t1": t1,
                    "audio_bytes": audio_bytes,
                },
            ))

        def maybe_submit_more_preproc() -> None:
            nonlocal next_submit_idx

            buffered = queue.qsize() + len(inflight_preproc)
            if buffered >= low_watermark:
                return

            target = high_watermark
            while next_submit_idx < len(miss_items) and buffered < target:
                miss_pos = next_submit_idx
                _, w = miss_items[miss_pos]

                task = asyncio.create_task(preprocess_one(miss_pos, w))
                inflight_preproc.add(task)
                task.add_done_callback(inflight_preproc.discard)

                next_submit_idx += 1
                buffered += 1

        async def upload_worker(worker_id: int) -> None:
            while True:
                item = await queue.get()

                if item is None:
                    queue.task_done()
                    return

                idx, sliced = item
                parent_t0 = sliced["parent_t0"]
                parent_t1 = sliced["parent_t1"]
                t0 = sliced["t0"]
                t1 = sliced["t1"]
                audio_bytes = sliced["audio_bytes"]

                try:
                    async with upload_sem:
                        result = await self._upload_one(
                            t0=t0,
                            t1=t1,
                            audio_bytes=audio_bytes,
                        )

                    if self.enable_result_cache:
                        cache_key = self._result_cache_key(video_path, t0, t1)
                        self._result_cache[cache_key] = dict(result)

                    result["parent_t0"] = parent_t0
                    result["parent_t1"] = parent_t1

                except Exception as e:
                    print(f"[ASRClient] upload worker {worker_id} failed on [{t0:.2f}, {t1:.2f}]: {e}")
                    result = {
                        "parent_t0": parent_t0,
                        "parent_t1": parent_t1,
                        "t0": t0,
                        "t1": t1,
                        "transcript": "",
                        "segments": [],
                    }

                results[idx] = result
                queue.task_done()
                maybe_submit_more_preproc()

        upload_workers = [
            asyncio.create_task(upload_worker(i))
            for i in range(self.upload_concurrency)
        ]

        t0_sched = time.time()
        maybe_submit_more_preproc()

        while next_submit_idx < len(miss_items) or inflight_preproc or not queue.empty():
            maybe_submit_more_preproc()

            if inflight_preproc:
                await asyncio.wait(
                    inflight_preproc,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            else:
                await asyncio.sleep(0.01)

        t1_sched = time.time()

        await queue.join()

        for _ in upload_workers:
            await queue.put(None)

        await asyncio.gather(*upload_workers)

        t1_all = time.time()
        chunk_latency_s = t1_all - t0_all
        self._chunk_latencies_s.append(chunk_latency_s)

        final_results = [r for r in results if r is not None]

        print(
            f"[ASRClient] chunk summary windows={len(windows)} "
            f"misses={len(miss_items)} "
            f"queue_maxsize={queue_maxsize} low={low_watermark} high={high_watermark} "
            f"schedule_loop={t1_sched - t0_sched:.3f}s "
            f"chunk_latency={chunk_latency_s:.3f}s"
        )

        return final_results, chunk_latency_s

    async def transcribe_windows_batch(
        self,
        *,
        video_path: str,
        windows: list[dict],
        request_chunk_len_s: float = 10.0,
    ) -> list[dict[str, Any]]:
        if not windows:
            return []

        t_expand0 = time.time()
        request_chunks = self._expand_windows_into_request_chunks(
            windows,
            request_chunk_len_s,
        )
        t_expand1 = time.time()
        self._timing["request_chunk_expand_s"] += (t_expand1 - t_expand0)

        print(
            f"[ASRClient] starting request-chunk transcription: "
            f"num_windows={len(windows)} "
            f"request_chunk_len_s={request_chunk_len_s} "
            f"num_request_chunks={len(request_chunks)}"
        )

        # Here, each request chunk is one request.
        chunk_results, _chunk_latency_s = await self._transcribe_one_chunk(
            video_path=video_path,
            windows=request_chunks,
        )

        t_group0 = time.time()
        grouped_results = self._group_request_chunks_back_to_windows(chunk_results)
        t_group1 = time.time()
        self._timing["group_results_s"] += (t_group1 - t_group0)

        return grouped_results

    def _group_request_chunks_back_to_windows(
        self,
        chunk_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}

        for r in chunk_results:
            key = (float(r["parent_t0"]), float(r["parent_t1"]))
            grouped.setdefault(key, []).append(r)

        out = []
        for (parent_t0, parent_t1), items in grouped.items():
            items = sorted(items, key=lambda x: x["t0"])
            joined = " ".join(
                str(x.get("transcript", "")).strip()
                for x in items
                if str(x.get("transcript", "")).strip()
            ).strip()

            out.append({
                "t0": parent_t0,
                "t1": parent_t1,
                "transcript": joined,
                "segments": items,
            })

        out.sort(key=lambda x: (x["t0"], x["t1"]))
        return out    

    async def transcribe_window(
        self,
        *,
        video_path: str,
        t0: float,
        t1: float,) -> dict[str, Any]:
        results = await self.transcribe_windows_batch(
            video_path=video_path,
            windows=[{"t0": t0, "t1": t1}],
            request_chunk_len_s=max(t1 - t0, 0.001),
        )
        return results[0] if results else {"transcript": "", "segments": []}

    def get_chunk_latency_summary(self) -> dict[str, float | int]:
        if not self._chunk_latencies_s:
            return {
                "count": 0,
                "mean_s": 0.0,
                "p50_s": 0.0,
                "p95_s": 0.0,
                "p99_s": 0.0,
                "max_s": 0.0,
            }

        xs = sorted(self._chunk_latencies_s)
        n = len(xs)

        def pct(p: float) -> float:
            if n == 1:
                return xs[0]
            idx = int(round((p / 100.0) * (n - 1)))
            return xs[idx]

        return {
            "count": n,
            "mean_s": sum(xs) / n,
            "p50_s": pct(50),
            "p95_s": pct(95),
            "p99_s": pct(99),
            "max_s": xs[-1],
        }

    def get_request_latency_summary(self) -> dict[str, float | int]:
        if not self._request_latencies_s:
            return {
                "count": 0,
                "mean_s": 0.0,
                "p50_s": 0.0,
                "p95_s": 0.0,
                "p99_s": 0.0,
                "max_s": 0.0,
            }

        xs = sorted(self._request_latencies_s)
        n = len(xs)

        def pct(p: float) -> float:
            if n == 1:
                return xs[0]
            idx = int(round((p / 100.0) * (n - 1)))
            return xs[idx]

        return {
            "count": n,
            "mean_s": sum(xs) / n,
            "p50_s": pct(50),
            "p95_s": pct(95),
            "p99_s": pct(99),
            "max_s": xs[-1],
        }
    
    def get_cache_summary(self) -> dict[str, float | int]:
        return {
            "enable_audio_cache": int(self.enable_audio_cache),
            "audio_cache_hits": self._audio_cache_hits,
            "audio_cache_misses": self._audio_cache_misses,
            "full_audio_extract_time_s": self._full_audio_extract_time_s,
        }
    
    def get_result_cache_summary(self) -> dict[str, float | int]:
        total_requests_seen = self._result_cache_hits + self._result_cache_misses
        hit_rate = 0.0 if total_requests_seen == 0 else self._result_cache_hits / total_requests_seen
        return {
            "result_cache_hits": self._result_cache_hits,
            "result_cache_misses": self._result_cache_misses,
            "result_cache_size": len(self._result_cache),
            "model_calls": self._model_calls,
            "total_requests_seen": total_requests_seen,
            "result_cache_hit_rate": hit_rate,
        }
    
    def reset_timing(self) -> None:
        for k in self._timing:
            self._timing[k] = 0.0

    def get_timing_summary(self) -> dict[str, float]:
        return dict(self._timing)