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

    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_preproc_workers = num_preproc_workers
        self.upload_concurrency = upload_concurrency
        self.chunk_concurrency = chunk_concurrency

        self._session: aiohttp.ClientSession | None = None
        self._audio_cache: dict[str, str] = {}
        self._preproc_pool = ProcessPoolExecutor(max_workers=num_preproc_workers)

        # Per-request latency to the ASR server.
        self._request_latencies_s: list[float] = []

        # Per-chunk latency, where one chunk contains N windows.
        self._chunk_latencies_s: list[float] = []

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=600)
            connector = aiohttp.TCPConnector(limit=0)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

        for audio_path in self._audio_cache.values():
            if os.path.exists(audio_path):
                os.remove(audio_path)
        self._audio_cache.clear()

        self._preproc_pool.shutdown(wait=True, cancel_futures=True)

    async def _ensure_full_audio(self, video_path: str) -> str:
        cached = self._audio_cache.get(video_path)
        if cached and os.path.exists(cached):
            return cached

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

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
            raise RuntimeError(f"ffmpeg full-audio extraction failed for {video_path}")

        self._audio_cache[video_path] = audio_path
        print(f"[ASRClient] cached full audio for {video_path} -> {audio_path}")
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

        next_submit_idx = 0
        inflight_preproc: set[asyncio.Task] = set()
        upload_sem = asyncio.Semaphore(self.upload_concurrency)

        async def preprocess_one(idx: int, w: dict) -> None:
            t0 = float(w["t0"])
            t1 = float(w["t1"])

            item = await loop.run_in_executor(
                self._preproc_pool,
                _slice_audio_bytes,
                full_audio_path,
                t0,
                t1,
            )

            await queue.put((idx, item))

        def maybe_submit_more_preproc() -> None:
            nonlocal next_submit_idx

            buffered = queue.qsize() + len(inflight_preproc)
            if buffered >= low_watermark:
                return

            target = high_watermark
            while next_submit_idx < len(windows) and buffered < target:
                task = asyncio.create_task(preprocess_one(next_submit_idx, windows[next_submit_idx]))
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
                t0, t1, audio_bytes = sliced

                try:
                    async with upload_sem:
                        result = await self._upload_one(
                            t0=t0,
                            t1=t1,
                            audio_bytes=audio_bytes,
                        )
                except Exception as e:
                    print(f"[ASRClient] upload worker {worker_id} failed on [{t0:.2f}, {t1:.2f}]: {e}")
                    result = {
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

        while next_submit_idx < len(windows) or inflight_preproc or not queue.empty():
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
    chunk_size: int = 8,
) -> list[dict[str, Any]]:
        if not windows:
            return []

        window_chunks = self._chunk_windows(windows, chunk_size)

        print(
            f"[ASRClient] starting chunked transcription: "
            f"total_windows={len(windows)} chunk_size={chunk_size} "
            f"num_chunks={len(window_chunks)} chunk_concurrency={self.chunk_concurrency}"
        )

        sem = asyncio.Semaphore(self.chunk_concurrency)
        chunk_results_ordered: list[list[dict[str, Any]] | None] = [None] * len(window_chunks)

        async def run_one_chunk(chunk_idx: int, chunk: list[dict]) -> None:
            async with sem:
                print(
                    f"[ASRClient] running chunk {chunk_idx + 1}/{len(window_chunks)} "
                    f"with {len(chunk)} windows"
                )

                results, chunk_latency_s = await self._transcribe_one_chunk(
                    video_path=video_path,
                    windows=chunk,
                )

                print(
                    f"[ASRClient] finished chunk {chunk_idx + 1}/{len(window_chunks)} "
                    f"latency={chunk_latency_s:.3f}s"
                )

                chunk_results_ordered[chunk_idx] = results

        tasks = [
            asyncio.create_task(run_one_chunk(chunk_idx, chunk))
            for chunk_idx, chunk in enumerate(window_chunks)
        ]

        await asyncio.gather(*tasks)

        all_results: list[dict[str, Any]] = []
        for r in chunk_results_ordered:
            if r is not None:
                all_results.extend(r)

        return all_results

    async def transcribe_window(
        self,
        *,
        video_path: str,
        t0: float,
        t1: float,
    ) -> dict[str, Any]:
        results = await self.transcribe_windows_batch(
            video_path=video_path,
            windows=[{"t0": t0, "t1": t1}],
            chunk_size=1,
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