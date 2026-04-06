from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

from faster_whisper import WhisperModel


class ASRClient:
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    async def transcribe_window(self, *, video_path: str, t0: float, t1: float) -> dict[str, Any]:
        if t1 <= t0:
            return {"transcript": "", "segments": []}

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(t0),
                "-to",
                str(t1),
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                wav_path,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            rc = await proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed for window [{t0}, {t1}]")

            segments_iter, _info = self.model.transcribe(wav_path)

            segments = []
            transcript_parts = []

            for seg in segments_iter:
                text = seg.text.strip()
                if not text:
                    continue

                transcript_parts.append(text)
                segments.append(
                    {
                        "start": round(t0 + float(seg.start), 3),
                        "end": round(t0 + float(seg.end), 3),
                        "text": text,
                    }
                )

            return {
                "transcript": " ".join(transcript_parts).strip(),
                "segments": segments,
            }

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)