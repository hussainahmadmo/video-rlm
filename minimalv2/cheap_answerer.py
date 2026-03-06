from dataclasses import dataclass
from typing import List, Tuple, Optional
import requests
import json

@dataclass(frozen=True)
class TextAnswererConfig:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 256
    temperature: float = 0.0
    timeout_s: int = 60

class TextAnswerer:
    def __init__(self, cfg: TextAnswererConfig):
        self.cfg = cfg
        root = cfg.base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        self.endpoint = root + "/v1/chat/completions"

    def answer_with_confidence(self, *, question: str, evidence: str) -> tuple[str, float, dict]:
        prompt = (
            "Return STRICT JSON only: {\"answer\": <string>, \"confidence\": <0..1>}.\n"
            "Answer the question using ONLY the evidence. "
            "If evidence is insufficient, still guess but set confidence low.\n\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{evidence}\n"
        )

        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": "You answer questions from provided evidence. Output STRICT JSON only."},
                {"role": "user", "content": prompt},
            ],
        }

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.cfg.api_key}"}
        r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()

        # small extractor like your profiler's _extract_json
        i, j = text.find("{"), text.rfind("}")
        raw = json.loads(text[i:j+1]) if i >= 0 and j > i else json.loads(text)

        ans = str(raw.get("answer", "")).strip()
        conf = float(raw.get("confidence", 0.0) or 0.0)
        conf = max(0.0, min(1.0, conf))
        return ans, conf, raw