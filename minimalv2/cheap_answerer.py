from dataclasses import dataclass
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


def _extract_json(text: str) -> dict:
    text = text.strip()

    if not text:
        raise ValueError("Empty model output")

    try:
        return json.loads(text)
    except Exception:
        pass

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()

    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        candidate = text[i:j+1]

        try:
            return json.loads(candidate)
        except Exception:
            pass

        cleaned = (
            candidate
            .replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
        )
        return json.loads(cleaned)

    raise ValueError(f"Could not parse JSON from model output: {text[:500]}")


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
                {
                    "role": "system",
                    "content": "You answer questions from provided evidence. Output STRICT JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }

        r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.cfg.timeout_s)
        if r.status_code >= 400:
            print("CheapAnswerer status:", r.status_code)
            print("CheapAnswerer response:", r.text[:8000])
        r.raise_for_status()

        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()

        print("\n===== RAW CHEAP ANSWER OUTPUT =====")
        print(repr(text))
        print("===== END RAW CHEAP ANSWER OUTPUT =====\n")

        try:
            raw = _extract_json(text)
            ans = str(raw.get("answer", "")).strip()

            conf = raw.get("confidence", 0.0)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0

            conf = max(0.0, min(1.0, conf))
            return ans, conf, raw

        except Exception as e:
            print("Cheap answer parse failed:", e)
            return "", 0.0, {"error": str(e), "raw_text": text[:1000]}