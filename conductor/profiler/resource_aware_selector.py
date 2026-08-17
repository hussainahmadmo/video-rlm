from __future__ import annotations

import math
import pickle
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


TIER_ORDER = [
    "budget2",
    "scan0.0039_k8_budget32",
    "w4_k8_budget16",
    "budget32",
    "scan0.03125_k8_budget32",
]


CONFIG_TO_TIER = {
    "budget2": "budget2",
    "scan0.0039_k8_budget32": "scan0.0039_k8_budget32",
    "w4_k8_budget16": "w4_k8_budget16",
    "budget32": "budget32",
    "scan0.03125_k8_budget32": "scan0.03125_k8_budget32",
}


QUESTION_TERMS = [
    "after",
    "before",
    "compare",
    "first",
    "how",
    "key",
    "overall",
    "pattern",
    "primary",
    "process",
    "sequence",
    "summarize",
    "then",
    "why",
    "workflow",
]


def tier_index(config_name: str) -> int:
    tier = CONFIG_TO_TIER.get(config_name, config_name)
    if tier not in TIER_ORDER:
        raise KeyError(f"unknown selector tier/config: {config_name}")
    return TIER_ORDER.index(tier)


def clamp_tier(config_name: str) -> str:
    return CONFIG_TO_TIER.get(config_name, config_name)


def upgrade_tier(config_name: str, steps: int = 1) -> str:
    idx = tier_index(config_name)
    return TIER_ORDER[min(len(TIER_ORDER) - 1, idx + steps)]


def downgrade_tier(config_name: str, steps: int = 1) -> str:
    idx = tier_index(config_name)
    return TIER_ORDER[max(0, idx - steps)]


def text_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def feature_names() -> list[str]:
    names = [
        "bias",
        "duration_log",
        "duration_short",
        "duration_medium",
        "duration_long",
        "question_len_log",
        "choice_count",
        "avg_choice_len_log",
        "category_reasoning",
        "category_temporal",
        "category_sequence",
        "category_process",
    ]
    names.extend(f"term_{term}" for term in QUESTION_TERMS)
    return names


def extract_features(row: dict[str, Any]) -> dict[str, float]:
    question = str(row.get("question") or "")
    tokens = text_tokens(question)
    token_set = set(tokens)
    choices = row.get("choices") or []
    if not isinstance(choices, list):
        choices = []

    duration_s = safe_float(row.get("duration_s"))
    category = " ".join(
        str(row.get(key) or "").lower()
        for key in (
            "question_category",
            "topic_category",
            "vimio_profile",
            "reasoning_type",
            "scheduler_query_class",
        )
    )

    avg_choice_len = (
        mean(len(text_tokens(str(choice))) for choice in choices)
        if choices
        else 0.0
    )

    feats = {
        "bias": 1.0,
        "duration_log": math.log1p(max(duration_s, 0.0)),
        "duration_short": 1.0 if 0 < duration_s < 300 else 0.0,
        "duration_medium": 1.0 if 300 <= duration_s < 1200 else 0.0,
        "duration_long": 1.0 if duration_s >= 1200 else 0.0,
        "question_len_log": math.log1p(len(tokens)),
        "choice_count": float(len(choices)),
        "avg_choice_len_log": math.log1p(avg_choice_len),
        "category_reasoning": 1.0 if "reason" in category else 0.0,
        "category_temporal": 1.0 if "temporal" in category else 0.0,
        "category_sequence": 1.0 if "sequence" in category else 0.0,
        "category_process": 1.0 if "process" in category else 0.0,
    }

    for term in QUESTION_TERMS:
        feats[f"term_{term}"] = 1.0 if term in token_set else 0.0

    return feats


def vectorize(row: dict[str, Any], names: list[str] | None = None) -> list[float]:
    names = names or feature_names()
    feats = extract_features(row)
    return [float(feats.get(name, 0.0)) for name in names]


def fit_standardizer(vectors: list[list[float]]) -> tuple[list[float], list[float]]:
    if not vectors:
        raise ValueError("cannot fit selector with no training vectors")
    cols = list(zip(*vectors))
    means = [mean(col) for col in cols]
    stds = []
    for col, mu in zip(cols, means):
        var = mean((x - mu) ** 2 for x in col)
        stds.append(math.sqrt(var) or 1.0)
    return means, stds


def standardize(
    vector: list[float],
    means: list[float],
    stds: list[float],
) -> list[float]:
    return [
        (value - mu) / sd
        for value, mu, sd in zip(vector, means, stds)
    ]


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class ResourceAwareSelector:
    def __init__(self, model: dict[str, Any]):
        self.model = model
        self.features = list(model["features"])
        self.means = list(model["means"])
        self.stds = list(model["stds"])
        self.centroids = {
            label: list(values)
            for label, values in model["centroids"].items()
        }
        self.examples = [
            {
                "label": str(example["label"]),
                "vector": list(example["vector"]),
            }
            for example in model.get("examples", [])
        ]
        self.k_neighbors = int(model.get("k_neighbors") or 7)
        self.priors = dict(model.get("priors") or {})
        self.default_config = str(
            model.get("default_config")
            or "scan0.0039_k8_budget32"
        )

    @classmethod
    def load(cls, path: str | Path) -> "ResourceAwareSelector":
        with Path(path).open("rb") as handle:
            return cls(pickle.load(handle))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as handle:
            pickle.dump(self.model, handle)

    def predict_base_config(self, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if not self.centroids and not self.examples:
            return self.default_config, {
                "selector": "nearest_neighbor",
                "fallback": "no_examples",
            }

        raw = vectorize(row, self.features)
        vec = standardize(raw, self.means, self.stds)

        if self.examples:
            neighbors = []
            for example in self.examples:
                distance = euclidean(vec, example["vector"])
                neighbors.append((distance, example["label"]))
            neighbors.sort()

            votes: dict[str, float] = defaultdict(float)
            for distance, label in neighbors[: self.k_neighbors]:
                votes[label] += 1.0 / (distance + 1e-6)
            ranked = sorted(
                votes.items(),
                key=lambda item: (-item[1], tier_index(item[0])),
            )
            label = ranked[0][0]
            return label, {
                "selector": "nearest_neighbor",
                "k_neighbors": self.k_neighbors,
                "votes": [
                    {"config": key, "score": value}
                    for key, value in ranked
                ],
                "neighbors": [
                    {"config": label, "distance": distance}
                    for distance, label in neighbors[: self.k_neighbors]
                ],
            }

        scored = []
        for label, centroid in self.centroids.items():
            distance = euclidean(vec, centroid)
            prior = float(self.priors.get(label, 0.0))
            scored.append((distance - 0.05 * prior, distance, label))
        scored.sort()
        _, distance, label = scored[0]
        return label, {
            "selector": "nearest_centroid",
            "distance": distance,
            "scores": [
                {"config": label, "distance": dist}
                for _, dist, label in scored[:5]
            ],
        }

    def choose_config(
        self,
        row: dict[str, Any],
        gpu_state: dict[str, Any] | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        predicted, meta = self.predict_base_config(row)
        chosen, reason = apply_gpu_pressure(predicted, gpu_state or {})
        if chosen != predicted:
            reason = f"learned={predicted};{reason}"
        else:
            reason = f"learned={predicted};gpu_keep"
        return chosen, reason, meta


def train_nearest_centroid(
    rows: list[dict[str, Any]],
    *,
    label_key: str = "label_config",
    k_neighbors: int = 7,
) -> ResourceAwareSelector:
    names = feature_names()
    training = [
        row
        for row in rows
        if row.get(label_key) in TIER_ORDER
    ]
    if not training:
        raise ValueError(
            f"no rows with {label_key} in selector tier set"
        )

    labels = [str(row[label_key]) for row in training]
    vectors = [vectorize(row, names) for row in training]
    means, stds = fit_standardizer(vectors)
    standardized = [
        standardize(vector, means, stds)
        for vector in vectors
    ]

    by_label: dict[str, list[list[float]]] = defaultdict(list)
    for label, vector in zip(labels, standardized):
        by_label[label].append(vector)

    centroids = {}
    for label, label_vectors in by_label.items():
        cols = list(zip(*label_vectors))
        centroids[label] = [mean(col) for col in cols]

    counts = Counter(labels)
    total = sum(counts.values())
    priors = {
        label: count / total
        for label, count in counts.items()
    }
    default_config = counts.most_common(1)[0][0]

    return ResourceAwareSelector(
        {
            "model_type": "nearest_centroid_config_selector",
            "features": names,
            "means": means,
            "stds": stds,
            "centroids": centroids,
            "examples": [
                {"label": label, "vector": vector}
                for label, vector in zip(labels, standardized)
            ],
            "k_neighbors": k_neighbors,
            "priors": priors,
            "default_config": default_config,
            "label_key": label_key,
            "training_examples": len(training),
            "label_counts": dict(counts),
        }
    )


def query_gpu_state() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": repr(exc),
            "avg_gpu_util_pct": None,
            "avg_mem_used_pct": None,
            "gpus": [],
        }

    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        idx, util, mem_used, mem_total = parts
        try:
            mem_used_f = float(mem_used)
            mem_total_f = float(mem_total)
            gpus.append(
                {
                    "index": int(idx),
                    "gpu_util_pct": float(util),
                    "mem_used_mib": mem_used_f,
                    "mem_total_mib": mem_total_f,
                    "mem_used_pct": (
                        100.0 * mem_used_f / mem_total_f
                        if mem_total_f
                        else 0.0
                    ),
                }
            )
        except ValueError:
            continue

    if not gpus:
        return {
            "available": False,
            "error": "nvidia-smi returned no parseable GPUs",
            "avg_gpu_util_pct": None,
            "avg_mem_used_pct": None,
            "gpus": [],
        }

    return {
        "available": True,
        "error": None,
        "avg_gpu_util_pct": mean(gpu["gpu_util_pct"] for gpu in gpus),
        "avg_mem_used_pct": mean(gpu["mem_used_pct"] for gpu in gpus),
        "gpus": gpus,
    }


def apply_gpu_pressure(
    config_name: str,
    gpu_state: dict[str, Any],
    *,
    high_util_pct: float = 75.0,
    critical_util_pct: float = 92.0,
    high_mem_pct: float = 88.0,
) -> tuple[str, str]:
    config_name = clamp_tier(config_name)
    if not gpu_state.get("available"):
        return config_name, "gpu_unknown_keep"

    util = safe_float(gpu_state.get("avg_gpu_util_pct"))
    mem = safe_float(gpu_state.get("avg_mem_used_pct"))

    if util >= critical_util_pct or mem >= high_mem_pct:
        return "budget2", (
            f"gpu_critical util={util:.1f} mem={mem:.1f}"
        )
    if util >= high_util_pct:
        return min(
            config_name,
            "scan0.0039_k8_budget32",
            key=tier_index,
        ), f"gpu_high util={util:.1f} mem={mem:.1f}"
    return config_name, f"gpu_ok util={util:.1f} mem={mem:.1f}"
