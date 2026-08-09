from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from PIL import Image
from decord.video_reader import VideoReader
from openai import OpenAI

from transformers import (
    AutoModel,
    AutoProcessor,
)


FORBIDDEN_RUNTIME_KEYS = {
    "gold",
    "correct",
    "answer_label",
    "answer_idx",
    "oracle_correct",
    "oracle_selected_config",
    "execution_tier",
    "tier",
    "num_correct_configs",
    "num_configs_evaluated",
}

# ============================================================
# Agent actions
# ============================================================

VALID_ACTIONS = {
    "SEARCH_LOCAL",
    "SEARCH_BEFORE",
    "SEARCH_AFTER",
    "GLOBAL_SCAN",
    "INCREASE_DENSITY",
    "ANSWER",
}


# ============================================================
# State
# ============================================================

@dataclass
class Evidence:
    action: str
    query: str
    start_s: float
    end_s: float
    timestamps: list[float]
    observation: str
    prediction: str
    confidence: str
    latency_s: float


@dataclass
class AgentState:
    question: str
    choices: list[str]
    semantic_profile: dict[str, Any]
    duration_s: float

    evidence: list[Evidence] = field(
        default_factory=list
    )

    candidate_times: list[float] = field(
        default_factory=list
    )

    final_answer: str | None = None

    total_latency_s: float = 0.0

    global_scan_count: int = 0


# ============================================================
# Utilities
# ============================================================


def sanitize_runtime_profile(
    profile,
):
    if not profile:
        return {}

    return {
        key: value
        for key, value in profile.items()
        if key not in FORBIDDEN_RUNTIME_KEYS
    }


def assert_no_forbidden_keys(
    obj,
    path="runtime_input",
):
    if isinstance(obj, dict):

        for key, value in obj.items():

            if key in FORBIDDEN_RUNTIME_KEYS:
                raise RuntimeError(
                    f"Oracle leakage detected: "
                    f"{path}.{key}"
                )

            assert_no_forbidden_keys(
                value,
                f"{path}.{key}",
            )

    elif isinstance(obj, list):

        for index, value in enumerate(
            obj
        ):
            assert_no_forbidden_keys(
                value,
                f"{path}[{index}]",
            )

def load_jsonl(path):
    with open(path) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def normalize_answer(value):
    if value is None:
        return ""

    value = str(value).strip()

    match = re.search(
        r"\b([A-H])\b",
        value.upper(),
    )

    if match:
        return match.group(1)

    return value


def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")

        if text.lower().startswith(
            "json"
        ):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if (
        start >= 0
        and end > start
    ):
        try:
            return json.loads(
                text[start:end + 1]
            )
        except Exception:
            pass

    return None


def format_choices(choices):
    if not choices:
        return "No choices provided."

    return "\n".join(
        f"{chr(ord('A') + i)}. {choice}"
        for i, choice in enumerate(
            choices
        )
    )


def image_to_data_uri(
    image: Image.Image,
):
    buf = io.BytesIO()

    image.save(
        buf,
        format="JPEG",
        quality=85,
    )

    encoded = base64.b64encode(
        buf.getvalue()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


def action_counts(
    state: AgentState,
):
    counts = {}

    for evidence in state.evidence:
        counts[evidence.action] = (
            counts.get(
                evidence.action,
                0,
            )
            + 1
        )

    return counts


def same_range(
    a_start,
    a_end,
    b_start,
    b_end,
    tolerance_s=1.0,
):
    return (
        abs(a_start - b_start)
        <= tolerance_s
        and
        abs(a_end - b_end)
        <= tolerance_s
    )


def normalize_query(
    value: str,
):
    return " ".join(
        str(value)
        .strip()
        .lower()
        .split()
    )


# ============================================================
# Video backend
# ============================================================

class VideoBackend:

    def __init__(
        self,
        path: str,
    ):
        self.path = path

        self.vr = VideoReader(
            path
        )

        self.fps = float(
            self.vr.get_avg_fps()
        )

        self.num_frames = len(
            self.vr
        )

        self.duration_s = (
            self.num_frames
            / self.fps
        )

    def frame_at(
        self,
        timestamp_s: float,
    ):
        timestamp_s = max(
            0.0,
            min(
                timestamp_s,
                self.duration_s
                - 1.0 / self.fps,
            ),
        )

        index = int(
            round(
                timestamp_s
                * self.fps
            )
        )

        index = max(
            0,
            min(
                index,
                self.num_frames - 1,
            ),
        )

        array = (
            self.vr[index]
            .asnumpy()
        )

        return Image.fromarray(
            array
        )

    def sample_range(
        self,
        start_s: float,
        end_s: float,
        num_frames: int,
    ):
        start_s = max(
            0.0,
            start_s,
        )

        end_s = min(
            self.duration_s,
            end_s,
        )

        if end_s <= start_s:
            end_s = min(
                self.duration_s,
                start_s + 1.0,
            )

        timestamps = np.linspace(
            start_s,
            end_s,
            max(
                1,
                num_frames,
            ),
        )

        frames = [
            self.frame_at(
                float(t)
            )
            for t in timestamps
        ]

        return (
            [
                float(t)
                for t in timestamps
            ],
            frames,
        )


# ============================================================
# SigLIP retriever
#
# Used ONLY for SEARCH_LOCAL.
# ============================================================

class SigLIPRetriever:

    def __init__(
        self,
        model_name: str,
        device: str,
    ):
        self.device = device

        print(
            "Loading SigLIP:",
            model_name,
        )

        self.processor = (
            AutoProcessor
            .from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModel
            .from_pretrained(
                model_name
            )
            .eval()
            .to(device)
        )

    @torch.no_grad()
    def rank_frames(
        self,
        query: str,
        timestamps: list[float],
        frames: list[Image.Image],
        topk: int,
        batch_size: int = 4,
    ):
        if not frames:
            return []

        all_scores = []

        for start in range(
            0,
            len(frames),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(frames),
            )

            batch_frames = (
                frames[start:end]
            )

            inputs = self.processor(
                text=[
                    query
                ] * len(batch_frames),
                images=batch_frames,
                padding="max_length",
                return_tensors="pt",
            )

            inputs = {
                key: value.to(
                    self.device
                )
                for key, value
                in inputs.items()
            }

            outputs = self.model(
                **inputs
            )

            if hasattr(
                outputs,
                "logits_per_image",
            ):
                scores = (
                    outputs
                    .logits_per_image
                    .diagonal()
                )

            elif hasattr(
                outputs,
                "logits_per_text",
            ):
                scores = (
                    outputs
                    .logits_per_text
                    .diagonal()
                )

            else:
                raise RuntimeError(
                    "SigLIP output has no "
                    "logits_per_image/text"
                )

            all_scores.extend(
                scores
                .detach()
                .float()
                .cpu()
                .tolist()
            )

            del inputs
            del outputs
            del scores

            if str(
                self.device
            ).startswith("cuda"):
                torch.cuda.empty_cache()

        scores = np.asarray(
            all_scores,
            dtype=np.float32,
        )

        order = np.argsort(
            -scores
        )

        return [
            {
                "timestamp_s": (
                    timestamps[idx]
                ),
                "score": float(
                    scores[idx]
                ),
            }
            for idx in order[
                : min(
                    topk,
                    len(order),
                )
            ]
        ]


# ============================================================
# VLM evidence inspector
# ============================================================

class EvidenceInspector:

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ):
        self.client = client
        self.model = model

    def inspect(
        self,
        *,
        question: str,
        choices: list[str],
        timestamps: list[float],
        frames: list[Image.Image],
        action: str,
        query: str,
    ):
        if action in {
            "GLOBAL_SCAN",
            "GLOBAL_SCAN_CHUNK",
        }:

            inspection_instruction = """
This evidence comes from a uniform scan of the whole video.

The frames are chronological.

They were NOT selected by semantic retrieval.

This may be only one temporal chunk of the whole-video scan,
so describe what happens in THIS chunk precisely.

For workflow / summary questions:
identify the major stage or transition visible here.

For sequence / ordering questions:
preserve chronological order.

For counting / repetition questions:
note candidate occurrences, but do not infer an exact whole-video
count from one chunk.

For conceptual or metaphorical questions:
record any visually relevant event, text, scene, or context that
could help interpret the question.

Do not guess beyond the visible evidence.
"""

        elif action in {
            "SEARCH_BEFORE",
            "SEARCH_AFTER",
        }:

            inspection_instruction = """
This is a targeted temporal-relation inspection.

Pay close attention to chronological order and determine what
occurs immediately before or after the relevant anchor event.
"""

        elif action == "INCREASE_DENSITY":

            inspection_instruction = """
This range has already been identified as potentially useful.

You are now seeing it at higher temporal density.

Focus on details or transitions that may have been missed by
the earlier sparse inspection.
"""

        else:

            inspection_instruction = """
These frames correspond to a query-conditioned local search.

Determine whether the requested event, object, person, or
visual concept is actually present in this range.
"""

        content = []

        content.append({
            "type": "text",
            "text": f"""
You are inspecting video evidence for a question.

Question:
{question}

Choices:
{format_choices(choices)}

Action:
{action}

Search intent:
{query}

{inspection_instruction}

Frames are ordered chronologically.

Timestamps:
{timestamps}

Describe the useful evidence first.

Then give your current best multiple-choice answer only if the
current evidence supports one.

Return ONLY JSON:

{{
  "observation": "concise evidence description",
  "prediction": "A",
  "confidence": "low"
}}

prediction may be an empty string if evidence is insufficient.

confidence must be:
low | medium | high
""",
        })

        for timestamp, frame in zip(
            timestamps,
            frames,
        ):
            content.append({
                "type": "text",
                "text": (
                    f"Frame at "
                    f"{timestamp:.2f}s"
                ),
            })

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": (
                        image_to_data_uri(
                            frame
                        )
                    )
                },
            })

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                temperature=0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        text = (
            response
            .choices[0]
            .message.content
            .strip()
        )

        parsed = extract_json(
            text
        )

        if parsed is None:
            return {
                "observation": text,
                "prediction": "",
                "confidence": "low",
                "latency_s": latency,
            }

        confidence = str(
            parsed.get(
                "confidence",
                "low",
            )
        ).strip().lower()

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
            confidence = "low"

        return {
            "observation": str(
                parsed.get(
                    "observation",
                    "",
                )
            ),
            "prediction": (
                normalize_answer(
                    parsed.get(
                        "prediction",
                        "",
                    )
                )
            ),
            "confidence": (
                confidence
            ),
            "latency_s": (
                latency
            ),
        }

    def aggregate_window(
        self,
        *,
        question: str,
        choices: list[str],
        action: str,
        query: str,
        start_s: float,
        end_s: float,
        chunk_results: list[dict],
    ):
        """
        Text-only aggregation for a temporally localized window
        that had to be split into several multimodal requests.
        """

        summaries = []

        for index, result in enumerate(
            chunk_results,
            start=1,
        ):
            summaries.append({
                "chunk": index,
                "timestamps": (
                    result[
                        "timestamps"
                    ]
                ),
                "observation": (
                    result[
                        "observation"
                    ]
                ),
                "local_prediction": (
                    result[
                        "prediction"
                    ]
                ),
                "local_confidence": (
                    result[
                        "confidence"
                    ]
                ),
            })

        prompt = f"""
    You are combining chronological observations from ONE temporal
    window of a video.

    Question:
    {question}

    Choices:
    {format_choices(choices)}

    Action:
    {action}

    Search intent:
    {query}

    Temporal range:
    {start_s:.2f}s to {end_s:.2f}s

    The window was split into consecutive frame chunks because all
    frames could not fit into one multimodal request.

    Chunk evidence:
    {json.dumps(
        summaries,
        ensure_ascii=False,
    )}

    Reason across the chunks as ONE continuous temporal interval.

    Important:

    1. Chunks are chronological.

    2. Do not majority-vote the local predictions.

    3. Use the observations as the primary evidence.

    4. If action is SEARCH_AFTER:
    determine what occurs after the anchor.

    5. If action is SEARCH_BEFORE:
    determine what occurs before the anchor.

    6. If action is INCREASE_DENSITY:
    use the denser temporal evidence to resolve fine actions,
    transitions, gestures, ordering, or repeated events.

    7. If action is SEARCH_LOCAL:
    determine whether this temporal region actually contains
    the target event/object and what it implies for the answer.

    8. Compare the complete evidence against every answer choice.

    Return ONLY JSON:

    {{
    "observation": "combined evidence for this temporal window",
    "prediction": "A",
    "confidence": "medium"
    }}

    prediction may be empty if evidence is insufficient.

    confidence must be:
    low | medium | high
    """

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        text = (
            response
            .choices[0]
            .message.content
            .strip()
        )

        parsed = extract_json(
            text
        )

        if parsed is None:
            return {
                "observation": text,
                "prediction": "",
                "confidence": "low",
                "latency_s": latency,
            }

        confidence = str(
            parsed.get(
                "confidence",
                "low",
            )
        ).strip().lower()

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
            confidence = "low"

        return {
            "observation": str(
                parsed.get(
                    "observation",
                    "",
                )
            ),
            "prediction": (
                normalize_answer(
                    parsed.get(
                        "prediction",
                        "",
                    )
                )
            ),
            "confidence": (
                confidence
            ),
            "latency_s": (
                latency
            ),
        }
    def aggregate_global(
        self,
        *,
        question: str,
        choices: list[str],
        chunk_results: list[dict],
    ):
        """
        Text-only aggregation over the chronological observations
        produced by small multimodal GLOBAL_SCAN chunks.
        """

        summaries = []

        for index, result in enumerate(
            chunk_results,
            start=1,
        ):
            summaries.append({
                "chunk": index,
                "timestamps": (
                    result[
                        "timestamps"
                    ]
                ),
                "observation": (
                    result[
                        "observation"
                    ]
                ),
                "local_prediction": (
                    result[
                        "prediction"
                    ]
                ),
                "local_confidence": (
                    result[
                        "confidence"
                    ]
                ),
            })

        prompt = f"""
You are aggregating chronological evidence from one video.

Question:
{question}

Choices:
{format_choices(choices)}

The whole video was sampled uniformly and divided into
chronological chunks.

Here are the observations from those chunks:

{json.dumps(
    summaries,
    ensure_ascii=False,
)}

Reason across ALL chunks.

Important rules:

1. Chunks are in chronological order.

2. Do NOT majority-vote the local predictions.

3. The local predictions are weak hints only.
   Base the final answer primarily on the observations.

4. For workflow or sequence questions:
   reconstruct the major events from beginning to end.

5. For before/after questions:
   preserve temporal relations carefully.

6. For counting/repetition questions:
   combine candidate occurrences across chunks, but do not
   double-count the same occurrence.

7. For conceptual/metaphorical questions:
   identify the relevant evidence or context and infer the
   intended meaning from the supplied observations.

8. Compare the reconstructed evidence against EVERY answer
   choice.

Return ONLY JSON:

{{
  "observation": "combined chronological evidence",
  "prediction": "A",
  "confidence": "medium"
}}

prediction may be empty if the evidence is genuinely
insufficient.

confidence must be:
low | medium | high
"""

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        text = (
            response
            .choices[0]
            .message.content
            .strip()
        )

        parsed = extract_json(
            text
        )

        if parsed is None:
            return {
                "observation": text,
                "prediction": "",
                "confidence": "low",
                "latency_s": latency,
            }

        confidence = str(
            parsed.get(
                "confidence",
                "low",
            )
        ).strip().lower()

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
            confidence = "low"

        return {
            "observation": str(
                parsed.get(
                    "observation",
                    "",
                )
            ),
            "prediction": (
                normalize_answer(
                    parsed.get(
                        "prediction",
                        "",
                    )
                )
            ),
            "confidence": (
                confidence
            ),
            "latency_s": (
                latency
            ),
        }


# ============================================================
# Controller
# ============================================================

class VideoAgentController:

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ):
        self.client = client
        self.model = model

    def choose_action(
        self,
        state: AgentState,
    ):
        history = []

        for index, evidence in enumerate(
            state.evidence,
            start=1,
        ):
            history.append({
                "evidence_id": (
                    index - 1
                ),
                "action": (
                    evidence.action
                ),
                "query": (
                    evidence.query
                ),
                "range": [
                    evidence.start_s,
                    evidence.end_s,
                ],
                "timestamps": (
                    evidence.timestamps
                ),
                "observation": (
                    evidence.observation
                ),
                "prediction": (
                    evidence.prediction
                ),
                "confidence": (
                    evidence.confidence
                ),
            })

        counts = action_counts(
            state
        )

        prompt = f"""
You control an adaptive video-question answering agent.

Question:
{state.question}

Choices:
{format_choices(state.choices)}

Video duration:
{state.duration_s:.2f} seconds

Question profile:
{json.dumps(
    state.semantic_profile,
    ensure_ascii=False,
)}

Evidence acquired so far:
{json.dumps(
    history,
    ensure_ascii=False,
)}

Action usage:
{json.dumps(counts)}

Available actions:

SEARCH_LOCAL
    Query-conditioned semantic retrieval using SigLIP.
    Use this when a specific event/object/person must be
    located in time.

GLOBAL_SCAN
    Uniform temporal sampling across the ENTIRE video.
    GLOBAL_SCAN does NOT use semantic retrieval.
    Use for:
      summary,
      workflow,
      sequence/order,
      counting,
      repeated events,
      dispersed evidence,
      whole-video context.

SEARCH_BEFORE
    Inspect time immediately BEFORE an already located anchor.

SEARCH_AFTER
    Inspect time immediately AFTER an already located anchor.

INCREASE_DENSITY
    Reinspect a known temporal range using more frames.

ANSWER
    Return the final answer.

Rules:

1. Do not ANSWER until evidence is sufficient.

2. For summary/workflow/global/sequence/multi-event questions,
   prefer GLOBAL_SCAN first.

3. For localized questions, prefer SEARCH_LOCAL.

4. Multiple SEARCH_LOCAL actions are allowed when they use
   materially different search intents.

5. Do not repeat the exact same SEARCH_LOCAL query.

6. GLOBAL_SCAN first uses coarse uniform temporal coverage.

7. A second GLOBAL_SCAN is allowed only if more global temporal
   resolution is genuinely needed.

8. Do not use GLOBAL_SCAN more than twice.

9. After GLOBAL_SCAN:
      - ANSWER if evidence is sufficient.
      - INCREASE_DENSITY if a particular range is ambiguous.
      - SEARCH_LOCAL if a specific missing event must be found.
      - SEARCH_BEFORE / SEARCH_AFTER for temporal relations.

10. For after questions:
      locate the anchor first,
      then SEARCH_AFTER.

11. For before questions:
      locate the anchor first,
      then SEARCH_BEFORE.

12. For counting:
      GLOBAL_SCAN first.
      If candidate occurrences are unclear, densify relevant
      regions before answering.

13. Do not repeat an identical action over the same evidence
    without a clear reason.

14. INCREASE_DENSITY requires:
      start_s
      end_s
      query

15. SEARCH_BEFORE / SEARCH_AFTER require:
      anchor_s
      query

16. Prefer timestamps/ranges already present in evidence.

17. Never use ground truth.

Return ONLY JSON.

Examples:

{{
  "action": "GLOBAL_SCAN",
  "query": "reconstruct the major stages of the art workflow",
  "reason": "The question asks for whole-video sequence."
}}

{{
  "action": "SEARCH_LOCAL",
  "query": "two men sitting on stage",
  "reason": "Need to locate the relevant scene."
}}

{{
  "action": "SEARCH_LOCAL",
  "query": "left man raising his right hand",
  "reason": "Need a more specific anchor within the located scene."
}}

{{
  "action": "SEARCH_AFTER",
  "anchor_s": 64.2,
  "query": "what happens immediately after the hand is lowered",
  "reason": "The anchor has been localized."
}}

{{
  "action": "INCREASE_DENSITY",
  "start_s": 40.0,
  "end_s": 56.0,
  "query": "clarify the transition in this interval",
  "reason": "Sparse evidence was ambiguous."
}}

{{
  "action": "ANSWER",
  "answer": "C",
  "reason": "The accumulated evidence supports C."
}}
"""

        response = (
            self.client
            .chat.completions
            .create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )
        )

        text = (
            response
            .choices[0]
            .message.content
            .strip()
        )

        parsed = extract_json(
            text
        )

        if parsed is None:
            return {
                "action": "GLOBAL_SCAN",
                "query": (
                    state.question
                ),
                "reason": (
                    "controller parse failure"
                ),
            }

        action = str(
            parsed.get(
                "action",
                "GLOBAL_SCAN",
            )
        ).upper()

        if action not in VALID_ACTIONS:
            action = "GLOBAL_SCAN"

        parsed["action"] = (
            action
        )

        return parsed

    def assess_sufficiency(
        self,
        state: AgentState,
    ):
        history = []

        for i, evidence in enumerate(
            state.evidence
        ):
            history.append({
                "evidence_id": i,
                "action": evidence.action,
                "range": [
                    evidence.start_s,
                    evidence.end_s,
                ],
                "observation":
                    evidence.observation,
                "prediction":
                    evidence.prediction,
                "confidence":
                    evidence.confidence,
            })

        prompt = f"""
    You are evaluating whether the current VIDEO EVIDENCE is
    sufficient to distinguish the answer choices.

    Question:
    {state.question}

    Choices:
    {format_choices(state.choices)}

    Evidence:
    {json.dumps(
        history,
        ensure_ascii=False,
    )}

    Do not simply trust previous predictions.

    Compare the actual visual observations against EVERY answer
    choice.

    Many answer choices may agree on the general activity but
    differ in one small visual detail. General agreement is NOT
    sufficient.

    Examples of discriminating facts:

    - laptop vs cloth vs art board
    - pointing toward another person vs toward the stage
    - reaching into a box vs cutting it with a knife
    - child vs woman vs man holding an object
    - black shirt vs black scrubs

    Determine:

    1. Which answer choices remain plausible?
    2. Does the current evidence distinguish them?
    3. If not, what exact visual fact is missing?
    4. Should we search for that fact elsewhere or inspect an
    existing range more densely?

    Return ONLY JSON:

    {{
    "sufficient": false,
    "best_answer": "",
    "plausible_choices": ["A", "D"],
    "missing_visual_fact":
        "whether the young man uses a knife to cut the box",
    "recommended_action": "SEARCH_LOCAL",
    "recommended_query":
        "young man using a knife to cut cardboard box",
    "evidence_id": null
    }}

    For an existing promising range:

    {{
    "sufficient": false,
    "best_answer": "",
    "plausible_choices": ["A", "D"],
    "missing_visual_fact":
        "direction of the man's pointing gesture",
    "recommended_action": "INCREASE_DENSITY",
    "recommended_query":
        "determine whether he points toward the man or stage",
    "evidence_id": 1
    }}

    If the evidence genuinely distinguishes the choices:

    {{
    "sufficient": true,
    "best_answer": "D",
    "plausible_choices": ["D"],
    "missing_visual_fact": "",
    "recommended_action": "ANSWER",
    "recommended_query": "",
    "evidence_id": null
    }}
    """

        response = (
            self.client
            .chat.completions
            .create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )
        )

        parsed = extract_json(
            response
            .choices[0]
            .message.content
        )

        return parsed
# ============================================================
# Agent
# ============================================================


class VideoAgent:

    def __init__(
        self,
        *,
        controller,
        inspector,
        retriever,
        coarse_fps: float,
        local_window_s: float,
        temporal_window_s: float,
        coarse_topk: int,
        inspect_frames: int,
        dense_frames: int,
        max_rounds: int,
        global_chunk_size: int,
    ):
        self.controller = controller
        self.inspector = inspector
        self.retriever = retriever

        self.coarse_fps = coarse_fps
        self.local_window_s = local_window_s
        self.temporal_window_s = temporal_window_s
        self.coarse_topk = coarse_topk
        self.inspect_frames = inspect_frames
        self.dense_frames = dense_frames
        self.max_rounds = max_rounds
        self.global_chunk_size = global_chunk_size

    # ========================================================
    # SEARCH_LOCAL
    # ========================================================

    def evidence_relevance_score(
        self,
        evidence: Evidence,
    ):
        score = 0

        if evidence.prediction:
            score += 2

        if evidence.confidence == "high":
            score += 2
        elif evidence.confidence == "medium":
            score += 1

        observation = (
            evidence.observation
            .lower()
        )

        negative_phrases = [
            "no evidence",
            "not seen",
            "not present",
            "does not appear",
            "no mention",
            "not visible",
            "cannot identify",
        ]

        if any(
            phrase in observation
            for phrase in negative_phrases
        ):
            score -= 3

        return score
    def local_search(
        self,
        *,
        video,
        query,
    ):
        duration = video.duration_s

        count = max(
            2,
            min(
                64,
                int(
                    math.ceil(
                        duration
                        * self.coarse_fps
                    )
                ),
            ),
        )

        timestamps, frames = (
            video.sample_range(
                0.0,
                duration,
                count,
            )
        )

        return self.retriever.rank_frames(
            query=query,
            timestamps=timestamps,
            frames=frames,
            topk=self.coarse_topk,
        )

    # ========================================================
    # GLOBAL_SCAN
    # ========================================================

    def uniform_global_scan(
        self,
        *,
        video,
        num_frames: int,
    ):
        return video.sample_range(
            0.0,
            video.duration_s,
            num_frames,
        )

    def choose_global_frame_budget(
        self,
        *,
        duration_s: float,
        scan_index: int,
    ):
        if scan_index <= 1:

            if duration_s <= 60:
                return 16

            if duration_s <= 300:
                return 16

            if duration_s <= 1200:
                return 24

            return 32

        if duration_s <= 300:
            return 32

        if duration_s <= 1200:
            return 40

        return 48

    def choose_window_chunk_size(
        self,
        video,
    ):
        """
        Short videos benefit from seeing more consecutive
        frames jointly.

        Long videos keep small chunks to avoid context
        overflow.
        """

        if video.duration_s <= 60:
            return 8

        if video.duration_s <= 300:
            return 6

        return self.global_chunk_size

    def inspect_global_scan(
        self,
        *,
        video,
        state,
        query,
        num_frames,
    ):
        timestamps, frames = (
            self.uniform_global_scan(
                video=video,
                num_frames=num_frames,
            )
        )

        print(
            "    uniform global "
            f"frames={len(timestamps)}"
        )

        print(
            "    timestamps:",
            [
                round(t, 2)
                for t in timestamps
            ],
        )

        chunk_results = []
        chunk_latency = 0.0

        for start in range(
            0,
            len(frames),
            self.global_chunk_size,
        ):
            end = min(
                start
                + self.global_chunk_size,
                len(frames),
            )

            chunk_timestamps = (
                timestamps[start:end]
            )

            chunk_frames = (
                frames[start:end]
            )

            print(
                "    global chunk:",
                start,
                "->",
                end,
                "timestamps=",
                [
                    round(t, 2)
                    for t
                    in chunk_timestamps
                ],
            )

            chunk_result = (
                self.inspector.inspect(
                    question=state.question,
                    choices=state.choices,
                    timestamps=chunk_timestamps,
                    frames=chunk_frames,
                    action="GLOBAL_SCAN_CHUNK",
                    query=query,
                )
            )

            chunk_results.append({
                "timestamps": (
                    chunk_timestamps
                ),
                "observation": (
                    chunk_result[
                        "observation"
                    ]
                ),
                "prediction": (
                    chunk_result[
                        "prediction"
                    ]
                ),
                "confidence": (
                    chunk_result[
                        "confidence"
                    ]
                ),
            })

            chunk_latency += float(
                chunk_result[
                    "latency_s"
                ]
            )

        combined = (
            self.inspector
            .aggregate_global(
                question=state.question,
                choices=state.choices,
                chunk_results=(
                    chunk_results
                ),
            )
        )

        total_latency = (
            chunk_latency
            + float(
                combined[
                    "latency_s"
                ]
            )
        )

        evidence = Evidence(
            action="GLOBAL_SCAN",
            query=query,
            start_s=0.0,
            end_s=float(
                video.duration_s
            ),
            timestamps=timestamps,
            observation=(
                combined[
                    "observation"
                ]
            ),
            prediction=(
                combined[
                    "prediction"
                ]
            ),
            confidence=(
                combined[
                    "confidence"
                ]
            ),
            latency_s=(
                total_latency
            ),
        )

        state.evidence.append(
            evidence
        )

        state.total_latency_s += (
            total_latency
        )

        return evidence

    # ========================================================
    # WINDOW INSPECTION
    # ========================================================

    def inspect_window(
        self,
        *,
        video,
        state,
        action,
        query,
        start_s,
        end_s,
        num_frames,
    ):
        start_s = max(
            0.0,
            float(start_s),
        )

        end_s = min(
            video.duration_s,
            float(end_s),
        )

        timestamps, frames = (
            video.sample_range(
                start_s,
                end_s,
                num_frames,
            )
        )

        chunk_size = (
            self.choose_window_chunk_size(
                video
            )
        )

        # ----------------------------------------------------
        # One multimodal request if it fits.
        # ----------------------------------------------------

        if len(frames) <= chunk_size:

            result = (
                self.inspector.inspect(
                    question=state.question,
                    choices=state.choices,
                    timestamps=timestamps,
                    frames=frames,
                    action=action,
                    query=query,
                )
            )

            total_latency = float(
                result[
                    "latency_s"
                ]
            )

        # ----------------------------------------------------
        # Otherwise split chronologically and aggregate.
        # ----------------------------------------------------

        else:

            chunk_results = []
            chunk_latency = 0.0

            for chunk_start in range(
                0,
                len(frames),
                chunk_size,
            ):
                chunk_end = min(
                    chunk_start
                    + chunk_size,
                    len(frames),
                )

                chunk_timestamps = (
                    timestamps[
                        chunk_start:
                        chunk_end
                    ]
                )

                chunk_frames = (
                    frames[
                        chunk_start:
                        chunk_end
                    ]
                )

                print(
                    "    window chunk:",
                    chunk_start,
                    "->",
                    chunk_end,
                    "timestamps=",
                    [
                        round(t, 2)
                        for t
                        in chunk_timestamps
                    ],
                )

                chunk_result = (
                    self.inspector.inspect(
                        question=(
                            state.question
                        ),
                        choices=(
                            state.choices
                        ),
                        timestamps=(
                            chunk_timestamps
                        ),
                        frames=(
                            chunk_frames
                        ),
                        action=action,
                        query=query,
                    )
                )

                chunk_results.append({
                    "timestamps": (
                        chunk_timestamps
                    ),
                    "observation": (
                        chunk_result[
                            "observation"
                        ]
                    ),
                    "prediction": (
                        chunk_result[
                            "prediction"
                        ]
                    ),
                    "confidence": (
                        chunk_result[
                            "confidence"
                        ]
                    ),
                })

                chunk_latency += float(
                    chunk_result[
                        "latency_s"
                    ]
                )

            combined = (
                self.inspector
                .aggregate_window(
                    question=(
                        state.question
                    ),
                    choices=(
                        state.choices
                    ),
                    action=action,
                    query=query,
                    start_s=start_s,
                    end_s=end_s,
                    chunk_results=(
                        chunk_results
                    ),
                )
            )

            result = combined

            total_latency = (
                chunk_latency
                + float(
                    combined[
                        "latency_s"
                    ]
                )
            )

        evidence = Evidence(
            action=action,
            query=query,
            start_s=start_s,
            end_s=end_s,
            timestamps=timestamps,
            observation=(
                result[
                    "observation"
                ]
            ),
            prediction=(
                result[
                    "prediction"
                ]
            ),
            confidence=(
                result[
                    "confidence"
                ]
            ),
            latency_s=(
                total_latency
            ),
        )

        state.evidence.append(
            evidence
        )

        state.total_latency_s += (
            total_latency
        )

        return evidence

    # ========================================================
    # HELPERS
    # ========================================================

    def last_nonempty_prediction(
        self,
        state,
    ):
        for evidence in reversed(
            state.evidence
        ):
            if evidence.prediction:
                return (
                    evidence.prediction
                )

        return ""

    def find_unrefined_local_evidence(
        self,
        state,
    ):
        candidates = [
            evidence
            for evidence
            in state.evidence
            if evidence.action
            in {
                "SEARCH_LOCAL",
                "SEARCH_BEFORE",
                "SEARCH_AFTER",
            }
        ]

        for candidate in reversed(
            candidates
        ):
            already_dense = any(
                (
                    evidence.action
                    == "INCREASE_DENSITY"
                    and same_range(
                        evidence.start_s,
                        evidence.end_s,
                        candidate.start_s,
                        candidate.end_s,
                    )
                )
                for evidence
                in state.evidence
            )

            if not already_dense:
                return candidate

        return None

    def resolve_anchor_evidence(
        self,
        state,
        decision,
    ):
        """
        Temporal operators must point to evidence already
        discovered by the agent.

        Never trust an arbitrary timestamp invented by the
        controller.
        """

        evidence_id = decision.get(
            "evidence_id"
        )

        if evidence_id is None:
            return None

        try:
            evidence_id = int(
                evidence_id
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

        if not (
            0
            <= evidence_id
            < len(state.evidence)
        ):
            return None

        return state.evidence[
            evidence_id
        ]

    def choose_best_local_for_refinement(
        self,
        state,
    ):
        local = [
            evidence
            for evidence
            in state.evidence
            if evidence.action
            == "SEARCH_LOCAL"
        ]

        if not local:
            return None

        confidence_rank = {
            "low": 0,
            "medium": 1,
            "high": 2,
        }

        return min(
            local,
            key=lambda evidence: (
                confidence_rank.get(
                    evidence.confidence,
                    0,
                )
            ),
        )

    # ========================================================
    # MAIN AGENT LOOP
    # ========================================================

    def run(
        self,
        *,
        video,
        question,
        choices,
        semantic_profile,
    ):
        state = AgentState(
            question=question,
            choices=choices or [],
            semantic_profile=(
                semantic_profile
                or {}
            ),
            duration_s=(
                video.duration_s
            ),
        )

        trajectory = []

        max_global_scans = (
            1
            if video.duration_s <= 60
            else 2
        )

        for round_idx in range(
            self.max_rounds
        ):
            controller_start = (
                time.time()
            )

            decision = (
                self.controller
                .choose_action(
                    state
                )
            )

            controller_latency = (
                time.time()
                - controller_start
            )

            state.total_latency_s += (
                controller_latency
            )

            # =================================================
            # Before accepting ANSWER, check whether the
            # evidence actually distinguishes the choices.
            # =================================================

            if (
                decision.get("action")
                == "ANSWER"
                and state.evidence
            ):
                sufficiency = (
                    self.controller
                    .assess_sufficiency(
                        state
                    )
                )

                if sufficiency:
                    print(
                        "    sufficiency:",
                        sufficiency,
                    )

                    if not sufficiency.get(
                        "sufficient",
                        False,
                    ):
                        recommended = str(
                            sufficiency.get(
                                "recommended_action",
                                "SEARCH_LOCAL",
                            )
                        ).upper()

                        if (
                            recommended
                            == "INCREASE_DENSITY"
                        ):
                            evidence_id = (
                                sufficiency.get(
                                    "evidence_id"
                                )
                            )

                            if (
                                evidence_id is not None
                                and 0
                                <= int(evidence_id)
                                < len(state.evidence)
                            ):
                                target = (
                                    state.evidence[
                                        int(evidence_id)
                                    ]
                                )

                                decision = {
                                    "action":
                                        "INCREASE_DENSITY",

                                    "start_s":
                                        target.start_s,

                                    "end_s":
                                        target.end_s,

                                    "query":
                                        sufficiency.get(
                                            "recommended_query",
                                            sufficiency.get(
                                                "missing_visual_fact",
                                                question,
                                            ),
                                        ),

                                    "reason":
                                        sufficiency.get(
                                            "missing_visual_fact",
                                            "Need discriminative evidence.",
                                        ),
                                }

                            else:
                                decision = {
                                    "action":
                                        "SEARCH_LOCAL",

                                    "query":
                                        sufficiency.get(
                                            "recommended_query",
                                            sufficiency.get(
                                                "missing_visual_fact",
                                                question,
                                            ),
                                        ),

                                    "reason":
                                        sufficiency.get(
                                            "missing_visual_fact",
                                            "Need discriminative evidence.",
                                        ),
                                }

                        else:
                            decision = {
                                "action":
                                    "SEARCH_LOCAL",

                                "query":
                                    sufficiency.get(
                                        "recommended_query",
                                        sufficiency.get(
                                            "missing_visual_fact",
                                            question,
                                        ),
                                    ),

                                "reason":
                                    sufficiency.get(
                                        "missing_visual_fact",
                                        "Need discriminative evidence.",
                                    ),
                            }

                    else:
                        decision[
                            "answer"
                        ] = normalize_answer(
                            sufficiency.get(
                                "best_answer",
                                decision.get(
                                    "answer",
                                    "",
                                ),
                            )
                        )

            action = decision[
                "action"
            ]

            # =================================================
            # Temporal operations REQUIRE prior evidence.
            # =================================================

            if action in {
                "SEARCH_BEFORE",
                "SEARCH_AFTER",
            }:
                anchor_evidence = (
                    self.resolve_anchor_evidence(
                        state,
                        decision,
                    )
                )

                if anchor_evidence is None:

                    print(
                        "    temporal action without valid "
                        "evidence_id -> forcing SEARCH_LOCAL"
                    )

                    action = (
                        "SEARCH_LOCAL"
                    )

                    decision[
                        "action"
                    ] = action

                    decision[
                        "query"
                    ] = (
                        "locate the anchor event "
                        "referenced by the temporal "
                        "question"
                    )

                    decision[
                        "reason"
                    ] = (
                        "Before/after reasoning requires "
                        "a previously observed anchor."
                    )

            # =================================================
            # Duplicate SEARCH_LOCAL guard.
            # =================================================

            if action == "SEARCH_LOCAL":

                proposed_query = (
                    normalize_query(
                        decision.get(
                            "query",
                            question,
                        )
                    )
                )

                previous_queries = {
                    normalize_query(
                        evidence.query
                    )
                    for evidence
                    in state.evidence
                    if evidence.action
                    == "SEARCH_LOCAL"
                }

                if (
                    proposed_query
                    in previous_queries
                ):
                    candidate = (
                        self
                        .choose_best_local_for_refinement(
                            state
                        )
                    )

                    if candidate is not None:

                        print(
                            "    duplicate SEARCH_LOCAL "
                            "-> refining existing candidate"
                        )

                        action = (
                            "INCREASE_DENSITY"
                        )

                        decision[
                            "action"
                        ] = action

                        decision[
                            "start_s"
                        ] = (
                            candidate.start_s
                        )

                        decision[
                            "end_s"
                        ] = (
                            candidate.end_s
                        )

                        decision[
                            "query"
                        ] = (
                            "inspect this previously "
                            "retrieved candidate more "
                            "densely to distinguish the "
                            "remaining answer choices"
                        )

                        decision[
                            "reason"
                        ] = (
                            "The same semantic retrieval "
                            "query has already been used."
                        )

                    else:

                        print(
                            "    duplicate SEARCH_LOCAL "
                            "without local evidence "
                            "-> GLOBAL_SCAN"
                        )

                        action = (
                            "GLOBAL_SCAN"
                        )

                        decision[
                            "action"
                        ] = action

                        decision[
                            "query"
                        ] = question

            # =================================================
            # Limit GLOBAL_SCAN.
            # =================================================

            if (
                action
                == "GLOBAL_SCAN"
                and state.global_scan_count
                >= max_global_scans
            ):
                candidate = (
                    self
                    .find_unrefined_local_evidence(
                        state
                    )
                )

                if candidate is not None:

                    print(
                        "    GLOBAL_SCAN limit reached "
                        "-> INCREASE_DENSITY"
                    )

                    action = (
                        "INCREASE_DENSITY"
                    )

                    decision[
                        "action"
                    ] = action

                    decision[
                        "start_s"
                    ] = (
                        candidate.start_s
                    )

                    decision[
                        "end_s"
                    ] = (
                        candidate.end_s
                    )

                    decision[
                        "query"
                    ] = (
                        "inspect this promising "
                        "region more densely to "
                        "resolve the answer choices"
                    )

                else:

                    print(
                        "    GLOBAL_SCAN limit reached "
                        "-> ANSWER"
                    )

                    action = (
                        "ANSWER"
                    )

                    decision[
                        "action"
                    ] = action

                    decision[
                        "answer"
                    ] = (
                        self
                        .last_nonempty_prediction(
                            state
                        )
                    )

            # =================================================
            # Duplicate INCREASE_DENSITY guard.
            # =================================================

            if action == "INCREASE_DENSITY":

                requested_start = float(
                    decision.get(
                        "start_s",
                        0.0,
                    )
                )

                requested_end = float(
                    decision.get(
                        "end_s",
                        min(
                            video.duration_s,
                            requested_start
                            + self.local_window_s,
                        ),
                    )
                )

                already_dense = any(
                    (
                        evidence.action
                        == "INCREASE_DENSITY"
                        and same_range(
                            evidence.start_s,
                            evidence.end_s,
                            requested_start,
                            requested_end,
                        )
                    )
                    for evidence
                    in state.evidence
                )

                if already_dense:

                    print(
                        "    duplicate density "
                        "range -> ANSWER"
                    )

                    action = (
                        "ANSWER"
                    )

                    decision[
                        "action"
                    ] = action

                    decision[
                        "answer"
                    ] = (
                        self
                        .last_nonempty_prediction(
                            state
                        )
                    )

            trajectory.append({
                "round": (
                    round_idx + 1
                ),
                "decision": (
                    decision
                ),
                "controller_latency_s": (
                    controller_latency
                ),
            })

            print(
                f"  round={round_idx + 1} "
                f"action={action}"
            )

            # =================================================
            # ANSWER
            # =================================================

            if action == "ANSWER":

                answer = (
                    normalize_answer(
                        decision.get(
                            "answer",
                            "",
                        )
                    )
                )

                if not answer:
                    answer = (
                        self
                        .last_nonempty_prediction(
                            state
                        )
                    )

                state.final_answer = answer

                break

            # =================================================
            # SEARCH_LOCAL
            # =================================================

            if action == "SEARCH_LOCAL":

                query = str(
                    decision.get(
                        "query",
                        question,
                    )
                )

                ranked = (
                    self.local_search(
                        video=video,
                        query=query,
                    )
                )

                if not ranked:
                    continue

                num_candidates = min(
                    3,
                    len(ranked),
                )

                print(
                    "    local candidates:",
                    [
                        round(
                            float(
                                result[
                                    "timestamp_s"
                                ]
                            ),
                            2,
                        )
                        for result
                        in ranked[
                            :num_candidates
                        ]
                    ],
                )

                for (
                    candidate_rank,
                    candidate,
                ) in enumerate(
                    ranked[
                        :num_candidates
                    ],
                    start=1,
                ):
                    anchor = float(
                        candidate[
                            "timestamp_s"
                        ]
                    )

                    half = (
                        self.local_window_s
                        / 2.0
                    )

                    start_s = max(
                        0.0,
                        anchor - half,
                    )

                    end_s = min(
                        video.duration_s,
                        anchor + half,
                    )

                    print(
                        "    inspect candidate "
                        f"{candidate_rank}: "
                        f"{start_s:.2f} "
                        f"-> {end_s:.2f}"
                    )

                    # IMPORTANT:
                    # Keep the original semantic query.
                    # Do NOT append "(retrieval candidate N)".
                    self.inspect_window(
                        video=video,
                        state=state,
                        action="SEARCH_LOCAL",
                        query=query,
                        start_s=start_s,
                        end_s=end_s,
                        num_frames=(
                            self.inspect_frames
                        ),
                    )

                continue

            # =================================================
            # GLOBAL_SCAN
            # =================================================

            if action == "GLOBAL_SCAN":

                state.global_scan_count += 1

                query = str(
                    decision.get(
                        "query",
                        question,
                    )
                )

                global_num_frames = (
                    self
                    .choose_global_frame_budget(
                        duration_s=(
                            video.duration_s
                        ),
                        scan_index=(
                            state.global_scan_count
                        ),
                    )
                )

                self.inspect_global_scan(
                    video=video,
                    state=state,
                    query=query,
                    num_frames=(
                        global_num_frames
                    ),
                )

                continue

            # =================================================
            # SEARCH_AFTER
            # =================================================

            if action == "SEARCH_AFTER":

                anchor_evidence = (
                    self.resolve_anchor_evidence(
                        state,
                        decision,
                    )
                )

                if anchor_evidence is None:
                    continue

                # Search immediately after the evidence
                # interval containing the anchor.
                anchor = float(
                    anchor_evidence.end_s
                )

                start_s = anchor

                end_s = min(
                    video.duration_s,
                    anchor
                    + self.temporal_window_s,
                )

                if end_s <= start_s:
                    continue

                query = str(
                    decision.get(
                        "query",
                        "what happens immediately "
                        "after the anchor event",
                    )
                )

                self.inspect_window(
                    video=video,
                    state=state,
                    action="SEARCH_AFTER",
                    query=query,
                    start_s=start_s,
                    end_s=end_s,
                    num_frames=(
                        self.dense_frames
                    ),
                )

                continue

            # =================================================
            # SEARCH_BEFORE
            # =================================================

            if action == "SEARCH_BEFORE":

                anchor_evidence = (
                    self.resolve_anchor_evidence(
                        state,
                        decision,
                    )
                )

                if anchor_evidence is None:
                    continue

                anchor = float(
                    anchor_evidence.start_s
                )

                start_s = max(
                    0.0,
                    anchor
                    - self.temporal_window_s,
                )

                end_s = anchor

                if end_s <= start_s:
                    continue

                query = str(
                    decision.get(
                        "query",
                        "what happens immediately "
                        "before the anchor event",
                    )
                )

                self.inspect_window(
                    video=video,
                    state=state,
                    action="SEARCH_BEFORE",
                    query=query,
                    start_s=start_s,
                    end_s=end_s,
                    num_frames=(
                        self.dense_frames
                    ),
                )

                continue

            # =================================================
            # INCREASE_DENSITY
            # =================================================

            if action == "INCREASE_DENSITY":

                start_s = float(
                    decision.get(
                        "start_s",
                        0.0,
                    )
                )

                end_s = float(
                    decision.get(
                        "end_s",
                        min(
                            video.duration_s,
                            start_s
                            + self.local_window_s,
                        ),
                    )
                )

                query = str(
                    decision.get(
                        "query",
                        question,
                    )
                )

                print(
                    "    densifying:",
                    round(
                        start_s,
                        2,
                    ),
                    "->",
                    round(
                        end_s,
                        2,
                    ),
                )

                self.inspect_window(
                    video=video,
                    state=state,
                    action="INCREASE_DENSITY",
                    query=query,
                    start_s=start_s,
                    end_s=end_s,
                    num_frames=(
                        self.dense_frames
                        * 2
                    ),
                )

                continue

        # =====================================================
        # Forced final answer
        # =====================================================

        if state.final_answer is None:

            decision = (
                self.controller
                .choose_action(
                    state
                )
            )

            if (
                decision["action"]
                == "ANSWER"
                and state.evidence
            ):
                sufficiency = (
                    self.controller
                    .assess_sufficiency(
                        state
                    )
                )

                if sufficiency:
                    print(
                        "    sufficiency:",
                        sufficiency
                    )

                    if not sufficiency.get(
                        "sufficient",
                        False,
                    ):
                        recommended = str(
                            sufficiency.get(
                                "recommended_action",
                                "SEARCH_LOCAL",
                            )
                        ).upper()

                        if recommended == "INCREASE_DENSITY":

                            evidence_id = (
                                sufficiency.get(
                                    "evidence_id"
                                )
                            )

                            if (
                                evidence_id is not None
                                and 0
                                <= int(evidence_id)
                                < len(state.evidence)
                            ):
                                target = (
                                    state.evidence[
                                        int(evidence_id)
                                    ]
                                )

                                decision = {
                                    "action":
                                        "INCREASE_DENSITY",
                                    "start_s":
                                        target.start_s,
                                    "end_s":
                                        target.end_s,
                                    "query":
                                        sufficiency.get(
                                            "recommended_query",
                                            sufficiency.get(
                                                "missing_visual_fact",
                                                question,
                                            ),
                                        ),
                                    "reason":
                                        sufficiency.get(
                                            "missing_visual_fact",
                                            "Need discriminative evidence.",
                                        ),
                                }

                        else:
                            decision = {
                                "action":
                                    "SEARCH_LOCAL",
                                "query":
                                    sufficiency.get(
                                        "recommended_query",
                                        sufficiency.get(
                                            "missing_visual_fact",
                                            question,
                                        ),
                                    ),
                                "reason":
                                    sufficiency.get(
                                        "missing_visual_fact",
                                        "Need discriminative evidence.",
                                    ),
                            }

                    else:
                        decision[
                            "answer"
                        ] = normalize_answer(
                            sufficiency.get(
                                "best_answer",
                                decision.get(
                                    "answer",
                                    "",
                                ),
                            )
                        )

            if (
                decision.get(
                    "action"
                )
                == "ANSWER"
            ):
                state.final_answer = (
                    normalize_answer(
                        decision.get(
                            "answer",
                            "",
                        )
                    )
                )

            if not state.final_answer:
                state.final_answer = (
                    self
                    .last_nonempty_prediction(
                        state
                    )
                )

        if state.final_answer is None:
            state.final_answer = ""

        return (
            state,
            trajectory,
        )
    
# ============================================================
# Video path resolution
# ============================================================

def build_video_index(
    sweep_results,
):
    index = {}

    for row in sweep_results:

        dataset = str(
            row.get(
                "dataset",
                row.get(
                    "source_dataset",
                    "unknown",
                ),
            )
        )

        video_id = str(
            row.get(
                "video_id",
                "",
            )
        )

        value = row.get(
            "video"
        )

        if value:
            index[
                (
                    dataset,
                    video_id,
                )
            ] = str(
                value
            )

    return index


def evidence_relevance_score(
    evidence: Evidence,
):
    score = 0

    if evidence.prediction:
        score += 2

    if evidence.confidence == "high":
        score += 2
    elif evidence.confidence == "medium":
        score += 1

    observation = (
        evidence.observation
        .lower()
    )

    negative_phrases = [
        "no evidence",
        "not seen",
        "not present",
        "does not appear",
        "no mention",
    ]

    if any(
        phrase in observation
        for phrase in negative_phrases
    ):
        score -= 3

    return score


def resolve_video_path(
    candidate,
    video_index,
):
    dataset = str(
        candidate.get(
            "dataset",
            "unknown",
        )
    )

    video_id = str(
        candidate.get(
            "video_id",
            "",
        )
    )

    for field_name in (
        "video",
        "video_path",
    ):
        value = candidate.get(
            field_name
        )

        if (
            value
            and Path(
                value
            ).exists()
        ):
            return str(
                value
            )

    candidate_path = (
        video_index.get(
            (
                dataset,
                video_id,
            )
        )
    )

    if (
        candidate_path
        and Path(
            candidate_path
        ).exists()
    ):
        return (
            candidate_path
        )

    raise FileNotFoundError(
        f"Could not resolve video "
        f"dataset={dataset} "
        f"video_id={video_id}"
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = (
        argparse
        .ArgumentParser()
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--sweep-results",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--base-url",
        default=(
            "http://localhost:9000/v1"
        ),
    )

    parser.add_argument(
        "--vlm-model",
        default=(
            "Qwen/"
            "Qwen2.5-VL-7B-Instruct"
        ),
    )

    parser.add_argument(
        "--siglip-model",
        default=(
            "google/"
            "siglip-so400m-patch14-384"
        ),
    )

    parser.add_argument(
        "--clip-device",
        default="cuda:0",
    )

    parser.add_argument(
        "--coarse-fps",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--coarse-topk",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--local-window-s",
        type=float,
        default=16.0,
    )

    parser.add_argument(
        "--temporal-window-s",
        type=float,
        default=16.0,
    )

    parser.add_argument(
        "--inspect-frames",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--dense-frames",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--global-chunk-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    candidates = load_jsonl(
        args.input
    )

    sweep_rows = load_jsonl(
        args.sweep_results
    )

    # ========================================================
    # Gold lookup
    #
    # Used only after execution.
    # ========================================================

    gold_by_question = {}

    for row in sweep_rows:

        key = (
            str(
                row.get(
                    "dataset",
                    row.get(
                        "source_dataset",
                        "unknown",
                    ),
                )
            ),
            str(
                row.get(
                    "video_id",
                    "",
                )
            ),
            str(
                row.get(
                    "qid",
                    "",
                )
            ),
            str(
                row.get(
                    "question",
                    "",
                )
            ),
        )

        gold = row.get(
            "answer_label"
        )

        if (
            gold is not None
        ):
            gold_by_question[
                key
            ] = (
                normalize_answer(
                    gold
                )
            )

    video_index = (
        build_video_index(
            sweep_rows
        )
    )

    if (
        args.limit > 0
    ):
        candidates = (
            candidates[
                : args.limit
            ]
        )

    client = OpenAI(
        base_url=(
            args.base_url
        ),
        api_key="EMPTY",
    )

    retriever = (
        SigLIPRetriever(
            model_name=(
                args.siglip_model
            ),
            device=(
                args.clip_device
            ),
        )
    )

    inspector = (
        EvidenceInspector(
            client=client,
            model=(
                args.vlm_model
            ),
        )
    )

    controller = (
        VideoAgentController(
            client=client,
            model=(
                args.vlm_model
            ),
        )
    )

    agent = VideoAgent(
        controller=controller,
        inspector=inspector,
        retriever=retriever,
        coarse_fps=(
            args.coarse_fps
        ),
        local_window_s=(
            args.local_window_s
        ),
        temporal_window_s=(
            args.temporal_window_s
        ),
        coarse_topk=(
            args.coarse_topk
        ),
        inspect_frames=(
            args.inspect_frames
        ),
        dense_frames=(
            args.dense_frames
        ),
        max_rounds=(
            args.max_rounds
        ),
        global_chunk_size=(
            args.global_chunk_size
        ),
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Resume
    # ========================================================

    done = set()

    if output.exists():

        for row in load_jsonl(
            output
        ):

            done.add((
                str(
                    row.get(
                        "dataset",
                        "unknown",
                    )
                ),
                str(
                    row.get(
                        "video_id",
                        "",
                    )
                ),
                str(
                    row.get(
                        "qid",
                        "",
                    )
                ),
                str(
                    row.get(
                        "question",
                        "",
                    )
                ),
            ))

    pending = []

    for candidate in candidates:

        key = (
            str(
                candidate.get(
                    "dataset",
                    "unknown",
                )
            ),
            str(
                candidate.get(
                    "video_id",
                    "",
                )
            ),
            str(
                candidate.get(
                    "qid",
                    "",
                )
            ),
            str(
                candidate.get(
                    "question",
                    "",
                )
            ),
        )

        if key not in done:
            pending.append(
                candidate
            )

    print(
        "candidates:",
        len(candidates),
    )

    print(
        "already done:",
        len(done),
    )

    print(
        "remaining:",
        len(pending),
    )

    total_correct = 0
    total_finished = 0
    total_latency = 0.0
    total_rounds = 0

    with output.open(
        "a"
    ) as output_file:

        for index, candidate in enumerate(
            pending,
            start=1,
        ):

            question = str(
                candidate[
                    "question"
                ]
            )

            choices = (
                candidate.get(
                    "choices"
                )
                or []
            )

            profile = (
                sanitize_runtime_profile(
                    candidate.get(
                        "semantic_profile"
                    )
                    or {}
                )
            )

            dataset = str(
                candidate.get(
                    "dataset",
                    "unknown",
                )
            )

            video_id = str(
                candidate.get(
                    "video_id",
                    "",
                )
            )

            qid = str(
                candidate.get(
                    "qid",
                    "",
                )
            )

            key = (
                dataset,
                video_id,
                qid,
                question,
            )

            video_path = (
                resolve_video_path(
                    candidate,
                    video_index,
                )
            )

            print()
            print(
                "=" * 80
            )

            print(
                f"[{index}/{len(pending)}]"
            )

            print(
                "dataset:",
                dataset,
            )

            print(
                "video:",
                video_path,
            )

            print(
                "question:",
                question,
            )

            video = VideoBackend(
                video_path
            )

            start = time.time()

            runtime_input = {
                "question": question,
                "choices": choices,
                "semantic_profile": profile,
            }

            assert_no_forbidden_keys(
                runtime_input
            )

            state, trajectory = (
                agent.run(
                    video=video,
                    **runtime_input,
                )
            )

            wall_latency = (
                time.time()
                - start
            )

            gold = (
                gold_by_question.get(
                    key
                )
            )

            prediction = (
                normalize_answer(
                    state.final_answer
                )
            )

            correct = (
                gold is not None
                and prediction
                == gold
            )

            result = {
                "dataset": (
                    dataset
                ),
                "video_id": (
                    video_id
                ),
                "qid": qid,
                "question": (
                    question
                ),
                "video_path": (
                    video_path
                ),

                # Offline/debug label only.
                # Never exposed to the runtime agent.
                "oracle_tier_for_analysis": (
                    candidate.get(
                        "tier"
                    )
                ),
                "semantic_profile": (
                    profile
                ),
                "gold": gold,
                "prediction": (
                    prediction
                ),
                "correct": bool(
                    correct
                ),
                "rounds": len(
                    trajectory
                ),
                "global_scan_count": (
                    state.global_scan_count
                ),
                "evidence": [
                    {
                        "action": (
                            evidence.action
                        ),
                        "query": (
                            evidence.query
                        ),
                        "start_s": (
                            evidence.start_s
                        ),
                        "end_s": (
                            evidence.end_s
                        ),
                        "timestamps": (
                            evidence.timestamps
                        ),
                        "observation": (
                            evidence.observation
                        ),
                        "prediction": (
                            evidence.prediction
                        ),
                        "confidence": (
                            evidence.confidence
                        ),
                        "latency_s": (
                            evidence.latency_s
                        ),
                    }
                    for evidence
                    in state.evidence
                ],
                "trajectory": (
                    trajectory
                ),
                "agent_internal_latency_s": (
                    state.total_latency_s
                ),
                "wall_latency_s": (
                    wall_latency
                ),
                "source": (
                    "online_dynamic_video_agent_v3"
                ),
            }

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_file.flush()

            total_finished += 1

            total_correct += int(
                correct
            )

            total_latency += (
                wall_latency
            )

            total_rounds += (
                len(
                    trajectory
                )
            )

            print(
                "FINAL:",
                prediction,
                "gold:",
                gold,
                "correct:",
                correct,
            )

            print(
                "rounds:",
                len(
                    trajectory
                ),
            )

            print(
                "global scans:",
                state.global_scan_count,
            )

            print(
                "latency:",
                wall_latency,
            )

    print()
    print(
        "=" * 80
    )

    print(
        "===== VIDEO AGENT SUMMARY ====="
    )

    print(
        "finished:",
        total_finished,
    )

    if total_finished:

        print(
            "accuracy:",
            total_correct
            / total_finished,
        )

        print(
            "avg_latency_s:",
            total_latency
            / total_finished,
        )

        print(
            "avg_rounds:",
            total_rounds
            / total_finished,
        )


if __name__ == "__main__":
    main()