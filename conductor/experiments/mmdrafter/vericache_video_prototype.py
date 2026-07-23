from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    DynamicCache,
    Qwen2_5_VLForConditionalGeneration,
)


Cache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential VeriCache-style prototype using "
            "low/medium video inputs as the drafter and "
            "high-fidelity inputs as the verifier."
        )
    )

    parser.add_argument("--draft-inputs", required=True)
    parser.add_argument("--high-inputs", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--draft-name",
        choices=("low", "medium"),
        required=True,
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )

    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def advance_token_block(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    cache: Any,
    attention_mask: torch.Tensor,
    token_ids: torch.Tensor,
    rope_deltas: torch.Tensor,
) -> dict[str, Any]:
    new_mask = append_attention_mask(
        attention_mask,
        int(token_ids.shape[1]),
    )

    position_ids, cache_position = build_cached_positions(
        cache=cache,
        token_ids=token_ids,
        rope_deltas=rope_deltas,
    )

    outputs = model(
        input_ids=token_ids,
        attention_mask=new_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=cache,
        rope_deltas=rope_deltas,
        use_cache=True,
        return_dict=True,
    )

    return {
        "cache": outputs.past_key_values,
        "next_logits": outputs.logits[:, -1, :],
        "attention_mask": new_mask,
        "rope_deltas": outputs.rope_deltas,
    }

def move_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)

    if isinstance(value, dict):
        return {
            key: move_to_device(item, device)
            for key, item in value.items()
        }

    return value


def legacy_to_dynamic_cache(
    cache: Cache,
    model: Qwen2_5_VLForConditionalGeneration,
) -> DynamicCache:
    try:
        return DynamicCache.from_legacy_cache(
            cache
        )
    except TypeError:
        # Some Transformers versions require config
        # during DynamicCache construction.
        return DynamicCache(
            cache,
            config=model.config.text_config,
        )
    
def as_legacy_cache(cache: Any) -> Cache:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()

    return tuple(
        (
            layer[0],
            layer[1],
        )
        for layer in cache
    )


def cache_to_cpu(
    cache: Any,
    *,
    pin_memory: bool = True,
) -> Cache:
    legacy = as_legacy_cache(cache)

    cpu_layers: list[
        tuple[torch.Tensor, torch.Tensor]
    ] = []

    for key, value in legacy:
        key_cpu = (
            key.detach()
            .to("cpu")
            .contiguous()
        )
        value_cpu = (
            value.detach()
            .to("cpu")
            .contiguous()
        )

        if (
            pin_memory
            and torch.cuda.is_available()
        ):
            key_cpu = key_cpu.pin_memory()
            value_cpu = value_cpu.pin_memory()

        cpu_layers.append(
            (key_cpu, value_cpu)
        )

    return tuple(cpu_layers)


def cache_to_device(
    cache: Cache,
    device: torch.device,
    model: Qwen2_5_VLForConditionalGeneration,
) -> DynamicCache:
    legacy_gpu = tuple(
        (
            key.to(
                device,
                non_blocking=True,
            ),
            value.to(
                device,
                non_blocking=True,
            ),
        )
        for key, value in cache
    )

    return legacy_to_dynamic_cache(
        legacy_gpu,
        model,
    )


def cache_num_bytes(cache: Cache) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in cache
        for tensor in layer
    )


def crop_cache(
    cache: Any,
    sequence_length: int,
    model: Qwen2_5_VLForConditionalGeneration,
) -> DynamicCache:
    if isinstance(cache, DynamicCache):
        dynamic_cache = cache
    else:
        dynamic_cache = legacy_to_dynamic_cache(
            as_legacy_cache(cache),
            model,
        )

    dynamic_cache.crop(sequence_length)
    return dynamic_cache


def append_attention_mask(
    attention_mask: torch.Tensor,
    token_count: int,
) -> torch.Tensor:
    if token_count <= 0:
        return attention_mask

    extra = torch.ones(
        (
            attention_mask.shape[0],
            token_count,
        ),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    return torch.cat(
        [attention_mask, extra],
        dim=1,
    )

def build_cached_positions(
    *,
    cache: Any,
    token_ids: torch.Tensor,
    rope_deltas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    past_length = int(
        cache.get_seq_length()
    )

    token_count = int(
        token_ids.shape[1]
    )

    cache_position = torch.arange(
        past_length,
        past_length + token_count,
        dtype=torch.long,
        device=token_ids.device,
    )

    position_ids = (
        cache_position
        .view(1, 1, token_count)
        .expand(
            3,
            token_ids.shape[0],
            token_count,
        )
    )

    position_ids = (
        position_ids
        + rope_deltas
        .to(token_ids.device)
        .view(
            1,
            token_ids.shape[0],
            1,
        )
    )

    return position_ids, cache_position


@torch.inference_mode()
def prefill(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    outputs = model(
        **inputs,
        use_cache=True,
        return_dict=True,
    )

    return {
        "cache": outputs.past_key_values,
        "next_logits": outputs.logits[:, -1, :],
        "attention_mask": inputs["attention_mask"],
        "rope_deltas": outputs.rope_deltas,
        "sequence_length": int(
            inputs["input_ids"].shape[1]
        ),
    }

@torch.inference_mode()
def generate_draft_block(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    cache: Any,
    next_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    rope_deltas: torch.Tensor,
    block_size: int,
    eos_token_id: int | None,
) -> dict[str, Any]:
    tokens: list[torch.Tensor] = []
    cache_snapshots: list[Cache] = []
    logits_snapshots: list[torch.Tensor] = []
    mask_snapshots: list[torch.Tensor] = []
    rope_snapshots: list[torch.Tensor] = []

    current_cache = cache
    current_logits = next_logits
    current_mask = attention_mask
    current_rope_deltas = rope_deltas

    for _ in range(block_size):
        token = (
            current_logits
            .argmax(dim=-1)
            .unsqueeze(1)
        )

        tokens.append(token)

        current_mask = append_attention_mask(
            current_mask,
            1,
        )

        position_ids, cache_position = (
            build_cached_positions(
                cache=current_cache,
                token_ids=token,
                rope_deltas=current_rope_deltas,
            )
        )

        outputs = model(
            input_ids=token,
            attention_mask=current_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=current_cache,
            rope_deltas=current_rope_deltas,
            use_cache=True,
            return_dict=True,
        )

        current_cache = outputs.past_key_values
        current_logits = outputs.logits[:, -1, :]
        current_rope_deltas = outputs.rope_deltas

        cache_snapshots.append(
            as_legacy_cache(current_cache)
        )
        logits_snapshots.append(
            current_logits
        )
        mask_snapshots.append(
            current_mask
        )
        rope_snapshots.append(
            current_rope_deltas
        )

        if (
            eos_token_id is not None
            and int(token[0, 0].item())
            == eos_token_id
        ):
            break

    return {
        "draft_ids": torch.cat(tokens, dim=1),
        "cache_snapshots": cache_snapshots,
        "logits_snapshots": logits_snapshots,
        "mask_snapshots": mask_snapshots,
        "rope_snapshots": rope_snapshots,
    }

@torch.inference_mode()
def verify_block(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    high_cache: Any,
    high_next_logits: torch.Tensor,
    high_attention_mask: torch.Tensor,
    high_rope_deltas: torch.Tensor,
    draft_ids: torch.Tensor,
) -> dict[str, Any]:
    """
    Verify K drafted tokens with one high-model block forward.

    The prompt's final logits predict draft token 0.
    Feeding draft tokens 0..K-1 produces logits for tokens
    1..K and advances the high cache through the full block.
    """

    draft_length = int(
        draft_ids.shape[1]
    )

    first_prediction = (
        high_next_logits
        .argmax(dim=-1)
        .unsqueeze(1)
    )

    combined_mask = append_attention_mask(
        high_attention_mask,
        draft_length,
    )

    position_ids, cache_position = (
        build_cached_positions(
            cache=high_cache,
            token_ids=draft_ids,
            rope_deltas=high_rope_deltas,
        )
    )

    outputs = model(
        input_ids=draft_ids,
        attention_mask=combined_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=high_cache,
        rope_deltas=high_rope_deltas,
        use_cache=True,
        return_dict=True,
    )

    if draft_length > 1:
        remaining_predictions = (
            outputs.logits[
                :,
                :-1,
                :,
            ]
            .argmax(dim=-1)
        )

        verifier_ids = torch.cat(
            [
                first_prediction,
                remaining_predictions,
            ],
            dim=1,
        )
    else:
        verifier_ids = first_prediction

    matches = verifier_ids.eq(
        draft_ids
    )

    rejected = (
        ~matches[0]
    ).nonzero(as_tuple=False)

    if rejected.numel() == 0:
        accepted_length = draft_length
    else:
        accepted_length = int(
            rejected[0].item()
        )

    # outputs.logits[:, accepted_length - 1] predicts
    # the token after accepted_length consumed draft tokens.
    if accepted_length == 0:
        correction_id = first_prediction
    else:
        correction_id = (
            outputs.logits[
                :,
                accepted_length - 1,
                :,
            ]
            .argmax(dim=-1)
            .unsqueeze(1)
        )

    return {
        "verifier_ids": verifier_ids,
        "matches": matches,
        "accepted_length": accepted_length,
        "correction_id": correction_id,
        "full_block_cache": outputs.past_key_values,
        "full_block_next_logits": (
            outputs.logits[:, -1, :]
        ),
        "full_block_attention_mask": (
            combined_mask
        ),
        "full_block_rope_deltas": outputs.rope_deltas,
    }

@torch.inference_mode()
def advance_one_token(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    cache: Any,
    attention_mask: torch.Tensor,
    token_id: torch.Tensor,
    rope_deltas: torch.Tensor,
) -> dict[str, Any]:
    new_mask = append_attention_mask(
        attention_mask,
        token_id.shape[1],
    )

    position_ids, cache_position = (
        build_cached_positions(
            cache=cache,
            token_ids=token_id,
            rope_deltas=rope_deltas,
        )
    )

    outputs = model(
        input_ids=token_id,
        attention_mask=new_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=cache,
        rope_deltas=rope_deltas,
        use_cache=True,
        return_dict=True,
    )

    return {
        "cache": outputs.past_key_values,
        "next_logits": outputs.logits[:, -1, :],
        "attention_mask": new_mask,
        "rope_deltas": outputs.rope_deltas,
    }


@torch.inference_mode()
def high_only_generate(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    inputs: dict[str, Any],
    max_new_tokens: int,
) -> torch.Tensor:
    prompt_length = int(
        inputs["input_ids"].shape[1]
    )

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
    )

    return generated.sequences[
        :,
        prompt_length:,
    ]

@torch.inference_mode()
def manual_high_generate(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    inputs: dict[str, Any],
    max_new_tokens: int,
) -> torch.Tensor:
    state = prefill(
        model=model,
        inputs=inputs,
    )

    generated: list[torch.Tensor] = []

    for _ in range(max_new_tokens):
        token_id = (
            state["next_logits"]
            .argmax(dim=-1)
            .unsqueeze(1)
        )

        generated.append(token_id)

        state = advance_one_token(
            model=model,
            cache=state["cache"],
            attention_mask=state[
                "attention_mask"
            ],
            token_id=token_id,
            rope_deltas=state[
                "rope_deltas"
            ],
        )

    return torch.cat(
        generated,
        dim=1,
    )

@torch.inference_mode()
def manual_high_generate_with_offload(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    inputs: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
) -> torch.Tensor:
    state = prefill(
        model=model,
        inputs=inputs,
    )

    cache_cpu = cache_to_cpu(
        state["cache"]
    )

    next_logits_cpu = (
        state["next_logits"]
        .detach()
        .cpu()
        .pin_memory()
    )

    attention_mask_cpu = (
        state["attention_mask"]
        .detach()
        .cpu()
        .pin_memory()
    )

    rope_deltas_cpu = (
        state["rope_deltas"]
        .detach()
        .cpu()
        .pin_memory()
    )

    del state

    generated: list[torch.Tensor] = []

    for step in range(max_new_tokens):
        cache_gpu = cache_to_device(
            cache_cpu,
            device,
            model,
        )

        next_logits_gpu = next_logits_cpu.to(
            device,
            non_blocking=True,
        )

        attention_mask_gpu = attention_mask_cpu.to(
            device,
            non_blocking=True,
        )

        rope_deltas_gpu = rope_deltas_cpu.to(
            device,
            non_blocking=True,
        )

        synchronize(device)

        token_id = (
            next_logits_gpu
            .argmax(dim=-1)
            .unsqueeze(1)
        )

        generated.append(
            token_id.detach().cpu()
        )

        print(
            f"offload step={step}, "
            f"cache_len={cache_gpu.get_seq_length()}, "
            f"token={int(token_id[0, 0])}",
            flush=True,
        )

        state = advance_one_token(
            model=model,
            cache=cache_gpu,
            attention_mask=attention_mask_gpu,
            token_id=token_id,
            rope_deltas=rope_deltas_gpu,
        )

        cache_cpu = cache_to_cpu(
            state["cache"]
        )

        next_logits_cpu = (
            state["next_logits"]
            .detach()
            .cpu()
            .pin_memory()
        )

        attention_mask_cpu = (
            state["attention_mask"]
            .detach()
            .cpu()
            .pin_memory()
        )

        rope_deltas_cpu = (
            state["rope_deltas"]
            .detach()
            .cpu()
            .pin_memory()
        )
        del state
        del cache_gpu
        del next_logits_gpu
        del attention_mask_gpu
        del rope_deltas_gpu
        del token_id

    return torch.cat(
        generated,
        dim=1,
    )


@torch.inference_mode()
def advance_token_sequence(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    cache: Any,
    attention_mask: torch.Tensor,
    token_ids: torch.Tensor,
    rope_deltas: torch.Tensor,
) -> dict[str, Any]:
    state = {
        "cache": cache,
        "attention_mask": attention_mask,
        "rope_deltas": rope_deltas,
        "next_logits": None,
    }

    for token_index in range(token_ids.shape[1]):
        token_id = token_ids[
            :,
            token_index:token_index + 1,
        ]

        state = advance_one_token(
            model=model,
            cache=state["cache"],
            attention_mask=state["attention_mask"],
            token_id=token_id,
            rope_deltas=state["rope_deltas"],
        )

    return state

    
@torch.inference_mode()
def vericache_decode(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Any,
    draft_inputs_cpu: dict[str, Any],
    high_inputs_cpu: dict[str, Any],
    device: torch.device,
    block_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    eos_token_id = processor.tokenizer.eos_token_id

    draft_inputs = move_to_device(
        draft_inputs_cpu,
        device,
    )

    high_inputs = move_to_device(
        high_inputs_cpu,
        device,
    )

    synchronize(device)
    total_start = time.perf_counter()

    # Full high prompt is computed once.
    high_prefill_start = time.perf_counter()

    high_state_gpu = prefill(
        model=model,
        inputs=high_inputs,
    )

    synchronize(device)

    high_prefill_s = (
        time.perf_counter()
        - high_prefill_start
    )

    high_prompt_length = int(
        high_state_gpu["sequence_length"]
    )

    high_next_logits_cpu = (
        high_state_gpu["next_logits"]
        .detach()
        .cpu()
        .pin_memory()
    )

    high_attention_mask_cpu = (
        high_state_gpu["attention_mask"]
        .detach()
        .cpu()
        .pin_memory()
    )

    high_rope_deltas_cpu = (
        high_state_gpu["rope_deltas"]
        .detach()
        .cpu()
        .pin_memory()
    )

    high_cache_cpu = cache_to_cpu(
        high_state_gpu["cache"]
    )

    high_kv_bytes = cache_num_bytes(
        high_cache_cpu
    )

    # Remove full high KV from GPU between verification rounds.
    del high_state_gpu

    synchronize(device)

    draft_prefill_start = time.perf_counter()

    draft_state = prefill(
        model=model,
        inputs=draft_inputs,
    )

    synchronize(device)

    draft_prefill_s = (
        time.perf_counter()
        - draft_prefill_start
    )

    del high_inputs
    del draft_inputs

    output_tokens: list[int] = []

    total_drafted = 0
    total_accepted = 0
    verification_rounds = 0
    full_block_accepts = 0

    total_draft_s = 0.0
    total_transfer_in_s = 0.0
    total_verify_s = 0.0
    total_transfer_out_s = 0.0

    while len(output_tokens) < max_new_tokens:
        remaining = (
            max_new_tokens
            - len(output_tokens)
        )

        current_block_size = min(
            block_size,
            remaining,
        )

        synchronize(device)
        draft_start = time.perf_counter()
        draft_round_cache_cpu = cache_to_cpu(
            draft_state["cache"],
            pin_memory=False,
        )

        draft_round_mask = draft_state[
            "attention_mask"
        ]

        draft_round_rope_deltas = draft_state[
            "rope_deltas"
        ]

        proposal = generate_draft_block(
            model=model,
            cache=draft_state["cache"],
            next_logits=draft_state["next_logits"],
            attention_mask=draft_state["attention_mask"],
            rope_deltas=draft_state["rope_deltas"],
            block_size=current_block_size,
            eos_token_id=eos_token_id,
        )

        synchronize(device)

        total_draft_s += (
            time.perf_counter()
            - draft_start
        )

        draft_ids = proposal["draft_ids"]
        drafted_count = int(
            draft_ids.shape[1]
        )

        total_drafted += drafted_count

        # Reload full high KV for this verification round.
        synchronize(device)
        transfer_in_start = time.perf_counter()

        high_cache_gpu = cache_to_device(
            high_cache_cpu,
            device,
            model,
        )

        high_next_logits_gpu = (
            high_next_logits_cpu.to(
                device,
                non_blocking=True,
            )
        )

        high_attention_mask_gpu = (
            high_attention_mask_cpu.to(
                device,
                non_blocking=True,
            )
        )

        high_rope_deltas_gpu = (
            high_rope_deltas_cpu.to(
                device,
                non_blocking=True,
            )
        )

        synchronize(device)

        total_transfer_in_s += (
            time.perf_counter()
            - transfer_in_start
        )

        synchronize(device)
        verify_start = time.perf_counter()

        verification = verify_block_sequential(
            model=model,
            high_cache=high_cache_gpu,
            high_next_logits=high_next_logits_gpu,
            high_attention_mask=high_attention_mask_gpu,
            high_rope_deltas=high_rope_deltas_gpu,
            draft_ids=draft_ids,
        )

        synchronize(device)

        total_verify_s += (
            time.perf_counter()
            - verify_start
        )

        verification_rounds += 1

        accepted_length = int(
            verification[
                "accepted_length"
            ]
        )

        verification_full_accept = bool(
            verification["full_accept"]
        )

        total_accepted += accepted_length

        accepted_ids = draft_ids[
            :,
            :accepted_length,
        ]

        output_tokens.extend(
            accepted_ids[0]
            .detach()
            .cpu()
            .tolist()
        )

        if verification_full_accept:
            full_block_accepts += 1

            draft_state = {
                "cache": legacy_to_dynamic_cache(
                    proposal["cache_snapshots"][-1],
                    model,
                ),
                "next_logits": proposal[
                    "logits_snapshots"
                ][-1],
                "attention_mask": proposal[
                    "mask_snapshots"
                ][-1],
                "rope_deltas": proposal[
                    "rope_snapshots"
                ][-1],
            }

            next_high_cache = verification[
                "state"
            ]["cache"]

            next_high_logits = verification[
                "state"
            ]["next_logits"]

            next_high_mask = verification[
                "state"
            ]["attention_mask"]

            next_high_rope_deltas = verification[
                "state"
            ]["rope_deltas"]

            reached_eos = (
                eos_token_id is not None
                and int(draft_ids[0, -1].item())
                == eos_token_id
            )

        else:
            correction_id = verification[
                "correction_id"
            ]

            output_tokens.append(
                int(correction_id[0, 0].item())
            )

            next_high_cache = verification[
                "state"
            ]["cache"]

            next_high_logits = verification[
                "state"
            ]["next_logits"]

            next_high_mask = verification[
                "state"
            ]["attention_mask"]

            next_high_rope_deltas = verification[
                "state"
            ]["rope_deltas"]

            committed_ids = torch.cat(
                [
                    draft_ids[:, :accepted_length],
                    correction_id,
                ],
                dim=1,
            )

            clean_draft_cache = cache_to_device(
                draft_round_cache_cpu,
                device,
                model,
            )

            draft_state = advance_token_sequence(
                model=model,
                cache=clean_draft_cache,
                attention_mask=draft_round_mask,
                token_ids=committed_ids,
                rope_deltas=draft_round_rope_deltas,
            )

            reached_eos = (
                eos_token_id is not None
                and int(correction_id[0, 0].item())
                == eos_token_id
            )
       

        synchronize(device)
        transfer_out_start = (
            time.perf_counter()
        )

        high_cache_cpu = cache_to_cpu(
            next_high_cache
        )

        high_next_logits_cpu = (
            next_high_logits
            .detach()
            .cpu()
            .pin_memory()
        )

        high_attention_mask_cpu = (
            next_high_mask
            .detach()
            .cpu()
            .pin_memory()
        )

        high_rope_deltas_cpu = (
            next_high_rope_deltas
            .detach()
            .cpu()
            .pin_memory()
        )

        synchronize(device)

        total_transfer_out_s += (
            time.perf_counter()
            - transfer_out_start
        )

        

        del high_cache_gpu
        del verification

        if not verification_full_accept:
            del clean_draft_cache
            del committed_ids
            del correction_id

        del high_next_logits_gpu
        del high_attention_mask_gpu
        del high_rope_deltas_gpu

        del next_high_cache
        del next_high_logits
        del next_high_mask
        del next_high_rope_deltas

        del proposal
        del draft_ids
        del draft_round_cache_cpu
        del draft_round_mask
        del draft_round_rope_deltas

        if reached_eos:
            break

    synchronize(device)

    total_s = (
        time.perf_counter()
        - total_start
    )

    output_tokens = output_tokens[
        :max_new_tokens
    ]

    return {
        "output_token_ids": output_tokens,
        "output_text": processor.decode(
            output_tokens,
            skip_special_tokens=True,
        ),
        "output_tokens": len(
            output_tokens
        ),
        "high_prefill_s": high_prefill_s,
        "draft_prefill_s": draft_prefill_s,
        "draft_s": total_draft_s,
        "high_kv_transfer_in_s": (
            total_transfer_in_s
        ),
        "verification_s": total_verify_s,
        "high_kv_transfer_out_s": (
            total_transfer_out_s
        ),
        "total_s": total_s,
        "tokens_per_s": (
            len(output_tokens) / total_s
            if total_s > 0
            else 0.0
        ),
        "total_drafted_tokens": (
            total_drafted
        ),
        "total_accepted_tokens": (
            total_accepted
        ),
        "acceptance_fraction": (
            total_accepted / total_drafted
            if total_drafted > 0
            else 0.0
        ),
        "verification_rounds": (
            verification_rounds
        ),
        "full_block_accepts": (
            full_block_accepts
        ),
        "high_kv_bytes": high_kv_bytes,
        "high_kv_mib": (
            high_kv_bytes / 1024**2
        ),
    }


def load_inputs(path: str) -> dict[str, Any]:
    resolved = (
        Path(path)
        .expanduser()
        .resolve()
    )

    payload = torch.load(
        resolved,
        map_location="cpu",
        weights_only=False,
    )

    if "inputs" not in payload:
        raise KeyError(
            f"'inputs' missing from {resolved}"
        )

    return payload["inputs"]

@torch.inference_mode()
def verify_block_sequential(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    high_cache: Any,
    high_next_logits: torch.Tensor,
    high_attention_mask: torch.Tensor,
    high_rope_deltas: torch.Tensor,
    draft_ids: torch.Tensor,
) -> dict[str, Any]:
    state = {
        "cache": high_cache,
        "next_logits": high_next_logits,
        "attention_mask": high_attention_mask,
        "rope_deltas": high_rope_deltas,
    }

    accepted_length = 0
    verifier_tokens: list[torch.Tensor] = []

    for token_index in range(draft_ids.shape[1]):
        verifier_id = (
            state["next_logits"]
            .argmax(dim=-1)
            .unsqueeze(1)
        )

        verifier_tokens.append(verifier_id)

        draft_id = draft_ids[
            :,
            token_index:token_index + 1,
        ]

        if not torch.equal(
            verifier_id,
            draft_id,
        ):
            correction_id = verifier_id

            corrected_state = advance_one_token(
                model=model,
                cache=state["cache"],
                attention_mask=state[
                    "attention_mask"
                ],
                token_id=correction_id,
                rope_deltas=state[
                    "rope_deltas"
                ],
            )

            return {
                "accepted_length": accepted_length,
                "correction_id": correction_id,
                "state": corrected_state,
                "verifier_ids": torch.cat(
                    verifier_tokens,
                    dim=1,
                ),
                "full_accept": False,
            }

        state = advance_one_token(
            model=model,
            cache=state["cache"],
            attention_mask=state[
                "attention_mask"
            ],
            token_id=draft_id,
            rope_deltas=state[
                "rope_deltas"
            ],
        )

        accepted_length += 1

    return {
        "accepted_length": accepted_length,
        "correction_id": None,
        "state": state,
        "verifier_ids": torch.cat(
            verifier_tokens,
            dim=1,
        ),
        "full_accept": True,
    }

def main() -> None:
    args = parse_args()

    if args.block_size <= 0:
        raise ValueError(
            "--block-size must be positive"
        )

    if args.max_new_tokens <= 0:
        raise ValueError(
            "--max-new-tokens must be positive"
        )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but unavailable"
        )

    processor = AutoProcessor.from_pretrained(
        args.model,
        use_fast=False,
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )

    draft_inputs_cpu = load_inputs(
        args.draft_inputs
    )

    high_inputs_cpu = load_inputs(
        args.high_inputs
    )

    high_inputs_gpu = move_to_device(
        high_inputs_cpu,
        device,
    )

    synchronize(device)
    baseline_start = time.perf_counter()

    baseline_ids = high_only_generate(
        model=model,
        inputs=high_inputs_gpu,
        max_new_tokens=args.max_new_tokens,
    )

    synchronize(device)

    baseline_s = (
        time.perf_counter()
        - baseline_start
    )

    manual_ids = manual_high_generate(
        model=model,
        inputs=high_inputs_gpu,
        max_new_tokens=args.max_new_tokens,
    )

    synchronize(device)

    manual_match = torch.equal(
        baseline_ids,
        manual_ids,
    )

    print(
        "generate token ids:",
        baseline_ids[0]
        .detach()
        .cpu()
        .tolist(),
        flush=True,
    )

    print(
        "manual token ids:",
        manual_ids[0]
        .detach()
        .cpu()
        .tolist(),
        flush=True,
    )

    print(
        "manual cached match:",
        manual_match,
        flush=True,
    )
    if not manual_match:
        raise RuntimeError(
            "Manual cached high decoding does not "
            "match model.generate()."
        )

    baseline_token_ids = (
        baseline_ids[0]
        .detach()
        .cpu()
        .tolist()
    )

    del manual_ids

    baseline_text = processor.decode(
        baseline_token_ids,
        skip_special_tokens=True,
    )

    del high_inputs_gpu
    del baseline_ids

    result = vericache_decode(
        model=model,
        processor=processor,
        draft_inputs_cpu=draft_inputs_cpu,
        high_inputs_cpu=high_inputs_cpu,
        device=device,
        block_size=args.block_size,
        max_new_tokens=args.max_new_tokens,
    )

    result.update(
        {
            "draft_name": args.draft_name,
            "block_size": args.block_size,
            "max_new_tokens": (
                args.max_new_tokens
            ),
            "high_only_token_ids": (
                baseline_token_ids
            ),
            "high_only_text": baseline_text,
            "high_only_s": baseline_s,
            "high_only_tokens_per_s": (
                len(baseline_token_ids)
                / baseline_s
                if baseline_s > 0
                else 0.0
            ),
            "lossless_match": (
                result["output_token_ids"]
                == baseline_token_ids
            ),
            "speedup_over_high_only": (
                baseline_s / result["total_s"]
                if result["total_s"] > 0
                else 0.0
            ),
        }
    )

    output_path = (
        Path(args.output)
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nHigh-only output:")
    print(baseline_text)

    print("\nVeriCache-style output:")
    print(result["output_text"])

    print(
        f"\nLossless match: "
        f"{result['lossless_match']}"
    )
    print(
        f"Acceptance: "
        f"{result['total_accepted_tokens']}/"
        f"{result['total_drafted_tokens']} "
        f"({result['acceptance_fraction']:.2%})"
    )
    print(
        f"Verification rounds: "
        f"{result['verification_rounds']}"
    )
    print(
        f"High-only: "
        f"{baseline_s:.3f}s, "
        f"{result['high_only_tokens_per_s']:.2f} tok/s"
    )
    print(
        f"Prototype: "
        f"{result['total_s']:.3f}s, "
        f"{result['tokens_per_s']:.2f} tok/s"
    )
    print(
        f"Current speedup: "
        f"{result['speedup_over_high_only']:.3f}x"
    )
    print(
        f"Full high KV: "
        f"{result['high_kv_mib']:.1f} MiB on CPU"
    )
    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
