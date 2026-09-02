#!/usr/bin/env python3
"""Mock up preparation-only versus cross-stage admission with continuous batching."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


OUTPUT = Path("analysis/figures/evaluation_mockups/12_continuous_batching_admission_mockup")
COLOR_A = "#D55E00"
COLOR_B = "#0072B2"


def draw_batch(axis, epochs, title, service_summary, explanation) -> None:
    slots = len(epochs[0])
    steps_per_request = 4
    total_steps = len(epochs) * steps_per_request

    axis.set_xlim(0, total_steps)
    axis.set_ylim(slots, 0)
    axis.set_xticks(range(total_steps + 1))
    axis.set_xticklabels([str(i + 1) if i < total_steps else "" for i in range(total_steps + 1)])
    axis.set_yticks([i + 0.5 for i in range(slots)], [f"Batch slot {i + 1}" for i in range(slots)])
    axis.set_xlabel("Decode step — every batch slot in a column executes concurrently")
    axis.set_title(f"{title}\n{service_summary}", fontsize=12.5, fontweight="bold", pad=12)

    for epoch_index, requests in enumerate(epochs):
        start = epoch_index * steps_per_request
        for slot, request in enumerate(requests):
            tenant = request[0]
            color = COLOR_A if tenant == "A" else COLOR_B
            axis.add_patch(
                Rectangle(
                    (start, slot),
                    steps_per_request,
                    1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.5,
                )
            )
            axis.text(
                start + steps_per_request / 2,
                slot + 0.5,
                request,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    for boundary in range(steps_per_request, total_steps, steps_per_request):
        axis.axvline(boundary, color="black", linestyle="--", linewidth=1.2)
    axis.grid(axis="x", alpha=0.12)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 8.8))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.68, hspace=0.82)

    # All requests use four decode steps to isolate the effect of admission.
    # The native ready-queue order repeatedly exposes three A requests followed
    # by one B request even though both tenants remain backlogged.
    prep_only_epochs = [
        ["A1", "A2", "A3", "B1"],
        ["A4", "A5", "A6", "B2"],
        ["A7", "A8", "A9", "B3"],
    ]
    cross_stage_epochs = [
        ["A1", "A2", "B1", "B2"],
        ["A3", "A4", "B3", "B4"],
        ["A5", "A6", "B5", "B6"],
    ]

    draw_batch(
        axes[0],
        prep_only_epochs,
        "Preparation-only: native/FCFS inference admission",
        "A = 36 slot-steps, B = 12 slot-steps; service gap = 24",
        "Equal request cost, but the ready-queue order gives A three of four newly available slots.",
    )
    draw_batch(
        axes[1],
        cross_stage_epochs,
        "Cross-stage max-min: tenant-fair inference admission",
        "A = 24 slot-steps, B = 24 slot-steps; service gap = 0",
        "At each admission event, the scheduler fills slots from the tenant with less accounted inference service.",
    )

    fig.legend(
        handles=[
            Patch(facecolor=COLOR_A, label="Tenant A request"),
            Patch(facecolor=COLOR_B, label="Tenant B request"),
        ],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.81),
    )
    fig.text(
        0.5,
        0.855,
        "Both tenants are model-ready and continuously backlogged; four batch slots; "
        "each request generates four decode tokens. Dashed lines mark new admissions.",
        fontsize=10.5,
        ha="center",
    )
    fig.suptitle(
        "MOCKUP — Tenant Fairness with vLLM Continuous Batching\n"
        "Inference requests execute concurrently; fairness is enforced when batch slots become available",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=230, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
