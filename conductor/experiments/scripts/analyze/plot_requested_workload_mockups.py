#!/usr/bin/env python3
"""Create complete, annotated mockups for the requested fairness workloads."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("analysis/figures/requested_workload_mockups")
POLICIES = ["FCFS", "Tenant RR", "Prep-only", "Inference-only", "Cross-stage"]
COLORS = {
    "FCFS": "#D55E00",
    "Tenant RR": "#CC79A7",
    "Prep-only": "#E69F00",
    "Inference-only": "#56B4E9",
    "Cross-stage": "#009E73",
}
MARKERS = {"FCFS": "s", "Tenant RR": "D", "Prep-only": "o", "Inference-only": "^", "Cross-stage": "v"}


def finish(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=230, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def decorate(axis, xlabel: str, ylabel: str, title: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontweight="bold")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def increasing(time: np.ndarray, endpoint: float, exponent: float = 1.18) -> np.ndarray:
    return endpoint * (time / time[-1]) ** exponent


def bounded(time: np.ndarray, level: float, amplitude: float, phase: float = 0) -> np.ndarray:
    values = level + amplitude * np.sin(time / 42 + phase)
    values[0] = 0
    return values


def constant_rate_service_gap() -> None:
    time = np.arange(0, 601, 50, dtype=float)
    cpu = {
        "FCFS": increasing(time, 620, 1.25),
        "Tenant RR": increasing(time, 300, 1.15),
        "Prep-only": bounded(time, 34, 7, 0.8),
        "Inference-only": increasing(time, 575, 1.22),
        "Cross-stage": bounded(time, 27, 5, 0.1),
    }
    inference = {
        "FCFS": increasing(time, 480, 1.22),
        "Tenant RR": increasing(time, 225, 1.13),
        "Prep-only": increasing(time, 430, 1.20),
        "Inference-only": bounded(time, 32, 6, 0.7),
        "Cross-stage": bounded(time, 25, 5, 0.0),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.66, wspace=0.18)
    for policy in POLICIES:
        style = dict(color=COLORS[policy], marker=MARKERS[policy], linewidth=2.4)
        axes[0].plot(time, cpu[policy], label=policy, **style)
        axes[1].plot(time, inference[policy], label=policy, **style)
    decorate(
        axes[0],
        "Time (seconds)",
        "Service gap between A and B\n(CPU worker-seconds; lower is fairer)",
        "(a) Preparation-heavy configuration",
    )
    decorate(
        axes[1],
        "Time (seconds)",
        "Service gap between A and B\n(inference-service units; lower is fairer)",
        "(b) Inference-heavy configuration",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.79))
    fig.text(
        0.5,
        0.715,
        "CPU run: equal rates, A=128 frames and B=32. Inference run: equal rates and 32 frames, "
        "A requests ~100 generated tokens and B one word. Both tenants remain stage-backlogged.",
        ha="center",
        fontsize=10.5,
    )
    fig.suptitle(
        "MOCKUP 1 — Constant-Rate Two-Tenant Service Fairness\n"
        "Heterogeneous CPU and inference costs are measured in separate matched bottleneck runs",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "01_constant_rate_stage_service_gap")


def identical_cost_rate_asymmetry() -> None:
    """Mock up the control case where equal turns imply equal service."""
    time = np.arange(0, 601, 50, dtype=float)
    cpu = {
        "FCFS": increasing(time, 610, 1.22),
        "Tenant RR": bounded(time, 25, 5, 0.4),
        "Prep-only": bounded(time, 31, 6, 0.9),
        # This policy leaves preparation in FCFS order.
        "Inference-only": increasing(time, 565, 1.20),
        "Cross-stage": bounded(time, 22, 4, 0.1),
    }
    inference = {
        "FCFS": increasing(time, 475, 1.20),
        "Tenant RR": bounded(time, 24, 5, 0.5),
        # This ablation leaves inference admission in FCFS order.
        "Prep-only": increasing(time, 425, 1.18),
        "Inference-only": bounded(time, 28, 5, 0.3),
        "Cross-stage": bounded(time, 21, 4, 0.0),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
    fig.subplots_adjust(
        left=0.08, right=0.99, bottom=0.12, top=0.66, wspace=0.18
    )
    for policy in POLICIES:
        style = dict(
            color=COLORS[policy], marker=MARKERS[policy], linewidth=2.4
        )
        axes[0].plot(time, cpu[policy], label=policy, **style)
        axes[1].plot(time, inference[policy], label=policy, **style)
    decorate(
        axes[0],
        "Time (seconds)",
        "Service gap between A and B\n(CPU worker-seconds; lower is fairer)",
        "(a) Preparation-service gap",
    )
    decorate(
        axes[1],
        "Time (seconds)",
        "Service gap between A and B\n(inference-service units; lower is fairer)",
        "(b) Inference-service gap",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, ncol=5, loc="upper center",
        bbox_to_anchor=(0.5, 0.79),
    )
    fig.text(
        0.5,
        0.715,
        "Control workload: continuously backlogged A and B; deterministic A:B request rate = 2:1; "
        "every request has the same 32 frames, prompt, and output limit.",
        ha="center",
        fontsize=10.5,
    )
    fig.suptitle(
        "MOCKUP — Identical-Cost Requests with Unequal Arrival Rates\n"
        "Round-robin approximates service fairness because every request costs the same",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "00_identical_cost_rate_asymmetry")


def on_off_work_conservation() -> None:
    time = np.arange(0, 241, 5, dtype=float)
    on = ((time // 30) % 2 == 0).astype(float)
    a_rate = 0.25 * on
    b_rate = 1.0 - a_rate
    total = a_rate + b_rate
    # This is an ideal expected-allocation mockup. Measured windowed rates will
    # have request-granularity noise and short pipeline transition delays.
    infer_a = a_rate.copy()
    infer_b = b_rate.copy()
    # Draw a continuous protected-latency envelope for visual comparison.
    # In measured results, A has observations only in ON windows; OFF-window
    # values must be omitted rather than interpreted as real requests.
    a_p95 = 1.4 + 0.08 * np.sin(time / 12)
    b_p95 = 2.5 + 0.055 * time

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.2))
    fig.subplots_adjust(
        left=0.07, right=0.99, bottom=0.12, top=0.57, wspace=0.24
    )
    for axis, rates, title in [
        (axes[0], (a_rate, b_rate), "(a) Preparation service rate"),
        (axes[1], (infer_a, infer_b), "(b) Inference service rate"),
    ]:
        for start in range(0, 240, 60):
            axis.axvspan(
                start, start + 30, color="#D55E00", alpha=0.07,
                label="A ON" if start == 0 else None,
            )
            axis.axvspan(
                start + 30, start + 60, color="#777777", alpha=0.05,
                label="A OFF" if start == 0 else None,
            )
        axis.step(
            time, rates[0], where="post", color="#D55E00", linewidth=2.3,
            label="Tenant A received",
        )
        axis.step(
            time, rates[1], where="post", color="#0072B2", linewidth=2.3,
            label="Tenant B received",
        )
        axis.plot(
            time, total, color="black", linestyle="--", linewidth=1.5,
            label="Total utilized capacity",
        )
        axis.set_ylim(-0.05, 1.08)
        decorate(axis, "Time (seconds)", "Received service rate\n(fraction of stage capacity)", title)
    axes[2].plot(
        time, a_p95, color="#D55E00", linewidth=2.3, marker="o",
        markevery=6, label="Tenant A protected trend",
    )
    axes[2].plot(time, b_p95, color="#0072B2", linewidth=2.3, label="Tenant B")
    for start in range(0, 240, 60):
        axes[2].axvspan(start, start + 30, color="#D55E00", alpha=0.07)
        axes[2].axvspan(start + 30, start + 60, color="#777777", alpha=0.05)
    decorate(
        axes[2], "Request-arrival time (seconds)",
        "Windowed P95 end-to-end latency (s)",
        "(c) A is protected; B's overload backlog grows",
    )
    axes[2].legend(frameon=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, 0.76),
    )
    fig.text(
        0.5,
        0.645,
        "Two matched runs with empirically calibrated stage capacity C; identical 32-frame requests.\n"
        "A offers 0.25C for 30 s ON, then zero for 30 s OFF; B remains backlogged. Use 30 s windows.",
        ha="center",
        fontsize=10.5,
    )
    fig.suptitle(
        "MOCKUP 2 — ON/OFF Work Conservation Under Cross-Stage Max-Min\n"
        "Preparation and inference are evaluated in separate capacity-calibrated runs",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "02_on_off_work_conservation")


def heterogeneous_video_cost() -> None:
    time = np.arange(0, 301, 10, dtype=float)
    a_share = 0.5 + 0.055 * np.sin(time / 22)
    b_share = 1.0 - a_share
    completed_a = 0.125 * time
    # Deliberately not 4x: frame count is not assumed to scale worker-time
    # perfectly. The real slope must come from measured completions.
    completed_b = 0.39 * time

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.13, top=0.64, wspace=0.22)
    axes[0].plot(time, a_share, color="#D55E00", linewidth=2.4, label="Tenant A: 128 frames")
    axes[0].plot(time, b_share, color="#0072B2", linewidth=2.4, label="Tenant B: 32 frames")
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1.3, label="Equal service share")
    axes[0].set_ylim(0.35, 0.65)
    decorate(
        axes[0],
        "Time (seconds)",
        "Received preparation service rate\n(fraction of worker capacity)",
        "(a) Equal resource service",
    )
    axes[1].plot(time, completed_a, color="#D55E00", linewidth=2.4, label="Tenant A: expensive")
    axes[1].plot(time, completed_b, color="#0072B2", linewidth=2.4, label="Tenant B: inexpensive")
    decorate(
        axes[1],
        "Time (seconds)",
        "Completed video preparations",
        "(b) Unequal completion counts are expected",
    )
    for axis in axes:
        axis.legend(frameon=False)
    fig.text(
        0.5,
        0.695,
        "Workload: both tenants continuously preparation-backlogged; same codec, resolution, prompt, and output limit. "
        "A samples 128 frames/request; B samples 32. Cross-stage max-min uses observed worker-time.",
        ha="center",
        fontsize=10.5,
    )
    fig.suptitle(
        "MOCKUP 3 — Heterogeneous Video Preparation Cost\n"
        "Fair service means equal worker-time—not equal completed-request counts",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "03_heterogeneous_video_cost")


def constant_underloaded_overloaded() -> None:
    time = np.arange(0, 301, 10, dtype=float)
    a = np.full_like(time, 0.25)
    b = np.full_like(time, 0.75)
    total = a + b
    accumulated_a = 0.25 * time
    accumulated_b = 0.75 * time

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.64, wspace=0.24)
    for axis, title in [(axes[0], "(a) Preparation allocation"), (axes[1], "(b) Inference allocation")]:
        axis.plot(time, a, color="#D55E00", linewidth=2.4, label="Tenant A received")
        axis.plot(time, b, color="#0072B2", linewidth=2.4, label="Tenant B received")
        axis.plot(time, total, color="black", linestyle="--", linewidth=1.4, label="Total utilization")
        axis.axhline(0.5, color="#777777", linestyle=":", linewidth=1.2, label="Nominal equal share")
        axis.set_ylim(0, 1.08)
        decorate(axis, "Time (seconds)", "Received service rate\n(fraction of stage capacity)", title)
    axes[2].plot(time, accumulated_a, color="#D55E00", linewidth=2.4, label="Tenant A")
    axes[2].plot(time, accumulated_b, color="#0072B2", linewidth=2.4, label="Tenant B")
    decorate(axes[2], "Time (seconds)", "Accumulated received service\n(stage-capacity seconds)", "(c) Demand-proportional totals")
    axes[2].legend(frameon=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.77))
    fig.text(
        0.5,
        0.695,
        "Workload: identical 32-frame requests. Tenant A continuously offers 0.25× stage capacity and is not backlogged; "
        "Tenant B continuously offers at least 1.0× capacity and remains backlogged.",
        ha="center",
        fontsize=10.5,
    )
    fig.suptitle(
        "MOCKUP 4 — Constant Underloaded/Overloaded Work Conservation\n"
        "A receives all requested service; B consumes the unused share; total utilization remains 100%",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "04_constant_work_conservation")


def cpu_inference_fairness_plane() -> None:
    """Show the expected two-stage fairness position of each policy."""
    # Illustrative positions only. Replace these values with measured service
    # scores and seed-level confidence intervals before using the figure as a
    # result.
    expected = {
        "FCFS": (34, 30),
        "Tenant RR": (70, 66),
        "Prep-only": (94, 43),
        "Inference-only": (41, 94),
        "Cross-stage": (95, 96),
    }

    fig, axis = plt.subplots(figsize=(9.2, 7.8))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.19, top=0.76)

    # Make the desirable region and the two single-stage limitations explicit.
    axis.axvspan(85, 100, color="#009E73", alpha=0.055)
    axis.axhspan(85, 100, color="#009E73", alpha=0.055)
    axis.axvline(85, color="#777777", linestyle=":", linewidth=1.2)
    axis.axhline(85, color="#777777", linestyle=":", linewidth=1.2)

    label_offsets = {
        "FCFS": (3, -7),
        "Tenant RR": (3, -7),
        "Prep-only": (-24, -7),
        "Inference-only": (3, -1),
        "Cross-stage": (-27, -7),
    }
    for policy in POLICIES:
        x_value, y_value = expected[policy]
        axis.scatter(
            x_value,
            y_value,
            s=190 if policy == "Cross-stage" else 145,
            marker=MARKERS[policy],
            color=COLORS[policy],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        dx, dy = label_offsets[policy]
        label = "Cross-stage (ours)" if policy == "Cross-stage" else policy
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=(x_value + dx, y_value + dy),
            fontsize=11,
            fontweight="bold" if policy == "Cross-stage" else "normal",
            arrowprops=dict(arrowstyle="-", color=COLORS[policy], alpha=0.7),
        )

    axis.annotate(
        "High fairness at both stages",
        xy=(97, 98),
        xytext=(63, 103),
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="#007A59",
        arrowprops=dict(arrowstyle="->", color="#007A59", linewidth=1.5),
    )
    axis.text(91.5, 13, "Preparation fair\nInference unfair", ha="center", color="#666666")
    axis.text(13, 91.5, "Inference fair\nPreparation unfair", va="center", color="#666666")

    axis.set_xlim(0, 105)
    axis.set_ylim(0, 108)
    axis.set_aspect("equal", adjustable="box")
    decorate(
        axis,
        "Preparation-service fairness (%)",
        "Inference-service fairness (%)",
        "Expected policy positions under simultaneous heterogeneous contention",
    )
    axis.set_xticks(np.arange(0, 101, 20))
    axis.set_yticks(np.arange(0, 101, 20))

    fig.text(
        0.5,
        0.805,
        "Workload: equal-rate, continuously backlogged tenants; A uses 128 frames + short outputs, "
        "B uses 32 frames + long outputs. Both stages are overloaded; identical trace across policies and seeds.",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.075,
        r"Per-stage score for two tenants:  $F_s=100\left(1-\frac{|S_A^s-S_B^s|}{S_A^s+S_B^s}\right)$.  "
        "100% means equal received service; 0% means one tenant received all service.",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.035,
        "Illustrative mockup—not measured results. Call this a fairness plane unless measurements establish a Pareto tradeoff.",
        ha="center",
        fontsize=9.5,
        color="#555555",
        style="italic",
    )
    fig.suptitle(
        "MOCKUP 5 — Cross-Stage Tenant Fairness Plane\n"
        "Cross-stage scheduling targets the upper-right corner",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "05_cpu_inference_fairness_plane")


def identical_cost_fairness_plane() -> None:
    """Show why homogeneous cost is a control rather than a headline result."""
    expected = {
        "FCFS": (35, 31),
        "Tenant RR": (93, 92),
        "Prep-only": (95, 43),
        "Inference-only": (39, 95),
        "Cross-stage": (97, 97),
    }

    fig, axis = plt.subplots(figsize=(9.2, 7.8))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.19, top=0.76)
    axis.axvspan(85, 100, color="#009E73", alpha=0.055)
    axis.axhspan(85, 100, color="#009E73", alpha=0.055)
    axis.axvline(85, color="#777777", linestyle=":", linewidth=1.2)
    axis.axhline(85, color="#777777", linestyle=":", linewidth=1.2)

    label_offsets = {
        "FCFS": (3, -7),
        "Tenant RR": (-15, 3),
        "Prep-only": (-25, -7),
        "Inference-only": (3, -1),
        "Cross-stage": (-27, 5),
    }
    for policy in POLICIES:
        x_value, y_value = expected[policy]
        axis.scatter(
            x_value,
            y_value,
            s=190 if policy == "Cross-stage" else 145,
            marker=MARKERS[policy],
            color=COLORS[policy],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        dx, dy = label_offsets[policy]
        label = "Cross-stage (ours)" if policy == "Cross-stage" else policy
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=(x_value + dx, y_value + dy),
            fontsize=11,
            fontweight="bold" if policy == "Cross-stage" else "normal",
            arrowprops=dict(arrowstyle="-", color=COLORS[policy], alpha=0.7),
        )

    axis.text(
        71,
        67,
        "Round-robin approaches cross-stage\nbecause equal request turns\napproximate equal service",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    axis.set_xlim(0, 105)
    axis.set_ylim(0, 108)
    axis.set_aspect("equal", adjustable="box")
    decorate(
        axis,
        "Preparation-service fairness (%)",
        "Inference-service fairness (%)",
        "Expected policy positions for identical-cost requests",
    )
    axis.set_xticks(np.arange(0, 101, 20))
    axis.set_yticks(np.arange(0, 101, 20))

    fig.text(
        0.5,
        0.805,
        "Control workload: continuously backlogged A and B; deterministic A:B request rate = 2:1; "
        "identical 32-frame videos, prompts, and output limits.",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.075,
        r"Per-stage score: $F_s=100\left(1-\frac{|S_A^s-S_B^s|}{S_A^s+S_B^s}\right)$. "
        "100% means equal received service.",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.035,
        "Illustrative mockup—not measured results. This control is not evidence of a CPU–GPU Pareto tradeoff.",
        ha="center",
        fontsize=9.5,
        color="#555555",
        style="italic",
    )
    fig.suptitle(
        "MOCKUP 6 — Identical-Cost Tenant Fairness Plane\n"
        "Round-robin approaches max–min when every request has the same cost",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "06_identical_cost_fairness_plane")


def on_off_fairness_plane() -> None:
    """Summarize ON/OFF work-conserving fairness at both stages."""
    expected = {
        "FCFS": (42, 38),
        "Tenant RR": (91, 89),
        "Prep-only": (96, 44),
        "Inference-only": (45, 96),
        "Cross-stage": (97, 97),
    }

    fig, axis = plt.subplots(figsize=(9.2, 7.8))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.20, top=0.75)
    axis.axvspan(85, 100, color="#009E73", alpha=0.055)
    axis.axhspan(85, 100, color="#009E73", alpha=0.055)
    axis.axvline(85, color="#777777", linestyle=":", linewidth=1.2)
    axis.axhline(85, color="#777777", linestyle=":", linewidth=1.2)

    label_offsets = {
        "FCFS": (3, -7),
        "Tenant RR": (-18, -7),
        "Prep-only": (-25, -7),
        "Inference-only": (3, -1),
        "Cross-stage": (-27, 5),
    }
    for policy in POLICIES:
        x_value, y_value = expected[policy]
        axis.scatter(
            x_value,
            y_value,
            s=190 if policy == "Cross-stage" else 145,
            marker=MARKERS[policy],
            color=COLORS[policy],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        dx, dy = label_offsets[policy]
        label = "Cross-stage (ours)" if policy == "Cross-stage" else policy
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=(x_value + dx, y_value + dy),
            fontsize=11,
            fontweight="bold" if policy == "Cross-stage" else "normal",
            arrowprops=dict(arrowstyle="-", color=COLORS[policy], alpha=0.7),
        )

    axis.text(
        72,
        65,
        "The target changes with demand:\nA receives 0.25C while ON and 0 while OFF;\nB receives all remaining capacity",
        ha="center",
        fontsize=10.3,
        color="#555555",
    )
    axis.set_xlim(0, 105)
    axis.set_ylim(0, 108)
    axis.set_aspect("equal", adjustable="box")
    decorate(
        axis,
        "Preparation allocation fairness (%)",
        "Inference allocation fairness (%)",
        "Expected policy positions for an ON/OFF workload",
    )
    axis.set_xticks(np.arange(0, 101, 20))
    axis.set_yticks(np.arange(0, 101, 20))

    fig.text(
        0.5,
        0.795,
        "Workload: identical 32-frame requests; A offers 0.25C for 30 s ON then 0 for 30 s OFF; "
        "B remains backlogged; total target utilization is 100%.",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.075,
        r"Demand-aware score: $F_s=100\left(1-\frac{\sum_i|S_i^s-S_i^{s,*}|}"
        r"{2\sum_i S_i^{s,*}}\right)$, where $S_i^{s,*}$ is the work-conserving max–min target.",
        ha="center",
        fontsize=9.6,
    )
    fig.text(
        0.5,
        0.035,
        "Illustrative mockup—not measured results. Single-stage ablations protect only their controlled stage.",
        ha="center",
        fontsize=9.5,
        color="#555555",
        style="italic",
    )
    fig.suptitle(
        "MOCKUP 7 — ON/OFF Work-Conserving Fairness Plane\n"
        "Fairness follows active demand rather than forcing equal service",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "07_on_off_fairness_plane")


def shifting_contention_service_gap() -> None:
    """Show whether fairness remains bounded as the bottleneck changes."""
    time = np.arange(0, 601, 25, dtype=float)

    def piecewise_gap(slopes, base=0):
        values = np.zeros_like(time)
        for index, point in enumerate(time):
            first = min(point, 200)
            second = min(max(point - 200, 0), 200)
            third = max(point - 400, 0)
            values[index] = base + slopes[0] * first + slopes[1] * second + slopes[2] * third
        values[0] = 0
        return values

    cpu = {
        "FCFS": piecewise_gap((0.72, 0.34, 0.82)),
        "Tenant RR": piecewise_gap((0.35, 0.14, 0.42)),
        "Prep-only": bounded(time, 25, 5, 0.5),
        "Inference-only": piecewise_gap((0.67, 0.30, 0.73)),
        "Cross-stage": bounded(time, 18, 4, 0.0),
    }
    inference = {
        "FCFS": piecewise_gap((0.20, 0.67, 0.75)),
        "Tenant RR": piecewise_gap((0.10, 0.32, 0.39)),
        "Prep-only": piecewise_gap((0.17, 0.62, 0.70)),
        "Inference-only": bounded(time, 23, 5, 0.6),
        "Cross-stage": bounded(time, 17, 4, 0.1),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.13, top=0.62, wspace=0.20)
    phase_specs = [
        (0, 200, "Preparation-heavy", "#E69F00"),
        (200, 400, "Inference-heavy", "#56B4E9"),
        (400, 600, "Both contended", "#CC79A7"),
    ]
    for axis, values, title, ylabel in [
        (
            axes[0],
            cpu,
            "(a) Preparation-service fairness",
            "Preparation-service gap between A and B\n(CPU worker-seconds; lower is fairer)",
        ),
        (
            axes[1],
            inference,
            "(b) Inference-service fairness",
            "Inference-service gap between A and B\n(service-seconds; lower is fairer)",
        ),
    ]:
        for start, end, label, color in phase_specs:
            axis.axvspan(start, end, color=color, alpha=0.055)
            axis.text(
                (start + end) / 2,
                0.98,
                label,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9.5,
                color="#555555",
            )
        axis.axvline(200, color="#777777", linestyle="--", linewidth=1.1)
        axis.axvline(400, color="#777777", linestyle="--", linewidth=1.1)
        for policy in POLICIES:
            axis.plot(
                time,
                values[policy],
                label=policy,
                color=COLORS[policy],
                marker=MARKERS[policy],
                markevery=2,
                linewidth=2.4,
            )
        decorate(axis, "Time (seconds)", ylabel, title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.79),
    )
    fig.text(
        0.5,
        0.685,
        "Workload: two equal-rate, continuously backlogged tenants. Phase 1 varies frame cost; "
        "Phase 2 varies output cost; Phase 3 varies both. Identical trace across policies and seeds.",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.055,
        "Expected result: cross-stage keeps both gaps bounded without identifying the active bottleneck in advance.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
    )
    fig.suptitle(
        "MOCKUP 8 — Fairness Under Shifting Multimodal Contention\n"
        "Cross-stage scheduling keeps tenant service imbalance bounded at both stages",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "08_shifting_contention_service_gap")


def fairness_efficiency_work_conservation() -> None:
    """Combine the fairness/efficiency tradeoff with work conservation."""
    expected = {
        "FCFS": (30, 100.0),
        "Tenant RR": (67, 98.4),
        "Prep-only": (43, 99.0),
        "Inference-only": (40, 98.7),
        "Cross-stage": (95, 97.8),
    }

    time = np.arange(0, 241, 5, dtype=float)
    a_on = ((time // 30) % 2 == 0).astype(float)
    a_received = 0.25 * a_on
    b_received = 1.0 - a_received
    total = a_received + b_received

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.5))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.65, wspace=0.22)

    offsets = {
        "FCFS": (-6, 0.3),
        "Tenant RR": (2, 0.4),
        "Prep-only": (3, 0.5),
        "Inference-only": (2, -0.9),
        "Cross-stage": (-12, -0.4),
    }
    for policy in POLICIES:
        fairness_value, throughput_value = expected[policy]
        axes[0].scatter(
            fairness_value,
            throughput_value,
            s=190 if policy == "Cross-stage" else 145,
            marker=MARKERS[policy],
            color=COLORS[policy],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        dx, dy = offsets[policy]
        label = "Cross-stage (ours)" if policy == "Cross-stage" else policy
        axes[0].annotate(
            label,
            (fairness_value, throughput_value),
            xytext=(fairness_value + dx, throughput_value + dy),
            fontsize=10.5,
            fontweight="bold" if policy == "Cross-stage" else "normal",
            arrowprops=dict(arrowstyle="-", color=COLORS[policy], alpha=0.7),
        )
    axes[0].annotate(
        "Desired region",
        xy=(98, 100),
        xytext=(76, 101.4),
        color="#007A59",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#007A59", linewidth=1.4),
    )
    axes[0].set_xlim(20, 103)
    axes[0].set_ylim(95.5, 102.2)
    decorate(
        axes[0],
        r"Worst-stage fairness, $\min(F_{prep},F_{inference})$ (%)",
        "Normalized aggregate throughput\n(% of FCFS; higher is better)",
        "(a) Fairness–efficiency tradeoff",
    )

    for start in range(0, 240, 60):
        axes[1].axvspan(
            start,
            start + 30,
            color="#D55E00",
            alpha=0.07,
        )
        axes[1].axvspan(
            start + 30,
            start + 60,
            color="#777777",
            alpha=0.05,
        )
    axes[1].stackplot(
        time,
        a_received,
        b_received,
        step="post",
        colors=["#D55E00", "#0072B2"],
        alpha=0.65,
        labels=["Tenant A received", "Tenant B received"],
    )
    axes[1].plot(
        time,
        total,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Total utilized capacity",
    )
    axes[1].set_ylim(0, 1.08)
    decorate(
        axes[1],
        "Time (seconds)",
        "Received service rate\n(fraction of available capacity)",
        "(b) Work conservation under ON/OFF demand",
    )
    axes[1].legend(frameon=False, loc="lower right", fontsize=9.5)

    fig.text(
        0.5,
        0.715,
        "Pareto workload: equal-rate, backlogged tenants with 32/128-frame videos and short/long outputs. "
        "ON/OFF check: A demands 0.25C while ON; B is always backlogged.",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.065,
        "Expected conclusion: cross-stage approaches the upper-right fairness/throughput region and reallocates unused service instead of idling capacity.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
    )
    fig.suptitle(
        "MOCKUP 9 — Fairness, Efficiency, and Work Conservation\n"
        "Cross-stage fairness targets tenant isolation with little throughput cost",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "09_fairness_efficiency_work_conservation")


def normalized_throughput_cost() -> None:
    """Show the expected throughput cost of each fairness policy."""
    throughput = {
        "FCFS": 100.0,
        "Tenant RR": 98.4,
        "Prep-only": 99.0,
        "Inference-only": 98.7,
        "Cross-stage": 97.8,
    }
    values = [throughput[policy] for policy in POLICIES]
    colors = [COLORS[policy] for policy in POLICIES]
    labels = ["Cross-stage\n(ours)" if policy == "Cross-stage" else policy for policy in POLICIES]

    fig, axis = plt.subplots(figsize=(9.8, 6.6))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.18, top=0.68)
    bars = axis.bar(labels, values, color=colors, width=0.66, alpha=0.88)
    axis.axhline(100, color="black", linestyle="--", linewidth=1.4, label="FCFS reference")
    for bar, value in zip(bars, values):
        loss = 100.0 - value
        annotation = "100.0%" if loss == 0 else f"{value:.1f}%\n({loss:.1f}% cost)"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.25,
            annotation,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold" if value == throughput["Cross-stage"] else "normal",
        )
    axis.set_ylim(94, 102)
    decorate(
        axis,
        "Scheduling policy",
        "Normalized aggregate throughput\n(% of FCFS; higher is better)",
        "Expected throughput under heterogeneous multimodal contention",
    )
    axis.legend(frameon=False, loc="lower left")
    fig.text(
        0.5,
        0.745,
        "Workload: two equal-rate, continuously backlogged tenants with heterogeneous frame and output costs; "
        "identical trace across policies and seeds.",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.065,
        "Expected conclusion: cross-stage fairness substantially improves isolation while retaining throughput close to FCFS.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
    )
    fig.suptitle(
        "MOCKUP 10 — Throughput Cost of Fair Scheduling\n"
        "Fairness should not substantially reduce completed requests per second",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "10_normalized_throughput_cost")


def standalone_fairness_throughput_pareto() -> None:
    """Standalone expected fairness/throughput plane."""
    expected = {
        "FCFS": (30, 100.0),
        "Tenant RR": (67, 98.4),
        "Prep-only": (43, 99.0),
        "Inference-only": (40, 98.7),
        "Cross-stage": (95, 97.8),
    }
    fig, axis = plt.subplots(figsize=(9.6, 7.0))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.72)
    offsets = {
        "FCFS": (-7, 0.35),
        "Tenant RR": (2, 0.35),
        "Prep-only": (2, 0.5),
        "Inference-only": (2, -0.8),
        "Cross-stage": (-18, -0.65),
    }
    for policy in POLICIES:
        x_value, y_value = expected[policy]
        axis.scatter(
            x_value,
            y_value,
            s=190 if policy == "Cross-stage" else 145,
            marker=MARKERS[policy],
            color=COLORS[policy],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        dx, dy = offsets[policy]
        label = "Cross-stage (ours)" if policy == "Cross-stage" else policy
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=(x_value + dx, y_value + dy),
            fontsize=11,
            fontweight="bold" if policy == "Cross-stage" else "normal",
            arrowprops=dict(arrowstyle="-", color=COLORS[policy], alpha=0.7),
        )
    axis.axhline(100, color="black", linestyle="--", linewidth=1.2, label="FCFS throughput")
    axis.annotate(
        "Desired region",
        xy=(98, 100),
        xytext=(72, 101.4),
        color="#007A59",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#007A59", linewidth=1.4),
    )
    axis.set_xlim(20, 103)
    axis.set_ylim(95.5, 102.2)
    decorate(
        axis,
        r"Worst-stage fairness, $\min(F_{prep},F_{inference})$ (%)",
        "Normalized aggregate throughput\n(% of FCFS; higher is better)",
        "Expected fairness–efficiency tradeoff",
    )
    axis.legend(frameon=False, loc="lower right")
    fig.text(
        0.5,
        0.78,
        "Workload: equal-rate, continuously backlogged tenants; A uses 128 frames + short outputs, "
        "B uses 32 frames + long outputs; both stages are overloaded.",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.065,
        "Illustrative mockup—not measured results. Upper-right means high fairness at both stages with little throughput cost.",
        ha="center",
        fontsize=9.8,
        color="#555555",
        style="italic",
    )
    fig.suptitle(
        "MOCKUP 11 — Cross-Stage Fairness–Throughput Plane\n"
        "Fairness gain versus aggregate performance cost",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "11_fairness_throughput_pareto")


def standalone_work_conservation_trace() -> None:
    """Standalone ON/OFF capacity-allocation trace."""
    time = np.arange(0, 241, 5, dtype=float)
    a_on = ((time // 30) % 2 == 0).astype(float)
    a_received = 0.25 * a_on
    b_received = 1.0 - a_received
    total = a_received + b_received

    fig, axis = plt.subplots(figsize=(10.8, 6.2))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.18, top=0.68)
    for start in range(0, 240, 60):
        axis.axvspan(start, start + 30, color="#D55E00", alpha=0.07)
        axis.axvspan(start + 30, start + 60, color="#777777", alpha=0.05)
        axis.text(start + 15, 1.035, "A ON", ha="center", fontsize=9, color="#8C3B00")
        axis.text(start + 45, 1.035, "A OFF", ha="center", fontsize=9, color="#555555")
    axis.stackplot(
        time,
        a_received,
        b_received,
        step="post",
        colors=["#D55E00", "#0072B2"],
        alpha=0.68,
        labels=["Tenant A received", "Tenant B received"],
    )
    axis.plot(
        time,
        total,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Total utilized capacity",
    )
    axis.set_ylim(0, 1.08)
    decorate(
        axis,
        "Time (seconds)",
        "Received service rate\n(fraction of available capacity)",
        "Cross-stage max–min reallocates unused service",
    )
    axis.legend(frameon=False, loc="lower right")
    fig.text(
        0.5,
        0.745,
        "Workload: identical 32-frame requests. A offers 0.25C for 30 s ON and zero for 30 s OFF; "
        "B remains backlogged. The dashed total should remain at 1.0.",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.065,
        "Illustrative mockup—not measured results. When A is idle, B receives the released capacity instead of leaving the stage idle.",
        ha="center",
        fontsize=9.8,
        color="#555555",
        style="italic",
    )
    fig.suptitle(
        "MOCKUP 12 — Work Conservation Under ON/OFF Demand\n"
        "Fair sharing does not reserve unused capacity for an idle tenant",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "12_work_conservation_trace")


def heterogeneity_fairness_sweep() -> None:
    """Show how service fairness changes as request costs diverge."""
    ratios = np.asarray([1, 2, 4, 8], dtype=float)
    scores = {
        "FCFS": [96, 72, 46, 25],
        "Tenant RR": [97, 79, 57, 36],
        "Prep-only": [96, 76, 53, 34],
        "Inference-only": [96, 75, 52, 33],
        "Cross-stage": [97, 96, 94, 92],
    }
    fig, axis = plt.subplots(figsize=(10.2, 6.7))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.18, top=0.70)
    for policy in POLICIES:
        axis.plot(
            ratios,
            scores[policy],
            color=COLORS[policy],
            marker=MARKERS[policy],
            markersize=8,
            linewidth=2.5,
            label="Cross-stage (ours)" if policy == "Cross-stage" else policy,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(ratios, ["1×", "2×", "4×", "8×"])
    axis.set_ylim(0, 103)
    decorate(
        axis,
        "Tenant request-cost ratio (expensive / inexpensive)",
        r"Worst-stage fairness, $\min(F_{prep},F_{inference})$ (%)",
        "Expected robustness to heterogeneous multimodal request costs",
    )
    axis.legend(frameon=False, ncol=2, loc="lower left")
    fig.text(
        0.5,
        0.765,
        "Workload: two equal-rate, continuously backlogged tenants. Increase A:B preparation and output-cost ratios "
        "while holding the arrival trace fixed across policies and seeds.",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.065,
        "Illustrative mockup—not measured results. Equal request turns become less fair as service costs diverge.",
        ha="center",
        fontsize=9.8,
        color="#555555",
        style="italic",
    )
    fig.suptitle(
        "MOCKUP 13 — Fairness Versus Request-Cost Heterogeneity\n"
        "Cross-stage accounting should remain stable as CPU and inference costs diverge",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    finish(fig, "13_heterogeneity_fairness_sweep")


def main() -> None:
    identical_cost_rate_asymmetry()
    constant_rate_service_gap()
    on_off_work_conservation()
    heterogeneous_video_cost()
    constant_underloaded_overloaded()
    cpu_inference_fairness_plane()
    identical_cost_fairness_plane()
    on_off_fairness_plane()
    shifting_contention_service_gap()
    fairness_efficiency_work_conservation()
    normalized_throughput_cost()
    standalone_fairness_throughput_pareto()
    standalone_work_conservation_trace()
    heterogeneity_fairness_sweep()
    print(OUT)


if __name__ == "__main__":
    main()
