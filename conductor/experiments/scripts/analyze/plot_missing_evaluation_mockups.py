#!/usr/bin/env python3
"""Generate illustrative mockups for the remaining fairness evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("analysis/figures/evaluation_mockups")
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
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def decorate(axis, *, xlabel: str, ylabel: str, title: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def mockup_note(fig) -> None:
    fig.suptitle("Illustrative mockup — not measured results", fontsize=10, color="#555555")


def victim_isolation() -> None:
    values = [5.5, 3.2, 1.6, 5.3, 1.4]
    fig, axis = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    bars = axis.bar(POLICIES, values, color=[COLORS[p] for p in POLICIES])
    axis.bar_label(bars, fmt="%.1fx", padding=3)
    axis.axhline(1, color="black", linestyle="--", linewidth=1, label="No-contention latency")
    axis.set_ylim(0, 6.2)
    decorate(axis, xlabel="Scheduling policy", ylabel="Victim P95 slowdown (x)", title="1. Aggressive-tenant isolation")
    axis.legend(frameon=False)
    mockup_note(fig)
    finish(fig, "01_victim_isolation")


def moving_bottleneck() -> None:
    cases = ["Preparation-heavy", "Inference-heavy", "Mixed"]
    values = {
        "FCFS": [5.5, 4.8, 5.8],
        "Tenant RR": [3.1, 3.0, 3.5],
        "Prep-only": [1.4, 4.5, 3.2],
        "Inference-only": [5.1, 1.4, 3.4],
        "Cross-stage": [1.3, 1.3, 1.5],
    }
    x = np.arange(len(cases))
    width = 0.16
    fig, axis = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    for index, policy in enumerate(POLICIES):
        axis.bar(x + (index - 2) * width, values[policy], width, label=policy, color=COLORS[policy])
    axis.set_xticks(x, cases)
    decorate(axis, xlabel="Bottleneck location", ylabel="Victim P95 slowdown (x)", title="2. Fairness when the bottleneck moves")
    axis.legend(frameon=False, ncol=3)
    mockup_note(fig)
    finish(fig, "02_moving_bottleneck")


def load_sweep() -> None:
    load = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
    prep_gap = {
        "FCFS": [8, 35, 120, 310, 620],
        "Tenant RR": [7, 25, 80, 180, 330],
        "Prep-only": [6, 12, 20, 28, 38],
        "Inference-only": [8, 34, 115, 290, 580],
        "Cross-stage": [6, 11, 18, 24, 32],
    }
    gpu_gap = {
        "FCFS": [5, 25, 90, 230, 480],
        "Tenant RR": [5, 18, 60, 140, 260],
        "Prep-only": [5, 24, 85, 215, 450],
        "Inference-only": [5, 10, 17, 25, 35],
        "Cross-stage": [5, 9, 15, 21, 29],
    }
    victim = {
        "FCFS": [1.0, 1.2, 2.2, 4.5, 8.0],
        "Tenant RR": [1.0, 1.1, 1.8, 3.0, 5.0],
        "Prep-only": [1.0, 1.0, 1.2, 1.5, 2.0],
        "Inference-only": [1.0, 1.2, 2.1, 4.2, 7.4],
        "Cross-stage": [1.0, 1.0, 1.1, 1.3, 1.6],
    }
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.5), constrained_layout=True)
    for policy in POLICIES:
        style = dict(color=COLORS[policy], marker=MARKERS[policy], linewidth=2)
        axes[0].plot(load, prep_gap[policy], label=policy, **style)
        axes[1].plot(load, gpu_gap[policy], label=policy, **style)
        axes[2].plot(load, victim[policy], label=policy, **style)
    decorate(axes[0], xlabel="Offered load / measured capacity", ylabel="CPU service gap (worker-s)", title="(a) Preparation fairness")
    decorate(axes[1], xlabel="Offered load / measured capacity", ylabel="GPU service gap (service-s)", title="(b) Inference fairness")
    decorate(axes[2], xlabel="Offered load / measured capacity", ylabel="Victim P95 slowdown (x)", title="(c) End-to-end user impact")
    axes[0].axvline(1, color="black", linestyle="--", linewidth=1)
    axes[1].axvline(1, color="black", linestyle="--", linewidth=1)
    axes[2].axvline(1, color="black", linestyle="--", linewidth=1, label="Capacity")
    axes[2].legend(frameon=False, ncol=2)
    mockup_note(fig)
    finish(fig, "03_load_sensitivity")


def system_cost() -> None:
    metrics = [
        ("Throughput", "Requests/s", [0.160, 0.158, 0.159, 0.157, 0.159]),
        ("CPU utilization", "Percent", [89, 88, 89, 88, 89]),
        ("GPU utilization", "Percent", [82, 81, 82, 81, 82]),
        ("Scheduler overhead", "Milliseconds/request", [0.05, 0.10, 0.14, 0.13, 0.22]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), constrained_layout=True)
    for axis, (title, ylabel, values) in zip(axes.flat, metrics):
        axis.bar(POLICIES, values, color=[COLORS[p] for p in POLICIES])
        axis.tick_params(axis="x", rotation=18)
        decorate(axis, xlabel="Policy", ylabel=ylabel, title=title)
    mockup_note(fig)
    finish(fig, "04_system_cost")


def inflight_accounting() -> None:
    concurrency = np.array([1, 2, 4, 8])
    variants = {
        "Completion-only": ("#D55E00", [25, 90, 240, 560]),
        "Fixed reservation": ("#E69F00", [22, 45, 85, 150]),
        "Profiled + reconciled": ("#009E73", [20, 24, 30, 42]),
    }
    fig, axis = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    for label, (color, values) in variants.items():
        axis.plot(concurrency, values, marker="o", linewidth=2.5, color=color, label=label)
    axis.set_xticks(concurrency)
    decorate(axis, xlabel="Concurrent preparation workers", ylabel="CPU service gap (worker-s)", title="5. In-flight accounting")
    axis.legend(frameon=False)
    mockup_note(fig)
    finish(fig, "05_inflight_accounting")


def profiler_accuracy() -> None:
    predicted = np.array([0.4, 0.7, 1.0, 1.4, 1.8, 2.3, 2.9, 3.5, 4.2, 5.0])
    observed = np.array([0.5, 0.65, 1.15, 1.3, 2.0, 2.15, 3.15, 3.3, 4.45, 4.8])
    fig, axis = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    axis.scatter(predicted, observed, s=65, color="#0072B2", label="Requests")
    axis.plot([0, 5.4], [0, 5.4], color="black", linestyle="--", label="Perfect prediction")
    axis.set_xlim(0, 5.4)
    axis.set_ylim(0, 5.4)
    decorate(axis, xlabel="Predicted preparation service (s)", ylabel="Observed preparation service (s)", title="6. Online cost-profiler accuracy")
    axis.legend(frameon=False)
    mockup_note(fig)
    finish(fig, "06_profiler_accuracy")


def stochastic_arrivals() -> None:
    rng = np.random.default_rng(7)
    time = np.arange(0, 601, 50)
    endpoints = {
        "FCFS": (620, 350),
        "Tenant RR": (330, 220),
        "Prep-only": (40, 310),
        "Inference-only": (580, 15),
        "Cross-stage": (32, 12),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), constrained_layout=True)
    for policy in POLICIES:
        cpu_end, gpu_end = endpoints[policy]
        base = time / time[-1]
        cpu = cpu_end * base + rng.normal(0, max(cpu_end * 0.025, 3), len(time))
        gpu = gpu_end * base + rng.normal(0, max(gpu_end * 0.025, 1.5), len(time))
        cpu[0] = gpu[0] = 0
        axes[0].plot(time, np.maximum(cpu, 0), color=COLORS[policy], marker=MARKERS[policy], label=policy)
        axes[1].plot(time, np.maximum(gpu, 0), color=COLORS[policy], marker=MARKERS[policy], label=policy)
    decorate(axes[0], xlabel="Time (s)", ylabel="CPU service gap (worker-s)", title="(a) Poisson heterogeneous workload")
    decorate(axes[1], xlabel="Time (s)", ylabel="GPU service gap (service-s)", title="(b) Poisson heterogeneous workload")
    axes[1].legend(frameon=False, ncol=2)
    mockup_note(fig)
    finish(fig, "07_stochastic_arrivals")


def tenant_scalability() -> None:
    tenants = np.array([2, 4, 8, 16])
    p95 = {
        "FCFS": [2.0, 3.0, 5.0, 8.5],
        "Tenant RR": [1.7, 2.4, 3.5, 5.5],
        "Prep-only": [1.4, 1.7, 2.2, 3.2],
        "Inference-only": [1.8, 2.8, 4.5, 7.5],
        "Cross-stage": [1.3, 1.4, 1.6, 2.0],
    }
    overhead = [0.10, 0.14, 0.22, 0.38]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for policy in POLICIES:
        axes[0].plot(tenants, p95[policy], color=COLORS[policy], marker=MARKERS[policy], label=policy)
    axes[1].plot(tenants, overhead, color=COLORS["Cross-stage"], marker="o", linewidth=2.5)
    axes[0].set_xticks(tenants)
    axes[1].set_xticks(tenants)
    decorate(axes[0], xlabel="Number of tenants", ylabel="Worst-tenant P95 slowdown (x)", title="(a) Isolation at scale")
    decorate(axes[1], xlabel="Number of tenants", ylabel="Scheduling overhead (ms/request)", title="(b) Scheduler scalability")
    axes[0].legend(frameon=False, ncol=2)
    mockup_note(fig)
    finish(fig, "08_tenant_scalability")


def both_tenants_cross_stage_overload() -> None:
    # A and B are continuously backlogged at every point. Their offered-service
    # rates have a fixed 1.5:1 ratio; even B exceeds its 50% fair share.
    load = np.array([1.3, 1.5, 1.7, 1.9, 2.1])
    cpu_gap = {
        "FCFS": [90, 170, 290, 470, 720],
        "Tenant RR": [50, 90, 155, 250, 390],
        "Prep-only": [18, 22, 27, 34, 44],
        "Inference-only": [85, 160, 275, 445, 680],
        "Cross-stage": [15, 18, 22, 27, 34],
    }
    gpu_gap = {
        "FCFS": [60, 110, 190, 310, 490],
        "Tenant RR": [34, 60, 105, 175, 275],
        "Prep-only": [56, 103, 180, 295, 460],
        "Inference-only": [12, 15, 19, 25, 32],
        "Cross-stage": [10, 13, 16, 20, 26],
    }
    slowdown = {
        "FCFS": [2.2, 3.4, 5.2, 7.8, 11.5],
        "Tenant RR": [1.7, 2.3, 3.3, 4.8, 7.0],
        "Prep-only": [1.5, 1.8, 2.3, 3.1, 4.4],
        "Inference-only": [2.0, 3.1, 4.7, 7.0, 10.2],
        "Cross-stage": [1.3, 1.4, 1.6, 2.0, 2.6],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.8), constrained_layout=True)
    for policy in POLICIES:
        style = dict(color=COLORS[policy], marker=MARKERS[policy], linewidth=2)
        axes[0].plot(load, cpu_gap[policy], label=policy, **style)
        axes[1].plot(load, gpu_gap[policy], label=policy, **style)
        axes[2].plot(load, slowdown[policy], label=policy, **style)

    decorate(
        axes[0],
        xlabel="Total offered load / end-to-end capacity",
        ylabel="CPU service gap (worker-s)",
        title="(a) Preparation fairness",
    )
    decorate(
        axes[1],
        xlabel="Total offered load / end-to-end capacity",
        ylabel="GPU service gap (service-s)",
        title="(b) Inference fairness",
    )
    decorate(
        axes[2],
        xlabel="Total offered load / end-to-end capacity",
        ylabel="Worst-tenant P95 slowdown (x)",
        title="(c) End-to-end isolation",
    )
    axes[2].legend(frameon=False, ncol=2)
    fig.suptitle(
        "Both Tenants and Both Pipeline Stages Are Overloaded\n"
        "A:B offered-service ratio = 1.5:1; both tenants remain backlogged",
        fontsize=13,
        fontweight="bold",
    )
    finish(fig, "09_both_tenants_cross_stage_overload")


def contact_sheet() -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13.0, 17.0), constrained_layout=True)
    # 1: victim isolation
    axes[0, 0].bar(POLICIES, [5.5, 3.2, 1.6, 5.3, 1.4], color=[COLORS[p] for p in POLICIES])
    decorate(axes[0, 0], xlabel="Policy", ylabel="Victim slowdown (x)", title="1. Isolation")
    # 2: moving bottleneck, cross-stage summary versus single-stage policies
    cases = np.arange(3)
    axes[0, 1].plot(cases, [1.4, 4.5, 3.2], marker="o", color=COLORS["Prep-only"], label="Prep-only")
    axes[0, 1].plot(cases, [5.1, 1.4, 3.4], marker="^", color=COLORS["Inference-only"], label="Inference-only")
    axes[0, 1].plot(cases, [1.3, 1.3, 1.5], marker="v", color=COLORS["Cross-stage"], label="Cross-stage")
    axes[0, 1].set_xticks(cases, ["Prep", "Inference", "Mixed"])
    decorate(axes[0, 1], xlabel="Bottleneck", ylabel="Victim slowdown (x)", title="2. Moving bottleneck")
    axes[0, 1].legend(frameon=False)
    # 3: load
    load = [0.5, 0.8, 1.0, 1.2, 1.5]
    axes[1, 0].plot(load, [1, 1.2, 2.2, 4.5, 8], marker="s", color=COLORS["FCFS"], label="FCFS")
    axes[1, 0].plot(load, [1, 1, 1.1, 1.3, 1.6], marker="v", color=COLORS["Cross-stage"], label="Cross-stage")
    decorate(axes[1, 0], xlabel="Offered load / capacity", ylabel="Victim slowdown (x)", title="3. Load sensitivity")
    axes[1, 0].legend(frameon=False)
    # 4: cost
    axes[1, 1].bar(POLICIES, [0.160, 0.158, 0.159, 0.157, 0.159], color=[COLORS[p] for p in POLICIES])
    decorate(axes[1, 1], xlabel="Policy", ylabel="Throughput (requests/s)", title="4. Fairness cost")
    # 5: in-flight
    conc = [1, 2, 4, 8]
    axes[2, 0].plot(conc, [25, 90, 240, 560], marker="o", color="#D55E00", label="Completion-only")
    axes[2, 0].plot(conc, [20, 24, 30, 42], marker="o", color="#009E73", label="Profiled + reconciled")
    decorate(axes[2, 0], xlabel="Concurrency", ylabel="CPU service gap (worker-s)", title="5. In-flight accounting")
    axes[2, 0].legend(frameon=False)
    # 6: profiler
    pred = np.array([0.4, 0.8, 1.3, 1.9, 2.5, 3.2, 4.1, 5.0])
    obs = np.array([0.5, 0.7, 1.5, 1.8, 2.7, 3.0, 4.3, 4.8])
    axes[2, 1].scatter(pred, obs, color="#0072B2")
    axes[2, 1].plot([0, 5.2], [0, 5.2], color="black", linestyle="--")
    decorate(axes[2, 1], xlabel="Predicted service (s)", ylabel="Observed service (s)", title="6. Profiler accuracy")
    # 7: stochastic
    time = np.arange(0, 601, 50)
    axes[3, 0].plot(time, 620 * time / 600, color=COLORS["FCFS"], label="FCFS")
    bounded_gap = 30 + 6 * np.sin(time / 50)
    bounded_gap[0] = 0
    axes[3, 0].plot(time, bounded_gap, color=COLORS["Cross-stage"], label="Cross-stage")
    decorate(axes[3, 0], xlabel="Time (s)", ylabel="CPU service gap (worker-s)", title="7. Stochastic robustness")
    axes[3, 0].legend(frameon=False)
    # 8: scale
    tenant_count = [2, 4, 8, 16]
    axes[3, 1].plot(tenant_count, [0.10, 0.14, 0.22, 0.38], marker="o", color=COLORS["Cross-stage"])
    decorate(axes[3, 1], xlabel="Number of tenants", ylabel="Overhead (ms/request)", title="8. Tenant scalability")
    for axis in axes.flat:
        axis.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Missing Evaluation Experiments — Mockup Sequence (Not Measured)", fontsize=17, fontweight="bold")
    finish(fig, "00_all_missing_experiments")


def main() -> None:
    victim_isolation()
    moving_bottleneck()
    load_sweep()
    system_cost()
    inflight_accounting()
    profiler_accuracy()
    stochastic_arrivals()
    tenant_scalability()
    both_tenants_cross_stage_overload()
    contact_sheet()
    print(OUT)


if __name__ == "__main__":
    main()
