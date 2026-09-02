#!/usr/bin/env python3
"""VTC-style fairness mockup for heterogeneous multimodal requests."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def service_rates(time_s: np.ndarray, phase: float) -> tuple[np.ndarray, np.ndarray]:
    warmup = 1.0 - np.exp(-time_s / 25.0)
    variation = 0.025 * np.sin(time_s / 33.0 + phase)
    tenant_a = warmup * (0.5 + variation)
    tenant_b = warmup * (0.5 - variation)
    return tenant_a, tenant_b


def bounded_difference(time_s: np.ndarray, scale: float, offset: float = 1.0) -> np.ndarray:
    return scale * (0.025 * offset + 0.012 * np.sin(time_s / 38.0) ** 2)


def growing_difference(time_s: np.ndarray, scale: float, factor: float) -> np.ndarray:
    return factor * scale * (0.00065 * time_s + 0.0000007 * time_s**2)


def main() -> None:
    time_s = np.arange(0, 601, 10, dtype=float)
    tenant_a_color = "#0072b2"
    tenant_b_color = "#e69f00"
    maxmin_color = "#009e73"
    fcfs_color = "#d55e00"

    prep_a, prep_b = service_rates(time_s, 0.0)
    infer_a, infer_b = service_rates(time_s, 1.2)
    prep_differences = {
        "FCFS": growing_difference(time_s, 100.0, 1.0),
        "Tenant round-robin": growing_difference(time_s, 100.0, 0.62),
        "Preparation-only": bounded_difference(time_s, 100.0, 1.35),
        "Inference-only / VTC": growing_difference(time_s, 100.0, 0.88),
        "Cross-stage max-min": bounded_difference(time_s, 100.0),
    }
    infer_differences = {
        "FCFS": growing_difference(time_s, 80.0, 1.0),
        "Tenant round-robin": growing_difference(time_s, 80.0, 0.62),
        "Preparation-only": growing_difference(time_s, 80.0, 0.88),
        "Inference-only / VTC": bounded_difference(time_s, 80.0, 1.35),
        "Cross-stage max-min": bounded_difference(time_s, 80.0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True)
    rate_panels = [
        (axes[0, 0], prep_a, prep_b, "(a) Preparation service rate"),
        (axes[1, 0], infer_a, infer_b, "(c) Inference service rate"),
    ]
    for ax, tenant_a, tenant_b, title in rate_panels:
        ax.plot(
            time_s,
            tenant_a,
            marker="v",
            markevery=4,
            color=tenant_a_color,
            linewidth=2.2,
            label="Tenant A: high-rate, 32 frames",
        )
        ax.plot(
            time_s,
            tenant_b,
            marker="s",
            markevery=4,
            color=tenant_b_color,
            linewidth=2.2,
            label="Tenant B: low-rate, 128 frames",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Received service rate\n(normalized; 60 s window)")
        ax.set_ylim(0, 0.62)

    difference_panels = [
        (
            axes[0, 1],
            prep_differences,
            "(b) Preparation service difference",
            "Cumulative difference\n(worker-seconds)",
        ),
        (
            axes[1, 1],
            infer_differences,
            "(d) Inference service difference",
            "Cumulative difference\n(inference-service units)",
        ),
    ]
    policy_styles = {
        "FCFS": (fcfs_color, "s", "-"),
        "Tenant round-robin": ("#cc79a7", "D", "--"),
        "Preparation-only": ("#e69f00", "o", "-."),
        "Inference-only / VTC": ("#56b4e9", "^", ":"),
        "Cross-stage max-min": (maxmin_color, "v", "-"),
    }
    for ax, differences, title, ylabel in difference_panels:
        for policy, values in differences.items():
            color, marker, linestyle = policy_styles[policy]
            ax.plot(
                time_s,
                values,
                marker=marker,
                markevery=6,
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
                markersize=4.2,
                label=policy,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)

    for ax in axes.flat:
        ax.set_xlim(0, 600)
        ax.set_xlabel("Time (seconds)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(
        "MOCKUP — Cross-stage fairness with stochastic heterogeneous requests\n"
        "Both tenants backlogged; Poisson arrivals (CV=1); illustrative values only",
        y=1.01,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output = Path("analysis/figures/vtc_style_cross_stage_fairness_mockup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
