#!/usr/bin/env python3
"""Mock up two-tenant ON/OFF fairness and work conservation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    time_s = np.arange(0, 241, 1, dtype=float)
    on = (time_s % 60) < 30

    # Illustrative normalized demand and service rates, not measurements.
    demand_a = np.where(on, 1.0, 0.0)
    demand_b = np.ones_like(time_s)
    service_a = np.where(on, 0.5, 0.0)
    service_b = np.where(on, 0.5, 1.0)

    blue = "#0072b2"
    orange = "#e69f00"
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 8.0), sharex=True)

    axes[0].step(time_s, demand_a, where="post", color=blue, linewidth=2.4, label="Tenant A (ON/OFF)")
    axes[0].step(time_s, demand_b, where="post", color=orange, linewidth=2.4, label="Tenant B (always ON)")
    axes[0].set_title("(a) Offered request demand", fontweight="bold")
    axes[0].set_ylabel("Offered load\n(fraction of capacity)")

    for ax, title, ylabel in [
        (axes[1], "(b) Preparation service allocation", "Fraction of preparation\ncapacity received"),
        (axes[2], "(c) Inference service allocation", "Fraction of inference\ncapacity received"),
    ]:
        ax.step(time_s, service_a, where="post", color=blue, linewidth=2.4, label="Tenant A (ON/OFF)")
        ax.step(time_s, service_b, where="post", color=orange, linewidth=2.4, label="Tenant B (always ON)")
        ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1.4, label="Equal share")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)

    for ax in axes:
        for start in (0, 60, 120, 180):
            ax.axvspan(start, start + 30, color=blue, alpha=0.06)
        ax.set_ylim(-0.04, 1.10)
        ax.set_xlim(0, 240)
        ax.grid(alpha=0.22)

    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_xticks(np.arange(0, 241, 30))
    axes[0].text(15, 1.045, "A ON", ha="center", va="bottom", color=blue, fontweight="bold")
    axes[0].text(45, 1.045, "A OFF", ha="center", va="bottom", color="#555555", fontweight="bold")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "MOCKUP — Two-tenant ON/OFF fairness and work conservation\n"
        "Identical 32-frame requests; 30 s ON / 30 s OFF",
        y=0.995,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))

    output = Path("analysis/figures/on_off_work_conservation_mockup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
