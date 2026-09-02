#!/usr/bin/env python3
"""Create a simple ON/OFF received-service-versus-time mockup."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    time_s = np.arange(0, 241, 1, dtype=float)
    tenant_a_on = (time_s % 60) < 30

    # Normalized service rate: both split service while A is ON; B receives
    # all service while A is OFF. Illustrative values, not measurements.
    service_a = np.where(tenant_a_on, 0.5, 0.0)
    service_b = np.where(tenant_a_on, 0.5, 1.0)

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.step(
        time_s,
        service_a,
        where="post",
        color="#0072b2",
        linewidth=2.8,
        label="Tenant A (ON/OFF)",
    )
    ax.step(
        time_s,
        service_b,
        where="post",
        color="#e69f00",
        linewidth=2.8,
        label="Tenant B (always backlogged)",
    )
    ax.axhline(
        0.5,
        color="#777777",
        linestyle=":",
        linewidth=1.6,
        label="Equal share when both are active",
    )
    for start in (0, 60, 120, 180):
        ax.axvspan(start, start + 30, color="#0072b2", alpha=0.07)

    ax.text(15, 1.04, "A ON", ha="center", color="#0072b2", fontweight="bold")
    ax.text(45, 1.04, "A OFF", ha="center", color="#555555", fontweight="bold")
    ax.set_title(
        "MOCKUP — Received service over time under cross-stage max-min\n"
        "Tenant A alternates 30 s ON / 30 s OFF",
        fontweight="bold",
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Received service rate\n(fraction of stage capacity)")
    ax.set_xlim(0, 240)
    ax.set_ylim(-0.04, 1.10)
    ax.set_xticks(np.arange(0, 241, 30))
    ax.grid(alpha=0.24)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    output = Path("analysis/figures/on_off_service_over_time_mockup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
