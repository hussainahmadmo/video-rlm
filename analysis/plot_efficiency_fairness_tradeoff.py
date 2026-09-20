#!/usr/bin/env python3
"""Plot efficiency versus light-tenant protection for a noisy-neighbor run."""
import argparse
import json
from pathlib import Path


LABELS = {
    'static_fcfs': 'Static FCFS',
    'adaptive_fcfs': 'Adaptive FCFS',
    'current_full_conductor': 'Current Conductor',
    'efficient_full_conductor': 'Efficient Conductor',
}
COLORS = {
    'static_fcfs': '#777777',
    'adaptive_fcfs': '#4c78a8',
    'current_full_conductor': '#e45756',
    'efficient_full_conductor': '#2ca02c',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--light-tenants', nargs='+', default=['b', 'c'])
    args = parser.parse_args()

    rows = json.loads(args.comparison.read_text())
    by_name = {row['variant']: row for row in rows}
    baseline = by_name['static_fcfs']
    points = []
    for name in LABELS:
        row = by_name[name]
        ratios = {
            tenant: (
                row['tenants'][tenant]['end_to_end_s']['p95'] /
                baseline['tenants'][tenant]['end_to_end_s']['p95']
            )
            for tenant in args.light_tenants
        }
        points.append({
            'variant': name,
            'throughput_qps': row['throughput_qps'],
            'worst_light_p95_ratio': max(ratios.values()),
            'tenant_p95_ratios': ratios,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'tradeoff_points.json').write_text(
        json.dumps(points, indent=2) + '\n'
    )

    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/efficient-fairness-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    offsets = {
        'static_fcfs': (-68, 7),
        'adaptive_fcfs': (5, 7),
        'current_full_conductor': (5, 7),
        'efficient_full_conductor': (5, 7),
    }
    for point in points:
        name = point['variant']
        x = point['throughput_qps']
        y = point['worst_light_p95_ratio']
        ax.scatter(x, y, s=70, color=COLORS[name], edgecolor='white',
                   linewidth=.8, zorder=3)
        ax.annotate(LABELS[name], (x, y), xytext=offsets[name],
                    textcoords='offset points', fontsize=8)

    # Current and efficient Conductor form the measured local tradeoff.
    current = next(p for p in points if p['variant'] == 'current_full_conductor')
    efficient = next(p for p in points if p['variant'] == 'efficient_full_conductor')
    ax.plot(
        [current['throughput_qps'], efficient['throughput_qps']],
        [current['worst_light_p95_ratio'], efficient['worst_light_p95_ratio']],
        color='#777777', linewidth=1, linestyle='--', zorder=1,
    )
    ax.axhline(1, color='#bbbbbb', linewidth=1, linestyle=':')
    ax.margins(x=.08, y=.12)
    ax.set_xlabel('Aggregate throughput (requests/s)  →', fontsize=9)
    ax.set_ylabel('Worst light-tenant p95 / static FCFS  (lower ↓)', fontsize=9)
    ax.grid(alpha=.18, zorder=0)
    ax.set_title('Noisy neighbor: efficiency–protection tradeoff', fontsize=9)
    fig.tight_layout()
    for extension in ('pdf', 'png'):
        fig.savefig(args.output_dir / f'efficiency_fairness_tradeoff.{extension}',
                    dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
