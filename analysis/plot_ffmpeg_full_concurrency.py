"""Plot the complete sustained-backlog FFmpeg concurrency sweep."""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    ROOT / 'large_sweeps/ffmpeg_worker_sweep_cpu64_requests96_20260918/summary.csv',
    ROOT / 'large_sweeps/ffmpeg_worker_sweep_cpu64_high_concurrency_20260918/summary.csv',
]
OUT = ROOT / 'analysis/figures/ffmpeg_full_concurrency'


def main():
    rows = []
    for path in INPUTS:
        for row in csv.DictReader(path.open()):
            rows.append({key: int(value) if key in {'workers', 'repetitions'} else float(value)
                         for key, value in row.items()})
    rows.sort(key=lambda row: row['workers'])
    if [row['workers'] for row in rows] != [8, 12, 16, 24, 32, 48, 64, 96]:
        raise RuntimeError('incomplete or duplicate concurrency sweep')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    os.environ.setdefault('MPLCONFIGDIR', '/tmp/ffmpeg-full-concurrency-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    # Keep the motivation figure legible at one-column width. Throughput shows
    # where useful parallelism ends; active latency shows the contention cost.
    fig, axes = plt.subplots(2, 1, figsize=(3.33, 2.30), sharex=True)
    x = list(range(len(rows))); labels = [row['workers'] for row in rows]
    metrics = [
        ('throughput_rps', 'Throughput', 'Videos completed per second'),
        ('mean_decode_s', 'Time/video', 'Time to decode one video'),
    ]
    for ax, (key, ylabel, title) in zip(axes, metrics):
        mean_key = key if key == 'child_cpu_s_per_request' else key + '_mean'
        sd_key = None if key == 'child_cpu_s_per_request' else key + '_sd'
        ax.errorbar(x, [row[mean_key] for row in rows],
                    yerr=None if sd_key is None else [row[sd_key] for row in rows],
                    color='#0877b5', marker='o', linewidth=2, capsize=3)
        ax.axvspan(2.5, 7.5, color='#df6500', alpha=.09)
        ax.axvline(2, color='#555555', linestyle='--', linewidth=1)
        ax.set_title(title, loc='left', fontsize=10.5)
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 2, 7], ['Low', '16', 'High'])
        ax.set_yticks([])
        ax.set_ylim(bottom=0)
    axes[0].text(4.9, .82, 'No more throughput', transform=axes[0].get_xaxis_transform(),
                 fontsize=8, ha='center', color='#9a4700')
    axes[1].text(5.0, .77, 'Each video gets slower', transform=axes[1].get_xaxis_transform(),
                 fontsize=8, ha='center', color='#9a4700')
    axes[-1].set_xlabel('Concurrent decodes')
    fig.subplots_adjust(top=.95, bottom=.21, left=.20, right=.97, hspace=.54)
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / ('ffmpeg_full_concurrency.' + extension), dpi=240,
                    bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
