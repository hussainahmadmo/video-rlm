"""Figure 1: validated waiting/execution times, not inferred CPU utilization."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/paper-preparation-gap-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_cross_gpu_poisson_time import load_runs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '69f58e9744973ad74e35062f/figures/motivation'
SOURCES = [
    ('A40', 2020, 'a40_client_limit_comparison_timeout600_20260917/bounded'),
    ('H100', 2022, 'h100_sxm_cpu8_collection_20260917/results/h100_sxm_cpu8_client_comparison_timeout1800_20260917/bounded'),
    ('B300', 2025, 'b300_cpu8_collection_20260917/results/b300_cpu8_client_comparison_timeout1800_20260917/bounded'),
]
runs = load_runs([f'{gpu}={year}={ROOT}/large_sweeps/{path}' for gpu, year, path in SOURCES])
parts = [('prep_queue_s', 'Waiting before preparation', '#6baed6'),
         ('later_work_s', 'All later work', '#d9d9d9')]
plt.rcParams.update({'font.size': 10.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(3, 1, figsize=(3.33, 3.05), sharex=True)
rows = []
for ax, run in zip(axes, runs):
    bottom = [0.0] * 3
    for key, label, color in parts:
        values = [
            (g['metrics']['prep_queue_s'] if key == 'prep_queue_s' else
             g['metrics']['prep_execution_s'] +
             g['metrics']['admission_queue_s'] +
             g['metrics']['inference_service_s'])
            for g in run['groups']
        ]
        ax.barh(range(3), values, left=bottom, color=color, label=label, height=.65)
        bottom = [a+b for a,b in zip(bottom, values)]
    ax.set_yticks(range(3), ['Low', 'Medium', 'High'])
    ax.invert_yaxis()
    ax.set_title(run['label'], loc='left', fontsize=10.5, pad=4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([])
    for g in run['groups']:
        rows.append(dict(system=run['label'], rate=g['rate'], **g['metrics']))
axes[1].set_ylabel('Offered load')
axes[-1].set_xlabel('End-to-end delay')
axes[-1].set_xlim(0, 300)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
           bbox_to_anchor=(.5, 1), ncol=1, frameon=False, columnspacing=.8,
           handlelength=1.0, handletextpad=.4)
fig.subplots_adjust(left=.27, right=.96, bottom=.15, top=.74, hspace=.50)
OUT.mkdir(parents=True, exist_ok=True)
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'inference_preparation_gap.{ext}', dpi=220)
(OUT / 'inference_preparation_gap_metrics.json').write_text(json.dumps(rows, indent=2)+'\n')
