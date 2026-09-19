"""Same-decoder comparison of FCFS, fixed-width fairness and joint allocation."""
import argparse
import json
import os
import random
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from run_fairness_tail_pilot import ROOT, MODEL, SNAPSHOT, VIDEO, measurements


def resource_metrics(rows, cpu_capacity=2):
    tenants = sorted({r['tenant'] for r in rows})
    result = {t: dict(cpu_worker_s=sum(r['prep_cpu_worker_s'] for r in rows if r['tenant']==t),
                     gpu_reserved_lane_s=sum(r['prep_gpu_reserved_lane_s'] for r in rows if r['tenant']==t))
              for t in tenants}
    for key in ['cpu_worker_s', 'gpu_reserved_lane_s']:
        total = sum(v[key] for v in result.values())
        for v in result.values():
            v[key+'_share'] = v[key]/total if total else 0
    times=sorted({r[key] for r in rows for key in ['arrival_s','prep_started_s','prep_ready_s']})
    contenders = sorted({r['tenant'] for r in rows if r.get('frame_count', r.get('frames', 128)) >= 32})
    common={t:0 for t in tenants}
    common_duration=0
    for start,end in zip(times,times[1:]):
        midpoint=(start+end)/2
        active=[r for r in rows if r['prep_started_s']<=midpoint<r['prep_ready_s']]
        if sum(r['prep_gpu_lanes'] for r in active)>4 or sum(r['prep_backend']=='seek_cpu' for r in active)>cpu_capacity:
            raise RuntimeError('observed reservation capacity exceeded')
        both=all(any(r['tenant']==t and r['arrival_s']<=midpoint<r['prep_ready_s'] for r in rows)
                 for t in contenders) and len(contenders) > 1
        if both:
            common_duration+=end-start
            for r in active:
                common[r['tenant']]+=(end-start)*r['prep_gpu_lanes']
    total=sum(common.values())
    return dict(tenants=result, joint_heavy_backlog_window_s=common_duration,
                gpu_service_during_joint_heavy_backlog=common,
                gpu_shares_during_joint_heavy_backlog={t:v/total if total else 0 for t,v in common.items()})


def report(output, results):
    (output/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    manifest=json.loads((output/'manifest.json').read_text())
    tenant_names=manifest.get('tenants', ['light','heavy_a','heavy_b'])
    lines=['# Joint acceleration allocation comparison','',
        manifest.get('description', 'Staged 60-request pilot with two CPU workers.'), '',
        'Fixed-width fairness uses the same separate CPU/GPU reservation accounts as joint allocation. Tenant priority is the maximum of accounted CPU service / CPU capacity and GPU lane service / lane capacity. It is an experimental dominant-service heuristic, not a proven DRF allocation. Service counters include in-flight reservations and reconcile at completion. GPU costs are reserved lane-seconds across full preparation, not hardware engine busy time; GPU-path host instructions are not CPU-pool worker service. Aggregate shares are demand-dependent and need not be equal. A separate window reports GPU shares while all tenants with heavy requests have outstanding preparation.', '',
        '| Policy | Overall mean (s) | Overall median (s) | Overall p95 (s) | Throughput (req/s) |',
        '|---|---:|---:|---:|---:|']
    for r in results:
        a=r['all']['end_to_end_s'];lines.append(f"| {r['variant']} | {a['mean']:.2f} | {a['median']:.2f} | {a['p95']:.2f} | {r['throughput_qps']:.3f} |")
    lines+=['','| Policy / tenant | Median (s) | p95 (s) | CPU share | Reserved GPU lane share |','|---|---:|---:|---:|---:|']
    for r in results:
        for t,a in r['tenants'].items():
            v=r['resources']['tenants'][t];e=a['end_to_end_s']
            lines.append(f"| {r['variant']} / {t} | {e['median']:.2f} | {e['p95']:.2f} | {v['cpu_worker_s_share']:.1%} | {v['gpu_reserved_lane_s_share']:.1%} |")
    lines+=['','GPU shares during overlapping preparation backlog:','']
    for r in results:
        v=r['resources']; shares=v['gpu_shares_during_joint_heavy_backlog']
        lines.append(f"- {r['variant']}: " + ', '.join(f'{t} {shares[t]:.1%}' for t in tenant_names) + f"; window {v['joint_heavy_backlog_window_s']:.2f}s")
    if results and 'sizes' in results[0]:
        lines+=['','| Policy / size | Median (s) | p95 (s) |','|---|---:|---:|']
        for r in results:
            for size, metrics in r['sizes'].items():
                e=metrics['end_to_end_s']
                lines.append(f"| {r['variant']} / {size} | {e['median']:.2f} | {e['p95']:.2f} |")
    lines+=['','Only completed valid runs are included. Inspect `status.json`, `COMPLETE` or `FAILED`. This is a one-trial, one-video pilot in fixed policy order, with 20 samples per tenant; no statistical significance or p99 claim. A short full-pipeline calibration measures every evaluated frame count on CPU and every GPU-eligible frame count at each GPU width before the comparison; profiles are stored in `lane_profile.json`. New online profiles start per policy. The factorial policies isolate tenant selection, placement, and lane-width decisions while using the same decoder implementation. No FlashCodec implementation is used; the experiment does not claim to outperform FlashCodec.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    if not results:return
    os.environ.setdefault('MPLCONFIGDIR','/tmp/joint-allocation-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['median','p95']):
        for tenant in tenant_names:
            ax.plot(range(len(results)),[r['tenants'][tenant]['end_to_end_s'][key] for r in results],marker='o',label=tenant)
        ax.set_xticks(range(len(results)),[r['variant'].replace('_','\n',1) for r in results],fontsize=9)
        ax.set_ylabel(f'End-to-end {key} (s)');ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.suptitle(f'Same decoder and hardware: allocation pilot ({len(results)}/{len(manifest["runs"])} complete)')
    fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(output/f'latency_comparison.{extension}',dpi=180)
    plt.close(fig)
    labels=[r['variant'].replace('_','\n',1) for r in results]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for tenant,color in zip(tenant_names,['#69b96c','#3182bd','#e99b35']):
        for ax,key in zip(axes[:2],['cpu_worker_s_share','gpu_reserved_lane_s_share']):
            bottoms=[sum(r['resources']['tenants'][t][key] for t in tenant_names[:tenant_names.index(tenant)])*100 for r in results]
            ax.bar(range(len(results)),[r['resources']['tenants'][tenant][key]*100 for r in results],bottom=bottoms,label=tenant,color=color)
    for ax,title in zip(axes[:2],['CPU worker service','Reserved GPU lane service']):
        ax.set_title(title);ax.set_ylabel('Tenant share (%)');ax.set_ylim(0,100)
    axes[0].legend(frameon=False,fontsize=9)
    axes[2].bar(range(len(results)),[r['throughput_qps'] for r in results],color='#3182bd')
    axes[2].set_title('Throughput');axes[2].set_ylabel('Completed requests / second')
    for ax in axes:ax.set_xticks(range(len(results)),labels,fontsize=9)
    fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(output/f'service_and_throughput.{extension}',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--video', type=Path, default=VIDEO)
    parser.add_argument('--cpu-workers', type=int, default=2)
    parser.add_argument('--arrival-trace', type=Path)
    parser.add_argument('--variable-frame-workload', action='store_true',
                        help='Generate a 60-request, three-tenant 1/16/128-frame trace')
    parser.add_argument('--routing-ablation', action='store_true')
    parser.add_argument('--mechanism-ablation', action='store_true',
                        help='Isolate fairness, CPU/GPU placement, and GPU lane width')
    parser.add_argument('--minimum-lane-ablation', action='store_true',
                        help='Compare placement FCFS and fair allocation with minimum GPU widths one and two')
    parser.add_argument('--adaptive-all-frame-counts', action='store_true',
                        help='Allow adaptive policies to consider GPU preparation below the static routing threshold')
    parser.add_argument('--conservative-placement', action='store_true',
                        help='Require adaptive routes to beat the calibrated static route by a confidence margin')
    parser.add_argument('--integrated-ablation', action='store_true',
                        help='Cross optimization/capacity adaptation with cross-stage fairness')
    parser.add_argument('--bypass-ablation', action='store_true',
                        help='Compare optimized FCFS, tenant FIFO, and profiled within-tenant bypass')
    parser.add_argument('--gpu-backend', choices=['parallel_nvdec','flashstyle_nvdec'], default='parallel_nvdec')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--model-snapshot', default=SNAPSHOT,
                        help='Local model snapshot passed to vLLM')
    parser.add_argument('--profiled-light-ablation', action='store_true')
    parser.add_argument('--max-cpu-workers', type=int, default=8)
    parser.add_argument('--max-handoff', type=int, default=16)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists():parser.error('fresh output directory required')
    if not args.video.is_file():parser.error(f'video does not exist: {args.video}')
    output.mkdir(parents=True)
    trace=output/'trace.jsonl';rows=[]
    for i in range(60):
        tenant='heavy_a' if i<20 else 'heavy_b' if i<40 else 'light'
        rows.append(dict(request_id=f'joint-{i}',qid=f'joint-{i}',tenant=tenant,modality='video',video=str(args.video),
                         frame_count=1 if tenant=='light' else 128,arrival_s=i*.1,max_tokens=32,priority=0,
                         prompt_override='Briefly describe the video.',**{'class':'background'}))
    if args.cpu_workers < 1 or args.max_cpu_workers < args.cpu_workers:
        parser.error('positive CPU workers and max-cpu-workers >= cpu-workers required')
    if args.max_handoff < 8 or args.max_handoff < args.max_cpu_workers:
        parser.error('max-handoff must cover eight fixed slots and max-cpu-workers')
    if sum(map(bool, [args.routing_ablation, args.mechanism_ablation,
                      args.minimum_lane_ablation,
                      args.profiled_light_ablation, args.integrated_ablation,
                      args.bypass_ablation])) > 1:
        parser.error('choose only one ablation')
    if args.arrival_trace and args.variable_frame_workload:
        parser.error('choose --arrival-trace or --variable-frame-workload')
    if args.variable_frame_workload:
        rng = random.Random(11)
        rows = []
        for tenant in ['a', 'b', 'c']:
            sizes = [1] * 5 + [16] * 10 + [128] * 5
            rng.shuffle(sizes)
            for i, frame_count in enumerate(sizes):
                rows.append(dict(request_id=f'{tenant}-{i}', qid=f'{tenant}-{i}',
                                 tenant=tenant, modality='video', video=str(args.video),
                                 frame_count=frame_count, arrival_s=rng.uniform(0, 15),
                                 max_tokens=32, priority=0,
                                 prompt_override='Briefly describe the video.',
                                 **{'class':'background'}))
        rows.sort(key=lambda r: (r['arrival_s'], r['request_id']))
    if args.arrival_trace:
        rows=[json.loads(line) for line in args.arrival_trace.read_text().splitlines() if line.strip()]
        if len(rows) != 60: parser.error('this pilot requires 60 requests')
        for row in rows:
            row['video'] = str(args.video)
    tenant_names=sorted({r['tenant'] for r in rows})
    frame_counts=sorted({int(r['frame_count']) for r in rows})
    trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def command(name,trace_path,flags,prep_workers=None,handoff=None):
        prep_workers = args.cpu_workers if prep_workers is None else prep_workers
        handoff = 8 if handoff is None else handoff
        return [sys.executable,str(ROOT/'conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py'),
                '--arrival-trace',str(trace_path),'--output',str(output/name),'--port','9001','--model',MODEL,
                '--prep-workers',str(prep_workers),'--vlm-concurrency','4','--prepared-queue-depth',str(handoff),'--urgent-prep-reserve','0',
                '--decode-backend','seek_cpu','--cpu-decoder-threads','1','--prep-placement','joint','--gpu-prep-backend',args.gpu_backend,
                '--gpu-prep-limit','2','--gpu-decoder-budget','4','--prep-cost-profiler','frame_ewma',
                '--joint-lane-profile',str(output/'lane_profile.json')]+flags
    runs=[dict(variant=name,command=command(name,trace,flags)) for name,flags in [
        ('parallel_fcfs',['--prep-policy','fcfs','--joint-allocation-order','fcfs','--joint-fixed-routing','--joint-fixed-lanes','2']),
        ('parallel_fixed_fair',['--prep-policy','prep_max_min','--joint-fixed-routing','--joint-fixed-lanes','2']),
        ('adaptive_fcfs',['--prep-policy','fcfs','--joint-allocation-order','fcfs']),
        ('joint_allocator',['--prep-policy','prep_max_min'])]]
    if args.mechanism_ablation:
        adaptive_eligibility = (['--gpu-prep-frame-threshold','1']
                                if args.adaptive_all_frame_counts else [])
        conservative = (['--joint-conservative-routing',
                         '--joint-preferred-gpu-frame-threshold','32',
                         '--joint-switch-margin-s','2',
                         '--joint-switch-margin-ratio','.2']
                        if args.conservative_placement else [])
        runs=[dict(variant=name,command=command(name,trace,flags)) for name,flags in [
            ('fixed_fcfs',['--prep-policy','fcfs','--joint-allocation-order','fcfs',
                           '--joint-fixed-routing','--joint-fixed-lanes','2']),
            ('fixed_fair',['--prep-policy','prep_max_min',
                           '--joint-fixed-routing','--joint-fixed-lanes','2']),
            ('placement_only_fcfs',['--prep-policy','fcfs','--joint-allocation-order','fcfs',
                                    '--joint-fixed-lanes','2'] + adaptive_eligibility + conservative),
            ('width_only_fcfs',['--prep-policy','fcfs','--joint-allocation-order','fcfs',
                                '--joint-fixed-routing']),
            ('adaptive_fcfs',['--prep-policy','fcfs','--joint-allocation-order','fcfs'] + adaptive_eligibility + conservative),
            ('full_conductor',['--prep-policy','prep_max_min'] + adaptive_eligibility + conservative)]]
    if args.minimum_lane_ablation:
        adaptive = [
            '--gpu-prep-frame-threshold','1',
            '--joint-conservative-routing',
            '--joint-preferred-gpu-frame-threshold','32',
            '--joint-switch-margin-s','2',
            '--joint-switch-margin-ratio','.2',
        ]
        runs=[dict(variant=name,command=command(name,trace,flags)) for name,flags in [
            ('placement_only_fcfs',['--prep-policy','fcfs',
                                    '--joint-allocation-order','fcfs',
                                    '--joint-fixed-lanes','2'] + adaptive),
            ('full_conductor_min1',['--prep-policy','prep_max_min'] + adaptive),
            ('full_conductor_min2',['--prep-policy','prep_max_min',
                                    '--joint-min-gpu-lanes','2'] + adaptive),
        ]]
    if args.integrated_ablation:
        online = [
            '--capacity-adaptation','online',
            '--capacity-initial-prep-workers',str(args.cpu_workers),
            '--capacity-initial-handoff','8',
            '--capacity-control-interval-s','10',
        ]
        runs = [
            dict(variant='fixed_fcfs', command=command(
                'fixed_fcfs', trace,
                ['--prep-policy','fcfs','--joint-allocation-order','fcfs',
                 '--joint-fixed-routing','--joint-fixed-lanes','2'])),
            dict(variant='fixed_conductor', command=command(
                'fixed_conductor', trace,
                ['--prep-policy','max_min','--joint-allocation-order','fair',
                 '--joint-fixed-routing','--joint-fixed-lanes','2'])),
            dict(variant='optimized_fcfs', command=command(
                'optimized_fcfs', trace,
                ['--prep-policy','fcfs','--joint-allocation-order','fcfs'] + online,
                prep_workers=args.max_cpu_workers, handoff=args.max_handoff)),
            dict(variant='full_conductor', command=command(
                'full_conductor', trace,
                ['--prep-policy','max_min','--joint-allocation-order','fair'] + online,
                prep_workers=args.max_cpu_workers, handoff=args.max_handoff)),
        ]
    if args.bypass_ablation:
        # Keep capacity identical and fixed in this ablation.  Otherwise the
        # online controller can react differently to each request order and
        # confound the effect of within-tenant bypass.
        fair = ['--prep-policy','max_min','--joint-allocation-order','fair']
        runs = [
            dict(variant='optimized_fcfs', command=command(
                'optimized_fcfs', trace,
                ['--prep-policy','fcfs','--joint-allocation-order','fcfs'],
                prep_workers=args.max_cpu_workers, handoff=args.max_handoff)),
            dict(variant='full_conductor_fifo', command=command(
                'full_conductor_fifo', trace, fair,
                prep_workers=args.max_cpu_workers, handoff=args.max_handoff)),
            dict(variant='full_conductor_bypass', command=command(
                'full_conductor_bypass', trace,
                fair + ['--joint-profile-light-s','1',
                        '--joint-light-bypass-age-s','5'],
                prep_workers=args.max_cpu_workers, handoff=args.max_handoff)),
        ]
    if args.routing_ablation:
        if args.mechanism_ablation:parser.error('choose only one ablation')
        runs=[dict(variant=name,command=command(name,trace,['--prep-policy','prep_max_min']+flags)) for name,flags in [
            ('original_joint', []),
            ('bypass_only', ['--joint-bypass-heavy']),
            ('routing_only', ['--joint-fixed-routing']),
            ('corrected_joint', ['--joint-fixed-routing','--joint-bypass-heavy'])]]
    if args.profiled_light_ablation:
        if args.routing_ablation or args.mechanism_ablation:parser.error('choose only one ablation')
        runs=[dict(variant=name,command=command(name,trace,['--prep-policy','prep_max_min']+flags)) for name,flags in [
            ('original_joint',[]),
            ('profiled_light_joint',['--joint-profile-light-s','1','--joint-cpu-light-reserve','1','--joint-light-bypass-age-s','5'])]]
    gpu_query = subprocess.run(
        ['nvidia-smi', f'--id={args.gpu}', '--query-gpu=name', '--format=csv,noheader'],
        capture_output=True, text=True,
    )
    gpu_name = gpu_query.stdout.strip() if gpu_query.returncode == 0 else 'GPU model unavailable'
    frame_mix=', '.join(f'{n}x{sum(int(r["frame_count"]) == n for r in rows)}' for n in frame_counts)
    interleaved = args.arrival_trace or args.variable_frame_workload
    (output/'manifest.json').write_text(json.dumps(dict(runs=runs,requests_per_policy=60,tenants=tenant_names,frame_counts=frame_counts,cpu_workers=args.cpu_workers,description=f'{gpu_name} physical GPU {args.gpu}, Qwen2.5-VL-7B; 60 requests per policy with frame-count mix {frame_mix}. ' + ('Interleaved requests; every tenant receives the recorded mix. ' if interleaved else 'Staged tenant arrivals. ') + f'{args.cpu_workers} CPU workers, two GPU preparation job slots, four GPU decoder lanes, four inference slots, FCFS inference admission and handoff capacity eight. Identical video, sampling, prompt, 32-token budget, trace and GOP-parallel NVDEC backend across all policies. Fixed placement routes requests below 32 frames to CPU and requests at or above 32 frames to GPU with two lanes. Adaptive policies choose CPU/GPU and one/two/four lanes. Fair policies use tenant service accounts; adaptive FCFS has no tenant lane-share cap. The same inference occupancy guard applies to all policies.'),indent=2)+'\n')
    if args.routing_ablation:
        manifest=json.loads((output/'manifest.json').read_text())
        manifest['description']=f'Four-policy routing/bypass ablation on the identical 60-request mixed trace; {args.cpu_workers} CPU workers, two GPU preparation job slots, four GPU decoder lanes, four inference slots, FCFS inference admission and handoff eight. All variants retain tenant service fairness and adaptive one/two/four GPU lanes. Original joint permits heavy CPU preparation and one candidate per tenant. Bypass-only considers the oldest light and heavy request per tenant. Routing-only sends few-frame requests to CPU and GPU-eligible requests (32+ frames) exclusively to GPU. Corrected joint combines strict routing and bypass. Same video, sampling, prompt and output budget across variants.'
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    manifest=json.loads((output/'manifest.json').read_text())
    manifest['gpu_backend']=args.gpu_backend
    manifest['physical_gpu']=args.gpu
    manifest['description']=manifest['description'].replace('A40 GPU 1',f'A40 GPU {args.gpu}')
    if args.gpu_backend=='flashstyle_nvdec':
        manifest['description']=manifest['description'].replace('GOP-parallel NVDEC backend','FlashCodec-inspired shared-chunk NVDEC backend')
        manifest['description']+=' This uses our independently implemented chunk scheduler, not the authors\' FlashCodec code; inference remains monolithic vLLM, not the E/P/D prototype.'
    if args.profiled_light_ablation:
        manifest['description']=manifest['description'].replace('across four policies','across two policies')
        manifest['description']+=' Two-policy ablation: original joint versus profiled-light joint. Both retain tenant service accounts and adaptive placement/width. The latter uses a 1-second predicted CPU-cost threshold, protects one CPU slot from expensive work, and permits within-tenant cheap bypass with 5-second aging. Both receive the same calibration for 1/16/128 frames. This bundles three changes; it does not isolate them separately.'
    if args.mechanism_ablation:
        manifest['description']=(
            f'Six-policy mechanism ablation on the identical 60-request trace with '
            f'{args.cpu_workers} CPU workers, two GPU preparation jobs, four decoder '
            'lanes, four inference slots, and handoff capacity eight. Fixed FCFS is '
            'the reference. Fixed fair changes only tenant selection; placement-only '
            'changes CPU/GPU placement with width fixed at two lanes; width-only keeps '
            'fixed CPU-small/GPU-heavy routing but chooses one/two/four lanes; adaptive '
            'FCFS changes placement and width without fairness; full Conductor combines '
            'adaptive placement/width with tenant service accounting.' +
            (' Adaptive policies consider both backends for every frame count; fixed '
             'policies use the 32-frame routing threshold.' if args.adaptive_all_frame_counts else '') +
            (' Adaptive policies retain that calibrated route unless another option '
             'improves predicted readiness by at least two seconds and 20%.'
             if args.conservative_placement else '')
        )
    if args.minimum_lane_ablation:
        manifest['description'] = (
            f'Three-policy minimum-lane ablation on the identical 60-request trace '
            f'with {args.cpu_workers} CPU workers, two GPU preparation jobs, four '
            'decoder lanes, four inference slots, and handoff capacity eight. All '
            'policies use conservative load-aware CPU/GPU placement. Placement-only '
            'FCFS fixes GPU width at two lanes. Full Conductor min1 uses fair tenant '
            'selection with dynamic widths one/two/four; min2 uses the same policy '
            'but restricts dynamic widths to two/four. The comparison isolates the '
            'effect of allowing one-lane execution in the fair allocator.'
        )
    if args.integrated_ablation:
        manifest['description'] = (
            f'Integrated four-policy comparison on {gpu_name}. Fixed policies use '
            f'{args.cpu_workers} CPU workers, handoff eight, fixed CPU-small/GPU-heavy '
            'routing, and two GPU decoder lanes. Optimized policies start from the '
            f'same CPU/handoff configuration and may adapt up to {args.max_cpu_workers} '
            f'CPU workers and handoff {args.max_handoff}; they also select CPU/GPU '
            'placement and one/two/four decoder lanes. FCFS variants use arrival order. '
            'Conductor variants use separate preparation and inference tenant-service '
            'accounts; preparation placement additionally uses CPU worker-second and '
            'reserved GPU lane-second accounts. GPU job and lane limits remain two and '
            'four for every policy.'
        )
    if args.bypass_ablation:
        manifest['description'] = (
            f'Profile-guided within-tenant bypass ablation on {gpu_name}. All '
            f'policies use a fixed capacity of {args.max_cpu_workers} CPU workers '
            f'and handoff {args.max_handoff}, and use identical profiled CPU/GPU '
            'placement and one/two/four-lane selection. Optimized FCFS uses '
            'arrival order. Conductor FIFO first selects the tenant with least '
            'accounted service and then its oldest request. Conductor bypass '
            'retains tenant-first selection but permits requests with predicted '
            'CPU preparation cost at most one second to bypass more expensive '
            'requests within that tenant; after five seconds, the oldest aged '
            'request regains precedence. No CPU slot reservation is enabled, so '
            'this comparison isolates request bypass and its aging rule.'
        )
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    snapshot=output/'code_snapshot';snapshot.mkdir()
    for f in [Path(__file__),ROOT/'conductor/experiments/scripts/run/joint_preparation_allocation.py',ROOT/'conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py',ROOT/'conductor/experiments/scripts/run/parallel_nvdec_decode.py',ROOT/'conductor/experiments/scripts/run/batched_nvdec_decode.py']:
        shutil.copy2(f,snapshot/f.name)
    if args.gpu_backend=='flashstyle_nvdec':
        shutil.copy2(ROOT/'conductor/experiments/scripts/run/flashstyle_nvdec_decode.py',snapshot/'flashstyle_nvdec_decode.py')
    results=[];report(output,results)
    if args.dry_run:
        gpu_calibration_frames = frame_counts if args.adaptive_all_frame_counts else [n for n in frame_counts if n >= 32]
        calibration_count = len(frame_counts) + 3 * len(gpu_calibration_frames)
        print(f'Prepared {len(runs)} policies, {60*len(runs)} measured requests and {calibration_count} calibration requests');return
    try:urllib.request.urlopen('http://127.0.0.1:9001/health',timeout=2)
    except Exception:pass
    else:raise RuntimeError('port 9001 already hosts a server')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),VIDEO_RLM_NVDEC_GPU_ID='0',VIDEO_RLM_NVDEC_MAX_SESSIONS='4',
             VIDEO_RLM_FLASH_GPU_IDS='0',VIDEO_RLM_FLASH_WORKERS_PER_GPU='4')
    server=telemetry=None
    try:
        (output/'status.json').write_text(json.dumps(dict(state='starting',active='model_server',completed=0,total=len(runs)))+'\n')
        with (output/'server.log').open('w') as log:
            server=subprocess.Popen([str(Path(sys.executable).with_name('vllm')),'serve',args.model_snapshot,'--served-model-name',MODEL,
                '--host','127.0.0.1','--port','9001','--api-key','EMPTY','--dtype','auto','--tensor-parallel-size','1',
                '--max-model-len','16384','--gpu-memory-utilization','.80','--enforce-eager','--max-num-seqs','4',
                '--max-num-batched-tokens','4096','--limit-mm-per-prompt','{"image":128,"video":0}',
                '--no-enable-prefix-caching','--mm-processor-cache-gb','0','--scheduling-policy','priority'],
                stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        for attempt in range(150):
            if server.poll() is not None:raise RuntimeError('server failed')
            try:urllib.request.urlopen('http://127.0.0.1:9001/health',timeout=2);break
            except Exception:time.sleep(2)
        else:raise RuntimeError('server startup timeout')
        with (output/'gpu.csv').open('w') as log:
            telemetry=subprocess.Popen(['nvidia-smi',f'--id={args.gpu}','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv','--loop=1'],stdout=log)
        calibration=output/'calibration_trace.jsonl'
        calibration.write_text(json.dumps(dict(rows[0],arrival_s=0))+'\n')
        profile={str(n):{} for n in frame_counts}
        (output/'lane_profile.json').write_text(json.dumps(profile))
        calibration_frames=frame_counts
        gpu_calibration_frames = (calibration_frames if args.adaptive_all_frame_counts else
                                  [n for n in calibration_frames if n >= 32])
        calibration_options = ([(n, 0) for n in calibration_frames] +
                               [(n, w) for n in gpu_calibration_frames for w in [1, 2, 4]])
        for frame_count,width in calibration_options:
            calibration.write_text(json.dumps(dict(rows[0],arrival_s=0,frame_count=frame_count))+'\n')
            profile.setdefault(str(frame_count),{})
            name=f'calibration_{frame_count}_{width}'
            (output/'status.json').write_text(json.dumps(dict(state='calibrating',active=name,completed=0,total=len(runs)))+'\n')
            print('Starting '+name,flush=True)
            cmd=command(name,calibration,['--prep-policy','fcfs','--joint-allocation-order','fcfs','--joint-fixed-routing','--joint-fixed-lanes',str(width)])
            if width > 0:
                cmd += ['--gpu-prep-frame-threshold','1']
            if width==0:
                cmd += ['--prep-placement','fixed']
            with (output/f'{name}.log').open('w') as log:subprocess.run(cmd,check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
            raw=[json.loads(l) for l in (output/name/'results.jsonl').read_text().splitlines() if l.strip()]
            if len(raw)!=1 or raw[0].get('error'):raise RuntimeError('calibration failed')
            if width==0:
                profile.setdefault('cpu',{})[str(frame_count)]=raw[0]['prep_service_s']
            else:
                profile[str(frame_count)][str(width)]=raw[0]['prep_service_s']
            (output/'lane_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
        for run in runs:
            name=run['variant'];print('Starting '+name,flush=True)
            (output/'status.json').write_text(json.dumps(dict(active=name,completed=len(results),total=len(runs)))+'\n')
            started=time.time()
            with (output/f'{name}.log').open('w') as log:subprocess.run(run['command'],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
            summary=json.loads((output/name/'summary.json').read_text())
            raw=[json.loads(l) for l in (output/name/'results.jsonl').read_text().splitlines() if l.strip()]
            if summary['errors'] or len(raw)!=60 or {r['request_id'] for r in raw}!={r['request_id'] for r in rows}:raise RuntimeError('failed or missing requests')
            resources=resource_metrics(raw,summary['prep_workers'])
            results.append(dict(variant=name,started_epoch_s=started,finished_epoch_s=time.time(),
                                all=measurements(raw),tenants={t:measurements([r for r in raw if r['tenant']==t]) for t in tenant_names},
                                sizes={f'{n}_frames':measurements([r for r in raw if r['frame_count']==n]) for n in frame_counts},
                                throughput_qps=summary['throughput_qps'],resources=resources,
                                capacity_adaptation=summary.get('capacity_adaptation'),
                                capacity_initial_prep_workers=summary.get('capacity_initial_prep_workers'),
                                capacity_final_prep_workers=summary.get('capacity_final_prep_workers'),
                                capacity_initial_handoff=summary.get('capacity_initial_handoff'),
                                capacity_final_handoff=summary.get('capacity_final_handoff'),
                                capacity_history=summary.get('capacity_history', []),
                                prep_backend_counts=summary.get('prep_backend_counts', {}),
                                lane_counts={str(n):sum(r['prep_gpu_lanes']==n for r in raw) for n in [0,1,2,4]}))
            report(output,results)
        (output/'status.json').write_text(json.dumps(dict(active=None,completed=len(results),total=len(runs)))+'\n')
        (output/'COMPLETE').write_text(f'All {len(runs)} comparison runs completed without errors.\n')
    except BaseException as error:
        (output/'status.json').write_text(json.dumps(dict(state='failed',error=repr(error),completed=len(results),total=len(runs)))+'\n')
        (output/'FAILED').write_text(repr(error)+'\n');raise
    finally:
        if telemetry is not None:telemetry.terminate();telemetry.wait()
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL)


if __name__=='__main__':main()
