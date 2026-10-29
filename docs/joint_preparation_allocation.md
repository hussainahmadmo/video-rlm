# Joint preparation allocation prototype

`--prep-placement joint --gpu-prep-backend parallel_nvdec` enables the optional
resource-aware allocator. It supports equal-weight tenants and FCFS inference
admission with `--prep-policy fcfs` or `prep_max_min`. It does not modify vLLM's
GPU execution scheduler or implement a formally proven fairness primitive.

The allocator maintains separate CPU-pool worker-second and GPU reserved
decoder-lane-second accounts. Reservations are added at dispatch and reconciled
against observed preparation occupancy at completion. GPU charge is configured
lane count times full preparation duration, including indexing/remuxing. This
is reservation occupancy, not measured NVDEC hardware busy time, and does not
account for GPU-path host CPU instructions or CUDA compute used for conversion.

Fair tenant selection compares each tenant's maximum capacity-normalized
CPU/GPU accumulated service, including reservations. Among resource-feasible
oldest requests from eligible tenants, the least-served tenant wins. Candidate
execution choices are CPU or any width in the configured GPU-width set. CPU
worker capacity, GPU lane capacity, GPU-job capacity, and legal widths are all
runtime parameters. The evaluated four-lane configuration uses widths one, two,
and four. The lowest predicted
execution time among available choices wins. A CPU option can be deferred if
waiting for predicted GPU releases would make the request ready sooner; other
feasible tenants can still proceed. The inference-occupancy guard applies to all
new GPU admission and does not guarantee inference SLOs.

`--gpu-decoder-budget` caps total active reserved lanes and
`--joint-gpu-widths` supplies the legal per-request widths.
`--gpu-prep-limit` additionally caps simultaneous GPU preparation requests.
GPU-capable requests require the configured frame threshold (default 32).
When multiple such tenants contend, each tenant's concurrent reservations are
capped at the ceiling of the budget divided by the contender count. Eligible
contenders include outstanding GPU-capable pending work and active GPU requests.
With one remaining contender it may borrow the whole budget. Existing work is
nonpreemptive. Ceilings cannot satisfy equal simultaneous shares when tenants
outnumber lanes; service-based admission supplies time sharing. Historical
service counters do not currently reset on idle-tenant reactivation.

`--joint-lane-profile` accepts JSON with full preparation costs such as
`{"128":{"1":11.0,"2":6.2,"4":5.5},"cpu":{"128":30.0}}` as initial estimates.
CPU and every configured lane width then learn independently with online
profilers.

The comparison runner `analysis/run_joint_allocation_comparison.py` uses the same
GPU, CPU pool and GOP decoder for three configurations:

1. FCFS, static CPU-small/GPU-heavy routing, fixed two-lane GPU execution.
2. Resource-aware tenant fairness, the same static routing and lane count.
3. Resource-aware tenant fairness with joint backend/lane allocation.

FCFS is work-conserving across feasible backend choices; a blocked heavy request
does not prevent independent CPU-small requests from starting. The fixed fairness
baseline uses the same CPU/GPU service accounts as joint allocation, isolating
the benefit of execution choices rather than a change of accounting alone.
Per-tenant candidate FIFO and resource constraints can still limit utilization.

The pilot uses 20 requests each from one light and two heavy tenants. It records
per-tenant median/p95 E2E latency and TTFT, overall metrics, burst throughput,
CPU and reserved GPU service totals/shares, and GPU shares during periods when
both heavy tenants have outstanding preparation. Final aggregate resource shares
need not be equal because tenants have different demands and eligibility.
The experiment does not use FlashCodec itself, so cannot claim a measured win
against FlashCodec. Multiple seeds, broader traces, calibrated hardware resource
units, idle-reactivation handling and interference controls remain necessary
before a general fairness/performance claim.
