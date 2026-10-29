"""Resource-aware preparation admission; decoder lanes are reservations.

GPU charges are reserved lane-seconds, not hardware engine busy time. CPU
charges refer to the CPU preparation pool, not GPU-path host instructions.
"""
import math


class JointPreparationAllocator:
    def __init__(self, cpu_capacity, gpu_capacity, gpu_jobs, frame_threshold=32,
                 gpu_widths=None):
        if min(cpu_capacity, gpu_capacity, gpu_jobs, frame_threshold) < 1:
            raise ValueError('positive capacities required')
        if gpu_widths is None:
            gpu_widths = [width for width in (1, 2, 4)
                          if width <= gpu_capacity]
        try:
            supplied_widths = tuple(gpu_widths)
        except TypeError:
            raise ValueError('GPU widths must be positive integers') from None
        if any(isinstance(width, bool) or not isinstance(width, int)
               for width in supplied_widths):
            raise ValueError('GPU widths must be positive integers')
        normalized_widths = tuple(sorted(set(supplied_widths)))
        if (not normalized_widths or
                any(width < 1 or width > gpu_capacity for width in normalized_widths)):
            raise ValueError('GPU widths must be within configured lane capacity')
        self.cpu_capacity = cpu_capacity
        self.gpu_capacity = gpu_capacity
        self.gpu_jobs = gpu_jobs
        self.gpu_widths = normalized_widths
        self.frame_threshold = frame_threshold
        self.cpu_service = {}
        self.gpu_service = {}
        self.active_tenants = set()
        self.service_frontier = 0.0
        # Number of younger requests dispatched ahead of a tenant's current
        # head request. Entries disappear when that head is selected or leaves
        # the pending queue. This state bounds within-tenant reordering without
        # changing inter-tenant service accounting.
        self.within_tenant_bypasses = {}

    def score(self, tenant):
        return max(self.cpu_service.get(tenant, 0) / self.cpu_capacity,
                   self.gpu_service.get(tenant, 0) / self.gpu_capacity)

    def gpu_eligible(self, job):
        return job['modality'] == 'video' and job['frame_count'] >= self.frame_threshold

    def update_active_tenants(self, jobs, active, initialize_frontier=False):
        """Track preparation demand and optionally initialize entering tenants.

        The frontier is virtual normalized service used only for scheduling.  It
        does not change the observed per-request resource accounting emitted by
        the runner.
        """
        current = {str(job['tenant']) for job in list(jobs) + list(active)}
        if initialize_frontier:
            continuing = current & self.active_tenants
            if continuing:
                self.service_frontier = max(
                    self.service_frontier,
                    min(self.score(tenant) for tenant in continuing),
                )
            for tenant in current - self.active_tenants:
                self.cpu_service[tenant] = self.service_frontier * self.cpu_capacity
                self.gpu_service[tenant] = self.service_frontier * self.gpu_capacity
            if current:
                self.service_frontier = max(
                    self.service_frontier,
                    min(self.score(tenant) for tenant in current),
                )
        self.active_tenants = current

    def choose(self, jobs, active, cpu_backend, gpu_backend, cpu_estimate,
               gpu_estimate, allow_gpu=True, order='fair', fixed_lanes=0,
               fixed_routing=False, now_s=0, bypass_heavy=False,
               profile_light_s=0, cpu_light_reserve=0, light_bypass_age_s=5,
               cpu_limit=None, conservative_routing=False,
               preferred_gpu_frame_threshold=32, switch_margin_s=0,
               switch_margin_ratio=0, min_gpu_lanes=1,
               fairness_slack_s=0, initialize_active_frontier=False,
               fair_work_conserving_borrow=False, gpu_job_limit=None,
               gpu_lane_limit=None, selection_policy=None,
               tenant_dispatches=None, sjf_aging_s=30,
               within_tenant_sjf=False, within_tenant_bypass_age_s=30,
               within_tenant_bypass_limit=4,
               placement_policy='predicted_readiness', gpu_uncertainty=None):
        cpu_limit = self.cpu_capacity if cpu_limit is None else int(cpu_limit)
        gpu_job_limit = self.gpu_jobs if gpu_job_limit is None else int(gpu_job_limit)
        gpu_lane_limit = self.gpu_capacity if gpu_lane_limit is None else int(gpu_lane_limit)
        if selection_policy is None:
            selection_policy = order
        allowed_selection_policies = {
            'fair', 'fcfs', 'tenant_round_robin', 'prep_sjf',
            'prep_sjf_aging',
        }
        if (order not in ('fair', 'fcfs') or
                selection_policy not in allowed_selection_policies or
                placement_policy not in ('predicted_readiness', 'presto') or
                fixed_lanes < 0 or
                fixed_lanes > self.gpu_capacity or min_gpu_lanes < 1 or
                min_gpu_lanes > self.gpu_capacity or
                (fixed_lanes and fixed_lanes < min_gpu_lanes) or
                (fixed_lanes and fixed_lanes not in self.gpu_widths)):
            raise ValueError('invalid allocation settings')
        if not math.isfinite(sjf_aging_s) or sjf_aging_s <= 0:
            raise ValueError('SJF aging threshold must be positive')
        if (not math.isfinite(within_tenant_bypass_age_s)
                or within_tenant_bypass_age_s <= 0
                or isinstance(within_tenant_bypass_limit, bool)
                or not isinstance(within_tenant_bypass_limit, int)
                or within_tenant_bypass_limit < 1
                or (within_tenant_sjf and
                    (selection_policy != 'fair' or order != 'fair'))
                or (within_tenant_sjf and profile_light_s)):
            raise ValueError('invalid bounded within-tenant SJF settings')
        tenant_dispatches = tenant_dispatches or {}
        gpu_uncertainty = gpu_uncertainty or (lambda job, lanes: 0.0)
        if not 1 <= cpu_limit <= self.cpu_capacity:
            raise ValueError('CPU limit must be within configured capacity')
        if not 1 <= gpu_job_limit <= self.gpu_jobs:
            raise ValueError('GPU job limit must be within configured capacity')
        if not 1 <= gpu_lane_limit <= self.gpu_capacity:
            raise ValueError('GPU lane limit must be within configured capacity')
        if (not math.isfinite(profile_light_s) or profile_light_s < 0
                or not 0 <= cpu_light_reserve <= self.cpu_capacity
                or not math.isfinite(light_bypass_age_s) or light_bypass_age_s <= 0
                or (cpu_light_reserve and not profile_light_s)):
            raise ValueError('invalid profiled-light settings')
        if (preferred_gpu_frame_threshold < 1
                or not math.isfinite(switch_margin_s) or switch_margin_s < 0
                or not math.isfinite(switch_margin_ratio)
                or not 0 <= switch_margin_ratio < 1
                or not math.isfinite(fairness_slack_s)
                or fairness_slack_s < 0):
            raise ValueError('invalid conservative-routing settings')
        self.update_active_tenants(
            jobs, active, initialize_frontier=initialize_active_frontier,
        )
        cpu_costs = {}
        if profile_light_s:
            for job in list(jobs) + list(active):
                estimate = float(cpu_estimate(job))
                if not math.isfinite(estimate) or estimate < 0:
                    raise ValueError('invalid CPU service estimate')
                cpu_costs[id(job)] = estimate
        def light(job):
            return bool(profile_light_s and cpu_costs[id(job)] <= profile_light_s)
        oldest = {}
        for job in sorted(jobs, key=lambda j: j['sequence']):
            key = ((str(job['tenant']), light(job)) if profile_light_s else
                   (str(job['tenant']), self.gpu_eligible(job)) if bypass_heavy else str(job['tenant']))
            oldest.setdefault(key, job)
        cpu_used = sum(job['prep_backend'] == cpu_backend for job in active)
        expensive_cpu_used = sum(job['prep_backend'] == cpu_backend and not
                                 job.get('joint_decision', {}).get('profile_light', light(job))
                                 for job in active) if profile_light_s else cpu_used
        gpu_active = [job for job in active if job['prep_backend'] == gpu_backend]
        cpu_active = [job for job in active if job['prep_backend'] == cpu_backend]
        used_lanes = sum(job['prep_gpu_lanes'] for job in gpu_active)
        available = gpu_lane_limit - used_lanes
        if used_lanes > self.gpu_capacity or cpu_used > self.cpu_capacity:
            raise RuntimeError('preparation allocation exceeds capacity')
        def preferred_gpu_demand(job):
            return (self.gpu_eligible(job) and not light(job) and
                    (not conservative_routing or
                     job['frame_count'] >= preferred_gpu_frame_threshold))
        gpu_tenants = {str(job['tenant']) for job in jobs
                       if preferred_gpu_demand(job)}
        gpu_tenants.update(str(job['tenant']) for job in gpu_active)
        fair_share_cap = (math.ceil(gpu_lane_limit / max(1, len(gpu_tenants)))
                          if order == 'fair' else gpu_lane_limit)
        def tenant_cap(tenant):
            if order != 'fair' or not fair_work_conserving_borrow:
                return fair_share_cap
            tenant_score = self.score(tenant)
            another_is_behind = any(
                other != tenant and self.score(other) < tenant_score - 1e-9
                for other in gpu_tenants
            )
            return fair_share_cap if another_is_behind else gpu_lane_limit
        candidates = []
        # FCFS chooses the oldest feasible request. A saturated GPU lane does
        # not deliberately prevent CPU-routed work from using a free CPU slot.
        if selection_policy == 'fair' and within_tenant_sjf:
            # Tenant selection remains service-first, but every request of the
            # selected tenant must be visible to the bounded local policy.
            considered = list(jobs)
        elif selection_policy in {'fair', 'tenant_round_robin'}:
            considered = list(oldest.values())
        elif selection_policy == 'fcfs':
            considered = sorted(jobs, key=lambda j: j['sequence'])
        else:
            # SJF must compare every pending request, including multiple jobs
            # from the same tenant.
            considered = list(jobs)
        for job in considered:
            tenant = str(job['tenant'])
            cap = tenant_cap(tenant)
            heavy = self.gpu_eligible(job)
            prefers_gpu = (job['modality'] == 'video' and
                           job['frame_count'] >= preferred_gpu_frame_threshold)
            choices = []
            assigned_backend = (
                job.get('presto_backend') if placement_policy == 'presto'
                else None
            )
            if (placement_policy == 'presto' and
                    assigned_backend not in (cpu_backend, gpu_backend)):
                raise ValueError('PRESTO placement requires an assigned backend')
            cpu_allowed = (not cpu_light_reserve or light(job) or
                           expensive_cpu_used < max(0, cpu_limit - cpu_light_reserve))
            if (cpu_used < cpu_limit and cpu_allowed and
                    (not fixed_routing or not heavy) and
                    assigned_backend in (None, cpu_backend)):
                cpu_service = float(cpu_estimate(job))
                choices.append((cpu_service, cpu_backend, 0, cpu_service))
            if (allow_gpu and heavy and not light(job) and
                    len(gpu_active) < gpu_job_limit and
                    assigned_backend in (None, gpu_backend)):
                held = sum(j['prep_gpu_lanes'] for j in gpu_active if str(j['tenant']) == tenant)
                # Fixed-lane baselines retain their width; dynamic allocation
                # caps per-tenant concurrent reservations under contention.
                available = gpu_lane_limit - used_lanes
                widths = ([fixed_lanes] if fixed_lanes else
                          [n for n in self.gpu_widths
                           if min_gpu_lanes <= n <= cap - held])
                for lanes in widths:
                    if 0 < lanes <= available:
                        gpu_service = float(gpu_estimate(job, lanes))
                        uncertainty = float(gpu_uncertainty(job, lanes))
                        if not math.isfinite(uncertainty) or uncertainty < 0:
                            raise ValueError('invalid GPU uncertainty estimate')
                        choices.append((gpu_service + uncertainty, gpu_backend,
                                        lanes, gpu_service))
            if choices:
                # A free CPU need not receive heavy work that would become
                # ready sooner after a near-term GPU release. Retain it in
                # the fair queue, but allow other feasible tenants to proceed.
                if (placement_policy == 'predicted_readiness' and
                        not fixed_routing and allow_gpu and heavy and
                        not light(job) and gpu_active):
                    future_widths = ([fixed_lanes] if fixed_lanes else
                                     [n for n in self.gpu_widths
                                      if min_gpu_lanes <= n <= cap])
                    releases = sorted((max(.05, j['predicted_prep_service_s'] - (now_s-j['prep_started_s'])), j['prep_gpu_lanes'], str(j['tenant']))
                                      for j in gpu_active if 'prep_started_s' in j and 'predicted_prep_service_s' in j)
                    future_ready = float('inf')
                    for width in future_widths:
                        free = gpu_lane_limit-used_lanes
                        active_jobs = len(gpu_active)
                        future_held = sum(j['prep_gpu_lanes'] for j in gpu_active if str(j['tenant']) == tenant)
                        for waiting, released, releasing_tenant in releases:
                            free += released
                            active_jobs -= 1
                            if releasing_tenant == tenant:
                                future_held -= released
                            if free >= width and active_jobs < gpu_job_limit and future_held + width <= cap:
                                future_ready = min(
                                    future_ready,
                                    waiting + gpu_estimate(job, width)
                                    + gpu_uncertainty(job, width),
                                )
                                break
                    choices = [choice for choice in choices if choice[1] != cpu_backend or choice[0] <= future_ready]
                    if not choices:
                        continue
                if (placement_policy == 'predicted_readiness' and
                        conservative_routing and not fixed_routing):
                    preferred_backend = gpu_backend if prefers_gpu else cpu_backend
                    preferred = [choice for choice in choices if choice[1] == preferred_backend]
                    alternatives = [choice for choice in choices if choice[1] != preferred_backend]
                    if preferred and alternatives:
                        best_preferred = min(preferred, key=lambda c: (c[0], c[2]))
                        margin = max(switch_margin_s, switch_margin_ratio * best_preferred[0])
                        better_alternatives = [choice for choice in alternatives
                                               if choice[0] + margin < best_preferred[0]]
                        choices = preferred + better_alternatives
                    elif alternatives and prefers_gpu:
                        # A heavy request waits for its preferred GPU path when
                        # the guard blocks new GPU work or no release estimate
                        # justifies paying the much larger CPU service cost.
                        if not allow_gpu or not gpu_active:
                            continue
                        future_widths = ([fixed_lanes] if fixed_lanes else
                                         [n for n in self.gpu_widths
                                          if min_gpu_lanes <= n <= cap])
                        releases = sorted(
                            (max(.05, j['predicted_prep_service_s'] -
                                 (now_s-j['prep_started_s'])),
                             j['prep_gpu_lanes'], str(j['tenant']))
                            for j in gpu_active
                            if 'prep_started_s' in j and
                            'predicted_prep_service_s' in j)
                        future_ready = float('inf')
                        for width in future_widths:
                            free = gpu_lane_limit-used_lanes
                            active_jobs = len(gpu_active)
                            future_held = sum(
                                j['prep_gpu_lanes'] for j in gpu_active
                                if str(j['tenant']) == tenant)
                            for waiting, released, releasing_tenant in releases:
                                free += released
                                active_jobs -= 1
                                if releasing_tenant == tenant:
                                    future_held -= released
                                if (free >= width and active_jobs < gpu_job_limit and
                                        future_held + width <= cap):
                                    future_ready = min(
                                        future_ready,
                                        waiting + gpu_estimate(job, width)
                                        + gpu_uncertainty(job, width))
                                    break
                        choices = [choice for choice in alternatives
                                   if choice[0] + max(
                                       switch_margin_s,
                                       switch_margin_ratio * future_ready,
                                   ) < future_ready]
                        if not choices:
                            continue
                    elif alternatives and not prefers_gpu:
                        # CPU-preferred work uses spare GPU capacity only when
                        # doing so clearly beats the next predicted CPU release.
                        releases = [
                            max(.05, j['predicted_prep_service_s'] -
                                (now_s-j['prep_started_s']))
                            for j in cpu_active
                            if 'prep_started_s' in j and
                            'predicted_prep_service_s' in j
                        ]
                        if not releases:
                            continue
                        future_ready = min(releases) + float(cpu_estimate(job))
                        choices = [choice for choice in alternatives
                                   if choice[0] + max(
                                       switch_margin_s,
                                       switch_margin_ratio * future_ready,
                                   ) < future_ready]
                        if not choices:
                            continue
                readiness_cost, backend, lanes, duration = min(
                    choices, key=lambda c: (c[0], c[2]))
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError('invalid service estimate')
                if selection_policy == 'fair' and profile_light_s:
                    aged = now_s - job.get('arrival_s', now_s) >= light_bypass_age_s
                    # Tenant service stays first. Within a tenant, cheap work
                    # bypasses expensive work until the latter ages.
                    rank = 0 if aged else 1 if light(job) else 2
                    key = (self.score(tenant), rank, job['sequence'])
                elif selection_policy == 'fair':
                    # Tenant ID is the deterministic tie break before any
                    # within-tenant ordering decision.
                    key = (self.score(tenant), tenant, job['sequence'])
                elif selection_policy == 'tenant_round_robin':
                    key = (tenant_dispatches.get(tenant, 0.0), job['sequence'])
                elif selection_policy == 'prep_sjf':
                    key = (duration, job['sequence'])
                elif selection_policy == 'prep_sjf_aging':
                    waited = max(0.0, now_s - float(job.get('arrival_s', now_s)))
                    key = ((0, float(job.get('arrival_s', 0.0)), job['sequence'])
                           if waited >= sjf_aging_s else
                           (1, duration, job['sequence']))
                else:
                    key = (job['sequence'],)
                candidates.append((key, job, dict(backend=backend, lanes=lanes,
                    predicted_service_s=duration, tenant_score=self.score(tenant),
                    placement_uncertainty_s=readiness_cost-duration,
                    risk_adjusted_service_s=readiness_cost,
                    profile_light=light(job), predicted_cpu_service_s=cpu_costs.get(id(job)),
                    tenant_lane_cap=cap, free_gpu_lanes=available if heavy and allow_gpu else self.gpu_capacity-used_lanes)))
                candidates[-1][2].update(
                    placement_policy=placement_policy,
                    presto_assignment=job.get('presto_assignment'),
                    conservative_routing=conservative_routing,
                    preferred_backend=(gpu_backend if prefers_gpu else cpu_backend),
                    switch_margin_s=switch_margin_s,
                    switch_margin_ratio=switch_margin_ratio,
                    min_gpu_lanes=min_gpu_lanes,
                    fairness_slack_s=fairness_slack_s,
                    service_frontier=self.service_frontier,
                    fair_work_conserving_borrow=fair_work_conserving_borrow,
                )
        if not candidates:
            return None
        if selection_policy == 'fair' and within_tenant_sjf:
            # Fairness first defines the eligible tenant set. Bounded SJF
            # chooses one request per eligible tenant. With nonzero slack, the
            # shortest of those bounded choices runs, allowing the scheduler
            # to see cheap work behind tenant heads without unbounded bypass.
            tenant_scores = {
                str(item[1]['tenant']): item[2]['tenant_score']
                for item in candidates
            }
            minimum_score = min(tenant_scores.values())
            if fairness_slack_s:
                eligible_tenants = {
                    tenant for tenant, score in tenant_scores.items()
                    if score <= minimum_score + fairness_slack_s
                }
            else:
                # Preserve strict Conductor's deterministic tenant choice when
                # several tenants have the same service score. Bounded SJF is
                # then confined to that one tenant.
                eligible_tenants = {
                    min(tenant_scores, key=lambda tenant: (
                        tenant_scores[tenant], tenant,
                    ))
                }
            live_heads = set()
            for tenant in {str(job['tenant']) for job in jobs}:
                tenant_head = min(
                    (job for job in jobs if str(job['tenant']) == tenant),
                    key=lambda job: job['sequence'],
                )
                live_heads.add((tenant, tenant_head['sequence']))
            self.within_tenant_bypasses = {
                key: value for key, value in self.within_tenant_bypasses.items()
                if key in live_heads
            }
            bounded_choices = []
            for tenant in eligible_tenants:
                tenant_candidates = [
                    item for item in candidates
                    if str(item[1]['tenant']) == tenant
                ]
                tenant_jobs = [
                    job for job in jobs if str(job['tenant']) == tenant
                ]
                head = min(tenant_jobs, key=lambda job: job['sequence'])
                head_key = (tenant, head['sequence'])
                head_candidate = next(
                    (item for item in tenant_candidates if item[1] is head),
                    None,
                )
                waited = max(
                    0.0, now_s - float(head.get('arrival_s', now_s)),
                )
                bypasses = self.within_tenant_bypasses.get(head_key, 0)
                force_head = (
                    head_candidate is not None
                    and (waited >= within_tenant_bypass_age_s
                         or bypasses >= within_tenant_bypass_limit)
                )
                if force_head:
                    choice = head_candidate
                    mode = ('aged_head'
                            if waited >= within_tenant_bypass_age_s
                            else 'bypass_limit_head')
                else:
                    choice = min(
                        tenant_candidates,
                        key=lambda item: (
                            item[2]['predicted_service_s'],
                            item[1]['sequence'],
                        ),
                    )
                    mode = 'shortest_feasible'
                bounded_choices.append(
                    (choice, head, head_key, waited, bypasses, mode)
                )
            choice, head, head_key, waited, bypasses, mode = min(
                bounded_choices,
                key=lambda value: (
                    value[0][2]['predicted_service_s'],
                    value[0][1]['sequence'],
                ),
            )
            _, selected, decision = choice
            if selected is head:
                self.within_tenant_bypasses.pop(head_key, None)
            else:
                self.within_tenant_bypasses[head_key] = bypasses + 1
            decision.update(
                within_tenant_selection='bounded_sjf',
                within_tenant_selection_mode=mode,
                within_tenant_head_sequence=head['sequence'],
                within_tenant_head_wait_s=waited,
                within_tenant_head_bypasses=(
                    bypasses if selected is head else bypasses + 1
                ),
                within_tenant_bypass_age_s=within_tenant_bypass_age_s,
                within_tenant_bypass_limit=within_tenant_bypass_limit,
                minimum_tenant_score=minimum_score,
                fairness_eligible_tenant_count=len(eligible_tenants),
            )
            return selected, decision
        if order == 'fair' and fairness_slack_s:
            minimum_score = min(item[2]['tenant_score'] for item in candidates)
            candidates = [
                item for item in candidates
                if item[2]['tenant_score'] <= minimum_score + fairness_slack_s
            ]
            # Fairness defines the eligible tenant set.  Within that bounded
            # set, prefer the work predicted to release its resource first.
            _, job, decision = min(
                candidates,
                key=lambda item: (
                    item[2]['predicted_service_s'], item[1]['sequence'],
                ),
            )
            decision['minimum_tenant_score'] = minimum_score
            return job, decision
        _, job, decision = min(candidates, key=lambda item: item[0])
        return job, decision

    def reserve(self, job, cpu_backend):
        tenant = str(job['tenant'])
        duration = job['predicted_prep_service_s']
        gpu = job['prep_backend'] != cpu_backend
        charge = duration * job['prep_gpu_lanes'] if gpu else duration
        account = self.gpu_service if gpu else self.cpu_service
        account[tenant] = account.get(tenant, 0) + charge
        job['joint_resource_reservation_s'] = charge

    def reconcile(self, job, duration, cpu_backend):
        tenant = str(job['tenant'])
        gpu = job['prep_backend'] != cpu_backend
        actual = duration * job['prep_gpu_lanes'] if gpu else duration
        account = self.gpu_service if gpu else self.cpu_service
        account[tenant] = max(0, account.get(tenant, 0) + actual - job['joint_resource_reservation_s'])
        job['prep_cpu_worker_s'] = 0 if gpu else actual
        job['prep_gpu_reserved_lane_s'] = actual if gpu else 0
