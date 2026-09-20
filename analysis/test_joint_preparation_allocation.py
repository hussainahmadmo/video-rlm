import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'conductor/experiments/scripts/run'))
from joint_preparation_allocation import JointPreparationAllocator


def job(tenant, sequence, frames=128):
    return dict(tenant=tenant, sequence=sequence, frame_count=frames, modality='video')


class JointTest(unittest.TestCase):
    def setUp(self):
        self.allocator = JointPreparationAllocator(2, 4, 2)

    def choose(self, jobs, active=None, **settings):
        return self.allocator.choose(jobs, active or [], 'cpu', 'gpu', lambda j: 30,
                                     lambda j, n: 12/n, **settings)

    def test_fair_share_caps_and_borrowing(self):
        _, decision = self.choose([job('a', 0), job('b', 1)])
        self.assertEqual(decision['lanes'], 2)
        _, decision = self.choose([job('a', 0)])
        self.assertEqual(decision['lanes'], 4)

    def test_adaptive_fcfs_has_no_tenant_lane_share_cap(self):
        jobs = [job('a', 0), job('b', 1), job('c', 2)]
        _, fair = self.choose(jobs)
        _, fcfs = self.choose(jobs, order='fcfs')
        self.assertEqual(fair['lanes'], 2)
        self.assertEqual(fcfs['lanes'], 4)

    def test_concurrent_reservations_count_against_tenant_cap(self):
        active = dict(job('a', 0), prep_backend='gpu', prep_gpu_lanes=2)
        selected, decision = self.choose([job('a', 1), job('b', 2)], [active])
        self.assertEqual(selected['tenant'], 'a')
        self.assertEqual(decision['backend'], 'cpu')
        self.allocator.gpu_service['a'] = 20
        selected, decision = self.choose([job('a', 1), job('b', 2)], [active])
        self.assertEqual(selected['tenant'], 'b')
        self.assertEqual(decision['lanes'], 2)

    def test_resource_reservation_and_reconciliation(self):
        j=dict(job('a', 0), prep_backend='gpu', prep_gpu_lanes=4, predicted_prep_service_s=3)
        self.allocator.reserve(j, 'cpu')
        self.assertEqual(self.allocator.gpu_service['a'], 12)
        self.allocator.reconcile(j, 5, 'cpu')
        self.assertEqual(self.allocator.gpu_service['a'], 20)
        self.assertEqual(j['prep_cpu_worker_s'], 0)
        self.assertEqual(j['prep_gpu_reserved_lane_s'], 20)

    def test_fcfs_and_fairness_both_use_available_cpu_capacity(self):
        active=[dict(job('a', 0), prep_backend='gpu', prep_gpu_lanes=4)]
        jobs=[job('b', 1), job('light', 2, 1)]
        selected, decision=self.choose(jobs, active, fixed_routing=True, fixed_lanes=2, order='fcfs')
        self.assertEqual(selected['tenant'], 'light')
        selected, decision=self.choose(jobs, active, fixed_routing=True, fixed_lanes=2)
        self.assertEqual(selected['tenant'], 'light')
        self.assertEqual(decision['backend'], 'cpu')

    def test_guard_and_lane_budget(self):
        _, decision=self.choose([job('a', 0)], allow_gpu=False)
        self.assertEqual(decision['backend'], 'cpu')

    def test_online_cpu_limit_does_not_change_gpu_capacity(self):
        active = [dict(job('light', 0, 1), prep_backend='cpu', prep_gpu_lanes=0)]
        selected, decision = self.choose(
            [job('heavy', 1), job('light', 2, 1)], active,
            cpu_limit=1, fixed_routing=True, fixed_lanes=2,
        )
        self.assertEqual(selected['tenant'], 'heavy')
        self.assertEqual(decision['backend'], 'gpu')
        self.assertEqual(decision['lanes'], 2)

    def profiled_choose(self, jobs, active=None, **settings):
        return self.allocator.choose(jobs, active or [], 'cpu', 'gpu',
                                     lambda j:j['cpu_cost'], lambda j,n:12/n,
                                     profile_light_s=1, **settings)

    def test_profiled_light_selection_uses_cost_not_frame_count(self):
        expensive=dict(job('a',0,1),cpu_cost=30)
        cheap=dict(job('a',1,128),cpu_cost=.2)
        selected,decision=self.profiled_choose([expensive,cheap])
        self.assertIs(selected,cheap)
        self.assertEqual(decision['backend'],'cpu')
        self.assertTrue(decision['profile_light'])

    def test_profiled_light_keeps_tenant_service_first(self):
        self.allocator.cpu_service['a']=20
        selected,_=self.profiled_choose([dict(job('a',0,1),cpu_cost=.2),
                                        dict(job('b',1),cpu_cost=30)])
        self.assertEqual(selected['tenant'],'b')

    def test_cpu_reserve_blocks_heavy_but_allows_light(self):
        active=[dict(job('a',0),cpu_cost=30,prep_backend='cpu',prep_gpu_lanes=0)]
        heavy=dict(job('b',1),cpu_cost=30)
        cheap=dict(job('b',2,1),cpu_cost=.2)
        self.assertIsNone(self.profiled_choose([heavy],active,allow_gpu=False,cpu_light_reserve=1))
        selected,decision=self.profiled_choose([heavy,cheap],active,allow_gpu=False,cpu_light_reserve=1)
        self.assertIs(selected,cheap)
        self.assertEqual(decision['backend'],'cpu')

    def test_aging_limits_within_tenant_light_bypass(self):
        heavy=dict(job('a',0),cpu_cost=30,arrival_s=0)
        cheap=dict(job('a',1,1),cpu_cost=.2,arrival_s=5.5)
        selected,_=self.profiled_choose([heavy,cheap],now_s=6,light_bypass_age_s=5)
        self.assertIs(selected,heavy)

    def test_light_bypasses_same_tenant_heavy_when_gpu_full(self):
        active=[dict(job('other', 0), prep_backend='gpu', prep_gpu_lanes=4)]
        jobs=[job('a', 1), job('a', 2, 1)]
        self.assertIsNone(self.choose(jobs, active, fixed_routing=True))
        selected, decision=self.choose(jobs, active, fixed_routing=True, bypass_heavy=True)
        self.assertEqual(selected['sequence'], 2)
        self.assertEqual(decision['backend'], 'cpu')

    def test_strict_routing_keeps_heavy_off_cpu_when_guard_blocks_gpu(self):
        self.assertIsNone(self.choose([job('a',0)], fixed_routing=True,
                                     bypass_heavy=True, allow_gpu=False))

    def test_waits_for_faster_gpu_without_blocking_light_tenant(self):
        active=[dict(job('a', 0), prep_backend='gpu', prep_gpu_lanes=4,
                     prep_started_s=0, predicted_prep_service_s=2)]
        self.assertIsNone(self.choose([job('b', 1)], active, now_s=1))
        selected, decision=self.choose([job('b', 1),job('light',2,1)], active, now_s=1)
        self.assertEqual(selected['tenant'], 'light')
        self.assertEqual(decision['backend'], 'cpu')
        active=[dict(job('a', 0), prep_backend='gpu', prep_gpu_lanes=4)]
        _, decision=self.choose([job('b', 1)], active)
        self.assertEqual(decision['backend'], 'cpu')

    def test_conservative_heavy_waits_for_preferred_gpu(self):
        active=[dict(job('a', 0), prep_backend='gpu', prep_gpu_lanes=4,
                     prep_started_s=0, predicted_prep_service_s=10)]
        choice=self.choose([job('b', 1)], active, now_s=1,
                           conservative_routing=True,
                           preferred_gpu_frame_threshold=32,
                           switch_margin_s=2, switch_margin_ratio=.2)
        self.assertIsNone(choice)

    def test_conservative_heavy_uses_cpu_when_gpu_wait_is_clearly_worse(self):
        active=[dict(job('a', 0), prep_backend='gpu', prep_gpu_lanes=4,
                     prep_started_s=0, predicted_prep_service_s=50)]
        _, decision=self.choose([job('b', 1)], active, now_s=1,
                                conservative_routing=True,
                                preferred_gpu_frame_threshold=32,
                                switch_margin_s=2, switch_margin_ratio=.2)
        self.assertEqual(decision['backend'], 'cpu')

    def test_conservative_route_requires_margin_for_cpu_preferred_work(self):
        allocator=JointPreparationAllocator(2,4,2,frame_threshold=1)
        settings=dict(conservative_routing=True,
                      preferred_gpu_frame_threshold=32,
                      switch_margin_s=2,switch_margin_ratio=.2)
        _, decision=allocator.choose([job('light',0,16)], [], 'cpu', 'gpu',
                                     lambda j:5, lambda j,n:3, **settings)
        self.assertEqual(decision['backend'],'cpu')
        _, decision=allocator.choose([job('light',0,16)], [], 'cpu', 'gpu',
                                     lambda j:5, lambda j,n:2, **settings)
        self.assertEqual(decision['backend'],'gpu')

    def test_conservative_guard_does_not_spill_heavy_to_cpu(self):
        choice=self.choose([job('a',0)], allow_gpu=False,
                           conservative_routing=True,
                           preferred_gpu_frame_threshold=32,
                           switch_margin_s=2,switch_margin_ratio=.2)
        self.assertIsNone(choice)

    def test_minimum_gpu_lanes_excludes_one_lane_choice(self):
        allocator=JointPreparationAllocator(2,4,2,frame_threshold=1)
        _, decision=allocator.choose(
            [job('a',0)], [], 'cpu', 'gpu',
            lambda j:30, lambda j,n:{1:3,2:4,4:8}[n],
            min_gpu_lanes=2,
        )
        self.assertEqual(decision['backend'],'gpu')
        self.assertEqual(decision['lanes'],2)

    def test_minimum_two_lanes_still_borrows_four_when_uncontended(self):
        _, decision = self.choose([job('a', 0)], min_gpu_lanes=2)
        self.assertEqual(decision['lanes'], 4)
        _, decision = self.choose(
            [job('a', 0), job('b', 1)], min_gpu_lanes=2,
        )
        self.assertEqual(decision['lanes'], 2)

    def test_fair_borrowing_uses_all_lanes_for_a_least_served_tenant(self):
        self.allocator.gpu_service['a'] = 8
        selected, decision = self.choose(
            [job('a', 0), job('b', 1)], min_gpu_lanes=2,
            fair_work_conserving_borrow=True,
        )
        self.assertEqual(selected['tenant'], 'b')
        self.assertEqual(decision['lanes'], 4)

    def test_fair_borrowing_caps_an_ahead_tenant_selected_with_slack(self):
        self.allocator.gpu_service['a'] = 4
        selected, decision = self.allocator.choose(
            [job('a', 0), job('b', 1)], [], 'cpu', 'gpu',
            lambda j: 30,
            lambda j, n: (1 if j['tenant'] == 'a' else 20) / n,
            min_gpu_lanes=2, fairness_slack_s=2,
            fair_work_conserving_borrow=True,
        )
        self.assertEqual(selected['tenant'], 'a')
        self.assertEqual(decision['lanes'], 2)

    def test_fairness_slack_prefers_shorter_feasible_work(self):
        self.allocator.cpu_service['fast'] = 4
        jobs = [job('slow', 0), job('fast', 1, 1)]
        selected, _ = self.allocator.choose(
            jobs, [], 'cpu', 'gpu',
            lambda j: 20 if j['tenant'] == 'slow' else 1,
            lambda j, n: 30,
            allow_gpu=False, fairness_slack_s=3,
        )
        self.assertEqual(selected['tenant'], 'fast')
        selected, _ = self.allocator.choose(
            jobs, [], 'cpu', 'gpu',
            lambda j: 20 if j['tenant'] == 'slow' else 1,
            lambda j, n: 30,
            allow_gpu=False, fairness_slack_s=1,
        )
        self.assertEqual(selected['tenant'], 'slow')

    def test_reactivated_tenant_starts_at_active_service_frontier(self):
        self.allocator.update_active_tenants(
            [job('a', 0)], [], initialize_frontier=True,
        )
        self.allocator.cpu_service.update(a=20, returning=100)
        self.allocator.update_active_tenants(
            [job('a', 0)], [], initialize_frontier=True,
        )
        self.allocator.update_active_tenants([], [], initialize_frontier=True)
        self.allocator.update_active_tenants(
            [job('returning', 1)], [], initialize_frontier=True,
        )
        self.assertEqual(self.allocator.score('returning'), 10)

    def test_arbitrary_cpu_capacity_and_gpu_width_set(self):
        allocator=JointPreparationAllocator(
            cpu_capacity=7, gpu_capacity=5, gpu_jobs=3,
            frame_threshold=1, gpu_widths=[2,3,5],
        )
        _, decision=allocator.choose(
            [job('a',0)], [], 'cpu', 'gpu',
            lambda j:30, lambda j,n:{2:5,3:2,5:3}[n],
        )
        self.assertEqual(allocator.cpu_capacity,7)
        self.assertEqual(allocator.gpu_widths,(2,3,5))
        self.assertEqual(decision['lanes'],3)

    def test_rejects_gpu_width_outside_lane_budget(self):
        with self.assertRaises(ValueError):
            JointPreparationAllocator(4,5,2,gpu_widths=[1,6])


if __name__ == '__main__':
    unittest.main()
