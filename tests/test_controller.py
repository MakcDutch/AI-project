import unittest
from dataclasses import replace

import numpy as np

from tello_nav.controller import ControlCommand, ControllerConfig, ReactiveController
from tello_nav.controller.sectors import SectorConfig, analyze


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.cfg = ControllerConfig(allow_lateral=True, recovery_frames=2,
                                    brake_duration_s=.2, avoid_hold_s=.3)
        self.controller = ReactiveController(self.cfg, lambda: self.now)
        self.depth = np.full((60, 90), 4.0)

    def compute(self, depth=None, dt=.1):
        self.now += dt
        return self.controller.compute(self.depth if depth is None else depth, self.now)

    def forward(self):
        self.assertEqual(self.compute().state, "STOP")
        self.assertEqual(self.compute().state, "FORWARD")

    def test_free_path_and_bounds(self):
        self.forward()
        command = self.compute()
        command.validate(self.cfg.command_limit)
        self.assertEqual(command.forward_backward, 15)

    def test_center_brakes_then_avoids_left(self):
        self.forward()
        self.depth[:, 30:60] = .5
        self.assertEqual(self.compute().state, "BRAKE")
        self.assertEqual(self.compute().channels, (0, 0, 0, 0))
        self.assertEqual(self.compute(dt=.21).state, "AVOID_LEFT")

    def test_choose_right_when_left_blocked(self):
        self.depth[:, :60] = .5
        self.assertEqual(self.compute().state, "BRAKE")
        self.assertEqual(self.compute(dt=.21).state, "AVOID_RIGHT")

    def test_choose_left_when_right_blocked(self):
        self.depth[:, 30:] = .5
        self.assertEqual(self.compute().state, "BRAKE")
        self.assertEqual(self.compute(dt=.21).state, "AVOID_LEFT")

    def test_left_or_right_obstacle_does_not_force_center_avoidance(self):
        for crop in (slice(0, 30), slice(60, 90)):
            self.setUp()
            self.depth[:, crop] = .5
            self.forward()

    def test_all_blocked_stops(self):
        self.depth[:] = .5
        self.assertEqual(self.compute().state, "BRAKE")
        self.assertEqual(self.compute(dt=.21).state, "STOP")

    def test_lateral_disabled_by_default(self):
        self.controller = ReactiveController(replace(self.cfg, allow_lateral=False), lambda: self.now)
        self.depth[:, 30:60] = .5
        self.compute()
        self.assertEqual(self.compute(dt=.21).state, "STOP")
        self.assertFalse(any(self.compute().channels))

    def test_invalid_values_and_missing_maps(self):
        for value in (np.nan, np.inf, -np.inf, 0, -1):
            with self.subTest(value=value):
                self.depth[:] = value
                self.assertEqual(self.compute().state, "STOP")
        self.now += .1
        self.assertEqual(self.controller.compute(None, self.now).state, "STOP")
        for bad in (np.zeros((1,)), np.ones((2, 2, 3)), np.array([["x"]]), np.ones((2, 2), complex)):
            self.assertEqual(self.compute(bad).state, "STOP")

    def test_partial_coverage_unknown(self):
        self.depth[:, :10] = np.nan
        cmd = self.compute()
        self.assertEqual(cmd.state, "STOP")
        self.assertIn("coverage", cmd.reason)

    def test_small_invalid_patch_is_ignored(self):
        self.depth[20, 10] = np.nan
        self.forward()
        self.assertLess(self.controller.sectors["left"].valid_coverage, 1)

    def test_stale_future_and_duplicate(self):
        self.now = 2
        for stamp in (0, 3, float("nan")):
            self.assertEqual(self.controller.compute(self.depth, stamp).state, "STOP")
        self.controller.compute(self.depth, 2)
        self.assertEqual(self.controller.compute(self.depth, 2).state, "STOP")

    def test_failure_requires_recovery(self):
        self.forward()
        self.now += .1
        self.assertEqual(self.controller.compute(None, self.now).state, "STOP")
        self.assertEqual(self.compute().state, "STOP")
        self.assertEqual(self.compute().state, "FORWARD")

    def test_hysteresis(self):
        self.forward()
        self.depth[:, 30:60] = .5
        self.compute()
        self.depth[:, 30:60] = 1.1
        self.assertEqual(self.compute(dt=.3).state, "STOP")
        for _ in range(5):
            self.assertEqual(self.compute().state, "STOP")
        self.depth[:] = 1.3
        self.assertEqual(self.compute().state, "STOP")
        self.assertEqual(self.compute().state, "FORWARD")

    def test_no_direct_left_right_switch(self):
        self.depth[:, 30:60] = .5
        self.depth[:, :30] = 5
        self.compute()
        self.assertEqual(self.compute(dt=.21).state, "AVOID_LEFT")
        self.depth[:, 60:] = 8
        self.assertEqual(self.compute().state, "AVOID_LEFT")
        self.depth[:, :30] = .5
        self.assertEqual(self.compute().state, "BRAKE")
        self.assertEqual(self.compute().state, "BRAKE")
        self.assertEqual(self.compute(dt=.21).state, "AVOID_RIGHT")

    def test_avoidance_recovery_brakes_before_forward(self):
        self.depth[:, 30:60] = .5
        self.compute()
        self.assertEqual(self.compute(dt=.21).state, "AVOID_LEFT")
        self.depth[:] = 4
        self.compute()
        self.assertEqual(self.compute(dt=.31).state, "BRAKE")
        self.assertEqual(self.compute(dt=.21).state, "FORWARD")

    def test_single_pixel_obstacle_is_not_lost_in_percentile(self):
        self.forward()
        self.depth[20, 45] = .2
        self.assertEqual(self.compute().state, "BRAKE")
        stats = self.controller.sectors["center"]
        self.assertEqual(stats.distance_m, 4)
        self.assertEqual(stats.minimum_m, .2)

    def test_dynamic_distance(self):
        self.controller = ReactiveController(replace(self.cfg, braking_deceleration_mps2=1), lambda: self.now)
        self.now = 1
        self.controller.compute(self.depth, .9, {"reliable": True, "speed_mps": 2, "timestamp": 1})
        self.assertAlmostEqual(self.controller.safe_distance_m, 3.2)
        self.now = 1.1
        self.controller.compute(self.depth, 1, {"reliable": False, "speed_mps": 100, "timestamp": 1})
        self.assertEqual(self.controller.safe_distance_m, 1)
        self.now = 2
        self.controller.compute(self.depth, 2, {"reliable": True, "speed_mps": 100, "timestamp": 0})
        self.assertEqual(self.controller.safe_distance_m, 1)

    def test_config_and_command_validation(self):
        for kwargs in ({"min_distance_m": 0}, {"max_frame_age_s": float("nan")},
                       {"forward_speed": 30}, {"braking_deceleration_mps2": -1},
                       {"recovery_frames": 0}, {"command_limit": 101},
                       {"allow_lateral": "false"}):
            with self.assertRaises(ValueError):
                ControllerConfig(**kwargs)
        for cmd in (ControlCommand(left_right=101), ControlCommand(yaw=1),
                    ControlCommand(state="FORWARD", forward_backward=2.5)):
            with self.assertRaises(ValueError):
                cmd.validate()

    def test_dynamic_distance_overflow_stops(self):
        self.controller = ReactiveController(replace(self.cfg, braking_deceleration_mps2=1), lambda: self.now)
        self.now = 1
        command = self.controller.compute(self.depth, 1, {
            "reliable": True, "speed_mps": 1e300, "timestamp": 1,
        })
        self.assertEqual(command.state, "STOP")
        self.assertIn("overflow", command.reason)


class SectorTests(unittest.TestCase):
    def test_roi_excludes_unobserved_area(self):
        depth = np.ones((60, 90)) * 3
        depth[:10] = .1
        self.assertTrue(analyze(depth, SectorConfig())["center"].clear(1))
        self.assertFalse(analyze(depth, SectorConfig(roi=(0, 0, 1, 1)))["center"].clear(1))

    def test_tiny_or_empty_roi_is_unknown(self):
        stats = analyze(np.ones((1, 1)), SectorConfig())
        self.assertFalse(all(s.known for s in stats.values()))

    def test_invalid_roi(self):
        with self.assertRaises(ValueError):
            SectorConfig(roi=(1, 0, 0, 1))
