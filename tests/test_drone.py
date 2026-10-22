import time
import unittest
from unittest.mock import patch

import numpy as np

from tello_nav.controller import ControlCommand
from tello_nav.drone import MockDroneAdapter, TelloAdapter, create_adapter
from tello_nav.replay.providers import MockDepthProvider, synthetic_frames
from tello_nav.replay.runner import ReplayRunner


class FakeSDK:
    def __init__(self):
        self.calls = []
        self.fail_connect = False
        self.fail_send = False

    def connect(self):
        if self.fail_connect:
            raise OSError("not reachable")
        self.calls.append("connect")

    def send_rc_control(self, *channels):
        if self.fail_send:
            raise OSError("link failed")
        self.calls.append(channels)

    def streamon(self):
        self.calls.append("streamon")

    def streamoff(self):
        self.calls.append("streamoff")

    def get_current_state(self):
        return {"bat": 80, "vgx": 2}

    def get_frame_read(self):
        class Reader:
            frame = np.ones((60, 90, 3), np.uint8)
        return Reader()


class DroneTests(unittest.TestCase):
    def setUp(self):
        self.now = 1.0
        self.command = ControlCommand(forward_backward=10, state="FORWARD")

    def adapter(self):
        adapter = MockDroneAdapter(clock=lambda: self.now)
        adapter.connect()
        return adapter

    def test_factory_defaults_to_mock_and_replay_commands_work(self):
        adapter = create_adapter(clock=lambda: self.now)
        self.assertIsInstance(adapter, MockDroneAdapter)
        adapter.connect()
        rows = ReplayRunner(MockDepthProvider()).run(synthetic_frames(3), "test")
        for row in rows:
            command = ControlCommand(**row["command"])
            self.assertTrue(adapter.send(command, timestamp=self.now, observation_timestamp=self.now))
            self.now += .1
        self.assertEqual(adapter.commands[-1].state, "FORWARD")

    def test_disconnected_adapter_cannot_send(self):
        adapter = MockDroneAdapter()
        self.assertFalse(adapter.send(self.command, timestamp=0, observation_timestamp=0))
        self.assertEqual(adapter.commands, [])

    def test_freshness_and_command_limits(self):
        for command, stamp, observation in (
            (self.command, 0, 0), (self.command, 1, 0), (self.command, 2, 1),
            (self.command, 1, float("nan")),
            (ControlCommand(forward_backward=21, state="FORWARD"), 1, 1),
            (ControlCommand(forward_backward=10, state="STOP"), 1, 1),
        ):
            adapter = self.adapter()
            self.assertFalse(adapter.send(command, timestamp=stamp, observation_timestamp=observation))
            self.assertTrue(adapter.inhibited)
            self.assertEqual(adapter.commands[-1].channels, (0, 0, 0, 0))

    def test_watchdog_and_operator_override_latch(self):
        adapter = self.adapter()
        adapter.send(self.command, timestamp=1, observation_timestamp=1)
        self.now = 1.4
        adapter.watchdog()
        self.assertTrue(adapter.inhibited)
        self.assertEqual(adapter.commands[-1].state, "STOP")
        self.assertFalse(adapter.send(self.command, timestamp=1.4, observation_timestamp=1.4))
        with self.assertRaises(PermissionError):
            adapter.reset_operator_stop()
        adapter.reset_operator_stop(operator_authorized=True)
        self.assertTrue(adapter.send(self.command, timestamp=1.4, observation_timestamp=1.4))
        adapter.manual_override()
        self.assertTrue(adapter.inhibited)
        adapter.close()
        self.assertFalse(adapter.connected)

    def test_repeated_new_commands_cannot_extend_old_observation(self):
        adapter = self.adapter()
        self.now = 1.2
        self.assertTrue(adapter.send(self.command, timestamp=1.2, observation_timestamp=1))
        self.now = 1.31
        adapter.watchdog()
        self.assertTrue(adapter.inhibited)

    def test_live_requires_two_explicit_gates(self):
        for mode, auth in (("mock", True), ("live", False)):
            with self.assertRaises(PermissionError):
                TelloAdapter(mode=mode, operator_authorized=auth, sdk_factory=FakeSDK)
        with self.assertRaises(PermissionError):
            create_adapter("live")

    def live(self, sdk):
        adapter = TelloAdapter(mode="live", operator_authorized=True, sdk_factory=lambda: sdk,
                               clock=lambda: self.now)
        self.addCleanup(adapter.close)
        return adapter

    def test_fake_live_sdk_video_telemetry_commands_and_close(self):
        sdk = FakeSDK()
        adapter = self.live(sdk)
        adapter.connect()
        adapter.start_video()
        image, received = adapter.video_frame()
        self.assertEqual(image.shape, (60, 90, 3))
        self.assertEqual(received, 1)
        self.assertFalse(adapter.telemetry()["reliable"])
        self.assertTrue(adapter.send(self.command, timestamp=1, observation_timestamp=1))
        self.assertIn((0, 10, 0, 0), sdk.calls)
        adapter.close()
        self.assertEqual(sdk.calls[-2:], [(0, 0, 0, 0), "streamoff"])
        self.assertFalse(adapter.worker.is_alive())

    def test_connection_error(self):
        sdk = FakeSDK()
        sdk.fail_connect = True
        adapter = self.live(sdk)
        with self.assertRaises(ConnectionError):
            adapter.connect()
        self.assertFalse(adapter.connected)
        self.assertTrue(adapter.inhibited)
        self.assertFalse(adapter.send(self.command, timestamp=1, observation_timestamp=1))

    def test_command_error_latches_and_watchdog_retries_stop(self):
        sdk = FakeSDK()
        adapter = self.live(sdk)
        adapter.connect()
        sdk.fail_send = True
        self.assertFalse(adapter.send(self.command, timestamp=1, observation_timestamp=1))
        self.assertTrue(adapter.inhibited)
        self.assertIn("failed", adapter.last_error)
        sdk.fail_send = False
        adapter.watchdog()
        self.assertEqual(sdk.calls[-1], (0, 0, 0, 0))

    def test_video_and_telemetry_errors_inhibit_motion(self):
        for method in ("get_current_state", "get_frame_read", "streamon"):
            sdk = FakeSDK()
            adapter = self.live(sdk)
            adapter.connect()
            if method == "get_frame_read":
                adapter.start_video()
            with patch.object(sdk, method, side_effect=OSError("connection lost")):
                with self.assertRaises(OSError):
                    if method == "get_current_state":
                        adapter.telemetry()
                    elif method == "streamon":
                        adapter.start_video()
                    else:
                        adapter.video_frame()
            self.assertTrue(adapter.inhibited)
            self.assertEqual(sdk.calls[-1], (0, 0, 0, 0))
            adapter.close()

    def test_watchdog_runs_without_new_controller_commands(self):
        sdk = FakeSDK()
        adapter = self.live(sdk)
        adapter.connect()
        self.now = 2
        deadline = time.monotonic() + 1
        while not adapter.inhibited and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(adapter.inhibited)
        self.assertEqual(sdk.calls[-1], (0, 0, 0, 0))

    def test_mock_and_replay_do_not_import_sdk(self):
        import builtins
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.startswith("djitellopy"):
                raise AssertionError("mock/replay attempted hardware import")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guard):
            create_adapter().connect()
            ReplayRunner(MockDepthProvider()).run(synthetic_frames(2), "safe")
