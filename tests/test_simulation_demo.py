"""Lifecycle guards; actual GUI/flight verification is recorded separately."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tello_nav.simulation import demo


class DemoLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        self.patch = patch.object(demo, 'STATE', self.state)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_recycled_pid_is_never_signalled(self):
        record = {'pid': os.getpid(), 'start': 'incorrect-start-tick'}
        with patch.object(demo.os, 'kill') as kill, patch.object(demo.os, 'killpg') as group:
            demo.terminate(record)
            demo.terminate(record, group=True)
        kill.assert_not_called()
        group.assert_not_called()

    def test_owned_process_can_be_stopped_without_affecting_other_process(self):
        owned = subprocess.Popen(['sleep', '30'], start_new_session=True)
        unrelated = subprocess.Popen(['sleep', '30'], start_new_session=True)
        self.addCleanup(lambda: unrelated.poll() is None and unrelated.terminate())
        self.addCleanup(lambda: unrelated.wait(timeout=5))
        try:
            demo.terminate(demo.identity(owned.pid), group=True)
            owned.wait(timeout=5)
            self.assertIsNone(unrelated.poll())
        finally:
            if owned.poll() is None:
                owned.terminate()
                owned.wait()
            unrelated.terminate()
            unrelated.wait()

    def test_smoke_and_flight_commands_are_separate_and_use_verified_config(self):
        config = {'root': '/tmp/px4', 'px4': {'pid': 123}}
        smoke = demo.runner_command(config, smoke=True)
        flight = demo.runner_command(config)
        self.assertIn('--smoke', smoke)
        self.assertNotIn('--smoke', flight)
        for command in (smoke, flight):
            self.assertIn('--simulation-only', command)
            self.assertEqual(command[command.index('--experiment') + 1], 'wall')
            self.assertEqual(command[command.index('--config') + 1],
                             str(demo.PROJECT / 'simulation/configs/x500.json'))

    def test_start_rejects_missing_demo_before_spawning_anything(self):
        with patch.object(demo.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'No live prepared demo'):
                demo.start()
        spawn.assert_not_called()

    def test_start_rejects_reusing_flown_spawn(self):
        with patch.object(demo, 'require_prepared', return_value={'flown': True}), \
                patch.object(demo.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'already been used'):
                demo.start()
        spawn.assert_not_called()

    def test_prepare_is_idempotent_and_never_starts_flight(self):
        demo.save('supervisor', demo.identity(os.getpid()))
        with patch.object(demo, 'require_prepared', return_value={'flown': False}), \
                patch.object(demo.subprocess, 'Popen') as spawn, \
                patch.object(demo.subprocess, 'run') as run:
            demo.prepare('/unused')
        spawn.assert_not_called()
        run.assert_not_called()

    def test_stop_with_no_live_owned_process_is_harmless(self):
        demo.save('supervisor', {'pid': os.getpid(), 'start': 'old'})
        with patch.object(demo.os, 'kill') as kill:
            demo.stop()
        kill.assert_not_called()

    def test_gui_config_does_not_offer_pause_reset_controls(self):
        config = (demo.PROJECT / 'simulation/configs/demo_gui.config').read_text()
        self.assertIn('GzSceneManager', config)
        self.assertNotIn('filename="WorldControl"', config)
        self.assertIn('<camera_pose>-4 -7 4', config)


if __name__ == '__main__':
    unittest.main()
