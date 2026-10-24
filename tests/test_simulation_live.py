"""Opt-in integration test: real Gazebo + PX4 communication, NEVER arms."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import json


@unittest.skipUnless(os.environ.get('AI_PROJECT_SITL_SMOKE')=='1', 'requires running real Gazebo/PX4 SITL')
class LiveSmoke(unittest.TestCase):
    def test_sensor_and_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'smoke.json'
            result = subprocess.run([sys.executable,'-B','-m','tello_nav.simulation.runner',
                '--simulation-only','--px4-root',os.environ['PX4_ROOT'],
                '--px4-pid',os.environ['PX4_PID'],'--smoke','--output',str(log)],
                capture_output=True,text=True,timeout=65)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            data = json.loads(log.read_text())
            self.assertTrue(data['metadata']['summary']['success'])
            self.assertFalse(data['metadata']['summary']['armed'])
