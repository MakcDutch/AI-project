from dataclasses import replace
import unittest
from tello_nav.simulation.safety_monitor import SafetyMonitor, Limits, Velocity
from tello_nav.simulation.runner import summarize


class SafetyTests(unittest.TestCase):
    def inputs(self, **changes):
        args = dict(now=10., sensor_received=9.9, telemetry_received=9.9,
                    command_received=9.9, connected=True, altitude=1.5, travel=1.,clearance=3.)
        args.update(changes)
        return args

    def test_valid(self):
        v = Velocity(.45)
        self.assertEqual(SafetyMonitor().check(v,**self.inputs()),(v,None))

    def test_independent_faults_latch_zero(self):
        for changes in [dict(connected=False),dict(sensor_received=9),dict(telemetry_received=None),
                        dict(command_received=9),dict(sensor_received=11),dict(altitude=4),
                        dict(altitude=.1),dict(travel=9),dict(clearance=.5),dict(collision=True),
                        dict(emergency=True),dict(now=float('nan')),dict(clearance=float('inf'))]:
            with self.subTest(changes=changes):
                monitor = SafetyMonitor()
                v,reason = monitor.check(Velocity(.4),**self.inputs(**changes))
                self.assertEqual(v,Velocity())
                self.assertTrue(reason)
                self.assertEqual(monitor.check(Velocity(.4),**self.inputs()),(Velocity(),reason))

    def test_envelope_rejection(self):
        for v in [Velocity(.61),Velocity(.5,.5),Velocity(down_m_s=.31),Velocity(yaw_deg_s=16),Velocity(float('nan'))]:
            self.assertEqual(SafetyMonitor().check(v,**self.inputs())[0],Velocity())

    def test_invalid_config(self):
        for changes in [dict(horizontal_m_s=0),dict(sensor_timeout_s=-1),dict(altitude_m=float('inf')),dict(rc_limit=0)]:
            with self.assertRaises(ValueError):
                Limits(**changes)

    def test_empty_log_cannot_report_flight_success(self):
        for experiment in ['clear','wall','stale']:
            summary = summarize([],experiment)
            self.assertFalse(summary['success'])
            self.assertIsNone(summary['collision'])
            self.assertIsNone(summary['actual_stopping_distance_m'])


class MeasurementTests(unittest.TestCase):
    def rows(self):
        rows=[]
        # Test inputs only; never exported as actual simulation evidence.
        for i in range(51):
            stamp=i*.1
            moving=i<20
            x=.04*i if moving else .8+min((i-20)*.01,.1)
            rows.append(dict(position={'x':x,'y':0,'z':1.5},simulation_time_s=stamp,
                actual_speed_m_s=.4 if moving else (.1 if i<30 else .02),
                controller_state='FORWARD' if moving else 'BRAKE',
                safety_event=None,collision=False,contact_monitor_available=True,
                travel_m=x,wall_clearance_m=6-x-.5,center_depth_m=6-x,
                depth_interrupted=False,flight_mode='OFFBOARD'))
        return rows

    def test_measured_stopping_metrics(self):
        summary=summarize(self.rows(),'wall')
        self.assertTrue(summary['success'])
        self.assertAlmostEqual(summary['actual_stopping_distance_m'],.1)
        self.assertAlmostEqual(summary['detection_to_stop_s'],1)
        self.assertEqual(summary['peak_velocity_m_s'],.4)

    def test_no_contact_monitor_cannot_claim_collision_free(self):
        rows=self.rows()
        rows[0]['contact_monitor_available']=False
        summary=summarize(rows,'wall')
        self.assertFalse(summary['success'])
        self.assertIsNone(summary['collision'])

    def test_zero_setpoint_does_not_prove_stop(self):
        rows=self.rows()
        for row in rows:
            row['actual_speed_m_s']=.4
        summary=summarize(rows,'wall')
        self.assertFalse(summary['success'])
        self.assertIsNone(summary['actual_stopping_distance_m'])

    def test_failure_invalidates_good_measurements(self):
        self.assertFalse(summarize(self.rows(),'wall','landing failed')['success'])


class PlotTests(unittest.TestCase):
    def test_plot_measured_log_schema(self):
        import importlib.util
        if importlib.util.find_spec('matplotlib') is None:
            self.skipTest('optional simulation plotting dependency missing')
        import json
        from pathlib import Path
        import tempfile
        from tello_nav.simulation.plots import main
        rows=MeasurementTests().rows()
        for r in rows:
            r['elapsed_s']=r['simulation_time_s']
            r['desired_velocity']={'forward_m_s':.4 if r['controller_state']=='FORWARD' else 0}
            r['applied_velocity']=r['desired_velocity']
            r['actual_forward_m_s']=r['actual_speed_m_s']
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'test_only.json'
            output=Path(directory)/'test_only.png'
            log.write_text(json.dumps({'records':rows,'metadata':{'experiment':'unit test input',
                'summary':{'success':False}}}))
            main([str(log),'--output',str(output)])
            self.assertEqual(output.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            self.assertGreater(output.stat().st_size,10000)


class MissingTelemetrySummaryTests(unittest.TestCase):
    def test_invalid_telemetry_keeps_failure_report_serializable(self):
        import json
        rows=MeasurementTests().rows()
        rows[-1]['actual_speed_m_s']=None
        rows[-1]['safety_event']='nonfinite vehicle telemetry'
        summary=summarize(rows,'wall')
        self.assertFalse(summary['success'])
        self.assertTrue(summary['measurement_gaps'])
        json.dumps(summary,allow_nan=False)
