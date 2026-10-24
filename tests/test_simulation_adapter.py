import asyncio
from dataclasses import replace
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock

import numpy as np

from tello_nav.controller import ControlCommand, ReactiveController
from tello_nav.simulation.gazebo_camera import decode_depth, GazeboCamera
from tello_nav.simulation.safety_monitor import Limits, Velocity, map_command, enu_to_ned, ned_to_enu
from tello_nav.simulation.px4_adapter import PX4Adapter, verify_sitl


def image(values=None, **kwargs):
    values = np.array([[1, 2], [3, 4]], dtype='<f4') if values is None else values
    fields = dict(width=2, height=2, step=8, data=values.tobytes(), pixel_format_type=13,
                  header=NS(stamp=NS(sec=12, nsec=250000000)))
    fields.update(kwargs)
    return NS(**fields)


class MappingTests(unittest.TestCase):
    def test_forward_si(self):
        v = map_command(ControlCommand(forward_backward=15, state='FORWARD'))
        self.assertAlmostEqual(v.forward_m_s, .45)
        self.assertEqual(v.right_m_s, 0)

    def test_right_up_yaw_and_diagonal_limit(self):
        v = map_command(ControlCommand(left_right=20, forward_backward=20, up_down=20,
                                       yaw=20, state='FORWARD'))
        self.assertAlmostEqual(np.hypot(v.forward_m_s,v.right_m_s), .6)
        self.assertEqual(v.down_m_s, -.3)
        self.assertEqual(v.yaw_deg_s, 15)

    def test_stop_brake(self):
        for state in ['STOP','BRAKE']:
            self.assertEqual(map_command(ControlCommand(state=state)), Velocity())
            with self.assertRaises(ValueError):
                map_command(ControlCommand(forward_backward=1,state=state))

    def test_invalid_rc(self):
        for rc in [21, True, float('nan')]:
            with self.assertRaises(ValueError):
                map_command(ControlCommand(forward_backward=rc,state='FORWARD'))

    def test_coordinates(self):
        self.assertEqual(enu_to_ned(1,2,3), (2,1,-3))
        self.assertEqual(ned_to_enu(*enu_to_ned(1,2,3)), (1,2,3))

    def test_guard_rejects_unrelated_process(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises((PermissionError, FileNotFoundError)):
                verify_sitl(directory, 1)
        with self.assertRaises(PermissionError):
            PX4Adapter(root='/',pid=1)


class CameraTests(unittest.TestCase):
    def test_units_shape_stamp(self):
        depth, stamp = decode_depth(image())
        self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(depth.shape,(2,2))
        self.assertEqual(depth[1,1],4)
        self.assertEqual(stamp,12.25)

    def test_invalid_values_unknown(self):
        d,_ = decode_depth(image(np.array([[0,-1],[np.inf,np.nan]],dtype='<f4')))
        self.assertTrue(np.isnan(d).all())

    def test_padding(self):
        data = np.array([[1,2,999],[3,4,999]], dtype='<f4').tobytes()
        d,_ = decode_depth(image(step=12,data=data))
        np.testing.assert_array_equal(d,[[1,2],[3,4]])

    def test_invalid_metadata(self):
        for changes in [dict(width=0),dict(width=5000),dict(step=7),dict(data=b'bad'),
                        dict(pixel_format_type=3),dict(header=NS(stamp=NS(sec=-1,nsec=0))),
                        dict(header=NS(stamp=NS(sec=0,nsec=1000000000)))]:
            with self.assertRaises(ValueError):
                decode_depth(image(**changes))

    def test_controller_accepts_decoded_metric_frame(self):
        data = np.full((60,90),4,dtype='<f4')
        d,stamp = decode_depth(image(data,width=90,height=60,step=360))
        controller = ReactiveController(clock=lambda: stamp)
        controller.compute(d,stamp)
        controller.clock = lambda: stamp+.1
        self.assertEqual(controller.compute(d,stamp+.1).state,'FORWARD')


class AdapterFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_setpoint_is_propagated(self):
        adapter = object.__new__(PX4Adapter)
        adapter.drone = NS(offboard=NS(set_velocity_body=AsyncMock(side_effect=ConnectionError('lost'))))
        adapter.failure = None
        # Test the SDK boundary without requiring MAVSDK for the unit suite.
        import sys
        from unittest.mock import patch
        with patch.dict(sys.modules, {'mavsdk.offboard':NS(VelocityBodyYawspeed=lambda *x:x)}):
            with self.assertRaises(ConnectionError):
                await adapter.send(Velocity(.2))
        self.assertIn('lost',adapter.failure)

    async def test_landing_failure_never_disarms(self):
        adapter = object.__new__(PX4Adapter)
        disarm = AsyncMock()
        adapter.drone = NS(action=NS(land=AsyncMock(side_effect=ConnectionError('lost')),disarm=disarm))
        adapter.armed = True
        adapter.send = AsyncMock()
        with self.assertRaises(ConnectionError):
            await adapter.land()
        disarm.assert_not_awaited()

    async def test_confirmed_landing_disarms(self):
        async def states():
            for name in ['IN_AIR','LANDING','ON_GROUND']:
                yield NS(name=name)
        adapter = object.__new__(PX4Adapter)
        adapter.drone = NS(action=NS(land=AsyncMock(),disarm=AsyncMock()),
                           telemetry=NS(landed_state=states))
        adapter.send = AsyncMock()
        adapter.armed = True
        await adapter.land()
        self.assertFalse(adapter.armed)
        adapter.drone.action.disarm.assert_awaited_once()


class DiscoveryTests(unittest.TestCase):
    def test_transport_publisher_uses_property(self):
        from unittest.mock import patch
        import sys
        camera = object.__new__(GazeboCamera)
        camera.node = NS(topic_list=lambda:['/depth_camera','/depth_camera/points',
                             '/world/w/model/x500_depth_0/link/camera/sensor/IMX214/image'],
                         topic_info=lambda t:([NS(msg_type_name='gz.msgs.Image')],[]),
                         subscribe=lambda *args:True)
        camera.topic = None
        with patch.dict(sys.modules, {'gz.msgs10.image_pb2':NS(Image=object)}):
            self.assertEqual(camera.discover(),'/depth_camera')

    def test_ambiguous_depth_topics_rejected(self):
        from unittest.mock import patch
        import sys
        camera = object.__new__(GazeboCamera)
        camera.node = NS(topic_list=lambda:['/depth_one','/depth_two'],
                         topic_info=lambda t:([NS(msg_type_name='gz.msgs.Image')],[]))
        with patch.dict(sys.modules, {'gz.msgs10.image_pb2':NS(Image=object)}):
            with self.assertRaises(RuntimeError):
                camera.discover()


class IndependentWatchdogTests(unittest.TestCase):
    def test_stalled_command_loop_terminates_only_owned_sdk(self):
        from unittest.mock import Mock
        adapter = object.__new__(PX4Adapter)
        adapter.offboard = True
        adapter.last_send = 1.
        adapter.failure = None
        adapter.server_process = NS(poll=lambda:None,kill=Mock())
        adapter._watchdog_step(1.1)
        adapter.server_process.kill.assert_not_called()
        adapter._watchdog_step(1.31)
        adapter.server_process.kill.assert_called_once()
        self.assertFalse(adapter.offboard)
        self.assertIn('watchdog',adapter.failure)


class ContactTests(unittest.TestCase):
    def test_entity_ids_without_names_still_latch_wall_collision(self):
        import threading
        camera=object.__new__(GazeboCamera)
        camera.lock=threading.Lock()
        camera.collision=False
        camera.collision_events=[]
        message=NS(header=NS(stamp=NS(sec=4,nsec=500000000)), contact=[
            NS(collision1=NS(id=1,name=''),collision2=NS(id=2,name=''),position=[object()])])
        camera._contact(message)
        self.assertTrue(camera.collision)
        self.assertEqual(camera.collision_events[0]['collision2']['id'],2)
        self.assertEqual(camera.collision_events[0]['timestamp_s'],4.5)


class OffboardEntryTests(unittest.IsolatedAsyncioTestCase):
    async def test_delayed_mode_telemetry_keeps_watchdog_fresh_with_zero_stream(self):
        from unittest.mock import Mock
        import time
        adapter=object.__new__(PX4Adapter)
        adapter.drone=NS(offboard=NS(start=AsyncMock()))
        adapter.telemetry=NS(mode='HOLD')
        adapter.offboard=False
        adapter.last_send=time.monotonic()
        adapter.lifecycle=[]
        adapter.failure=None
        adapter.server_process=NS(poll=lambda:None,kill=Mock())
        sent=[]
        async def send(value):
            sent.append(value)
            adapter.last_send=time.monotonic()
        adapter.send=send
        async def monitor():
            start=time.monotonic()
            while time.monotonic()-start<.45:
                adapter._watchdog_step(time.monotonic())
                await asyncio.sleep(.02)
            adapter.telemetry.mode='OFFBOARD'
        task=asyncio.create_task(monitor())
        await adapter._enter_offboard()
        await task
        self.assertGreaterEqual(len(sent),5)
        self.assertTrue(all(v==Velocity() for v in sent))
        adapter.server_process.kill.assert_not_called()
        self.assertTrue(adapter.offboard)


class TelemetryClockTests(unittest.TestCase):
    def test_receipt_clocks_stay_in_their_original_domains(self):
        from unittest.mock import patch
        from tello_nav.simulation.px4_adapter import Telemetry
        adapter=object.__new__(PX4Adapter)
        adapter.telemetry=Telemetry()
        adapter.telemetry_history=[]
        adapter.simulation_snapshot=lambda:{'sim_time':10.,'pose':None}
        state=NS(position=NS(north_m=1.,east_m=2.,down_m=-1.5),
                 velocity=NS(north_m_s=0.,east_m_s=.45,down_m_s=0.))
        with patch('tello_nav.simulation.px4_adapter.time.monotonic',return_value=100000.):
            adapter._position(state)
        self.assertEqual(adapter.telemetry.received,100000.)
        self.assertEqual(adapter.telemetry.sim_received,10.)
        self.assertEqual(adapter.telemetry_history[0]['simulation_time_s'],10.)
