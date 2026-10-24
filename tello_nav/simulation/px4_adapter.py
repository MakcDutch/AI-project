"""MAVSDK boundary restricted to a verified local PX4 SITL process."""
import asyncio
from contextlib import suppress
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
import threading
import subprocess

from .safety_monitor import Velocity


def verify_sitl(root, pid):
    root = Path(root).resolve(strict=True)
    binary = root / 'build/px4_sitl_default/bin/px4'
    if not (root / 'ROMFS/px4fmu_common/init.d-posix/airframes/4002_gz_x500_depth').is_file():
        raise PermissionError('PX4 checkout with X500 depth airframe required')
    if pid <= 1 or Path(f'/proc/{pid}/exe').resolve(strict=True) != binary.resolve(strict=True):
        raise PermissionError('PID must identify this checkout\'s running PX4 SITL binary')
    os.kill(pid, 0)
    return root


@dataclass
class Telemetry:
    connected: bool = False
    north: float = 0.0
    east: float = 0.0
    down: float = 0.0
    vn: float = 0.0
    ve: float = 0.0
    vd: float = 0.0
    received: float | None = None
    sim_received: float | None = None
    connection_received: float | None = None
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    attitude_received: float | None = None
    mode: str = 'UNKNOWN'
    mode_received: float | None = None
    in_air: bool = False

    @property
    def speed(self):
        return math.sqrt(self.vn**2 + self.ve**2 + self.vd**2)

    @property
    def forward_speed(self):
        yaw = math.radians(self.yaw_deg)
        return self.vn*math.cos(yaw) + self.ve*math.sin(yaw)


class PX4Adapter:
    def __init__(self, *, root, pid, simulation_only=False):
        if simulation_only is not True:
            raise PermissionError('explicit --simulation-only required')
        self.root = verify_sitl(root, pid)
        self.pid = pid
        self.telemetry = Telemetry()
        self.tasks = []
        self.telemetry_history = []
        self.lifecycle = []
        self.simulation_snapshot = None
        self.drone = None
        self.offboard = False
        self.armed = False
        self.failure = None
        self.last_send = None
        self.server_process = None
        self.watchdog_stop = threading.Event()
        self.watchdog_thread = None

    async def _consume(self, stream, update):
        try:
            async for item in stream:
                update(item)
            self.failure = 'telemetry stream ended unexpectedly'
            self.telemetry.connected = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = f'telemetry stream failed: {exc}'
            self.telemetry.connected = False

    def _connection(self, state):
        self.telemetry.connected = state.is_connected
        self.telemetry.connection_received = time.monotonic()

    def _position(self, state):
        t = self.telemetry
        t.north, t.east, t.down = state.position.north_m, state.position.east_m, state.position.down_m
        t.vn, t.ve, t.vd = state.velocity.north_m_s, state.velocity.east_m_s, state.velocity.down_m_s
        t.received = time.monotonic()
        snapshot=self.simulation_snapshot() if self.simulation_snapshot else {}
        t.sim_received=snapshot.get('sim_time')
        if len(self.telemetry_history)<20000:
            self.telemetry_history.append(dict(monotonic_time_s=t.received,
                simulation_time_s=snapshot.get('sim_time'),gazebo_pose=snapshot.get('pose'),
                north_m=t.north,east_m=t.east,down_m=t.down,
                vn_m_s=t.vn,ve_m_s=t.ve,vd_m_s=t.vd,yaw_deg=t.yaw_deg,
                in_air=t.in_air,flight_mode=t.mode))

    def _attitude(self, state):
        t = self.telemetry
        t.yaw_deg, t.roll_deg, t.pitch_deg = state.yaw_deg, state.roll_deg, state.pitch_deg
        t.attitude_received = time.monotonic()

    def _mode(self, mode):
        self.telemetry.mode = mode.name
        self.telemetry.mode_received = time.monotonic()

    async def connect(self):
        verify_sitl(self.root, self.pid)
        # PX4 stream rates use simulation seconds, MAVSDK disconnect deadlines use
        # wall seconds. At 0.25x a default 1Hz heartbeat arrives every 4 wall seconds
        # and repeatedly trips MAVSDK's 3s connection timer, resetting home health.
        await asyncio.to_thread(subprocess.run,
            [str(self.root/'build/px4_sitl_default/bin/px4-mavlink'),
             'stream','-u','14580','-s','HEARTBEAT','-r','5'],
            check=True,timeout=5,capture_output=True)
        self.lifecycle.append(dict(event='sitl_heartbeat_5hz',monotonic_time_s=time.monotonic()))
        from mavsdk import System
        self.drone = System()
        # No configurable remote host, serial URL, or physical vehicle option.
        await self.drone.connect(system_address='udpin://127.0.0.1:14540')
        # Pinned MAVSDK 3.17.4 owns this child. Never kill an unrelated process.
        self.server_process = self.drone._server_process
        if not isinstance(self.server_process, subprocess.Popen):
            raise RuntimeError('expected adapter-owned MAVSDK server process')
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True,
                                                name='sitl-command-watchdog')
        self.watchdog_thread.start()
        streams = [(self.drone.core.connection_state(), self._connection),
                   (self.drone.telemetry.position_velocity_ned(), self._position),
                   (self.drone.telemetry.attitude_euler(), self._attitude),
                   (self.drone.telemetry.flight_mode(), self._mode),
                   (self.drone.telemetry.in_air(), lambda x: setattr(self.telemetry, 'in_air', x))]
        self.tasks = [asyncio.create_task(self._consume(s, f)) for s, f in streams]
        async with asyncio.timeout(20):
            while not self.telemetry.connected or self.telemetry.received is None:
                await asyncio.sleep(0.1)
        await self.drone.telemetry.set_rate_position_velocity_ned(20.0)
        await self.drone.telemetry.set_rate_attitude_euler(20.0)
        # HOME_POSITION may have been broadcast before this SDK connected.
        await self.drone.telemetry.set_rate_home(1.0)

    async def takeoff(self, altitude=1.5):
        verify_sitl(self.root, self.pid)
        health = None
        try:
            async with asyncio.timeout(60):
                async for health in self.drone.telemetry.health():
                    if health.is_global_position_ok and health.is_home_position_ok:
                        break
        except TimeoutError as exc:
            raise TimeoutError(f'PX4 health gate timed out: {health}') from exc
        # Verified against v1.16.0 commander_params.c: action 4 is controlled Land.
        await self.drone.param.set_param_float('COM_OF_LOSS_T', .5)
        await self.drone.param.set_param_int('COM_OBL_RC_ACT', 4)
        if (abs(await self.drone.param.get_param_float('COM_OF_LOSS_T')-.5)>1e-5
                or await self.drone.param.get_param_int('COM_OBL_RC_ACT')!=4):
            raise RuntimeError('PX4 offboard-loss landing parameters were not confirmed')
        await self.drone.action.set_takeoff_altitude(altitude)
        if abs(await self.drone.param.get_param_float('MIS_TAKEOFF_ALT')-altitude)>.01:
            raise RuntimeError('takeoff altitude parameter not confirmed')
        # Navigator consumes parameter updates asynchronously in simulation time.
        # A parameter ACK alone does not mean Navigator has refreshed its cache.
        if self.simulation_snapshot is None:
            raise RuntimeError('simulation clock is required before takeoff')
        start=self.simulation_snapshot()['sim_time']
        async with asyncio.timeout(15):
            while self.simulation_snapshot()['sim_time']-start<1.2:
                await asyncio.sleep(.05)
        self.lifecycle.append(dict(event='arming_requested',monotonic_time_s=time.monotonic()))
        self.armed = True  # cleanup is required even if cancellation races with the arm ACK
        await self.drone.action.arm()
        await self.drone.action.takeoff()
        self.lifecycle.append(dict(event='takeoff_acknowledged',monotonic_time_s=time.monotonic()))
        stable_since = None
        takeoff_start = self.simulation_snapshot()['sim_time']
        async with asyncio.timeout(180):
            while True:
                t, now = self.telemetry, time.monotonic()
                sim_now = self.simulation_snapshot()['sim_time']
                if sim_now-takeoff_start>35:
                    raise TimeoutError(f'takeoff/hover did not stabilize: altitude={-t.down:.3f} speed={t.speed:.3f}')
                if (t.connected and t.received is not None and now-t.received < .5
                        and t.in_air and abs(-t.down-altitude) < .3 and t.speed < .15):
                    stable_since = sim_now if stable_since is None else stable_since
                    if sim_now-stable_since > 2:
                        break
                else:
                    stable_since = None
                await asyncio.sleep(.1)
        self.lifecycle.append(dict(event='hover_stable',monotonic_time_s=time.monotonic()))
        prestream_start = self.simulation_snapshot()['sim_time']
        async with asyncio.timeout(15):
            while self.simulation_snapshot()['sim_time']-prestream_start<1.25:
                await self.send(Velocity())
                await asyncio.sleep(.05)
        await self._enter_offboard()

    async def _enter_offboard(self):
        await self.drone.offboard.start()
        self.offboard = True
        self.lifecycle.append(dict(event='offboard_started',monotonic_time_s=time.monotonic()))
        async with asyncio.timeout(3):
            while self.telemetry.mode != 'OFFBOARD':
                await self.send(Velocity())
                await asyncio.sleep(.05)

    async def send(self, velocity):
        from mavsdk.offboard import VelocityBodyYawspeed
        try:
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(*velocity.values()))
            self.last_send = time.monotonic()
        except Exception as exc:
            self.failure = f'setpoint failed: {exc}'
            raise

    def _watchdog_step(self, now):
        if self.offboard and (self.last_send is None or not 0<=now-self.last_send<=.3):
            self.failure = 'independent command watchdog timeout; SDK stream terminated for PX4 landing failsafe'
            self.offboard = False
            if self.server_process is not None and self.server_process.poll() is None:
                self.server_process.kill()

    def _watchdog_loop(self):
        while not self.watchdog_stop.wait(.05):
            self._watchdog_step(time.monotonic())

    async def land(self):
        """Never disarm airborne. PX4's own offboard-loss action remains essential."""
        if self.drone is None or not self.armed:
            return
        with suppress(Exception):
            await asyncio.wait_for(self.send(Velocity()), 2)
        self.offboard = False  # landing action owns control; stop Offboard watchdog
        # Land action can switch out of Offboard even when offboard.stop fails.
        await asyncio.wait_for(self.drone.action.land(), 5)
        self.offboard = False
        async with asyncio.timeout(45):
            async for state in self.drone.telemetry.landed_state():
                if state.name == 'ON_GROUND':
                    await self.drone.action.disarm()
                    self.armed = False
                    return

    async def close(self):
        self.watchdog_stop.set()
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=.2)
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.server_process is not None and self.server_process.poll() is None:
            self.server_process.terminate()
            try:
                await asyncio.to_thread(self.server_process.wait, timeout=2)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        self.drone = None
