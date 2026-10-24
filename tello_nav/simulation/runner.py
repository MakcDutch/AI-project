"""Real Gazebo/PX4 experiment coordinator; no fake backend or hardware path."""
import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import time

from tello_nav.controller import ControllerConfig, ReactiveController, ControlCommand
from tello_nav.controller.sectors import analyze, SectorConfig
from tello_nav.replay.logging import plain, write_results
from .gazebo_camera import GazeboCamera
from .px4_adapter import PX4Adapter
from .safety_monitor import Limits, SafetyMonitor, Velocity, map_command


async def wait_ready(camera, timeout=25):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if camera.topic is None:
            try:
                camera.discover()
            except RuntimeError:
                pass
        s = camera.snapshot()
        if (s['frame'] and s['pose'] and s['sim_time'] is not None
                and '/wall_contacts' in camera.node.topic_list()):
            if s['error']:
                raise RuntimeError(s['error'])
            if not all(v.known for v in analyze(s['frame'].depth, SectorConfig()).values()):
                raise RuntimeError('depth stream has insufficient valid coverage before arming')
            if not 0 <= s['sim_time']-s['frame'].timestamp < .5:
                raise RuntimeError('depth and world clock do not share a fresh simulation epoch')
            return s
        await asyncio.sleep(.1)
    raise TimeoutError(f'Gazebo depth/clock/X500 pose/wall contact not ready: {camera.snapshot()["error"]}')


def summarize(records, experiment, error=None):
    flying = [r for r in records if r.get('position') and
              all(isinstance(v,(int,float)) and math.isfinite(v) for v in
                  (r.get('actual_speed_m_s'), r.get('simulation_time_s'),r.get('travel_m'),
                   r.get('wall_clearance_m'),*(r['position'].get(k) for k in ('x','y','z'))))]
    data_gaps = len(flying)!=len(records)
    moving = next((r for r in flying if r['controller_state']=='FORWARD' and r['actual_speed_m_s']>.1), None)
    detection = next((r for r in flying if moving and r['simulation_time_s']>moving['simulation_time_s']
                      and (r['controller_state']=='BRAKE'
                           or (r['safety_event'] and experiment=='stale'))), None)
    stopped = None
    if detection:
        # Require a continuous second below 8cm/s, rather than one noisy sample.
        since = None
        for r in flying:
            if r['simulation_time_s'] < detection['simulation_time_s']:
                continue
            if r['actual_speed_m_s'] < .08:
                since = r if since is None else since
                if r['simulation_time_s']-since['simulation_time_s'] >= 1:
                    stopped = since
                    break
            else:
                since = None
    collisions = any(r['collision'] for r in records)
    contact_verified = bool(records) and all(r['contact_monitor_available'] for r in records)
    movement = max((r['travel_m'] for r in flying), default=0)
    peak = max((r['actual_speed_m_s'] for r in flying), default=0)
    min_clearance = min((r['wall_clearance_m'] for r in flying), default=None)
    safety = sorted({r['safety_event'] for r in records if r['safety_event']})
    depths = [r['center_depth_m'] for r in flying if r['center_depth_m'] is not None]
    decreasing_depth = len(depths)>2 and depths[0]-min(depths)>.5
    common = error is None and not data_gaps and not collisions and contact_verified and peak>.1
    if experiment == 'clear':
        success = common and movement>.75 and not safety and any(r['controller_state']=='FORWARD' for r in flying)
    elif experiment == 'wall':
        success = (common and decreasing_depth and detection is not None and stopped is not None
                   and min_clearance>.65 and not safety)
    else:
        success = (common and stopped is not None and any(r['depth_interrupted'] for r in flying)
                   and any('timeout' in event for event in safety))
    return dict(experiment=experiment, success=success, error=error, measurement_gaps=data_gaps,
                collision=collisions if contact_verified else None,
                contact_monitor_verified=contact_verified, minimum_clearance_m=min_clearance,
                actual_stopping_distance_m=(stopping_path(flying, detection, stopped)
                                            if stopped and detection else None),
                detection_to_stop_s=(stopped['simulation_time_s']-detection['simulation_time_s']
                                     if stopped and detection else None),
                peak_velocity_m_s=peak, travel_m=movement, depth_decreased=decreasing_depth,
                safety_events=safety, final_controller_state=flying[-1]['controller_state'] if flying else None,
                final_flight_mode=flying[-1]['flight_mode'] if flying else None)


def stopping_path(rows, detection, stopped):
    segment = [r for r in rows if detection['simulation_time_s']<=r['simulation_time_s']<=stopped['simulation_time_s']]
    return sum(math.dist(tuple(a['position'][k] for k in ('x','y')),
                         tuple(b['position'][k] for k in ('x','y')))
               for a,b in zip(segment,segment[1:]))


def finite_json(value):
    if isinstance(value, dict):
        return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [finite_json(v) for v in value]
    return None if isinstance(value,float) and not math.isfinite(value) else value


async def guarded(operation, emergency):
    task = asyncio.create_task(operation)
    stop = asyncio.create_task(emergency.wait())
    try:
        done, _ = await asyncio.wait((task,stop),return_when=asyncio.FIRST_COMPLETED)
        if stop in done:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
            raise RuntimeError('manual emergency stop during initialization')
        return await task
    finally:
        stop.cancel()
        await asyncio.gather(stop,return_exceptions=True)


async def run(args):
    config = json.loads(Path(args.config).read_text())
    cfg = ControllerConfig(**config['controller'])
    limits = Limits(**config['limits'])
    if cfg.command_limit != limits.rc_limit:
        raise ValueError('controller and mapping RC limits must match')
    adapter = PX4Adapter(root=args.px4_root, pid=args.px4_pid, simulation_only=args.simulation_only)
    camera = GazeboCamera()
    adapter.simulation_snapshot = camera.snapshot
    monitor = SafetyMonitor(limits)
    emergency = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, emergency.set)
    records, events = [], []
    startup_evidence = None
    error = None
    now_sim = 0.0
    controller = ReactiveController(cfg, clock=lambda: now_sim)
    command, desired = ControlCommand(), Velocity()
    command_at, last_index, interrupted = None, None, False
    contact_available = False
    graph_checked = 0.0
    start_pose = None
    injection_at = None
    terminal_at = None
    last_frame = None
    try:
        initial = await wait_ready(camera)
        # This world/pose check precedes all arm/takeoff operations.
        p = initial['pose']
        if math.hypot(p['x'], p['y'])>.3 or abs(p['z'])>.5:
            raise RuntimeError('restart simulation: X500 must be at the documented origin on the ground')
        await guarded(adapter.connect(), emergency)
        if args.smoke:
            events.append('actual Gazebo depth + clock + model pose + MAVSDK telemetry received; no arming')
            startup_evidence = dict(depth_shape=list(initial['frame'].depth.shape),
                frame_timestamp_s=initial['frame'].timestamp,simulation_time_s=initial['sim_time'],
                gazebo_pose=initial['pose'],telemetry=finite_json(asdict(adapter.telemetry)))
        else:
            await guarded(asyncio.wait_for(adapter.takeoff(), 300), emergency)
            initial = await wait_ready(camera)
            start_pose = initial['pose']
            # Yaw=0 ENU means east-facing, i.e. yaw=90 degrees in PX4 NED.
            if abs((adapter.telemetry.yaw_deg-90+180)%360-180)>10:
                raise RuntimeError('vehicle does not face world +X/east; restart simulation')
            start_sim = initial['sim_time']
            deadline = time.monotonic()+120
            while time.monotonic()<deadline:
                s, now = camera.snapshot(), time.monotonic()
                now_sim = s['sim_time'] if s['sim_time'] is not None else now_sim
                elapsed = now_sim-start_sim
                t = adapter.telemetry
                frame = s['frame']
                if args.experiment == 'stale' and elapsed>=4 and t.speed>.1:
                    if not interrupted:
                        injection_at = now_sim
                        events.append('depth consumption interrupted while camera publishing continues')
                    interrupted = True
                if not interrupted:
                    last_frame = frame
                frame = last_frame
                if frame and frame.index != last_index:
                    command = controller.compute(frame.depth, frame.timestamp,
                        {'reliable': t.sim_received is not None, 'speed_mps': t.speed,
                         'timestamp': t.sim_received})
                    desired = map_command(command, limits)
                    command_at = now
                    last_index = frame.index
                if s['error']:
                    monitor.stop('invalid depth stream: '+s['error'])
                sim_age = now_sim-frame.timestamp if frame else math.inf
                if not 0<=sim_age<=limits.sensor_timeout_s:
                    monitor.stop('depth simulation timestamp timeout or future frame')
                required_stamps = (s['clock_received'], s['pose_received'], t.attitude_received)
                if any(stamp is None or not 0<=now-stamp<=limits.telemetry_timeout_s for stamp in required_stamps):
                    monitor.stop('world clock, pose, attitude or mode timeout')
                if t.mode != 'OFFBOARD':
                    monitor.stop('offboard mode lost')
                if adapter.failure:
                    monitor.stop(adapter.failure)
                if not all(math.isfinite(v) for v in (t.vn,t.ve,t.vd,t.north,t.east,t.down,t.yaw_deg)):
                    monitor.stop('nonfinite vehicle telemetry')
                elif math.hypot(t.vn,t.ve)>limits.horizontal_m_s+.2:
                    monitor.stop('measured horizontal speed exceeds envelope')
                p = s['pose'] or start_pose
                travel = math.hypot(p['x']-start_pose['x'], p['y']-start_pose['y'])
                clearance = 6.0-p['x']-.5  # conservative body radius, not camera depth
                applied, event = monitor.check(desired, now=now,
                    sensor_received=frame.received if frame else None,
                    telemetry_received=t.received, command_received=command_at,
                    connected=t.connected, altitude=p['z'], travel=travel, clearance=clearance,
                    collision=s['collision'], emergency=emergency.is_set())
                if event:
                    command = controller.compute(None, now_sim)
                    desired = Velocity()
                await asyncio.wait_for(adapter.send(applied), .25)
                if now-graph_checked>1:
                    pubs, _ = camera.node.topic_info('/wall_contacts')
                    contact_available = bool(pubs) and all(p.msg_type_name=='gz.msgs.Contacts' for p in pubs)
                    graph_checked = now
                if not contact_available:
                    monitor.stop('wall contact publisher unavailable')
                    await asyncio.wait_for(adapter.send(Velocity()),.25)
                    applied, event = Velocity(), monitor.reason
                center = controller.sectors.get('center')
                records.append(finite_json(dict(simulation_time_s=now_sim, elapsed_s=elapsed,
                    monotonic_time_s=now, frame_index=frame.index if frame else None,
                    frame_timestamp_s=frame.timestamp if frame else None,
                    sensor_sim_latency_s=sim_age if math.isfinite(sim_age) else None,
                    sensor_receive_age_s=now-frame.received if frame else None,
                    sector_distances=plain(controller.sectors), center_depth_m=center.distance_m if center else None,
                    controller_state=command.state, controller_reason=command.reason,
                    desired_velocity=asdict(desired), applied_velocity=asdict(applied),
                    actual_velocity_ned={'north':t.vn,'east':t.ve,'down':t.vd},
                    actual_forward_m_s=t.forward_speed, actual_speed_m_s=t.speed,
                    telemetry_receive_time_s=t.received, telemetry_receive_sim_time_s=t.sim_received,
                    safe_distance_m=controller.safe_distance_m, position=p,
                    orientation_deg={'roll':t.roll_deg,'pitch':t.pitch_deg,'yaw':t.yaw_deg},
                    flight_mode=t.mode, wall_clearance_m=clearance, travel_m=travel,
                    safety_event=event, collision=s['collision'], contact_monitor_available=contact_available,
                    depth_interrupted=interrupted, depth_interruption_time_s=injection_at)))
                if event or command.state=='BRAKE':
                    terminal_at = terminal_at if terminal_at is not None else now_sim
                if terminal_at is not None and now_sim-terminal_at>4:
                    break
                if args.experiment=='clear' and travel>=1.0:
                    break
                if elapsed>=args.duration or emergency.is_set():
                    break
                await asyncio.sleep(.05)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        monitor.stop(error)
    finally:
        try:
            await adapter.land()
            events.append('landing confirmed and disarmed' if not adapter.armed else 'landing not confirmed')
        except Exception as exc:
            error = (error+'; ' if error else '')+f'landing failed: {exc}'
            events.append('landing failed; operator must inspect SITL console')
        await adapter.close()
        final_camera = camera.snapshot()
        camera.close()
        summary = summarize(records, args.experiment, error)
        if final_camera['collision']:
            summary['collision'] = True
            summary['success'] = False
        summary['final_vehicle_state'] = ('landed_disarmed' if (records or any(t['in_air'] for t in adapter.telemetry_history)) and not adapter.armed
                                          else 'landing_unconfirmed' if adapter.armed else 'not_armed')
        if args.smoke:
            summary = dict(smoke=True, success=error is None, error=error, armed=False)
        write_results(args.output, records, dict(source='actual Gazebo/PX4 only',
            experiment=args.experiment, depth_topic=camera.topic, controller=plain(cfg),
            limits=plain(limits), events=events, startup_evidence=startup_evidence, collision_events=final_camera['collision_events'],
            telemetry_history=finite_json(adapter.telemetry_history),lifecycle=adapter.lifecycle,summary=summary))
        print(json.dumps(summary, indent=2))
    return 0 if summary['success'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation-only', action='store_true', required=True)
    parser.add_argument('--px4-root', required=True)
    parser.add_argument('--px4-pid', required=True, type=int)
    parser.add_argument('--experiment', choices=['clear','wall','stale'], default='wall')
    parser.add_argument('--duration', type=float, default=25)
    parser.add_argument('--config', default='simulation/configs/x500.json')
    parser.add_argument('--smoke', action='store_true', help='read actual sensor/telemetry, never arm')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or not 1<=args.duration<=60:
        parser.error('duration must be in [1,60] simulation seconds')
    if Path(args.output).suffix not in {'.json','.csv'}:
        parser.error('output must end in .json or .csv')
    try:
        return asyncio.run(run(args))
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f'simulation startup rejected: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
