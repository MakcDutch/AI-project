"""SI command mapping and an independent, latched simulation supervisor."""
from dataclasses import dataclass, asdict
import math

from tello_nav.controller import ControlCommand


@dataclass(frozen=True)
class Velocity:
    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    down_m_s: float = 0.0
    yaw_deg_s: float = 0.0

    def values(self):
        return tuple(asdict(self).values())


def enu_to_ned(x, y, z):
    return y, x, -z


def ned_to_enu(north, east, down):
    return east, north, -down


@dataclass(frozen=True)
class Limits:
    rc_limit: int = 20
    horizontal_m_s: float = 0.6
    vertical_m_s: float = 0.3
    yaw_deg_s: float = 15.0
    altitude_m: float = 3.0
    travel_m: float = 8.0
    min_clearance_m: float = 0.65
    sensor_timeout_s: float = 0.5
    telemetry_timeout_s: float = 0.5
    command_timeout_s: float = 0.3

    def __post_init__(self):
        ControlCommand().validate(self.rc_limit)
        if self.rc_limit == 0 or any(not math.isfinite(v) or v <= 0
                                   for k, v in asdict(self).items() if k != 'rc_limit'):
            raise ValueError('limits must be positive and finite')


def map_command(command, limits=Limits()):
    command.validate(limits.rc_limit)
    if command.state in {'STOP', 'BRAKE'}:
        return Velocity()
    forward = command.forward_backward / limits.rc_limit * limits.horizontal_m_s
    right = command.left_right / limits.rc_limit * limits.horizontal_m_s
    norm = math.hypot(forward, right)
    scale = min(1.0, limits.horizontal_m_s / norm) if norm else 1.0
    return Velocity(forward * scale, right * scale,
                    -command.up_down / limits.rc_limit * limits.vertical_m_s,
                    command.yaw / limits.rc_limit * limits.yaw_deg_s)


class SafetyMonitor:
    def __init__(self, limits=Limits()):
        self.limits = limits
        self.reason = None

    def stop(self, reason):
        self.reason = self.reason or reason
        return Velocity()

    def check(self, desired, *, now, sensor_received, telemetry_received,
              command_received, connected, altitude, travel, clearance,
              collision=False, emergency=False):
        if self.reason:
            return Velocity(), self.reason
        cfg = self.limits
        problem = None
        if emergency:
            problem = 'manual emergency stop'
        elif not connected:
            problem = 'connection lost'
        elif collision:
            problem = 'wall contact'
        elif any(not math.isfinite(v) for v in (now, altitude, travel, clearance, *desired.values())):
            problem = 'invalid safety input'
        elif any(stamp is None or not math.isfinite(stamp) or not 0 <= now-stamp <= timeout
                 for stamp, timeout in ((sensor_received, cfg.sensor_timeout_s),
                                        (telemetry_received, cfg.telemetry_timeout_s),
                                        (command_received, cfg.command_timeout_s))):
            problem = 'sensor, telemetry or command timeout'
        elif altitude < 0.3 or altitude > cfg.altitude_m:
            problem = 'altitude envelope exceeded'
        elif travel > cfg.travel_m:
            problem = 'maximum approach travel exceeded'
        elif clearance <= cfg.min_clearance_m:
            problem = 'minimum wall clearance reached'
        elif (math.hypot(desired.forward_m_s, desired.right_m_s) > cfg.horizontal_m_s + 1e-9
              or abs(desired.down_m_s) > cfg.vertical_m_s
              or abs(desired.yaw_deg_s) > cfg.yaw_deg_s):
            problem = 'velocity envelope exceeded'
        if problem:
            self.stop(problem)
        return (Velocity(), self.reason) if self.reason else (desired, None)
