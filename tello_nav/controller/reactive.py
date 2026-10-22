from dataclasses import dataclass, field
import math
import time
from typing import Callable, Mapping

import numpy as np

from .commands import ControlCommand
from .sectors import SectorConfig, SectorStats, analyze


@dataclass(frozen=True)
class ControllerConfig:
    min_distance_m: float = 1.0
    hysteresis_m: float = 0.25
    max_frame_age_s: float = 0.5
    brake_duration_s: float = 0.3
    avoid_hold_s: float = 0.5
    recovery_frames: int = 2
    forward_speed: int = 15
    lateral_speed: int = 10
    command_limit: int = 20
    # Lateral translation is opt-in: a forward camera cannot see side clearance.
    allow_lateral: bool = False
    braking_deceleration_mps2: float | None = None
    reaction_latency_s: float = 0.0
    telemetry_max_age_s: float = 0.25
    sectors: SectorConfig = field(default_factory=SectorConfig)

    def __post_init__(self) -> None:
        positive = (self.min_distance_m, self.max_frame_age_s, self.telemetry_max_age_s)
        nonnegative = (self.hysteresis_m, self.brake_duration_s, self.avoid_hold_s,
                       self.reaction_latency_s)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("distance and freshness bounds must be positive and finite")
        if any(not math.isfinite(v) or v < 0 for v in nonnegative):
            raise ValueError("durations and hysteresis must be nonnegative and finite")
        if self.braking_deceleration_mps2 is not None and (
            not math.isfinite(self.braking_deceleration_mps2) or self.braking_deceleration_mps2 <= 0
        ):
            raise ValueError("braking deceleration must be positive")
        if type(self.allow_lateral) is not bool:
            raise ValueError("allow_lateral must be an explicit boolean")
        if type(self.recovery_frames) is not int or self.recovery_frames < 1:
            raise ValueError("recovery_frames must be a positive integer")
        for value in (self.command_limit, self.forward_speed, self.lateral_speed):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 100:
                raise ValueError("command limits/speeds must be integers in [1, 100]")
        if max(self.forward_speed, self.lateral_speed) > self.command_limit:
            raise ValueError("configured speed exceeds command limit")


class ReactiveController:
    """One instance per sequence. Observation timestamps and clock share seconds/epoch.

    Fresh observations must have strictly increasing timestamps. STOP/BRAKE are
    zero RC requests, not evidence that the aircraft has physically stopped.
    """

    def __init__(self, config: ControllerConfig | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.config = config or ControllerConfig()
        self.clock = clock
        self.state = "STOP"
        self.entered_at = float("-inf")
        self.last_timestamp: float | None = None
        self.clear_frames = 0
        self.sectors: dict[str, SectorStats] = {}
        self.safe_distance_m = self.config.min_distance_m

    def _command(self, state: str, reason: str, now: float) -> ControlCommand:
        if state != self.state:
            self.entered_at = now
        self.state = state
        cfg = self.config
        cmd = ControlCommand(
            left_right=-cfg.lateral_speed if state == "AVOID_LEFT" else
            cfg.lateral_speed if state == "AVOID_RIGHT" else 0,
            forward_backward=cfg.forward_speed if state == "FORWARD" else 0,
            state=state, reason=reason,
        )
        cmd.validate(cfg.command_limit)
        return cmd

    def _stop(self, reason: str, now: float) -> ControlCommand:
        self.clear_frames = 0
        return self._command("STOP", reason, now)

    def _safe_distance(self, now: float, age: float, telemetry: Mapping | None) -> float:
        cfg = self.config
        distance = cfg.min_distance_m
        if cfg.braking_deceleration_mps2 is None or not telemetry:
            return distance
        # Only explicit SI, trusted, fresh telemetry is accepted; raw Tello vgx
        # is deliberately not used without a documented coordinate/unit adapter.
        try:
            speed = float(telemetry["speed_mps"])
            stamp = float(telemetry["timestamp"])
            if telemetry.get("reliable") is True and math.isfinite(speed) and speed >= 0 and (
                math.isfinite(stamp) and 0 <= now - stamp <= cfg.telemetry_max_age_s
            ):
                distance += speed * (age + cfg.reaction_latency_s) + speed**2 / (
                    2 * cfg.braking_deceleration_mps2)
        except OverflowError:
            return math.inf
        except (KeyError, TypeError, ValueError):
            pass
        return distance

    def compute(self, depth_map: np.ndarray | None, timestamp: float,
                telemetry: Mapping | None = None) -> ControlCommand:
        now = self.clock()
        self.sectors = {}
        self.safe_distance_m = self.config.min_distance_m
        if not math.isfinite(now):
            return self._stop("invalid controller clock", 0.0)
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            return self._stop("invalid observation timestamp", now)
        age = now - timestamp
        if age < 0 or age > self.config.max_frame_age_s:
            return self._stop("stale or future observation", now)
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return self._stop("non-increasing observation timestamp", now)
        self.last_timestamp = timestamp
        if depth_map is None:
            return self._stop("missing depth or inference failure", now)
        try:
            self.sectors = analyze(depth_map, self.config.sectors)
        except (TypeError, ValueError):
            return self._stop("invalid depth map", now)
        if not all(s.known for s in self.sectors.values()):
            return self._stop("insufficient valid sector coverage", now)
        cfg = self.config
        threshold = self._safe_distance(now, age, telemetry)
        if not math.isfinite(threshold):
            return self._stop("dynamic safety distance overflow", now)
        self.safe_distance_m = threshold
        center = self.sectors["center"]
        release = center.clear(threshold + cfg.hysteresis_m)
        blocked = not center.clear(threshold)
        if self.state == "FORWARD":
            if blocked:
                self.clear_frames = 0
                return self._command("BRAKE", "center obstacle; zero translation before avoidance", now)
            return self._command("FORWARD", "center remains clear", now)
        if release:
            self.clear_frames += 1
        else:
            self.clear_frames = 0
        if self.state in {"AVOID_LEFT", "AVOID_RIGHT"}:
            side = "left" if self.state == "AVOID_LEFT" else "right"
            if not self.sectors[side].clear(threshold + cfg.hysteresis_m):
                self.clear_frames = 0
                return self._command("BRAKE", "selected side became unsafe; brake before changing direction", now)
            if now - self.entered_at < cfg.avoid_hold_s:
                return self._command(self.state, "hold selected avoidance direction", now)
            if self.clear_frames >= cfg.recovery_frames:
                return self._command("BRAKE", "center recovered; brake lateral translation", now)
            return self._command(self.state, "continue selected avoidance direction", now)
        if self.state == "BRAKE" and now - self.entered_at < cfg.brake_duration_s:
            return self._command("BRAKE", "braking dwell", now)
        if self.clear_frames >= cfg.recovery_frames:
            return self._command("FORWARD", "center clear beyond hysteresis for recovery frames", now)
        if not blocked:
            return self._stop("waiting for stable center clearance", now) if not release else self._command(
                "STOP", "waiting for recovery frames", now)
        if self.state != "BRAKE":
            return self._command("BRAKE", "obstacle; brake before considering avoidance", now)
        if not cfg.allow_lateral:
            return self._stop("lateral avoidance disabled; side clearance is unobserved", now)
        choices = [name for name in ("left", "right")
                   if self.sectors[name].clear(threshold + cfg.hysteresis_m)]
        if not choices:
            return self._stop("all candidate directions blocked", now)
        side = max(choices, key=lambda name: self.sectors[name].distance_m)
        self.clear_frames = 0
        return self._command("AVOID_" + side.upper(), "clear side selected after braking dwell", now)
