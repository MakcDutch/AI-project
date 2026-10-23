from dataclasses import dataclass
import math
import time
from typing import Callable, Iterable, Iterator

from tello_nav.controller import ControllerConfig, ReactiveController
from tello_nav.drone import MockDroneAdapter
from .logging import plain
from .providers import DepthProvider, validate_depth
from .video import Frame


@dataclass(frozen=True)
class ReplayConfig:
    mode: str = "offline"  # offline, realtime, virtual
    inference_delay_s: float = 0.0
    communication_delay_s: float = 0.0
    late_policy: str = "process"  # process stale frames or drop after max_lateness_s
    max_lateness_s: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in {"offline", "realtime", "virtual"}:
            raise ValueError("invalid replay mode")
        if self.late_policy not in {"process", "drop"}:
            raise ValueError("invalid late policy")
        if any(not math.isfinite(v) or v < 0 for v in (
            self.inference_delay_s, self.communication_delay_s, self.max_lateness_s
        )):
            raise ValueError("delays/lateness must be nonnegative and finite")


class ReplayRunner:
    """Single consumer, sequential inference; each iter_records call resets state.

    offline: timeline follows capture times; measured runtime is logged but does
    not affect decisions. virtual: queue and artificial delays use virtual time.
    realtime: wall clock pacing, actual work + artificial delays affect age.
    In all modes communication delay is applied *before* computing the command,
    so stale observations cannot generate a fresh motion command on delivery.
    """

    def __init__(self, provider: DepthProvider, controller_config: ControllerConfig | None = None,
                 config: ReplayConfig | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, *, mock_drone: bool = False):
        self.provider = provider
        self.controller_config = controller_config or ControllerConfig()
        self.config = config or ReplayConfig()
        self.clock, self.sleep = clock, sleep
        self.mock_drone = mock_drone
        if self.config.mode == "offline" and (
            self.config.inference_delay_s or self.config.communication_delay_s
        ):
            raise ValueError("artificial delay requires virtual or realtime mode")

    def iter_records(self, frames: Iterable[Frame], scenario_id: str,
                     visualize: bool = False) -> Iterator[dict]:
        cfg = self.config
        decision_time = 0.0
        controller = ReactiveController(self.controller_config, clock=lambda: decision_time)
        drone = MockDroneAdapter(clock=lambda: decision_time,
                                 command_limit=self.controller_config.command_limit) if self.mock_drone else None
        if drone:
            drone.connect()
        wall_start = self.clock()
        first_timestamp = None
        previous_timestamp = None
        previous_index = -1
        ready_at = 0.0
        viewer = None
        if visualize:
            from .visualization import ReplayViewer
            viewer = ReplayViewer(self.controller_config.sectors)
        try:
            for frame in frames:
                if (not math.isfinite(frame.timestamp) or frame.timestamp < 0
                        or frame.index <= previous_index or frame.index < 0
                        or (previous_timestamp is not None and frame.timestamp <= previous_timestamp)):
                    raise ValueError("frames must have increasing nonnegative timestamps and indices")
                previous_timestamp, previous_index = frame.timestamp, frame.index
                if first_timestamp is None:
                    first_timestamp = frame.timestamp
                arrival = frame.timestamp - first_timestamp
                if cfg.mode == "realtime":
                    wait = arrival - (self.clock() - wall_start)
                    if wait > 0:
                        self.sleep(wait)
                    start = max(arrival, self.clock() - wall_start)
                elif cfg.mode == "virtual":
                    start = max(arrival, ready_at)
                else:
                    start = arrival
                dropped = cfg.late_policy == "drop" and start - arrival > cfg.max_lateness_s
                depth, error, measured = None, None, 0.0
                if dropped:
                    error = "late frame dropped; safety STOP"
                else:
                    before = self.clock()
                    try:
                        depth = validate_depth(self.provider.predict(frame), frame)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    measured = max(0.0, self.clock() - before)
                simulated = 0.0 if dropped else cfg.inference_delay_s + cfg.communication_delay_s
                if cfg.mode == "realtime":
                    if simulated:
                        self.sleep(simulated)
                    decision_time = max(start, self.clock() - wall_start)
                else:
                    decision_time = start + simulated
                ready_at = decision_time
                command = controller.compute(depth, arrival)
                mock_execution = None
                if drone:
                    drone.watchdog()
                    accepted = drone.send(command, timestamp=decision_time, observation_timestamp=arrival)
                    mock_execution = {"accepted": accepted, "inhibited": drone.inhibited,
                                      "error": drone.last_error,
                                      "effective_command": plain(drone.commands[-1]) if drone.commands else None}
                record = {
                    "scenario_id": scenario_id, "frame_index": frame.index,
                    "frame_timestamp_s": frame.timestamp, "timestamp_source": frame.timestamp_source,
                    "depth_model": self.provider.name, "sectors": plain(controller.sectors),
                    "safe_distance_m": controller.safe_distance_m,
                    "controller_state": command.state, "command": plain(command),
                    "reason": command.reason, "measured_inference_s": measured,
                    "simulated_inference_s": 0.0 if dropped else cfg.inference_delay_s,
                    "simulated_communication_s": 0.0 if dropped else cfg.communication_delay_s,
                    "frame_age_s": decision_time - arrival, "decision_time_s": decision_time,
                    "dropped": dropped, "error": error,
                    "safety_event": command.reason if command.state in {"STOP", "BRAKE"} else None,
                }
                if drone:
                    record["mock_execution"] = mock_execution
                frame_metadata = getattr(self.provider, "frame_metadata", None)
                if callable(frame_metadata):
                    record["prediction"] = frame_metadata(frame)
                if viewer and not viewer.show(frame, depth, record):
                    yield record
                    break
                yield record
        finally:
            if drone:
                drone.close()
            if viewer:
                viewer.close()

    def run(self, frames: Iterable[Frame], scenario_id: str, visualize: bool = False) -> list[dict]:
        return list(self.iter_records(frames, scenario_id, visualize))

    def metadata(self) -> dict:
        metadata = {"schema_version": 1, "provider": self.provider.name,
                      "controller_config": self.controller_config, "replay_config": self.config,
                      "mock_drone": self.mock_drone,
                      "determinism": "offline/virtual decisions deterministic; measured runtime is observational"}
        provider_metadata = getattr(self.provider, "metadata", None)
        if callable(provider_metadata):
            metadata["provider_metadata"] = provider_metadata()
        return plain(metadata)
