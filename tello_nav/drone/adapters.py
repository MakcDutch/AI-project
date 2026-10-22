import math
import threading
import time
from typing import Callable

from tello_nav.controller import ControlCommand


class MockDroneAdapter:
    """No import of DJITelloPy, sockets, or hardware references."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 command_limit: int = 20, freshness_s: float = 0.3):
        ControlCommand().validate(command_limit)
        if not math.isfinite(freshness_s) or freshness_s <= 0:
            raise ValueError("freshness_s must be positive and finite")
        self.clock, self.command_limit, self.freshness_s = clock, command_limit, freshness_s
        self.connected = False
        self.inhibited = False
        self.commands: list[ControlCommand] = []
        self.last_received: float | None = None
        self.last_observation: float | None = None
        self.last_error: str | None = None
        self.lock = threading.RLock()

    def connect(self) -> None:
        with self.lock:
            self.connected = True
            self.last_received = self.clock()

    def _emit(self, command: ControlCommand) -> None:
        self.commands.append(command)

    def _halt(self, reason: str) -> None:
        self.inhibited = True
        stop = ControlCommand(reason=reason)
        try:
            self._emit(stop)
        except Exception as exc:
            self.last_error = f"zero-command transmission failed: {exc}"

    def send(self, command: ControlCommand, *, timestamp: float,
             observation_timestamp: float) -> bool:
        """Both times are monotonic seconds in this adapter's clock epoch.

        Caller must preserve the original observation timestamp: constructing a
        new command must never make an old depth observation fresh again.
        """
        with self.lock:
            if not self.connected or self.inhibited:
                return False
            now = self.clock()
            try:
                command.validate(self.command_limit)
                if (not math.isfinite(now) or any(not math.isfinite(t) or not 0 <= now - t <= self.freshness_s
                        for t in (timestamp, observation_timestamp)) or observation_timestamp > timestamp):
                    raise ValueError("stale, future or invalid command/observation timestamp")
                self._emit(command)
                self.last_received = now
                self.last_observation = observation_timestamp
                return True
            except Exception as exc:
                self.last_error = str(exc)
                self._halt(f"command rejected: {exc}")
                return False

    def watchdog(self) -> None:
        with self.lock:
            if not self.connected:
                return
            now = self.clock()
            times = [t for t in (self.last_received, self.last_observation) if t is not None]
            if not math.isfinite(now) or any(not 0 <= now - t <= self.freshness_s for t in times):
                self._halt("watchdog timeout; operator reset required")
            elif self.inhibited:
                self._halt("operator inhibition active")

    def emergency_stop(self) -> None:
        """Latch zero RC translation/yaw; this is NOT motor cutoff or landing."""
        with self.lock:
            if self.connected:
                self._halt("operator emergency STOP")
            else:
                self.inhibited = True

    def manual_override(self) -> None:
        """Stop and revoke this adapter's motion authority for external piloting."""
        self.emergency_stop()

    def reset_operator_stop(self, *, operator_authorized: bool = False) -> None:
        with self.lock:
            if operator_authorized is not True or not self.connected:
                raise PermissionError("connected operator must explicitly reset inhibition")
            self.inhibited = False
            self.last_received = self.clock()
            self.last_observation = None

    def telemetry(self) -> dict:
        return {"timestamp": self.clock(), "reliable": False, "source": "mock"}

    def close(self) -> None:
        with self.lock:
            if self.connected:
                self._halt("adapter closing")
            self.connected = False


class TelloAdapter(MockDroneAdapter):
    """Explicitly gated optional SDK boundary. No takeoff/land/motor-cutoff API.

    SDK UDP RC sends are unacknowledged: a successful call does not prove packet
    delivery. The watchdog is a local best-effort zero-command mechanism.
    """

    def __init__(self, *, mode: str, operator_authorized: bool = False,
                 sdk_factory: Callable | None = None, **kwargs):
        if mode != "live" or operator_authorized is not True:
            raise PermissionError("live mode AND explicit operator authorization are required")
        super().__init__(**kwargs)
        self.sdk_factory = sdk_factory
        self.sdk = None
        self.streaming = False
        self.shutdown = threading.Event()
        self.worker: threading.Thread | None = None

    def connect(self) -> None:
        if self.connected:
            raise RuntimeError("adapter already connected")
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("previous watchdog is still running")
        try:
            factory = self.sdk_factory
            if factory is None:
                from djitellopy import Tello
                factory = Tello
            self.sdk = factory()
            self.sdk.connect()
            super().connect()
            self.shutdown.clear()
            self.worker = threading.Thread(target=self._watch_loop, name="tello-watchdog", daemon=True)
            self.worker.start()
        except Exception as exc:
            self.connected = False
            self.inhibited = True
            self.last_error = str(exc)
            raise ConnectionError("Tello connection failed") from exc

    def _watch_loop(self) -> None:
        while not self.shutdown.wait(min(0.05, self.freshness_s / 3)):
            self.watchdog()

    def _emit(self, command: ControlCommand) -> None:
        if self.sdk is None:
            raise ConnectionError("SDK is not connected")
        self.sdk.send_rc_control(*command.channels)
        super()._emit(command)

    def start_video(self) -> None:
        if not self.connected:
            raise ConnectionError("connect first")
        try:
            self.sdk.streamon()
            self.streaming = True
        except Exception:
            self.emergency_stop()
            raise

    def video_frame(self):
        """Return (SDK image copy, receive time); not a sensor capture timestamp.

        DJITelloPy's cached frame can be repeated or old. This timestamp MUST NOT
        be used as capture freshness for autonomous flight without a separately
        validated acquisition pipeline. No autonomous live runner is provided.
        """
        if not self.streaming:
            raise RuntimeError("start video first")
        try:
            frame = self.sdk.get_frame_read().frame
            if frame is None:
                raise ValueError("no video frame available")
            return frame.copy(), self.clock()
        except Exception:
            self.emergency_stop()
            raise

    def telemetry(self) -> dict:
        if not self.connected:
            raise ConnectionError("not connected")
        try:
            return {"raw_tello": dict(self.sdk.get_current_state()), "timestamp": self.clock(),
                    "reliable": False, "timestamp_source": "local_read_not_sensor_time"}
        except Exception:
            self.emergency_stop()
            raise

    def close(self) -> None:
        self.shutdown.set()
        if self.worker is not None:
            self.worker.join(timeout=1.0)
        super().close()
        if self.streaming:
            try:
                self.sdk.streamoff()
            except Exception as exc:
                self.last_error = f"stream cleanup failed: {exc}"
            self.streaming = False
        # Do not call sdk.end(): SDK cleanup may land an aircraft marked flying.
        # SDK socket lifetime remains owned by DJITelloPy/process; see docs.


def create_adapter(mode: str = "mock", *, operator_authorized: bool = False, **kwargs):
    if mode == "mock":
        return MockDroneAdapter(**kwargs)
    if mode == "live":
        return TelloAdapter(mode=mode, operator_authorized=operator_authorized, **kwargs)
    raise ValueError("adapter mode must be mock or live")
