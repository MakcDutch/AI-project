from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class ControlCommand:
    """Tello RC units [-100, 100]; positive LR is right, FB is forward.

    These are SDK setpoints, not measured velocities or SI units.
    """

    left_right: int = 0
    forward_backward: int = 0
    up_down: int = 0
    yaw: int = 0
    state: str = "STOP"
    reason: str = "safety stop"

    @property
    def channels(self) -> tuple[int, int, int, int]:
        return self.left_right, self.forward_backward, self.up_down, self.yaw

    def validate(self, limit: int = 100) -> None:
        if not isinstance(limit, Integral) or isinstance(limit, bool) or not 0 <= limit <= 100:
            raise ValueError("RC limit must be an integer in [0, 100]")
        if any(isinstance(v, bool) or not isinstance(v, Integral) or abs(v) > limit
               for v in self.channels):
            raise ValueError("invalid or out-of-bounds RC command")
        if self.state not in {"FORWARD", "BRAKE", "AVOID_LEFT", "AVOID_RIGHT", "STOP"}:
            raise ValueError("unknown controller state")
        if self.state in {"STOP", "BRAKE"} and any(self.channels):
            raise ValueError("STOP/BRAKE must command zero motion")
