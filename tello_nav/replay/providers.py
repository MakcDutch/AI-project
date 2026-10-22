from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .video import Frame


class DepthProvider(Protocol):
    """Return aligned HxW camera-axis Z depth in meters; invalid <=0/nonfinite.

    Implementations own preprocessing, intrinsics, crop restoration and metric
    scaling. A relative-depth model cannot satisfy this contract without an
    independently validated metric reconstruction.
    """

    name: str

    def predict(self, frame: Frame) -> np.ndarray: ...


def validate_depth(depth: np.ndarray, frame: Frame) -> np.ndarray:
    if (not isinstance(depth, np.ndarray) or depth.shape != frame.rgb.shape[:2]
            or depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number)
            or np.iscomplexobj(depth)):
        raise ValueError(f"frame {frame.index}: expected aligned real HxW metric depth")
    return depth


@dataclass
class CallableDepthProvider:
    """Bridge to a teammate's already metric, aligned public inference function.

    The callable receives Frame (RGB uint8, index, timestamp). This wrapper does
    no scaling, resizing, inference, or automatic model selection.
    """

    name: str
    infer: Callable[[Frame], np.ndarray]
    convention: str = "camera_z_m"

    def __post_init__(self) -> None:
        if self.convention != "camera_z_m":
            raise ValueError("adapter must explicitly produce camera_z_m")

    def predict(self, frame: Frame) -> np.ndarray:
        return validate_depth(self.infer(frame), frame)


class NpyDepthProvider:
    """Exact mapping: directory/000000.npy corresponds to decoded frame index 0."""

    def __init__(self, directory: str | Path, name: str = "precomputed"):
        self.directory, self.name = Path(directory), name
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)

    def predict(self, frame: Frame) -> np.ndarray:
        return validate_depth(np.load(self.directory / f"{frame.index:06d}.npy", allow_pickle=False), frame)


class MockDepthProvider:
    """Synthetic camera-Z meters; no learned inference."""

    name = "synthetic-mock"
    SCENARIOS = ("empty_corridor", "center_obstacle", "left_obstacle", "right_obstacle",
                 "multiple_obstacles", "narrow_passage", "reflective_surface", "invalid_depth")

    def __init__(self, scenario: str = "empty_corridor"):
        if scenario not in self.SCENARIOS:
            raise ValueError(f"unknown synthetic scenario: {scenario}")
        self.scenario = scenario

    def predict(self, frame: Frame) -> np.ndarray:
        h, w = frame.rgb.shape[:2]
        depth = np.full((h, w), 4.0, dtype=np.float32)
        a, b = w // 3, 2 * w // 3
        if self.scenario == "center_obstacle":
            depth[:, a:b] = 0.5
        elif self.scenario == "left_obstacle":
            depth[:, :a] = 0.5
        elif self.scenario == "right_obstacle":
            depth[:, b:] = 0.5
        elif self.scenario == "multiple_obstacles":
            depth[:] = 0.5
        elif self.scenario == "narrow_passage":
            depth[:, :a] = depth[:, b:] = 0.5
            depth[:, a:a + 1] = 0.5
            depth[:, b - 1:b] = 0.5
        elif self.scenario == "reflective_surface":
            # Missing estimates stand in for a possible failure; this is not
            # a physical model of glass or of a network's behavior on glass.
            depth[:, a:b] = np.nan
        elif self.scenario == "invalid_depth":
            depth[:] = np.nan
        return depth


def synthetic_frames(count: int = 30, fps: float = 10.0):
    if count < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("count and FPS must be positive")
    for index in range(count):
        yield Frame(index, index / fps, np.zeros((60, 90, 3), dtype=np.uint8), "synthetic_exact")
