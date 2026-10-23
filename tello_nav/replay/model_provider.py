"""Adapters for the team's unchanged predict(rgb, fx) model wrappers.

No torch, model package, weights or network is used until explicit construction.
"""
from dataclasses import dataclass, field
import importlib
import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from .providers import validate_depth
from .video import Frame


class TeamDepthModel(Protocol):
    name: str

    def predict(self, rgb: np.ndarray, fx: float) -> np.ndarray: ...


@dataclass(frozen=True)
class CameraCalibration:
    """Existing size/K/dist JSON schema; same camera, orientation and crop required.

    Device identity and optical settings cannot be inferred from this schema.
    No undistortion is applied: callers must keep RGB/calibration geometry aligned.
    """

    path: Path
    size: tuple[int, int] = field(init=False)
    focal: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        data = json.loads(self.path.read_text())
        size = data["size"]
        matrix = np.asarray(data["K"], dtype=float)
        distortion = np.asarray(data.get("dist", []), dtype=float)
        if (len(size) != 2 or any(type(v) is not int or v <= 0 for v in size)
                or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0, 0, 1])
                or not np.isfinite(distortion).all()):
            raise ValueError("invalid calibration size, K, or distortion")
        # Freeze calibration contents for an entire run; no mid-run file reload.
        object.__setattr__(self, "size", tuple(size))
        object.__setattr__(self, "focal", float(matrix[0, 0] / 2 + matrix[1, 1] / 2))

    def focal_for(self, width: int, height: int) -> float:
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        cal_w, cal_h = self.size
        sx, sy = width / cal_w, height / cal_h
        # Allow integer resize rounding, not rotation, crop, or aspect distortion.
        if not math.isclose(sx, sy, rel_tol=1e-3, abs_tol=1 / max(cal_w, cal_h)):
            raise ValueError("frame aspect/orientation differs from calibration; supply matching intrinsics")
        # Preserve calibration.intrinsics.load_focal's public convention exactly:
        # average fx/fy, scaled by longest side. No empirical correction factor.
        return self.focal * max(width, height) / max(cal_w, cal_h)


class TeamModelDepthProvider:
    """Turn the existing predict(RGB, focal_px) into predict(Frame), no rescaling.

    Metric3D's wrapper clamps +inf and excessive depths to 300m. Treat that
    saturation as unknown here so a numerical failure cannot become free space.
    Confidence and pre-resize invalid masks are not exposed by either wrapper.
    """

    def __init__(self, model: TeamDepthModel, calibration: CameraCalibration, *, kind: str):
        if kind not in {"da3", "metric3d"}:
            raise ValueError("kind must be da3 or metric3d")
        self.model, self.calibration, self.kind = model, calibration, kind
        self.name = model.name

    def predict(self, frame: Frame) -> np.ndarray:
        rgb = frame.rgb
        if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3
                or rgb.dtype != np.uint8 or not all(rgb.shape[:2])):
            raise ValueError("model input must be a nonempty HxWx3 uint8 RGB array")
        height, width = rgb.shape[:2]
        focal = self.calibration.focal_for(width, height)
        # A copy prevents a model from modifying subsequent recorded observations.
        depth = validate_depth(self.model.predict(np.array(rgb, order="C", copy=True), focal), frame)
        depth = depth.astype(np.float32, copy=True)
        invalid = ~np.isfinite(depth) | (depth <= 0)
        if self.kind == "metric3d":
            invalid |= depth >= 300.0
        depth[invalid] = np.nan
        return depth

    def metadata(self) -> dict:
        return {"kind": self.kind, "model": self.name, "convention": "camera_z_m",
                "calibration": str(self.calibration.path), "calibration_size": self.calibration.size,
                "focal_convention": "mean_fx_fy_times_longest_side_ratio",
                "distortion_handling": "no_undistortion",
                "metric3d_saturation_is_unknown": self.kind == "metric3d"}


def create_model_provider(kind: str, calibration_path: str | Path, *, device: str = "cpu",
                          allow_model_download: bool = False) -> TeamModelDepthProvider:
    """The existing constructors may fetch weights/code, even if caches exist.

    Require explicit permission before importing model packages or constructing
    either wrapper. Supplying an already initialized model to the provider above
    is the alternative for applications managing their own local model lifecycle.
    """
    if kind not in {"da3", "metric3d"}:
        raise ValueError("unknown depth model")
    if allow_model_download is not True:
        raise PermissionError("model constructors may download weights/code; explicitly allow model download")
    calibration = CameraCalibration(Path(calibration_path))
    try:
        if kind == "da3":
            wrapper = importlib.import_module("src.depth_models.da3").DA3Metric(device=device)
        else:
            wrapper = importlib.import_module("src.depth_models.metric3d").Metric3D(
                "metric3d_vit_small", device=device)
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(f"{kind} initialization failed; install the team's model dependencies and weights: {exc}") from exc
    return TeamModelDepthProvider(wrapper, calibration, kind=kind)
