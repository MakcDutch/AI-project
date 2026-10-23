"""Version-1 prediction exchange. No neural dependencies or unit conversion."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from .providers import validate_depth
from .video import Frame


class PredictionError(ValueError):
    """Invalid exchange artifact, provenance or frame alignment."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_sha256(rgb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def frame_id(scenario_id: str, index: int) -> str:
    """Canonical ID assigned by this format, independent of timestamps."""
    return f"{scenario_id}:{index:06d}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PredictionError(message)


def _text(value: Any, field: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} must be a nonempty string")


def _finite(value: Any, field: str) -> None:
    _require(type(value) in (int, float) and math.isfinite(value), f"{field} must be a finite number")


def _hash(value: Any, field: str) -> None:
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), f"invalid {field} SHA-256")


def _load_json(path: Path) -> dict:
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                _require(key not in result, f"duplicate JSON key: {key}")
                result[key] = value
            return result
        data = json.loads(path.read_text(), object_pairs_hook=unique)
        _require(isinstance(data, dict), "JSON root must be an object")
        return data
    except (OSError, ValueError, TypeError) as exc:
        raise PredictionError(f"cannot read {path.name}: {exc}") from exc


def _relative(root: Path, value: Any) -> Path:
    _text(value, "artifact filename")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts, "artifact paths must remain inside bundle")
    resolved = (root / path).resolve()
    _require(resolved.is_relative_to(root.resolve()), "artifact path/symlink escapes bundle")
    return resolved


def _calibration(path: Path) -> tuple[dict, np.ndarray]:
    data = _load_json(path)
    size = data.get("size")
    _require(isinstance(size, list) and len(size) == 2 and
             all(type(v) is int and v > 0 for v in size), "invalid calibration size")
    try:
        k = np.asarray(data.get("K"), dtype=float)
        dist = np.asarray(data["dist"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionError("calibration requires numeric K and dist") from exc
    _require(k.shape == (3, 3) and np.isfinite(k).all() and k[0, 0] > 0 and k[1, 1] > 0
             and np.allclose(k[2], [0, 0, 1]) and dist.ndim == 1 and np.isfinite(dist).all(),
             "invalid calibration K or distortion")
    return data, k


def _metadata(data: dict) -> None:
    try:
        _require(type(data["schema_version"]) is int and data["schema_version"] == 1,
                 "unsupported prediction schema_version")
        _text(data["scenario_id"], "scenario_id")
        _require(type(data["frame_count"]) is int and data["frame_count"] > 0, "invalid frame_count")
        _hash(data["video_sha256"], "video")
        model, cal, depth = data["model"], data["calibration"], data["depth"]
        for field in ("name", "checkpoint_id", "checkpoint_revision"):
            _text(model[field], f"model.{field}")
        _require(model["kind"] in ("da3", "metric3d"), "model.kind must be da3 or metric3d")
        _require(model["origin"] in ("pretrained", "synthetic"), "invalid model.origin")
        for field in ("id", "camera_id", "geometry"):
            _text(cal[field], f"calibration.{field}")
        _require(cal["geometry"] in ("distorted", "undistorted"), "unsupported calibration geometry")
        _require(cal["status"] in ("verified", "provisional", "synthetic"), "invalid calibration.status")
        _hash(cal["sha256"], "calibration")
        _text(cal["file"], "calibration.file")
        for field in ("units", "convention"):
            _text(depth[field], f"depth.{field}")
        _require(depth["dtype"] == "float32", "depth dtype must be float32")
        _require(depth["invalid_values"] == "nonfinite_or_nonpositive", "unsupported invalid-depth policy")
        saturation = depth["saturation_m"]
        if saturation is not None:
            _finite(saturation, "saturation_m")
            _require(saturation > 0, "saturation_m must be positive")
        if model["kind"] == "metric3d":
            _require(saturation == 300.0, "existing Metric3D wrapper requires declared 300m saturation")
        json.dumps(data, allow_nan=False)
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionError(f"invalid prediction metadata: {exc}") from exc


def _record(row: dict, data: dict, calibration: dict, k_ref: np.ndarray) -> None:
    try:
        index = row["frame_index"]
        _require(type(index) is int and 0 <= index < data["frame_count"], "invalid frame index")
        _require(row["frame_id"] == frame_id(data["scenario_id"], index), "frame ID/index/scenario mismatch")
        _require(row["calibration_id"] == data["calibration"]["id"], "frame calibration ID mismatch")
        _finite(row["timestamp_s"], "timestamp_s")
        _require(row["timestamp_s"] >= 0, "negative frame timestamp")
        _require(row["timestamp_source"] in ("external", "fps_approximation", "synthetic_exact"),
                 "unsupported timestamp source")
        h, w = row["image_shape"]
        _require(type(h) is int and type(w) is int and min(h, w) > 0, "invalid image shape")
        _require(row["depth_shape"] == [h, w], "depth/image geometry mismatch; no implicit resizing")
        cw, ch = calibration["size"]
        sx, sy = w / cw, h / ch
        _require(math.isclose(sx, sy, rel_tol=1e-3, abs_tol=1 / max(cw, ch)),
                 "frame aspect/orientation incompatible with calibration")
        expected = k_ref.copy()
        expected[0] *= sx
        expected[1] *= sy
        k = np.asarray(row["camera_intrinsics"], dtype=float)
        _require(k.shape == (3, 3) and np.isfinite(k).all() and
                 np.allclose(k, expected, rtol=1e-6, atol=1e-6), "frame camera intrinsics mismatch")
        _finite(row["inference_focal_px"], "inference_focal_px")
        _require(row["inference_focal_px"] > 0, "inference focal must be positive")
        preprocessing = row["preprocessing"]
        _require(isinstance(preprocessing, dict), "preprocessing must be an object")
        _require(preprocessing["input_color"] == "RGB" and
                 preprocessing["output_alignment"] == "original_rgb", "unsupported color/alignment")
        _text(preprocessing["description"], "preprocessing.description")
        for field in ("rgb_sha256", "depth_sha256"):
            _hash(row[field], field)
        _text(row["depth_file"], "depth_file")
        if row["inference_latency_s"] is not None:
            _finite(row["inference_latency_s"], "inference latency")
            _require(row["inference_latency_s"] >= 0, "negative inference latency")
        json.dumps(row, allow_nan=False)
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionError(f"invalid frame metadata: {exc}") from exc


class PredictionExporter:
    """New bundle only; publish manifest only after all frames are exported.

    Metadata is supplied by the inference owner. This does not certify its truth.
    Float32 arrays are saved exactly, including NaN/Inf/zero/negative sentinels.
    """

    def __init__(self, directory: str | Path, *, scenario_id: str, frame_count: int,
                 video_path: str | Path, model: dict, calibration_path: str | Path,
                 calibration_id: str, camera_id: str, calibration_status: str,
                 calibration_geometry: str, depth: dict):
        source = Path(calibration_path)
        self.calibration, self.k_ref = _calibration(source)
        self.data = {"schema_version": 1, "scenario_id": scenario_id, "frame_count": frame_count,
                     "video_sha256": file_sha256(video_path), "model": model,
                     "calibration": {"id": calibration_id, "camera_id": camera_id,
                                     "status": calibration_status, "geometry": calibration_geometry,
                                     "file": "calibration.json", "sha256": file_sha256(source)},
                     "depth": depth, "frames": []}
        # A JSON round trip freezes caller-owned metadata.
        _metadata(self.data)
        _require(calibration_geometry != "undistorted" or np.allclose(self.calibration["dist"], 0),
                 "undistorted geometry requires a matching zero-distortion calibration")
        self.data = json.loads(json.dumps(self.data, allow_nan=False))
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "depth").mkdir()
        with (self.root / "calibration.json").open("xb") as handle:
            handle.write(source.read_bytes())
        self.closed = False

    def add(self, frame: Frame, depth_map: np.ndarray, *, camera_intrinsics: list,
            inference_focal_px: float, preprocessing: dict, inference_latency_s: float | None = None) -> None:
        _require(not self.closed, "exporter is finalized")
        validate_depth(depth_map, frame)
        _require(depth_map.dtype == np.dtype("float32"), "export requires float32; cast explicitly before export")
        _require(frame.rgb.ndim == 3 and frame.rgb.shape[2] == 3 and frame.rgb.dtype == np.uint8,
                 "export frame must be uint8 RGB HxWx3")
        rows = self.data["frames"]
        _require(not rows or frame.index > rows[-1]["frame_index"], "duplicate or out-of-order export frame")
        _require(not rows or frame.timestamp > rows[-1]["timestamp_s"], "non-increasing export timestamp")
        filename = f"depth/{frame.index:06d}.npy"
        row = {"frame_id": frame_id(self.data["scenario_id"], frame.index), "frame_index": frame.index,
               "timestamp_s": frame.timestamp, "timestamp_source": frame.timestamp_source,
               "image_shape": list(frame.rgb.shape[:2]), "depth_shape": list(depth_map.shape),
               "calibration_id": self.data["calibration"]["id"], "camera_intrinsics": camera_intrinsics,
               "inference_focal_px": inference_focal_px, "preprocessing": preprocessing,
               "rgb_sha256": rgb_sha256(frame.rgb), "depth_sha256": "0" * 64,
               "depth_file": filename, "inference_latency_s": inference_latency_s}
        _record(row, self.data, self.calibration, self.k_ref)
        with (self.root / filename).open("xb") as handle:
            np.save(handle, depth_map, allow_pickle=False)
        row["depth_sha256"] = file_sha256(self.root / filename)
        rows.append(json.loads(json.dumps(row, allow_nan=False)))

    def finalize(self) -> Path:
        _require(not self.closed, "exporter is already finalized")
        _require([r["frame_index"] for r in self.data["frames"]] == list(range(self.data["frame_count"])),
                 "missing predictions; cannot finalize incomplete bundle")
        path = self.root / "predictions.json"
        with path.open("x") as handle:
            json.dump(self.data, handle, indent=2, allow_nan=False)
            handle.write("\n")
        self.closed = True
        return path


class ManifestDepthProvider:
    """Strict prediction import. No shape, unit, ray-depth or scale conversion.

    Invalid pixel values reach the existing controller as unknown. Structural or
    provenance failures reject the bundle; per-frame file/alignment failures raise
    PredictionError, which ReplayRunner logs while issuing STOP.
    """

    def __init__(self, manifest: str | Path, *, scenario_id: str, video_path: str | Path,
                 calibration_path: str | Path, calibration_id: str, camera_id: str,
                 allow_synthetic: bool = False):
        self.path = Path(manifest)
        self.root = self.path.parent
        self.data = _load_json(self.path)
        _metadata(self.data)
        data = self.data
        _require(data["scenario_id"] == scenario_id, "prediction scenario mismatch")
        _require(data["video_sha256"] == file_sha256(video_path), "source video SHA-256 mismatch")
        cal = data["calibration"]
        _require(cal["id"] == calibration_id and cal["camera_id"] == camera_id,
                 "expected calibration/camera identifier mismatch")
        _require(cal["sha256"] == file_sha256(calibration_path), "expected calibration SHA-256 mismatch")
        cal_file = _relative(self.root, cal["file"])
        _require(file_sha256(cal_file) == cal["sha256"], "bundled calibration SHA-256 mismatch")
        calibration, k_ref = _calibration(cal_file)
        _require(cal["geometry"] != "undistorted" or np.allclose(calibration["dist"], 0),
                 "undistorted geometry requires a matching zero-distortion calibration")
        _require(data["depth"]["units"] == "m" and data["depth"]["convention"] == "optical_axis_z",
                 "unsuitable for metric controller: require meters and optical_axis_z; convert explicitly upstream")
        synthetic = data["model"]["origin"] == "synthetic"
        _require((synthetic and allow_synthetic is True and cal["status"] == "synthetic") or
                 (not synthetic and cal["status"] == "verified"),
                 "unsuitable calibration/provenance: real predictions require verified calibration; synthetic requires explicit opt-in")
        self.synthetic = synthetic
        self.name = data["model"]["name"]
        try:
            rows = data["frames"]
            _require(isinstance(rows, list), "frames must be a list")
            self.rows = {}
            filenames = set()
            previous_time = -1.0
            for row in rows:
                _record(row, data, calibration, k_ref)
                index = row["frame_index"]
                _require(index not in self.rows, "duplicate frame prediction")
                _require(row["depth_file"] not in filenames, "duplicate depth filename")
                _relative(self.root, row["depth_file"])
                _require(row["timestamp_s"] > previous_time, "non-increasing prediction timestamps")
                previous_time = row["timestamp_s"]
                filenames.add(row["depth_file"])
                self.rows[index] = row
            _require(list(self.rows) == list(range(data["frame_count"])), "missing or out-of-order frame predictions")
        except (KeyError, TypeError) as exc:
            raise PredictionError(f"invalid prediction records: {exc}") from exc
        capture = cv2.VideoCapture(str(video_path))
        try:
            _require(capture.isOpened(), "cannot decode source video")
            count = 0
            while capture.read()[0]:
                count += 1
            _require(count == data["frame_count"], "prediction count differs from decoded video frame count")
        finally:
            capture.release()

    def predict(self, frame: Frame) -> np.ndarray:
        row = self.rows.get(frame.index)
        _require(row is not None, f"missing prediction for frame {frame.index}")
        _require(row["frame_id"] == frame_id(self.data["scenario_id"], frame.index), "frame ID mismatch")
        _require(math.isclose(frame.timestamp, row["timestamp_s"], rel_tol=0, abs_tol=1e-6),
                 f"timestamp mismatch at frame {frame.index}")
        _require(frame.timestamp_source == row["timestamp_source"], "timestamp source mismatch")
        _require(list(frame.rgb.shape[:2]) == row["image_shape"] and rgb_sha256(frame.rgb) == row["rgb_sha256"],
                 f"RGB geometry/content mismatch at frame {frame.index}")
        path = _relative(self.root, row["depth_file"])
        try:
            _require(file_sha256(path) == row["depth_sha256"], f"depth SHA-256 mismatch at frame {frame.index}")
            depth = np.load(path, allow_pickle=False)
            _require(isinstance(depth, np.ndarray) and depth.dtype == np.dtype("float32") and list(depth.shape) == row["depth_shape"],
                     "depth file dtype/shape mismatch")
            depth = validate_depth(depth, frame).copy()
        except (OSError, TypeError, ValueError, EOFError) as exc:
            raise PredictionError(f"cannot load depth for frame {frame.index}: {exc}") from exc
        saturation = self.data["depth"]["saturation_m"]
        if saturation is not None:
            depth[depth >= saturation] = np.nan
        return depth

    def frame_metadata(self, frame: Frame) -> dict:
        row = self.rows.get(frame.index)
        return {"frame_id": row["frame_id"] if row else None,
                "source_scenario_id": self.data["scenario_id"],
                "exported_inference_latency_s": row["inference_latency_s"] if row else None,
                "prediction_origin": self.data["model"]["origin"],
                "calibration_status": self.data["calibration"]["status"]}

    def metadata(self) -> dict:
        return {"exchange_schema_version": 1, "manifest_sha256": file_sha256(self.path),
                "scenario_id": self.data["scenario_id"], "model": self.data["model"],
                "calibration": self.data["calibration"], "depth": self.data["depth"],
                "synthetic": self.synthetic,
                "metric_evaluation_eligible_by_metadata": not self.synthetic,
                "verification_limit": "metadata and identity checks do not independently validate camera calibration or network accuracy"}
