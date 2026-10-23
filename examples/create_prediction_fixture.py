"""Synthetic exchange artifacts, never learned outputs or real calibration."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from tello_nav.replay.predictions import PredictionExporter
from tello_nav.replay.video import VideoReader


def create_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    video = root / "video.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (90, 60))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPG encoder required")
    try:
        for index in range(8):
            rgb = np.full((60, 90, 3), (index * 15, 70, 140), np.uint8)
            cv2.putText(rgb, str(index), (35, 35), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    times = root / "timestamps.csv"
    with times.open("w", newline="") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["frame_index", "timestamp_s"])
        csv_writer.writerows((i, 100 + i / 10) for i in range(8))
    k = [[90.0, 0, 45.0], [0, 90.0, 30.0], [0, 0, 1.0]]
    calibration = root / "synthetic_calibration.json"
    calibration.write_text(json.dumps({"size": [90, 60], "K": k, "dist": [0, 0, 0, 0, 0],
                                       "notice": "Synthetic fixture; not DJI Tello calibration"}, indent=2) + "\n")
    specs = []
    for kind in ("da3", "metric3d"):
        exporter = PredictionExporter(
            root / kind, scenario_id="exchange-demo", frame_count=8, video_path=video,
            model={"kind": kind, "name": f"{kind.upper()}-LABELED-SYNTHETIC",
                   "checkpoint_id": "synthetic:no-neural-checkpoint", "checkpoint_revision": "fixture-v1",
                   "origin": "synthetic"}, calibration_path=calibration,
            calibration_id="synthetic-camera-v1", camera_id="synthetic-camera",
            calibration_status="synthetic", calibration_geometry="undistorted",
            depth={"dtype": "float32", "units": "m", "convention": "optical_axis_z",
                   "invalid_values": "nonfinite_or_nonpositive", "saturation_m": 300.0 if kind == "metric3d" else None})
        for frame in VideoReader(video, times):
            depth = np.full((60, 90), 4.0, np.float32)
            if frame.index in (2, 3):
                depth[:, 30:60] = .5
            if frame.index == 3:
                depth[:, :30] = .5
            if kind == "metric3d" and frame.index == 4:
                depth[:] = 300.0
            if frame.index == 6:
                depth[20, 45] = .2
            if frame.index == 7:
                depth[:] = np.nan
            exporter.add(frame, depth, camera_intrinsics=k, inference_focal_px=90.0,
                         preprocessing={"input_color": "RGB", "output_alignment": "original_rgb",
                                        "description": "Synthetic arrays; no neural preprocessing, resizing or scaling"})
        exporter.finalize()
        specs.append({"id": f"exchange-demo-{kind}", "scenario_id": "exchange-demo", "video": "video.avi",
                      "timestamps": "timestamps.csv", "predictions": f"{kind}/predictions.json",
                      "calibration": "synthetic_calibration.json", "calibration_id": "synthetic-camera-v1",
                      "camera_id": "synthetic-camera"})
    (root / "scenarios.json").write_text(json.dumps(specs, indent=2) + "\n")
    # Explicitly broken copies for diagnosis; never used as valid predictions.
    for case in ("missing_prediction", "timestamp_mismatch", "calibration_mismatch", "corrupted_file", "radial_depth"):
        directory = root / "invalid" / case
        shutil.copytree(root / "da3", directory)
        manifest = directory / "predictions.json"
        data = json.loads(manifest.read_text())
        if case == "missing_prediction":
            data["frames"].pop()
        elif case == "timestamp_mismatch":
            data["frames"][0]["timestamp_s"] += .01
        elif case == "calibration_mismatch":
            data["calibration"]["id"] = "another-camera-calibration"
        elif case == "corrupted_file":
            (directory / "depth/000000.npy").write_bytes(b"intentionally corrupted fixture")
        else:
            data["depth"]["convention"] = "radial_range"
        manifest.write_text(json.dumps(data, indent=2) + "\n")
    return root / "scenarios.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction_handoff"))
    args = parser.parse_args()
    print(f"Created synthetic exchange fixtures: {create_fixture(args.output)}")


if __name__ == "__main__":
    main()
