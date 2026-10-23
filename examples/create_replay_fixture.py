"""Generate explicitly synthetic videos/depth, not model predictions or flights.

Run from repository root: python -m examples.create_replay_fixture
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from tello_nav.replay.providers import MockDepthProvider
from tello_nav.replay.video import Frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/integration_demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for name in ("empty_corridor", "center_obstacle", "invalid_depth"):
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        writer = cv2.VideoWriter(str(directory / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (90, 60))
        if not writer.isOpened():
            raise RuntimeError("OpenCV MJPG video writer unavailable")
        provider = MockDepthProvider(name)
        try:
            with (directory / "timestamps.csv").open("w", newline="") as handle:
                timestamps = csv.writer(handle)
                timestamps.writerow(["frame_index", "timestamp_s"])
                for index in range(12):
                    rgb = np.full((60, 90, 3), (30, 60, 90), np.uint8)
                    cv2.putText(rgb, str(index), (30, 35), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                    timestamp = 100 + index / 10
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    np.save(directory / f"{index:06d}.npy", provider.predict(Frame(index, timestamp, rgb)))
                    timestamps.writerow([index, f"{timestamp:.1f}"])
        finally:
            writer.release()
        scenarios.append({"id": f"synthetic-fixture-{name}", "video": f"{name}/video.avi",
                          "timestamps": f"{name}/timestamps.csv", "depth_dir": name,
                          "model_name": provider.name})
    manifest = args.output / "scenarios.json"
    manifest.write_text(json.dumps(scenarios, indent=2) + "\n")
    print(f"Generated synthetic fixtures: {manifest}")


if __name__ == "__main__":
    main()
