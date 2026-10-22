import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp: float
    rgb: np.ndarray
    timestamp_source: str = "external"


def read_timestamps(path: str | Path) -> list[float]:
    """CSV columns frame_index,timestamp_s, contiguous zero-based indices."""
    result = []
    with Path(path).open(newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != ["frame_index", "timestamp_s"]:
            raise ValueError("timestamp CSV requires exactly frame_index,timestamp_s columns")
        for index, row in enumerate(rows):
            try:
                value = float(row["timestamp_s"])
                if int(row["frame_index"]) != index or not math.isfinite(value) or value < 0:
                    raise ValueError()
                if result and value <= result[-1]:
                    raise ValueError()
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid timestamp row {index + 2}") from exc
            result.append(value)
    if not result:
        raise ValueError("empty timestamp file")
    return result


class VideoReader:
    def __init__(self, path: str | Path, timestamps: str | Path | None = None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.timestamps = read_timestamps(timestamps) if timestamps else None

    def __iter__(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise ValueError(f"cannot open video: {self.path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            advertised_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if self.timestamps is None and (not math.isfinite(fps) or fps <= 0):
                raise ValueError("video has no usable FPS; supply timestamp CSV")
            index = 0
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                if self.timestamps is not None:
                    if index >= len(self.timestamps):
                        raise ValueError("timestamp file shorter than decoded video")
                    timestamp, source = self.timestamps[index], "external"
                else:
                    timestamp, source = index / fps, "fps_approximation"
                yield Frame(index, timestamp, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), source)
                index += 1
            if index == 0:
                raise ValueError("video contains no decodable frames")
            if self.timestamps is not None and index != len(self.timestamps):
                raise ValueError("timestamp file longer than decoded video (or decoding failed)")
            if math.isfinite(advertised_count) and advertised_count > 0 and index < round(advertised_count):
                raise ValueError("video decoding ended before advertised frame count")
        finally:
            capture.release()
