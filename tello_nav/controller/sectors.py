from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SectorConfig:
    # Normalized crop (x0, y0, x1, y1), then normalized horizontal divisions.
    roi: tuple[float, float, float, float] = (0.0, 0.2, 1.0, 0.85)
    boundaries: tuple[float, float] = (1 / 3, 2 / 3)
    percentile: float = 10.0
    min_coverage: float = 0.9
    min_valid_pixels: int = 16
    # Any close valid pixel triggers blockage, even below percentile support.
    # This intentionally favors false stops over missing thin obstacles.

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.roi
        a, b = self.boundaries
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and 0 < a < b < 1):
            raise ValueError("invalid normalized ROI or boundaries")
        if not (0 <= self.percentile <= 100 and 0 < self.min_coverage <= 1):
            raise ValueError("invalid percentile or coverage")
        if type(self.min_valid_pixels) is not int or self.min_valid_pixels < 1:
            raise ValueError("min_valid_pixels must be a positive integer")


@dataclass(frozen=True)
class SectorStats:
    distance_m: float | None
    minimum_m: float | None
    valid_coverage: float
    valid_pixels: int
    total_pixels: int
    known: bool

    def clear(self, threshold: float) -> bool:
        return (self.known and self.distance_m is not None and self.minimum_m is not None
                and self.distance_m > threshold and self.minimum_m > threshold)


def sector_rectangles(shape: tuple[int, ...], config: SectorConfig) -> dict[str, tuple[int, int, int, int]]:
    h, w = shape[:2]
    x0, y0, x1, y1 = config.roi
    left, top, right, bottom = int(w * x0), int(h * y0), int(w * x1), int(h * y1)
    width = right - left
    a, b = (left + int(width * f) for f in config.boundaries)
    return {"left": (left, top, a, bottom), "center": (a, top, b, bottom),
            "right": (b, top, right, bottom)}


def analyze(depth_map: np.ndarray, config: SectorConfig) -> dict[str, SectorStats]:
    if (not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2
            or not all(depth_map.shape) or not np.issubdtype(depth_map.dtype, np.number)
            or np.iscomplexobj(depth_map)):
        raise ValueError("depth must be a nonempty real numeric HxW array")
    result = {}
    for name, (x0, y0, x1, y1) in sector_rectangles(depth_map.shape, config).items():
        values = depth_map[y0:y1, x0:x1]
        valid = values[np.isfinite(values) & (values > 0)]
        count, total = int(valid.size), int(values.size)
        coverage = count / total if total else 0.0
        result[name] = SectorStats(
            float(np.percentile(valid, config.percentile)) if count else None,
            float(valid.min()) if count else None, coverage, count, total,
            count >= config.min_valid_pixels and coverage >= config.min_coverage,
        )
    return result
