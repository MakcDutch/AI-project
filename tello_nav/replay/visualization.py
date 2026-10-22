import cv2
import numpy as np

from tello_nav.controller.sectors import SectorConfig, sector_rectangles
from .video import Frame


class ReplayViewer:
    def __init__(self, sectors: SectorConfig):
        self.sectors = sectors

    def show(self, frame: Frame, depth: np.ndarray | None, record: dict) -> bool:
        rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        if depth is None:
            color = np.zeros_like(rgb)
        else:
            good = np.isfinite(depth) & (depth > 0)
            scaled = np.zeros(depth.shape, np.uint8)
            scaled[good] = (np.clip(depth[good], 0, 10) * 25.5).astype(np.uint8)
            color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
            color[~good] = 0
        for name, (x0, y0, x1, y1) in sector_rectangles(frame.rgb.shape, self.sectors).items():
            for panel in (rgb, color):
                cv2.rectangle(panel, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
            stats = record["sectors"].get(name)
            label = f"{name}: unknown" if not stats else (
                f"{name}: p={stats['distance_m']} min={stats['minimum_m']} valid={stats['valid_coverage']:.0%}")
            cv2.putText(color, label, (x0 + 2, max(12, y0)), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
        joined = np.hstack((rgb, color))
        width = max(1000, joined.shape[1])
        canvas = np.zeros((joined.shape[0] + 90, width, 3), np.uint8)
        canvas[:joined.shape[0], :joined.shape[1]] = joined
        lines = [f"frame={frame.index} t={frame.timestamp:.3f}s ({frame.timestamp_source}) age={record['frame_age_s']:.3f}s",
                 f"{record['controller_state']} {record['command']}", record['reason']]
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (8, joined.shape[0] + 20 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv2.imshow("Tello replay | q / Esc exits", canvas)
        return cv2.waitKey(1) & 0xFF not in (ord("q"), 27)

    def close(self) -> None:
        cv2.destroyAllWindows()
