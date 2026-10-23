import os
import tempfile

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

class DA3Metric:
    name = "DA3-metric-large"

    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
        self.model = model.to(device=self.device)

    def predict(self, rgb, fx):
        h0, w0 = rgb.shape[:2]

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            cv2.imwrite(path, np.ascontiguousarray(rgb[:, :, ::-1]))
            prediction = self.model.inference([path])
        finally:
            os.remove(path)

        raw = np.asarray(prediction.depth[0], dtype=np.float32)
        fx_proc = fx * raw.shape[1] / w0
        metric = fx_proc * raw / 300.0
        return cv2.resize(metric, (w0, h0), interpolation=cv2.INTER_LINEAR)