import sys

import cv2
import numpy as np
import torch


class Metric3D:
    INPUT_SIZE = (616, 1064)

    def __init__(self, variant="metric3d_vit_small", device=None):
        sys.modules["xformers"] = None
        sys.modules["xformers.ops"] = None

        self.name = variant
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.hub.load("yvanyin/metric3d", variant, pretrain=True, trust_repo=True)
        self.model.to(self.device).eval()

        self.mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        self.std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    def predict(self, rgb, fx):
        h0, w0 = rgb.shape[:2]
        ih, iw = self.INPUT_SIZE

        scale = min(ih / h0, iw / w0)
        img = cv2.resize(rgb, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
        fx_scaled = fx * scale

        h, w = img.shape[:2]
        pad_h, pad_w = ih - h, iw - w
        top, left = pad_h // 2, pad_w // 2
        img = cv2.copyMakeBorder(img, top, pad_h - top, left, pad_w - left,
                                 cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])

        x = torch.from_numpy(img.transpose((2, 0, 1))).float()
        x = torch.div(x - self.mean, self.std)[None].to(self.device)

        with torch.no_grad():
            pred, _, _ = self.model.inference({"input": x})

        pred = pred.squeeze()
        pred = pred[top: pred.shape[0] - (pad_h - top), left: pred.shape[1] - (pad_w - left)]
        pred = torch.nn.functional.interpolate(pred[None, None], (h0, w0), mode="bilinear").squeeze()
        pred = torch.clamp(pred * (fx_scaled / 1000.0), 0, 300)
        return pred.cpu().numpy()