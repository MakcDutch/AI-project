"""Розвідувальна перевірка Metric3D на одному знімку (ранній етап роботи).

Фокусна відстань тут — груба оцінка, НЕ з калібрування. Для вимірювань
використовуйте src/eval/validate.py і src/eval/compare_models.py. Цей скрипт
лишено як історію роботи.

Запуск з кореня репозиторію:  python src/eval/check_metric3d.py
"""

import sys
from pathlib import Path

sys.modules["xformers"] = None
sys.modules["xformers.ops"] = None

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
IMG_PATH = str(ROOT / "data" / "test.jpg")
OUT_PNG = ROOT / "results" / "m3d_test.png"
FX_ORIG = 876.0  # ГРУБА оцінка для оригінального кадру 960x1280 (відповідає 345 px для DA3), після калібровки замінимо

device = "cuda" if torch.cuda.is_available() else "cpu"

rgb_origin = cv2.imread(IMG_PATH)[:, :, ::-1]  # BGR -> RGB
h0, w0 = rgb_origin.shape[:2]

# --- передобробка (як у hubconf.py) ---
input_size = (616, 1064)  # для ViT
scale = min(input_size[0] / h0, input_size[1] / w0)
rgb = cv2.resize(rgb_origin, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
fx = FX_ORIG * scale  # фокусна для зменшеного кадру

padding = [123.675, 116.28, 103.53]
h, w = rgb.shape[:2]
pad_h = input_size[0] - h
pad_w = input_size[1] - w
pad_h_half = pad_h // 2
pad_w_half = pad_w // 2
rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                         cv2.BORDER_CONSTANT, value=padding)
pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
rgb = torch.div((rgb - mean), std)
rgb = rgb[None, :, :, :].to(device)

# --- модель ---
model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True, trust_repo=True)
model.to(device).eval()

with torch.no_grad():
    pred_depth, confidence, output_dict = model.inference({"input": rgb})

# --- післяобробка ---
pred_depth = pred_depth.squeeze()
pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1],
                        pad_info[2]: pred_depth.shape[1] - pad_info[3]]
pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], (h0, w0),
                                             mode="bilinear").squeeze()
pred_depth = pred_depth * (fx / 1000.0)  # з канонічної камери в метри
pred_depth = torch.clamp(pred_depth, 0, 300)
depth = pred_depth.cpu().numpy()

print("метри: min %.2f  max %.2f  mean %.2f" % (depth.min(), depth.max(), depth.mean()))

# значення в двох точках оригінального кадру (x, y), медіана у вікні 21x21
def sample(x, y, r=10):
    return float(np.median(depth[y - r:y + r + 1, x - r:x + r + 1]))

print("крісло (сидіння): %.2f м" % sample(330, 760))
print("двері: %.2f м" % sample(520, 450))

fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].imshow(rgb_origin)
ax[0].set_title("фото")
ax[0].axis("off")
im = ax[1].imshow(depth, cmap="turbo", vmin=0.7, vmax=4.7)
ax[1].set_title("Metric3D, глибина, м (приблизно)")
ax[1].axis("off")
fig.colorbar(im, ax=ax[1], fraction=0.046)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.show()