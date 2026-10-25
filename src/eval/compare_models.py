"""Порівняння DA3 і Metric3D на одному знімку.

Запуск з кореня репозиторію:  python src/eval/compare_models.py
Потребує ваги моделей, matplotlib і знімок за шляхом IMG_PATH.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from depth_models.da3 import DA3Metric
from depth_models.metric3d import Metric3D
from calibration.intrinsics import load_focal

ROOT = Path(__file__).resolve().parents[2]
IMG_PATH = ROOT / "data" / "test.jpg"
CALIB_JSON = ROOT / "results" / "calib_phone_960.json"
OUT_PNG = ROOT / "results" / "compare.png"
POINTS = {"Chair": (330, 760), "Doors": (520, 450)}

if not IMG_PATH.exists():
    raise SystemExit(f"Немає знімка {IMG_PATH}. Покладіть його туди або змініть IMG_PATH.")

rgb = np.ascontiguousarray(cv2.imread(str(IMG_PATH))[:, :, ::-1])

FX = load_focal(CALIB_JSON, rgb.shape[1], rgb.shape[0])
print(f"fx for this photo: {FX:.1f} px")

models = [DA3Metric(), Metric3D("metric3d_vit_small")]


def sample(depth, x, y, r=10):
    return float(np.median(depth[y - r:y + r + 1, x - r:x + r + 1]))


results = {}
for m in models:
    d = m.predict(rgb, FX)
    results[m.name] = d
    print(f"\n{m.name}: min {d.min():.2f}  max {d.max():.2f}  mean {d.mean():.2f}")
    for label, (x, y) in POINTS.items():
        print(f"  {label}: {sample(d, x, y):.2f} м")

names = list(results)
a, b = results[names[0]], results[names[1]]
print(f"\naverage relative difference between models: {np.mean(np.abs(a - b) / b):.1%}")

fig, ax = plt.subplots(1, 3, figsize=(15, 7))
ax[0].imshow(rgb)
ax[0].set_title("photo")
for i, n in enumerate(names, start=1):
    im = ax[i].imshow(results[n], cmap="turbo", vmin=0.7, vmax=4.7)
    ax[i].set_title(n)
for a_ in ax:
    a_.axis("off")
fig.colorbar(im, ax=ax[1:], fraction=0.02)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.show()