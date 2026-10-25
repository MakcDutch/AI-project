"""Розвідувальна перевірка DA3 на одному знімку (ранній етап роботи).

Фокусна відстань тут — груба оцінка, НЕ з калібрування. Для вимірювань
використовуйте src/eval/validate.py і src/eval/compare_models.py, які беруть
фокусну з results/calib_phone_960.json. Цей скрипт лишено як історію роботи.

Запуск з кореня репозиторію:  python src/eval/check_da3.py
"""

from pathlib import Path

import torch
import matplotlib.pyplot as plt
from depth_anything_3.api import DepthAnything3

ROOT = Path(__file__).resolve().parents[2]
IMG_PATH = ROOT / "data" / "test.jpg"
OUT_PNG = ROOT / "results" / "da3_test.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
model = model.to(device=device)

prediction = model.inference([str(IMG_PATH)])

raw = prediction.depth[0]
img = prediction.processed_images[0]

focal = 345.0  # ГРУБА оцінка для кадру 504x378, після калібровки замінимо
metric = focal * raw / 300.0

fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].imshow(img)
ax[0].set_title("фото")
ax[0].axis("off")
im = ax[1].imshow(metric, cmap="turbo")
ax[1].set_title("глибина, м (приблизно)")
ax[1].axis("off")
fig.colorbar(im, ax=ax[1], fraction=0.046)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.show()