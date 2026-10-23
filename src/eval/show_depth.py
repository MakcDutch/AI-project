import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAL_DIR = ROOT / "data" / "validation"
DEPTH_DIR = ROOT / "results" / "depth"
WIN = 7


def median_at(depth, x, y, win=WIN):
    h, w = depth.shape
    r = win // 2
    xi, yi = int(round(x)), int(round(y))
    return float(np.median(depth[max(0, yi - r): min(h, yi + r + 1), max(0, xi - r): min(w, xi + r + 1)]))


def main():
    img_name = sys.argv[1] if len(sys.argv) > 1 else "wall_150.jpg"
    stem = Path(img_name).stem

    maps = sorted(DEPTH_DIR.glob(f"*__{stem}.npy"))
    if not maps:
        raise SystemExit(f"No cached depth maps for {img_name}. Run validate.py first")
    gt = pd.read_csv(VAL_DIR / "gt_points.csv")
    gt = gt[gt["image"] == img_name]
    bgr = cv2.imread(str(VAL_DIR / img_name))
    if bgr is None:
        raise SystemExit(f"Could not read {VAL_DIR / img_name}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depths = [np.load(p) for p in maps]
    vmin, vmax = np.percentile(np.concatenate([d.ravel() for d in depths]), [1, 99])

    fig, ax = plt.subplots(1, 1 + len(maps), figsize=(5 * (1 + len(maps)), 7.5))
    ax[0].imshow(rgb)
    ax[0].set_title(img_name)
    for a, p, d in zip(ax[1:], maps, depths):
        im = a.imshow(d, cmap="turbo", vmin=vmin, vmax=vmax)
        a.set_title(p.stem.split("__")[0])
        for _, r in gt.iterrows():
            pred = median_at(d, r["x"], r["y"])
            a.plot(r["x"], r["y"], "w+", ms=9)
            a.text(r["x"], r["y"] - 14, f"{pred:.2f} ({r['z_geo_m']:.2f})", color="black", fontsize=7,
                   ha="center", bbox=dict(facecolor="white", alpha=0.75, pad=1, edgecolor="none"))
        fig.colorbar(im, ax=a, fraction=0.046, label="м")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    out = ROOT / "results" / f"depth_{stem}.png"
    fig.savefig(out, dpi=130)
    print("Saved:", out)


if __name__ == "__main__":
    main()