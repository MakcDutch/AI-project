import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAL_DIR = ROOT / "data" / "validation"
RES_DIR = ROOT / "results"
CALIB_JSON = RES_DIR / "calib_iphone_960.json"
GT_CSV = VAL_DIR / "gt_points.csv"

WIN = 7
DELTA_THR = 1.25
N_BOOT = 2000


def metrics(pred, gt):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    rel = np.abs(pred - gt) / gt
    return {
        "n": len(gt),
        "AbsRel": rel.mean(),
        "RMSE_m": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "delta1": float(np.mean(np.maximum(pred / gt, gt / pred) < DELTA_THR)),
        "bias": float(np.mean((pred - gt) / gt)),
    }


def loo_scaled(df, gt_col):
    out = pd.Series(index=df.index, dtype=float)
    for img in df["image"].unique():
        m = df["image"] == img
        other = df[~m]
        s = np.median(other[gt_col] / other["z_pred"]) if len(other) else 1.0
        out[m] = df.loc[m, "z_pred"] * s
    return out


def boot_ci_absrel(df, gt_col, rng):
    imgs = df["image"].unique()
    if len(imgs) < 3:
        return (np.nan, np.nan)
    groups = {i: df[df["image"] == i] for i in imgs}
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(imgs, size=len(imgs), replace=True)
        d = pd.concat([groups[i] for i in pick])
        vals.append(np.mean(np.abs(d["z_pred"] - d[gt_col]) / d[gt_col]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def evaluate(df):
    rng = np.random.default_rng(0)
    rows = []
    for model, d in df.groupby("model", sort=False):
        d = d.copy()
        m = metrics(d["z_pred"], d["z_gt"])
        lo, hi = boot_ci_absrel(d, "z_gt", rng)
        rows.append({"model": model, "scope": "all frames (GT=geometry)", **m, "AbsRel_lo": lo, "AbsRel_hi": hi})

        if d["z_tape"].notna().all():
            m = metrics(d["z_pred"], d["z_tape"])
            lo, hi = boot_ci_absrel(d, "z_tape", rng)
            rows.append({"model": model, "scope": "all frames (GT=tape)", **m, "AbsRel_lo": lo, "AbsRel_hi": hi})

        d["z_pred_loo"] = loo_scaled(d, "z_gt")
        m = metrics(d["z_pred_loo"], d["z_gt"])
        rows.append({"model": model, "scope": "after global scale (leave-one-frame-out)", **m,
                     "AbsRel_lo": np.nan, "AbsRel_hi": np.nan})

        for img, g in d.groupby("image", sort=False):
            m = metrics(g["z_pred"], g["z_gt"])
            rows.append({"model": model, "scope": f"{img} (z≈{g['z_gt'].mean():.2f} м)", **m,
                         "AbsRel_lo": np.nan, "AbsRel_hi": np.nan})
    return pd.DataFrame(rows)

def plot_error_vs_distance(df, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for model, d in df.groupby("model", sort=False):
        zs, bias, sd, ab = [], [], [], []
        for _, x in d.groupby("image"):
            rel = (x["z_pred"] - x["z_gt"]) / x["z_gt"]
            zs.append(x["z_gt"].mean())
            bias.append(rel.mean() * 100)
            sd.append(rel.std() * 100)
            ab.append(rel.abs().mean() * 100)
        o = np.argsort(zs)
        zs, bias, sd, ab = (np.array(a)[o] for a in (zs, bias, sd, ab))
        ax[0].errorbar(zs, bias, yerr=sd, marker="o", capsize=3, label=model)
        ax[1].plot(zs, ab, marker="o", label=model)
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_xlabel("distance to wall (geometry), m")
    ax[0].set_ylabel("bias (pred-gt)/gt, %")
    ax[0].set_title("Systematic bias (+/- spread over 8 markers)")
    ax[1].set_xlabel("distance to wall (geometry), m")
    ax[1].set_ylabel("AbsRel, %")
    ax[1].set_title("Error vs distance")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def sample_depth(depth, x, y, win=WIN):
    h, w = depth.shape
    r = win // 2
    xi, yi = int(round(x)), int(round(y))
    patch = depth[max(0, yi - r): min(h, yi + r + 1), max(0, xi - r): min(w, xi + r + 1)]
    return float(np.median(patch))


def run_models(gt):
    import cv2
    from calibration.intrinsics import load_focal
    from depth_models.da3 import DA3Metric
    from depth_models.metric3d import Metric3D

    models = [DA3Metric(), Metric3D("metric3d_vit_small")]

    (RES_DIR / "depth").mkdir(parents=True, exist_ok=True)
    records = []
    for img_name, g in gt.groupby("image", sort=False):
        bgr = cv2.imread(str(VAL_DIR / img_name))
        if bgr is None:
            raise SystemExit(f"Could not read {VAL_DIR / img_name}")
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        w0, h0 = int(g["w"].iloc[0]), int(g["h"].iloc[0])
        sx, sy = w / w0, h / h0
        fx = load_focal(CALIB_JSON, w, h)
        print(f"{img_name}: {w}x{h}, fx={fx:.1f}")
        for m in models:
            cache = RES_DIR / "depth" / f"{m.name}__{Path(img_name).stem}.npy"
            if cache.exists():
                depth = np.load(cache)
            else:
                depth = m.predict(rgb, fx)
                np.save(cache, depth.astype(np.float32))
            for _, r in g.iterrows():
                records.append({
                    "model": m.name, "image": img_name, "marker_id": int(r["marker_id"]),
                    "x": r["x"] * sx, "y": r["y"] * sy,
                    "z_gt": r["z_geo_m"],
                    "z_tape": r["z_tape_m"] if pd.notna(r["z_tape_m"]) else np.nan,
                    "z_pred": sample_depth(depth, r["x"] * sx, r["y"] * sy),
                })
    return pd.DataFrame(records)


def main():
    if not GT_CSV.exists():
        raise SystemExit("gt_points.csv not found. Run first: python src\\validation\\ground_truth.py")
    gt = pd.read_csv(GT_CSV)
    if (gt["scale_src"] != "ruler").any():
        print("WARNING: the ground truth scale is not pinned by the ruler (see RULER_CM in ground_truth.py).\n")

    df = run_models(gt)
    df.to_csv(RES_DIR / "validation_points.csv", index=False)

    summary = evaluate(df)
    summary.to_csv(RES_DIR / "validation_summary.csv", index=False)
    plot_error_vs_distance(df, RES_DIR / "error_vs_distance.png")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    show = summary.copy()
    for c in ("AbsRel", "bias", "delta1", "AbsRel_lo", "AbsRel_hi"):
        show[c] = (show[c] * 100).round(1)
    show["RMSE_m"] = show["RMSE_m"].round(3)
    show = show.rename(columns={"AbsRel": "AbsRel,%", "delta1": "δ1,%", "bias": "зсув,%",
                                "AbsRel_lo": "CI_lo,%", "AbsRel_hi": "CI_hi,%"})
    print("\n" + show.to_string(index=False))

if __name__ == "__main__":
    main()