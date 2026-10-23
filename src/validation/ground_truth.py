import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
VAL_DIR = ROOT / "data" / "validation"
CALIB_JSON = ROOT / "results" / "calib_iphone_960.json"
MARKERS_CSV = VAL_DIR / "markers.csv"
FRAMES_CSV = VAL_DIR / "frames.csv"
OUT_CSV = VAL_DIR / "gt_points.csv"

RULER_CM = {(1, 4): 102, (5, 8): 98, (1, 5): 52}

USE_DISTORTION = False
RESID_WARN_PX = 4.0
MIN_MARKERS = 5


def load_markers():
    frames = {}
    with open(MARKERS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            fr = frames.setdefault(r["image"], {"wh": (int(r["w"]), int(r["h"])), "pts": {}})
            fr["pts"][int(r["marker_id"])] = (float(r["x"]), float(r["y"]))
    return frames


def load_tape():
    tape = {}
    if FRAMES_CSV.exists():
        with open(FRAMES_CSV, newline="") as f:
            for r in csv.DictReader(f):
                tape[r["image"]] = float(r["z_wall_m"])
    return tape


def camera_for_size(w, h):
    with open(CALIB_JSON) as f:
        d = json.load(f)
    cw, ch = d["size"]
    if (w < h) != (cw < ch):
        raise SystemExit("Orientation of photos (vertical/gorisontal) is different from calibration.")
    s = max(w, h) / max(cw, ch)
    K = np.array(d["K"], dtype=np.float64)
    K[:2, :] *= s
    return K, np.array(d["dist"], dtype=np.float64)


def project(K, P, pose):
    R, _ = cv2.Rodrigues(pose[:3])
    X = (R @ np.c_[P, np.zeros(len(P))].T).T + pose[3:]
    uv = np.c_[K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2], K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]]
    return uv, X[:, 2]


def main():
    frames = load_markers()
    tape = load_tape()

    names = [n for n, fr in frames.items() if len(fr["pts"]) >= MIN_MARKERS]
    for n in frames:
        if n not in names:
            print(f"Missed {n}: less {MIN_MARKERS} markers")
    if not names:
        raise SystemExit("No photos")

    w, h = frames[names[0]]["wh"]
    for n in names:
        if frames[n]["wh"] != (w, h):
            raise SystemExit(f"{n}: size {frames[n]['wh']} differ from {(w, h)}")
    K, dist = camera_for_size(w, h)

    ids = sorted({i for n in names for i in frames[n]["pts"]})
    if 1 not in ids or 2 not in ids:
        raise SystemExit("Markers 1 and 2 needed (they fix coordinate system).")
    free_ids = [i for i in ids if i not in (1, 2)]
    idx = {i: k for k, i in enumerate(ids)}

    obs = {}
    for n in names:
        mid = sorted(frames[n]["pts"])
        pts = np.array([frames[n]["pts"][i] for i in mid], dtype=np.float64)
        if USE_DISTORTION:
            pts = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        obs[n] = (mid, pts)

    ref = max(names, key=lambda n: (len(obs[n][0]), tape.get(n, 0)))
    mid, pts = obs[ref]
    if len(mid) < len(ids):
        raise SystemExit("At least 1 photo needed, where all markers are placed")
    z_ref = tape.get(ref, 3.0)
    P_init = np.zeros((len(ids), 2))
    for i, p in zip(mid, pts):
        P_init[idx[i]] = [(p[0] - K[0, 2]) * z_ref / K[0, 0], (p[1] - K[1, 2]) * z_ref / K[1, 1]]
    SC = 0.35
    v = P_init[idx[2]] - P_init[idx[1]]
    f = SC / complex(v[0], v[1])
    for k in range(len(ids)):
        c = (complex(*(P_init[k] - P_init[idx[1]]))) * f
        P_init[k] = [c.real, c.imag]

    poses0 = []
    for n in names:
        mid, pts = obs[n]
        obj = np.c_[P_init[[idx[i] for i in mid]], np.zeros(len(mid))]
        ok, rv, tv = cv2.solvePnP(obj, pts, K, None)
        if not ok:
            raise SystemExit(f"solvePnP не спрацював для {n}")
        poses0.append(np.r_[rv.ravel(), tv.ravel()])
    poses0 = np.array(poses0)

    def unpack(x):
        P = np.zeros((len(ids), 2))
        P[idx[2]] = [SC, 0.0]
        for j, i in enumerate(free_ids):
            P[idx[i]] = x[2 * j: 2 * j + 2]
        return P, x[2 * len(free_ids):].reshape(len(names), 6)

    def resid(x):
        P, poses = unpack(x)
        out = []
        for n, pose in zip(names, poses):
            mid, pts = obs[n]
            uv, _ = project(K, P[[idx[i] for i in mid]], pose)
            out.append((uv - pts).ravel())
        return np.concatenate(out)

    x0 = np.r_[np.array([P_init[idx[i]] for i in free_ids]).ravel(), poses0.ravel()]
    sol = least_squares(resid, x0, method="trf", loss="soft_l1", f_scale=2.0)
    P, poses = unpack(sol.x)

    z_unit, res_px = {}, {}
    for n, pose in zip(names, poses):
        mid, pts = obs[n]
        uv, z = project(K, P[[idx[i] for i in mid]], pose)
        for i, zi, e in zip(mid, z, np.linalg.norm(uv - pts, axis=1)):
            z_unit[(n, i)] = zi
            res_px[(n, i)] = e
    all_res = np.array(list(res_px.values()))
    print(f"Photos: {len(names)}, points: {len(all_res)}, size {w}x{h}")
    print(f"Reproection error: RMS {np.sqrt(np.mean(all_res**2)):.2f} px, "
          f"max {all_res.max():.1f} px")
    for (n, i), e in sorted(res_px.items()):
        if e > RESID_WARN_PX:
            print(f"Bad click: {n}, marker {i}, error {e:.1f} px")

    ks = []
    for (a, b), cm in RULER_CM.items():
        if cm is None or a not in idx or b not in idx:
            continue
        d_unit = np.linalg.norm(P[idx[a]] - P[idx[b]])
        ks.append(cm / 100.0 / d_unit)
        print(f"  Line {a}->{b}: {cm:.1f} sm, k={ks[-1]:.4f}")
    if ks:
        k = float(np.median(ks))
        spread = (max(ks) - min(ks)) / k
        print(f"Line: k={k:.4f} (spread: {spread:.1%})")
        if spread > 0.02:
            print("Spread >2%")
        scale_src = "ruler"
    else:
        ratios = []
        for n in names:
            zs = [z_unit[(n, i)] for i in obs[n][0]]
            if n in tape:
                ratios.append(tape[n] / np.mean(zs))
        if not ratios:
            raise SystemExit("No measures")
        k = float(np.mean(ratios))
        scale_src = "tape_mean"

    print("\nphoto            | line, м | z from geometry, м | line vs geometry | up/down of markers")
    for n in names:
        mid = obs[n][0]
        zg = np.array([z_unit[(n, i)] * k for i in mid])
        top = [z_unit[(n, i)] * k for i in mid if i <= 4]
        bot = [z_unit[(n, i)] * k for i in mid if i >= 5]
        zt = tape.get(n)
        diff = f"{100 * (zt / zg.mean() - 1):+5.1f}%" if zt else "   n/a"
        tb = f"{np.mean(top):.3f}/{np.mean(bot):.3f}" if top and bot else "n/a"
        print(f"{n:15s} | {zt if zt else float('nan'):10.2f} | {zg.mean():16.3f} | {diff:>20s} | {tb}")

    for i in ids:
        print(f"  {i}: x={P[idx[i]][0] * k * 100:6.1f}  y={P[idx[i]][1] * k * 100:6.1f}")

    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "marker_id", "x", "y", "w", "h", "z_geo_m", "z_tape_m", "resid_px", "scale_src"])
        for n in names:
            for i in obs[n][0]:
                x, y = frames[n]["pts"][i]
                wr.writerow([n, i, f"{x:.1f}", f"{y:.1f}", w, h,
                             f"{z_unit[(n, i)] * k:.4f}", tape.get(n, ""), f"{res_px[(n, i)]:.2f}", scale_src])
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()