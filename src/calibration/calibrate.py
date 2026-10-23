import glob
import json
import os

import cv2
import numpy as np

IMG_DIR = "data/calibration/from_camera_check"
PATTERN = (7, 7)
SQUARE_MM = 52.0
LONG_SIDE = 960
OUT_JSON = "results/calib_iphone_960.json"

objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints, imgpoints, used = [], [], []
size = None

for p in sorted(glob.glob(os.path.join(IMG_DIR, "*.*"))):
    img = cv2.imread(p)
    if img is None:
        continue
    h, w = img.shape[:2]
    s = LONG_SIDE / max(h, w)
    new_size = (round(w * s), round(h * s))
    if size is None:
        size = new_size
    elif new_size != size:
        print(f"missed; another size of photo {new_size}): {os.path.basename(p)}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        print(f"angles not found: {os.path.basename(p)}")
        continue

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp)
    imgpoints.append((corners * s).astype(np.float32))
    used.append(os.path.basename(p))

print(f"\nphotos used: {len(used)}, size: {size}")
if len(used) < 10:
    raise SystemExit("Too few succeed photos (at least 10-15 needed")

rms, K, dist, rvecs, tvecs, std_int, std_ext, per_view = cv2.calibrateCameraExtended(
    objpoints, imgpoints, size, None, None, flags=cv2.CALIB_FIX_K3)
errs = per_view.ravel()

print(f"fx = {K[0, 0]:.1f} ± {std_int[0, 0]:.1f}   fy = {K[1, 1]:.1f} ± {std_int[1, 0]:.1f}")
print(f"cx = {K[0, 2]:.1f} ± {std_int[2, 0]:.1f}   cy = {K[1, 2]:.1f} ± {std_int[3, 0]:.1f}")
print("distortion:", np.round(dist.ravel(), 4))
print(f"reprojection RMS: {rms:.3f} px; through photos: average {errs.mean():.3f}, "
      f"The worst {errs.max():.3f} ({used[int(np.argmax(errs))]})")

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump({"size": size, "K": K.tolist(), "dist": dist.ravel().tolist(),
               "rms": rms, "images": used}, f, indent=2)
print("Saved:", OUT_JSON)