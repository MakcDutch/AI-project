import glob
import os

import cv2
import numpy as np

IMG_DIR = "data/calibration/from_camera_check"
PAT = (7, 7)

for p in sorted(glob.glob(os.path.join(IMG_DIR, "*.*"))):
    img = cv2.imread(p)
    if img is None:
        continue
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f, c = cv2.findChessboardCorners(
        gray, PAT, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    name = os.path.basename(p)
    if not f:
        print(f"{name}: angles not found")
        continue
    c = c.reshape(PAT[1], PAT[0], 2)
    step = lambda a: np.mean(np.linalg.norm(np.diff(a, axis=0), axis=1))
    r1 = max(step(c[0]), step(c[-1])) / min(step(c[0]), step(c[-1]))
    r2 = max(step(c[:, 0]), step(c[:, -1])) / min(step(c[:, 0]), step(c[:, -1]))
    cx, cy = c[..., 0].mean() / w, c[..., 1].mean() / h
    print(f"{name}: titl {max(r1, r2):.2f}, centre of mesh ({cx:.2f}, {cy:.2f})")