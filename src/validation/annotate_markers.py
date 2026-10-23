import csv
import os
import sys

import cv2

IMG_DIR = "data/validation"
FRAMES_CSV = os.path.join(IMG_DIR, "frames.csv")
OUT_CSV = os.path.join(IMG_DIR, "markers.csv")
N_MARKERS = 8
MAX_W, MAX_H = 1400, 900
FIELDS = ["image", "marker_id", "x", "y", "w", "h"]
REDO = "--redo" in sys.argv


def read_frames():
    with open(FRAMES_CSV, newline="", encoding="utf-8") as f:
        return [row["image"].strip() for row in csv.DictReader(f)]


def read_markers():
    if not os.path.exists(OUT_CSV):
        return []
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_markers(rows):
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def annotate(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not open {path}")
        return None
    h, w = img.shape[:2]
    scale = min(1.0, MAX_W / w, MAX_H / h)
    shown = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img.copy()

    points = []
    win = "markers"

    def redraw():
        canvas = shown.copy()
        for mid, x, y in points:
            if x is None:
                continue
            cx, cy = int(x * scale), int(y * scale)
            cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(canvas, str(mid), (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
        nxt = len(points) + 1
        status = (f"{os.path.basename(path)} | click on the marker {nxt}/{N_MARKERS}"
                  if nxt <= N_MARKERS else
                  f"{os.path.basename(path)} | all {N_MARKERS} are ready - Enter to save")
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(canvas, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, canvas)

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < N_MARKERS:
            points.append((len(points) + 1, mx / scale, my / scale))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("u"), 8) and points:
            points.pop()
            redraw()
        elif key == 32 and len(points) < N_MARKERS:
            points.append((len(points) + 1, None, None))
            redraw()
        elif key in (13, 10):  # Enter
            if len(points) == N_MARKERS:
                break
            print(f"Marked {len(points)} з {N_MARKERS}, more needed (or space to miss)")
        elif key == ord("n"):
            cv2.destroyWindow(win)
            return None
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            raise KeyboardInterrupt

    cv2.destroyWindow(win)
    return [(mid, x, y, w, h) for mid, x, y in points if x is not None]


def main():
    frames = read_frames()
    rows = read_markers()
    done = {r["image"] for r in rows}

    try:
        for name in frames:
            if name in done and not REDO:
                print(f"{name}: already marked (run with --redo to repeat)")
                continue
            path = os.path.join(IMG_DIR, name)
            print(f"{name}: marker...")
            result = annotate(path)
            if result is None:
                print("  missed")
                continue
            rows = [r for r in rows if r["image"] != name]
            for mid, x, y, w, h in result:
                rows.append({"image": name, "marker_id": mid, "x": f"{x:.1f}", "y": f"{y:.1f}",
                             "w": w, "h": h})
            write_markers(rows)
            print(f" Saved {len(result)} points")
    except KeyboardInterrupt:
        print("suspended, saved is in", OUT_CSV)
    cv2.destroyAllWindows()
    print("Ready:", OUT_CSV)


if __name__ == "__main__":
    main()