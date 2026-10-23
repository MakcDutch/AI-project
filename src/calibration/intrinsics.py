import json


def load_focal(json_path, img_w, img_h):
    with open(json_path) as f:
        d = json.load(f)
    K = d["K"]
    cal_w, cal_h = d["size"]
    f_cal = 0.5 * (K[0][0] + K[1][1])
    scale = max(img_w, img_h) / max(cal_w, cal_h)
    return f_cal * scale