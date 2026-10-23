import cv2

IMG = "data/calibration/from_camera_check/photo_2026-10-22_23-08-17.jpg"
PATTERN = (7, 7)

img = cv2.imread(IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
found, corners = cv2.findChessboardCorners(
    gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
print("знайдено:", found)
if found:
    cv2.drawChessboardCorners(img, PATTERN, corners, found)
    cv2.imwrite("results/check_corners.jpg", img)
    print("збережено results/check_corners.jpg")