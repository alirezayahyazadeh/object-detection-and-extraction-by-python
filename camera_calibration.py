import numpy as np
import cv2
import glob

# تنظیمات چک‌بورد
chessboard_size = (9, 6)  # تعداد خانه‌های چک‌بورد (عرض, ارتفاع)
square_size = 0.025  # اندازه هر مربع در متر

# ایجاد نقاط دنیای واقعی (3D) برای چک‌بورد
obj_points = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
obj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
obj_points *= square_size

# لیست برای ذخیره نقاط چک‌بورد واقعی و تصویر
object_points = []
image_points = []

# بارگذاری تصاویر چک‌بورد
images = glob.glob('calibration_images/*.jpg')  # مسیر تصاویر کالیبراسیون

for image_path in images:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to load image: {image_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # پیدا کردن نقاط چک‌بورد
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        print(f"Chessboard corners found in image: {image_path}")
        object_points.append(obj_points)
        image_points.append(corners)

        # رسم نقاط چک‌بورد در تصویر
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)
    else:
        print(f"No chessboard corners found in image: {image_path}")

cv2.destroyAllWindows()

# بررسی اینکه نقاط کافی برای کالیبراسیون یافت شده‌اند
if not object_points or not image_points:
    print("Object points collected:", len(object_points))
    print("Image points collected:", len(image_points))
    raise ValueError("No points found for calibration. Make sure you have enough calibration images with visible chessboard corners.")

# کالیبره کردن دوربین
if not gray.size:
    raise RuntimeError("Gray image is not initialized.")

h, w = gray.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (w, h), None, None)

# ذخیره کردن نتایج کالیبراسیون
np.savez('camera_calibration.npz', camera_matrix=mtx, dist_coeffs=dist)

print("کالیبراسیون با موفقیت انجام شد.")