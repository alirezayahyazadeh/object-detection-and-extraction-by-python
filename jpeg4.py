import cv2
import numpy as np
import pyrealsense2 as rs

ca = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

ca.start(config)

try:
    while True:
        frames = ca.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        hh, ww = color_image.shape[:2]

        lower = np.array([200, 200, 200])
        upper = np.array([255, 255, 255])

        thresh = cv2.inRange(color_image, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros((hh, ww), dtype=np.uint8)
        for cntr in contours:
            cv2.drawContours(mask, [cntr], 0, (255, 255, 255), -1)

        points = np.column_stack(np.where(thresh.transpose() > 0))

        if len(points) >= 15:
            hullpts = cv2.convexHull(points)
            ((centx, centy), (width, height), angle) = cv2.fitEllipse(hullpts)
            print("center x,y:", centx, centy)
            print("diameters:", width, height)
            print("orientation angle:", angle)

            hull = color_image.copy()
            cv2.polylines(hull, [hullpts], True, (100, 100, 255), 1)

            circle = np.zeros((hh, ww), dtype=np.uint8)
            cx = int(centx)
            cy = int(centy)
            radius = (width + height) / 4
            cv2.circle(circle, (cx, cy), int(radius), 255, -1)


            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
            circle = cv2.morphologyEx(circle, cv2.MORPH_ERODE, kernel)

            mask2 = cv2.bitwise_and(255-morph, 255-morph, mask=circle)
            result = cv2.bitwise_and(color_image, color_image, mask=mask2)

            cv2.imshow('Result', result)
            cv2.imshow('mask2',mask2)

        else:
            print("Not enough points to fit an ellipse.")
            cv2.imshow('Original Image', color_image)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    ca.stop()
    cv2.destroyAllWindows()
