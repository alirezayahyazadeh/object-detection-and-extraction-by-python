import cv2
import numpy as np
import pyrealsense2 as rs

ca = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

ca.start(config)

try:
    while True:
        frames = ca.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask = thresh
        
        mask = cv2.bitwise_not(mask)

        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('RealSense Camera - Original RGB Image', img)
        cv2.imshow('Thresholded Image (Otsu)', thresh)
        cv2.imshow('Binary Mask', mask)
        cv2.imshow('Background Removed Image', result)

        if cv2.waitKey(1) & 0xFF == ord('X'):
            break

finally:
    ca.stop()
    cv2.destroyAllWindows()
