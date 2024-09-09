import cv2
import numpy as np
import pyrealsense2 as rs

ca = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

ca.start(cfg)

try:
    while True:
        frames = ca.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        background_mask = cv2.bitwise_not(mask)
        object_extracted = cv2.bitwise_and(color_image, color_image, mask=mask)
        background_replaced = cv2.bitwise_and(color_image, color_image, mask=background_mask)
        background_replaced += np.ones_like(color_image) * 255

        final_image = object_extracted + background_replaced

        cv2.imshow("Original RGB Image", color_image)
        cv2.imshow("Edges", edges)
        cv2.imshow("Extracted Object", object_extracted)
        cv2.imshow("Final Image", final_image)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

finally:
    ca.stop()
    cv2.destroyAllWindows()
