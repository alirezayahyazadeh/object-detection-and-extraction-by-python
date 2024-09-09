import cv2
import numpy as np
import pyrealsense2 as r

ca = r.pipeline()
config = r.config()

config.enable_stream(r.stream.color, 640, 480, r.format.bgr8, 30)
config.enable_stream(r.stream.depth, 640, 480, r.format.z16, 30)

ca.start(config)

try:
    while True:
        frameset = ca.wait_for_frames(timeout_ms=10000)

        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        if not color_frame or not depth_frame:
            print("No color or depth frame captured.")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        cv2.imshow("Original RGB Image", color_image)

        depth_threshold = 1.0

        depth_mask = np.where((depth_image > 0) & (depth_image < depth_threshold * 1000), 255, 0).astype(np.uint8)

        foreground = cv2.bitwise_and(color_image, color_image, mask=depth_mask)

        background_mask = cv2.bitwise_not(depth_mask)
        black_background = np.zeros_like(color_image)

        final_image = cv2.bitwise_or(foreground, black_background, mask=depth_mask)

        cv2.imshow("Depth Image", depth_image)
        cv2.imshow("Foreground", foreground)
        cv2.imshow("Final Image", final_image)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

finally:
    ca.stop()
    cv2.destroyAllWindows()
