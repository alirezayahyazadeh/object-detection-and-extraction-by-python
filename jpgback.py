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
        frameset = ca.wait_for_frames(timeout_ms=10000)

        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        if not color_frame or not depth_frame:
            print("No color or depth frame captured.")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalize the depth image for better visualization
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_normalized = np.uint8(depth_image_normalized)

        cv2.imshow("Depth Image (Normalized)", depth_image_normalized)

        depth_threshold = 1.0  # Adjust this value depending on your scene

        depth_mask = np.where((depth_image > 0) & (depth_image < depth_threshold * 1000), 255, 0).astype(np.uint8)

        # Visualize the depth mask
        cv2.imshow("Depth Mask", depth_mask)

        foreground = cv2.bitwise_and(color_image, color_image, mask=depth_mask)

        background_mask = cv2.bitwise_not(depth_mask)
        black_background = np.zeros_like(color_image)

        final_image = cv2.bitwise_or(foreground, black_background, mask=depth_mask)

        cv2.imshow("Foreground", foreground)
        cv2.imshow("Final Image", final_image)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

finally:
    ca.stop()
    cv2.destroyAllWindows()
