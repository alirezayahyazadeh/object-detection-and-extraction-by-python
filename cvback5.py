import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

try:
    while True:
        frameset = pipeline.wait_for_frames()
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_threshold = 1.0

        depth_mask = cv2.inRange(depth_image, 0, depth_threshold * 1000)

        foreground = cv2.bitwise_and(color_image, color_image, mask=depth_mask)

        background_mask = cv2.bitwise_not(depth_mask)

        background_replaced = cv2.bitwise_and(color_image, color_image, mask=background_mask)
        background_replaced[depth_mask == 255] = [5, 5, 10]

        final_image = cv2.add(foreground, background_replaced)

        cv2.imshow("Original RGB Image", color_image)
        cv2.imshow("Depth Image", depth_image)
        cv2.imshow("Foreground", foreground)
        cv2.imshow("Final Image", final_image)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
