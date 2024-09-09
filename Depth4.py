import cv2
import numpy as np
import pyrealsense2 as rs

ca = rs.pipeline()
config = rs.config()

# Enable depth and color streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
ca.start(config)

# Define the clipping distance in meters (0.7 meters for closer object detection)
clipping_distance_meters = 0.7
clipping_distance = clipping_distance_meters * 1000  # Convert to millimeters

try:
    while True:
        # Get frames from RealSense
        frames = ca.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Create a depth mask: object closer than clipping distance
        depth_mask = np.where((depth_image > 0) & (depth_image < clipping_distance), 255, 0).astype(np.uint8)

        # Refine mask using morphological operations
        kernel = np.ones((7, 7), np.uint8)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)

        # Find contours to isolate the object
        contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by contour area to remove small regions (background noise)
        mask = np.zeros_like(depth_mask)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # You can adjust the area threshold
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Apply the refined mask to the color image
        foreground = cv2.bitwise_and(color_image, color_image, mask=mask)

        # Create a black background
        background_mask = cv2.bitwise_not(mask)
        black_background = np.zeros_like(color_image)

        # Combine the foreground (object) with the black background
        final_image = cv2.bitwise_or(foreground, black_background)

        # Display the images
        cv2.imshow("Original RGB Image", color_image)
        cv2.imshow("Depth Mask", mask)
        cv2.imshow("Masked (Object Isolated) Image", final_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    ca.stop()
    cv2.destroyAllWindows()
