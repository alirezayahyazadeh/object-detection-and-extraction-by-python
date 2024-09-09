import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize the RealSense pipeline
ca = rs.pipeline()
config = rs.config()

# Configure the color and depth streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
ca.start(config)

try:
    while True:
        # Wait for a coherent set of frames
        frames = ca.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert the images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Clip the depth values to a maximum of 1 meter (1000 mm)
        depth_image_clipped = np.clip(depth_image, 0, 1000)

        # Normalize the depth image to a grayscale image
        depth_image_normalized = cv2.normalize(depth_image_clipped, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_normalized = np.uint8(depth_image_normalized)

        # Threshold the depth image to create a binary mask
        _, depth_mask = cv2.threshold(depth_image_normalized, 50, 255, cv2.THRESH_BINARY)

        # Invert the mask to get the object in focus
        depth_mask_inv = cv2.bitwise_not(depth_mask)

        # Apply the mask to the color image to extract the object
        foreground = cv2.bitwise_and(color_image, color_image, mask=depth_mask)

        # Display the results
        cv2.imshow('Original RGB Image', color_image)
        cv2.imshow('Depth Image (Clipped and Normalized)', depth_image_normalized)
        cv2.imshow('Foreground', foreground)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and close windows
    ca.stop()
    cv2.destroyAllWindows()
