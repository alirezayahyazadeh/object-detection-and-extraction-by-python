import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Depth clipping distance (in meters)
clip_distance_in_meters = 2.0  # Adjust this value as needed

# Convert meters to depth units (depth values are in millimeters)
clip_distance = clip_distance_in_meters * 1000

try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Ensure both frames are valid
        if not color_frame or not depth_frame:
            continue

        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Clip depth image to the specified distance
        depth_clipped = np.clip(depth_image, 0, clip_distance)

        # Normalize depth image to 0-255 range for grayscale visualization
        depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Convert color image to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Thresholding to create a binary mask
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.bitwise_not(thresh)

        # Apply mask to color image to remove background
        result = cv2.bitwise_and(color_image, color_image, mask=mask)

        # Display the images
        cv2.imshow('RealSense Camera - Original RGB Image', color_image)
        cv2.imshow('Thresholded Image (Otsu)', thresh)
        cv2.imshow('Binary Mask', mask)
        cv2.imshow('Background Removed Image', result)
        cv2.imshow('Normalized Depth Image', depth_normalized)  # Depth image clipped and normalized

        # Break loop on 'X' key press
        if cv2.waitKey(1) & 0xFF == ord('X'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
