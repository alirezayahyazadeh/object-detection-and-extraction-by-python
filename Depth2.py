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

# Distance threshold to segment object (adjust as needed)
min_object_distance_in_meters = 0.2  # Minimum distance from the camera (in meters)
max_object_distance_in_meters = 0.8  # Maximum distance from the camera (in meters)
min_depth_threshold = int(min_object_distance_in_meters * 1000)  # Convert to millimeters
max_depth_threshold = int(max_object_distance_in_meters * 1000)  # Convert to millimeters

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

        # Create a binary mask where depth values within the object's range are set to 255
        object_mask = np.where((depth_image > min_depth_threshold) & (depth_image < max_depth_threshold), 255, 0).astype(np.uint8)

        # Remove noise from the mask using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

        # Find connected components in the mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(object_mask, connectivity=8)

        # Filter the mask to keep only the largest connected component (likely the object)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Label 0 is the background
        largest_mask = np.zeros_like(object_mask)
        largest_mask[labels == largest_label] = 255

        # Apply the mask to the color image
        result = cv2.bitwise_and(color_image, color_image, mask=largest_mask)

        # Display the original color image, object mask, and result
        cv2.imshow('RealSense Camera - Original RGB Image', color_image)
        cv2.imshow('Object Mask', largest_mask)
        cv2.imshow('Object Segmented Image', result)

        # Break loop on 'X' key press
        if cv2.waitKey(1) & 0xFF == ord('X'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
