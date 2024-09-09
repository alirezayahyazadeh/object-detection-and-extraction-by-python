import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure depth stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Set depth clipping distance (in meters)
clipping_distance_in_meters = 2  # 2 meters
clipping_distance = clipping_distance_in_meters * 1000  # Convert to mm

try:
    while True:
        # Wait for a coherent pair of frames: depth
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalize the depth image to the range 0-255 for display
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Apply a Gaussian blur to smooth out the depth image and reduce noise
        depth_blurred = cv2.GaussianBlur(depth_normalized, (5, 5), 0)

        # Clip the depth image to the desired range (0 to clipping_distance)
        depth_clipped = np.where((depth_image > 0) & (depth_image < clipping_distance), 255, 0).astype(np.uint8)

        # Apply morphological operations to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        depth_cleaned = cv2.morphologyEx(depth_clipped, cv2.MORPH_CLOSE, kernel)
        depth_cleaned = cv2.morphologyEx(depth_cleaned, cv2.MORPH_OPEN, kernel)

        # Display the depth images
        cv2.imshow('Normalized Depth Image', depth_blurred)
        cv2.imshow('Clipped Depth Image', depth_cleaned)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
