import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize the RealSense pipeline
ca = rs.pipeline()
config = rs.config()

# Configure only the depth stream (no RGB stream)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
ca.start(config)

# Set the depth clipping distance to 2 meters (adjustable to 1 or 2 meters)
depth_clipping_distance = 2.0  # Change to 1.0 or 2.0 meters based on your requirement
depth_scale = 1000  # Convert meters to millimeters for the depth sensor

try:
    while True:
        # Wait for a coherent set of frames (depth frame)
        frames = ca.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # Convert the depth frame to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Clip the depth values to the set distance (1 or 2 meters)
        depth_clipped = np.where((depth_image > 0) & (depth_image < depth_clipping_distance * depth_scale), 255, 0).astype(np.uint8)

        # Apply Gaussian blur to reduce noise
        depth_clipped_blurred = cv2.GaussianBlur(depth_clipped, (5, 5), 0)

        # Apply morphological operations (erosion followed by dilation)
        kernel = np.ones((3, 3), np.uint8)
        depth_cleaned = cv2.morphologyEx(depth_clipped_blurred, cv2.MORPH_OPEN, kernel)
        depth_cleaned = cv2.morphologyEx(depth_cleaned, cv2.MORPH_CLOSE, kernel)

        # Normalize the depth data for better visualization (0 to 255 grayscale)
        depth_normalized = cv2.normalize(depth_cleaned, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Display the cleaned and normalized depth images
        cv2.imshow("Clipped Depth Image (Cleaned)", depth_cleaned)
        cv2.imshow("Normalized Depth Image", depth_normalized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and close windows
    ca.stop()
    cv2.destroyAllWindows()
