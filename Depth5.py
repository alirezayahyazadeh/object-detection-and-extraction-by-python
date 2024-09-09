import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize the Intel RealSense camera pipeline
ca = rs.pipeline()
config = rs.config()

# Configure the streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
ca.start(config)

# Set the depth clipping distance to 1 meter for close objects (adjustable to 2 meters)
depth_clipping_distance = 1.0  # Change to 1.0 or 2.0 meters based on the desired distance
depth_scale = 1000  # Convert meters to millimeters for the depth sensor

try:
    while True:
        # Wait for frames from the camera
        frames = ca.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Clip the depth array to a fixed distance (1 meter or 2 meters)
        depth_mask = np.where((depth_image > 0) & (depth_image < depth_clipping_distance * depth_scale), 255, 0).astype(np.uint8)

        # Normalize the depth array to a stable gray image (0 to 255)
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Apply morphological operations to clean the mask (closing holes and smoothing edges)
        kernel = np.ones((5, 5), np.uint8)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)

        # Invert the depth mask to focus on the object
        mask_inverted = cv2.bitwise_not(depth_mask)

        # Extract the object using the mask
        masked_image = cv2.bitwise_and(color_image, color_image, mask=depth_mask)

        # Create a completely black background
        background = np.zeros_like(color_image)

        # Combine the extracted object with the black background
        final_image = cv2.add(masked_image, background)

        # Display the images
        cv2.imshow("Normalized Depth Image", depth_normalized)  # Grayscale normalized depth image
        cv2.imshow("Depth Mask", depth_mask)
        cv2.imshow("Original RGB Image", color_image)
        cv2.imshow("Masked (Object Isolated) Image", masked_image)
        cv2.imshow("Background Removed Image", final_image)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the camera pipeline
    ca.stop()
    cv2.destroyAllWindows()
