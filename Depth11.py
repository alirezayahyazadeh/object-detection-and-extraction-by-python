import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable streams for both color and depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Create depth and color aligner
align_to = rs.stream.color
align = rs.align(align_to)

# Set depth scale (used for distance measurement)
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Set distance for depth clipping (e.g., 1 meter)
target_distance = 1.0  # 1 meter
depth_threshold = int(target_distance / depth_scale)

# Capture a single image
try:
    # Wait for frames
    frames = pipeline.wait_for_frames()

    # Align depth frame to color frame
    aligned_frames = align.process(frames)

    # Get both aligned color and depth frames
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        raise Exception("Failed to capture frames")

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Create a mask based on the depth threshold (around 1 meter)
    mask = np.where((depth_image > 0) & (depth_image < depth_threshold), 255, 0).astype(np.uint8)

    # Convert to 3 channels for masking with color image
    mask_3d = np.stack((mask, mask, mask), axis=-1)

    # Remove background by applying the mask
    extracted_object = cv2.bitwise_and(color_image, mask_3d)

    # Display the extracted object
    cv2.imshow('Extracted Object', extracted_object)

    # Wait until a key is pressed
    cv2.waitKey(0)

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()