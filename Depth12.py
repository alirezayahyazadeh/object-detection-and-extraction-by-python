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
target_distance = 1.0  # 1 meter distance for the object
depth_threshold_near = int((target_distance - 0.3) / depth_scale)  # Set near depth threshold (30 cm tolerance)
depth_threshold_far = int((target_distance + 0.3) / depth_scale)   # Set far depth threshold (30 cm tolerance)

# Apply filters for better depth quality
decimation = rs.decimation_filter()  # Reduce depth map resolution
spatial = rs.spatial_filter()        # Edge-preserving spatial smoothing
temporal = rs.temporal_filter()      # Temporal smoothing

# Capture a single frame (color + depth)
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

    # Apply depth filters
    depth_frame = decimation.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Resize depth image to match color image if needed
    depth_image_resized = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]))

    # Create a binary mask based on the depth threshold (object within the range)
    mask = np.where((depth_image_resized > depth_threshold_near) & (depth_image_resized < depth_threshold_far), 255, 0).astype(np.uint8)

    # Apply GaussianBlur to smooth the mask and remove noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Morphology to clean up the mask (closing small holes, smoothing edges)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Convert to 3 channels to apply to the color image
    mask_3d = np.stack((mask, mask, mask), axis=-1)

    # Enhance the visibility of the object with contrast adjustment
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 40    # Brightness control (0-100)
    contrast_image = cv2.convertScaleAbs(color_image, alpha=alpha, beta=beta)

    # Apply the mask to the color image (extract the object)
    object_only = cv2.bitwise_and(contrast_image, mask_3d)

    # Set background pixels to black for a clean object extraction
    background_removed = cv2.addWeighted(object_only, 1, np.zeros_like(color_image), 0, 0)

    # Save and display the final image with the background removed
    cv2.imwrite('background_removed_image.png', background_removed)
    cv2.imshow('Background Removed Image', background_removed)

    # Wait until a key is pressed
    cv2.waitKey(0)

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
