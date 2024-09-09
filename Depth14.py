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

    # Manually set the depth thresholds (tightened range)
    center_depth_value = depth_image_resized[240, 320]
    print(f"Center Depth Value: {center_depth_value}")

    # Narrow depth range aggressively
    depth_threshold_near = center_depth_value - 20  # Nearer depth threshold
    depth_threshold_far = center_depth_value + 20   # Farther depth threshold

    print(f"Depth Thresholds: Near={depth_threshold_near}, Far={depth_threshold_far}")

    # Create a binary mask based on the depth threshold
    mask = np.where((depth_image_resized > depth_threshold_near) & 
                    (depth_image_resized < depth_threshold_far), 120, 250).astype(np.uint8)

    # Apply GaussianBlur to smooth the mask and remove noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Morphology to clean up the mask (closing small holes, smoothing edges)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise

    # Edge detection to refine object boundaries
    edges = cv2.Canny(mask, 100, 200)
    mask_with_edges = cv2.add(mask, edges)

    # Refine the mask by applying the edges back to the mask
    mask_refined = cv2.morphologyEx(mask_with_edges, cv2.MORPH_CLOSE, kernel)

    # Convert to 3 channels to apply to the color image
    mask_3d = np.stack((mask_refined, mask_refined, mask_refined), axis=-1)

    # Apply CLAHE for better contrast
    lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Apply the mask to the color image (extract the object)
    object_only = cv2.bitwise_and(enhanced_image, mask_3d)

    # Increase brightness and contrast on the masked object
    alpha = 3.0  # Strong contrast control
    beta = 100   # Higher brightness control
    object_only = cv2.convertScaleAbs(object_only, alpha=alpha, beta=beta)

    # Set background pixels to black for a clean object extraction
    background_removed = cv2.addWeighted(object_only, 1, np.zeros_like(color_image), 0, 0)

    # Save and display the final image with the background removed
    cv2.imwrite('background_removed_image.png', background_removed)
    cv2.imshow('Background Removed Image', background_removed)

    # Visualize the depth map
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_resized, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imshow('Depth Map', depth_colormap)

    # Wait until a key is pressed
    cv2.waitKey(0)

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
