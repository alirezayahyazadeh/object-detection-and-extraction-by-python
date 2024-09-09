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

# Capture a single frame (color + depth) and stop streaming
print("Capturing image...")

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

# Save the color image and depth image for reference
cv2.imwrite('color_image.png', color_image)
np.save('depth_image.npy', depth_image)  # Save depth image in numpy format

# Stop streaming after capturing the frame
pipeline.stop()

print("Image captured successfully. Processing...")

# Step 1: Convert depth values to meters
depth_in_meters = depth_image * depth_scale

# Print the minimum and maximum depth values for debugging
min_depth = np.min(depth_in_meters)
max_depth = np.max(depth_in_meters)
print(f"Min Depth: {min_depth} meters, Max Depth: {max_depth} meters")

# Print the depth value at the center of the image for debugging
center_x = depth_image.shape[1] // 2  # Middle of the width
center_y = depth_image.shape[0] // 2  # Middle of the height
center_depth_value = depth_in_meters[center_y, center_x]
print(f"Center Depth Value: {center_depth_value} meters")

# Step 2: Set depth range to isolate the object (manual tuning based on the printed values)
depth_threshold_near = 0.9  # Near threshold (set to 0.9 meters)
depth_threshold_far = 1.1   # Far threshold (set to 1.1 meters)

# Create a mask based on the current depth thresholds
mask = np.where((depth_in_meters > depth_threshold_near) & 
                (depth_in_meters < depth_threshold_far), 2, 250).astype(np.uint8)

# Apply morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)  # Remove noise

# Step 3: Resize the mask to match the color image dimensions
mask_resized = cv2.resize(mask_cleaned, (color_image.shape[1], color_image.shape[0]))

# Step 4: Convert the resized mask to 3 channels to match the color image format
mask_3d = np.stack((mask_resized, mask_resized, mask_resized), axis=-1)

# Step 5: Multiply the mask by the original color image to isolate the object
object_only = cv2.bitwise_and(color_image, mask_3d)

# Step 6: Further increase brightness and contrast for better visibility
alpha = 3.0  # Contrast control (increased to 3.0 for more contrast)
beta = 80    # Brightness control (increased to 80 for more brightness)
enhanced_object = cv2.convertScaleAbs(object_only, alpha=alpha, beta=beta)

# Set the background to black by masking the inverse of the object
background_black = cv2.addWeighted(enhanced_object, 1, np.zeros_like(color_image), 0, 0)

# Save the final isolated object image
cv2.imwrite('object_isolated_enhanced.png', background_black)

# Display the final images
cv2.imshow('Color Image', color_image)
cv2.imshow('Mask', mask_cleaned)
cv2.imshow('Object Isolated', background_black)

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
