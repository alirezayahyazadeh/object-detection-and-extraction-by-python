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

# Capture the background depth frame (without the subject)
print("Capturing background image...")
for i in range(10):
    frames = pipeline.wait_for_frames()  # Allow some frames to stabilize

background_frames = pipeline.wait_for_frames()
background_depth_frame = background_frames.get_depth_frame()
if not background_depth_frame:
    raise Exception("Failed to capture background depth frame")
background_depth_image = np.asanyarray(background_depth_frame.get_data())

# Capture the depth frame with the subject
print("Capturing subject image...")
for i in range(10):
    frames = pipeline.wait_for_frames()  # Allow some frames to stabilize

subject_frames = pipeline.wait_for_frames()
subject_depth_frame = subject_frames.get_depth_frame()
subject_color_frame = subject_frames.get_color_frame()
if not subject_depth_frame or not subject_color_frame:
    raise Exception("Failed to capture subject depth frame")

subject_depth_image = np.asanyarray(subject_depth_frame.get_data())
subject_color_image = np.asanyarray(subject_color_frame.get_data())

# Stop streaming after capturing the frames
pipeline.stop()

# Step 1: Subtract background from subject depth image
depth_diff = cv2.absdiff(subject_depth_image, background_depth_image)

# Step 2: Apply thresholding to isolate the subject (you can adjust the threshold value)
_, mask = cv2.threshold(depth_diff, 50, 255, cv2.THRESH_BINARY)

# Convert the mask to 8-bit (CV_8UC1) for contour detection
mask_8bit = cv2.convertScaleAbs(mask)

# Step 3: Apply median filtering to reduce noise
mask_8bit = cv2.medianBlur(mask_8bit, 5)

# Step 4: Find contours and filter to keep only the largest contour (subject)
contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)
    mask_8bit = np.zeros_like(mask_8bit)
    cv2.drawContours(mask_8bit, [largest_contour], -1, 255, thickness=cv2.FILLED)

# Step 5: Use the mask to extract the subject from the color image
extracted_subject = cv2.bitwise_and(subject_color_image, subject_color_image, mask=mask_8bit)

# Step 6: Display the results
cv2.imshow('Background Depth Image', background_depth_image)
cv2.imshow('Subject Depth Image', subject_depth_image)
cv2.imshow('Depth Difference', depth_diff)
cv2.imshow('Mask after Median Filtering', mask_8bit)
cv2.imshow('Extracted Subject', extracted_subject)

# Save the result
cv2.imwrite('extracted_subject.png', extracted_subject)

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
