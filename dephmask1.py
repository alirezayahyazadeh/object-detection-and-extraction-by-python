import cv2
import numpy as np
import pyrealsense2 as rs

# Step 1: Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Step 2: Configure the color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Step 3: Start the pipeline
pipeline.start(config)

try:
    while True:
        # Step 4: Wait for frames and get the color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Check if the color frame is valid
        if not color_frame:
            continue

        # Convert the frame to a numpy array
        img = np.asanyarray(color_frame.get_data())

        # Step 5: Convert image to HSV color space for color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the color range for the object (adjust as needed)
        lower_color = np.array([5, 100, 100])  # Example values
        upper_color = np.array([15, 255, 255])

        # Step 6: Threshold the image to get only the object color
        color_mask = cv2.inRange(hsv, lower_color, upper_color)

        # Step 7: Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        # Step 8: Initialize the GrabCut mask
        mask = np.zeros(img.shape[:2], np.uint8)

        # Ensure that some pixels are definitively marked
        mask[color_mask == 255] = cv2.GC_PR_FGD  # Mark probable foreground
        mask[color_mask == 0] = cv2.GC_PR_BGD  # Mark probable background

        # Mark the central region as sure foreground to help GrabCut
        rect = (10, 10, img.shape[1] - 10, img.shape[0] - 10)

        # Step 9: Initialize background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Step 10: Run GrabCut algorithm
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Convert the GrabCut mask to binary (1 for foreground, 0 for background)
        final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

        # Step 11: Apply the mask to the original image
        result = cv2.bitwise_and(img, img, mask=final_mask)

        # Display the results
        cv2.imshow('Original RGB Image', img)
        cv2.imshow('Color Mask', color_mask)
        cv2.imshow('Final Mask after GrabCut', final_mask)
        cv2.imshow('Object Isolated', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Step 12: Stop the pipeline and close all windows
    pipeline.stop()
    cv2.destroyAllWindows()
