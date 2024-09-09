import cv2
import numpy as np
import pyrealsense2 as rs

# Step 1: Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Step 2: Configure the color stream (640x480 resolution, 30 FPS)
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

        # Step 5: Convert the frame to a numpy array
        img = np.asanyarray(color_frame.get_data())

        # Step 6: Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 7: Apply Otsu's thresholding to the grayscale image to create a binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 8: Create a binary mask
        mask = thresh
        
        mask = cv2.bitwise_not(mask)
        

        # Step 9: Invert the mask to create a mask for the background (optional, if needed)
        # mask_inv = cv2.bitwise_not(mask)

        # Step 10: Apply the mask to the original image to isolate the object
        result = cv2.bitwise_and(img, img, mask=mask)

        # Step 11: Display the original, thresholded, mask, and final result images
        cv2.imshow('RealSense Camera - Original RGB Image', img)
       
        cv2.imshow('Thresholded Image (Otsu)', thresh)
        cv2.imshow('Binary Mask', mask)
        cv2.imshow('Background Removed Image', result)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('X'):
            break

finally:
    # Step 12: Stop the pipeline and close the windows
    pipeline.stop()
    cv2.destroyAllWindows()
