import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize the Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

try:
    while True:
        # Wait for a coherent set of frames
        frameset = pipeline.wait_for_frames()
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        if not color_frame or not depth_frame:
            print("No frames captured.")
            continue

        # Convert the frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Define a depth threshold range in meters
        depth_threshold = 1.0  # 1 meter

        # Create a binary mask where the depth is within the threshold
        depth_mask = cv2.inRange(depth_image, 0, depth_threshold * 1000)

        # Apply the mask to the color image to remove the background
        foreground = cv2.bitwise_and(color_image, color_image, mask=depth_mask)

        # Create an inverted mask for the background
        background_mask = cv2.bitwise_not(depth_mask)

        # Replace the background with black
        background_replaced = cv2.bitwise_and(color_image, color_image, mask=background_mask)
        background_replaced[depth_mask == 0] = [0, 0, 0]

        # Combine the foreground with the new black background
        final_image = cv2.add(foreground, background_replaced)

        # Display the images
        cv2.imshow("Original RGB Image", color_image)
        cv2.imshow("Depth Image", depth_image)
        cv2.imshow("Foreground", foreground)
        cv2.imshow("Final Image", final_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
