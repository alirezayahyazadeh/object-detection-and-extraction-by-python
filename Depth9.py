import cv2
import numpy as np
import pyrealsense2 as rs  # Intel RealSense SDK

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream

# Start the RealSense camera
pipeline.start(config)

try:
    while True:
        # Wait for frames from the camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # Convert the depth frame to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalize the depth image for better visualization
        depth_image_normalized = cv2.normalize(depth_image, 250, 250, 20,  cv2.NORM_MINMAX, cv2.CV_8UC1)

        # Save the depth image to a file
        cv2.imwrite('depth_image.png', depth_image_normalized)
        cv2.imshow(  'depth_image', depth_image)
        # Display the depth image
        cv2.imshow('Depth Image1', depth_image_normalized)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the camera pipeline and close all OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
