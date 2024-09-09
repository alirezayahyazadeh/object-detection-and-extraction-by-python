import pyrealsense2 as rs
import numpy as np
import cv2
from matplotlib import pyplot as plt

class DepthMap:
    def __init__(self, showImages: bool = False):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise Exception("Failed to capture depth or color frames")

        # Convert images to numpy arrays
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())

        if showImages:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.color_image)
            plt.subplot(122)
            plt.imshow(self.depth_image, cmap='gray')
            plt.show()

        # Stop the pipeline after use
        self.pipeline.stop()

def demoViewPics():
    # Show pictures captured by the depth camera
    dp = DepthMap(showImages=True)
   

if __name__ == '__main__':
    demoViewPics()
