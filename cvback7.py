import cv2
import numpy as np
import pyrealsense2 as r

ca = r.pipeline()
config = r.config()

config.enable_stream(r.stream.color, 640, 480, r.format.bgr8, 30)
config.enable_stream(r.stream.depth, 640, 480, r.format.z16, 30)

ca.start(config)

def grabcut_foreground_extraction(image, rect):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]
    return image

try:
    while True:
        frameset = ca.wait_for_frames(timeout_ms=10000)

        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        if not color_frame or not depth_frame:
            print("No color or depth frame captured.")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        cv2.imshow("Original RGB Image", color_image)

        depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        depth_threshold = 1.0

        depth_mask = np.where((depth_image > 0) & (depth_image < depth_threshold * 1000), 255, 0).astype(np.uint8)

        foreground = cv2.bitwise_and(color_image, color_image, mask=depth_mask)

        rect = (50, 50, color_image.shape[1]-100, color_image.shape[0]-100)
        grabcut_image = grabcut_foreground_extraction(foreground, rect)

        cv2.imshow("GrabCut Foreground Extraction", grabcut_image)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

finally:
    ca.stop()
    cv2.destroyAllWindows()
