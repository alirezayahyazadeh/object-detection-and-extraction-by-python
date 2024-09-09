import pyrealsense2 as rs
import numpy as np
import cv2
from rembg import remove
from PIL import Image
import json


pipeline = rs.pipeline()
config = rs.config()


config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)


point_cloud_list = []

try:
    while True:
       
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        
        color_image = np.asanyarray(color_frame.get_data())
       
        pil_image = Image.fromarray(color_image)
        
        output_image = remove(pil_image)
        
        output_np = np.array(output_image)


        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        for y in range(depth_frame.get_height()):
            for x in range(depth_frame.get_width()):
                distance = depth_frame.get_distance(x, y)
                if distance > 0:
                   
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], distance)
                    point_cloud_list.append(point)

        
        cv2.imshow("Original RGB Image", color_image)
        cv2.imshow("Background Removed Image", output_np)

        #
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    
    with open("point_cloud.json", "w") as f:
        json.dump(point_cloud_list, f)
    
    
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Point cloud saved to 'point_cloud.txt'")
