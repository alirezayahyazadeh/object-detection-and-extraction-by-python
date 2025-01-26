# Object Detection and Depth Processing Library  

## Overview  
This repository provides a suite of Python scripts designed for object detection, depth image processing, and computer vision tasks. By leveraging powerful libraries such as OpenCV, PyRealSense2, and NumPy, these scripts enable users to capture video streams, extract objects from depth images, and perform advanced image processing techniques.  

## Features  
- **Real-Time Camera Stream Processing**: Capture and process live video streams, including depth data, using OpenCV.  
- **Thresholding and Masking**: Generate masks based on intensity thresholds for effective object detection and filtering.  
- **GrabCut Algorithm**: Perform precise foreground extraction with OpenCV’s GrabCut implementation.  
- **Depth Image Background Subtraction**: Isolate objects by subtracting background depth images, making object extraction easier.  
- **Custom Depth Image Filters**: Apply custom masks and filters to focus on objects of interest.  

## Files  

### `1.camera.py`  
- Captures and processes a live camera feed using OpenCV.  
- Demonstrates real-time image acquisition and display functionality.  

### `3.thershold+mask.py` and `3.thershold+mask4.py`  
- Implements thresholding and masking techniques to isolate objects based on pixel intensity.  
- Useful for filtering images in object detection workflows.  

### `4.GrabCut.py`  
- Utilizes OpenCV’s GrabCut algorithm to separate the foreground from the background.  
- Ideal for extracting objects from cluttered images.  

### `camera_calibration.py`  
- Provides tools for calibrating cameras using a chessboard pattern.  
- Computes the camera matrix and distortion coefficients for lens distortion correction.  

### `Depth 1 to 16.py`  
- Processes depth images captured with Intel RealSense cameras.  
- Captures a background frame, subtracts it from a subject frame, and isolates the subject.  
- Outputs include a binary mask and the extracted object.  

### `jpgback.py`  
- Demonstrates how depth images can create binary masks to separate the foreground from the background.  
- Uses depth thresholds to generate visualizations.  

## Requirements  

### Software  
- **Python**: Version 3.8+  
- **Libraries**:  
  - OpenCV  
  - NumPy  
  - PyRealSense2  

### Hardware  
- Intel RealSense camera (for depth-related scripts).  
