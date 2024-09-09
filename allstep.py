import cv2
import numpy as np
import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Step 1: Convert to grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Step 2: Apply Otsu's thresholding
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 3: Morphological closing to remove small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Step 4: Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 5: Draw contours as white filled on black background
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

        # Step 6: Get the convex hull of the white filled contours
        hull = cv2.convexHull(contours[0])

        # Step 7: Fit an ellipse to the convex hull if there are enough points
        if len(hull) >= 5:
            ellipse = cv2.fitEllipse(hull)
            (center, axes, angle) = ellipse
            print(f"Ellipse Center: {center}, Axes: {axes}, Angle: {angle}")
            # Print the ellipse shape to check if it's close to a circle
            if abs(axes[0] - axes[1]) / max(axes) < 0.1:  # Check if axes are almost equal
                print("Ellipse is close to a circle")
            else:
                print("Ellipse is not close to a circle")

        # Step 8: Draw the convex hull outline in red on the original image
        hull_image = color_image.copy()
        cv2.polylines(hull_image, [hull], True, (0, 0, 255), 2)

        # Step 9: Draw a circle using the average ellipse radii and center
        circle_mask = np.zeros_like(gray_image)
        if len(hull) >= 5:
            radius = int((axes[0] + axes[1]) / 4)
            cv2.circle(circle_mask, (int(center[0]), int(center[1])), radius, 255, thickness=cv2.FILLED)

        # Step 10: Erode the circle a little to avoid leaving a partial white ring
        eroded_circle_mask = cv2.erode(circle_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)))

        # Step 11: Combine the inverted morph image and the circle mask to make a final mask
        final_mask = cv2.bitwise_and(255 - morph, eroded_circle_mask)

        # Step 12: Apply the final mask to the original image
        result = cv2.bitwise_and(color_image, color_image, mask=final_mask)

        # Display the different stages
        cv2.imshow('Original RGB Image', color_image)
        cv2.imshow('Thresholded Image (Otsu)', thresh)
        cv2.imshow('Morphology Close', morph)
        cv2.imshow('Filled Contours', mask)
        cv2.imshow('Convex Hull', hull_image)
        cv2.imshow('Circle Mask', eroded_circle_mask)
        cv2.imshow('Final Mask', final_mask)
        cv2.imshow('Final Result', result)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
