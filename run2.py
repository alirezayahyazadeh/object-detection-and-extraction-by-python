import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try using DSHOW as the backend
# If this doesn't work, try changing `0` to `1` or `-1` to use a different camera.

if not cap.isOpened():
    print("Error: Could not open video device.")
else:
    print("Camera is opened successfully.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    cv2.imshow('Camera Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
