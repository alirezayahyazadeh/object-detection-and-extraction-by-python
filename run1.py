import cv2
import numpy as np

# تنظیمات
background_video_path = 'background.avi'
object_video_path = 'object.avi'
output_path = 'extracted_object.avi'
recording_time = 10  # زمان ضبط پس‌زمینه به ثانیه

# اتصال به دوربین
cap = cv2.VideoCapture(0)

# دریافت تنظیمات دوربین
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# تعریف کدک و ایجاد شیء ضبط ویدیو
fourcc = cv2.VideoWriter_fourcc(*'XVID')
background_out = cv2.VideoWriter(background_video_path, fourcc, 20.0, (frame_width, frame_height))

# ضبط ویدیو پس‌زمینه
print("Recording background...")
start_time = cv2.getTickCount()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    background_out.write(frame)
    cv2.imshow('Background Recording', frame)
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    if elapsed_time > recording_time:
        break

background_out.release()
print("Background recording completed.")

# بارگذاری ویدیو پس‌زمینه
background_cap = cv2.VideoCapture(background_video_path)

# ضبط ویدیو با آبجکت
object_out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

print("Recording object video...")
while True:
    ret_object, frame_object = cap.read()
    if not ret_object:
        break

    ret_background, background_frame = background_cap.read()
    if not ret_background:
        background_frame = np.zeros_like(frame_object)

    # تبدیل به خاکستری
    gray_object = cv2.cvtColor(frame_object, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)

    # اعمال تفاوت
    diff = cv2.absdiff(gray_background, gray_object)
    _, fg_mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # اعمال فیلتر و پیدا کردن کانتورها
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_with_contours = frame_object.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame_with_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # نمایش ویدیو
    cv2.imshow('Object Video', frame_object)
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Extracted Object', frame_with_contours)
    object_out.write(frame_with_contours)

    # خروج از حلقه با فشردن کلید 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

background_cap.release()
cap.release()
object_out.release()
cv2.destroyAllWindows()
print("Object extraction completed.")