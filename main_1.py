# import cv2
# import pytesseract
# import numpy as np
#
# # ---------------- CONFIG ----------------
# VIDEO_PATH = "sample_label_video.mp4"  # Your video filename
# FRAME_SKIP = 3                        # Process every nth frame
# BLUR_THRESHOLD = 130                  # Stricter for fast conveyor motion
# TESSERACT_CONFIG = "--psm 6"          # Assume block of text
# MIN_LABEL_AREA = 800                  # Ignore small detections
#
# # Tuned HSV range for green label (from frame_15)
# LOWER_GREEN = np.array([63, 28, 53])
# UPPER_GREEN = np.array([87, 232, 145])
#
# # ---------------- FUNCTIONS ----------------
#
# def is_clear(image, threshold=BLUR_THRESHOLD):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return var > threshold
#
# def detect_label_roi(frame):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > MIN_LABEL_AREA:
#             x, y, w, h = cv2.boundingRect(cnt)
#             roi = frame[y:y+h, x:x+w]
#             return roi
#     return None
#
# def extract_text(roi):
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     text = pytesseract.image_to_string(gray, config=TESSERACT_CONFIG)
#     return text.strip()
#
# # ---------------- MAIN ----------------
#
# def main():
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     if not cap.isOpened():
#         print(f"❌ Cannot open video: {VIDEO_PATH}")
#         return
#
#     print("🔍 Starting OCR extraction...\n")
#     frame_id = 0
#     extracted_count = 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         if frame_id % FRAME_SKIP == 0:
#             roi = detect_label_roi(frame)
#
#             if roi is not None and is_clear(roi):
#                 text = extract_text(roi)
#                 if text:
#                     extracted_count += 1
#                     print(f"[Frame {frame_id}] ✅ Text: {text}")
#
#         frame_id += 1
#
#     cap.release()
#     print(f"\n✅ Done. Total extracted entries: {extracted_count}")
#
# if __name__ == "__main__":
#     main()


#------------------------------------------------------------------------------------------
import cv2
import pytesseract
import numpy as np

# Path to your video file
VIDEO_PATH = "sample_label_video.mp4"

# HSV range for detecting green background
LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])

# Minimum contour area and size to ignore tiny regions
MIN_AREA = 500
MIN_WIDTH = 80
MIN_HEIGHT = 20

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
label_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue

        # Extract green region and preprocess
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Optional dilation to connect characters
        binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)

        # OCR
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(binary, config=custom_config)

        if text.strip():
            print(f"[Frame {frame_count} | Label {label_count}] Text:")
            print(text.strip())
            print("-" * 40)

        # Optional: draw bounding box for debug
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_count += 1

    # Optional: Uncomment below to visualize frames
    # cv2.imshow("Detected Green Labels", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()






