'''
import cv2
import numpy as np


def mark_target_text(image_path):
    """Draws a red rectangle around the target text area and saves the marked image"""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return False

    # Get image dimensions
    height, width = img.shape[:2]

    # Coordinates for the target text area (adjust these values as needed)
    # Format: (start_x, start_y, end_x, end_y)
    text_area = (
        int(width * 0.2),  # left
        int(height * 0.7),  # top
        int(width * 0.8),  # right
        int(height * 0.8)  # bottom
    )

    # Draw red rectangle (BGR color format)
    cv2.rectangle(img,
                  (text_area[0], text_area[1]),
                  (text_area[2], text_area[3]),
                  (0, 0, 255),  # Red color
                  2)  # Line thickness

    # Save and show the output
    output_path = "marked_" + image_path
    cv2.imwrite(output_path, img)

    # Display the result (optional)
    cv2.imshow("Marked Image", img)
    cv2.waitKey(2000)  # Display for 2 seconds
    cv2.destroyAllWindows()

    return True


if __name__ == "__main__":
    input_image = "frame_15.jpg"
    success = mark_target_text(input_image)

    if success:
        print(f"Successfully marked target area in {input_image}")
        print(f"Saved as marked_{input_image}")
    else:
        print("Failed to process the image")

#-------------------------------------------------------

import cv2
import pytesseract
import numpy as np


def extract_text_from_region(image_path, region_coords):
    """Extracts text from a specific region of an image"""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # Crop to the specified region
    x1, y1, x2, y2 = region_coords
    cropped = img[y1:y2, x1:x2]

    # Preprocess for better OCR
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform OCR
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    return text.strip()


def mark_and_extract_text(image_path):
    """Marks the target area and extracts text from it"""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # Get image dimensions
    height, width = img.shape[:2]

    # Coordinates for the target text area (same as your working values)
    region_coords = (
        int(width * 0.2),  # left
        int(height * 0.7),  # top
        int(width * 0.8),  # right
        int(height * 0.8)  # bottom
    )

    # Draw red rectangle (for visualization)
    cv2.rectangle(img,
                  (region_coords[0], region_coords[1]),
                  (region_coords[2], region_coords[3]),
                  (0, 0, 255), 2)

    # Save the marked image
    cv2.imwrite("marked_" + image_path, img)

    # Extract text from the region
    extracted_text = extract_text_from_region(image_path, region_coords)

    return extracted_text


if __name__ == "__main__":
    input_image = "frame_15.jpg"
    extracted_text = mark_and_extract_text(input_image)

    if extracted_text:
        print("Successfully extracted text:")
        print(extracted_text)
    else:
        print("Failed to extract text")



#--------------------------------------------------------

import cv2
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def detect_text_roi(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Could not read image.")
        return None, None

    # Resize for consistent processing
    orig = image.copy()
    (H, W) = image.shape[:2]
    newW, newH = 320, 320
    rW, rH = W / float(newW), H / float(newH)
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # Load EAST text detector
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                      "feature_fusion/concat_3"])

    # Decode predictions
    def decode_predictions(scores, geometry, confThreshold=0.5):
        rects = []
        confidences = []

        for y in range(0, scores.shape[2]):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, scores.shape[3]):
                if scoresData[x] < confThreshold:
                    continue

                offsetX, offsetY = x * 4.0, y * 4.0
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
        return rects, confidences

    rects, confidences = decode_predictions(scores, geometry)
    boxes = cv2.dnn.NMSBoxesRotated([(r[0], r[1], r[2] - r[0], r[3] - r[1], 0) for r in rects],
                                    confidences, 0.5, 0.4)

    if len(boxes) == 0:
        print("[INFO] No text regions found, falling back to entire image.")
        return orig, orig

    # Take largest area box
    biggest = max(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    (startX, startY, endX, endY) = biggest
    startX, startY = max(0, startX), max(0, startY)
    endX, endY = min(orig.shape[1], endX), min(orig.shape[0], endY)
    roi = orig[startY:endY, startX:endX]

    return orig, roi

def extract_text(image):
    result = ocr.ocr(image, cls=True)
    lines = []
    for line in result[0]:
        text = line[1][0]
        lines.append(text)
    return "\n".join(lines)

if __name__ == "__main__":
    image_path = "frame_15.jpg"  # Update this if needed
    original_image, roi = detect_text_roi(image_path)

    if roi is not None:
        extracted_text = extract_text(roi)
        print("Extracted Text from ROI:\n", extracted_text)

        # Optional: show ROI
        # plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        # plt.title("Detected Text ROI")
        # plt.axis("off")
        # plt.show()
    else:
        print("[ERROR] No ROI found.")



#----------------------------------------------------------------------------
import cv2
import pytesseract
import numpy as np

# If tesseract is not in your PATH, specify its location here:
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

VIDEO_PATH = "sample_label_video.mp4"
# ROI coordinates for the red rectangle (x, y, w, h).
# Adjust these based on your marked_frame_15.jpg. Example values provided.
ROI_X = 225
ROI_Y = 315
ROI_W = 335
ROI_H = 62

# Minimum OCR text length to consider frame as "clear"
MIN_TEXT_LENGTH = 6

def is_frame_clear(gray_roi, threshold=120):
    # Uses Laplacian variance as a sharpness metric
    laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    return laplacian_var > threshold

def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Try both global and adaptive thresholding for best results
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0

    print("Extracted data from clear frames:")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop to ROI (red rectangle)
        roi = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]

        # Preprocess for OCR
        processed = preprocess_roi(roi)

        # Optional: Skip blurry frames
        if not is_frame_clear(processed):
            frame_count += 1
            continue

        # OCR
        text = pytesseract.image_to_string(processed, config="--psm 7")
        text = text.strip()

        if len(text) >= MIN_TEXT_LENGTH:
            print(f"Frame {frame_count}: {text}")

        frame_count += 1

    cap.release()

if __name__ == "__main__":
    main()

#-------------------------------------
import cv2
import numpy as np
import pytesseract

# Path to your marked frame image
IMAGE_PATH = "marked_frame_15.jpg"

# Automatically get ROI from proportional coordinates
def get_roi(img):
    height, width = img.shape[:2]
    left = int(width * 0.2)
    top = int(height * 0.7)
    right = int(width * 0.8)
    bottom = int(height * 0.8)
    return img[top:bottom, left:right]

# Upscaling factor to enhance small text
UPSCALE_FACTOR = 3

def preprocess_roi(roi):
    # Upscale the ROI for better OCR accuracy
    roi = cv2.resize(roi, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image',gray)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()

    # Reduce noise
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    # Adaptive thresholding for variable lighting/background
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 11
    )
    # Morphological closing to join breaks in characters
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Invert: Tesseract expects black-on-white
    inverted = cv2.bitwise_not(closed)
    return inverted

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Could not read the image. Check path and filename!")
        return
    roi = get_roi(img)
    processed_roi = preprocess_roi(roi)
    # Use Tesseract LSTM OCR, single block mode
    config = r'--oem 1 --psm 6'
    extracted_text = pytesseract.image_to_string(processed_roi, config=config)
    print("Extracted Text:\n")
    print(extracted_text.strip())

if __name__ == "__main__":
    main()

#--------------------------------
import cv2
import numpy as np
import pytesseract

# Path to your marked frame image
IMAGE_PATH = "marked_frame_15.jpg"

def get_roi(img):
    height, width = img.shape[:2]
    left = int(width * 0.2)
    top = int(height * 0.7)
    right = int(width * 0.8)
    bottom = int(height * 0.8)
    return img[top:bottom, left:right]

def preprocess_roi(roi):
    # Convert to grayscale (simple and robust)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Optionally, apply slight denoising if needed
    denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    return denoised

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Could not read the image. Check the path and filename!")
        return
    roi = get_roi(img)
    processed_roi = preprocess_roi(roi)
    # Tesseract config: LSTM OCR, assume a block of text
    config = r'--oem 1 --psm 6'
    extracted_text = pytesseract.image_to_string(processed_roi, config=config)
    print("Extracted Text:\n")
    print(extracted_text.strip())

    # Optional: Save or display the grayscale image for debugging
    # cv2.imwrite('debug_roi_gray.png', processed_roi)
    # cv2.imshow('ROI Grayscale', processed_roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#--------------------------------
import cv2
import numpy as np
import pytesseract

IMAGE_PATH = "marked_frame_15.jpg"

def get_roi(img):
    height, width = img.shape[:2]
    left = int(width * 0.2)
    top = int(height * 0.7)
    right = int(width * 0.8)
    bottom = int(height * 0.8)
    return img[top:bottom, left:right]

def preprocess_roi(roi):
    # Upscale
    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 17, 8
    )
    # Morphological Closing
    kernel = np.ones((2,2), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closed

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Could not read the image.")
        return
    roi = get_roi(img)
    preprocessed = preprocess_roi(roi)
    config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./+*-'
    extracted_text = pytesseract.image_to_string(preprocessed, config=config)
    print("Extracted Text:\n")
    print(extracted_text.strip())

if __name__ == "__main__":
    main()

#-----------------------
import cv2
import numpy as np
import pytesseract

# Path to your grayscale label image
IMAGE_PATH = "marked_image_gray_scale_version.png"

def get_roi(img):
    height, width = img.shape[:2]
    left = int(width * 0.2)
    top = int(height * 0.7)
    right = int(width * 0.8)
    bottom = int(height * 0.8)
    return img[top:bottom, left:right]

def preprocess_roi(gray_roi):
    # Denoise to reduce minor noise (optional, but helps with packaging labels)
    denoised = cv2.fastNlMeansDenoising(gray_roi, None, 20, 7, 21)
    # Upscale for better OCR accuracy
    upscaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return upscaled

def main():
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image could not be loaded. Please check the file path.")
        return

    # roi = get_roi(img)

    # processed_roi = preprocess_roi(roi)
    # Tesseract config: LSTM engine, block of text mode
    config = r'--oem 1 --psm 6'
    extracted_text = pytesseract.image_to_string(img, config=config)
    print("Extracted Text:\n")
    print(extracted_text.strip())

    # For visual debugging (optional)
    cv2.imshow("ROI", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''
#----------------------------------------------------------------------
import cv2
import pytesseract

# Path to your grayscale image
IMAGE_PATH = "marked_image_gray_scale_version.png"

def preprocess_image(img):
    # Optional: denoise (useful for packaging images)
    denoised = cv2.fastNlMeansDenoising(img, None, 20, 7, 21)
    # Upscaling to improve small text recognition
    upscaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return upscaled

def main():
    # Read as grayscale
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load the image. Check the path and filename.")
        return
    processed_img = preprocess_image(img)
    config = r'--oem 1 --psm 6'
    extracted_text = pytesseract.image_to_string(processed_img, config=config)
    print("Extracted Text:\n")
    print(extracted_text.strip())

    # Optional: display or save the processed image for troubleshooting
    cv2.imshow("Processed Image", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("debug_full_processed.png", processed_img)

if __name__ == "__main__":
    main()
#-----------------------------------------------------------------
'''
import cv2
import pytesseract

IMAGE_PATH = "marked_image_gray_scale_version.png"

def preprocess(img):
    # Upscale
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    # Sharpen
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(sharpened, None, 25, 7, 21)
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 9)
    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closed

def main():
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    processed = preprocess(img)
    config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./+*-:'
    text = pytesseract.image_to_string(processed, config=config)
    text = text.replace('Fe.', 'Rs.').replace('MCI', 'MFG')  # Basic post-correction
    print("Extracted Text:\n")
    print(text.strip())

if __name__ == "__main__":
    main()
    '''
