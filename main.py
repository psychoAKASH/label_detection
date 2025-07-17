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

        '''

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

