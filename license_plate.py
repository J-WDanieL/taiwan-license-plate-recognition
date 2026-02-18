# -*- coding: utf-8 -*-
"""
Taiwan License Plate Recognition

Recognizes Taiwan standard license plates (format: AAA-1234) using
classical computer vision: edge detection, morphological operations,
connected component labeling, and template matching.

Usage:
    python license_plate.py <image_path>
"""

import sys
import os

import cv2
import numpy as np
from skimage import measure


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLATE_W, PLATE_H = 550, 200
SIMILARITY_THRESHOLD = 99.0  # minimum pixel similarity (%) to accept a match

# Maps template index → character label (matches top-to-bottom order in template_fonts.bmp)
CHAR_TABLE = [
    [0,  'A'], [1,  'B'], [2,  'C'], [3,  'D'], [4,  'E'], [5,  'F'], [6,  'G'],
    [7,  'H'], [8,  'I'], [9,  'J'], [10, 'L'], [11, 'K'], [12, 'M'], [13, 'N'],
    [14, 'O'], [15, 'P'], [16, 'R'], [17, 'T'], [18, 'S'], [19, 'Q'], [20, 'U'],
    [21, 'V'], [22, 'W'], [23, 'X'], [24, 'Y'], [25, 'Z'], [26, '3'], [27, '1'],
    [28, '2'], [29, '4'], [30, '5'], [31, '6'], [32, '7'], [33, '8'], [34, '9'],
    [35, '0'],
]


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def detect_plate(img):
    """Locate the license plate region and return its crop, or None if not found."""
    # Focus on the center region (45%–85% height, 33%–66% width)
    thy = round(img.shape[0] * 0.45)
    thh = round(img.shape[0] * 0.85)
    thx = img.shape[1] // 3
    roi = img[thy:thh, thx:thx * 2]

    # Grayscale → Gaussian blur → Sobel edge → Canny → binary threshold
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 10)
    sobel   = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=1)
    edges   = cv2.Canny(sobel, 110, 100)
    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)

    # Morphological operations to consolidate the plate blob
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 10))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 38))
    m = cv2.dilate(binary, k1, 0)
    m = cv2.erode(m,       k1, 4)
    m = cv2.dilate(m,      k1, 6)
    m = cv2.erode(m,       k2, 1)
    m = cv2.dilate(m,      k2, 2)

    # Keep only the contour that matches a plate's aspect ratio
    contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plate_crop = None
    matches = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w > 2 * h
                and h >= m.shape[0] * 0.15
                and w >= m.shape[1] * 0.4
                and 2.5 < w / h < 4.8):
            matches += 1
            plate_crop = roi[y:y + h, x:x + w]

    return plate_crop if matches == 1 else None


def standardize_plate(plate):
    """Convert a plate crop to a normalized grayscale image (550×200 px)."""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (PLATE_W, PLATE_H), interpolation=cv2.INTER_LINEAR)


def enhance_characters(plate_gray):
    """Apply Black-Hat transform and morphology to isolate character pixels."""
    # Black-Hat highlights dark characters against a bright background
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    blackhat = cv2.morphologyEx(plate_gray, cv2.MORPH_BLACKHAT, se)

    # Amplify pixels above the noise floor (vectorised — avoids nested Python loops)
    hot = blackhat > 60
    blackhat[hot] = np.clip(blackhat[hot].astype(np.int32) * 2, 0, 255).astype(np.uint8)

    # Binary threshold → open (separate touching chars) → open (remove noise) → close (fill gaps)
    _, threshold = cv2.threshold(blackhat, 127, 255, cv2.THRESH_BINARY)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT,    (5, 10))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN,  k1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN,  k2)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, k3)

    return threshold


def _cc_label_filter(binary, lower, upper):
    """Return a mask keeping only connected components whose pixel count is in (lower, upper)."""
    labels = measure.label(binary, background=0)
    mask = np.zeros(binary.shape, dtype="uint8")
    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = (labels == label).astype(np.uint8) * 255
        if lower < cv2.countNonZero(label_mask) < upper:
            mask = cv2.add(mask, label_mask)
    return mask


def segment_characters(threshold):
    """Return individual character sub-images sorted left-to-right."""
    lower = 400
    upper = threshold.shape[0] * threshold.shape[1] // 12

    # Tighten area bounds until at least 7 components survive
    while True:
        mask = _cc_label_filter(threshold, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) >= 7:
            break
        lower += 50
        upper -= 10

    # Remove blobs that are too wide, too small, or too large to be characters
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if w / h >= 0.6 or area < 1000 or area > 10000:
            mask[y:y + h, x:x + w] = 0

    # Extract and sort bounding boxes left-to-right
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        chars.append((x, mask[y:y + h, x:x + w]))
    chars.sort(key=lambda c: c[0])
    return [img for _, img in chars]


def load_templates():
    """Load template_fonts.bmp and return character sub-images in CHAR_TABLE order."""
    temp = cv2.imread("template_fonts.bmp", 0)
    if temp is None:
        raise FileNotFoundError('Cannot read "template_fonts.bmp".')

    _, thresh = cv2.threshold(temp, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    templates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        templates.append((y, x, thresh[y:y + h, x:x + w]))

    # Sort top-to-bottom then left-to-right to match CHAR_TABLE ordering
    templates.sort(key=lambda t: (t[0], t[1]))
    return [img for _, _, img in templates]


def _match_character(char_img, templates, t_start, t_end):
    """Return the best-matching label for char_img within templates[t_start:t_end],
    or None if no template exceeds SIMILARITY_THRESHOLD."""
    white_pixels = int(np.sum(char_img == 255))
    if white_pixels == 0:
        return None

    best = []
    for t in range(t_start, t_end):
        resized = cv2.resize(templates[t], (char_img.shape[1], char_img.shape[0]))
        matching = int(np.sum(resized == char_img))
        similarity = matching / white_pixels * 100
        if similarity > SIMILARITY_THRESHOLD:
            best.append((CHAR_TABLE[t][1], similarity))

    if not best:
        return None
    best.sort(key=lambda s: s[1], reverse=True)
    return best[0][0]


def recognize_plate(char_imgs, templates):
    """Match segmented character images and return the plate string (e.g. 'ABC-1234').

    Taiwan plates: first 3 characters are letters (templates 0–25),
    next 4 characters are digits (templates 26–35).
    """
    n = 0
    letters = []

    # Match up to 3 letters
    while n < len(char_imgs) and len(letters) < 3:
        ch = _match_character(char_imgs[n], templates, 0, 26)
        if ch is not None:
            letters.append(ch)
        n += 1

    digits = []

    # Match up to 4 digits from where the letter scan left off
    while n < len(char_imgs) and len(digits) < 4:
        ch = _match_character(char_imgs[n], templates, 26, len(templates))
        if ch is not None:
            digits.append(ch)
        n += 1

    return "".join(letters) + "-" + "".join(digits)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python license_plate.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: cannot read image '{sys.argv[1]}'.")
        sys.exit(1)

    plate = detect_plate(img)
    if plate is None:
        print("Error: license plate not found in image.")
        sys.exit(1)

    plate_gray = standardize_plate(plate)
    threshold  = enhance_characters(plate_gray)
    char_imgs  = segment_characters(threshold)

    if len(char_imgs) < 7:
        print("Error: could not segment 7 characters from the plate.")
        sys.exit(1)

    try:
        templates = load_templates()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    result = recognize_plate(char_imgs, templates)
    print(f"{result}")

    os.makedirs("results", exist_ok=True)
    cv2.imwrite(f"results/{result}.jpg", plate)


if __name__ == "__main__":
    main()
