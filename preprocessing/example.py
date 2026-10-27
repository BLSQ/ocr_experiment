"""Example approach: cut out each box and read the digits by shape matching.

Only works on the demo forms from make_demo_data.py (printed digits, known box
positions). It shows the shape every approach follows: one `read_form`
function that takes a photo and returns {field: value}.
"""

from pathlib import Path

import cv2
import numpy as np

# Where each field's box is on the form: (x, y, width, height), inside the border.
FIELD_BOXES = {
    "tested": (305, 165, 210, 50),
    "positive": (305, 255, 210, 50),
    "negative": (305, 345, 210, 50),
}
SIZE = 24  # every digit shape is resized to SIZE x SIZE before comparing


def _square(shape: np.ndarray) -> np.ndarray:
    """Center a shape in a square and resize it, so shapes can be compared.

    Args:
        shape: Black-and-white image of one character.

    Returns:
        A SIZE x SIZE float image.
    """
    height, width = shape.shape
    side = max(height, width)
    square = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    square[top : top + height, left : left + width] = shape
    return cv2.resize(square, (SIZE, SIZE)).astype(np.float32)


def _reference_digit(digit: str) -> np.ndarray:
    """Draw a digit in the same font as the demo forms, as a reference shape.

    Args:
        digit: The digit to draw.

    Returns:
        The digit's normalized shape.
    """
    canvas = np.zeros((100, 100), dtype=np.uint8)
    cv2.putText(canvas, digit, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 4)
    ys, xs = np.nonzero(canvas)
    return _square(canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])


REFERENCE_DIGITS = {digit: _reference_digit(digit) for digit in "0123456789"}


def read_box(box: np.ndarray) -> str:
    """Read the digits in one box, left to right.

    Args:
        box: Grayscale image of a single box.

    Returns:
        The digits read, or an empty string if none were found.
    """
    # Ink becomes white (255), paper black (0).
    _, ink = cv2.threshold(box, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Each separate blob of ink is a candidate digit; tiny blobs are specks.
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    blobs = sorted(
        (stats[i] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 20),
        key=lambda blob: blob[cv2.CC_STAT_LEFT],
    )
    digits = ""
    for x, y, width, height, _ in blobs:
        shape = _square(ink[y : y + height, x : x + width])
        # Pick the reference digit whose shape matches best.
        scores = {
            digit: cv2.matchTemplate(shape, reference, cv2.TM_CCOEFF_NORMED)[0, 0]
            for digit, reference in REFERENCE_DIGITS.items()
        }
        digits += max(scores, key=scores.__getitem__)
    return digits


def read_form(photo_path: Path) -> dict[str, str]:
    """Read every field of one form.

    Args:
        photo_path: Path to the photo.

    Returns:
        The value read for each field.
    """
    image = cv2.imread(str(photo_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.medianBlur(image, 3)  # remove small specks
    readings = {}
    for field, (x, y, width, height) in FIELD_BOXES.items():
        readings[field] = read_box(image[y : y + height, x : x + width])
    return readings
