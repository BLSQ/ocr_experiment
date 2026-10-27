"""Straighten a tilted form photo using the angle of its table lines.

Run from the project root:
    python preprocessing/deskew.py data/images_selection/<photo>.png

The straightened image is saved to results/debug_deskew/ so it can be checked by eye.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate how many degrees the page is tilted, from its long straight lines.

    Args:
        gray: Grayscale image.

    Returns:
        The tilt in degrees (positive = page rotated clockwise), 0.0 if no lines are found.
    """
    edges = cv2.Canny(gray, 50, 150)

    # Only keep long segments (table rules, page edges); text strokes are much shorter.
    min_length = min(gray.shape) // 4
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 1800, threshold=100, minLineLength=min_length, maxLineGap=20)
    if lines is None:
        return 0.0

    # One row per segment: x1, y1, x2, y2
    x1, y1, x2, y2 = lines.reshape(-1, 4).astype(float).T
    angles = np.degrees(np.arctan2(y2 - y1, x2 - x1))

    # Fold into [-45, 45): horizontal and vertical lines then both measure the same tilt.
    angles = (angles + 45) % 90 - 45

    # The median ignores the few lines that are not part of the table.
    return float(np.median(angles))


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image around its center, enlarging the canvas so no corner is cut off.

    Args:
        image: Image to rotate.
        angle: Degrees, positive = counter-clockwise.

    Returns:
        The rotated image, with the new border filled in white.
    """
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    # Size of the rotated canvas, then shift so the image stays centered in it.
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2

    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Straighten an image so its table lines are horizontal and vertical.

    Args:
        image: BGR image as loaded by cv2.imread.

    Returns:
        The straightened image and the tilt angle that was corrected, in degrees.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_angle(gray)
    # Rotating by the measured tilt (counter-clockwise in OpenCV) undoes it.
    return rotate(image, angle), angle


if __name__ == "__main__":
    photo_path = Path(sys.argv[1])
    image = cv2.imread(str(photo_path))
    if image is None:
        sys.exit(f"Could not read {photo_path}")

    straight, angle = deskew(image)
    print(f"Tilt found: {angle:.2f} degrees")

    out_dir = Path("results/debug_deskew")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{photo_path.stem}_deskewed.png"
    cv2.imwrite(str(out_path), straight)
    print(f"Saved {out_path}")
