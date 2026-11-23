"""Straighten a tilted form photo using the angle of its table lines.

deskew() only rotates. straighten() rotates so the horizontal lines are level, then unshears so the vertical lines
are vertical too, for photos taken at an angle. Neither fixes perspective where the vertical lines fan out (lean
one way on the left of the page and the other way on the right); that needs a warp from 4 corners.

Run from the project root:
    python preprocessing/deskew.py data/images_selection/<photo>.png

The straightened image is saved to results/debug_deskew/ so it can be checked by eye.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def line_angles(gray: np.ndarray) -> np.ndarray:
    """Find the long straight lines of an image and return their angles.

    Args:
        gray: Grayscale image.

    Returns:
        One angle per line, in degrees from -180 to 180 (0 = horizontal); empty if no lines are found.
    """
    edges = cv2.Canny(gray, 50, 150)

    # Only keep long segments (table rules, page edges); text strokes are much shorter.
    min_length = min(gray.shape) // 4
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 1800, threshold=100, minLineLength=min_length, maxLineGap=20)
    if lines is None:
        return np.array([])

    # One row per segment: x1, y1, x2, y2
    x1, y1, x2, y2 = lines.reshape(-1, 4).astype(float).T
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate how many degrees the page is tilted, from its long straight lines.

    Args:
        gray: Grayscale image.

    Returns:
        The tilt in degrees (positive = page rotated clockwise), 0.0 if no lines are found.
    """
    angles = line_angles(gray)
    if angles.size == 0:
        return 0.0

    # Fold into [-45, 45): horizontal and vertical lines then both measure the same tilt.
    angles = (angles + 45) % 90 - 45

    # The median ignores the few lines that are not part of the table.
    return float(np.median(angles))


def tilt_and_lean(gray: np.ndarray) -> tuple[float, float]:
    """Measure the horizontal and the vertical lines separately.

    In a photo taken at an angle the two are not at right angles, so one rotation cannot straighten both:
    e.g. the horizontal lines are level but the vertical lines lean.

    Args:
        gray: Grayscale image.

    Returns:
        The tilt of the horizontal lines (degrees, positive = going down to the right) and the lean of the vertical
        lines (degrees from vertical, positive = top leaning right); 0.0 for a direction without lines.
    """
    angles = (line_angles(gray) + 90) % 180 - 90  # -90 to 90, the direction of a line does not matter
    horizontal = angles[np.abs(angles) < 45]
    vertical = angles[np.abs(angles) >= 45]
    # A vertical line is at +90 or -90; its lean is how far it is from that, with the top leaning right positive.
    lean = np.where(vertical > 0, vertical - 90, vertical + 90)
    tilt = float(np.median(horizontal)) if horizontal.size else 0.0
    return tilt, float(np.median(lean)) if lean.size else 0.0


def unshear(image: np.ndarray, lean: float) -> np.ndarray:
    """Slant an image sideways so leaning vertical lines become vertical, leaving horizontal lines as they are.

    Args:
        image: Image whose vertical lines lean.
        lean: Lean of the vertical lines, in degrees (as returned by tilt_and_lean()).

    Returns:
        The unsheared image, widened so nothing is cut off, with the new border filled in white.
    """
    h, w = image.shape[:2]
    # A leaning line has x = x0 - y * tan(lean); moving each row by y * tan(lean) makes x constant.
    shift = np.tan(np.radians(lean))
    offset = max(0.0, -shift * h)  # keep the moved rows inside the canvas
    matrix = np.array([[1, shift, offset], [0, 1, 0]], np.float64)
    new_w = int(np.ceil(w + abs(shift) * h))
    return cv2.warpAffine(image, matrix, (new_w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


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


def straighten(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Straighten the horizontal lines by rotating, then the vertical lines by unshearing.

    Unlike deskew(), this also fixes photos taken at an angle, where the vertical lines still lean after the
    horizontal lines are level.

    Args:
        image: BGR image as loaded by cv2.imread.

    Returns:
        The straightened image, the rotation applied and the lean removed, in degrees.
    """
    tilt, _ = tilt_and_lean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    rotated = rotate(image, tilt)
    # Measure the lean again after the rotation, which changes it.
    _, lean = tilt_and_lean(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY))
    return unshear(rotated, lean), tilt, lean


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
