"""Find the table in a form photo and crop it out, flattened to a rectangle.

Run from the project root, ideally on photos already turned upright by orientation.py:
    python preprocessing/table_crop.py results/debug_orientation/*_upright.png

Steps:
1. Lines: keep only the long horizontal and vertical strokes of the image (ruled lines).
2. Table: among the connected groups of lines, keep the one where the most lines cross.
   A table grid has many crossings; the page edge or a single box has only a few.
3. Crop: fit 4 corners around that group and warp them to a rectangle, which also fixes
   perspective (a table photographed at an angle looks like a trapezoid).

For each photo it saves the crop and the original with the table outline drawn on it to
results/debug_table/, so the result can be checked by eye.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def line_masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the long horizontal and vertical lines of an image.

    Args:
        gray: Grayscale image, roughly upright.

    Returns:
        Two binary masks (255 = line): horizontal lines and vertical lines.
    """
    h, w = gray.shape
    # Ink becomes white on black. The neighbourhood size follows the image size.
    block = max(15, min(h, w) // 50) | 1
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 10)

    # Opening with a long, thin kernel keeps only strokes at least that long (lines, not text).
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 30)))
    return horizontal, vertical


def find_table(gray: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Find the 4 corners of the table: the group of lines with the most crossings.

    Args:
        gray: Grayscale image, roughly upright.

    Returns:
        The corners as a (4, 2) array (top-left, top-right, bottom-right, bottom-left) and the
        number of line crossings in the table, or None if no lines cross.
    """
    horizontal, vertical = line_masks(gray)

    # Thicken a little so lines that almost touch join, and their crossings overlap.
    kernel = np.ones((5, 5), np.uint8)
    horizontal, vertical = cv2.dilate(horizontal, kernel), cv2.dilate(vertical, kernel)
    grid = horizontal | vertical
    crossings = horizontal & vertical

    # Group connected lines, then count the crossing pixels inside each group.
    n_groups, labels = cv2.connectedComponents(grid)
    crossing_pixels = np.bincount(labels[crossings > 0], minlength=n_groups)
    crossing_pixels[0] = 0  # label 0 is the background
    best = int(crossing_pixels.argmax())
    if crossing_pixels[best] == 0:
        return None
    table_mask = np.where(labels == best, 255, 0).astype(np.uint8)
    n_crossings = cv2.connectedComponents(crossings & table_mask)[0] - 1

    # Outline of the group, simplified to 4 corners; fall back to the tightest rotated rectangle.
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    corners = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, closed=True), closed=True)
    if len(corners) != 4:
        corners = cv2.boxPoints(cv2.minAreaRect(hull))
    return order_corners(corners.reshape(4, 2).astype(np.float32)), n_crossings


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Sort 4 corners as top-left, top-right, bottom-right, bottom-left.

    Args:
        corners: (4, 2) array of x, y points in any order.

    Returns:
        The same points, sorted.
    """
    total = corners.sum(axis=1)  # x + y: smallest at top-left, largest at bottom-right
    diff = corners[:, 1] - corners[:, 0]  # y - x: smallest at top-right, largest at bottom-left
    return np.array([corners[total.argmin()], corners[diff.argmin()], corners[total.argmax()], corners[diff.argmax()]])


def warp_to_rectangle(image: np.ndarray, corners: np.ndarray, margin: int = 10) -> np.ndarray:
    """Cut out the area inside 4 corners and flatten it to a rectangle.

    Args:
        image: Image to crop.
        corners: (4, 2) array: top-left, top-right, bottom-right, bottom-left.
        margin: Pixels kept around the table so its border lines are not cut.

    Returns:
        The flattened crop.
    """
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32) + margin
    matrix = cv2.getPerspectiveTransform(corners, target)
    size = (width + 2 * margin, height + 2 * margin)
    return cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


if __name__ == "__main__":
    out_dir = Path("results/debug_table")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'photo':<55}{'table area':>11}{'crossings':>11}{'crop size':>12}")
    # One or more photos, e.g. results/debug_orientation/*_upright.png
    for photo_path in map(Path, sys.argv[1:]):
        image = cv2.imread(str(photo_path))
        if image is None:
            print(f"{photo_path.name:<55}could not read")
            continue

        found = find_table(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if found is None:
            print(f"{photo_path.name:<55}no table found")
            continue
        corners, n_crossings = found

        crop = warp_to_rectangle(image, corners)
        outline = cv2.polylines(image.copy(), [corners.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=4)
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_table.png"), crop)
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_outline.png"), outline)

        area = cv2.contourArea(corners) / (image.shape[0] * image.shape[1])
        crop_size = f"{crop.shape[1]}x{crop.shape[0]}"
        print(f"{photo_path.name:<55}{area:>10.0%}{n_crossings:>11}{crop_size:>12}")

    print(f"Saved results to {out_dir}/")
