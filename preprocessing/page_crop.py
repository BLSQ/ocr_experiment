"""Find the sheet of paper in a photo and crop it out, flattened to a rectangle.

Meant as the first step, before orientation.py. Run from the project root:
    python preprocessing/page_crop.py data/images_selection/*.png

Steps:
1. Paper pixels: bright, nearly colourless (white or grey, not wood or fabric) and smooth
   (little texture), compared with the rest of the photo.
2. Page: fill the holes left by the writing, keep the largest paper region and fit 4 corners.
3. Decide: if that region covers almost the whole photo, the photo shows only the paper and is
   kept as it is. If it is too small, no page was found and the photo is also kept. Otherwise
   the page is warped to a rectangle, which also fixes perspective.

For each photo it saves to results/debug_page/: the page (or the unchanged photo), the photo with
the page outline drawn on it, and the paper mask, so the result can be checked by eye.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from table_crop import order_corners, warp_to_rectangle

MIN_PAGE_AREA = 0.20  # below this share of the photo, the region is not taken as the page
MAX_PAGE_AREA = 0.90  # above it, the photo already shows only the paper
MAX_SATURATION = 70  # 0-255: paper is white or grey, so its colour saturation is low
MAX_TEXTURE = 12  # grey-level standard deviation around a pixel; paper is smooth (255 = no test)
BRIGHTNESS_OFFSET = 0  # added to the automatic (Otsu) brightness threshold; negative = darker allowed
CLOSE_FRACTION = 1 / 25  # size of the hole-filling step, as a share of the photo's shorter side


def paper_mask(
    image: np.ndarray,
    max_saturation: float = MAX_SATURATION,
    max_texture: float = MAX_TEXTURE,
    brightness_offset: float = BRIGHTNESS_OFFSET,
    close_fraction: float = CLOSE_FRACTION,
) -> np.ndarray:
    """Mark the pixels that look like paper: bright, nearly colourless and smooth.

    Args:
        image: BGR image.
        max_saturation: Pixels more colourful than this (0-255) are not paper.
        max_texture: Pixels with more texture than this are not paper; 255 turns the test off.
        brightness_offset: Shifts the automatic brightness threshold up (stricter) or down.
        close_fraction: Size of the hole-filling step, as a share of the photo's shorter side.
            Larger fills bigger gaps (dense tables, big writing) but can join paper to background.

    Returns:
        Binary mask (255 = paper), with the holes left by writing filled in.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    saturation, brightness = hsv[:, :, 1], hsv[:, :, 2]

    # Bright: Otsu picks the brightness threshold that best splits the photo in two groups.
    otsu, _ = cv2.threshold(brightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = brightness > otsu + brightness_offset
    colourless = saturation < max_saturation

    # Smooth: standard deviation of the grey level in a small window around each pixel.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (7, 7))
    texture = np.sqrt(np.maximum(cv2.blur(gray * gray, (7, 7)) - mean * mean, 0))
    smooth = texture < max_texture

    mask = np.where(bright & colourless & smooth, 255, 0).astype(np.uint8)

    # Closing fills the gaps left by text and lines (small bright specks elsewhere don't matter:
    # find_page keeps only the largest region).
    size = max(5, int(min(h, w) * close_fraction))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # The texture test marks the page border itself as "not smooth"; grow back by that width.
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def find_page(mask: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Fit 4 corners around the largest paper region.

    Args:
        mask: Binary paper mask from paper_mask().

    Returns:
        The corners as a (4, 2) array (top-left, top-right, bottom-right, bottom-left) and the
        share of the photo the region covers, or None if there is no paper region.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    region = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(region) / (mask.shape[0] * mask.shape[1])

    # Simplify the outline more and more until it has 4 corners; else use the tightest rectangle.
    hull = cv2.convexHull(region)
    for tolerance in (0.02, 0.04, 0.06):
        corners = cv2.approxPolyDP(hull, tolerance * cv2.arcLength(hull, closed=True), closed=True)
        if len(corners) == 4:
            break
    else:
        corners = cv2.boxPoints(cv2.minAreaRect(hull))
    return order_corners(corners.reshape(4, 2).astype(np.float32)), area


def crop_page(image: np.ndarray, **mask_params: float) -> tuple[np.ndarray, np.ndarray | None, str, float]:
    """Crop the page out of a photo, or keep the photo as it is if that is not needed or not possible.

    Args:
        image: BGR image as loaded by cv2.imread.
        **mask_params: Settings passed to paper_mask() (its defaults are used for any not given).

    Returns:
        The page (or the unchanged photo), the page corners (None if not cropped), what was
        decided, and the share of the photo covered by paper.
    """
    found = find_page(paper_mask(image, **mask_params))
    if found is None:
        return image, None, "kept: no paper found", 0.0

    corners, area = found
    if area > MAX_PAGE_AREA:
        return image, None, "kept: photo is all paper", area
    if area < MIN_PAGE_AREA:
        return image, None, "kept: paper too small", area
    return warp_to_rectangle(image, corners, margin=0), corners, "cropped", area


if __name__ == "__main__":
    out_dir = Path("results/debug_page")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'photo':<45}{'paper':>7}  {'decision':<26}{'size':>10}")
    # One or more photos, e.g. data/images_selection/*.png
    for photo_path in map(Path, sys.argv[1:]):
        image = cv2.imread(str(photo_path))
        if image is None:
            print(f"{photo_path.name:<45}could not read")
            continue

        page, corners, decision, area = crop_page(image)

        outline = image.copy()
        if corners is not None:
            cv2.polylines(outline, [corners.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=3)
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_page.png"), page)
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_outline.png"), outline)
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_mask.png"), paper_mask(image))

        size = f"{page.shape[1]}x{page.shape[0]}"
        print(f"{photo_path.name:<45}{area:>7.0%}  {decision:<26}{size:>10}")

    print(f"Saved results to {out_dir}/")
