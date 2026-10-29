"""Find the sheet of paper in a photo and crop it out, flattened to a rectangle.

Meant as the first step. Run from the project root:
    python preprocessing/page_crop.py data/images_selection/*.png

There are two ways to find the page:

"mask" (paper pixels), for a page on a coloured or textured background (wood, fabric):
1. Paper pixels: bright, nearly colourless (white or grey, not wood or fabric) and smooth
   (little texture), compared with the rest of the photo.
2. Page: fill the holes left by the writing, keep the largest paper region and fit 4 corners.

"edges" (page borders), which also works for a page lying on another sheet of paper:
1. Remove the writing and ruled lines, leaving only paper and background.
2. Near each side of the photo, look for a straight line where the image gets darker going
   outwards from the page, with paper-bright pixels on the page side. A side without one is
   taken as the photo's own edge (the page runs off the photo there).
3. The 4 corners are where these lines meet.

Then, for both: if the page covers almost the whole photo, the photo shows only the paper and is
kept as it is. If it is too small, no page was found and the photo is also kept. Otherwise
the page is warped to a rectangle, which also fixes perspective.

For each photo it runs both methods and saves to results/debug_page/: the page found by each
(or the unchanged photo), the photo with both outlines drawn on it (red = mask, blue = edges),
and the paper mask, so the results can be checked by eye.
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

# Settings of the "edges" method (find_page_edges()).
EDGE_BAND = 0.35  # each page border is searched in this share of the photo next to its side
MIN_STEP = 6  # grey levels the page must be brighter than what is outside it (a sheet on a sheet gives ~7)
PAPER_SHARE = 0.7  # the page side of a border must be at least this bright, relative to the middle of the photo
MIN_SUPPORT = 0.3  # share of the rows (or columns) that must lie on a border for it to count
MAX_BORDER_TILT = 20  # degrees; a border tilted more than this is not a page border


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


def remove_ink(gray: np.ndarray) -> np.ndarray:
    """Erase the writing and ruled lines, leaving only the paper and the background.

    A closing (a max filter then a min filter) removes dark strokes thinner than its kernel, but keeps the
    border between two large areas (paper and background) where it is.

    Args:
        gray: Grayscale image.

    Returns:
        The image without thin dark strokes, as float32.
    """
    size = max(9, min(gray.shape) // 30) | 1
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return closed.astype(np.float32)


def find_border(clean: np.ndarray, paper_level: float, min_step: float) -> tuple[float, float] | None:
    """Find the left border of the page: a straight line where the image gets brighter going right.

    Only the left side is searched; the other sides are found by flipping or transposing the image first.

    Args:
        clean: Image without ink (from remove_ink()), turned so the side to search is on the left.
        paper_level: Brightness of the paper; the side of the border towards the page must be close to it.
        min_step: How much brighter (grey levels) the page must be than what is outside it.

    Returns:
        The border as x = a * y + b, or None if no straight border is found.
    """
    h, w = clean.shape
    # Smooth along the border (vertically), so a row's value isn't thrown off by one speck.
    smooth = cv2.blur(clean, (1, 9))
    band = int(w * EDGE_BAND)
    d = 3  # compare pixels this far on each side of the possible border
    xs = np.arange(d, band)
    outside, inside = smooth[:, xs - d], smooth[:, xs + d]
    step = np.where(inside >= PAPER_SHARE * paper_level, inside - outside, 0)

    # In each row, the strongest step is a point on the border, if it is strong enough.
    best = step.argmax(axis=1)
    rows = np.flatnonzero(step[np.arange(h), best] >= min_step)
    if rows.size < MIN_SUPPORT * h:
        return None
    x, y = xs[best[rows]].astype(np.float32), rows.astype(np.float32)

    # Near the corners, some points come from the neighbouring border, so a plain fit is pulled off the line.
    # Instead (RANSAC): try lines through random pairs of points, keep the one most points are close to.
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(x), (200, 2))
    pairs = pairs[y[pairs[:, 0]] != y[pairs[:, 1]]]
    slopes = (x[pairs[:, 1]] - x[pairs[:, 0]]) / (y[pairs[:, 1]] - y[pairs[:, 0]])
    offsets = x[pairs[:, 0]] - slopes * y[pairs[:, 0]]
    n_close = (np.abs(x[None, :] - (slopes[:, None] * y[None, :] + offsets[:, None])) <= 3).sum(axis=1)
    best_line = n_close.argmax()
    if n_close[best_line] < MIN_SUPPORT * h:
        return None

    # Refit on the points close to the best line.
    close = np.abs(x - (slopes[best_line] * y + offsets[best_line])) <= 3
    a, b = np.polyfit(y[close], x[close], 1)
    if abs(np.degrees(np.arctan(a))) > MAX_BORDER_TILT:
        return None
    return float(a), float(b)


def find_page_edges(image: np.ndarray, min_step: float = MIN_STEP) -> tuple[np.ndarray, float] | None:
    """Find the page from its straight borders, where the paper meets something darker.

    Unlike paper_mask(), this does not need the background to be coloured or textured: it works when the page
    lies on another sheet, as long as the border shows as a step in brightness (often a thin shadow). A side
    with no border found is taken as the photo's own edge (the page runs off the photo there).

    Args:
        image: BGR image.
        min_step: How much brighter (grey levels) the page must be than what is outside it.

    Returns:
        The corners as a (4, 2) array (top-left, top-right, bottom-right, bottom-left) and the share of the
        photo inside them, or None if no border is found on any side.
    """
    clean = remove_ink(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    h, w = clean.shape
    # The page is in the middle of the photo, so the middle tells how bright the paper is.
    paper_level = float(np.median(clean[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]))

    # Each side, turned so it is on the left. A line x = a*y + b in a turned image is written back as a line
    # in the photo, in homogeneous form (p, q, r): p*x + q*y + r = 0.
    sides = {
        "left": (clean, lambda a, b: (1, -a, -b)),
        "right": (clean[:, ::-1], lambda a, b: (-1, -a, w - 1 - b)),
        "top": (clean.T, lambda a, b: (-a, 1, -b)),
        "bottom": (clean.T[:, ::-1], lambda a, b: (-a, -1, h - 1 - b)),
    }
    photo_edges = {"left": (1, 0, 0), "right": (1, 0, -(w - 1)), "top": (0, 1, 0), "bottom": (0, 1, -(h - 1))}
    lines, n_found = {}, 0
    for side, (turned, to_photo) in sides.items():
        border = find_border(np.ascontiguousarray(turned), paper_level, min_step)
        lines[side] = to_photo(*border) if border else photo_edges[side]
        n_found += border is not None
    if n_found == 0:
        return None

    # Corners where neighbouring borders meet, kept inside the photo.
    pairs = [("top", "left"), ("top", "right"), ("bottom", "right"), ("bottom", "left")]
    corners = []
    for one, two in pairs:
        x, y, z = np.cross(lines[one], lines[two])
        corners.append((np.clip(x / z, 0, w - 1), np.clip(y / z, 0, h - 1)))
    corners = np.array(corners, np.float32)
    return corners, cv2.contourArea(corners) / ((h - 1) * (w - 1))


def crop_page(
    image: np.ndarray, method: str = "mask", **params: float
) -> tuple[np.ndarray, np.ndarray | None, str, float]:
    """Crop the page out of a photo, or keep the photo as it is if that is not needed or not possible.

    Args:
        image: BGR image as loaded by cv2.imread.
        method: "mask" to find the page from its paper pixels (paper_mask()), or "edges" to find it from its
            straight borders (find_page_edges()).
        **params: Settings passed to paper_mask() or find_page_edges() (their defaults are used for any not given).

    Returns:
        The page (or the unchanged photo), the page corners (None if not cropped), what was
        decided, and the share of the photo covered by paper.
    """
    if method == "edges":
        found = find_page_edges(image, **params)
        if found is None:
            return image, None, "kept: no page border found", 0.0
    else:
        found = find_page(paper_mask(image, **params))
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

    # Both methods are run, to compare them: red outline = mask, blue outline = edges.
    colours = {"mask": (0, 0, 255), "edges": (255, 0, 0)}
    print(f"{'photo':<45}" + "".join(f"{m + ' page':>12}  {m + ' decision':<28}" for m in colours))
    # One or more photos, e.g. data/images_selection/*.png
    for photo_path in map(Path, sys.argv[1:]):
        image = cv2.imread(str(photo_path))
        if image is None:
            print(f"{photo_path.name:<45}could not read")
            continue

        outline = image.copy()
        line = f"{photo_path.name:<45}"
        for method, colour in colours.items():
            page, corners, decision, area = crop_page(image, method=method)
            if corners is not None:
                cv2.polylines(outline, [corners.astype(np.int32)], isClosed=True, color=colour, thickness=3)
            cv2.imwrite(str(out_dir / f"{photo_path.stem}_page_{method}.png"), page)
            line += f"{area:>12.0%}  {decision:<28}"
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_outline.png"), outline)
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_mask.png"), paper_mask(image))
        print(line)

    print(f"Saved results to {out_dir}/")
