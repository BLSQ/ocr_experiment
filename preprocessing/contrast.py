"""Even out the lighting and raise the contrast, so faint table lines stand out from the paper.

Shadows and tinted paper make one part of the page darker than another, so a line in a bright corner and the
paper in a dark corner can have the same grey level. Dividing by the paper level found around each pixel removes
that, and leaves the paper white and the ink dark everywhere.

Run from the project root:
    python preprocessing/contrast.py data/images_selection/*.png

Methods:
- flatten: estimate the paper level with a closing (a max filter then a min filter, which removes dark strokes
  thinner than the kernel), and divide the image by it. Removes shadows and uneven lighting.
- clahe: CLAHE (local histogram equalisation). Raises local contrast, but also raises the paper's noise.
- flatten_clahe: flatten, then CLAHE.
- flatten_stretch: flatten, then stretch so the darkest ink becomes black (faint lines get darker).

For each photo it prints how dark the ruled lines are compared with the paper before and after each method, and
saves a strip with the original and every method side by side to results/debug_contrast/.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from table_crop import line_masks

PAPER_FRACTION = 1 / 20  # size of the closing that finds the paper level, as a share of the shorter side
TILE_WIDTH = 300


def flatten(gray: np.ndarray, size_fraction: float = PAPER_FRACTION) -> np.ndarray:
    """Divide the image by its local paper level, so the paper becomes evenly white.

    Args:
        gray: Grayscale image.
        size_fraction: Size of the closing, as a share of the shorter side. It must be wider than the lines and
            handwriting, so they are removed from the paper level.

    Returns:
        The evened-out image: paper near 255, ink darker.
    """
    size = max(15, int(min(gray.shape) * size_fraction)) | 1
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return cv2.divide(gray, paper, scale=255)


def clahe(gray: np.ndarray) -> np.ndarray:
    """Raise the local contrast with CLAHE.

    Args:
        gray: Grayscale image.

    Returns:
        The equalised image.
    """
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def stretch_ink(gray: np.ndarray, ink_percentile: float = 1.0) -> np.ndarray:
    """Stretch the grey levels so the darkest ink becomes black and the paper stays white.

    Args:
        gray: Grayscale image, paper near white (e.g. after flatten()).
        ink_percentile: Share (%) of the darkest pixels that become fully black.

    Returns:
        The stretched image.
    """
    low = np.percentile(gray, ink_percentile)
    scaled = (gray.astype(np.float32) - low) * 255 / max(255 - low, 1.0)
    return np.clip(scaled, 0, 255).astype(np.uint8)


METHODS = {
    "flatten": flatten,
    "clahe": clahe,
    "flatten_clahe": lambda gray: clahe(flatten(gray)),
    "flatten_stretch": lambda gray: stretch_ink(flatten(gray)),
}


def line_contrast(gray: np.ndarray) -> float:
    """Measure how much darker the ruled lines are than the paper, in units of the paper's noise.

    Args:
        gray: Grayscale image.

    Returns:
        (median paper level - median line level) / paper standard deviation; 0.0 if no lines are found.
    """
    horizontal, vertical = line_masks(gray)
    lines = (horizontal | vertical) > 0
    paper = cv2.dilate((horizontal | vertical), np.ones((7, 7), np.uint8)) == 0
    if not lines.any() or not paper.any():
        return 0.0
    return float((np.median(gray[paper]) - np.median(gray[lines])) / max(gray[paper].std(), 1.0))


if __name__ == "__main__":
    out_dir = Path("results/debug_contrast")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'photo':<45}{'original':>10}" + "".join(f"{name:>17}" for name in METHODS))
    # One or more photos, e.g. data/images_selection/*.png
    for photo_path in map(Path, sys.argv[1:]):
        image = cv2.imread(str(photo_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"{photo_path.name:<45}could not read")
            continue

        results = {"original": image} | {name: method(image) for name, method in METHODS.items()}
        tiles = [cv2.resize(r, (TILE_WIDTH, int(r.shape[0] * TILE_WIDTH / r.shape[1]))) for r in results.values()]
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_strip.png"), np.hstack(tiles))
        print(
            f"{photo_path.name:<45}"
            + "".join(
                f"{line_contrast(r):>10.1f}" if name == "original" else f"{line_contrast(r):>17.1f}"
                for name, r in results.items()
            )
        )

    print(f"Saved results to {out_dir}/")
