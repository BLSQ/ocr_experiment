"""Try a few ways to remove noise from a form photo and compare them.

Run from the project root:
    python preprocessing/denoise.py data/images_selection/*.png

Methods:
- median: each pixel becomes the median of its 3 x 3 neighbours. Removes speckles, keeps edges fairly well.
- bilateral: averages neighbours with a similar grey level only, so edges (ink on paper) are not blurred.
- nlmeans: non-local means; averages small patches that look alike anywhere nearby. Usually the cleanest, slower.

For each photo it prints the estimated noise before and after each method and the time taken, and saves a strip
with the original and every method side by side to results/debug_denoise/.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

TILE_WIDTH = 300


def noise(gray: np.ndarray) -> float:
    """Estimate the standard deviation of the noise (Immerkær, 1996).

    The 3 x 3 filter below is the difference of two Laplacians: it gives about 0 on smooth areas, edges and lines,
    so what is left is mostly noise.

    Args:
        gray: Grayscale image.

    Returns:
        The noise standard deviation, in grey levels.
    """
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    response = cv2.filter2D(gray.astype(np.float32), -1, kernel)[1:-1, 1:-1]  # drop the border
    return float(np.sqrt(np.pi / 2) * np.abs(response).mean() / 6)


def nl_means(gray: np.ndarray, strength: float = 7.0) -> np.ndarray:
    """Remove noise with non-local means.

    Args:
        gray: Grayscale image.
        strength: How strongly to smooth (OpenCV's h); higher removes more noise but also faint strokes.

    Returns:
        The denoised image.
    """
    return cv2.fastNlMeansDenoising(gray, None, h=strength, templateWindowSize=7, searchWindowSize=21)


METHODS = {
    "median": lambda gray: cv2.medianBlur(gray, 3),
    "bilateral": lambda gray: cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=5),
    "nlmeans": nl_means,
}


if __name__ == "__main__":
    out_dir = Path("results/debug_denoise")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'photo':<45}{'original':>10}" + "".join(f"{name:>12}{'s':>6}" for name in METHODS))
    # One or more photos, e.g. data/images_selection/*.png
    for photo_path in map(Path, sys.argv[1:]):
        image = cv2.imread(str(photo_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"{photo_path.name:<45}could not read")
            continue

        line = f"{photo_path.name:<45}{noise(image):>10.2f}"
        results = [image]
        for method in METHODS.values():
            start = time.perf_counter()
            result = method(image)
            line += f"{noise(result):>12.2f}{time.perf_counter() - start:>6.2f}"
            results.append(result)
        print(line)

        tiles = [cv2.resize(r, (TILE_WIDTH, int(r.shape[0] * TILE_WIDTH / r.shape[1]))) for r in results]
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_strip.png"), np.hstack(tiles))

    print(f"Saved results to {out_dir}/")
