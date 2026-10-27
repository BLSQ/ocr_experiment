"""Try a few ways to reduce blur in a form photo and compare them.

Run from the project root:
    python preprocessing/deblur.py data/images_selection/<photo>.png

For each method it prints a sharpness score and the time taken, and saves the result to
results/debug_deblur/ so the methods can be compared by eye.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.restoration import richardson_lucy


def sharpness(gray: np.ndarray) -> float:
    """Score how sharp an image is: the variance of its Laplacian (higher = sharper).

    Sharpening also amplifies noise, which raises this score too, so a higher score does not
    always mean easier reading. Look at the images, and in the end compare OCR accuracy.

    Args:
        gray: Grayscale image.

    Returns:
        The sharpness score.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def unsharp_mask(gray: np.ndarray, sigma: float = 2.0, amount: float = 1.5) -> np.ndarray:
    """Sharpen by adding back the difference between the image and a blurred copy of it.

    Args:
        gray: Grayscale image.
        sigma: Size of the blur used to find edges; larger boosts thicker details.
        amount: How strongly edges are boosted.

    Returns:
        The sharpened image.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    # gray + amount * (gray - blurred), clipped to 0-255
    return cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)


def clahe_then_unsharp(gray: np.ndarray) -> np.ndarray:
    """Even out contrast across the page (CLAHE), then sharpen with an unsharp mask.

    Args:
        gray: Grayscale image.

    Returns:
        The contrast-adjusted, sharpened image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return unsharp_mask(clahe.apply(gray))


def deconvolve(gray: np.ndarray, sigma: float = 1.5, iterations: int = 15) -> np.ndarray:
    """Undo blur by Richardson-Lucy deconvolution, assuming the blur is Gaussian.

    Unlike sharpening, this tries to reverse the blur itself. It works best when `sigma` is close
    to the real blur; too large a value creates ringing around the strokes.

    Args:
        gray: Grayscale image.
        sigma: Assumed size of the blur, in pixels.
        iterations: More iterations sharpen more, but also amplify noise.

    Returns:
        The deblurred image.
    """
    size = int(6 * sigma) | 1  # odd kernel size covering the blur
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    psf = kernel_1d @ kernel_1d.T  # the assumed blur, as a 2D kernel

    restored = richardson_lucy(gray / 255.0, psf, num_iter=iterations)
    return (np.clip(restored, 0, 1) * 255).astype(np.uint8)


METHODS = {
    "unsharp": unsharp_mask,
    "clahe_unsharp": clahe_then_unsharp,
    "deconvolve": deconvolve,
}


if __name__ == "__main__":
    photo_path = Path(sys.argv[1])
    image = cv2.imread(str(photo_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        sys.exit(f"Could not read {photo_path}")

    out_dir = Path("results/debug_deblur")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'method':<15}{'sharpness':>12}{'seconds':>10}")
    print(f"{'original':<15}{sharpness(image):>12.1f}{'':>10}")

    for name, method in METHODS.items():
        start = time.perf_counter()
        result = method(image)
        seconds = time.perf_counter() - start

        print(f"{name:<15}{sharpness(result):>12.1f}{seconds:>10.2f}")
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_{name}.png"), result)

    print(f"Saved results to {out_dir}/")
