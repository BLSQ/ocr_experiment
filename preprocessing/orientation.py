"""Turn a form photo the right way up: fix the small tilt, then quarter turns (90/180/270).

Run from the project root:
    python preprocessing/orientation.py data/images_selection/*.png

Steps:
1. Lines: straighten the small tilt from the table lines (deskew() from deskew.py).
2. Text direction: detect text boxes. If the long ones run top-to-bottom, the page is on its
   side, so turn it 90 degrees. The text is now horizontal, but may still be upside down.
3. Text content: read the page as it is and turned 180 degrees, and keep the one the OCR
   reads most confidently. Upside-down text gets low confidence scores.

The result is saved to results/debug_orientation/ so it can be checked by eye.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr import ModelType, OCRVersion, RapidOCR
from deskew import deskew

ENGINE = RapidOCR(
    params={
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_type": ModelType.MOBILE,
    }
)


def text_is_vertical(image: np.ndarray) -> bool:
    """Check whether the text lines run top-to-bottom, meaning the page is on its side.

    Args:
        image: Image to check.

    Returns:
        True if most long text boxes are taller than they are wide.
    """
    boxes = ENGINE(image, use_det=True, use_cls=False, use_rec=False).boxes
    if boxes is None:
        return False

    # Each box is 4 (x, y) corners; the image is already straightened, so use its x and y extent.
    widths = np.ptp(boxes[:, :, 0], axis=1)
    heights = np.ptp(boxes[:, :, 1], axis=1)

    # Only long boxes (words, sentences) show the direction; single digits are nearly square.
    long_boxes = np.maximum(widths, heights) >= 2 * np.minimum(widths, heights)
    if not long_boxes.any():
        return False
    vertical = heights[long_boxes] > widths[long_boxes]
    return vertical.mean() > 0.5


def reading_confidence(image: np.ndarray) -> float:
    """Read the image and score how confident the OCR is, weighted by text length.

    Args:
        image: Image to read.

    Returns:
        Mean confidence per character (0-1), 0.0 if nothing was read.
    """
    # All three flags are given because the engine keeps the ones from its previous call.
    # use_cls=False: the text-line classifier would flip upside-down lines and hide the problem.
    result = ENGINE(image, use_det=True, use_cls=False, use_rec=True)
    if not result.txts:
        return 0.0
    lengths = np.array([len(text) for text in result.txts])
    return float(np.average(result.scores, weights=lengths)) if lengths.sum() else 0.0


def fix_orientation(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Straighten a photo and turn it so its text reads left-to-right, top-to-bottom.

    Args:
        image: BGR image as loaded by cv2.imread.

    Returns:
        The upright image, and a dict with the tilt, quarter turns and confidences found.
    """
    straight, tilt = deskew(image)

    turned_sideways = text_is_vertical(straight)
    if turned_sideways:
        straight = cv2.rotate(straight, cv2.ROTATE_90_CLOCKWISE)

    flipped = cv2.rotate(straight, cv2.ROTATE_180)
    confidence = reading_confidence(straight)
    confidence_flipped = reading_confidence(flipped)
    upside_down = confidence_flipped > confidence

    # Total clockwise turn applied after the tilt fix, in degrees.
    turn = (90 * turned_sideways + 180 * upside_down) % 360
    info = {
        "tilt": tilt,
        "turn_clockwise": turn,
        "confidence": confidence,
        "confidence_flipped": confidence_flipped,
    }
    return (flipped if upside_down else straight), info


if __name__ == "__main__":
    out_dir = Path("results/debug_orientation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'photo':<45}{'tilt':>7}{'turn':>6}{'conf':>7}{'flipped':>9}")
    # One or more photos, e.g. data/images_selection/*.png
    for photo_path in map(Path, sys.argv[1:]):
        image = cv2.imread(str(photo_path))
        if image is None:
            print(f"{photo_path.name:<45}could not read")
            continue

        upright, info = fix_orientation(image)
        print(
            f"{photo_path.name:<45}{info['tilt']:>7.2f}{info['turn_clockwise']:>6}"
            f"{info['confidence']:>7.3f}{info['confidence_flipped']:>9.3f}"
        )
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_upright.png"), upright)

    print(f"Saved results to {out_dir}/")
