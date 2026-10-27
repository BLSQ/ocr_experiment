"""Pick the colour channel where the ink stands out most from the paper, instead of plain grey.

Blue ballpoint absorbs red light, so it is dark in the red channel, while grey also mixes in the green and blue
channels, where blue ink is faint. Black ink and pencil look the same in every channel, and red ink almost
disappears in the red channel, so the best channel is measured per photo instead of fixed.

Lab b* (the blue-yellow axis) can separate blue ink even better, but black lines on white paper have the same b*
as the paper, so it can erase the table lines. It is shown for comparison and can be asked for, but "auto" only
picks among grey, red, green and blue.

Needs a colour image, so run it on original photos or page crops, before any step that makes the image grey
(deblur). Run from the project root:
    python preprocessing/channel.py results/debug_page/*_page.png

Steps:
1. Channels: grey, red, green, blue and Lab b*.
2. Contrast: in each channel, how far the darkest pixels (writing) are below the local paper level, in units of
   the paper's own noise. The ruled lines are left out (they are usually black, so the same in every channel),
   and measuring against the local paper keeps shadows out of the score; evening out the lighting is a separate
   job. Higher = writing easier to separate from paper.
3. Pick the channel with the highest contrast, and stretch it to the full 0-255 range.

For each photo it prints the contrast of every channel and saves to results/debug_channel/ a strip with every
channel side by side, labelled with its contrast (the chosen one marked with *), so the choice can be checked by eye.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from table_crop import line_masks

INK_SHARE = 0.03  # outside the ruled lines, the darkest 3% of pixels (relative to the local paper) are the writing
AUTO_CHANNELS = ["gray", "red", "green", "blue"]  # "auto" picks among these; lab_b can erase black lines
TILE_WIDTH = 300

# Each channel as a function of the BGR image. In all of them ink is darker than paper (blue ink has a low b*).
CHANNELS = {
    "gray": lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
    "red": lambda image: image[:, :, 2],
    "green": lambda image: image[:, :, 1],
    "blue": lambda image: image[:, :, 0],
    "lab_b": lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 2],
}


def not_lines(image: np.ndarray) -> np.ndarray:
    """Find the pixels that are not on a ruled line, so the score measures the writing.

    Args:
        image: BGR image.

    Returns:
        A boolean mask, True away from the lines.
    """
    horizontal, vertical = line_masks(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    lines = cv2.dilate(horizontal | vertical, np.ones((5, 5), np.uint8))  # also skip the blurred edges of lines
    return lines == 0


def ink_contrast(channel: np.ndarray, keep: np.ndarray) -> float:
    """Score how well the writing stands out from the paper in one channel.

    The local paper level is found by closing (a max filter followed by a min filter), which removes dark strokes
    thinner than the kernel. How far each pixel is below it is its darkness. Most pixels are paper, so the median
    darkness is the paper's noise; the score is the mean darkness of the darkest pixels in units of that noise.

    Args:
        channel: One channel of the image (ink darker than paper).
        keep: Boolean mask of the pixels to measure (from not_lines()).

    Returns:
        The contrast score (higher = easier to separate).
    """
    size = max(15, min(channel.shape) // 25) | 1
    paper = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    darkness = (paper.astype(np.float32) - channel.astype(np.float32))[keep]

    noise = max(1.4826 * float(np.median(darkness)), 1.0)  # 1.4826 scales it like a standard deviation
    ink = darkness[darkness >= np.quantile(darkness, 1 - INK_SHARE)].mean()
    return float(ink / noise)


def stretch(channel: np.ndarray) -> np.ndarray:
    """Stretch a channel so its darkest pixels become black and its brightest white.

    Lab b* only uses a small part of the 0-255 range, and later steps use fixed thresholds, so every channel is
    stretched the same way.

    Args:
        channel: One channel of the image.

    Returns:
        The stretched channel, as uint8.
    """
    low, high = np.percentile(channel, [0.5, 99.5])
    scaled = (channel.astype(np.float32) - low) * 255 / max(high - low, 1.0)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def select_channel(image: np.ndarray, channel: str = "auto") -> tuple[np.ndarray, str, dict[str, float]]:
    """Turn a colour image into the one channel with the best ink contrast.

    Args:
        image: BGR image.
        channel: A name from CHANNELS to always use that channel, or "auto" to pick the best of AUTO_CHANNELS.

    Returns:
        The chosen channel stretched to 0-255, its name, and the contrast of every channel.
    """
    channels = {name: get(image) for name, get in CHANNELS.items()}
    keep = not_lines(image)
    contrasts = {name: ink_contrast(values, keep) for name, values in channels.items()}
    chosen = max(AUTO_CHANNELS, key=contrasts.get) if channel == "auto" else channel
    return stretch(channels[chosen]), chosen, contrasts


def channel_strip(image: np.ndarray, chosen: str, contrasts: dict[str, float]) -> np.ndarray:
    """Put every channel side by side, labelled with its contrast, to compare them by eye.

    Args:
        image: BGR image.
        chosen: Name of the chosen channel, marked with *.
        contrasts: Contrast of every channel, from select_channel().

    Returns:
        The strip, as a grey image.
    """
    tiles = []
    for name, get in CHANNELS.items():
        tile = stretch(get(image))
        tile = cv2.resize(tile, (TILE_WIDTH, int(tile.shape[0] * TILE_WIDTH / tile.shape[1])))
        label = np.full((25, TILE_WIDTH), 255, np.uint8)
        mark = " *" if name == chosen else ""
        cv2.putText(label, f"{name}  {contrasts[name]:.1f}{mark}", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        tiles.append(np.vstack([label, tile]))
    return np.hstack(tiles)


if __name__ == "__main__":
    out_dir = Path("results/debug_channel")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'photo':<55}" + "".join(f"{name:>8}" for name in CHANNELS) + f"{'chosen':>8}")
    # One or more photos, e.g. results/debug_page/*_page.png
    for photo_path in map(Path, sys.argv[1:]):
        image = cv2.imread(str(photo_path))
        if image is None:
            print(f"{photo_path.name:<55}could not read")
            continue

        best, chosen, contrasts = select_channel(image)
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_channel.png"), best)
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_strip.png"), channel_strip(image, chosen, contrasts))
        print(f"{photo_path.name:<55}" + "".join(f"{value:>8.1f}" for value in contrasts.values()) + f"{chosen:>8}")

    print(f"Saved results to {out_dir}/")
