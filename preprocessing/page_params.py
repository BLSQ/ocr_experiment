"""Try different settings of the page crop (page_crop.py) and compare them side by side.

Run from the project root, on the photos that do not segment well:
    python preprocessing/page_params.py data/images_selection/<photo>.png ...

Every combination of the values in PARAM_GRID is tried. For each photo it saves one contact sheet
to results/debug_page_params/: one tile per combination, with the paper mask in green and the page
outline in red, labelled with its number. It also prints a table of the paper share found for each
photo and combination. Edit PARAM_GRID to try other values (keep it small: the number of tiles is
the product of the list lengths).
"""

import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
from page_crop import crop_page, paper_mask

# Values to try for each setting of paper_mask(); see its docstring for what each one does.
PARAM_GRID = {
    "max_saturation": [40, 70, 110],
    "max_texture": [12, 25, 255],  # 255 = no texture test
    "brightness_offset": [0],
    "close_fraction": [1 / 25, 1 / 10],
}
TILE_WIDTH = 300
COLUMNS = 6


def make_tile(image: np.ndarray, params: dict, number: int) -> np.ndarray:
    """Draw the result of one combination of settings on a copy of the photo.

    Args:
        image: BGR photo.
        params: Settings passed to paper_mask() and crop_page().
        number: Combination number shown on the tile.

    Returns:
        The tile: photo with the paper mask in green and page outline in red, and a label above.
    """
    _, corners, decision, area = crop_page(image, **params)

    tile = image.copy()
    paper = paper_mask(image, **params) > 0
    tile[paper] = (0.6 * tile[paper] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    if corners is not None:
        cv2.polylines(tile, [corners.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=3)

    scale = TILE_WIDTH / tile.shape[1]
    tile = cv2.resize(tile, (TILE_WIDTH, int(tile.shape[0] * scale)))

    label = np.full((40, TILE_WIDTH, 3), 255, np.uint8)
    cv2.putText(label, f"#{number}  {area:.0%}  {decision}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    settings = "  ".join(f"{short_name(k)}={v:g}" for k, v in params.items())
    cv2.putText(label, settings, (5, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    return np.vstack([label, tile])


def short_name(name: str) -> str:
    """Shorten a setting name for the tile labels, e.g. max_saturation -> sat.

    Args:
        name: Setting name.

    Returns:
        The short name.
    """
    return {"max_saturation": "sat", "max_texture": "tex", "brightness_offset": "bri", "close_fraction": "close"}[name]


def contact_sheet(tiles: list[np.ndarray]) -> np.ndarray:
    """Arrange tiles in a grid of COLUMNS columns, padding with white where sizes differ.

    Args:
        tiles: Tiles of width TILE_WIDTH (heights may differ).

    Returns:
        The contact sheet image.
    """
    height = max(tile.shape[0] for tile in tiles)
    padded = [np.vstack([t, np.full((height - t.shape[0], TILE_WIDTH, 3), 255, np.uint8)]) for t in tiles]
    padded += [np.full_like(padded[0], 255)] * (-len(padded) % COLUMNS)  # fill the last row
    rows = [np.hstack(padded[i : i + COLUMNS]) for i in range(0, len(padded), COLUMNS)]
    return np.vstack(rows)


if __name__ == "__main__":
    out_dir = Path("results/debug_page_params")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(PARAM_GRID)
    combinations = [dict(zip(names, values, strict=True)) for values in itertools.product(*PARAM_GRID.values())]

    print("Combinations:")
    for number, params in enumerate(combinations, start=1):
        print(f"  #{number:<3}" + "  ".join(f"{k}={v:g}" for k, v in params.items()))

    # Table: paper share found per photo and combination; * = page cropped, otherwise kept as is.
    print(f"\n{'photo':<45}" + "".join(f"{'#' + str(n):>6}" for n in range(1, len(combinations) + 1)))
    for photo_path in map(Path, sys.argv[1:]):
        image = cv2.imread(str(photo_path))
        if image is None:
            print(f"{photo_path.name:<45}could not read")
            continue

        cells = []
        for params in combinations:
            _, corners, _, area = crop_page(image, **params)
            cells.append(f"{area:>5.0%}{'*' if corners is not None else ' '}")
        print(f"{photo_path.name:<45}" + "".join(cells))

        tiles = [make_tile(image, params, number) for number, params in enumerate(combinations, start=1)]
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_sweep.png"), contact_sheet(tiles))

    print(f"\n* = page cropped. Saved one contact sheet per photo to {out_dir}/")
