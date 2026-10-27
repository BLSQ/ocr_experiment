"""Find the cells of a ruled table and save them as a grid: row, column, span and pixel box.

Run from the project root, on table crops from table_crop.py:
    python preprocessing/build_cell_grids.py results/debug_table/*_table.png

Steps:
1. Lines: keep only the long horizontal and vertical strokes (line_masks() from table_crop.py).
2. Grid: a row of pixels crossed by a horizontal line for much of its width is a row boundary
   (same for columns). This gives the y of every horizontal line and the x of every vertical one.
3. Merged cells: between two neighbouring grid cells, check whether the line separating them is
   really drawn. If not, the two belong to one merged cell (e.g. a header over two columns).

Anything outside the outer lines (titles, notes) is not part of any cell, so it is ignored.

For each table it saves to results/debug_cells/: a JSON file with the grid and cells, and the image
with every cell outlined and numbered (merged cells in orange), so the grid can be checked by eye.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from table_crop import line_masks

MIN_LINE_COVERAGE = 0.3  # share of the table width (or height) a line must cross to be a boundary
MIN_LINE_GAP = 6  # pixels; boundaries closer than this are one line (thick or double lines)
MIN_SEGMENT_DRAWN = 0.5  # share of a cell side that must be drawn for the side to count as a line


def line_positions(mask: np.ndarray, axis: int) -> list[int]:
    """Find the positions of the lines that cross most of the table.

    Args:
        mask: Binary mask of horizontal lines (axis=1) or vertical lines (axis=0).
        axis: 1 to find horizontal lines (returns y values), 0 for vertical lines (x values).

    Returns:
        Sorted line positions, in pixels.
    """
    # Thicken across the line so a line drifting by a few pixels still piles up in one band.
    kernel = np.ones((5, 1) if axis == 1 else (1, 5), np.uint8)
    coverage = (cv2.dilate(mask, kernel) > 0).mean(axis=axis)

    # Consecutive rows (or columns) above the threshold form one line; keep its centre.
    on = np.flatnonzero(coverage >= MIN_LINE_COVERAGE)
    if on.size == 0:
        return []
    bands = np.split(on, np.flatnonzero(np.diff(on) > 1) + 1)
    positions = [int(band.mean()) for band in bands]

    # Merge lines that are too close together to have a cell between them.
    merged = [positions[0]]
    for position in positions[1:]:
        if position - merged[-1] < MIN_LINE_GAP:
            merged[-1] = (merged[-1] + position) // 2
        else:
            merged.append(position)
    return merged


def side_is_drawn(mask: np.ndarray, fixed: int, start: int, end: int, vertical: bool) -> bool:
    """Check whether a line is drawn along one side of a cell.

    Args:
        mask: Vertical-line mask (for a vertical side) or horizontal-line mask (horizontal side).
        fixed: x of a vertical side, or y of a horizontal side.
        start: Where the side starts (y for a vertical side, x for a horizontal one).
        end: Where the side ends.
        vertical: True for a vertical side.

    Returns:
        True if enough of the side is covered by line pixels.
    """
    # Look in a narrow band around the side, skipping its ends where the crossing lines are.
    band = 3
    margin = (end - start) // 5
    start, end = start + margin, end - margin
    if vertical:
        strip = mask[start:end, max(fixed - band, 0) : fixed + band + 1]
        drawn = strip.any(axis=1)
    else:
        strip = mask[max(fixed - band, 0) : fixed + band + 1, start:end]
        drawn = strip.any(axis=0)
    return drawn.size > 0 and drawn.mean() >= MIN_SEGMENT_DRAWN


def find_cells(gray: np.ndarray) -> dict:
    """Find the grid of a ruled table and its cells, including merged cells.

    Args:
        gray: Grayscale image of the table (a straight, flat crop).

    Returns:
        The table model: number of rows and columns, line positions, and one entry per cell with
        its row, column, spans and pixel box (x0, y0 top-left; x1, y1 bottom-right).
    """
    horizontal, vertical = line_masks(gray)
    row_lines = line_positions(horizontal, axis=1)
    col_lines = line_positions(vertical, axis=0)
    n_rows, n_cols = max(len(row_lines) - 1, 0), max(len(col_lines) - 1, 0)

    # Start with every grid cell on its own, then join neighbours whose shared side is not drawn.
    # group[r, c] says which merged cell grid cell (r, c) belongs to.
    group = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)

    def join(a: int, b: int) -> None:
        group[group == group.flat[b]] = group.flat[a]

    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1, x0, x1 = row_lines[r], row_lines[r + 1], col_lines[c], col_lines[c + 1]
            if c + 1 < n_cols and not side_is_drawn(vertical, x1, y0, y1, vertical=True):
                join(r * n_cols + c, r * n_cols + c + 1)  # no line on the right: join right cell
            if r + 1 < n_rows and not side_is_drawn(horizontal, y1, x0, x1, vertical=False):
                join(r * n_cols + c, (r + 1) * n_cols + c)  # no line below: join cell below

    cells = []
    for label in np.unique(group):
        rows, cols = np.nonzero(group == label)
        row, col = int(rows.min()), int(cols.min())
        row_span, col_span = int(rows.max()) - row + 1, int(cols.max()) - col + 1
        cell = {
            "row": row,
            "col": col,
            "row_span": row_span,
            "col_span": col_span,
            "x0": col_lines[col],
            "y0": row_lines[row],
            "x1": col_lines[col + col_span],
            "y1": row_lines[row + row_span],
        }
        # A merged cell should be a rectangle; if not, a line was probably missed or faded.
        if rows.size != row_span * col_span:
            cell["irregular"] = True
        cells.append(cell)

    cells.sort(key=lambda cell: (cell["row"], cell["col"]))
    cells = [{"id": i, **cell} for i, cell in enumerate(cells)]
    return {"n_rows": n_rows, "n_cols": n_cols, "row_lines": row_lines, "col_lines": col_lines, "cells": cells}


def draw_cells(image: np.ndarray, table: dict, scale: int = 2) -> np.ndarray:
    """Draw every cell's box and number on an enlarged copy of the image.

    Args:
        image: BGR image the cells were found on.
        table: Table model from find_cells().
        scale: Enlargement, so the numbers stay readable on small images.

    Returns:
        The annotated image.
    """
    out = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for cell in table["cells"]:
        merged = cell["row_span"] > 1 or cell["col_span"] > 1
        colour = (0, 140, 255) if merged else (0, 180, 0)  # orange = merged, green = single
        top_left, bottom_right = (cell["x0"] * scale, cell["y0"] * scale), (cell["x1"] * scale, cell["y1"] * scale)
        cv2.rectangle(out, top_left, bottom_right, colour, 2)
        cv2.putText(out, str(cell["id"]), (top_left[0] + 4, top_left[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    return out


if __name__ == "__main__":
    out_dir = Path("results/debug_cells")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'table':<55}{'rows':>5}{'cols':>5}{'cells':>7}{'merged':>8}{'irregular':>11}")
    # One or more table crops, e.g. results/debug_table/*_table.png
    for image_path in map(Path, sys.argv[1:]):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"{image_path.name:<55}could not read")
            continue

        table = {"image": str(image_path), **find_cells(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))}
        (out_dir / f"{image_path.stem}_cells.json").write_text(json.dumps(table, indent=2))
        cv2.imwrite(str(out_dir / f"{image_path.stem}_cells.png"), draw_cells(image, table))

        cells = table["cells"]
        n_merged = sum(cell["row_span"] > 1 or cell["col_span"] > 1 for cell in cells)
        n_irregular = sum(cell.get("irregular", False) for cell in cells)
        print(
            f"{image_path.name:<55}{table['n_rows']:>5}{table['n_cols']:>5}"
            f"{len(cells):>7}{n_merged:>8}{n_irregular:>11}"
        )

    print(f"Saved results to {out_dir}/")
