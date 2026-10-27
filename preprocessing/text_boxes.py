"""Find where text is in an image and group the text boxes into rows and columns.

Unlike build_cell_grids.py, this does not rely on the table lines. Run from the project root, on upright pages or
table crops:
    python preprocessing/text_boxes.py results/debug_orientation/*_upright.png
    python preprocessing/text_boxes.py results/debug_table/*_table.png

Steps:
1. Text detection: the PP-OCRv5 text detector (DBNet, through rapidocr) marks every pixel that
   looks like text, then draws a box around each patch. Only detection runs, no reading.
2. Rows: boxes whose vertical centres are close (within half a typical box height) share a row.
3. Columns: boxes whose horizontal centres are close (within a typical box height) share a column.
   This is rougher than the rows: long labels and short numbers in one column have different
   centres.

For each image it saves to results/debug_text_boxes/: a JSON file with every box (pixel box,
corners, detector score, row and column), and the image with the boxes drawn, one colour per row.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr import ModelType, OCRVersion, RapidOCR

ENGINE = RapidOCR(
    params={
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_type": ModelType.MOBILE,
    }
)
BOX_THRESH = 0.5  # detector confidence needed to keep a box; lower finds fainter writing
UNCLIP_RATIO = 1.6  # how much each box is grown around its text; lower keeps neighbours apart


def detect_text(image: np.ndarray) -> list[dict]:
    """Find the text boxes in an image.

    Args:
        image: BGR image.

    Returns:
        One dict per box: pixel box (x0, y0 top-left; x1, y1 bottom-right), its 4 corners
        and the detector score.
    """
    result = ENGINE(image, use_det=True, use_cls=False, use_rec=False, box_thresh=BOX_THRESH, unclip_ratio=UNCLIP_RATIO)
    if result.boxes is None:
        return []

    boxes = []
    for corners, score in zip(result.boxes, result.scores, strict=True):
        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        boxes.append(
            {
                "x0": int(x0),
                "y0": int(y0),
                "x1": int(x1),
                "y1": int(y1),
                "corners": corners.astype(int).tolist(),
                "score": round(float(score), 3),
            }
        )
    return boxes


def group_positions(centres: np.ndarray, tolerance: float) -> np.ndarray:
    """Give the same group number to positions that lie close together along one axis.

    Args:
        centres: Box centres along one axis (y for rows, x for columns).
        tolerance: Largest distance from a group's mean for a centre to join that group.

    Returns:
        Group number of each centre, numbered from top to bottom (or left to right).
    """
    order = np.argsort(centres)
    groups = np.zeros(len(centres), int)
    members = [centres[order[0]]]  # centres in the current group
    group = 0
    for index in order[1:]:
        if centres[index] - np.mean(members) > tolerance:
            group += 1
            members = []
        members.append(centres[index])
        groups[index] = group
    return groups


def assign_rows_and_columns(boxes: list[dict]) -> None:
    """Add a row and column number to every box (in place).

    Args:
        boxes: Boxes from detect_text().
    """
    if not boxes:
        return
    x_centres = np.array([(box["x0"] + box["x1"]) / 2 for box in boxes])
    y_centres = np.array([(box["y0"] + box["y1"]) / 2 for box in boxes])
    # A typical text height sets how close is "the same row/column", whatever the image size.
    height = float(np.median([box["y1"] - box["y0"] for box in boxes]))

    rows = group_positions(y_centres, tolerance=0.5 * height)
    cols = group_positions(x_centres, tolerance=1.0 * height)
    for box, row, col in zip(boxes, rows, cols, strict=True):
        box["row"], box["col"] = int(row), int(col)


def draw_boxes(image: np.ndarray, boxes: list[dict], scale: int = 2) -> np.ndarray:
    """Draw every box with its number on an enlarged copy of the image, one colour per row.

    Args:
        image: BGR image the boxes were found on.
        boxes: Boxes with row numbers.
        scale: Enlargement, so the numbers stay readable on small images.

    Returns:
        The annotated image.
    """
    palette = [(0, 0, 230), (0, 160, 0), (230, 0, 0), (0, 140, 255), (200, 0, 200), (0, 170, 170)]
    out = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for box in boxes:
        colour = palette[box["row"] % len(palette)]
        corners = np.array(box["corners"]) * scale
        cv2.polylines(out, [corners.astype(np.int32)], isClosed=True, color=colour, thickness=2)
        label_at = (box["x0"] * scale, box["y0"] * scale - 3)
        cv2.putText(out, str(box["id"]), label_at, cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    return out


if __name__ == "__main__":
    out_dir = Path("results/debug_text_boxes")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'image':<55}{'boxes':>6}{'rows':>6}{'cols':>6}{'mean score':>12}")
    # One or more images, e.g. results/debug_orientation/*_upright.png
    for image_path in map(Path, sys.argv[1:]):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"{image_path.name:<55}could not read")
            continue

        boxes = detect_text(image)
        assign_rows_and_columns(boxes)
        boxes.sort(key=lambda box: (box["row"], box["col"]))
        boxes = [{"id": i, **box} for i, box in enumerate(boxes)]

        result = {"image": str(image_path), "boxes": boxes}
        (out_dir / f"{image_path.stem}_boxes.json").write_text(json.dumps(result, indent=2))
        cv2.imwrite(str(out_dir / f"{image_path.stem}_boxes.png"), draw_boxes(image, boxes))

        n_rows = len({box["row"] for box in boxes})
        n_cols = len({box["col"] for box in boxes})
        mean_score = np.mean([box["score"] for box in boxes]) if boxes else 0.0
        print(f"{image_path.name:<55}{len(boxes):>6}{n_rows:>6}{n_cols:>6}{mean_score:>12.2f}")

    print(f"Saved results to {out_dir}/")
