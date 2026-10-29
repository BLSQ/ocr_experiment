"""Run combinations of preprocessing steps on the same photos and compare how well each finds the table grid.

Usage (from the project root):
    python evaluate_preprocessing.py                       # every combination in pipelines.py
    python evaluate_preprocessing.py orient_table page_orient_table
    python evaluate_preprocessing.py --photos data/images_selection --expected data/images_selection/expected.csv
    python evaluate_preprocessing.py --images              # also save the final image and drawn grid per photo

Each combination runs its steps in order on every photo, then measures the table grid (rows, columns, cells)
on the result. The grid, and the turn found by the orientation step, are compared with expected.csv:

    photo,n_rows,n_cols,turn
    form_001.png,12,6,90

An empty value is not checked. `turn` is the clockwise turn (0, 90, 180 or 270) the photo needs to be upright.
Without expected.csv everything still runs, but nothing is scored.

Saves one JSON file per combination to results/preprocessing/<pipeline>_<time>.json, with the settings,
a summary, and for every photo what each step did, how long it took, the grid found and the checks.
Then rebuilds the HTML report of all runs, results/reports/preprocessing.html (see report.py).
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# The step scripts import each other by file name (e.g. `from deskew import deskew`), so their folder must be
# on the import path before importing them.
sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))

from preprocessing.build_cell_grids import draw_cells, find_cells
from preprocessing.channel import select_channel
from preprocessing.contrast import METHODS as CONTRAST_METHODS
from preprocessing.contrast import line_contrast
from preprocessing.deblur import METHODS, sharpness
from preprocessing.deskew import deskew
from preprocessing.orientation import fix_orientation
from preprocessing.page_crop import crop_page
from preprocessing.table_crop import find_table, warp_to_rectangle
from pipelines import PIPELINES
from report import build_report

RESULTS_DIR = Path("results/preprocessing")
CHECKS = ["grid", "rows", "cols", "turn"]


# --- Steps: each takes a BGR image and its settings, and returns the new image and what it did. ---


def step_page_crop(image: np.ndarray, **settings: float | str) -> tuple[np.ndarray, dict]:
    """Crop the sheet of paper out of the photo.

    Args:
        image: BGR image.
        **settings: Settings for crop_page(): method ("mask" or "edges"), and the settings of that method, e.g.
            max_texture for "mask" or min_step for "edges".

    Returns:
        The page (or the unchanged photo), and the decision and paper area.
    """
    page, _, status, area = crop_page(image, **settings)
    return page, {"status": status, "area": round(area, 3)}


def step_deskew(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Straighten the small tilt.

    Args:
        image: BGR image.

    Returns:
        The straightened image, and the tilt angle that was removed.
    """
    straight, angle = deskew(image)
    return straight, {"tilt": round(angle, 2)}


def step_orientation(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Straighten the tilt and turn the page upright.

    Args:
        image: BGR image.

    Returns:
        The upright image, and the tilt, turn and OCR confidences.
    """
    upright, info = fix_orientation(image)
    return upright, info


def step_deblur(image: np.ndarray, method: str = "clahe_unsharp") -> tuple[np.ndarray, dict]:
    """Sharpen a blurry photo with one of the methods from deblur.py.

    Args:
        image: BGR image.
        method: Name of the method in deblur.METHODS.

    Returns:
        The sharpened image (gray, stored as BGR so later steps work), and the sharpness before and after.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharp = METHODS[method](gray)
    info = {
        "method": method,
        "sharpness_before": round(sharpness(gray), 1),
        "sharpness_after": round(sharpness(sharp), 1),
    }
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR), info


def step_channel(image: np.ndarray, channel: str = "auto") -> tuple[np.ndarray, dict]:
    """Keep only the colour channel where the ink stands out most (see channel.py).

    Args:
        image: BGR image, still in colour.
        channel: "auto" to pick the best channel, or a name from channel.CHANNELS (e.g. "red").

    Returns:
        The chosen channel (stored as BGR so later steps work), its name and the contrast of every channel.
    """
    best, chosen, contrasts = select_channel(image, channel)
    info = {"channel": chosen, "contrast": {name: round(value, 1) for name, value in contrasts.items()}}
    return cv2.cvtColor(best, cv2.COLOR_GRAY2BGR), info


def step_contrast(image: np.ndarray, method: str = "flatten") -> tuple[np.ndarray, dict]:
    """Even out the lighting and raise the contrast with one of the methods from contrast.py.

    Args:
        image: BGR image.
        method: Name of the method in contrast.METHODS.

    Returns:
        The result (gray, stored as BGR so later steps work), and how dark the lines are before and after.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = CONTRAST_METHODS[method](gray)
    info = {
        "method": method,
        "line_contrast_before": round(line_contrast(gray), 1),
        "line_contrast_after": round(line_contrast(result), 1),
    }
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), info


def step_table_crop(
    image: np.ndarray, bridge: int = 0, join: int = 5, merge: float = 0.0, min_area: float = 0.0
) -> tuple[np.ndarray, dict]:
    """Crop the table out of the page and flatten it.

    Args:
        image: BGR image.
        bridge: Gaps along a line closed before finding lines (see table_crop.find_table()).
        join: Thickening that joins lines that almost touch (see table_crop.find_table()).
        merge: Also keep groups with this share of the best group's crossings (see table_crop.find_table()).
        min_area: If the table covers less than this share of the image, it is probably a piece of the table, so
            the image is kept as it is.

    Returns:
        The table (or the unchanged image), whether it was found, and the share of the image it covers.
    """
    found = find_table(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), bridge=bridge, join=join, merge=merge)
    if found is None:
        return image, {"found": False}
    corners, n_crossings = found
    area = cv2.contourArea(corners) / (image.shape[0] * image.shape[1])
    info = {"found": True, "crossings": n_crossings, "area": round(area, 3)}
    if area < min_area:
        return image, {**info, "kept": "table too small"}
    return warp_to_rectangle(image, corners), info


STEPS = {
    "page_crop": step_page_crop,
    "deskew": step_deskew,
    "orientation": step_orientation,
    "deblur": step_deblur,
    "channel": step_channel,
    "contrast": step_contrast,
    "table_crop": step_table_crop,
}


# --- Running and scoring ---


def parse_steps(pipeline: list) -> list[tuple[str, dict]]:
    """Turn a pipeline from pipelines.py into (name, settings) pairs and check every name exists.

    Args:
        pipeline: List of step names or (name, {settings}) pairs.

    Returns:
        One (name, settings) pair per step.
    """
    steps = [(step, {}) if isinstance(step, str) else (step[0], step[1]) for step in pipeline]
    for name, _ in steps:
        if name not in STEPS:
            sys.exit(f"Unknown step {name!r}. Known steps: {', '.join(STEPS)}")
    return steps


def load_expected(path: Path) -> dict[str, dict[str, int]]:
    """Read the expected values per photo; empty cells are left out.

    Args:
        path: Path to expected.csv.

    Returns:
        {photo: {"n_rows": ..., "n_cols": ..., "turn": ...}}, empty if the file does not exist.
    """
    if not path.exists():
        print(f"No {path}: nothing will be scored.")
        return {}
    with path.open(encoding="utf-8") as file:
        return {
            row["photo"]: {key: int(value) for key, value in row.items() if key != "photo" and value and value.strip()}
            for row in csv.DictReader(file)
        }


def check(result: dict | None, steps_done: list[dict], expected: dict, has_orientation: bool) -> dict[str, bool]:
    """Compare what was found with the expected values; only values that are known are checked.

    Args:
        result: Grid found (n_rows, n_cols, ...), or None if the photo failed.
        steps_done: What each step did, to find the turn from the orientation step.
        expected: Expected values for this photo.
        has_orientation: Whether the pipeline has an orientation step (otherwise the turn is not checked).

    Returns:
        {check name: correct}, e.g. {"rows": True, "cols": False, "grid": False, "turn": True}.
    """
    found = result or {}
    checks = {}
    if "n_rows" in expected:
        checks["rows"] = found.get("n_rows") == expected["n_rows"]
    if "n_cols" in expected:
        checks["cols"] = found.get("n_cols") == expected["n_cols"]
    if "rows" in checks and "cols" in checks:
        checks["grid"] = checks["rows"] and checks["cols"]
    if "turn" in expected and has_orientation:
        turns = [step["info"].get("turn_clockwise") for step in steps_done if step["step"] == "orientation"]
        checks["turn"] = bool(turns) and turns[-1] == expected["turn"] and result is not None
    return checks


def run_photo(photo_path: Path, steps: list[tuple[str, dict]]) -> tuple[dict, np.ndarray | None, dict | None]:
    """Run every step on one photo, then measure the table grid.

    Args:
        photo_path: Photo to process.
        steps: (name, settings) pairs, in order.

    Returns:
        The record for the JSON (steps, result, error), the final image and the grid (both None on failure).
    """
    steps_done: list[dict] = []
    record = {"steps": steps_done, "result": None, "error": None}
    try:
        image = cv2.imread(str(photo_path))
        if image is None:
            raise ValueError("could not read the image")
        for name, settings in steps:
            start = time.perf_counter()
            image, info = STEPS[name](image, **settings)
            steps_done.append({"step": name, "seconds": round(time.perf_counter() - start, 3), "info": info})

        start = time.perf_counter()
        table = find_cells(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        steps_done.append({"step": "measure_grid", "seconds": round(time.perf_counter() - start, 3), "info": {}})
    except Exception as exc:  # keep going with the other photos
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record, None, None

    record["result"] = {
        "n_rows": table["n_rows"],
        "n_cols": table["n_cols"],
        "n_cells": len(table["cells"]),
        "width": image.shape[1],
        "height": image.shape[0],
    }
    return record, image, table


def summarize(photos: dict[str, dict]) -> dict:
    """Score a pipeline over all photos: share correct per check, failures and time.

    Args:
        photos: Record per photo, as built in main().

    Returns:
        The summary: share correct per check (None if never checked), failed photos and mean seconds.
    """
    summary: dict = {"photos": len(photos), "failed_photos": sum(p["error"] is not None for p in photos.values())}
    for name in CHECKS:
        results = [p["checks"][name] for p in photos.values() if name in p["checks"]]
        summary[f"{name}_correct"] = round(sum(results) / len(results), 3) if results else None
        summary[f"{name}_checked"] = len(results)

    # How far off the counts are on average, so pipelines that are close but not exact can be told apart.
    # A failed photo counts as 0 rows and 0 columns found.
    for name, key in [("rows", "n_rows"), ("cols", "n_cols")]:
        errors = [
            abs((p["result"] or {}).get(key, 0) - p["expected"][key]) for p in photos.values() if key in p["expected"]
        ]
        summary[f"{name}_mean_error"] = round(sum(errors) / len(errors), 2) if errors else None

    step_seconds: dict[str, list[float]] = {}
    for record in photos.values():
        for step in record["steps"]:
            step_seconds.setdefault(step["step"], []).append(step["seconds"])
    totals = [sum(step["seconds"] for step in p["steps"]) for p in photos.values()]
    summary["mean_seconds_per_photo"] = round(sum(totals) / len(totals), 3) if totals else None
    summary["mean_seconds_per_step"] = {name: round(sum(s) / len(s), 3) for name, s in step_seconds.items()}
    return summary


def to_json(value: object) -> object:
    """Convert numpy numbers (e.g. from step info) so json.dumps can write them.

    Args:
        value: Value json.dumps does not know.

    Returns:
        A plain Python number, or the value as text.
    """
    return value.item() if isinstance(value, np.generic) else str(value)


def main() -> None:
    """Run the chosen pipelines on every photo, save one JSON per pipeline and print a comparison."""
    parser = argparse.ArgumentParser(description="Compare combinations of preprocessing steps.")
    parser.add_argument("pipelines", nargs="*", help="Names from pipelines.py (default: all)")
    parser.add_argument("--photos", type=Path, default=Path("data/images_selection"))
    parser.add_argument("--expected", type=Path, help="Expected values (default: <photos>/expected.csv)")
    parser.add_argument("--images", action="store_true", help="Also save the final image and drawn grid per photo")
    args = parser.parse_args()

    names = args.pipelines or list(PIPELINES)
    for name in names:
        if name not in PIPELINES:
            sys.exit(f"Unknown pipeline {name!r}. Known pipelines: {', '.join(PIPELINES)}")
    pipelines = {name: parse_steps(PIPELINES[name]) for name in names}

    photo_paths = sorted(p for p in args.photos.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    expected = load_expected(args.expected or args.photos / "expected.csv")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    summaries = {}
    for name, steps in pipelines.items():
        print(f"Running {name} on {len(photo_paths)} photos ...")
        has_orientation = any(step == "orientation" for step, _ in steps)
        image_dir = RESULTS_DIR / f"{name}_{stamp}"
        if args.images:
            image_dir.mkdir(exist_ok=True)

        photos = {}
        for photo_path in photo_paths:
            record, image, table = run_photo(photo_path, steps)
            record["expected"] = expected.get(photo_path.name, {})
            record["checks"] = check(record["result"], record["steps"], record["expected"], has_orientation)
            photos[photo_path.name] = record
            if args.images and image is not None:
                cv2.imwrite(str(image_dir / f"{photo_path.stem}_final.png"), image)
                cv2.imwrite(str(image_dir / f"{photo_path.stem}_cells.png"), draw_cells(image, table))

        summaries[name] = summarize(photos)
        output = {
            "pipeline": name,
            "steps": steps,
            "photos_dir": str(args.photos),
            "date": datetime.now().isoformat(timespec="seconds"),
            "summary": summaries[name],
            "photos": photos,
        }
        path = RESULTS_DIR / f"{name}_{stamp}.json"
        path.write_text(json.dumps(output, indent=2, default=to_json), encoding="utf-8")

    # Comparison table, best grid score first (pipelines never scored go last).
    print(f"\n{'pipeline':<30}{'grid':>7}{'rows':>7}{'cols':>7}{'turn':>7}{'off r/c':>10}{'failed':>8}{'s/photo':>9}")
    by_grid = sorted(
        summaries.items(), key=lambda item: -1 if item[1]["grid_correct"] is None else item[1]["grid_correct"]
    )
    for name, summary in reversed(by_grid):
        scores = "".join(
            f"{summary[f'{c}_correct']:>7.0%}" if summary[f"{c}_correct"] is not None else f"{'-':>7}" for c in CHECKS
        )
        off = f"{summary['rows_mean_error'] or 0:.1f}/{summary['cols_mean_error'] or 0:.1f}"
        print(f"{name:<30}{scores}{off:>10}{summary['failed_photos']:>8}{summary['mean_seconds_per_photo'] or 0:>9.2f}")
    print(f"\nSaved results to {RESULTS_DIR}/ (files ending in _{stamp})")
    build_report()


if __name__ == "__main__":
    main()
