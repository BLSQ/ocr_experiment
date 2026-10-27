"""Score one approach against the true values and save the result as JSON.

Usage:
    python evaluate.py <approach>                  # uses data/
    python evaluate.py <approach> --data data/demo

<approach> is a file in approaches/ (without .py) that defines
`read_form(photo_path) -> {field: value}`.

The data folder holds `photos/` and `labels.csv`, with one row per field:

    photo,field,value
    form_001.jpg,tested,125
    form_001.jpg,positive,40
"""

import argparse
import csv
import importlib
import json
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")


def load_labels(labels_path: Path) -> dict[str, dict[str, str]]:
    """Read the true values from labels.csv.

    Args:
        labels_path: Path to labels.csv.

    Returns:
        True values per photo: {photo: {field: value}}.
    """
    labels: dict[str, dict[str, str]] = {}
    with labels_path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            labels.setdefault(row["photo"], {})[row["field"]] = row["value"]
    return labels


def normalize(value: str | None) -> str:
    """Clean a value before comparing: ignore extra spaces and upper/lower case.

    Args:
        value: Value to clean; None counts as empty.

    Returns:
        The cleaned value.
    """
    return " ".join(str(value or "").split()).casefold()


def main() -> None:
    """Run the approach on every photo, compare with the labels, save JSON."""
    parser = argparse.ArgumentParser(description="Score one approach.")
    parser.add_argument("approach", help="File name in approaches/, without .py")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()

    approach = importlib.import_module(f"approaches.{args.approach}")
    labels = load_labels(args.data / "labels.csv")

    photos = []
    for photo, truth in labels.items():
        # 1. Let the approach read the photo (a crash counts as no readings).
        start = time.perf_counter()
        try:
            readings = approach.read_form(args.data / "photos" / photo)
            error = None
        except Exception as exc:  # keep going with the other photos
            readings = {}
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - start

        # 2. Compare each field with its true value.
        fields = [
            {
                "field": field,
                "true": true_value,
                "read": readings.get(field),
                "correct": normalize(readings.get(field)) == normalize(true_value),
            }
            for field, true_value in truth.items()
        ]
        photos.append(
            {
                "photo": photo,
                "seconds": round(seconds, 3),
                "error": error,
                "fields": fields,
            }
        )

    # 3. Summarize and save.
    all_fields = [field for photo in photos for field in photo["fields"]]
    n_correct = sum(field["correct"] for field in all_fields)
    summary = {
        "approach": args.approach,
        "data": str(args.data),
        "date": datetime.now().isoformat(timespec="seconds"),
        "accuracy": round(n_correct / len(all_fields), 3) if all_fields else None,
        "correct_fields": n_correct,
        "total_fields": len(all_fields),
        "photos": len(photos),
        "failed_photos": sum(photo["error"] is not None for photo in photos),
        "mean_seconds_per_photo": (
            round(sum(photo["seconds"] for photo in photos) / len(photos), 3)
            if photos
            else None
        ),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = RESULTS_DIR / f"{args.approach}_{stamp}.json"
    output.write_text(
        json.dumps({"summary": summary, "photos": photos}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
