"""Run the image-quality pipelines on the same photos and save the results side by side to compare by eye.

Usage (from the project root):
    python evaluate_quality.py                             # every pipeline in QUALITY_PIPELINES
    python evaluate_quality.py deskew_page_flatten page_deskew_flatten
    python evaluate_quality.py --photos data/images_selection
    python evaluate_quality.py --report results/quality/<time>   # only rebuild that run's report.html

The pipelines are in pipelines.py (QUALITY_PIPELINES) and reuse the steps of evaluate_preprocessing.py. Nothing is
scored against expected values: there is no grid or OCR here yet. Instead it prints a table with, per pipeline, the
mean over the photos of these numbers, measured on the final image (↑ = higher is better, ↓ = lower is better):
- sharp ↑: variance of the Laplacian. Noise raises it too, so read it together with noise (see deblur.sharpness()).
- noise ↓: estimated standard deviation of the noise, in grey levels (Immerkær's method, see denoise.noise()).
- lines ↑: how much darker the ruled lines are than the paper, in units of paper noise (see contrast.line_contrast()).
- separ ↑: how well one threshold splits ink from paper: between-class variance / total variance of Otsu's
  threshold, 0 to 1 (1 = only two grey levels, like clean black ink on white paper).
- uneven ↓: how much the paper level changes across the image (shadows, uneven light), in grey levels.
- ink: share of the image darker than Otsu's threshold. Not better or worse by itself, but a pipeline with clearly
  less ink than the others has made strokes thinner or lost faint ones (e.g. strong denoising).
- tilt ↓: the worse of the tilt of the horizontal lines and the lean of the vertical lines, in degrees. Both are
  measured because in a photo taken at an angle they differ, and rotating only fixes one of them.
- cropped ↑: share of photos where the page crop found and cut out the page.
- s/photo: mean seconds per photo; failed: photos where a step crashed.
The originals are measured too, for reference. They still include the background around the page, so their numbers
are not exactly comparable with cropped results.

Saves, in results/quality/<time>/:
- <photo>_strip.png: the original and the result of every pipeline side by side, with their names.
- <pipeline>/<photo>.png: the result of each pipeline, full size.
- summary.json: the steps of each pipeline, what each step did per photo, the numbers and the times.
- report.html: the table of the summary; hover a column for its meaning, or a pipeline for its steps.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from html import escape
from pathlib import Path

import cv2
import numpy as np

# The step scripts import each other by file name (e.g. `from deskew import deskew`), so their folder must be
# on the import path before importing them.
sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))

from evaluate_preprocessing import STEPS, parse_steps, to_json
from pipelines import QUALITY_PIPELINES
from preprocessing.contrast import line_contrast
from preprocessing.deblur import sharpness
from preprocessing.denoise import noise
from preprocessing.deskew import tilt_and_lean

RESULTS_DIR = Path("results/quality")
STRIP_HEIGHT = 400  # height of each image in the side-by-side strip
UNEVEN_TILES = 8  # the paper level is compared across an 8 x 8 grid of tiles

# Columns of the printed table and of report.html: (key, header, number format, description shown on hover).
COLUMNS = [
    (
        "sharpness",
        "sharp↑",
        ".0f",
        "Sharpness: variance of the Laplacian (how strong the edges are). Higher is sharper, but noise raises it "
        "too, so read it together with noise.",
    ),
    (
        "noise",
        "noise↓",
        ".2f",
        "Noise: estimated standard deviation of the noise, in grey levels (Immerkær's method). Lower is cleaner.",
    ),
    (
        "lines",
        "lines↑",
        ".1f",
        "Line contrast: how much darker the ruled table lines are than the paper, in units of the paper's noise. "
        "Higher means the lines stand out more.",
    ),
    (
        "separation",
        "separ↑",
        ".2f",
        "Ink/paper separation: how well one threshold (Otsu) splits ink from paper, from 0 to 1. 1 means only two "
        "grey levels, like clean black ink on white paper.",
    ),
    (
        "uneven",
        "uneven↓",
        ".1f",
        "Unevenness: how much the paper level changes across the image (shadows, uneven light), in grey levels. "
        "Lower is more even. It cannot see differences once the paper is pure white (255), e.g. after sharpening.",
    ),
    (
        "ink",
        "ink",
        ".1%",
        "Ink: share of the image darker than the ink/paper threshold (Otsu). Not better or worse by itself, but a "
        "pipeline with clearly less ink than the others has made strokes thinner or lost faint ones.",
    ),
    (
        "tilt",
        "tilt↓",
        ".2f",
        "Tilt left in the image, in degrees: the worse of the tilt of the horizontal lines and the lean of the "
        "vertical lines. In a photo taken at an angle they differ, and a rotation only fixes one of them.",
    ),
    ("cropped", "cropped↑", ".0%", "Share of photos where the page crop found the page and cut it out."),
    ("seconds", "s/photo", ".2f", "Mean seconds per photo for all the steps of the pipeline."),
    ("failed", "failed", "d", "Number of photos where a step crashed (not included in the means)."),
]


def separation(gray: np.ndarray) -> float:
    """Measure how well one threshold splits the ink from the paper (Otsu's effectiveness).

    Args:
        gray: Grayscale image.

    Returns:
        Between-class variance / total variance at Otsu's threshold, from 0 to 1 (1 = only two grey levels).
    """
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    total = gray.var()
    if total == 0:
        return 0.0
    dark, light = gray[gray <= threshold], gray[gray > threshold]
    if dark.size == 0 or light.size == 0:
        return 0.0
    share_dark = dark.size / gray.size
    between = share_dark * (1 - share_dark) * (float(light.mean()) - float(dark.mean())) ** 2
    return float(between / total)


def unevenness(gray: np.ndarray) -> float:
    """Measure how much the paper level changes across the image.

    The paper level of a tile is its 90th percentile, which ignores the ink (ink covers much less than 10% of a
    tile). Shadows and uneven light make these levels differ between tiles.

    Args:
        gray: Grayscale image.

    Returns:
        Standard deviation of the paper level between tiles, in grey levels.
    """
    rows = np.array_split(gray, UNEVEN_TILES, axis=0)
    levels = [np.percentile(tile, 90) for row in rows for tile in np.array_split(row, UNEVEN_TILES, axis=1)]
    return float(np.std(levels))


def measure(image: np.ndarray) -> dict[str, float]:
    """Measure the quality numbers of an image (see the top of this file).

    Args:
        image: BGR image.

    Returns:
        {"sharpness": ..., "noise": ..., "lines": ..., "separation": ..., "uneven": ..., "ink": ..., "tilt": ...},
        plus "tilt_h" and "lean_v", the tilt of the horizontal and the lean of the vertical lines.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tilt_h, lean_v = tilt_and_lean(gray)
    return {
        "sharpness": round(sharpness(gray), 1),
        "noise": round(noise(gray), 2),
        "lines": round(line_contrast(gray), 1),
        "separation": round(separation(gray), 3),
        "uneven": round(unevenness(gray), 1),
        "ink": round(float((gray <= threshold).mean()), 4),
        "tilt": round(max(abs(tilt_h), abs(lean_v)), 2),
        "tilt_h": round(tilt_h, 2),
        "lean_v": round(lean_v, 2),
    }


def was_cropped(steps_done: list[dict]) -> bool | None:
    """Tell whether the page crop step cut out the page.

    Args:
        steps_done: What each step did, from run_pipeline().

    Returns:
        True if the page was cropped, False if the photo was kept as it was, None if there is no page crop step.
    """
    crops = [step["info"]["status"] == "cropped" for step in steps_done if step["step"] == "page_crop"]
    return crops[-1] if crops else None


def summarize(records: list[dict]) -> dict[str, float]:
    """Average the numbers of one pipeline (or of the originals) over the photos.

    Args:
        records: One record per photo: {"quality": {...} or None, "cropped": ..., "seconds": ..., "error": ...}.

    Returns:
        The mean of every number; cropped is the share of photos cropped. Numbers never measured are left out.
    """
    done = [record for record in records if record["error"] is None]
    summary: dict[str, float] = {"failed": len(records) - len(done)}
    for key in ["sharpness", "noise", "lines", "separation", "uneven", "ink", "tilt"]:
        if done:
            summary[key] = float(np.mean([record["quality"][key] for record in done]))
    for key in ["cropped", "seconds"]:
        values = [record[key] for record in done if record.get(key) is not None]
        if values:
            summary[key] = float(np.mean(values))
    return summary


def run_pipeline(image: np.ndarray, steps: list[tuple[str, dict]]) -> tuple[np.ndarray, list[dict]]:
    """Run the steps of one pipeline in order.

    Args:
        image: BGR photo.
        steps: (name, settings) pairs, in order.

    Returns:
        The final image, and what each step did and how long it took.
    """
    steps_done = []
    for name, settings in steps:
        start = time.perf_counter()
        image, info = STEPS[name](image, **settings)
        steps_done.append({"step": name, "seconds": round(time.perf_counter() - start, 3), "info": info})
    return image, steps_done


def tile(image: np.ndarray, label: str) -> np.ndarray:
    """Resize an image to the strip height and write its label at the top.

    Args:
        image: BGR image.
        label: Text written on the image (the pipeline name).

    Returns:
        The labelled tile.
    """
    width = max(1, round(image.shape[1] * STRIP_HEIGHT / image.shape[0]))
    small = cv2.resize(image, (width, STRIP_HEIGHT), interpolation=cv2.INTER_AREA)
    banner = np.full((24, width, 3), 255, np.uint8)
    cv2.putText(banner, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
    # A thin grey column on the right separates the tiles.
    return cv2.copyMakeBorder(np.vstack([banner, small]), 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(128, 128, 128))


def format_value(value: float | None, fmt: str) -> str:
    """Write a number of the table, or "-" if it was not measured.

    Args:
        value: The number, or None.
        fmt: Python format spec, e.g. ".2f" or ".0%".

    Returns:
        The formatted number.
    """
    return "-" if value is None else format(value, fmt)


def write_report(run_dir: Path) -> None:
    """Write report.html next to summary.json: the summary table, with each column's meaning shown on hover.

    Args:
        run_dir: Folder of one run, results/quality/<time>/, holding summary.json.
    """
    data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    n_photos = len(data["originals"])

    header = "".join(f'<th data-tip="{escape(text)}">{escape(head)}</th>' for _, head, _, text in COLUMNS)
    rows = []
    for name, summary in data["summary"].items():
        # Hovering a pipeline name shows its steps, e.g. "page_crop(method=edges) → deskew → contrast(method=flatten)".
        steps = data["pipelines"].get(name)
        if steps is None:
            hover, css = "The photos as they are, before any step (still with the background around the page).", "ref"
        else:
            hover = " → ".join(
                f"{step}({', '.join(f'{k}={v}' for k, v in settings.items())})" if settings else step
                for step, settings in steps
            )
            css = ""
        cells = "".join(f"<td>{format_value(summary.get(key), fmt)}</td>" for key, _, fmt, _ in COLUMNS)
        rows.append(f'<tr class="{css}"><th data-tip="{escape(hover)}">{escape(name)}</th>{cells}</tr>')

    # The strips saved next to the report (original + every pipeline, side by side), one per photo. They are linked
    # by file name, so the report stays small and only works from its own folder.
    strips = [
        f"<figure><figcaption>{escape(path.name.removesuffix('_strip.png'))}</figcaption>"
        f'<a href="{escape(path.name)}"><img src="{escape(path.name)}" loading="lazy" alt="{escape(path.name)}"></a>'
        "</figure>"
        for path in sorted(run_dir.glob("*_strip.png"))
    ]

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image quality {escape(run_dir.name)}</title>
<style>
  :root {{ --bg: #fff; --fg: #1a1a1a; --muted: #666; --line: #ddd; --ref: #f4f4f4; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #1b1b1b; --fg: #e6e6e6; --muted: #999; --line: #3a3a3a; --ref: #262626; }}
  }}
  body {{ background: var(--bg); color: var(--fg); font: 14px/1.5 system-ui, sans-serif; margin: 24px 16px; }}
  p {{ color: var(--muted); margin: 4px 0 16px; }}
  table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 6px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
  td {{ text-align: right; }}
  thead th {{ text-align: right; text-decoration: underline dotted; }}
  thead th:first-child, tbody th {{ text-align: left; }}
  tbody th {{ font-weight: 500; }}
  tr.ref {{ background: var(--ref); color: var(--muted); }}
  /* Tooltips drawn by the page (CSS only), as some viewers do not show the browser's own title tooltips. */
  [data-tip] {{ position: relative; cursor: help; }}
  [data-tip]:hover::after {{
    content: attr(data-tip); position: absolute; top: 100%; right: 0; z-index: 1; margin-top: 4px;
    width: 300px; padding: 8px 10px; border-radius: 6px; background: var(--fg); color: var(--bg);
    font-weight: 400; font-size: 13px; line-height: 1.4; text-align: left; text-decoration: none;
    white-space: normal; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
  }}
  thead th:first-child:hover::after, tbody th:hover::after {{ left: 0; right: auto; }}
  h2 {{ margin: 32px 0 4px; }}
  figure {{ margin: 0 0 24px; }}
  figcaption {{ color: var(--muted); font-size: 13px; margin-bottom: 4px; }}
  figure img {{ display: block; max-width: 100%; height: auto; border: 1px solid var(--line); }}
</style>
</head>
<body>
<h1>Image quality, run {escape(run_dir.name)}</h1>
<p>{escape(data["date"])} · {n_photos} photos from {escape(data["photos_dir"])} · mean per photo ·
↑ higher is better, ↓ lower is better · hover a column for its meaning, a pipeline for its steps</p>
<table>
<thead><tr><th data-tip="Pipeline name, from QUALITY_PIPELINES in pipelines.py">pipeline</th>{header}</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
<h2>Photos</h2>
<p>Per photo: the original, then the result of each pipeline in the order of the table. Click a strip to open it
full size; each pipeline's result is also saved full size in the folder of the same name.</p>
{chr(10).join(strips) or "<p>No strips in this folder.</p>"}
</body>
</html>
"""
    (run_dir / "report.html").write_text(page, encoding="utf-8")


def main() -> None:
    """Run the chosen pipelines on every photo, save the images and a JSON summary, and print a comparison."""
    parser = argparse.ArgumentParser(description="Compare image-quality preprocessing pipelines.")
    parser.add_argument("pipelines", nargs="*", help="Names from QUALITY_PIPELINES in pipelines.py (default: all)")
    parser.add_argument("--photos", type=Path, default=Path("data/images_selection"))
    parser.add_argument("--report", type=Path, help="Only rebuild report.html of this run folder from its summary.json")
    args = parser.parse_args()

    if args.report:
        write_report(args.report)
        print(f"Saved {args.report / 'report.html'}")
        return

    names = args.pipelines or list(QUALITY_PIPELINES)
    for name in names:
        if name not in QUALITY_PIPELINES:
            sys.exit(f"Unknown pipeline {name!r}. Known pipelines: {', '.join(QUALITY_PIPELINES)}")
    pipelines = {name: parse_steps(QUALITY_PIPELINES[name]) for name in names}

    photo_paths = sorted(p for p in args.photos.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    out_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    for name in names:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    # results[pipeline][photo] = {"steps": [...], "quality": {...}, "cropped": ..., "seconds": ..., "error": ...}
    results: dict[str, dict[str, dict]] = {name: {} for name in names}
    originals: dict[str, dict] = {}
    for photo_path in photo_paths:
        image = cv2.imread(str(photo_path))
        if image is None:
            print(f"{photo_path.name}: could not read")
            continue
        originals[photo_path.name] = {"quality": measure(image), "error": None}
        tiles = [tile(image, "original")]
        for name, steps in pipelines.items():
            try:
                final, steps_done = run_pipeline(image, steps)
                quality = measure(final)
            except Exception as exc:  # keep going with the other pipelines and photos
                results[name][photo_path.name] = {"steps": [], "quality": None, "error": f"{type(exc).__name__}: {exc}"}
                continue
            results[name][photo_path.name] = {
                "steps": steps_done,
                "quality": quality,
                "cropped": was_cropped(steps_done),
                "seconds": round(sum(step["seconds"] for step in steps_done), 3),
                "error": None,
            }
            cv2.imwrite(str(out_dir / name / f"{photo_path.stem}.png"), final)
            tiles.append(tile(final, name))
        cv2.imwrite(str(out_dir / f"{photo_path.stem}_strip.png"), np.hstack(tiles))

    # Mean numbers per pipeline, with the original photos first for reference.
    # No steps ran on the originals, so "failed" means nothing there.
    summaries = {"original": summarize(list(originals.values())) | {"failed": None}}
    for name, photos in results.items():
        summaries[name] = summarize(list(photos.values()))

    print("\n" + f"{'pipeline':<30}" + "".join(f"{header:>10}" for _, header, _, _ in COLUMNS))
    for name, summary in summaries.items():
        print(f"{name:<30}{''.join(f'{format_value(summary.get(key), fmt):>10}' for key, _, fmt, _ in COLUMNS)}")
    print("↑ higher is better, ↓ lower is better (see the top of evaluate_quality.py)")

    output = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "photos_dir": str(args.photos),
        "pipelines": {name: steps for name, steps in pipelines.items()},
        "summary": summaries,
        "originals": originals,
        "photos": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(output, indent=2, default=to_json), encoding="utf-8")
    write_report(out_dir)
    print(f"\nSaved strips, images, summary.json and report.html to {out_dir}/")


if __name__ == "__main__":
    main()
