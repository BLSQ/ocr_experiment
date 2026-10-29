# OCR experiment: reading handwritten tables from photos

The goal is an **OCR system that reads photos of paper forms whose values are
written in a table, usually by hand**. Photos can be blurry or unevenly lit, and
there are several versions of the form.

The work is done in two steps:

1. **Segment the table into cells (current step).** Clean up the photo, find the
   table and split it into separate cells, each with its row, column and pixel
   box. This is what the preprocessing in this repo does, and the aim now is to
   find the combination of preprocessing steps that finds the right cells most
   often.
2. **Read each cell (later).** Once the segmentation is reliable, compare
   off-the-shelf pretrained OCR models (e.g. PP-OCR, TrOCR, a digit model) on the
   cell crops, and keep the one that reads the values best.

## What's in the repo

```
ocr_experiment/
├── data/                      not in git (may contain patient data)
│   ├── images/                all real form photos, also as images.tar.gz
│   └── images_selection/      10 photos for testing the preprocessing, and expected.csv
├── preprocessing/             one script per preprocessing step (see below)
├── pipelines.py               combinations of steps to compare
├── evaluate_preprocessing.py  runs the combinations and scores them against expected.csv
├── report.py                  builds the HTML report (run automatically by evaluate_preprocessing.py)
├── evaluate.py                for step 2: scores an OCR approach (not in use yet)
├── make_demo_data.py          for step 2: draws fake forms to try a reader on
├── requirements.txt           libraries the scripts use
├── pyproject.toml             candidate libraries for later, and ruff settings
└── results/                   not in git
    ├── debug_<step>/          images and JSON saved by each step script
    ├── preprocessing/         one JSON file per run of a combination
    └── reports/               preprocessing.html
```

## Setup

Use Python 3.12 or newer, in a virtual environment of your choice, and install
the libraries from the project folder:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

With conda instead: `conda create -n <name> python=3.13`, `conda activate <name>`,
then the same `pip install`.

`rapidocr` downloads its models the first time it runs, so that needs internet.
The photos in `data/` are not in git and have to be shared separately.

Run every command **from the project folder**, so the scripts find `data/` and
`results/`.

## The preprocessing steps

```
photo ─► page crop ─► deskew ─► colour channel ─► contrast ─► deblur ─► table crop ─► cells
```

Every step is optional except cells. A combination picks the steps it uses, in
this order.

| Step | Script | What it does |
|---|---|---|
| Page crop | `page_crop.py` | Cuts the sheet of paper out of the background, flattened to a rectangle. Two methods: `mask` finds the paper pixels (bright, colourless, smooth), for a page on wood or fabric; `edges` finds the page's straight borders, which also works for a page lying on another sheet or running off the photo. |
| Deskew | `deskew.py` | Measures the small tilt from the long straight lines and rotates it away, so table lines are horizontal and vertical. |
| Orientation | `orientation.py` | Deskews, then turns the page by 90° or 180° using the OCR text detector. Photos are expected upright, so it is not used in the current combinations. |
| Colour channel | `channel.py` | Keeps the colour channel (grey, red, green or blue) where the ink stands out most from the paper. Needs a colour image, so it goes before contrast and deblur. |
| Contrast | `contrast.py` | Evens out the lighting (removes shadows) and/or raises the contrast with CLAHE, so faint table lines are not lost. |
| Deblur | `deblur.py` | Sharpens a blurry photo (unsharp mask, CLAHE + unsharp mask, or deconvolution). |
| Table crop | `table_crop.py` | Finds the group of ruled lines with the most crossings (the table) and cuts it out, flattened to a rectangle. |
| Cells | `build_cell_grids.py` | Finds the row and column lines of the table and every cell, including merged cells. |

Two more scripts help with tuning: `page_params.py` compares page-crop settings
side by side, and `text_boxes.py` shows where the PP-OCRv5 detector finds text.
`preprocessing/example.py` is a teaching example for step 2 (it only reads the
fake demo forms).

The result of the cells step is a grid like this, with pixel boxes in the table
crop. A cell spanning several rows or columns is a merged cell:

```json
{
  "n_rows": 8, "n_cols": 5,
  "cells": [
    {"id": 0, "row": 0, "col": 0, "row_span": 2, "col_span": 1, "x0": 20, "y0": 30, "x1": 160, "y1": 115}
  ]
}
```

## Running the tests

### One step at a time

Each step script takes one or more images, prints a few numbers per image, and
saves images to look at in `results/debug_<step>/`. The steps chain through
those files:

```
python preprocessing/page_crop.py data/images_selection/*.png
python preprocessing/contrast.py data/images_selection/*.png
python preprocessing/table_crop.py data/images_selection/*.png
python preprocessing/build_cell_grids.py results/debug_table/*_table.png
```

`deskew.py` and `deblur.py` take one image each.

### Comparing combinations of steps

1. **Define the combinations** in `pipelines.py`. Each is a list of steps run in
   order; a step is its name, or `(name, {settings})`. The settings of each step
   are listed at the top of the file.

   ```python
   PIPELINES = {
       "deskew_table": ["deskew", "table_crop"],
       "deskew_clahe_join": ["deskew", ("contrast", {"method": "clahe"}), ("table_crop", {"join": 9})],
   }
   ```

2. **Give the expected grid** of each test photo in
   `data/images_selection/expected.csv`. Leave a value empty to skip that check
   (`turn` is only checked when the combination uses orientation).

   ```
   photo,n_rows,n_cols,turn
   form_001.png,12,6,
   ```

3. **Run** all combinations, or only the ones named:

   ```
   python evaluate_preprocessing.py
   python evaluate_preprocessing.py table_only page_edges_table
   python evaluate_preprocessing.py --images     # also save each photo's final image and drawn grid
   ```

   Every run ends by rebuilding the HTML report (see below), so there is nothing
   else to run.

## Reading the results

**In the terminal**, `evaluate_preprocessing.py` prints one line per combination,
best first:

| Column | Meaning |
|---|---|
| `grid` | share of photos where both the number of rows and of columns are right |
| `rows`, `cols` | share of photos where that count is right |
| `turn` | share of photos turned the right way (only with orientation) |
| `off r/c` | how many rows / columns the grid is off by on average (lower is better) |
| `failed` | photos where a step crashed (they count as wrong) |
| `s/photo` | mean seconds per photo |

**In `results/preprocessing/<combination>_<time>.json`**, the same summary, plus
for every photo what each step did (e.g. the tilt found, the share of the image
the table covers), how long it took, the grid found and which checks passed.
With `--images`, the folder of the same name has each photo's final image and
its grid drawn on it (green = single cell, orange = merged cell), to check by
eye.

**In the HTML report**, all runs side by side. `evaluate_preprocessing.py`
rebuilds it after every run (`python report.py` rebuilds it by hand, e.g. after
deleting old runs). Open `results/reports/preprocessing.html` in a browser (the
charts load Plotly from the internet). It shows a sortable table of the runs, the accuracy
per combination, accuracy against time, and the time spent in each step. By
default it shows the latest run of each combination; a switch shows every run,
and a menu picks which measure (grid, rows, columns) the charts use.

## Good to know

- **Patient data:** photos, expected values and results may contain patient
  information. They are excluded from git, and they shouldn't be sent to online
  services.
- **Adding a step:** write the step script in `preprocessing/`, then a small
  `step_<name>(image, **settings) -> (image, info)` function in
  `evaluate_preprocessing.py`, and add it to `STEPS`.
- **Libraries:** `requirements.txt` lists what the scripts use. Candidates for
  step 2 are grouped in `pyproject.toml` (install one group with
  `pip install --group <name>`, pip 25.1 or newer). When a script starts using a
  new library, also add it to `requirements.txt`.
- **Code checks:** `ruff check .` finds problems, and `ruff format .` fixes the
  layout.
