# OCR experiment: reading handwritten forms from photos

An exploratory experiment to find a **lightweight way to read photos of paper
forms filled in by hand**. The plan is simple: try several OCR approaches on the
same photos, score each one, and compare.

How and where the result will be used is not decided yet. This project only
answers: *which approach reads our forms best?*

## The idea in three steps

1. **Collect photos and their true values.** A table that says, for example,
   "photo 1: tested = 125, positive = 40".
2. **Let an approach read the photos.** For example Tesseract, or TrOCR on each
   box. It looks at each photo and says what it sees.
3. **Count how often it was right.** Compare its readings with the true values
   and get a score, e.g. "82% of fields correct".

Repeat step 2 with other approaches, and compare the scores.

## What's in the project

```
ocr_experiment/
├── data/                    not committed (may contain patient data)
│   ├── images/              all real form photos (4175), also as images.tar.gz
│   ├── images_selection/    11 real photos used to test the preprocessing steps
│   └── demo/                fake forms, created by make_demo_data.py when needed
├── preprocessing/
│   ├── example.py           teaching example (demo forms only)
│   └── <step>.py            preprocessing experiments, one step per file (see below)
├── pipelines.py             combinations of preprocessing steps to compare
├── evaluate_preprocessing.py  runs the combinations and scores them (see below)
├── evaluate.py              scores an approach and saves a JSON file
├── try_ppocr_whole_page.py  first try at reading a whole page with PP-OCRv5
├── make_demo_data.py        creates the fake demo forms
└── results/                 not committed
    ├── <approach>_<time>.json   one file per evaluation
    ├── preprocessing/           one JSON file per step combination
    └── debug_<step>/            images and JSON saved by each preprocessing script
```

There is no `labels.csv` for the real photos yet, so real photos can't be scored
with `evaluate.py` yet.

## Setup

Everything runs in the conda environment **`env_test`**. VS Code terminals in
this project open with it already active, and the debugger uses it too.

Run commands **from the project folder**, so the scripts find `approaches/`,
`data/` and `results/`.

## Try it in two commands

```
python make_demo_data.py
python evaluate.py example --data data/demo
```

The first command draws 20 fake forms with three number boxes (tested, positive,
negative) into `data/demo/`, some of them blurred or speckled like a bad photo.
The second runs the example approach on them and prints a summary like this:

```json
{
  "approach": "example",
  "accuracy": 0.867,
  "correct_fields": 52,
  "total_fields": 60,
  "photos": 20,
  "failed_photos": 0,
  "mean_seconds_per_photo": 0.005
}
```

## Preprocessing experiments (14 October 2026)

Before reading any values, a photo has to be turned into something easy to read:
the page cut out, turned the right way up, the table found and split into cells.
Each step is a separate script in `preprocessing/`, so it can be tested and tuned on
its own.

### The steps, in order

```
photo ─► page crop ─► orientation (tilt + quarter turns) ─► table crop ─► cells / text boxes
                                                   (deblur: separate, optional)
```

| Step | Script | What it does | Saves to |
|---|---|---|---|
| Page crop | `page_crop.py` | Finds the sheet of paper (bright, colourless, smooth pixels), fits 4 corners and flattens it. Keeps the photo as it is if it already shows only paper. | `results/debug_page/` |
| Page crop settings | `page_params.py` | Tries every combination of the page-crop settings and makes one contact sheet per photo to compare them. | `results/debug_page_params/` |
| Tilt | `deskew.py` | Measures the small tilt (up to ±45°) from the long straight lines and rotates it away. | `results/debug_deskew/` |
| Orientation | `orientation.py` | Fixes the tilt, then turns the page 90° if the text runs top-to-bottom, and 180° if the OCR reads it better upside down. | `results/debug_orientation/` |
| Colour channel | `channel.py` | Measures how well the writing stands out from the paper in grey, red, green, blue and Lab b*, and keeps the best of grey/red/green/blue (b* can erase black lines). Needs a colour image, so it goes after page crop and before deblur. | `results/debug_channel/` |
| Deblur | `deblur.py` | Compares three ways to sharpen a blurry photo: unsharp mask, CLAHE + unsharp mask, deconvolution. | `results/debug_deblur/` |
| Table crop | `table_crop.py` | Finds the group of ruled lines with the most crossings (the table), and flattens it to a rectangle. | `results/debug_table/` |
| Cells | `build_cell_grids.py` | Finds the row and column lines of the table and every cell, including merged cells. Saves the grid as JSON. | `results/debug_cells/` |
| Text boxes | `text_boxes.py` | Finds where text is with the PP-OCRv5 text detector (no reading) and groups the boxes into rows and columns. | `results/debug_text_boxes/` |

`try_ppocr_whole_page.py` is the first try at reading a whole page with PP-OCRv5.
Its last line uses `crop`, which isn't defined yet, so that line fails.

All step scripts take one or more images, so a whole folder can be run at once:

```
python preprocessing/page_crop.py data/images_selection/*.png
python preprocessing/orientation.py data/images_selection/*.png
python preprocessing/table_crop.py results/debug_orientation/*_upright.png
python preprocessing/build_cell_grids.py results/debug_table/*_table.png
python preprocessing/text_boxes.py results/debug_orientation/*_upright.png
```

`deskew.py` and `deblur.py` take one image each.

### Comparing step combinations

Instead of running the steps one by one, `evaluate_preprocessing.py` runs whole
combinations of steps on the same photos and scores how well each finds the table.

1. **Define the combinations** in `pipelines.py`. Each one is a list of steps, run
   in order. A step is its name, or `(name, {settings})`:

   ```python
   PIPELINES = {
       "orient_table": ["orientation", "table_crop"],
       "page_deblur_orient_table": ["page_crop", ("deblur", {"method": "clahe_unsharp"}), "orientation", "table_crop"],
   }
   ```

   The steps are `page_crop`, `deskew`, `orientation`, `deblur` and `table_crop`.
   After the last step, the table grid is always measured with `build_cell_grids.py`.

2. **Fill in what's expected** in `data/images_selection/expected.csv` (a template with
   the 11 photo names is there). `turn` is the clockwise turn the photo needs to be
   upright (0, 90, 180 or 270). Leave a value empty to skip that check.

   ```
   photo,n_rows,n_cols,turn
   form_001.png,12,6,90
   ```

3. **Run** (all combinations, or only the ones named):

   ```
   python evaluate_preprocessing.py
   python evaluate_preprocessing.py orient_table page_orient_table
   python evaluate_preprocessing.py --images     # also save the final image and drawn grid per photo
   ```

It prints one row per combination, best first: the share of photos where the grid
(rows and columns), the rows, the columns and the turn are right, failed photos,
and seconds per photo. Each combination is also saved to
`results/preprocessing/<pipeline>_<time>.json`, with what every step did on every
photo (e.g. the page-crop decision, the tilt, whether a table was found), how long it
took, the grid found and the checks. A photo where a step crashes is saved with its
error, and its checks count as wrong.

To add a step, write a small function `step_<name>(image, **settings) -> (image, info)`
in `evaluate_preprocessing.py` and add it to `STEPS`.

### The cell model

`build_cell_grids.py` saves one JSON file per table. Each cell has its row and column,
how many rows and columns it spans (for merged cells), and its box: top-left
(`x0`, `y0`) and bottom-right (`x1`, `y1`) pixel, in the table crop.

```json
{
  "n_rows": 8, "n_cols": 5,
  "row_lines": [30, 75, 115, ...],
  "col_lines": [20, 160, 250, ...],
  "cells": [
    {"id": 0, "row": 0, "col": 0, "row_span": 2, "col_span": 1, "x0": 20, "y0": 30, "x1": 160, "y1": 115},
    ...
  ]
}
```

A cell marked `"irregular": true` has a merged shape that isn't a rectangle,
which usually means a line was missed.

### What we found

Every step was first tested on synthetic images with a known answer, then run
on the 11 photos in `data/images_selection/`. The results on real photos below
are counts only. Whether each crop or grid is actually right still needs to be
checked by eye in the `results/debug_*` folders.

- **The photos are small:** at most 512 pixels on the longest side, both in
  `images_selection/` and in `images/`. Finding the page and the table works at
  this size, but handwritten digits may be only a few pixels tall, which will
  limit any reader. Full-resolution originals would help, if they exist.
- **Tilt:** synthetic tilts from −12° to +20° were measured within 0.15°. The
  real photos are tilted between −2.5° and +2.5°.
- **Orientation:** 24 of 24 synthetic cases were right (0°, 90°, 180°, 270°,
  with and without tilt). On the real photos: 9 left as they were, 1 turned 180°
  (`G932UjO6yne_…`) and 1 turned 90° (`GA49mHUaOF1_2026-05-28_…`). Three
  decisions were close calls, because the OCR is much less confident on real
  handwriting (0.6–0.8) than on printed demo text (0.99).
- **Deblur:** the sharpness score (variance of the Laplacian) also rises with
  noise, so it can't pick the best method by itself. The methods should be
  compared by how well the OCR reads their output.
- **Page crop:** synthetic page corners were found within 1–4 pixels. On the
  real photos, 10 were cropped and 1 was kept as all paper. Three need a look:
  - `GB2wYkCK6WB_…`: the dense table breaks the paper into pieces. A larger
    hole-filling setting (`close_fraction` 1/10) brings the page from 24% to
    83% of the photo.
  - `GA49mHUaOF1_2026-06-18_…`: the paper is tinted, so a strict colour limit
    rejects it.
  - `GhDQvinSZTM_…`: no setting fixes it (33–54% paper), possibly a shadow.
    Evening out the lighting before thresholding is the next thing to try.
- **Table crop:** on a synthetic photo it found all 42 line crossings, and the
  corners within about 6 pixels. On most real photos the "table" covers 84–96%
  of the image. That may be right, or the lines may merge with everything else.
  This hasn't been checked yet.
- **Cells:** exact on a synthetic table with merged headers. On the real table
  crops the grids are clearly wrong: many false merged cells, and some tables
  with almost no grid. Probable causes are wrong table crops, faint or broken
  lines at 512 pixels, or paper that isn't flat.
- **Text boxes:** printed text was found reliably on the synthetic table. The
  real pages give 21–150 boxes each. Whether the handwritten values are among
  them still needs checking.

### Next steps

1. **Check the debug images step by step,** starting with the page and table
   crops, since every later step depends on them.
2. **Tune the page crop** with `page_params.py`, and decide on new
   defaults.
3. **Form-version templates.** There are at least 3 form versions. Defining
   each version's grid once, with a field name for each cell, would make the
   cell step much more reliable. The detected grid can then identify which
   version a photo is.
4. **Combine text boxes and cells:** put each text box in the cell that contains
   its centre. This finds the cells with a value to read, and points to lines
   the grid missed.
5. **Read each cell** with a pretrained model (PP-OCRv5 recognition, TrOCR, an
   MNIST digit model), and write `labels.csv` for the real photos so the
   readers can be scored with `evaluate.py`.

## The data format

The true values go in a `labels.csv` file, with one row per field:

```
photo,field,value
form_001.jpg,tested,125
form_001.jpg,positive,40
form_002.jpg,tested,80
```

- `photo` is the file name inside the photos folder.
- `field` is the name of the box on the form. Use the same names in your
  approaches.
- `value` is what is really written there, as a person reads it.

By default `evaluate.py` looks for `data/photos/` and `data/labels.csv`. Only
photos listed in `labels.csv` are evaluated.

## Writing an approach

An approach is a file in `approaches/` with **one function**, `read_form`. It
receives the path of a photo and returns what it read, as a dictionary of
field → value:

```python
# approaches/my_idea.py
from pathlib import Path


def read_form(photo_path: Path) -> dict[str, str]:
    """Read every field of one form."""
    # ... open the photo, find the boxes, read them ...
    return {"tested": "125", "positive": "40"}
```

How it reads the photo is completely up to that file. Then score it:

```
python evaluate.py my_idea
```

A few rules about scoring:

- A field counts as correct when the value read equals the true value,
  ignoring extra spaces and upper/lower case.
- A field the approach didn't return counts as wrong.
- If the approach crashes on a photo, the error is saved, that photo's fields
  count as wrong, and the other photos still run.

## Reading the results

Each evaluation saves a file like `results/example_20261009-175739.json`. It has
two parts.

**`summary`** is the score shown above: accuracy, how many fields were right,
how many photos failed, and how long one photo takes on average.

**`photos`** lists every photo and field, with the true value next to what was
read:

```json
{
  "photo": "form_000.png",
  "seconds": 0.028,
  "error": null,
  "fields": [
    { "field": "tested",   "true": "340", "read": "34",  "correct": false },
    { "field": "positive", "true": "217", "read": "217", "correct": true }
  ]
}
```

This is where you learn *why* an approach fails. In the line above, the example
read "340" as "34": the photo was blurred, the "4" and "0" ran into each other,
and the two digits were read as one.

## How the example approach works

`approaches/example.py` only works on the fake demo forms. It can't read
handwriting. It's there to show the shape of an approach, and it uses ideas
that real approaches will need too:

1. **Open the photo** in grayscale and remove small specks with a median
   filter.
2. **Cut out each box.** The positions are written at the top of the file in
   `FIELD_BOXES`, as (x, y, width, height). This only works because the demo
   forms are perfectly straight. With real photos, the photo will first need
   to be aligned to a blank form.
3. **Separate ink from paper**, so ink becomes white and paper black (Otsu
   threshold).
4. **Find each separate blob of ink.** Each blob is treated as one digit, read
   from left to right. Tiny blobs are ignored as specks.
5. **Compare each blob with the digits 0–9** drawn in the same font as the demo
   forms, and keep the best match.

Step 4 is why blurred forms fail: when two digits touch, they become one blob.

## Ideas for real approaches

- **Crop each box, then read it with a pretrained model:**
  - TrOCR handwritten (`microsoft/trocr-small-handwritten`): reads a whole box
    at once, so digits don't need to be separated.
  - PaddleOCR PP-OCRv5 mobile models, through `rapidocr`: light, and also read
    printed labels. `rapidocr` also offers the newer PP-OCRv6 models (its
    default).
  - A pretrained MNIST digit model (ONNX): the lightest option, but each digit
    must be separated first.
- **Whole-page OCR engines as a baseline:** Tesseract, PaddleOCR, EasyOCR,
  docTR.
- **Check results with known data:** e.g. positive can't exceed tested, and
  totals must add up. This can catch some mistakes.

The target is about **85–90% of fields correct** on real photos. No model
training is planned: only existing pretrained models.

The candidate libraries are listed in `pyproject.toml`, in groups. Install one
group with `pip install --group <name>`. Groups: `core`, `onnx`, `ocr`, `hf`,
`postprocess`, `analysis`, `dev`. So far `core` (OpenCV, numpy, Pillow,
scikit-image), `dev` (ruff), `onnxruntime` and `rapidocr` are installed.

## Good to know

- **Patient data:** photos, labels and results may contain patient
  information. They are excluded from git, and they shouldn't be sent to
  online services.
- **Code checks:** `ruff check .` finds problems, and `ruff format .` fixes
  the layout.
