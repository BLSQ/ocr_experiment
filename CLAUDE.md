# CLAUDE.md

This file gives Claude Code context for working in this repository.

## Project: Reading handwritten paper forms from photos (exploratory experiment)

An exploratory experiment to find a **lightweight way to read photos of paper
forms filled in by hand**. Photos can be blurry, layouts differ between form
versions, and handwriting can be hard to read even for a person.

The work is **trying several OCR approaches and comparing how accurate they
are** on the same photos. How and where the result will be used (device, app,
integrations) is **not decided**. Don't assume a deployment target or a
destination system for the extracted data.

### Keep it simple

The user wants a **simple setup**, not a framework. An earlier version with
packages, registries, shared data types and tests was removed as too
complicated. Don't reintroduce that kind of structure:

- One file per approach, one evaluation script, results as JSON files.
- Add a shared helper file (e.g. `common.py`) only once two approaches need
  the same code.
- Prefer clear, short, well-commented code over abstractions.

### Status

- `approaches/example.py` is a teaching example that only works on the demo
  forms (printed digits, made by `make_demo_data.py`). No real reading
  approach (`read_form`) exists yet.
- **Real photos:** `data/images/` (4175 photos, also as `data/images.tar.gz`)
  and `data/images_selection/` (11 photos the user evaluates preprocessing
  on). They are at most **512 px** on the longest side. There is no
  `labels.csv` for them yet. `data/demo/` was removed; regenerate it with
  `make_demo_data.py` (it writes to `data/demo/` relative to the working
  directory, so run it from the scratchpad for throwaway tests).
- **Preprocessing experiments** (`preprocessing/*.py`) exist for each step,
  in this order: page crop → orientation (tilt + 90°/180° turns) → table crop
  → cells (grid from ruled lines, merged cells) / text boxes (PP-OCRv5
  detector). Deblur is a separate optional test. All work on synthetic tests;
  on real photos, page crop needs tuning and the cell grid is not right yet.
  None of the real-photo outputs has been checked by eye yet. See the
  README's "Preprocessing experiments" section for results and next steps.
- The forms are ruled tables with some merged cells and text outside the
  table to ignore. There are at least 3 form versions; per-version templates
  (grid + field name per cell) are a planned next step.

## Project structure

```
data/                   gitignored
  images/               all real form photos
  images_selection/     11 real photos used to test preprocessing
  demo/                 fake forms from make_demo_data.py (only when generated)
preprocessing/
  example.py            teaching example approach (read_form)
  <step>.py             preprocessing experiments, one step per file
pipelines.py            combinations of preprocessing steps (PIPELINES dict)
evaluate_preprocessing.py  runs and scores the combinations in pipelines.py
evaluate.py             scores one approach, saves JSON to results/
try_ppocr_whole_page.py first try at reading a whole page with PP-OCRv5
make_demo_data.py       creates demo forms to try things before real photos
results/                gitignored
  <approach>_<timestamp>.json
  preprocessing/        <pipeline>_<timestamp>.json from evaluate_preprocessing.py
  debug_<step>/         outputs of preprocessing/<step>.py
```

## Preprocessing test scripts

- Each `preprocessing/<step>.py` takes one or more image paths on the
  command line (e.g. `data/images_selection/*.png`), prints a short table of
  numbers per image, and saves debug images (and JSON where useful) to
  **`results/debug_<step>/`**. Keep new step scripts to that pattern; the user
  moved all debug output from `data/` to `results/`.
- Steps chain through files: orientation reads the originals, table crop reads
  `results/debug_orientation/*_upright.png`, cells read
  `results/debug_table/*_table.png`.
- Scripts reuse each other's functions with plain sibling imports (e.g.
  `from deskew import deskew`), which works because the script's folder is
  on `sys.path`. This replaces a `common.py` for now.
- **Comparing combinations:** `pipelines.py` lists combinations of steps
  (a step is a name or `(name, {settings})`). `evaluate_preprocessing.py`
  runs each on `data/images_selection/`, measures the grid with
  `find_cells()`, and scores rows/cols/turn against
  `data/images_selection/expected.csv` (hand-filled; empty = not checked).
  Each step there is a wrapper `step_<name>(image, **settings) -> (image, info)`
  in the `STEPS` dict, reusing the step scripts' functions; add new steps the
  same way.
- **Privacy when testing:** don't open real photos or their debug images with
  the Read tool (that sends them to Anthropic) unless the user approves. Test on
  synthetic images made in the scratchpad, and run real photos locally,
  printing numbers only.

## How it works

- **An approach** is a file in `approaches/` with one function:
  `read_form(photo_path) -> {field: value}`. Anything else in the file is up
  to that approach.
- **Evaluate** with `python evaluate.py <approach>` (or
  `--data data/demo` for the demo forms). It runs `read_form` on every photo
  in `labels.csv`, compares each field with its true value (ignoring extra
  spaces and upper/lower case), and saves a JSON file with:
  - `summary`: accuracy, correct/total fields, failed photos, mean seconds per
    photo.
  - `photos`: for each photo and field, the true value, the value read and
    whether it was correct, so mistakes can be inspected.
- If an approach crashes on a photo, the error is saved in the JSON and that
  photo's fields count as wrong; the other photos still run.

## Ideas for approaches

### Guiding principles

- **Lightweight first:** prefer classical image processing and small models on
  crops over large models on whole pages. Note processing time alongside
  accuracy.
- **Read known boxes, not whole pages:** crop each field at a known position
  (after aligning the photo to a blank form), then read that small crop.
- **Pretrained models only, no training.** Fine-tuning is only reconsidered if
  no pretrained model is accurate enough.
- Forms may contain patient-level health data. Don't commit real photos,
  labels or results, and don't send them to external services or APIs without
  explicit approval.

### Possible processing steps

1. **Quality check:** blur score (variance of the Laplacian), glare, brightness.
2. **Standardize:** perspective correction, grayscale, CLAHE or thresholding,
   deskew, resize.
3. **Align to a blank form:** ORB feature matching plus a homography, then
   crop fields at known coordinates.
4. **Read each field with a pretrained model:**
   - Handwritten values (mostly digits): models that read a whole box at once,
     such as TrOCR handwritten (`microsoft/trocr-small-handwritten`) or
     PaddleOCR PP-OCRv5 mobile models.
   - Lightest option: a pretrained MNIST CNN in ONNX (e.g. `mnist-12`), which
     needs each digit separated and shaped like MNIST.
   - Printed text: PaddleOCR or Tesseract.
   - Experimental: Florence-2-base, a small vision-language model.
5. **Check with known data:** fuzzy-match against known lists, parse dates,
   and use rules between fields (e.g. positive ≤ tested) to catch mistakes.

### Approaches to compare

- Crop known boxes and read each with a pretrained model (TrOCR, PP-OCR,
  MNIST CNN).
- Off-the-shelf OCR engines on the whole page (Tesseract, PaddleOCR, EasyOCR,
  docTR), as baselines.
- Each with and without the known-data checks.

An initial target is about **85–90% of fields correct** on real photos.
Large or cloud-hosted models are out of scope.

## Libraries

Candidates are listed as dependency groups in `pyproject.toml` (`core`, `onnx`,
`ocr`, `hf`, `postprocess`, `analysis`, `dev`); install one with
`pip install --group <name>`.

| Purpose | Library |
|---|---|
| Image processing, alignment, cropping | `opencv-python-headless`, `numpy`, `pillow`, `scikit-image` |
| Run pretrained ONNX models (e.g. MNIST CNN) | `onnxruntime` |
| OCR engines | `rapidocr` (PP-OCR models on onnxruntime), `paddleocr` + `paddlepaddle`, `pytesseract` (needs the `tesseract` binary), `easyocr`, `python-doctr` |
| Handwriting and vision-language models (TrOCR, Florence-2) | `torch`, `transformers`, `timm`, optionally `optimum` |
| Fuzzy matching against known lists | `rapidfuzz` |
| Plots | `matplotlib` |
| Linting | `ruff` |

## Environment

- **Python:** conda environment `env_test` (Python 3.13) is the environment
  for this project. Always run code and ruff with it
  (`/home/esteban/miniconda3/envs/env_test/bin/python`), never another conda
  env; if something is missing, ask before installing it there. VS Code
  terminals and the debugger are set to use it (see `.vscode/settings.json`
  and `.vscode/launch.json`). The user's shell is fish.
- **Installed in `env_test`:** the `core` and `dev` groups (OpenCV, numpy,
  Pillow, scikit-image, ruff), `onnxruntime` 1.31 and `rapidocr` 3.10, plus
  polars and pytest from earlier. The other groups aren't installed yet.
  OpenCV is the headless build: no `cv2.imshow` windows or trackbars, so
  compare settings with saved images (e.g. contact sheets).
- **`rapidocr` 3.10 quirks:**
  - The default models are PP-OCRv6 `small`. PP-OCRv5 only comes in `mobile`
    and `server`, so set `Det.model_type`/`Rec.model_type` together with
    `*.ocr_version`, or it fails with "Invalid OCR configuration".
  - The engine keeps `use_det`/`use_cls`/`use_rec` from its previous call:
    always pass all three.
  - Text crops taller than 1.5× their width are rotated 90° before reading,
    so sideways text can still get fair confidence scores.
- Run scripts from the project root (`python evaluate.py ...`), so
  `approaches/` and the relative `data/` and `results/` paths resolve.
- **Checks:** `ruff check .` and `ruff format .`

## Code Style

- Line length: 120 characters (ruff enforced)
- Python 3.12+
- Ruff rules enabled: `E`, `W`, `F`, `I` (isort), `UP`, `B`, `SIM`, `NPY`,
  `PTH`, `D` (pydocstyle, Google convention), `PL` (pylint), `RUF` (see
  `pyproject.toml`). Import order, complexity limits (`PLR09xx`), magic-value
  comparisons and several docstring/style rules are deliberately ignored.
- No Jupyter notebooks: experiments are plain Python scripts.
- **Docstrings:** Google-style. Every function and method docstring follows the
  same structure: a brief one-line description, an `Args:` section briefly
  describing each parameter, and a `Returns:` section if the function returns a
  value.
