"""Combinations of preprocessing steps to compare with evaluate_preprocessing.py.

Each combination is a list of steps, run in order. A step is its name, or (name, {settings}) to change its
settings. The step names are the keys of STEPS in evaluate_preprocessing.py:

    page_crop    settings: method = "mask" (paper pixels, default) or "edges" (page borders, also works for a page
                 on another sheet). For "mask": max_saturation, max_texture, brightness_offset, close_fraction
                 (see paper_mask()). For "edges": min_step (see find_page_edges())
    deskew       settings: shear = True to also straighten vertical lines that still lean after the rotation
                 (a photo taken at an angle; see deskew.straighten()). Default False: rotation only
    orientation  no settings (it also fixes the tilt, so deskew is not needed before it)
    deblur       settings: method = "unsharp", "clahe_unsharp" or "deconvolve", and that method's own settings
                 (sigma, amount for "unsharp"; sigma, iterations for "deconvolve"; see deblur.py)
    denoise      settings: method = "median", "bilateral" or "nlmeans", and strength for "nlmeans" (default 7;
                 lower keeps thin lines, see denoise.py)
    channel      settings: channel = "auto", "gray", "red", "green", "blue" or "lab_b" (needs colour: put it
                 after page_crop, which needs colour too, and before deblur or contrast, which make the image grey)
    contrast     settings: method = "flatten", "clahe", "flatten_clahe" or "flatten_stretch" (see contrast.py)
    table_crop   settings: bridge (px of line gaps to close, 0 = off), join (px, default 5), merge (0-1, 0 = off),
                 min_area (0-1: keep the image if the table found is smaller than this share of it)

After the last step, the table grid is always measured and compared with expected.csv.

Turning the photo upright is left to the user, so these combinations don't use orientation; deskew only removes
the small tilt, which the grid measurement needs (a tilted line doesn't cross one row of pixels).
"""

FLAT_CLAHE = ("contrast", {"method": "flatten_clahe"})
PAGE_EDGES = ("page_crop", {"method": "edges"})
RED = ("channel", {"channel": "red"})

# Second set (29 October 2026). The first set showed that the edges page crop helps, deskew hurts, and that
# page_edges_red_flatclahe_table was the closest. Its runs, and those of the first set, stay in the report.
# Each pipeline below changes one thing from that best one, so a difference in score points to that change.
PIPELINES = {
    # Reference: the best of the first set (same name, so the report links its runs).
    "page_edges_red_flatclahe_table": [PAGE_EDGES, RED, FLAT_CLAHE, "table_crop"],
    # Table crop: some crops were a small piece of the table (3% of the image on one photo).
    "page_edges_red_flatclahe_join": [PAGE_EDGES, RED, FLAT_CLAHE, ("table_crop", {"join": 9})],
    "page_edges_red_flatclahe_min": [PAGE_EDGES, RED, FLAT_CLAHE, ("table_crop", {"min_area": 0.2})],
    "page_edges_red_flatclahe_join_min": [PAGE_EDGES, RED, FLAT_CLAHE, ("table_crop", {"join": 9, "min_area": 0.2})],
    # Contrast: CLAHE may turn handwriting into extra "lines" (28 columns found for 12 on one photo).
    "page_edges_red_flatten_table": [PAGE_EDGES, RED, ("contrast", {"method": "flatten"}), "table_crop"],
    "page_edges_red_clahe_table": [PAGE_EDGES, RED, ("contrast", {"method": "clahe"}), "table_crop"],
    "page_edges_red_flatclahe_deblur_table": [
        PAGE_EDGES,
        RED,
        FLAT_CLAHE,
        ("deblur", {"method": "unsharp"}),
        "table_crop",
    ],
    # Colour channel: is red really better than grey, or than picking the channel per photo?
    "page_edges_flatclahe_table": [PAGE_EDGES, FLAT_CLAHE, "table_crop"],
    "page_edges_auto_flatclahe_table": [PAGE_EDGES, ("channel", {"channel": "auto"}), FLAT_CLAHE, "table_crop"],
    # Page crop: a stricter border test, in case creases or shadows inside the page are taken as its border.
    "page_strict_red_flatclahe_table": [
        ("page_crop", {"method": "edges", "min_step": 10}),
        RED,
        FLAT_CLAHE,
        "table_crop",
    ],
}

# Image quality only (23 November 2026): simple steps that make the photo easier to read, with no table crop and
# no grid. Compared with evaluate_quality.py, which saves the results side by side to look at by eye.
# The first one is the base; each of the others changes one thing from it.
FLATTEN = ("contrast", {"method": "flatten"})
QUALITY_PIPELINES = {
    # Base: straighten, crop the page, even out the lighting.
    "deskew_page_flatten": ["deskew", PAGE_EDGES, FLATTEN],
    # Order: crop first, so the background around the page does not affect the tilt found.
    "page_deskew_flatten": [PAGE_EDGES, "deskew", FLATTEN],
    # Colour channel: keep the red channel, where blue/black ink stands out more than in grey.
    "deskew_page_red_flatten": ["deskew", PAGE_EDGES, RED, FLATTEN],
    # Contrast: flatten, then make the darkest ink black so faint strokes get darker.
    "deskew_page_flatstretch": ["deskew", PAGE_EDGES, ("contrast", {"method": "flatten_stretch"})],
    # Sharpen: add an unsharp mask after the contrast, for blurry photos.
    "deskew_page_flatten_sharpen": ["deskew", PAGE_EDGES, FLATTEN, ("deblur", {"method": "unsharp"})],
    # Second round: the first run showed page crop before deskew leaves no tilt and is fastest, so
    # page_deskew_flatten is the new base, and each pipeline below changes one thing from it.
    # Milder sharpening (default sigma 2, amount 1.5), hoping for crisper strokes without doubling the noise.
    "page_deskew_flatten_mildsharpen": [
        PAGE_EDGES,
        "deskew",
        FLATTEN,
        ("deblur", {"method": "unsharp", "sigma": 1.0, "amount": 0.7}),
    ],
    # Remove the noise first (non-local means), before flatten raises it on darker paper.
    "page_deskew_denoise_flatten": [PAGE_EDGES, "deskew", ("denoise", {"method": "nlmeans"}), FLATTEN],
    # The other page crop method (paper pixels), for the photo the edges method does not crop.
    "page_mask_deskew_flatten": [("page_crop", {"method": "mask"}), "deskew", FLATTEN],
    # Third round. On a photo taken at an angle, page_deskew_flatten left the vertical lines leaning by 5 degrees
    # (the old tilt measure missed it), and denoise (strength 7) made the lines too thin.
    # Deskew with shear: rotate so the horizontal lines are level, then unshear so the vertical lines are too.
    "page_deskewshear_flatten": [PAGE_EDGES, ("deskew", {"shear": True}), FLATTEN],
    # Weaker denoise, to keep the thin lines.
    "page_deskew_denoise3_flatten": [PAGE_EDGES, "deskew", ("denoise", {"method": "nlmeans", "strength": 3}), FLATTEN],
}
