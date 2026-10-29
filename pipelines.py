"""Combinations of preprocessing steps to compare with evaluate_preprocessing.py.

Each combination is a list of steps, run in order. A step is its name, or (name, {settings}) to change its
settings. The step names are the keys of STEPS in evaluate_preprocessing.py:

    page_crop    settings: method = "mask" (paper pixels, default) or "edges" (page borders, also works for a page
                 on another sheet). For "mask": max_saturation, max_texture, brightness_offset, close_fraction
                 (see paper_mask()). For "edges": min_step (see find_page_edges())
    deskew       no settings
    orientation  no settings (it also fixes the tilt, so deskew is not needed before it)
    deblur       settings: method = "unsharp", "clahe_unsharp" or "deconvolve"
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
