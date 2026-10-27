"""Combinations of preprocessing steps to compare with evaluate_preprocessing.py.

Each combination is a list of steps, run in order. A step is its name, or (name, {settings}) to change its
settings. The step names are the keys of STEPS in evaluate_preprocessing.py:

    page_crop    settings: max_saturation, max_texture, brightness_offset, close_fraction (see paper_mask())
    deskew       no settings
    orientation  no settings (it also fixes the tilt, so deskew is not needed before it)
    deblur       settings: method = "unsharp", "clahe_unsharp" or "deconvolve"
    channel      settings: channel = "auto", "gray", "red", "green", "blue" or "lab_b" (needs colour: put it
                 after page_crop, which needs colour too, and before deblur, which makes the image grey)
    table_crop   no settings

After the last step, the table grid is always measured and compared with expected.csv.
"""

PIPELINES = {
    "table_only": ["table_crop"],
    "orient_table": ["orientation", "table_crop"],
    "page_orient_table": ["page_crop", "orientation", "table_crop"],
    "page_deblur_orient_table": ["page_crop", ("deblur", {"method": "clahe_unsharp"}), "orientation", "table_crop"],
    "page_strict_orient_table": [("page_crop", {"max_texture": 8}), "orientation", "table_crop"],
    "page_channel_orient_table": ["page_crop", "channel", "orientation", "table_crop"],
    "page_red_orient_table": ["page_crop", ("channel", {"channel": "red"}), "orientation", "table_crop"],
}
