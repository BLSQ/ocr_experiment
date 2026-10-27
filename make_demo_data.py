"""Create a few fake forms in data/demo/ to try evaluate.py before real photos.

Each form has three boxes (tested, positive, negative) with printed numbers,
some blurred or speckled. Usage: python make_demo_data.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

OUTPUT = Path("data/demo")
N_FORMS = 20
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_TOPS = {"tested": 160, "positive": 250, "negative": 340}  # y of each box


def main() -> None:
    """Draw the forms and write labels.csv."""
    rng = np.random.default_rng(0)
    (OUTPUT / "photos").mkdir(parents=True, exist_ok=True)

    with (OUTPUT / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["photo", "field", "value"])

        for index in range(N_FORMS):
            tested = int(rng.integers(1, 400))
            positive = int(rng.integers(0, tested + 1))
            values = {
                "tested": tested,
                "positive": positive,
                "negative": tested - positive,
            }

            # Blank form: title, labels and boxes.
            form = np.full((450, 600), 255, dtype=np.uint8)
            cv2.putText(form, "DEMO FORM", (30, 70), FONT, 1.3, 0, 2)
            for field, top in BOX_TOPS.items():
                cv2.putText(form, field, (40, top + 40), FONT, 0.9, 0, 2)
                cv2.rectangle(form, (300, top), (520, top + 60), 0, 2)
                # The filled-in number, with a bit of random size and position.
                x = 315 + int(rng.integers(0, 30))
                size = rng.uniform(1.0, 1.4)
                cv2.putText(form, str(values[field]), (x, top + 45), FONT, size, 0, 2)

            # Make it look like a poor photo: specks and blur.
            for _ in range(int(rng.integers(0, 40))):
                speck = (int(rng.integers(0, 600)), int(rng.integers(0, 450)))
                cv2.circle(form, speck, 1, 0, -1)
            blur = float(rng.choice([0, 0, 1, 2]))
            if blur:
                form = cv2.GaussianBlur(form, (0, 0), blur)

            photo = f"form_{index:03d}.png"
            cv2.imwrite(str(OUTPUT / "photos" / photo), form)
            for field, value in values.items():
                writer.writerow([photo, field, value])

    print(f"Wrote {N_FORMS} demo forms to {OUTPUT}")


if __name__ == "__main__":
    main()
