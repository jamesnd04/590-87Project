"""
Standalone OpenCV script that extracts the 12 character-portrait icons from
an in-game scoreboard screenshot (images look like ``assets/in-game.png``).

The scoreboard shows two vertical columns of 6 portraits each:

* **YOUR TEAM**  — left column, directly left of the level / rank badge
  and player name.
* **ENEMY TEAM** — right column, just right of the centre divider and
  left of each enemy player name.

Box coordinates are authored in a shared 1024×565 reference space (the
same space used by ``config/layouts/ingame_split_teams_1024.json``) and
scaled to whatever resolution the source image is.  That keeps the
coordinates portable across 1080p, 1440p, and 4K captures of the exact
same UI layout.

Each reference box tightly frames the character portrait — the small
level badges (``15``, ``23``), class shields, and the ``ACE`` chevron on
Bumper Dumper's row are intentionally **excluded**.

Usage
-----

Verify the default layout against ``assets/in-game.png`` and write an
overlay plus 12 individual JPG crops into ``.tmp-infer/verify-boxes/``::

    python3 scripts/verify_ingame_boxes.py

Point it at any other screenshot::

    python3 scripts/verify_ingame_boxes.py \\
        --source path/to/screenshot.png \\
        --out-dir .tmp-infer/my-check

The reference coordinates printed below can be copied verbatim into
``config/layouts/ingame_split_teams_1024.json``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Reference coordinates (1024×565 space).
#
# Measured directly from the in-game scoreboard using a pixel grid overlay
# on ``assets/in-game.png`` downscaled to 1024×565.  The left portraits sit
# between the left-hand class shield and the rank badge; the right portraits
# sit just past the centre divider, flush against each enemy name column.
#
# Boxes are *tight* on the portrait art — refine_icon_box() in
# scripts/image_infer.py grows them by ~18–22% before contour refinement,
# which is plenty of padding.  If you widen them further the crops start
# picking up neighbouring UI (class shields, rank badges, player text).
# ---------------------------------------------------------------------------
REF_W = 1024
REF_H = 565

LEFT_X = 170     # x-origin of YOUR TEAM portrait column
RIGHT_X = 525    # x-origin of ENEMY TEAM portrait column
BOX_W = 55       # portrait width
BOX_H = 40       # portrait height

# Row tops (y) in reference coords — 6 rows with ~38px pitch.
ROW_YS: tuple[int, ...] = (162, 200, 238, 276, 314, 352)

# (x, y, w, h) in reference pixels — edit LEFT_X/RIGHT_X/BOX_W/BOX_H/ROW_YS
# above to tune.  Order: YOUR 1-6 then ENEMY 1-6 (top → bottom per column).
BOXES: List[Tuple[int, int, int, int]] = [
    *[(LEFT_X, y, BOX_W, BOX_H) for y in ROW_YS],
    *[(RIGHT_X, y, BOX_W, BOX_H) for y in ROW_YS],
]

LABELS = [
    "your_1", "your_2", "your_3", "your_4", "your_5", "your_6",
    "enemy_1", "enemy_2", "enemy_3", "enemy_4", "enemy_5", "enemy_6",
]

# Cyan for YOUR TEAM, red-ish for ENEMY TEAM (BGR).
COLORS = [(0, 220, 255)] * 6 + [(80, 80, 255)] * 6


def scale_box(
    x: int, y: int, w: int, h: int,
    iw: int, ih: int,
) -> Tuple[int, int, int, int]:
    """Map a reference-space box into the actual image resolution."""
    sx = iw / float(REF_W)
    sy = ih / float(REF_H)
    x1 = max(0, min(int(round(x * sx)), iw - 1))
    y1 = max(0, min(int(round(y * sy)), ih - 1))
    x2 = max(x1 + 1, min(int(round((x + w) * sx)), iw))
    y2 = max(y1 + 1, min(int(round((y + h) * sy)), ih))
    return x1, y1, x2, y2


def draw_overlay(frame: np.ndarray, scaled: list[tuple[int, int, int, int]]) -> np.ndarray:
    overlay = frame.copy()
    thickness = max(2, frame.shape[1] // 900)
    font_scale = max(0.45, frame.shape[1] / 1600.0)
    for i, (x1, y1, x2, y2) in enumerate(scaled):
        color = COLORS[i]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            overlay, LABELS[i],
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA,
        )
    return overlay


def extract_crops(
    source: str,
    out_dir: str,
    jpg_quality: int = 95,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Extract the 12 character portraits and save them + an overlay.

    Returns a list of ``(output_path, (x, y, w, h))`` pairs in source-image
    pixel coordinates.
    """
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    ih, iw = frame.shape[:2]
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"Image size: {iw}×{ih}  "
        f"(scale {iw / REF_W:.3f}×{ih / REF_H:.3f} from {REF_W}×{REF_H} reference)"
    )

    scaled = [scale_box(x, y, w, h, iw, ih) for (x, y, w, h) in BOXES]
    overlay = draw_overlay(frame, scaled)
    overlay_path = os.path.join(out_dir, "verify_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"Overlay → {overlay_path}")

    results: list[tuple[str, tuple[int, int, int, int]]] = []
    for i, (x1, y1, x2, y2) in enumerate(scaled):
        crop = frame[y1:y2, x1:x2]
        path = os.path.join(out_dir, f"icon_{i:02d}_{LABELS[i]}.jpg")
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        w = x2 - x1
        h = y2 - y1
        results.append((path, (x1, y1, w, h)))
        print(f"  crop {i:02d} ({LABELS[i]}): (x={x1}, y={y1}, w={w}, h={h}) → {path}")

    print(f"\nAll done. Check '{out_dir}/' for verify_overlay.png and individual crops.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify + extract in-game icon boxes.")
    parser.add_argument(
        "--source", default="assets/in-game.png",
        help="Path to the in-game screenshot.",
    )
    parser.add_argument(
        "--out-dir", default=".tmp-infer/verify-boxes",
        help="Directory to write overlay + crop images.",
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG quality for saved crops (0-100).",
    )
    args = parser.parse_args()

    try:
        extract_crops(args.source, args.out_dir, jpg_quality=args.quality)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
