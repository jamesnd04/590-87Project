"""
Layout-based extraction of Marvel Rivals-style icon / portrait cells with OpenCV.

Reads a JSON layout describing where character icon boxes sit (relative to a
reference resolution), scales regions to the input image, crops each cell,
and prints one JSON object to stdout.

Layout modes (pick one):
  - "boxes": explicit list of {x, y, w, h} in reference pixels
  - "grid": vertical column — repeated rows (avatar column on scoreboards)
  - "horizontal_grid": one row of icons — repeated columns (team pick bar)

Usage:
  python3 scripts/image_infer.py --source /path/to/image.png
  python3 scripts/image_infer.py --source img.png --layout config/layouts/team_panel_1024.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import cv2
import numpy as np

DEFAULT_LAYOUT_REL = os.path.join("config", "scoreboard_layout.json")


def load_layout(path: str | None) -> tuple[dict[str, Any], str | None]:
    """Returns (layout dict, resolved path or None if embedded defaults)."""
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f), os.path.abspath(path)
    cwd = os.getcwd()
    default_path = os.path.join(cwd, DEFAULT_LAYOUT_REL)
    if os.path.isfile(default_path):
        with open(default_path, encoding="utf-8") as f:
            return json.load(f), os.path.abspath(default_path)
    embedded = {
        "reference_width": 1024,
        "reference_height": 479,
        "description": "Embedded default: post-match scoreboard icon column (tune via layout JSON).",
        "grid": {
            "x": 106,
            "y": 112,
            "width": 102,
            "height": 28,
            "row_stride": 28,
            "row_count": 12,
        },
    }
    return embedded, None


def boxes_from_layout(layout: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) in reference coordinates."""
    ref_boxes: list[tuple[int, int, int, int]] = []
    if "boxes" in layout and isinstance(layout["boxes"], list):
        for b in layout["boxes"]:
            ref_boxes.append(
                (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])),
            )
        return ref_boxes
    hg = layout.get("horizontal_grid")
    if isinstance(hg, dict):
        x = int(hg["x"])
        y = int(hg["y"])
        w = int(hg["width"])
        h = int(hg["height"])
        stride = int(hg["col_stride"])
        count = int(hg["col_count"])
        for i in range(count):
            ref_boxes.append((x + i * stride, y, w, h))
        return ref_boxes
    grid = layout.get("grid")
    if not isinstance(grid, dict):
        raise ValueError('Layout must contain "grid", "horizontal_grid", or "boxes"')
    x = int(grid["x"])
    y = int(grid["y"])
    w = int(grid["width"])
    h = int(grid["height"])
    stride = int(grid["row_stride"])
    count = int(grid["row_count"])
    for i in range(count):
        ref_boxes.append((x, y + i * stride, w, h))
    return ref_boxes


def scale_rect(
    x: int,
    y: int,
    w: int,
    h: int,
    iw: int,
    ih: int,
    rw: int,
    rh: int,
) -> tuple[int, int, int, int]:
    sx = iw / float(rw)
    sy = ih / float(rh)
    x1 = int(round(x * sx))
    y1 = int(round(y * sy))
    x2 = int(round((x + w) * sx))
    y2 = int(round((y + h) * sy))
    x1 = max(0, min(x1, iw - 1))
    y1 = max(0, min(y1, ih - 1))
    x2 = max(x1 + 1, min(x2, iw))
    y2 = max(y1 + 1, min(y2, ih))
    return x1, y1, x2, y2


def normalize_crop(crop: np.ndarray, size: int) -> np.ndarray:
    if crop.size == 0:
        return crop
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def summarize(count: int, iw: int, ih: int, rw: int, rh: int) -> str:
    return (
        f"OpenCV: {count} icon box(es); "
        f"screenshot {iw}×{ih}; layout reference {rw}×{rh}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--layout",
        default=None,
        help=f"JSON layout file (default: {DEFAULT_LAYOUT_REL} if present)",
    )
    parser.add_argument(
        "--normalize-size",
        type=int,
        default=128,
        help="Resize each crop to N×N RGB for payload (0 to skip crop_pixels)",
    )
    args = parser.parse_args()

    layout, layout_resolved = load_layout(args.layout)
    rw = int(layout.get("reference_width", 1920))
    rh = int(layout.get("reference_height", 1080))
    try:
        ref_boxes = boxes_from_layout(layout)
    except (KeyError, TypeError, ValueError) as e:
        print(json.dumps({"ok": False, "error": f"Invalid layout: {e}"}))
        sys.exit(1)

    frame = cv2.imread(args.source)
    if frame is None:
        print(json.dumps({"ok": False, "error": f"Could not read image: {args.source}"}))
        sys.exit(1)

    ih, iw = frame.shape[:2]
    icon_boxes: list[dict[str, Any]] = []
    norm_size = int(args.normalize_size)

    for i, (rx, ry, rw_, rh_) in enumerate(ref_boxes):
        x1, y1, x2, y2 = scale_rect(rx, ry, rw_, rh_, iw, ih, rw, rh)
        crop = frame[y1:y2, x1:x2]
        entry: dict[str, Any] = {
            "index": i,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "reference_box": {"x": rx, "y": ry, "w": rw_, "h": rh_},
        }
        if norm_size > 0 and crop.size > 0:
            small = normalize_crop(crop, norm_size)
            entry["crop_shape"] = {"width": norm_size, "height": norm_size, "channels": 3}
            entry["crop_mean_rgb"] = [
                round(float(small[:, :, c].mean()), 4) for c in range(3)
            ]
        icon_boxes.append(entry)

    payload = {
        "ok": True,
        "pipeline": "opencv_icon_boxes",
        "summary": summarize(len(icon_boxes), iw, ih, rw, rh),
        "image": {"width": iw, "height": ih},
        "layout": {
            "reference_width": rw,
            "reference_height": rh,
            "layout_file": layout_resolved,
        },
        "icon_boxes": icon_boxes,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
