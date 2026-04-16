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
import base64
import json
import os
import sys
import traceback
import urllib.request
from typing import Any

import cv2
import numpy as np

DEFAULT_LAYOUT_REL = os.path.join("config", "scoreboard_layout.json")
DEFAULT_HERO_ASSETS_REL = os.path.join("config", "hero_reference_assets.json")
DEFAULT_CLIP_CACHE_REL = os.path.join(".cache", "clip-reference-embeddings.json")
HERO_ALIAS_MAP: dict[str, str] = {
    "cloak-lord": "cloak-and-dagger-lord",
    "dagger-lord": "cloak-and-dagger-lord",
}
WORKER_RESPONSE_PREFIX = "__CLIP_WORKER_JSON__"
_CLIP_RUNTIME_CACHE: dict[str, dict[str, Any]] = {}


def load_local_env_file() -> None:
    """Load .env.local into process env for standalone CLI runs."""
    env_path = os.path.join(os.getcwd(), ".env.local")
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                if not key or key in os.environ:
                    continue
                os.environ[key] = v.strip().strip('"').strip("'")
    except Exception:
        pass


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


def clamp_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    iw: int,
    ih: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), iw - 1))
    y1 = max(0, min(int(y1), ih - 1))
    x2 = max(x1 + 1, min(int(x2), iw))
    y2 = max(y1 + 1, min(int(y2), ih))
    return x1, y1, x2, y2


def expand_box(
    box: tuple[int, int, int, int],
    expand_x: float,
    expand_y: float,
    iw: int,
    ih: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    dx = int(round(bw * expand_x))
    dy = int(round(bh * expand_y))
    return clamp_box(x1 - dx, y1 - dy, x2 + dx, y2 + dy, iw, ih)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    aa = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    ba = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / max(1.0, aa + ba - inter)


def detect_layout_mode(layout: dict[str, Any], layout_resolved: str | None) -> str:
    mode = layout.get("layout_mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip().lower()
    if layout_resolved:
        name = os.path.basename(layout_resolved).lower()
        if "post" in name:
            return "post_game"
        if "ingame" in name or "in_game" in name:
            return "in_game"
        if "character_select" in name or "pre" in name:
            return "pre_game"
    if layout.get("horizontal_grid"):
        return "pre_game"
    return "post_game"


def resolve_reference_image_path(layout: dict[str, Any], layout_resolved: str | None) -> str | None:
    ref = layout.get("reference_image")
    if not isinstance(ref, str) or not ref.strip():
        return None
    raw = ref.strip()
    if os.path.isabs(raw):
        return raw if os.path.isfile(raw) else None
    candidates: list[str] = [os.path.join(os.getcwd(), raw)]
    if layout_resolved:
        candidates.append(os.path.join(os.path.dirname(layout_resolved), raw))
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def homography_from_orb(ref: np.ndarray, frame: np.ndarray) -> np.ndarray | None:
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2500)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(frame_gray, None)
    if des1 is None or des2 is None or not kp1 or not kp2:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des1, des2, k=2)
    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 10:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or inlier_mask is None:
        return None
    inliers = int(inlier_mask.ravel().sum())
    ratio = inliers / float(max(1, len(good)))
    if inliers < 12 or ratio < 0.28:
        return None
    return H


def translation_shift_from_phasecorr(ref: np.ndarray, frame: np.ndarray) -> tuple[float, float]:
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ih, iw = frame_gray.shape[:2]
    resized_ref = cv2.resize(ref_gray, (iw, ih), interpolation=cv2.INTER_AREA)
    shift, _ = cv2.phaseCorrelate(
        np.float32(resized_ref),
        np.float32(frame_gray),
    )
    return float(shift[0]), float(shift[1])


def project_boxes_with_homography(
    ref_boxes: list[tuple[int, int, int, int]],
    H: np.ndarray,
    iw: int,
    ih: int,
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for (x, y, w, h) in ref_boxes:
        pts = np.float32(
            [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ],
        ).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        x1 = int(np.floor(np.min(warped[:, 0])))
        y1 = int(np.floor(np.min(warped[:, 1])))
        x2 = int(np.ceil(np.max(warped[:, 0])))
        y2 = int(np.ceil(np.max(warped[:, 1])))
        out.append(clamp_box(x1, y1, x2, y2, iw, ih))
    return out


def project_boxes_with_scale_shift(
    ref_boxes: list[tuple[int, int, int, int]],
    rw: int,
    rh: int,
    iw: int,
    ih: int,
    shift_x: float,
    shift_y: float,
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for (x, y, w, h) in ref_boxes:
        x1, y1, x2, y2 = scale_rect(x, y, w, h, iw, ih, rw, rh)
        out.append(
            clamp_box(
                int(round(x1 + shift_x)),
                int(round(y1 + shift_y)),
                int(round(x2 + shift_x)),
                int(round(y2 + shift_y)),
                iw,
                ih,
            ),
        )
    return out


def template_match_shift_scale(
    ref: np.ndarray,
    frame: np.ndarray,
    rw: int,
    rh: int,
) -> tuple[float, float, float, float] | None:
    """Estimate (shift_x, shift_y, sx, sy) with template matching around expected scale."""
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ih, iw = frame_gray.shape[:2]
    base_sx = iw / float(max(1, rw))
    base_sy = ih / float(max(1, rh))
    scales = [0.88, 0.94, 1.0, 1.06, 1.12]
    best: tuple[float, tuple[int, int], float, float] | None = None
    for m in scales:
        sx = base_sx * m
        sy = base_sy * m
        tw = int(round(ref_gray.shape[1] * sx))
        th = int(round(ref_gray.shape[0] * sy))
        if tw < 32 or th < 32 or tw >= iw or th >= ih:
            continue
        templ = cv2.resize(ref_gray, (tw, th), interpolation=cv2.INTER_AREA)
        try:
            res = cv2.matchTemplate(frame_gray, templ, cv2.TM_CCOEFF_NORMED)
        except Exception:
            continue
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if best is None or maxv > best[0]:
            best = (float(maxv), maxloc, sx, sy)
    if best is None:
        return None
    score, (x, y), sx, sy = best
    if score < 0.28:
        return None
    return float(x), float(y), float(sx), float(sy)


def project_boxes_with_template_alignment(
    ref_boxes: list[tuple[int, int, int, int]],
    iw: int,
    ih: int,
    shift_x: float,
    shift_y: float,
    sx: float,
    sy: float,
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for (x, y, w, h) in ref_boxes:
        x1 = int(round(x * sx + shift_x))
        y1 = int(round(y * sy + shift_y))
        x2 = int(round((x + w) * sx + shift_x))
        y2 = int(round((y + h) * sy + shift_y))
        out.append(clamp_box(x1, y1, x2, y2, iw, ih))
    return out


def snap_postgame_rows(
    boxes: list[tuple[int, int, int, int]],
    frame: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]]:
    if len(boxes) < 2:
        return boxes
    ordered = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2.0)
    centers = [((b[1] + b[3]) / 2.0) for b in ordered]
    x1_med = int(round(float(np.median([b[0] for b in ordered]))))
    x2_med = int(round(float(np.median([b[2] for b in ordered]))))
    tops: list[int] = []
    bottoms: list[int] = []
    for i, _ in enumerate(ordered):
        if i == 0:
            top = int(round(centers[i] - (centers[i + 1] - centers[i]) * 0.5))
        else:
            top = int(round((centers[i - 1] + centers[i]) * 0.5))
        if i == len(ordered) - 1:
            bot = int(round(centers[i] + (centers[i] - centers[i - 1]) * 0.5))
        else:
            bot = int(round((centers[i] + centers[i + 1]) * 0.5))
        tops.append(top)
        bottoms.append(bot)
    if frame is not None:
        # Refine row boundaries using horizontal divider peaks in Sobel-Y energy.
        try:
            ih, iw = frame.shape[:2]
            y_min = max(0, int(min(tops) - 8))
            y_max = min(ih, int(max(bottoms) + 8))
            sx1 = max(0, x1_med + int(0.12 * (x2_med - x1_med)))
            sx2 = min(iw, x2_med - int(0.12 * (x2_med - x1_med)))
            if y_max - y_min > 20 and sx2 - sx1 > 12:
                roi = frame[y_min:y_max, sx1:sx2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                soby = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                energy = np.mean(np.abs(soby), axis=1)
                for i in range(len(ordered) - 1):
                    target = int(round((centers[i] + centers[i + 1]) * 0.5)) - y_min
                    r0 = max(0, target - 8)
                    r1 = min(len(energy), target + 9)
                    if r1 - r0 < 3:
                        continue
                    local = energy[r0:r1]
                    peak = int(np.argmax(local)) + r0
                    divider_y = peak + y_min
                    bottoms[i] = divider_y
                    tops[i + 1] = divider_y
        except Exception:
            pass

    snapped: list[tuple[int, int, int, int]] = []
    for i in range(len(ordered)):
        y1 = tops[i] + 1
        y2 = bottoms[i] - 1
        if y2 <= y1:
            y1, y2 = ordered[i][1], ordered[i][3]
        snapped.append((x1_med, y1, x2_med, y2))
    return snapped


def recenter_ingame_row_windows(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Shift each in-game row window horizontally to maximize icon-like texture/color."""
    if len(boxes) < 2:
        return boxes
    ih, iw = frame.shape[:2]
    ordered = reorder_boxes(boxes, "left_right_columns_top_to_bottom")
    n = len(ordered)
    half = n // 2
    groups = [ordered[:half], ordered[half:]]
    recentered: list[tuple[int, int, int, int]] = []
    for group in groups:
        if not group:
            continue
        bw_med = int(round(float(np.median([b[2] - b[0] for b in group]))))
        for (x1, y1, x2, y2) in group:
            bw = max(1, x2 - x1)
            target_w = max(12, int(round(0.95 * bw_med)))
            ry1 = max(0, y1)
            ry2 = min(ih, y2)
            if ry2 - ry1 < 8:
                recentered.append((x1, y1, x2, y2))
                continue
            # Search around coarse X with an expanded horizontal band.
            sx0 = max(0, int(round(x1 - 0.9 * bw)))
            sx1 = min(iw, int(round(x2 + 0.9 * bw)))
            row = frame[ry1:ry2, sx0:sx1]
            if row.size == 0 or row.shape[1] <= target_w + 2:
                recentered.append((x1, y1, x2, y2))
                continue
            gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(row, cv2.COLOR_BGR2HSV)
            edges = cv2.Canny(gray, 70, 180)
            best_score = -1e9
            best_lx = int(round((x1 + x2) * 0.5)) - sx0 - (target_w // 2)
            for lx in range(0, row.shape[1] - target_w + 1, 2):
                ex = lx + target_w
                patch_g = gray[:, lx:ex]
                patch_s = hsv[:, lx:ex, 1]
                patch_e = edges[:, lx:ex]
                lap = float(cv2.Laplacian(patch_g, cv2.CV_32F).var())
                satv = float(patch_s.var())
                ed = float(np.count_nonzero(patch_e)) / float(max(1, patch_e.size))
                # Bias toward center to prevent large jumps unless score gain is meaningful.
                center = lx + target_w * 0.5
                center_ref = (x1 + x2) * 0.5 - sx0
                center_penalty = abs(center - center_ref) / float(max(1.0, bw))
                score = (0.018 * lap) + (0.0035 * satv) + (1.2 * ed) - (0.16 * center_penalty)
                if score > best_score:
                    best_score = score
                    best_lx = lx
            nx1 = sx0 + best_lx
            nx2 = nx1 + target_w
            recentered.append(clamp_box(nx1, y1, nx2, y2, iw, ih))
    return recentered


def register_layout_boxes(
    frame: np.ndarray,
    ref_boxes: list[tuple[int, int, int, int]],
    rw: int,
    rh: int,
    layout: dict[str, Any],
    layout_resolved: str | None,
    align_method: str,
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    ih, iw = frame.shape[:2]
    scaled = [scale_rect(x, y, w, h, iw, ih, rw, rh) for (x, y, w, h) in ref_boxes]
    info: dict[str, Any] = {"method": "scaled", "ok": True}
    ref_path = resolve_reference_image_path(layout, layout_resolved)
    if not ref_path:
        info["detail"] = "No reference_image provided; used scaled layout."
        return scaled, info
    ref = cv2.imread(ref_path)
    if ref is None:
        info["detail"] = f"Could not read reference image: {ref_path}"
        return scaled, info

    method = (align_method or "auto").strip().lower()
    if method == "auto":
        preferred = layout.get("preferred_align_method")
        if isinstance(preferred, str) and preferred.strip().lower() in {"template", "homography", "phasecorr", "scaled"}:
            method = preferred.strip().lower()
    if method not in {"auto", "homography", "phasecorr", "scaled"}:
        if method == "template":
            pass
        else:
            method = "auto"

    if method == "template":
        tmpl = template_match_shift_scale(ref, frame, rw=rw, rh=rh)
        if tmpl is not None:
            shift_x, shift_y, sx, sy = tmpl
            info["method"] = "template"
            info["reference_image"] = ref_path
            info["shift"] = {"x": round(shift_x, 3), "y": round(shift_y, 3)}
            info["scale"] = {"x": round(sx, 5), "y": round(sy, 5)}
            return project_boxes_with_template_alignment(
                ref_boxes=ref_boxes,
                iw=iw,
                ih=ih,
                shift_x=shift_x,
                shift_y=shift_y,
                sx=sx,
                sy=sy,
            ), info
        info["detail"] = "Template alignment failed; used scaled layout."
        return scaled, info

    if method not in {"auto", "homography", "phasecorr", "scaled"}:
        method = "auto"

    if method in {"auto", "homography"}:
        H = homography_from_orb(ref, frame)
        if H is not None:
            info["method"] = "homography"
            info["reference_image"] = ref_path
            return project_boxes_with_homography(ref_boxes, H, iw, ih), info
        if method == "homography":
            info["detail"] = "Homography alignment failed; used scaled layout."
            return scaled, info

    if method == "auto":
        tmpl = template_match_shift_scale(ref, frame, rw=rw, rh=rh)
        if tmpl is not None:
            shift_x, shift_y, sx, sy = tmpl
            info["method"] = "template"
            info["reference_image"] = ref_path
            info["shift"] = {"x": round(shift_x, 3), "y": round(shift_y, 3)}
            info["scale"] = {"x": round(sx, 5), "y": round(sy, 5)}
            return project_boxes_with_template_alignment(
                ref_boxes=ref_boxes,
                iw=iw,
                ih=ih,
                shift_x=shift_x,
                shift_y=shift_y,
                sx=sx,
                sy=sy,
            ), info

    if method in {"auto", "phasecorr"}:
        try:
            shift_x, shift_y = translation_shift_from_phasecorr(ref, frame)
            info["method"] = "phasecorr"
            info["reference_image"] = ref_path
            info["shift"] = {"x": round(shift_x, 3), "y": round(shift_y, 3)}
            return project_boxes_with_scale_shift(ref_boxes, rw, rh, iw, ih, shift_x, shift_y), info
        except Exception as e:
            if method == "phasecorr":
                info["detail"] = f"phasecorr failed: {e}"
                return scaled, info

    return scaled, info


def mode_inset(layout_mode: str) -> tuple[float, float]:
    if layout_mode == "pre_game":
        return 0.05, 0.08
    if layout_mode == "in_game":
        return 0.035, 0.06
    return 0.03, 0.07


def refine_icon_box(
    frame: np.ndarray,
    coarse_box: tuple[int, int, int, int],
    layout_mode: str,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    ih, iw = frame.shape[:2]
    coarse = clamp_box(*coarse_box, iw, ih)
    ex = 0.22 if layout_mode == "pre_game" else 0.18
    ey = 0.28 if layout_mode == "post_game" else 0.2
    roi_box = expand_box(coarse, ex, ey, iw, ih)
    rx1, ry1, rx2, ry2 = roi_box
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return coarse, {"used": "coarse", "reason": "empty_roi"}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 170)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx1, cy1, cx2, cy2 = coarse
    coarse_local = (cx1 - rx1, cy1 - ry1, cx2 - rx1, cy2 - ry1)
    coarse_area = float(max(1, (coarse_local[2] - coarse_local[0]) * (coarse_local[3] - coarse_local[1])))
    coarse_w = max(1, coarse_local[2] - coarse_local[0])
    coarse_h = max(1, coarse_local[3] - coarse_local[1])
    coarse_aspect = (coarse_local[2] - coarse_local[0]) / float(max(1, coarse_local[3] - coarse_local[1]))
    best: tuple[float, tuple[int, int, int, int]] | None = None

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8:
            continue
        area = float(w * h)
        if area < 0.35 * coarse_area or area > 1.7 * coarse_area:
            continue
        if abs(w - coarse_w) > coarse_w * 0.35:
            continue
        if abs(h - coarse_h) > coarse_h * 0.4:
            continue
        aspect = w / float(max(1, h))
        if aspect < coarse_aspect * 0.45 or aspect > coarse_aspect * 2.2:
            continue
        candidate = (x, y, x + w, y + h)
        overlap = iou(coarse_local, candidate)
        edge_density = float(np.count_nonzero(edges[y : y + h, x : x + w])) / float(max(1, w * h))
        score = (1.8 * overlap) + (0.35 * edge_density)
        if best is None or score > best[0]:
            best = (score, candidate)

    chosen = coarse
    meta: dict[str, Any] = {"used": "coarse"}
    if best is not None:
        bx1, by1, bx2, by2 = best[1]
        candidate_abs = clamp_box(bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1, iw, ih)
        if iou(coarse, candidate_abs) >= 0.3:
            chosen = candidate_abs
            meta = {"used": "contour", "score": round(float(best[0]), 4)}
        else:
            meta = {"used": "coarse", "reason": "low_iou_candidate"}

    inset_x, inset_y = mode_inset(layout_mode)
    px1, py1, px2, py2 = chosen
    bw = max(1, px2 - px1)
    bh = max(1, py2 - py1)
    ix = int(round(bw * inset_x))
    iy = int(round(bh * inset_y))
    inset = clamp_box(px1 + ix, py1 + iy, px2 - ix, py2 - iy, iw, ih)
    meta["inset"] = {"x": inset_x, "y": inset_y}
    return inset, meta


def quality_metrics(crop: np.ndarray, layout_mode: str) -> dict[str, Any]:
    if crop.size == 0:
        return {"ok": False, "reason": "empty_crop"}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat_var = float(hsv[:, :, 1].var())
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.count_nonzero(edges)) / float(max(1, gray.shape[0] * gray.shape[1]))
    if layout_mode == "in_game":
        # In-game icon strips can be dark and low-edge because of blur/shields.
        ok = bool(lap >= 1.5 and sat_var >= 40.0)
    elif layout_mode == "pre_game":
        ok = bool(lap >= 6.0 and sat_var >= 10.0 and edge_density >= 0.01)
    else:
        ok = bool(lap >= 8.0 and sat_var >= 12.0 and edge_density >= 0.012)
    return {
        "ok": ok,
        "laplacian_var": round(lap, 4),
        "saturation_var": round(sat_var, 4),
        "edge_density": round(edge_density, 6),
    }


def reorder_boxes(
    boxes: list[tuple[int, int, int, int]],
    order: str,
) -> list[tuple[int, int, int, int]]:
    mode = (order or "").strip().lower()
    if mode == "left_to_right":
        return sorted(boxes, key=lambda b: ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0))
    if mode == "left_right_columns_top_to_bottom":
        if len(boxes) <= 1:
            return boxes[:]
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, b) for b in boxes]
        xs = sorted(c[0] for c in centers)
        pivot = xs[len(xs) // 2]
        left = [c[2] for c in centers if c[0] <= pivot]
        right = [c[2] for c in centers if c[0] > pivot]
        left_sorted = sorted(left, key=lambda b: (b[1] + b[3]) / 2.0)
        right_sorted = sorted(right, key=lambda b: (b[1] + b[3]) / 2.0)
        return left_sorted + right_sorted
    return sorted(boxes, key=lambda b: ((b[1] + b[3]) / 2.0, (b[0] + b[2]) / 2.0))


def write_debug_outputs(
    frame: np.ndarray,
    source_path: str,
    debug_dir: str,
    coarse_boxes: list[tuple[int, int, int, int]],
    refined_boxes: list[tuple[int, int, int, int]],
) -> dict[str, Any]:
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source_path))[0]
    overlay = frame.copy()
    for i, b in enumerate(coarse_boxes):
        x1, y1, x2, y2 = b
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (64, 180, 255), 2)
        cv2.putText(overlay, f"c{i}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (64, 180, 255), 1, cv2.LINE_AA)
    for i, b in enumerate(refined_boxes):
        x1, y1, x2, y2 = b
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 240, 80), 2)
        cv2.putText(overlay, f"r{i}", (x1, min(overlay.shape[0] - 6, y2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 240, 80), 1, cv2.LINE_AA)
    overlay_path = os.path.join(debug_dir, f"{stem}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    crop_paths: list[str] = []
    for i, (x1, y1, x2, y2) in enumerate(refined_boxes):
        crop = frame[y1:y2, x1:x2]
        out = os.path.join(debug_dir, f"{stem}_crop_{i:02d}.png")
        cv2.imwrite(out, crop)
        crop_paths.append(out)
    return {"overlay": overlay_path, "crops": crop_paths}


def normalize_crop(crop: np.ndarray, size: int) -> np.ndarray:
    if crop.size == 0:
        return crop
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def build_crop_views_rgb(crop_bgr: np.ndarray) -> list[np.ndarray]:
    """Build multiple crop views so CLIP is less brittle on framing."""
    if crop_bgr.size == 0:
        return []
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return [rgb]

    def subrect(rx1: float, ry1: float, rx2: float, ry2: float) -> np.ndarray:
        x1 = max(0, min(w - 1, int(round(w * rx1))))
        y1 = max(0, min(h - 1, int(round(h * ry1))))
        x2 = max(x1 + 1, min(w, int(round(w * rx2))))
        y2 = max(y1 + 1, min(h, int(round(h * ry2))))
        return rgb[y1:y2, x1:x2]

    views = [
        rgb,  # full crop
        subrect(0.05, 0.05, 0.95, 0.95),  # center-tight
        subrect(0.0, 0.0, 0.9, 1.0),  # left-biased
        subrect(0.1, 0.0, 1.0, 1.0),  # right-biased
        subrect(0.0, 0.08, 1.0, 0.92),  # vertical trim
    ]
    return [v for v in views if v.size > 0]


def compute_hsv_histogram(image_bgr: np.ndarray) -> np.ndarray | None:
    if image_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    if hist is None:
        return None
    hist = cv2.normalize(hist, hist).astype(np.float32)
    return hist


def compute_hue_profile(image_bgr: np.ndarray, bins: int = 36) -> np.ndarray | None:
    if image_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # Emphasize colored, bright pixels so glow tint contributes more than dark UI.
    mask = (sat >= 40) & (val >= 35)
    if not np.any(mask):
        return None
    hue_vals = hue[mask]
    hist, _ = np.histogram(hue_vals, bins=bins, range=(0, 180))
    vec = hist.astype(np.float32)
    s = float(vec.sum())
    if s <= 0:
        return None
    return vec / s


def summarize(count: int, iw: int, ih: int, rw: int, rh: int) -> str:
    return (
        f"OpenCV: {count} icon box(es); "
        f"screenshot {iw}×{ih}; layout reference {rw}×{rh}."
    )


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return vec
    return vec / norm


def normalize_hero_id(hero_id: str) -> str:
    return HERO_ALIAS_MAP.get(hero_id, hero_id)


def load_hero_assets(path: str | None) -> list[dict[str, Any]]:
    if path and os.path.isfile(path):
        resolved = path
    else:
        resolved = os.path.join(os.getcwd(), DEFAULT_HERO_ASSETS_REL)
    if not os.path.isfile(resolved):
        return []
    with open(resolved, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        hid = item.get("id")
        files = item.get("files")
        if not isinstance(hid, str) or not isinstance(files, list):
            continue
        resolved_files = []
        for rel in files:
            if not isinstance(rel, str):
                continue
            full = rel if os.path.isabs(rel) else os.path.join(os.getcwd(), rel)
            if os.path.isfile(full):
                resolved_files.append(full)
        if resolved_files:
            out.append({"id": hid, "files": resolved_files})
    return out


def file_signature(path: str) -> str:
    st = os.stat(path)
    return f"{path}:{int(st.st_mtime_ns)}:{st.st_size}"


def build_reference_cache_key(
    model_name: str,
    include_lord_refs: bool,
    assets: list[dict[str, Any]],
) -> str:
    sigs: list[str] = []
    for hero in assets:
        sigs.append(str(hero.get("id", "")))
        for p in hero.get("files", []):
            if isinstance(p, str) and os.path.isfile(p):
                sigs.append(file_signature(p))
    payload = json.dumps(
        {
            "model": model_name,
            "include_lord_refs": include_lord_refs,
            "assets": sigs,
        },
        sort_keys=True,
    )
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_clip_reference_cache(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def write_clip_reference_cache(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def extract_feature_vector(image_features: Any) -> np.ndarray:
    """Handle CLIP image feature outputs across transformers versions."""
    if hasattr(image_features, "cpu"):
        return image_features.cpu().numpy().astype(np.float32)
    if hasattr(image_features, "image_embeds"):
        emb = image_features.image_embeds
        if hasattr(emb, "cpu"):
            return emb.cpu().numpy().astype(np.float32)
    if hasattr(image_features, "pooler_output"):
        emb = image_features.pooler_output
        if hasattr(emb, "cpu"):
            return emb.cpu().numpy().astype(np.float32)
    if isinstance(image_features, (tuple, list)) and len(image_features) > 0:
        head = image_features[0]
        if hasattr(head, "cpu"):
            return head.cpu().numpy().astype(np.float32)
    raise ValueError("Unsupported CLIP image_features output format.")


def maybe_gemini_tiebreak_prediction(
    crop_bgr: np.ndarray,
    candidates: list[str],
) -> tuple[str | None, dict[str, Any] | None]:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None, {"used": False, "reason": "GEMINI_API_KEY not set"}
    if crop_bgr.size == 0:
        return None, {"used": False, "reason": "empty_crop"}
    uniq_candidates = [c for c in dict.fromkeys(candidates) if isinstance(c, str) and c.strip()]
    if len(uniq_candidates) < 2:
        return None, {"used": False, "reason": "not_enough_candidates"}

    ok, enc = cv2.imencode(".png", crop_bgr)
    if not ok:
        return None, {"used": False, "reason": "encode_failed"}
    image_b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    model = (os.getenv("GEMINI_TIEBREAKER_MODEL") or "gemini-flash-latest").strip()
    endpoint = (
        os.getenv("GEMINI_TIEBREAKER_ENDPOINT")
        or f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    ).strip()

    allowed = ", ".join(f'"{c}"' for c in uniq_candidates)
    lower_candidates = {c.lower() for c in uniq_candidates}
    has_thor = any("thor" in c for c in lower_candidates)
    has_loki = any("loki" in c for c in lower_candidates)
    has_magik = any("magik" in c for c in lower_candidates)
    extra_rule = ""
    if has_thor and has_loki and has_magik:
        extra_rule = (
            '\nVisual disambiguation rule for these candidates: '
            'If you see lightning/blue electricity and silver wings, it is Thor. '
            'If you see more green it is Loki. If you see more yellow it is Magik.'
        )
    prompt = (
        "You are resolving a hero avatar classification tie. "
        "Return ONLY JSON with keys hero_id and confidence.\n"
        f"Allowed hero_id values: [{allowed}]\n"
        "Pick exactly one hero_id from that list. Never output anything else."
        f"{extra_rule}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                ],
            },
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "required": ["hero_id", "confidence"],
                "properties": {
                    "hero_id": {"type": "STRING", "enum": uniq_candidates},
                    "confidence": {"type": "NUMBER"},
                },
            },
        },
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        return None, {"used": False, "reason": f"request_failed: {e}"}
    try:
        parsed = json.loads(raw)
    except Exception:
        return None, {"used": False, "reason": "invalid_json_response"}

    choice: dict[str, Any] | None = None
    candidates_blob = parsed.get("candidates")
    if isinstance(candidates_blob, list):
        for cand in candidates_blob:
            if not isinstance(cand, dict):
                continue
            content = cand.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for p in parts:
                if not isinstance(p, dict):
                    continue
                text = p.get("text")
                if isinstance(text, str) and text.strip():
                    try:
                        maybe = json.loads(text)
                        if isinstance(maybe, dict):
                            choice = maybe
                            break
                    except Exception:
                        continue
            if choice is not None:
                break
    if choice is None:
        return None, {"used": False, "reason": "missing_candidates_json"}
    hero_id = choice.get("hero_id")
    if isinstance(hero_id, str) and hero_id in uniq_candidates:
        return hero_id, {"used": True, "provider": "gemini", "model": model, "candidates": uniq_candidates, "raw_choice": choice}
    return None, {"used": False, "reason": "invalid_choice", "raw_choice": choice}


def compute_clip_matches(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    hero_assets_path: str | None,
    model_name: str,
    top_k: int,
    margin_threshold: float,
    require_confidence: bool,
    include_lord_refs: bool,
    cache_path: str | None,
    include_embedding_debug: bool = False,
    use_multi_crop_ensemble: bool = True,
    low_score_label: str = "unknown",
    min_confidence_score: float = 0.8,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    assets = load_hero_assets(hero_assets_path)
    if not include_lord_refs:
        assets = [a for a in assets if not str(a.get("id", "")).endswith("-lord")]
    if not assets:
        return [], {"enabled": False, "skipped": "Hero reference asset map not found or empty."}
    try:
        runtime = _CLIP_RUNTIME_CACHE.get(model_name)
        if runtime is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            model_source = resolve_local_clip_model_source(model_name)
            model = CLIPModel.from_pretrained(model_source, local_files_only=model_source != model_name)
            processor = CLIPProcessor.from_pretrained(model_source, local_files_only=model_source != model_name)
            model.eval()
            runtime = {
                "torch": torch,
                "model": model,
                "processor": processor,
                "model_source": model_source,
            }
            _CLIP_RUNTIME_CACHE[model_name] = runtime
        torch = runtime["torch"]
        model = runtime["model"]
        processor = runtime["processor"]
    except Exception as e:
        return [], {
            "enabled": False,
            "skipped": f"CLIP dependencies unavailable: {e}. Install requirements-ml.txt",
        }

    resolved_cache_path = cache_path or os.path.join(os.getcwd(), DEFAULT_CLIP_CACHE_REL)
    cache_key = build_reference_cache_key(model_name, include_lord_refs, assets)
    cache_blob = read_clip_reference_cache(resolved_cache_path)
    cached_entry = cache_blob.get(cache_key) if isinstance(cache_blob, dict) else None

    ref_hero_ids: list[str] = []
    ref_image_paths: list[str] = []
    ref_color_hists: list[np.ndarray | None] = []
    ref_hue_profiles: list[np.ndarray | None] = []
    ref_features: np.ndarray | None = None
    used_cache = False
    if isinstance(cached_entry, dict):
        raw_ids = cached_entry.get("ref_hero_ids")
        raw_features = cached_entry.get("ref_features")
        if isinstance(raw_ids, list) and isinstance(raw_features, list):
            try:
                ids = [str(v) for v in raw_ids]
                feats = np.array(raw_features, dtype=np.float32)
                if feats.ndim == 2 and feats.shape[0] == len(ids):
                    ref_hero_ids = ids
                    ref_features = feats
                    used_cache = True
            except Exception:
                pass

    if ref_features is None:
        ref_images: list[np.ndarray] = []
        for hero in assets:
            hid = hero["id"]
            for p in hero["files"]:
                img = cv2.imread(p)
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ref_images.append(rgb)
                ref_hero_ids.append(hid)
                ref_image_paths.append(p)
                ref_color_hists.append(compute_hsv_histogram(img))
                ref_hue_profiles.append(compute_hue_profile(img))
        if not ref_images:
            return [], {"enabled": False, "skipped": "No readable hero reference images."}
        with torch.no_grad():
            ref_inputs = processor(images=ref_images, return_tensors="pt")
            ref_outputs = model.get_image_features(**ref_inputs)
            ref_features = extract_feature_vector(ref_outputs)
        ref_features = np.vstack([l2_normalize(v) for v in ref_features])
        cache_blob = cache_blob if isinstance(cache_blob, dict) else {}
        cache_blob[cache_key] = {
            "ref_hero_ids": ref_hero_ids,
            "ref_features": ref_features.tolist(),
        }
        write_clip_reference_cache(resolved_cache_path, cache_blob)

    if ref_features is None or len(ref_hero_ids) == 0:
        return [], {"enabled": False, "skipped": "No readable hero reference images."}
    if len(ref_image_paths) != len(ref_hero_ids):
        ref_image_paths = []
        ref_color_hists = []
        ref_hue_profiles = []
        for hero in assets:
            for p in hero["files"]:
                if os.path.isfile(p):
                    ref_image_paths.append(p)
                    img = cv2.imread(p)
                    ref_color_hists.append(compute_hsv_histogram(img) if img is not None else None)
                    ref_hue_profiles.append(compute_hue_profile(img) if img is not None else None)
        if len(ref_image_paths) != len(ref_hero_ids):
            ref_image_paths = ["" for _ in ref_hero_ids]
            ref_color_hists = [None for _ in ref_hero_ids]
            ref_hue_profiles = [None for _ in ref_hero_ids]

    rows: list[dict[str, Any]] = []
    pred_classes: list[str] = []
    color_prefilter_enabled = str(os.getenv("CLIP_COLOR_PREFILTER_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
    color_prefilter_topn = max(1, int(os.getenv("CLIP_COLOR_PREFILTER_TOPN", "200")))
    hue_bias_enabled = str(os.getenv("CLIP_HUE_BIAS_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
    hue_bias_weight = float(os.getenv("CLIP_HUE_BIAS_WEIGHT", "0.08"))
    hue_profile_by_index: dict[int, np.ndarray] = {
        i: hp for i, hp in enumerate(ref_hue_profiles) if hp is not None
    }
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            rows.append({"index": idx, "error": "empty crop"})
            continue
        crop_views = build_crop_views_rgb(crop) if use_multi_crop_ensemble else [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)]
        if not crop_views:
            rows.append({"index": idx, "error": "empty crop views"})
            continue
        with torch.no_grad():
            inp = processor(images=crop_views, return_tensors="pt")
            emb_outputs = model.get_image_features(**inp)
            emb_batch = extract_feature_vector(emb_outputs).astype(np.float32)
        emb_norm = np.vstack([l2_normalize(v) for v in emb_batch])
        emb = l2_normalize(np.mean(emb_norm, axis=0))
        consider_indices = np.arange(len(ref_hero_ids))
        if color_prefilter_enabled and len(ref_hero_ids) > color_prefilter_topn:
            crop_hist = compute_hsv_histogram(crop)
            if crop_hist is not None:
                color_scores = np.full(len(ref_hero_ids), -1.0, dtype=np.float32)
                for ridx, rh in enumerate(ref_color_hists):
                    if rh is None:
                        continue
                    dist = cv2.compareHist(crop_hist, rh, cv2.HISTCMP_BHATTACHARYYA)
                    color_scores[ridx] = float(max(-1.0, min(1.0, 1.0 - dist)))
                consider_indices = np.argsort(-color_scores)[:color_prefilter_topn]
        sims = ref_features @ emb
        final_scores = sims
        order = consider_indices[np.argsort(-final_scores[consider_indices])]
        matches: list[dict[str, Any]] = []
        best_by_hero: dict[str, float] = {}
        best_raw_by_alias: dict[str, str] = {}
        for ref_idx in order:
            raw_hero_id = ref_hero_ids[int(ref_idx)]
            hero_id = normalize_hero_id(raw_hero_id)
            score = float(final_scores[int(ref_idx)])
            prev = best_by_hero.get(hero_id)
            if prev is None or score > prev:
                best_by_hero[hero_id] = score
                best_raw_by_alias[hero_id] = raw_hero_id
            if len(best_by_hero) >= max(1, top_k * 2):
                break
        hero_ranked = sorted(best_by_hero.items(), key=lambda item: item[1], reverse=True)[: max(1, top_k)]
        for hero_id, score in hero_ranked:
            matches.append(
                {
                    "hero_id": hero_id,
                    "score": round(score, 6),
                    "raw_hero_id": best_raw_by_alias.get(hero_id, hero_id),
                },
            )
        chosen = "unknown"
        confident = False
        margin = 0.0
        if len(matches) > 0:
            chosen = matches[0]["hero_id"]
            top1 = float(matches[0]["score"])
            top2 = float(matches[1]["score"]) if len(matches) > 1 else -1.0
            margin = top1 - top2
            confident = margin >= margin_threshold
            if not confident and require_confidence:
                chosen = "unknown"
            if top1 < float(min_confidence_score):
                chosen = low_score_label
                confident = False
        tiebreak_info: dict[str, Any] | None = None
        tie_margin_max = float(os.getenv("GEMINI_TIEBREAKER_MARGIN_MAX", "0.03"))
        tie_topn = int(os.getenv("GEMINI_TIEBREAKER_TOPN", "5"))
        tie_enabled = str(os.getenv("GEMINI_TIEBREAKER_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
        if (
            tie_enabled
            and len(matches) >= 2
            and margin <= tie_margin_max
        ):
            allowed_rows = matches[: max(2, tie_topn)]
            if hue_bias_enabled:
                crop_hue_profile = compute_hue_profile(crop)
                if crop_hue_profile is not None:
                    rescored: list[tuple[float, dict[str, Any]]] = []
                    for m in allowed_rows:
                        raw_id = str(m.get("raw_hero_id") or m.get("hero_id") or "")
                        bonus = 0.0
                        for ridx, ref_raw in enumerate(ref_hero_ids):
                            if ref_raw != raw_id:
                                continue
                            hp = hue_profile_by_index.get(ridx)
                            if hp is None:
                                continue
                            dist = np.linalg.norm(crop_hue_profile - hp)
                            hue_sim = 1.0 - float(dist / np.sqrt(2.0))
                            bonus = max(bonus, hue_bias_weight * hue_sim)
                        rescored.append((float(m.get("score", 0.0)) + bonus, m))
                    rescored.sort(key=lambda x: x[0], reverse=True)
                    allowed_rows = [row for _, row in rescored]
            allowed = [str(m.get("hero_id")) for m in allowed_rows if isinstance(m.get("hero_id"), str)]
            tie_choice, tiebreak_info = maybe_gemini_tiebreak_prediction(crop, allowed)
            if isinstance(tie_choice, str) and tie_choice in allowed:
                chosen = tie_choice
        if chosen not in {"unknown", "unselected"}:
            pred_classes.append(chosen)
        row: dict[str, Any] = {
            "index": idx,
            "top_matches": matches,
            "predicted_hero": chosen,
            "confident": confident,
            "margin_top1_top2": round(margin, 6),
            "ensemble_views": len(crop_views),
            "hue_bias_weight": hue_bias_weight if hue_bias_enabled else 0.0,
            "min_confidence_score": float(min_confidence_score),
        }
        if tiebreak_info is not None:
            row["gemini_tiebreak"] = tiebreak_info
        if include_embedding_debug:
            top_ref_idx = int(order[0]) if len(order) > 0 else -1
            row["crop_embedding"] = emb.tolist()
            row["matched_asset_embedding"] = (
                ref_features[top_ref_idx].astype(np.float32).tolist()
                if top_ref_idx >= 0
                else None
            )
            row["matched_asset_path"] = (
                ref_image_paths[top_ref_idx]
                if top_ref_idx >= 0 and top_ref_idx < len(ref_image_paths)
                else None
            )
            row["matched_asset_name"] = (
                os.path.basename(ref_image_paths[top_ref_idx])
                if top_ref_idx >= 0 and top_ref_idx < len(ref_image_paths)
                else None
            )
        rows.append(row)

    return rows, {
        "enabled": True,
        "ok": True,
        "model": model_name,
        "model_source": str(runtime.get("model_source", model_name)) if isinstance(runtime, dict) else model_name,
        "reference_count": len(ref_hero_ids),
        "hero_count": len(assets),
        "top_k": max(1, top_k),
        "margin_threshold": margin_threshold,
        "require_confidence": require_confidence,
        "include_lord_refs": include_lord_refs,
        "reference_cache_path": resolved_cache_path,
        "used_reference_cache": used_cache,
        "prediction_classes": pred_classes,
    }


def resolve_local_clip_model_source(model_name: str) -> str:
    if os.path.isdir(model_name):
        return model_name
    if "/" not in model_name:
        return model_name
    owner, repo = model_name.split("/", 1)
    folder = f"models--{owner}--{repo}"
    candidates: list[str] = []
    cwd = os.getcwd()
    candidates.append(os.path.join(cwd, ".cache", "huggingface", folder, "snapshots"))
    candidates.append(os.path.join(cwd, ".cache", "huggingface", "hub", folder, "snapshots"))
    home_cache = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub", folder, "snapshots"))
    candidates.append(home_cache)
    for snaps in candidates:
        if not os.path.isdir(snaps):
            continue
        try:
            entries = sorted(os.listdir(snaps), reverse=True)
        except Exception:
            continue
        for entry in entries:
            p = os.path.join(snaps, entry)
            if not os.path.isdir(p):
                continue
            config_file = os.path.join(p, "config.json")
            pt_file = os.path.join(p, "pytorch_model.bin")
            if os.path.isfile(config_file) and os.path.isfile(pt_file):
                return p
    return model_name


def infer_payload(
    source: str,
    layout: str | None,
    normalize_size: int,
    hero_assets: str | None,
    clip_model: str,
    clip_top_k: int,
    clip_margin_threshold: float,
    clip_require_confidence: bool,
    clip_include_lord_refs: bool,
    clip_cache_file: str | None,
    align_method: str,
    strict_count: bool,
    debug_dir: str | None,
    debug_embedding_json_dir: str | None,
) -> dict[str, Any]:
    layout_data, layout_resolved = load_layout(layout)
    rw = int(layout_data.get("reference_width", 1920))
    rh = int(layout_data.get("reference_height", 1080))
    try:
        ref_boxes = boxes_from_layout(layout_data)
    except (KeyError, TypeError, ValueError) as e:
        return {"ok": False, "error": f"Invalid layout: {e}"}

    frame = cv2.imread(source)
    if frame is None:
        return {"ok": False, "error": f"Could not read image: {source}"}

    ih, iw = frame.shape[:2]
    icon_boxes: list[dict[str, Any]] = []
    layout_mode = detect_layout_mode(layout_data, layout_resolved)
    order = str(layout_data.get("order", "top_to_bottom"))
    expected_count = int(layout_data.get("expected_count", len(ref_boxes)))
    coarse_boxes, registration = register_layout_boxes(
        frame=frame,
        ref_boxes=ref_boxes,
        rw=rw,
        rh=rh,
        layout=layout_data,
        layout_resolved=layout_resolved,
        align_method=align_method,
    )
    packed: list[dict[str, Any]] = []
    for idx, b in enumerate(coarse_boxes):
        refined, meta = refine_icon_box(frame, b, layout_mode=layout_mode)
        packed.append(
            {
                "ref_box": ref_boxes[idx] if idx < len(ref_boxes) else (0, 0, 0, 0),
                "coarse": b,
                "refined": refined,
                "meta": meta,
            },
        )
    ordered_boxes = reorder_boxes([p["refined"] for p in packed], order)
    used = set()
    ordered_packed: list[dict[str, Any]] = []
    for ob in ordered_boxes:
        chosen_idx = -1
        for j, p in enumerate(packed):
            if j in used:
                continue
            if p["refined"] == ob:
                chosen_idx = j
                break
        if chosen_idx < 0:
            continue
        used.add(chosen_idx)
        ordered_packed.append(packed[chosen_idx])
    refined_boxes = [p["refined"] for p in ordered_packed]
    if layout_mode == "post_game":
        refined_boxes = snap_postgame_rows(refined_boxes, frame=frame)
    elif layout_mode == "in_game":
        refined_boxes = recenter_ingame_row_windows(frame, refined_boxes)
    scaled_boxes = refined_boxes

    for i, (x1, y1, x2, y2) in enumerate(refined_boxes):
        rb = ordered_packed[i]["ref_box"] if i < len(ordered_packed) else (0, 0, x2 - x1, y2 - y1)
        rx, ry, rw_, rh_ = rb
        crop = frame[y1:y2, x1:x2]
        quality = quality_metrics(crop, layout_mode=layout_mode)
        entry: dict[str, Any] = {
            "index": i,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "reference_box": {"x": rx, "y": ry, "w": rw_, "h": rh_},
            "quality": quality,
        }
        if i < len(ordered_packed):
            entry["refine"] = ordered_packed[i]["meta"]
        if normalize_size > 0 and crop.size > 0:
            small = normalize_crop(crop, normalize_size)
            entry["crop_shape"] = {"width": normalize_size, "height": normalize_size, "channels": 3}
            entry["crop_mean_rgb"] = [
                round(float(small[:, :, c].mean()), 4) for c in range(3)
            ]
        icon_boxes.append(entry)

    include_lord_refs_effective = bool(clip_include_lord_refs) or layout_mode == "pre_game"
    low_score_label = "unselected" if layout_mode == "pre_game" else "unknown"
    min_confidence_score = max(0.8, float(os.getenv("CLIP_MIN_SCORE", "0.8")))
    clip_rows, clip_meta = compute_clip_matches(
        frame=frame,
        boxes=scaled_boxes,
        hero_assets_path=hero_assets,
        model_name=str(clip_model),
        top_k=int(clip_top_k),
        margin_threshold=float(clip_margin_threshold),
        require_confidence=bool(clip_require_confidence),
        include_lord_refs=include_lord_refs_effective,
        cache_path=clip_cache_file,
        include_embedding_debug=bool(debug_embedding_json_dir),
        use_multi_crop_ensemble=True,
        low_score_label=low_score_label,
        min_confidence_score=min_confidence_score,
    )
    clip_predictions: list[str] = []
    if clip_meta and isinstance(clip_meta.get("prediction_classes"), list):
        clip_predictions = [str(v) for v in clip_meta["prediction_classes"] if isinstance(v, str)]
    clip_by_index = {int(row["index"]): row for row in clip_rows if "index" in row}
    for box in icon_boxes:
        row = clip_by_index.get(int(box["index"]))
        if row is not None:
            box["clip"] = row

    widths = [max(1, b[2] - b[0]) for b in refined_boxes]
    heights = [max(1, b[3] - b[1]) for b in refined_boxes]
    w_med = float(np.median(widths)) if widths else 1.0
    h_med = float(np.median(heights)) if heights else 1.0
    for item in icon_boxes:
        b = item["box"]
        w = max(1, int(b["x2"]) - int(b["x1"]))
        h = max(1, int(b["y2"]) - int(b["y1"]))
        within_size = (0.7 * w_med <= w <= 1.35 * w_med) and (0.7 * h_med <= h <= 1.35 * h_med)
        q = item.get("quality")
        base_ok = bool(isinstance(q, dict) and q.get("ok"))
        item["quality_ok"] = bool(base_ok and within_size)
        item["size_consistency_ok"] = bool(within_size)

    quality_ok_count = sum(1 for item in icon_boxes if bool(item.get("quality_ok")))
    count_ok = len(icon_boxes) == expected_count
    if strict_count and not count_ok:
        return {
            "ok": False,
            "error": f"Expected {expected_count} icon boxes but extracted {len(icon_boxes)}.",
            "layout": {
                "reference_width": rw,
                "reference_height": rh,
                "layout_file": layout_resolved,
            },
        }

    debug_meta: dict[str, Any] | None = None
    if debug_dir:
        debug_meta = write_debug_outputs(
            frame=frame,
            source_path=source,
            debug_dir=debug_dir,
            coarse_boxes=coarse_boxes,
            refined_boxes=refined_boxes,
        )
    embedding_debug_json_path: str | None = None
    if debug_embedding_json_dir and clip_rows:
        os.makedirs(debug_embedding_json_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(source))[0]
        embedding_debug_json_path = os.path.join(
            debug_embedding_json_dir,
            f"{stem}_crop_asset_embeddings.json",
        )
        crop_names: list[str] = []
        if isinstance(debug_meta, dict) and isinstance(debug_meta.get("crops"), list):
            crop_names = [os.path.basename(str(p)) for p in debug_meta["crops"]]
        mapping: dict[str, Any] = {}
        for row in clip_rows:
            idx = int(row.get("index", -1))
            crop_key = (
                crop_names[idx]
                if idx >= 0 and idx < len(crop_names)
                else f"{stem}_crop_{idx:02d}.png"
            )
            mapping[crop_key] = [
                row.get("crop_embedding"),
                row.get("matched_asset_name"),
                row.get("matched_asset_embedding"),
            ]
        with open(embedding_debug_json_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f)

    return {
        "ok": True,
        "pipeline": "opencv_icon_boxes",
        "summary": summarize(len(icon_boxes), iw, ih, rw, rh),
        "image": {"width": iw, "height": ih},
        "layout": {
            "reference_width": rw,
            "reference_height": rh,
            "layout_file": layout_resolved,
            "layout_mode": layout_mode,
            "order": order,
            "expected_count": expected_count,
            "count_ok": count_ok,
            "registration": registration,
        },
        "icon_boxes": icon_boxes,
        "quality": {
            "quality_ok_count": quality_ok_count,
            "total": len(icon_boxes),
            "all_quality_ok": quality_ok_count == len(icon_boxes),
        },
        "debug": debug_meta,
        "embedding_debug_json": embedding_debug_json_path,
        "clip": clip_meta,
        "prediction_classes": clip_predictions,
    }


def main() -> None:
    load_local_env_file()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=False)
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
    parser.add_argument(
        "--hero-assets",
        default=None,
        help=f"Hero reference assets JSON (default: {DEFAULT_HERO_ASSETS_REL} if present)",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="HF model id for CLIP image embeddings",
    )
    parser.add_argument(
        "--clip-top-k",
        type=int,
        default=5,
        help="Top-K hero matches to include for each crop",
    )
    parser.add_argument(
        "--clip-margin-threshold",
        type=float,
        default=0.03,
        help="Require top1-top2 score margin for confident prediction",
    )
    parser.add_argument(
        "--clip-require-confidence",
        action="store_true",
        help="Drop prediction when top1-top2 margin is below threshold",
    )
    parser.add_argument(
        "--clip-include-lord-refs",
        action="store_true",
        help="Include -lord reference avatars in matching pool",
    )
    parser.add_argument(
        "--clip-cache-file",
        default=None,
        help=f"Reference embedding cache file (default: {DEFAULT_CLIP_CACHE_REL})",
    )
    parser.add_argument(
        "--align-method",
        default="auto",
        choices=["auto", "homography", "phasecorr", "scaled"],
        help="Layout alignment mode before per-icon refinement",
    )
    parser.add_argument(
        "--strict-count",
        action="store_true",
        help="Fail payload when extracted icon count does not match layout expected_count",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory to write overlay/crop debug images",
    )
    parser.add_argument(
        "--debug-embedding-json-dir",
        default=None,
        help="Optional directory to write crop->(crop emb, matched asset, asset emb) JSON",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run as a long-lived JSONL worker process",
    )
    args = parser.parse_args()
    if args.worker:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            if line == "__QUIT__":
                break
            req_id = "unknown"
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise ValueError("worker request must be an object")
                req_id = str(req.get("id", "unknown"))
                payload = req.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("worker payload missing")
                result = infer_payload(
                    source=str(payload.get("source") or ""),
                    layout=str(payload.get("layout")) if payload.get("layout") is not None else None,
                    normalize_size=int(payload.get("normalize_size", 128)),
                    hero_assets=str(payload.get("hero_assets")) if payload.get("hero_assets") is not None else None,
                    clip_model=str(payload.get("clip_model") or "openai/clip-vit-base-patch32"),
                    clip_top_k=int(payload.get("clip_top_k", 5)),
                    clip_margin_threshold=float(payload.get("clip_margin_threshold", 0.03)),
                    clip_require_confidence=bool(payload.get("clip_require_confidence", False)),
                    clip_include_lord_refs=bool(payload.get("clip_include_lord_refs", False)),
                    clip_cache_file=str(payload.get("clip_cache_file")) if payload.get("clip_cache_file") is not None else None,
                    align_method=str(payload.get("align_method", "auto")),
                    strict_count=bool(payload.get("strict_count", False)),
                    debug_dir=str(payload.get("debug_dir")) if payload.get("debug_dir") is not None else None,
                    debug_embedding_json_dir=(
                        str(payload.get("debug_embedding_json_dir"))
                        if payload.get("debug_embedding_json_dir") is not None
                        else None
                    ),
                )
                print(f"{WORKER_RESPONSE_PREFIX}{json.dumps({'id': req_id, 'ok': True, 'result': result})}", flush=True)
            except Exception as e:
                print(
                    f"{WORKER_RESPONSE_PREFIX}{json.dumps({'id': req_id, 'ok': False, 'error': str(e), 'trace': traceback.format_exc(limit=2)})}",
                    flush=True,
                )
        return

    if not args.source:
        print(json.dumps({"ok": False, "error": "--source is required when not running in --worker mode"}))
        sys.exit(1)

    payload = infer_payload(
        source=args.source,
        layout=args.layout,
        normalize_size=int(args.normalize_size),
        hero_assets=args.hero_assets,
        clip_model=str(args.clip_model),
        clip_top_k=int(args.clip_top_k),
        clip_margin_threshold=float(args.clip_margin_threshold),
        clip_require_confidence=bool(args.clip_require_confidence),
        clip_include_lord_refs=bool(args.clip_include_lord_refs),
        clip_cache_file=args.clip_cache_file,
        align_method=str(args.align_method),
        strict_count=bool(args.strict_count),
        debug_dir=str(args.debug_dir) if args.debug_dir is not None else None,
        debug_embedding_json_dir=(
            str(args.debug_embedding_json_dir)
            if args.debug_embedding_json_dir is not None
            else None
        ),
    )
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
