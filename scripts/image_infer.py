"""
Headless quad-model inference on a single image. Prints one JSON object to stdout.

Usage:
  python3 scripts/image_infer.py \\
    --model1 models/MarvelRivals-Detection-Suite/pt/hero.pt \\
    --model2 models/MarvelRivals-Detection-Suite/pt/hp.pt \\
    --model3 models/MarvelRivals-Detection-Suite/pt/ui.pt \\
    --model4 models/MarvelRivals-Detection-Suite/pt/friendfoe.pt \\
    --source /path/to/image.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_KEYS = ("hero", "hp", "ui", "friendfoe")


def run_detection(index: int, frame: np.ndarray, models: list, results: list) -> None:
    results[index] = models[index](frame, verbose=False)[0]


def boxes_to_list(result, labels: dict, thresh: float) -> list[dict]:
    out: list[dict] = []
    for det in result.boxes:
        conf = float(det.conf.item())
        if conf < thresh:
            continue
        class_idx = int(det.cls.item())
        xyxy = det.xyxy.cpu().numpy().reshape(-1).astype(int).tolist()
        if len(xyxy) != 4:
            continue
        name = labels.get(class_idx, str(class_idx))
        out.append(
            {
                "label": name,
                "confidence": round(conf, 4),
                "box": {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
            }
        )
    return out


def summarize(models_payload: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    for key in MODEL_KEYS:
        dets = models_payload.get(key, [])
        if not dets:
            lines.append(f"{key}: (no detections above threshold)")
            continue
        top = sorted(dets, key=lambda x: x["confidence"], reverse=True)[:8]
        parts = [f"{x['label']} {x['confidence']:.0%}" for x in top]
        extra = len(dets) - len(top)
        suffix = f" … +{extra} more" if extra > 0 else ""
        lines.append(f"{key}: {', '.join(parts)}{suffix}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", required=True)
    parser.add_argument("--model2", required=True)
    parser.add_argument("--model3", required=True)
    parser.add_argument("--model4", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--thresh", type=float, default=0.25)
    args = parser.parse_args()

    paths = [args.model1, args.model2, args.model3, args.model4]
    for p in paths:
        if not os.path.isfile(p):
            print(json.dumps({"ok": False, "error": f"Model not found: {p}"}))
            sys.exit(1)

    frame = cv2.imread(args.source)
    if frame is None:
        print(json.dumps({"ok": False, "error": f"Could not read image: {args.source}"}))
        sys.exit(1)

    models = [YOLO(p) for p in paths]
    label_maps = [m.names for m in models]

    raw: list = [None, None, None, None]
    threads = []
    for i in range(4):
        t = threading.Thread(
            target=run_detection,
            args=(i, frame, models, raw),
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    models_out: dict[str, list[dict]] = {}
    for i, key in enumerate(MODEL_KEYS):
        models_out[key] = boxes_to_list(raw[i], label_maps[i], args.thresh)

    payload = {
        "ok": True,
        "threshold": args.thresh,
        "summary": summarize(models_out),
        "models": models_out,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
