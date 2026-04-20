#!/usr/bin/env python3
"""Merge detector.parquet rows with COCO-style auto_crop.json into raw JSONL for teacher labeling.

Expected parquet columns (aliases supported):
  - image: relative or absolute path; basename is matched to auto_crop images[].file_name
  - id_class or class_id: integer class id
  - box_cx, box_cy, box_w, box_h: normalized floats in [0, 1]

Optional parquet columns (ignored by downstream normalize_record if absent):
  - split: train|val|test

auto_crop.json: COCO-like { "images": [...], "annotations": [...] } with pixel bbox [x, y, w, h].
Per image, the annotation with the highest ``score`` is used as autocrop_top1 (normalized x1..y2).

Example:
  python scripts/merge_detector_parquet_autocrop_json.py \\
    --detector-parquet /path/to/detector.parquet \\
    --auto-crop-json /path/to/auto_crop.json \\
    --output-jsonl /path/to/raw_for_teacher.jsonl \\
    --root-dir /dataset/root
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detector-parquet", required=True, type=Path, help="Path to detector.parquet")
    p.add_argument("--auto-crop-json", required=True, type=Path, help="Path to COCO-style auto crop JSON")
    p.add_argument("--output-jsonl", required=True, type=Path, help="Output JSONL path")
    p.add_argument(
        "--root-dir",
        default=None,
        help="Dataset root prepended when resolving image paths (same as other pipeline scripts).",
    )
    p.add_argument(
        "--class-id-to-name",
        type=Path,
        default=None,
        help='Optional JSON mapping class id to name, e.g. {"0": "object"}. Default names: class_<id>.',
    )
    p.add_argument(
        "--match-key",
        choices=("basename", "file_name"),
        default="basename",
        help="Match parquet image column to auto_crop images[].file_name by basename (default) or exact string.",
    )
    p.add_argument(
        "--min-detector-objects",
        type=int,
        default=1,
        help="Skip images with fewer than this many detector rows (default: 1). Use 0 to allow empty lists.",
    )
    return p.parse_args()


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to read parquet. Install with: pip install pyarrow"
        ) from exc

    table = pq.read_table(path)
    return table.to_pylist()


def _class_name(class_id: int, mapping: dict[str, str] | None) -> str:
    if mapping is not None:
        key = str(class_id)
        if key in mapping:
            return mapping[key]
    return f"class_{class_id}"


def _get_int(row: dict[str, Any], *keys: str) -> int:
    for k in keys:
        if k in row and row[k] is not None:
            return int(row[k])
    raise KeyError(f"need one of: {keys}")


def _get_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _detector_object_from_row(row: dict[str, Any], class_names: dict[str, str] | None) -> dict[str, Any]:
    class_id = _get_int(row, "id_class", "class_id")
    return {
        "class_id": class_id,
        "class_name": _class_name(class_id, class_names),
        "box_cx": round(_get_float(row, "box_cx"), 4),
        "box_cy": round(_get_float(row, "box_cy"), 4),
        "box_w": round(_get_float(row, "box_w"), 4),
        "box_h": round(_get_float(row, "box_h"), 4),
    }


def _load_auto_crop(
    path: Path,
) -> tuple[
    dict[int, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    images = raw.get("images") or []
    annotations = raw.get("annotations") or []

    by_id: dict[int, dict[str, Any]] = {}
    for im in images:
        iid = int(im["id"])
        by_id[iid] = {
            "file_name": im["file_name"],
            "width": int(im["width"]),
            "height": int(im["height"]),
        }

    by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        by_image_id[int(ann["image_id"])].append(ann)

    # Precompute best annotation per image by score
    best_by_image_id: dict[int, dict[str, Any]] = {}
    for image_id, anns in by_image_id.items():
        if not anns:
            continue
        best = max(anns, key=lambda a: float(a.get("score", 0.0)))
        best_by_image_id[image_id] = best

    # file_name -> meta (for basename matching)
    by_file_name: dict[str, dict[str, Any]] = {}
    for iid, meta in by_id.items():
        fn = meta["file_name"]
        by_file_name[fn] = {**meta, "image_id": iid}

    return by_id, by_file_name, best_by_image_id


def _bbox_xywh_to_norm_top1(
    bbox: list[float | int],
    width: int,
    height: int,
    score: float,
) -> dict[str, Any]:
    x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    x1 = x / width
    y1 = y / height
    x2 = (x + w) / width
    y2 = (y + h) / height
    return {
        "x1": round(x1, 4),
        "y1": round(y1, 4),
        "x2": round(x2, 4),
        "y2": round(y2, 4),
        "score": round(float(score), 4),
    }


def main() -> int:
    args = parse_args()
    class_names: dict[str, str] | None = None
    if args.class_id_to_name is not None:
        class_names = json.loads(args.class_id_to_name.read_text(encoding="utf-8"))

    rows = _load_parquet(args.detector_parquet)
    if not rows:
        print("Warning: detector parquet is empty", file=sys.stderr)
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text("", encoding="utf-8")
        return 0

    _, by_file_name, best_by_image_id = _load_auto_crop(args.auto_crop_json)

    # Group detector rows by match key
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    canonical_image: dict[str, str] = {}
    for row in rows:
        img = row.get("image")
        if not isinstance(img, str) or not img.strip():
            continue
        key = img.strip() if args.match_key == "file_name" else Path(img).name
        grouped[key].append(row)
        # Keep first seen full path string for that key
        if key not in canonical_image:
            canonical_image[key] = img.strip()

    written = 0
    skipped_no_autocrop = 0
    skipped_no_detector = 0
    skipped_short_detector = 0

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for key, det_rows in sorted(grouped.items()):
            detector_objects = [_detector_object_from_row(r, class_names) for r in det_rows]
            if len(detector_objects) < args.min_detector_objects:
                skipped_short_detector += 1
                continue

            if args.match_key == "basename":
                file_name = key
                meta = by_file_name.get(file_name)
            else:
                # exact path: try basename lookup anyway for auto_crop which uses file_name only
                file_name = Path(key).name
                meta = by_file_name.get(file_name)

            if meta is None:
                skipped_no_autocrop += 1
                continue

            image_id = int(meta["image_id"])
            best_ann = best_by_image_id.get(image_id)
            if best_ann is None:
                skipped_no_autocrop += 1
                continue

            w, h = int(meta["width"]), int(meta["height"])
            autocrop_top1 = _bbox_xywh_to_norm_top1(
                list(best_ann["bbox"]),
                w,
                h,
                float(best_ann.get("score", 0.0)),
            )

            image_path = canonical_image[key]
            sample_id = Path(file_name).stem
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "image": image_path,
                "detector_objects": detector_objects,
                "autocrop_top1": autocrop_top1,
            }
            split = det_rows[0].get("split")
            if isinstance(split, str) and split.strip():
                record["split"] = split.strip().lower()

            out.write(json.dumps(record, ensure_ascii=True))
            out.write("\n")
            written += 1

    print(
        json.dumps(
            {
                "written": written,
                "skipped_no_matching_autocrop": skipped_no_autocrop,
                "skipped_detector_below_min": skipped_short_detector,
                "output": str(args.output_jsonl.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
