#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CLASS_MAPPING = {0: "human", 15: "cat", 16: "dog", 1000: "face"}
DETECTOR_COLUMNS = ("image", "id_class", "box_cx", "box_cy", "box_w", "box_h")


def _load_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required. Install with: python -m pip install -e .[teacher]") from exc
    return pd


def _read_class_mapping(path: str | None) -> dict[int, str]:
    if path is None:
        return dict(DEFAULT_CLASS_MAPPING)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("--class-map-json must contain a JSON object")
    mapping: dict[int, str] = {}
    for key, value in payload.items():
        try:
            class_id = int(key)
        except ValueError as exc:
            raise SystemExit(f"class map key must be an integer: {key}") from exc
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"class map value must be a non-empty string for class id {class_id}")
        mapping[class_id] = value.strip()
    return mapping


def _require_columns(columns: set[str], required: tuple[str, ...], source_name: str) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise SystemExit(f"{source_name} is missing required columns: {', '.join(missing)}")


def _unit_float(value: Any, field_name: str) -> float:
    number = round(float(value), 4)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be within [0, 1], got {number}")
    return number


def _prefixed_image_path(base_filename: str, image_prefix: str) -> str:
    if not image_prefix:
        return base_filename
    return f"{image_prefix.rstrip('/')}/{base_filename}"


def _load_autocrop_by_filename(autocrop_json: Path, category_id: int) -> dict[str, dict[str, float]]:
    pd = _load_pandas()
    crop_data = json.loads(autocrop_json.read_text(encoding="utf-8"))
    if not isinstance(crop_data, dict):
        raise SystemExit("autocrop JSON must be a COCO-style object")
    if "images" not in crop_data or "annotations" not in crop_data:
        raise SystemExit("autocrop JSON must contain images and annotations arrays")

    df_images = pd.DataFrame(crop_data["images"])
    df_annotations = pd.DataFrame(crop_data["annotations"])
    _require_columns(set(df_images.columns), ("id", "file_name", "width", "height"), "autocrop images")
    _require_columns(set(df_annotations.columns), ("image_id", "category_id", "bbox", "score"), "autocrop annotations")

    df_images = df_images.rename(columns={"id": "image_id"})
    df_annotations = df_annotations[df_annotations["category_id"] == category_id].copy()
    if df_annotations.empty:
        raise SystemExit(f"autocrop JSON has no annotations for category_id={category_id}")

    df_annotations = df_annotations.sort_values("score", ascending=False).drop_duplicates("image_id")
    df_crop = pd.merge(df_annotations, df_images, on="image_id", how="inner")
    if df_crop.empty:
        raise SystemExit("autocrop annotations do not match any image records")

    autocrop_by_filename: dict[str, dict[str, float]] = {}
    for row in df_crop.itertuples(index=False):
        bbox = list(row.bbox)
        if len(bbox) != 4:
            raise SystemExit(f"invalid autocrop bbox for {row.file_name}: {bbox}")
        width = float(row.width)
        height = float(row.height)
        if width <= 0 or height <= 0:
            raise SystemExit(f"invalid image size for {row.file_name}: {width}x{height}")
        x1 = max(0.0, min(1.0, bbox[0] / width))
        y1 = max(0.0, min(1.0, bbox[1] / height))
        x2 = max(0.0, min(1.0, (bbox[0] + bbox[2]) / width))
        y2 = max(0.0, min(1.0, (bbox[1] + bbox[3]) / height))
        if not (x1 < x2 and y1 < y2):
            raise SystemExit(f"invalid normalized autocrop bbox for {row.file_name}")
        autocrop_by_filename[os.path.basename(str(row.file_name))] = {
            "x1": round(x1, 4),
            "y1": round(y1, 4),
            "x2": round(x2, 4),
            "y2": round(y2, 4),
            "score": round(float(row.score), 4),
        }
    return autocrop_by_filename


def _load_detector_groups(detector_parquet: Path, min_saliency: float, class_mapping: dict[int, str]):
    pd = _load_pandas()
    df_detector = pd.read_parquet(detector_parquet)
    _require_columns(set(df_detector.columns), DETECTOR_COLUMNS, "detector parquet")
    if "saliency" in df_detector.columns:
        df_detector = df_detector[df_detector["saliency"] >= min_saliency].copy()
    else:
        df_detector = df_detector.copy()
    if df_detector.empty:
        raise SystemExit("detector parquet has no rows after filtering")

    def make_object(row: Any) -> dict[str, Any]:
        class_id = int(row.id_class)
        box_w = _unit_float(row.box_w, "box_w")
        box_h = _unit_float(row.box_h, "box_h")
        if box_w <= 0.0 or box_h <= 0.0:
            raise ValueError("box_w and box_h must be greater than 0")
        return {
            "class_id": class_id,
            "class_name": class_mapping.get(class_id, f"class_{class_id}"),
            "box_cx": _unit_float(row.box_cx, "box_cx"),
            "box_cy": _unit_float(row.box_cy, "box_cy"),
            "box_w": box_w,
            "box_h": box_h,
        }

    df_detector["base_filename"] = df_detector["image"].map(lambda value: os.path.basename(str(value)))
    df_detector["obj_dict"] = [make_object(row) for row in df_detector.itertuples(index=False)]
    return df_detector.groupby("base_filename")["obj_dict"].apply(list).reset_index()


def generate_dataset_jsonl(
    *,
    detector_parquet: Path,
    autocrop_json: Path,
    output: Path,
    split: str,
    class_mapping: dict[int, str],
    autocrop_category_id: int,
    min_saliency: float,
    missing_autocrop: str,
    image_prefix: str,
) -> dict[str, int]:
    autocrop_by_filename = _load_autocrop_by_filename(autocrop_json, autocrop_category_id)
    detector_groups = _load_detector_groups(detector_parquet, min_saliency, class_mapping)

    written = 0
    skipped_missing_autocrop = 0
    missing_examples: list[str] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in detector_groups.itertuples(index=False):
            base_filename = str(row.base_filename)
            autocrop_top1 = autocrop_by_filename.get(base_filename)
            if autocrop_top1 is None:
                if missing_autocrop == "skip":
                    skipped_missing_autocrop += 1
                    continue
                if missing_autocrop == "full_image":
                    autocrop_top1 = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "score": 0.0}
                else:
                    missing_examples.append(base_filename)
                    continue

            record = {
                "sample_id": Path(base_filename).stem,
                "image": _prefixed_image_path(base_filename, image_prefix),
                "split": split,
                "detector_objects": row.obj_dict,
                "autocrop_top1": autocrop_top1,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            written += 1

    if missing_examples:
        examples = ", ".join(missing_examples[:5])
        raise SystemExit(
            f"missing autocrop_top1 for {len(missing_examples)} detector images, examples: {examples}. "
            "Use --missing-autocrop skip or full_image if this is intentional."
        )

    return {
        "detector_images": len(detector_groups),
        "written_rows": written,
        "skipped_missing_autocrop": skipped_missing_autocrop,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create raw teacher-label input JSONL from detector parquet and COCO autocrop output.")
    parser.add_argument("--detector-parquet", required=True, help="Parquet with detector rows and normalized boxes.")
    parser.add_argument("--autocrop-json", required=True, help="COCO-style autocrop JSON containing images and annotations.")
    parser.add_argument("--output", required=True, help="Output raw JSONL path.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--class-map-json", default=None, help="Optional JSON object mapping class ids to names.")
    parser.add_argument("--autocrop-category-id", type=int, default=0)
    parser.add_argument("--min-saliency", type=float, default=0.2)
    parser.add_argument("--missing-autocrop", choices=["error", "skip", "full_image"], default="error")
    parser.add_argument(
        "--image-prefix",
        default="",
        help="Optional prefix written before each image basename, e.g. /data/coco/train2017.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = generate_dataset_jsonl(
        detector_parquet=Path(args.detector_parquet).resolve(),
        autocrop_json=Path(args.autocrop_json).resolve(),
        output=Path(args.output).resolve(),
        split=args.split,
        class_mapping=_read_class_mapping(args.class_map_json),
        autocrop_category_id=args.autocrop_category_id,
        min_saliency=args.min_saliency,
        missing_autocrop=args.missing_autocrop,
        image_prefix=args.image_prefix,
    )
    print(json.dumps({"output": str(Path(args.output).resolve()), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
