#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.data_utils import read_jsonl
from qwen_lora.reward_core import (
    ReasonFormatRules,
    bbox_iou,
    coerce_crop_recommendation,
    mean_abs_coord_error,
    parse_bbox_mapping,
)

FULL_IMAGE_CROP = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit teacher-labeled JSONL files for legacy vLLM generation risks."
    )
    parser.add_argument("--generated", required=True, help="JSONL produced by generate_teacher_labels_vllm_fast.py.")
    parser.add_argument(
        "--original",
        default=None,
        help="Optional pre-teacher raw JSONL. Enables detection of missing autocrop fill and coordinate clipping.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path for the audit report JSON.")
    parser.add_argument("--examples", type=int, default=10, help="Maximum examples to include per issue type.")
    return parser.parse_args()


def _sample_key(row: Mapping[str, Any], index: int) -> str:
    sample_id = row.get("sample_id")
    if isinstance(sample_id, str) and sample_id.strip():
        return sample_id.strip()
    image = row.get("image") or row.get("image_path")
    if isinstance(image, str) and image.strip():
        return image.strip()
    return f"__line_{index}"


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        key = _sample_key(row, index)
        if key in indexed:
            duplicates[key] += 1
            key = f"{key}#__duplicate_{duplicates[key]}"
        indexed[key] = row
    return indexed


def _coords(crop: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return tuple(round(float(crop[key]), 4) for key in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]


def _is_full_image(crop: Any) -> bool:
    if not isinstance(crop, Mapping):
        return False
    try:
        return _coords(crop) == (0.0, 0.0, 1.0, 1.0)
    except Exception:
        return False


def _score_is_zero(crop: Any) -> bool:
    if not isinstance(crop, Mapping):
        return False
    try:
        return round(float(crop.get("score")), 4) == 0.0
    except Exception:
        return False


def _autocrop_error(crop: Any) -> str | None:
    if not isinstance(crop, Mapping):
        return "autocrop_top1 must be an object"
    try:
        parse_bbox_mapping(crop)
    except Exception as exc:
        return str(exc)
    try:
        float(crop.get("score"))
    except Exception:
        return "autocrop_top1.score must be numeric"
    return None


def _teacher_error(teacher_answer: Any) -> str | None:
    if not isinstance(teacher_answer, Mapping):
        return "teacher_answer must be an object"
    try:
        coerce_crop_recommendation(teacher_answer, validate_reason=True)
    except Exception as exc:
        return str(exc)
    return None


def _append_example(examples: dict[str, list[dict[str, Any]]], issue: str, payload: dict[str, Any], limit: int) -> None:
    bucket = examples.setdefault(issue, [])
    if len(bucket) < limit:
        bucket.append(payload)


def audit_generated_rows(
    generated_rows: list[dict[str, Any]],
    original_rows: list[dict[str, Any]] | None,
    *,
    example_limit: int,
) -> dict[str, Any]:
    original_by_key = _index_rows(original_rows) if original_rows is not None else {}
    reason_rules = ReasonFormatRules()
    counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    autocrop_teacher_ious: list[float] = []
    autocrop_teacher_maes: list[float] = []

    generated_keys: set[str] = set()

    for index, row in enumerate(generated_rows, start=1):
        counts["generated_rows"] += 1
        key = _sample_key(row, index)
        generated_keys.add(key)
        split_counts[str(row.get("split", "missing"))] += 1

        autocrop = row.get("autocrop_top1")
        teacher_answer = row.get("teacher_answer")

        autocrop_error = _autocrop_error(autocrop)
        if autocrop_error:
            counts["invalid_autocrop"] += 1
            _append_example(examples, "invalid_autocrop", {"key": key, "error": autocrop_error}, example_limit)
        if _is_full_image(autocrop):
            counts["full_image_autocrop"] += 1
            if _score_is_zero(autocrop):
                counts["full_image_autocrop_score_zero"] += 1
                _append_example(examples, "full_image_autocrop_score_zero", {"key": key}, example_limit)

        teacher_error = _teacher_error(teacher_answer)
        if teacher_error:
            counts["invalid_teacher_answer"] += 1
            _append_example(examples, "invalid_teacher_answer", {"key": key, "error": teacher_error}, example_limit)
        else:
            teacher = coerce_crop_recommendation(teacher_answer, validate_reason=False)
            counts["valid_teacher_answer"] += 1
            if reason_rules.validate(teacher.reason):
                counts["invalid_teacher_reason_format"] += 1
            if autocrop_error is None and isinstance(autocrop, Mapping):
                autocrop_bbox = parse_bbox_mapping(autocrop)
                autocrop_teacher_ious.append(bbox_iou(autocrop_bbox, teacher.best_crop))
                autocrop_teacher_maes.append(mean_abs_coord_error(autocrop_bbox, teacher.best_crop))

        if original_rows is None:
            continue

        original = original_by_key.get(key)
        if original is None:
            counts["missing_original_match"] += 1
            _append_example(examples, "missing_original_match", {"key": key}, example_limit)
            continue

        original_autocrop = original.get("autocrop_top1")
        if original_autocrop is None and isinstance(autocrop, Mapping):
            counts["legacy_filled_missing_autocrop"] += 1
            _append_example(examples, "legacy_filled_missing_autocrop", {"key": key}, example_limit)
            continue

        if isinstance(original_autocrop, Mapping) and isinstance(autocrop, Mapping):
            try:
                original_coords = _coords(original_autocrop)
                generated_coords = _coords(autocrop)
            except Exception:
                continue
            if original_coords != generated_coords:
                counts["autocrop_changed_between_original_and_generated"] += 1
                issue = {
                    "key": key,
                    "original_autocrop": {k: original_autocrop.get(k) for k in ("x1", "y1", "x2", "y2", "score")},
                    "generated_autocrop": {k: autocrop.get(k) for k in ("x1", "y1", "x2", "y2", "score")},
                }
                if any(value < 0.0 or value > 1.0 for value in original_coords):
                    counts["legacy_clipped_autocrop"] += 1
                    _append_example(examples, "legacy_clipped_autocrop", issue, example_limit)
                else:
                    _append_example(examples, "autocrop_changed_between_original_and_generated", issue, example_limit)

    if original_rows is not None:
        counts["original_rows"] = len(original_rows)
        original_keys = set(original_by_key)
        counts["original_rows_without_generated_match"] = len(original_keys - generated_keys)

    mean_iou = sum(autocrop_teacher_ious) / len(autocrop_teacher_ious) if autocrop_teacher_ious else 0.0
    mean_mae = sum(autocrop_teacher_maes) / len(autocrop_teacher_maes) if autocrop_teacher_maes else 0.0

    risk_flags = {
        "teacher_answers_invalid": counts["invalid_teacher_answer"] > 0,
        "autocrop_was_filled_by_legacy_script": counts["legacy_filled_missing_autocrop"] > 0,
        "autocrop_was_clipped_by_legacy_script": counts["legacy_clipped_autocrop"] > 0,
        "many_full_image_autocrops": counts["full_image_autocrop_score_zero"] > max(10, counts["generated_rows"] * 0.01),
        "generated_missing_original_rows": counts["missing_original_match"] > 0,
    }

    return {
        "counts": dict(counts),
        "split_counts": dict(split_counts),
        "metrics": {
            "mean_autocrop_teacher_iou": mean_iou,
            "mean_autocrop_teacher_coord_mae": mean_mae,
        },
        "risk_flags": risk_flags,
        "risk_level": "high" if any(risk_flags.values()) else "low",
        "examples": examples,
    }


def main() -> int:
    args = parse_args()
    generated_path = Path(args.generated).resolve()
    original_path = Path(args.original).resolve() if args.original else None
    report = audit_generated_rows(
        read_jsonl(generated_path),
        read_jsonl(original_path) if original_path else None,
        example_limit=args.examples,
    )
    report["paths"] = {
        "generated": str(generated_path),
        "original": str(original_path) if original_path else None,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 1 if report["risk_level"] == "high" else 0


if __name__ == "__main__":
    raise SystemExit(main())
