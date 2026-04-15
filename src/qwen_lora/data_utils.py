from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .prompting import render_prompt
from .reward_core import (
    CropRecommendation,
    bbox_iou,
    parse_bbox_mapping,
    coerce_crop_recommendation,
    format_crop_recommendation,
    word_count,
)

IMAGE_KEYS = ("image", "image_path")


@dataclass(frozen=True)
class NormalizedSample:
    sample_id: str
    image_path: str
    detector_objects: list[dict[str, Any]]
    autocrop_top1: dict[str, Any]
    split: str
    teacher_answer: CropRecommendation | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def _get_first(record: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def resolve_image_path(image_value: str, root_dir: str | None = None) -> str:
    path = Path(image_value)
    if root_dir and not path.is_absolute():
        path = Path(root_dir) / path
    return str(path.expanduser().resolve())


def infer_split(identifier: str, val_ratio: float) -> str:
    digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def _as_unit_float(value: Any, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be within [0, 1]")
    return round(number, 4)


def _normalize_detector_object(index: int, value: Mapping[str, Any]) -> dict[str, Any]:
    class_name = value.get("class_name")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError(f"detector_objects[{index}].class_name must be a non-empty string")
    class_id = value.get("class_id")
    try:
        class_id = int(class_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"detector_objects[{index}].class_id must be an integer") from exc

    box_cx = _as_unit_float(value.get("box_cx"), field_name=f"detector_objects[{index}].box_cx")
    box_cy = _as_unit_float(value.get("box_cy"), field_name=f"detector_objects[{index}].box_cy")
    box_w = _as_unit_float(value.get("box_w"), field_name=f"detector_objects[{index}].box_w")
    box_h = _as_unit_float(value.get("box_h"), field_name=f"detector_objects[{index}].box_h")
    if box_w <= 0.0:
        raise ValueError(f"detector_objects[{index}].box_w must be greater than 0")
    if box_h <= 0.0:
        raise ValueError(f"detector_objects[{index}].box_h must be greater than 0")

    return {
        "class_id": class_id,
        "class_name": class_name.strip(),
        "box_cx": box_cx,
        "box_cy": box_cy,
        "box_w": box_w,
        "box_h": box_h,
    }


def normalize_detector_objects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("detector_objects must be a list")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"detector_objects[{index}] must be an object")
        normalized.append(_normalize_detector_object(index, item))
    return normalized


def normalize_autocrop_top1(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("autocrop_top1 must be an object")
    bbox = parse_bbox_mapping(value, round_values=True)
    try:
        score = float(value.get("score"))
    except (TypeError, ValueError) as exc:
        raise ValueError("autocrop_top1.score must be numeric") from exc
    return {
        **bbox.as_dict(),
        "score": round(score, 4),
    }


def normalize_teacher_answer(value: Any) -> CropRecommendation:
    if not isinstance(value, Mapping):
        raise ValueError("teacher_answer must be an object")
    return coerce_crop_recommendation(value, round_values=True, validate_reason=True)


def normalize_record(
    record: dict[str, Any],
    *,
    root_dir: str | None,
    val_ratio: float,
    require_teacher_answer: bool = True,
) -> NormalizedSample:
    image_value = _get_first(record, IMAGE_KEYS)
    if not isinstance(image_value, str) or not image_value.strip():
        raise ValueError("missing image or image_path")

    image_path = resolve_image_path(image_value, root_dir=root_dir)
    if not Path(image_path).exists():
        raise ValueError(f"image path does not exist: {image_path}")

    detector_objects = normalize_detector_objects(record.get("detector_objects"))
    autocrop_top1 = normalize_autocrop_top1(record.get("autocrop_top1"))

    teacher_answer_payload = record.get("teacher_answer")
    teacher_answer = None
    if teacher_answer_payload is None:
        if require_teacher_answer:
            raise ValueError("missing teacher_answer")
    else:
        teacher_answer = normalize_teacher_answer(teacher_answer_payload)

    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        sample_id = Path(image_path).stem

    split = record.get("split")
    if split is None:
        split = infer_split(sample_id, val_ratio=val_ratio)
    split = str(split).strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split: {split}")

    return NormalizedSample(
        sample_id=sample_id,
        image_path=image_path,
        detector_objects=detector_objects,
        autocrop_top1=autocrop_top1,
        split=split,
        teacher_answer=teacher_answer,
    )


def _require_teacher_answer(sample: NormalizedSample) -> CropRecommendation:
    if sample.teacher_answer is None:
        raise ValueError("teacher_answer is required for prepared datasets")
    return sample.teacher_answer


def to_sft_record(sample: NormalizedSample) -> dict[str, Any]:
    teacher_answer = _require_teacher_answer(sample)
    return {
        "sample_id": sample.sample_id,
        "messages": [
            {"role": "user", "content": render_prompt(sample.detector_objects, sample.autocrop_top1)},
            {"role": "assistant", "content": format_crop_recommendation(teacher_answer)},
        ],
        "images": [sample.image_path],
        "reference_bbox": teacher_answer.best_crop.as_dict(),
        "reference_reason": teacher_answer.reason,
        "autocrop_top1": sample.autocrop_top1,
        "detector_objects": sample.detector_objects,
    }


def to_grpo_record(sample: NormalizedSample) -> dict[str, Any]:
    teacher_answer = _require_teacher_answer(sample)
    return {
        "sample_id": sample.sample_id,
        "messages": [
            {"role": "user", "content": render_prompt(sample.detector_objects, sample.autocrop_top1)},
        ],
        "images": [sample.image_path],
        "reference_bbox": teacher_answer.best_crop.as_dict(),
        "reference_reason": teacher_answer.reason,
        "autocrop_top1": sample.autocrop_top1,
        "detector_objects": sample.detector_objects,
    }


def audit_record(sample: NormalizedSample) -> dict[str, Any]:
    teacher_answer = _require_teacher_answer(sample)
    return {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "teacher_valid": True,
        "reason_word_count": word_count(teacher_answer.reason),
        "autocrop_teacher_iou": round(
            bbox_iou(parse_bbox_mapping(sample.autocrop_top1), teacher_answer.best_crop),
            6,
        ),
    }
