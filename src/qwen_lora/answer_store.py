from __future__ import annotations

from typing import Any, Mapping

from .reward_core import (
    PredictionInspection,
    PredictionMetrics,
    coerce_crop_recommendation,
    parse_bbox_mapping,
)


def reference_answer_from_prepared(record: Mapping[str, Any]) -> dict[str, object]:
    return {
        "best_crop": parse_bbox_mapping(record["reference_bbox"], round_values=True).as_dict(),
        "reason": str(record["reference_reason"]).strip(),
    }


def build_review_record(
    prepared_record: Mapping[str, Any],
    *,
    prediction_text: str,
    metrics: PredictionMetrics,
    inspection: PredictionInspection,
) -> dict[str, object]:
    model_answer = inspection.recommendation.as_dict() if inspection.recommendation is not None else None
    images = prepared_record.get("images")
    image_path = str(images[0]) if isinstance(images, list) and images else None

    return {
        "sample_id": prepared_record.get("sample_id"),
        "image": image_path,
        "status": "predicted" if model_answer is not None else "needs_manual_fix",
        "final_answer": model_answer,
        "model_answer": model_answer,
        "reference_answer": reference_answer_from_prepared(prepared_record),
        "autocrop_top1": prepared_record.get("autocrop_top1"),
        "detector_objects": prepared_record.get("detector_objects"),
        "metrics": {
            "json_parse_ok": metrics.parse_ok,
            "valid_bbox": metrics.valid_bbox,
            "reason_format_ok": metrics.reason_format_ok,
            "iou_to_teacher": round(metrics.iou, 6),
            "coord_mae": round(metrics.coord_mae, 6),
            "reason_semantic_similarity": round(metrics.reason_similarity, 6),
        },
        "prediction_raw": prediction_text,
        "notes": "",
    }


def to_final_answer_record(review_record: Mapping[str, Any]) -> dict[str, object]:
    sample_id = review_record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be a non-empty string")

    image = review_record.get("image")
    if not isinstance(image, str) or not image.strip():
        raise ValueError("image must be a non-empty string")

    final_answer_payload = review_record.get("final_answer")
    if not isinstance(final_answer_payload, Mapping):
        raise ValueError("final_answer must be an object")
    final_answer = coerce_crop_recommendation(final_answer_payload, round_values=True, validate_reason=True)

    return {
        "sample_id": sample_id.strip(),
        "image": image.strip(),
        "final_answer": final_answer.as_dict(),
    }
