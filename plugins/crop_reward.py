from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.reward_core import (
    SemanticSimilarityScorer,
    bbox_iou,
    inspect_prediction_text,
    mean_abs_coord_error,
    parse_bbox_mapping,
)

try:  # pragma: no cover - exercised in training environments where ms-swift is installed
    from swift.rewards import ORM, orms
except Exception:  # pragma: no cover - legacy import paths
    try:
        from swift.plugin import ORM, orms  # type: ignore[no-redef,assignment]
    except Exception:
        try:
            from swift.plugin.orm import ORM, orms  # type: ignore[no-redef,assignment]
        except Exception:  # pragma: no cover - keeps local lint/tests lightweight
            class ORM:  # type: ignore[override]
                def __init__(self, args=None, **kwargs):
                    pass

            orms = {}


def _reference_bboxes(reference_bbox, kwargs: dict) -> Sequence[dict]:
    references = reference_bbox or kwargs.get("reference_bbox")
    if references is None:
        raise ValueError("reference_bbox column is required for crop reward functions")
    return references


def _reference_reasons(reference_reason, kwargs: dict) -> Sequence[str]:
    references = reference_reason or kwargs.get("reference_reason")
    if references is None:
        raise ValueError("reference_reason column is required for crop reward functions")
    return references


class CropIoUReward(ORM):
    def __call__(self, completions, reference_bbox=None, **kwargs):
        rewards = []
        for completion, bbox_payload in zip(completions, _reference_bboxes(reference_bbox, kwargs)):
            inspection = inspect_prediction_text(completion)
            if not inspection.valid_bbox or inspection.recommendation is None:
                rewards.append(0.0)
                continue
            rewards.append(bbox_iou(inspection.recommendation.best_crop, parse_bbox_mapping(bbox_payload)))
        return rewards


class CropCoordReward(ORM):
    def __call__(self, completions, reference_bbox=None, **kwargs):
        rewards = []
        for completion, bbox_payload in zip(completions, _reference_bboxes(reference_bbox, kwargs)):
            inspection = inspect_prediction_text(completion)
            if not inspection.valid_bbox or inspection.recommendation is None:
                rewards.append(0.0)
                continue
            mae = mean_abs_coord_error(inspection.recommendation.best_crop, parse_bbox_mapping(bbox_payload))
            rewards.append(max(0.0, 1.0 - mae))
        return rewards


class CropReasonSemanticReward(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.scorer = SemanticSimilarityScorer()

    def __call__(self, completions, reference_reason=None, **kwargs):
        rewards = []
        for completion, reason in zip(completions, _reference_reasons(reference_reason, kwargs)):
            inspection = inspect_prediction_text(completion)
            if not inspection.valid_bbox or inspection.recommendation is None:
                rewards.append(0.0)
                continue
            rewards.append(self.scorer.score(inspection.recommendation.reason, reason))
        return rewards


class CropGuidanceReward(ORM):
    def __call__(self, completions, reference_guidance_id=None, **kwargs):
        references = reference_guidance_id or kwargs.get("reference_guidance_id")
        rewards = []
        for completion, ref_gid in zip(completions, references):
            inspection = inspect_prediction_text(completion)
            if not inspection.valid_bbox or inspection.recommendation is None:
                rewards.append(0.0)
                continue
            predicted_gid = inspection.recommendation.guidance_id
            rewards.append(1.0 if (predicted_gid is not None and predicted_gid == ref_gid) else 0.0)
        return rewards


orms["crop_iou"] = CropIoUReward
orms["crop_coord"] = CropCoordReward
orms["crop_reason_semantic"] = CropReasonSemanticReward
orms["crop_guidance"] = CropGuidanceReward
