from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.reward_core import (
    BoundingBox,
    CropRecommendation,
    ReasonFormatRules,
    SemanticSimilarityScorer,
    acceptance_gate,
    evaluate_prediction,
    format_crop_recommendation,
    inspect_prediction_text,
    mean_abs_coord_error,
    parse_crop_response,
)


class RewardCoreTest(unittest.TestCase):
    def test_reason_format_rules_reject_markdown_and_long_reason(self) -> None:
        rules = ReasonFormatRules()
        self.assertEqual(rules.score("* bullet"), 0.0)
        self.assertEqual(rules.score(" ".join(["word"] * 41)), 0.0)

    def test_parse_and_format_round_trip(self) -> None:
        recommendation = CropRecommendation(
            best_crop=BoundingBox(x1=0.12, y1=0.1, x2=0.88, y2=0.93),
            reason="The crop keeps the main subject centered while trimming empty space.",
        )
        rendered = format_crop_recommendation(recommendation)
        parsed = parse_crop_response(rendered, validate_reason=True)
        self.assertEqual(parsed.best_crop.as_dict(), recommendation.best_crop.as_dict())
        self.assertEqual(parsed.reason, recommendation.reason)

    def test_inspect_prediction_distinguishes_parse_and_bbox_failures(self) -> None:
        parse_failure = inspect_prediction_text("not-json")
        bbox_failure = inspect_prediction_text('{"best_crop":{"x1":0.9,"y1":0.1,"x2":0.2,"y2":0.8},"reason":"Valid sentence."}')
        self.assertFalse(parse_failure.parse_ok)
        self.assertTrue(bbox_failure.parse_ok)
        self.assertFalse(bbox_failure.valid_bbox)

    def test_evaluate_prediction_prefers_better_bbox(self) -> None:
        scorer = SemanticSimilarityScorer(force_fallback=True)
        reference_bbox = BoundingBox(x1=0.12, y1=0.1, x2=0.88, y2=0.93)
        reference_reason = "The crop keeps the main subject centered while trimming empty space."
        exact = evaluate_prediction(
            format_crop_recommendation(
                CropRecommendation(best_crop=reference_bbox, reason=reference_reason)
            ),
            reference_bbox,
            reference_reason,
            semantic_scorer=scorer,
        )
        shifted = evaluate_prediction(
            '{"best_crop":{"x1":0.2,"y1":0.2,"x2":0.75,"y2":0.8},"reason":"The crop keeps the subject centered."}',
            reference_bbox,
            reference_reason,
            semantic_scorer=scorer,
        )
        invalid = evaluate_prediction("{}", reference_bbox, reference_reason, semantic_scorer=scorer)
        self.assertGreater(exact.iou, shifted.iou)
        self.assertLess(exact.coord_mae, shifted.coord_mae)
        self.assertEqual(invalid.iou, 0.0)
        self.assertEqual(invalid.coord_mae, 1.0)

    def test_mean_abs_coord_error_is_zero_for_exact_match(self) -> None:
        bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9)
        self.assertAlmostEqual(mean_abs_coord_error(bbox, bbox), 0.0)

    def test_acceptance_gate_requires_parse_bbox_and_quality_gains(self) -> None:
        baseline = {
            "json_parse_rate": 99.0,
            "valid_bbox_rate": 99.0,
            "mean_iou_to_teacher": 0.61,
            "reason_semantic_similarity": 0.83,
        }
        autocrop = {
            "mean_iou_to_teacher": 0.55,
        }
        candidate = {
            "json_parse_rate": 100.0,
            "valid_bbox_rate": 100.0,
            "mean_iou_to_teacher": 0.66,
            "reason_semantic_similarity": 0.82,
        }
        result = acceptance_gate(candidate, baseline, autocrop)
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
