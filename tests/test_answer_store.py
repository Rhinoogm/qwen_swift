from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.answer_store import build_review_record, to_final_answer_record
from qwen_lora.reward_core import BoundingBox, SemanticSimilarityScorer, evaluate_prediction, inspect_prediction_text


class AnswerStoreTest(unittest.TestCase):
    def _prepared_record(self) -> dict[str, object]:
        return {
            "sample_id": "street-001",
            "images": ["/tmp/street_001.ppm"],
            "reference_bbox": {
                "x1": 0.16,
                "y1": 0.1,
                "x2": 0.9,
                "y2": 0.95,
            },
            "reference_reason": "The crop keeps the car and pedestrians together as the focal street action while reducing empty border areas.",
            "autocrop_top1": {
                "x1": 0.12,
                "y1": 0.08,
                "x2": 0.92,
                "y2": 0.98,
                "score": 0.76,
            },
            "detector_objects": [
                {
                    "class_id": 2,
                    "class_name": "car",
                    "box_cx": 0.323,
                    "box_cy": 0.711,
                    "box_w": 0.229,
                    "box_h": 0.203,
                }
            ],
        }

    def test_build_review_record_prefills_final_answer_from_valid_prediction(self) -> None:
        prediction = (
            '{"best_crop":{"x1":0.1600,"y1":0.1000,"x2":0.9000,"y2":0.9500},'
            '"reason":"The crop keeps the car and pedestrians together as the focal street action while reducing empty border areas."}'
        )
        inspection = inspect_prediction_text(prediction)
        metrics = evaluate_prediction(
            prediction,
            BoundingBox(x1=0.16, y1=0.1, x2=0.9, y2=0.95),
            str(self._prepared_record()["reference_reason"]),
            semantic_scorer=SemanticSimilarityScorer(force_fallback=True),
        )

        review = build_review_record(
            self._prepared_record(),
            prediction_text=prediction,
            metrics=metrics,
            inspection=inspection,
        )

        self.assertEqual(review["status"], "predicted")
        self.assertEqual(review["image"], "/tmp/street_001.ppm")
        self.assertEqual(review["final_answer"], review["model_answer"])
        self.assertTrue(review["metrics"]["json_parse_ok"])

    def test_build_review_record_marks_invalid_predictions_for_manual_fix(self) -> None:
        prediction = "not-json"
        inspection = inspect_prediction_text(prediction)
        metrics = evaluate_prediction(
            prediction,
            BoundingBox(x1=0.16, y1=0.1, x2=0.9, y2=0.95),
            str(self._prepared_record()["reference_reason"]),
            semantic_scorer=SemanticSimilarityScorer(force_fallback=True),
        )

        review = build_review_record(
            self._prepared_record(),
            prediction_text=prediction,
            metrics=metrics,
            inspection=inspection,
        )

        self.assertEqual(review["status"], "needs_manual_fix")
        self.assertIsNone(review["final_answer"])

    def test_to_final_answer_record_validates_required_fields(self) -> None:
        exported = to_final_answer_record(
            {
                "sample_id": "street-001",
                "image": "/tmp/street_001.ppm",
                "final_answer": {
                    "best_crop": {
                        "x1": 0.16,
                        "y1": 0.1,
                        "x2": 0.9,
                        "y2": 0.95,
                    },
                    "reason": "The crop keeps the car and pedestrians together as the focal street action while reducing empty border areas.",
                },
            }
        )
        self.assertEqual(exported["sample_id"], "street-001")
        self.assertIn("final_answer", exported)

        with self.assertRaisesRegex(ValueError, "final_answer must be an object"):
            to_final_answer_record({"sample_id": "street-001", "image": "/tmp/street_001.ppm", "final_answer": None})


if __name__ == "__main__":
    unittest.main()
