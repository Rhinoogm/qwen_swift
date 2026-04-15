from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.data_utils import audit_record, normalize_record, to_grpo_record, to_sft_record
from qwen_lora.reward_core import parse_crop_response


class DataUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.tempdir.name) / "sample.jpg"
        self.image_path.write_bytes(b"fake")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _base_record(self) -> dict[str, object]:
        return {
            "image": str(self.image_path),
            "sample_id": "sample-001",
            "split": "train",
            "detector_objects": [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "box_cx": 0.5,
                    "box_cy": 0.5,
                    "box_w": 0.4,
                    "box_h": 0.6,
                }
            ],
            "autocrop_top1": {
                "x1": 0.1,
                "y1": 0.1,
                "x2": 0.9,
                "y2": 0.95,
                "score": 0.83,
            },
            "teacher_answer": {
                "best_crop": {
                    "x1": 0.12,
                    "y1": 0.1,
                    "x2": 0.88,
                    "y2": 0.93,
                },
                "reason": "The crop keeps the main person fully visible while removing empty border regions.",
            },
        }

    def test_normalize_record_requires_existing_image(self) -> None:
        record = self._base_record()
        record["image"] = str(Path(self.tempdir.name) / "missing.jpg")
        with self.assertRaisesRegex(ValueError, "image path does not exist"):
            normalize_record(record, root_dir=None, val_ratio=0.01)

    def test_normalize_record_rejects_invalid_detector_box(self) -> None:
        record = self._base_record()
        record["detector_objects"][0]["box_cx"] = 1.2
        with self.assertRaisesRegex(ValueError, "box_cx must be within"):
            normalize_record(record, root_dir=None, val_ratio=0.01)

    def test_normalize_record_rejects_invalid_teacher_reason(self) -> None:
        record = self._base_record()
        record["teacher_answer"]["reason"] = "One sentence. Another sentence."
        with self.assertRaisesRegex(ValueError, "single sentence"):
            normalize_record(record, root_dir=None, val_ratio=0.01)

    def test_normalize_record_allows_missing_teacher_for_generation(self) -> None:
        record = self._base_record()
        record.pop("teacher_answer")
        sample = normalize_record(record, root_dir=None, val_ratio=0.01, require_teacher_answer=False)
        self.assertIsNone(sample.teacher_answer)

    def test_sft_and_grpo_records_follow_expected_schema(self) -> None:
        sample = normalize_record(self._base_record(), root_dir=None, val_ratio=0.01)
        sft_record = to_sft_record(sample)
        grpo_record = to_grpo_record(sample)
        parsed_answer = parse_crop_response(sft_record["messages"][-1]["content"], validate_reason=True)

        self.assertEqual(sft_record["sample_id"], "sample-001")
        self.assertEqual(grpo_record["sample_id"], "sample-001")
        self.assertEqual(len(sft_record["messages"]), 2)
        self.assertEqual(len(grpo_record["messages"]), 1)
        self.assertIn("Return JSON only", sft_record["messages"][0]["content"])
        self.assertEqual(parsed_answer.best_crop.as_dict(), grpo_record["reference_bbox"])
        self.assertEqual(grpo_record["reference_reason"], parsed_answer.reason)
        self.assertEqual(grpo_record["images"], [str(self.image_path.resolve())])

    def test_audit_record_contains_teacher_stats(self) -> None:
        sample = normalize_record(self._base_record(), root_dir=None, val_ratio=0.01)
        audit = audit_record(sample)
        self.assertTrue(audit["teacher_valid"])
        self.assertGreater(audit["reason_word_count"], 0)
        self.assertGreaterEqual(audit["autocrop_teacher_iou"], 0.0)
        self.assertLessEqual(audit["autocrop_teacher_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
