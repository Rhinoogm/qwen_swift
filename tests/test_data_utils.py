from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.data_utils import normalize_record, to_grpo_record, to_sft_record


class DataUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.tempdir.name) / "sample.jpg"
        self.image_path.write_bytes(b"fake")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_normalize_record_requires_existing_image(self) -> None:
        with self.assertRaisesRegex(ValueError, "image path does not exist"):
            normalize_record(
                {
                    "image": str(Path(self.tempdir.name) / "missing.jpg"),
                    "description": "A dog running on grass.",
                    "caption": "A brown dog runs across a grassy field.",
                },
                root_dir=None,
                val_ratio=0.01,
            )

    def test_normalize_record_rejects_multisentence_caption(self) -> None:
        with self.assertRaisesRegex(ValueError, "single sentence"):
            normalize_record(
                {
                    "image": str(self.image_path),
                    "description": "A dog running on grass.",
                    "caption": "A brown dog runs across a grassy field. It looks happy.",
                },
                root_dir=None,
                val_ratio=0.01,
            )

    def test_sft_and_grpo_records_follow_expected_schema(self) -> None:
        sample = normalize_record(
            {
                "image": str(self.image_path),
                "description": "A dog running on grass.",
                "caption": "A brown dog runs across a grassy field.",
                "split": "train",
                "quality_score": 0.9,
                "sample_id": "dog-001",
            },
            root_dir=None,
            val_ratio=0.01,
        )
        sft_record = to_sft_record(sample)
        grpo_record = to_grpo_record(sample)

        self.assertEqual(sft_record["sample_id"], "dog-001")
        self.assertEqual(grpo_record["sample_id"], "dog-001")
        self.assertEqual(len(sft_record["messages"]), 2)
        self.assertEqual(len(grpo_record["messages"]), 1)
        self.assertEqual(grpo_record["reference_caption"], "A brown dog runs across a grassy field.")
        self.assertEqual(grpo_record["images"], [str(self.image_path.resolve())])


if __name__ == "__main__":
    unittest.main()
