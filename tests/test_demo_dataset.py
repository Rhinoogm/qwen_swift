from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.demo_dataset import DEMO_SAMPLES, demo_records, write_demo_dataset


class DemoDatasetTest(unittest.TestCase):
    def test_demo_records_match_sample_count(self) -> None:
        self.assertEqual(len(demo_records()), len(DEMO_SAMPLES))

    def test_write_demo_dataset_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            paths = write_demo_dataset(Path(tempdir))
            self.assertTrue(paths["annotations"].exists())
            self.assertTrue(paths["images_dir"].exists())
            for sample in DEMO_SAMPLES:
                self.assertTrue((paths["images_dir"] / sample.filename).exists())
            rows = [json.loads(line) for line in paths["annotations"].read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), len(DEMO_SAMPLES))
            self.assertIn("teacher_answer", rows[0])
            self.assertIn("detector_objects", rows[0])


if __name__ == "__main__":
    unittest.main()
