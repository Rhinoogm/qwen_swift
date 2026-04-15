from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.prompting import render_prompt


class PromptingTest(unittest.TestCase):
    def test_render_prompt_contains_detector_and_autocrop_json(self) -> None:
        prompt = render_prompt(
            [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "box_cx": 0.5,
                    "box_cy": 0.5,
                    "box_w": 0.4,
                    "box_h": 0.6,
                }
            ],
            {
                "x1": 0.1,
                "y1": 0.1,
                "x2": 0.9,
                "y2": 0.95,
                "score": 0.83,
            },
        )
        self.assertIn("<image>", prompt)
        self.assertIn("Return JSON only", prompt)
        self.assertIn('"class_name": "person"', prompt)
        self.assertIn('"score": 0.83', prompt)


if __name__ == "__main__":
    unittest.main()
