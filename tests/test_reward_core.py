from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.reward_core import (
    CaptionFormatRules,
    RougeLScorer,
    SemanticSimilarityScorer,
    acceptance_gate,
)


class RewardCoreTest(unittest.TestCase):
    def test_format_rules_reject_markdown_and_long_caption(self) -> None:
        rules = CaptionFormatRules()
        self.assertEqual(rules.score("* bullet"), 0.0)
        self.assertEqual(
            rules.score(" ".join(["word"] * 41)),
            0.0,
        )

    def test_format_rules_accept_single_grounded_sentence(self) -> None:
        rules = CaptionFormatRules()
        self.assertEqual(rules.score("A brown dog runs across a grassy field."), 1.0)

    def test_rouge_l_scores_exact_match_highest(self) -> None:
        scorer = RougeLScorer()
        exact = scorer.score("A brown dog runs across a grassy field.", "A brown dog runs across a grassy field.")
        partial = scorer.score("A dog runs on grass.", "A brown dog runs across a grassy field.")
        self.assertGreater(exact, partial)
        self.assertAlmostEqual(exact, 1.0)

    def test_semantic_fallback_prefers_related_text(self) -> None:
        scorer = SemanticSimilarityScorer(force_fallback=True)
        related = scorer.score("A dog runs on grass.", "A brown dog runs across a grassy field.")
        unrelated = scorer.score("A car is parked in snow.", "A brown dog runs across a grassy field.")
        self.assertGreater(related, unrelated)

    def test_acceptance_gate_requires_quality_and_format(self) -> None:
        baseline = {
            "bert_score_f1": 0.80,
            "rouge_l": 60.0,
            "format_pass_rate": 97.0,
            "empty_output_rate": 1.0,
        }
        candidate = {
            "bert_score_f1": 0.82,
            "rouge_l": 61.0,
            "format_pass_rate": 99.0,
            "empty_output_rate": 0.0,
        }
        result = acceptance_gate(candidate, baseline)
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
