#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.data_utils import read_jsonl
from qwen_lora.reward_core import (
    BertScoreScorer,
    CaptionFormatRules,
    RougeLScorer,
    acceptance_gate,
    average_word_count,
    empty_output_rate,
    percent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate caption predictions and optional baseline deltas.")
    parser.add_argument("--predictions", required=True, help="Candidate prediction JSONL.")
    parser.add_argument("--baseline-predictions", default=None, help="Optional baseline prediction JSONL.")
    parser.add_argument("--output-json", default=None, help="Optional metrics JSON output path.")
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, float]:
    rows = read_jsonl(path)
    predictions = [row["prediction"] for row in rows]
    references = [row["reference_caption"] for row in rows]

    rouge_scorer = RougeLScorer()
    bert_scorer = BertScoreScorer()
    format_rules = CaptionFormatRules()

    rouge_scores = rouge_scorer.batch_score(predictions, references)
    bert_scores = bert_scorer.batch_score(predictions, references)
    format_scores = format_rules.batch_score(predictions)

    return {
        "sample_count": len(rows),
        "rouge_l": percent(rouge_scores),
        "bert_score_f1": sum(bert_scores) / len(bert_scores) if bert_scores else 0.0,
        "format_pass_rate": percent(format_scores),
        "avg_word_count": average_word_count(predictions),
        "empty_output_rate": empty_output_rate(predictions),
    }


def main() -> int:
    args = parse_args()
    candidate_metrics = load_metrics(Path(args.predictions).resolve())
    payload: dict[str, object] = {"candidate": candidate_metrics}

    if args.baseline_predictions:
        baseline_metrics = load_metrics(Path(args.baseline_predictions).resolve())
        payload["baseline"] = baseline_metrics
        payload["acceptance_gate"] = acceptance_gate(candidate_metrics, baseline_metrics)

    rendered = json.dumps(payload, indent=2)
    print(rendered)

    if args.output_json:
        Path(args.output_json).resolve().write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
