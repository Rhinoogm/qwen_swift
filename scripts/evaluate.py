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
    ReasonFormatRules,
    SemanticSimilarityScorer,
    acceptance_gate,
    average,
    bbox_iou,
    evaluate_prediction,
    mean_abs_coord_error,
    parse_bbox_mapping,
    percent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate crop recommendation predictions and optional SFT baseline deltas.")
    parser.add_argument("--predictions", required=True, help="Candidate prediction JSONL.")
    parser.add_argument("--baseline-predictions", default=None, help="Optional SFT baseline prediction JSONL.")
    parser.add_argument("--output-json", default=None, help="Optional metrics JSON output path.")
    return parser.parse_args()


def load_prediction_metrics(path: Path) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows = read_jsonl(path)
    scorer = SemanticSimilarityScorer()
    reason_rules = ReasonFormatRules()
    evaluations = []
    for row in rows:
        reference_bbox = parse_bbox_mapping(row["reference_bbox"])
        reference_reason = str(row["reference_reason"])
        evaluations.append(
            evaluate_prediction(
                str(row["prediction"]),
                reference_bbox,
                reference_reason,
                semantic_scorer=scorer,
                reason_rules=reason_rules,
            )
        )

    metrics = {
        "sample_count": len(rows),
        "json_parse_rate": percent(item.parse_ok for item in evaluations),
        "valid_bbox_rate": percent(item.valid_bbox for item in evaluations),
        "reason_format_rate": percent(item.reason_format_ok for item in evaluations),
        "mean_iou_to_teacher": average(item.iou for item in evaluations),
        "iou_at_0_5": percent(item.iou >= 0.5 for item in evaluations),
        "coord_mae": average(item.coord_mae for item in evaluations),
        "reason_semantic_similarity": average(item.reason_similarity for item in evaluations),
    }
    return rows, metrics


def load_autocrop_baseline_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    ious = []
    maes = []
    hit_rates = []
    for row in rows:
        reference_bbox = parse_bbox_mapping(row["reference_bbox"])
        autocrop_bbox = parse_bbox_mapping(row["autocrop_top1"])
        iou = bbox_iou(autocrop_bbox, reference_bbox)
        ious.append(iou)
        maes.append(mean_abs_coord_error(autocrop_bbox, reference_bbox))
        hit_rates.append(iou >= 0.5)
    return {
        "sample_count": len(rows),
        "mean_iou_to_teacher": average(ious),
        "iou_at_0_5": percent(hit_rates),
        "coord_mae": average(maes),
    }


def main() -> int:
    args = parse_args()
    candidate_rows, candidate_metrics = load_prediction_metrics(Path(args.predictions).resolve())
    payload: dict[str, object] = {
        "candidate": candidate_metrics,
        "autocrop_baseline": load_autocrop_baseline_metrics(candidate_rows),
    }

    if args.baseline_predictions:
        _, baseline_metrics = load_prediction_metrics(Path(args.baseline_predictions).resolve())
        payload["baseline"] = baseline_metrics
        payload["acceptance_gate"] = acceptance_gate(
            candidate_metrics,
            baseline_metrics,
            payload["autocrop_baseline"],
        )

    rendered = json.dumps(payload, indent=2)
    print(rendered)

    if args.output_json:
        Path(args.output_json).resolve().write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
