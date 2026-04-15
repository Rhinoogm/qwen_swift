#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.answer_store import build_review_record
from qwen_lora.data_utils import read_jsonl, write_jsonl
from qwen_lora.reward_core import (
    ReasonFormatRules,
    SemanticSimilarityScorer,
    evaluate_prediction,
    inspect_prediction_text,
    parse_bbox_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch prediction on a prepared crop recommendation JSONL dataset.")
    parser.add_argument("--model", required=True, help="Merged model path or base model name.")
    parser.add_argument("--input", required=True, help="Prepared JSONL input path.")
    parser.add_argument("--output", required=True, help="Prediction JSONL output path.")
    parser.add_argument(
        "--review-output",
        default=None,
        help="Optional JSONL path for a human-editable review sheet with final_answer initialized from the model output.",
    )
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter path.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation length cap.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--image-max-token-num", type=int, default=1024, help="IMAGE_MAX_TOKEN_NUM override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(args.image_max_token_num)

    from peft import PeftModel
    from swift import get_model_processor, get_template
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

    model, processor = get_model_processor(args.model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    template = get_template(processor, enable_thinking=False)
    engine = TransformersEngine(model, template=template)

    request_config = RequestConfig(max_tokens=args.max_new_tokens, temperature=args.temperature)
    records = read_jsonl(Path(args.input).resolve())
    reason_rules = ReasonFormatRules()
    semantic_scorer = SemanticSimilarityScorer()
    outputs = []
    review_rows = []

    for record in records:
        infer_request = InferRequest(messages=record["messages"], images=record["images"])
        response = engine.infer([infer_request], request_config=request_config)[0]
        prediction = response.choices[0].message.content.strip()
        reference_bbox = parse_bbox_mapping(record["reference_bbox"])
        reference_reason = record["reference_reason"]
        metrics = evaluate_prediction(
            prediction,
            reference_bbox,
            reference_reason,
            semantic_scorer=semantic_scorer,
            reason_rules=reason_rules,
        )
        inspection = inspect_prediction_text(prediction)
        outputs.append(
            {
                "sample_id": record.get("sample_id"),
                "image": record["images"][0] if record.get("images") else None,
                "prediction": prediction,
                "reference_answer": {
                    "best_crop": record["reference_bbox"],
                    "reason": reference_reason,
                },
                "reference_bbox": record["reference_bbox"],
                "reference_reason": reference_reason,
                "autocrop_top1": record["autocrop_top1"],
                "json_parse_ok": metrics.parse_ok,
                "valid_bbox": metrics.valid_bbox,
                "reason_format_ok": metrics.reason_format_ok,
                "iou_to_teacher": round(metrics.iou, 6),
                "coord_mae": round(metrics.coord_mae, 6),
                "reason_semantic_similarity": round(metrics.reason_similarity, 6),
                "predicted_bbox": inspection.recommendation.best_crop.as_dict() if inspection.recommendation else None,
                "predicted_reason": inspection.recommendation.reason if inspection.recommendation else inspection.reason,
            }
        )
        if args.review_output:
            review_rows.append(
                build_review_record(
                    record,
                    prediction_text=prediction,
                    metrics=metrics,
                    inspection=inspection,
                )
            )

    write_jsonl(Path(args.output).resolve(), outputs)
    summary = {"predictions_written": len(outputs), "output": str(Path(args.output).resolve())}
    if args.review_output:
        review_output_path = Path(args.review_output).resolve()
        write_jsonl(review_output_path, review_rows)
        summary["review_output"] = str(review_output_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
