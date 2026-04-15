#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.data_utils import read_jsonl, write_jsonl
from qwen_lora.reward_core import CaptionFormatRules, word_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch prediction on a prepared image-caption JSONL dataset.")
    parser.add_argument("--model", required=True, help="Merged model path or base model name.")
    parser.add_argument("--input", required=True, help="Prepared JSONL input path.")
    parser.add_argument("--output", required=True, help="Prediction JSONL output path.")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter path.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation length cap.")
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
    format_rules = CaptionFormatRules()
    outputs = []

    for record in records:
        infer_request = InferRequest(messages=record["messages"], images=record["images"])
        response = engine.infer([infer_request], request_config=request_config)[0]
        prediction = response.choices[0].message.content.strip()
        outputs.append(
            {
                "sample_id": record.get("sample_id"),
                "prediction": prediction,
                "reference_caption": record.get("reference_caption") or record["messages"][-1]["content"],
                "format_pass": bool(format_rules.score(prediction)),
                "word_count": word_count(prediction),
            }
        )

    write_jsonl(Path(args.output).resolve(), outputs)
    print(json.dumps({"predictions_written": len(outputs), "output": str(Path(args.output).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
