#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.prompting import render_prompt, render_student_prompt
from qwen_lora.reward_core import format_crop_recommendation, parse_crop_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image crop recommendation inference.")
    parser.add_argument("--model", required=True, help="Merged model path or base model name.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--detector-objects-json", default=None, help="Inline JSON or a file path for detector_objects (teacher mode only).")
    parser.add_argument("--autocrop-top1-json", default=None, help="Inline JSON or a file path for autocrop_top1 (teacher mode only).")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter path.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation length cap.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--image-max-token-num", type=int, default=1024, help="IMAGE_MAX_TOKEN_NUM override.")
    return parser.parse_args()


def _load_json_argument(raw: str) -> object:
    path = Path(raw)
    payload = path.read_text(encoding="utf-8") if path.exists() else raw
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON argument: {exc}") from exc


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise SystemExit(f"Image path does not exist: {image_path}")

    teacher_mode = args.detector_objects_json is not None or args.autocrop_top1_json is not None
    if teacher_mode:
        if args.detector_objects_json is None or args.autocrop_top1_json is None:
            raise SystemExit("Both --detector-objects-json and --autocrop-top1-json must be provided together.")
        detector_objects = _load_json_argument(args.detector_objects_json)
        autocrop_top1 = _load_json_argument(args.autocrop_top1_json)
        if not isinstance(detector_objects, list):
            raise SystemExit("detector_objects must decode to a JSON list")
        if not isinstance(autocrop_top1, dict):
            raise SystemExit("autocrop_top1 must decode to a JSON object")
        prompt = render_prompt(detector_objects, autocrop_top1)
    else:
        prompt = render_student_prompt()

    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(args.image_max_token_num)

    from peft import PeftModel
    from swift import get_model_processor, get_template
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

    model, processor = get_model_processor(args.model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    template = get_template(processor, enable_thinking=False)
    engine = TransformersEngine(model, template=template)

    infer_request = InferRequest(
        messages=[{"role": "user", "content": prompt}],
        images=[str(image_path)],
    )
    request_config = RequestConfig(max_tokens=args.max_new_tokens, temperature=args.temperature)
    response = engine.infer([infer_request], request_config=request_config)[0]
    parsed = parse_crop_response(response.choices[0].message.content.strip())
    print(format_crop_recommendation(parsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
