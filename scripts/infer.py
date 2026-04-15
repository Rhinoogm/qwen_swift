#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.prompting import render_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image caption inference.")
    parser.add_argument("--model", required=True, help="Merged model path or base model name.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--description", required=True, help="User description text.")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter path.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation length cap.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--image-max-token-num", type=int, default=1024, help="IMAGE_MAX_TOKEN_NUM override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise SystemExit(f"Image path does not exist: {image_path}")

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
        messages=[{"role": "user", "content": render_prompt(args.description)}],
        images=[str(image_path)],
    )
    request_config = RequestConfig(max_tokens=args.max_new_tokens, temperature=args.temperature)
    response = engine.infer([infer_request], request_config=request_config)[0]
    print(response.choices[0].message.content.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
