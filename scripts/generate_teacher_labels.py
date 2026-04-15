#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.data_utils import normalize_record, read_jsonl, write_jsonl
from qwen_lora.prompting import render_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher pseudo labels for detector-aware crop recommendation data.")
    parser.add_argument(
        "--model",
        default=None,
        help="Teacher model path or name. Required only when rows without teacher_answer need generation.",
    )
    parser.add_argument("--input", required=True, help="Raw annotation JSONL path without teacher_answer.")
    parser.add_argument("--output", required=True, help="Output JSONL path with teacher_answer added.")
    parser.add_argument("--root-dir", default=None, help="Dataset root for resolving relative image paths.")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter path.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation length cap.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature. Use 0.0 for deterministic labels.")
    parser.add_argument("--image-max-token-num", type=int, default=1024, help="IMAGE_MAX_TOKEN_NUM override.")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="Fallback validation ratio when split is missing.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Regenerate teacher_answer even when it already exists.")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid rows and write them to the audit report.")
    parser.add_argument("--audit-json", default=None, help="Optional audit JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    audit_path = Path(args.audit_json).resolve() if args.audit_json else output_path.with_suffix(".audit.json")
    rows = read_jsonl(input_path)
    engine = None
    infer_request_cls = None
    request_config = None
    parse_crop_response = None

    def ensure_generation_runtime():
        nonlocal engine, infer_request_cls, request_config, parse_crop_response
        if engine is not None:
            return engine, infer_request_cls, request_config, parse_crop_response
        if not args.model:
            raise SystemExit("--model is required when generating missing teacher_answer rows")

        os.environ["IMAGE_MAX_TOKEN_NUM"] = str(args.image_max_token_num)

        from peft import PeftModel
        from swift import get_model_processor, get_template
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

        from qwen_lora.reward_core import parse_crop_response as parse_crop_response_impl

        model, processor = get_model_processor(args.model)
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter)
        template = get_template(processor, enable_thinking=False)
        engine = TransformersEngine(model, template=template)
        infer_request_cls = InferRequest
        request_config = RequestConfig(max_tokens=args.max_new_tokens, temperature=args.temperature)
        parse_crop_response = parse_crop_response_impl
        return engine, infer_request_cls, request_config, parse_crop_response

    generated_rows = []
    invalid_rows = []
    generated_count = 0
    reused_count = 0

    for index, row in enumerate(rows, start=1):
        try:
            if row.get("teacher_answer") is not None and not args.overwrite_existing:
                sample = normalize_record(
                    row,
                    root_dir=args.root_dir,
                    val_ratio=args.val_ratio,
                    require_teacher_answer=True,
                )
                if sample.teacher_answer is None:
                    raise ValueError("teacher_answer validation unexpectedly returned None")
                updated = dict(row)
                updated["teacher_answer"] = sample.teacher_answer.as_dict()
                generated_rows.append(updated)
                reused_count += 1
                continue

            sample = normalize_record(
                row,
                root_dir=args.root_dir,
                val_ratio=args.val_ratio,
                require_teacher_answer=False,
            )
            engine_obj, infer_request_cls_obj, request_config_obj, parse_crop_response_obj = ensure_generation_runtime()
            infer_request = infer_request_cls_obj(
                messages=[{"role": "user", "content": render_prompt(sample.detector_objects, sample.autocrop_top1)}],
                images=[sample.image_path],
            )
            response = engine_obj.infer([infer_request], request_config=request_config_obj)[0]
            teacher_answer = parse_crop_response_obj(response.choices[0].message.content.strip(), validate_reason=True)
            updated = dict(row)
            updated["teacher_answer"] = teacher_answer.as_dict()
            generated_rows.append(updated)
            generated_count += 1
        except ValueError as exc:
            error = {"line": index, "error": str(exc), "record": row}
            if not args.skip_invalid:
                raise SystemExit(f"Invalid record at line {index}: {exc}")
            invalid_rows.append(error)

    write_jsonl(output_path, generated_rows)
    audit = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "counts": {
            "raw_rows": len(rows),
            "written_rows": len(generated_rows),
            "generated_teacher_rows": generated_count,
            "reused_teacher_rows": reused_count,
            "invalid_rows": len(invalid_rows),
        },
        "settings": {
            "root_dir": args.root_dir,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "overwrite_existing": args.overwrite_existing,
        },
        "invalid_rows": invalid_rows,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(audit["counts"], indent=2))
    print(f"Wrote: {output_path}")
    print(f"Wrote: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
