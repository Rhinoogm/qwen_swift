#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import re
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used when optional teacher extras are missing
    def tqdm(iterable, **kwargs):
        return iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.data_utils import normalize_record, read_jsonl, write_jsonl
from qwen_lora.prompting import render_prompt
from qwen_lora.reward_core import extract_json_object_text, parse_crop_response

TEACHER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "best_crop": {
            "type": "object",
            "properties": {
                "x1": {"type": "number"},
                "y1": {"type": "number"},
                "x2": {"type": "number"},
                "y2": {"type": "number"},
            },
            "required": ["x1", "y1", "x2", "y2"],
            "additionalProperties": False,
        },
        "reason": {"type": "string"},
    },
    "required": ["best_crop", "reason"],
    "additionalProperties": False,
}


def normalize_teacher_payload(payload: dict[str, Any]) -> dict[str, Any]:
    best_crop = payload.get("best_crop")
    if isinstance(best_crop, list):
        if len(best_crop) != 4:
            raise ValueError("best_crop list must have 4 elements")
        payload = dict(payload)
        payload["best_crop"] = {
            "x1": float(best_crop[0]),
            "y1": float(best_crop[1]),
            "x2": float(best_crop[2]),
            "y2": float(best_crop[3]),
        }
    return payload


def fill_missing_autocrop(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("autocrop_top1") is not None:
        return row
    updated = dict(row)
    updated["autocrop_top1"] = {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 1.0,
        "y2": 1.0,
        "score": 0.0,
    }
    return updated


def clip01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def clip_autocrop_top1(row: dict[str, Any]) -> dict[str, Any]:
    autocrop = row.get("autocrop_top1")
    if not isinstance(autocrop, dict):
        return row

    updated = dict(row)
    updated_autocrop = dict(autocrop)
    for key in ("x1", "y1", "x2", "y2"):
        updated_autocrop[key] = clip01(updated_autocrop[key])

    if not (
        updated_autocrop["x1"] < updated_autocrop["x2"]
        and updated_autocrop["y1"] < updated_autocrop["y2"]
    ):
        raise ValueError("autocrop_top1 becomes invalid after clipping")

    updated["autocrop_top1"] = updated_autocrop
    return updated


def sanitize_prompt_for_vllm(prompt: str) -> str:
    prompt = (prompt or "").strip()
    prompt = re.sub(r"^\s*<image>\s*", "", prompt)
    return prompt + "\n\nReturn only a JSON object that exactly matches the required schema."


def image_to_data_url(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_image_url(image_path: str, image_url_mode: str) -> str:
    if image_url_mode == "local_path":
        return f"file://{Path(image_path).resolve()}"
    return image_to_data_url(image_path)


def extract_message_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(message_content).strip()


def extract_raw_response_bundle(completion: Any) -> dict[str, Any]:
    message = completion.choices[0].message
    content = extract_message_text(getattr(message, "content", ""))
    reasoning = getattr(message, "reasoning", None) or ""
    bundle = {
        "content": content,
        "reasoning": str(reasoning).strip(),
        "message_repr": str(message),
    }
    try:
        bundle["completion_dump"] = completion.model_dump(mode="json")
    except Exception:
        pass
    return bundle


def parse_response_to_teacher_answer(raw_text: str):
    payload = json.loads(extract_json_object_text(raw_text))
    payload = normalize_teacher_payload(payload)
    return parse_crop_response(json.dumps(payload, ensure_ascii=True), validate_reason=True)


def parse_response_bundle_to_teacher_answer(raw_bundle: dict[str, Any]):
    parse_errors = []
    for candidate in (raw_bundle.get("content", ""), raw_bundle.get("reasoning", "")):
        if not candidate:
            continue
        try:
            return parse_response_to_teacher_answer(str(candidate))
        except Exception as exc:
            parse_errors.append(str(exc))

    raise ValueError(
        "failed to parse teacher answer from both content and reasoning: "
        + " | ".join(parse_errors[:2] if parse_errors else ["empty response"])
    )


def create_completion_with_schema(
    client: Any,
    served_model_name: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_new_tokens: int,
):
    try:
        return client.chat.completions.create(
            model=served_model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "teacher_answer",
                    "schema": TEACHER_JSON_SCHEMA,
                    "strict": True,
                },
            },
        )
    except Exception as first_exc:
        try:
            return client.chat.completions.create(
                model=served_model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                extra_body={"guided_json": TEACHER_JSON_SCHEMA},
            )
        except Exception as second_exc:
            raise RuntimeError(
                f"structured output request failed; response_format error={first_exc}; guided_json error={second_exc}"
            ) from second_exc


def request_one_sample(
    client: Any,
    served_model_name: str,
    image_path: str,
    prompt_text: str,
    image_url_mode: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": build_image_url(image_path, image_url_mode)}},
            ],
        }
    ]
    completion = create_completion_with_schema(
        client=client,
        served_model_name=served_model_name,
        messages=messages,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return extract_raw_response_bundle(completion)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast teacher-label generation via a vLLM OpenAI-compatible server.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--audit-json", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only the first N input rows. Useful for quick partial teacher-label generation.",
    )

    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--served-model-name", required=True)

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--image-url-mode", choices=["data_url", "local_path"], default="data_url")
    parser.add_argument(
        "--fill-missing-autocrop",
        action="store_true",
        help="Use a full-image autocrop when autocrop_top1 is missing. Default is to fail fast.",
    )
    parser.add_argument(
        "--clip-autocrop",
        action="store_true",
        help="Clip autocrop_top1 coordinates into [0, 1]. Default is strict validation.",
    )
    parser.add_argument("--save-raw-response", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    audit_path = Path(args.audit_json).resolve() if args.audit_json else output_path.with_suffix(".audit.json")

    rows = read_jsonl(input_path)
    raw_row_count = len(rows)
    if args.max_samples is not None:
        if args.max_samples < 0:
            raise SystemExit("--max-samples must be >= 0")
        rows = rows[: args.max_samples]
    output_rows: list[dict[str, Any] | None] = [None] * len(rows)
    invalid_rows: list[dict[str, Any]] = []
    generated_count = 0
    reused_count = 0

    tasks: list[tuple[int, dict[str, Any], Any, str]] = []

    for index, row in enumerate(rows, start=1):
        try:
            if args.fill_missing_autocrop:
                row = fill_missing_autocrop(row)
            if args.clip_autocrop:
                row = clip_autocrop_top1(row)

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
                output_rows[index - 1] = updated
                reused_count += 1
                continue

            sample = normalize_record(
                row,
                root_dir=args.root_dir,
                val_ratio=args.val_ratio,
                require_teacher_answer=False,
            )
            prompt_text = sanitize_prompt_for_vllm(render_prompt(sample.detector_objects, sample.autocrop_top1))
            tasks.append((index, row, sample, prompt_text))
        except Exception as exc:
            error = {"line": index, "error": str(exc), "record": row}
            if not args.skip_invalid:
                raise SystemExit(f"Invalid record at line {index}: {exc}")
            invalid_rows.append(error)

    if tasks:
        from openai import OpenAI

        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.request_timeout)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_meta: dict[concurrent.futures.Future, tuple[int, dict[str, Any]]] = {}
            for index, row, sample, prompt_text in tasks:
                future = executor.submit(
                    request_one_sample,
                    client,
                    args.served_model_name,
                    sample.image_path,
                    prompt_text,
                    args.image_url_mode,
                    args.max_new_tokens,
                    args.temperature,
                )
                future_to_meta[future] = (index, row)

            for future in tqdm(
                concurrent.futures.as_completed(future_to_meta),
                total=len(future_to_meta),
                desc="Generating teacher labels via vLLM",
            ):
                index, row = future_to_meta[future]
                raw_bundle = None
                try:
                    raw_bundle = future.result()
                    teacher_answer = parse_response_bundle_to_teacher_answer(raw_bundle)
                    updated = dict(row)
                    updated["teacher_answer"] = teacher_answer.as_dict()
                    if args.save_raw_response:
                        updated["_teacher_raw_response"] = raw_bundle
                    output_rows[index - 1] = updated
                    generated_count += 1
                except Exception as exc:
                    error = {"line": index, "error": str(exc), "record": row}
                    if raw_bundle is not None and args.save_raw_response:
                        error["raw_response"] = raw_bundle
                    if not args.skip_invalid:
                        raise SystemExit(f"Invalid record at line {index}: {exc}")
                    invalid_rows.append(error)

    generated_rows = [row for row in output_rows if row is not None]
    write_jsonl(output_path, generated_rows)

    audit = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "counts": {
            "raw_rows": raw_row_count,
            "processed_rows": len(rows),
            "written_rows": len(generated_rows),
            "generated_teacher_rows": generated_count,
            "reused_teacher_rows": reused_count,
            "invalid_rows": len(invalid_rows),
        },
        "settings": {
            "root_dir": args.root_dir,
            "base_url": args.base_url,
            "served_model_name": args.served_model_name,
            "max_samples": args.max_samples,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "overwrite_existing": args.overwrite_existing,
            "max_workers": args.max_workers,
            "image_url_mode": args.image_url_mode,
            "fill_missing_autocrop": args.fill_missing_autocrop,
            "clip_autocrop": args.clip_autocrop,
            "save_raw_response": args.save_raw_response,
        },
        "invalid_rows": invalid_rows,
    }

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(audit["counts"], indent=2))
    print(f"Wrote: {output_path}")
    print(f"Wrote: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
