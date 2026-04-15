#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.data_utils import normalize_record, read_jsonl, to_grpo_record, to_sft_record, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw annotations and prepare SFT/GRPO JSONL files.")
    parser.add_argument("--input", required=True, help="Raw annotation JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Directory for processed JSONL outputs.")
    parser.add_argument("--root-dir", default=None, help="Dataset root for resolving relative image paths.")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="Fallback validation ratio when split is missing.")
    parser.add_argument("--rl-max-samples", type=int, default=5000, help="Maximum GRPO subset size.")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid rows and write them into the audit report instead of failing immediately.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = read_jsonl(input_path)

    normalized = []
    invalid = []
    for index, row in enumerate(rows, start=1):
        try:
            normalized.append(normalize_record(row, root_dir=args.root_dir, val_ratio=args.val_ratio))
        except ValueError as exc:
            error = {"line": index, "error": str(exc), "record": row}
            if not args.skip_invalid:
                raise SystemExit(f"Invalid record at line {index}: {exc}")
            invalid.append(error)

    train_samples = [sample for sample in normalized if sample.split == "train"]
    val_samples = [sample for sample in normalized if sample.split in {"val", "test"}]

    rl_samples = sorted(train_samples, key=lambda item: item.quality_score, reverse=True)[: args.rl_max_samples]

    sft_train_path = output_dir / "sft_train.jsonl"
    sft_val_path = output_dir / "sft_val.jsonl"
    grpo_train_path = output_dir / "grpo_train.jsonl"
    audit_path = output_dir / "audit.json"

    write_jsonl(sft_train_path, (to_sft_record(sample) for sample in train_samples))
    write_jsonl(sft_val_path, (to_sft_record(sample) for sample in val_samples))
    write_jsonl(grpo_train_path, (to_grpo_record(sample) for sample in rl_samples))

    audit = {
        "input_path": str(input_path),
        "counts": {
            "raw_rows": len(rows),
            "valid_rows": len(normalized),
            "invalid_rows": len(invalid),
            "sft_train_rows": len(train_samples),
            "sft_val_rows": len(val_samples),
            "grpo_train_rows": len(rl_samples),
        },
        "settings": {
            "val_ratio": args.val_ratio,
            "rl_max_samples": args.rl_max_samples,
            "root_dir": args.root_dir,
        },
        "invalid_rows": invalid,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")

    print(json.dumps(audit["counts"], indent=2))
    print(f"Wrote: {sft_train_path}")
    print(f"Wrote: {sft_val_path}")
    print(f"Wrote: {grpo_train_path}")
    print(f"Wrote: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
