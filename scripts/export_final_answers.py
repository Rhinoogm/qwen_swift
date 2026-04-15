#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.answer_store import to_final_answer_record
from qwen_lora.data_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export validated final answers from a human-editable review JSONL.")
    parser.add_argument("--input", required=True, help="Review JSONL path created by predict_dataset.py --review-output.")
    parser.add_argument("--output", required=True, help="Output JSONL path for final answers.")
    parser.add_argument(
        "--audit-json",
        default=None,
        help="Optional audit JSON path. Defaults to <output>.audit.json.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid rows and report them in the audit instead of failing immediately.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    audit_path = Path(args.audit_json).resolve() if args.audit_json else output_path.with_suffix(".audit.json")

    rows = read_jsonl(input_path)
    exported_rows = []
    invalid_rows = []

    for index, row in enumerate(rows, start=1):
        try:
            exported_rows.append(to_final_answer_record(row))
        except ValueError as exc:
            error = {"line": index, "error": str(exc), "record": row}
            if not args.skip_invalid:
                raise SystemExit(f"Invalid review row at line {index}: {exc}")
            invalid_rows.append(error)

    write_jsonl(output_path, exported_rows)
    audit = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "counts": {
            "raw_rows": len(rows),
            "exported_rows": len(exported_rows),
            "invalid_rows": len(invalid_rows),
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
