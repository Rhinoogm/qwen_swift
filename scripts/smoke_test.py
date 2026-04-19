#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight end-to-end smoke test for crop dataset prep and CLI plumbing.")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary smoke-test directory instead of deleting it.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    tempdir_obj = tempfile.TemporaryDirectory(prefix="qwen-lora-smoke-")
    tempdir = Path(tempdir_obj.name)
    try:
        workspace_dir = tempdir / "workspace"
        dataset_dir = tempdir / "demo_dataset"
        processed_dir = tempdir / "processed"
        predictions_path = tempdir / "predictions.jsonl"
        review_path = tempdir / "review.jsonl"
        final_answers_path = tempdir / "final_answers.jsonl"

        _run([sys.executable, "scripts/init_workspace.py", "--workspace-dir", str(workspace_dir)], cwd=ROOT)
        _run([sys.executable, "scripts/create_demo_dataset.py", "--output-dir", str(dataset_dir)], cwd=ROOT)
        _run(
            [
                sys.executable,
                "scripts/prepare_dataset.py",
                "--input",
                str(dataset_dir / "annotations.jsonl"),
                "--output-dir",
                str(processed_dir),
                "--root-dir",
                str(dataset_dir),
            ],
            cwd=ROOT,
        )
        _run([sys.executable, "scripts/run_sft.py", "--dry-run"], cwd=ROOT)
        _run([sys.executable, "scripts/run_grpo.py", "--dry-run"], cwd=ROOT)

        prepared_val = processed_dir / "sft_val.jsonl"
        rows = []
        if prepared_val.exists():
            for line in prepared_val.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                rows.append(
                    {
                        "sample_id": record.get("sample_id"),
                        "prediction": record["messages"][-1]["content"],
                        "reference_bbox": record["reference_bbox"],
                        "reference_reason": record["reference_reason"],
                        "autocrop_top1": record["autocrop_top1"],
                    }
                )
        predictions_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        _run([sys.executable, "scripts/evaluate.py", "--predictions", str(predictions_path), "--semantic-fallback"], cwd=ROOT)

        review_rows = []
        if prepared_val.exists():
            for line in prepared_val.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                review_rows.append(
                    {
                        "sample_id": record.get("sample_id"),
                        "image": record["images"][0],
                        "status": "predicted",
                        "final_answer": {
                            "best_crop": record["reference_bbox"],
                            "reason": record["reference_reason"],
                        },
                    }
                )
        review_path.write_text("".join(json.dumps(row) + "\n" for row in review_rows), encoding="utf-8")
        _run(
            [
                sys.executable,
                "scripts/export_final_answers.py",
                "--input",
                str(review_path),
                "--output",
                str(final_answers_path),
            ],
            cwd=ROOT,
        )

        summary = {
            "temp_dir": str(tempdir),
            "workspace_dir": str(workspace_dir),
            "dataset_dir": str(dataset_dir),
            "processed_dir": str(processed_dir),
            "predictions": str(predictions_path),
            "review": str(review_path),
            "final_answers": str(final_answers_path),
        }
        print(json.dumps(summary, indent=2))
        print("Smoke test passed.")
        if args.keep_temp:
            print(f"Temporary files kept at: {tempdir}")
        return 0
    finally:
        if not args.keep_temp:
            tempdir_obj.cleanup()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args))
