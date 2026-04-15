#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.demo_dataset import write_demo_dataset

RAW_TEMPLATE = ROOT / "examples" / "raw_annotations.example.jsonl"
WORKSPACE_DIRS = (
    "data/raw",
    "data/processed",
    "outputs/predictions",
    "outputs/reviews",
    "outputs/final_answers",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a predictable local workspace layout for datasets and outputs.")
    parser.add_argument(
        "--workspace-dir",
        default=".",
        help="Root directory where data/ and outputs/ will be created.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the copied template annotation file and demo dataset when they already exist.",
    )
    return parser.parse_args()


def _copy_file(source: Path, destination: Path, *, overwrite: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return "skipped"
    shutil.copy2(source, destination)
    return "copied"


def main() -> int:
    args = parse_args()
    workspace_dir = Path(args.workspace_dir).resolve()

    created_dirs = []
    for relative_path in WORKSPACE_DIRS:
        path = workspace_dir / relative_path
        path.mkdir(parents=True, exist_ok=True)
        created_dirs.append(str(path))

    template_path = workspace_dir / "data/raw/annotations.template.jsonl"
    template_status = _copy_file(RAW_TEMPLATE, template_path, overwrite=args.force)
    demo_paths = write_demo_dataset(workspace_dir / "data/demo_dataset", overwrite=args.force)

    summary = {
        "workspace_dir": str(workspace_dir),
        "created_dirs": created_dirs,
        "annotations_template": {
            "path": str(template_path),
            "status": template_status,
        },
        "demo_dataset": {key: str(value) for key, value in demo_paths.items()},
        "notes": [
            "Edit data/raw/annotations.template.jsonl to build your own dataset.",
            "Use data/demo_dataset/annotations.jsonl with --root-dir data/demo_dataset for a known-good local demo.",
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
