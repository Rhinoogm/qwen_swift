#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.demo_dataset import write_demo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy the bundled tiny demo detector-aware crop dataset to another directory.")
    parser.add_argument(
        "--output-dir",
        default="examples/generated_demo_dataset",
        help="Directory where demo images and annotations will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = write_demo_dataset(Path(args.output_dir))
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    print("This demo dataset is for pipeline smoke tests only, not for meaningful model quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
