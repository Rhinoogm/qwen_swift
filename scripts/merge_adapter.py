#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.swift_cli import build_swift_command, merged_env, run_command


def find_latest_checkpoint(path: Path) -> Path:
    if path.name.startswith("checkpoint-") and path.is_dir():
        return path
    checkpoints = [child for child in path.rglob("checkpoint-*") if child.is_dir()]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under {path}")
    checkpoints.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return checkpoints[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the latest ms-swift LoRA adapter into full weights.")
    parser.add_argument("--adapter-root", required=True, help="Checkpoint directory or parent directory containing checkpoints.")
    parser.add_argument("--output-dir", required=True, help="Merged model output directory.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter_path = find_latest_checkpoint(Path(args.adapter_root).resolve())
    cmd = build_swift_command(
        "export",
        {
            "adapters": str(adapter_path),
            "merge_lora": True,
            "output_dir": str(Path(args.output_dir).resolve()),
        },
    )
    forced_env = {"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices} if args.cuda_visible_devices is not None else None
    return run_command(cmd, env=merged_env(force=forced_env), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
