#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.swift_cli import build_swift_command, load_yaml, merged_env, run_command


def _require_jsonl(path_value: object, field_name: str) -> None:
    if not isinstance(path_value, str) or not path_value.endswith(".jsonl"):
        return
    path = Path(path_value)
    if not path.exists():
        raise SystemExit(f"{field_name} does not exist: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ms-swift SFT with the fixed Qwen3.5-0.8B config.")
    parser.add_argument("--config", default="configs/sft.yaml", help="YAML config path.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(Path(args.config))
    if not args.dry_run:
        config_args = config.get("args", {})
        _require_jsonl(config_args.get("dataset"), "dataset")
        _require_jsonl(config_args.get("val_dataset"), "val_dataset")
    cmd = build_swift_command("sft", config.get("args", {}))
    env = merged_env(config.get("env"))
    return run_command(cmd, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
