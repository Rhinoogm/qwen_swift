#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.swift_cli import build_swift_command, load_yaml, merged_env, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ms-swift SFT with the fixed Qwen3.5-0.8B config.")
    parser.add_argument("--config", default="configs/sft.yaml", help="YAML config path.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(Path(args.config))
    cmd = build_swift_command("sft", config.get("args", {}))
    env = merged_env(config.get("env"))
    return run_command(cmd, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
