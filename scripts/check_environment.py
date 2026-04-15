#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PACKAGES = (
    "torch",
    "swift",
    "peft",
    "transformers",
    "vllm",
    "sentence_transformers",
    "bert_score",
)


def _run_command(cmd: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 127, ""
    output = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode, output


def _package_info(name: str) -> dict[str, str | bool]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"installed": False}
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    return {"installed": True, "version": str(version)}


def build_report(expect_train: bool) -> dict[str, object]:
    report: dict[str, object] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "cwd": str(ROOT),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "paths": {
            "configs/sft.yaml": (ROOT / "configs/sft.yaml").exists(),
            "configs/grpo.yaml": (ROOT / "configs/grpo.yaml").exists(),
            "scripts/prepare_dataset.py": (ROOT / "scripts/prepare_dataset.py").exists(),
        },
    }

    nvidia_smi_path = shutil.which("nvidia-smi")
    nvidia = {"available": bool(nvidia_smi_path), "path": nvidia_smi_path}
    if nvidia_smi_path:
        code, output = _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
        nvidia["query_ok"] = code == 0
        nvidia["query_output"] = output
    report["nvidia_smi"] = nvidia

    package_names = ("numpy", "yaml") + (TRAIN_PACKAGES if expect_train else ())
    packages = {name: _package_info(name) for name in package_names}
    report["packages"] = packages

    if packages.get("torch", {}).get("installed"):
        import torch

        gpu_info = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            gpu_info["device_0"] = torch.cuda.get_device_name(0)
        report["torch"] = gpu_info

    checks = []
    checks.append(
        {
            "name": "python>=3.10",
            "ok": sys.version_info >= (3, 10),
        }
    )
    checks.append(
        {
            "name": "core_files_present",
            "ok": all(report["paths"].values()),
        }
    )
    if expect_train:
        checks.append(
            {
                "name": "train_packages_installed",
                "ok": all(bool(packages[name].get("installed")) for name in TRAIN_PACKAGES),
            }
        )
        torch_info = report.get("torch", {})
        checks.append(
            {
                "name": "cuda_available",
                "ok": bool(torch_info.get("cuda_available")),
            }
        )
    report["checks"] = checks
    report["all_checks_passed"] = all(item["ok"] for item in checks)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the current machine is ready for local dev or GPU training.")
    parser.add_argument(
        "--expect-train",
        action="store_true",
        help="Require GPU and train-time packages such as torch, ms-swift, and vllm.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(expect_train=args.expect_train)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))
        if report["all_checks_passed"]:
            print("Environment check passed.")
        else:
            print("Environment check found issues.")
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
