from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

# When set in the parent process, these should not be replaced by YAML `env:` defaults.
_ENV_SHELL_WINS_GPU: frozenset[str] = frozenset(
    {"CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"}
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _append_arg(cmd: list[str], key: str, value: Any) -> None:
    flag = f"--{key}"
    if value is None:
        return
    if isinstance(value, bool):
        cmd.extend([flag, str(value).lower()])
        return
    if isinstance(value, dict):
        cmd.extend([flag, json.dumps(value)])
        return
    if isinstance(value, (list, tuple)):
        cmd.append(flag)
        cmd.extend(str(item) for item in value)
        return
    cmd.extend([flag, str(value)])


def build_swift_command(subcommand: str, args: dict[str, Any]) -> list[str]:
    cmd = ["swift", subcommand]
    for key, value in args.items():
        _append_arg(cmd, key, value)
    return cmd


def merged_env(
    overrides: dict[str, str] | None = None,
    *,
    force: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build env for subprocess.run: start from os.environ, apply YAML/config overrides.

    Keys in ``_ENV_SHELL_WINS_GPU`` are not overwritten by ``overrides`` when already
    present in ``os.environ``, so e.g. ``CUDA_VISIBLE_DEVICES=1 python scripts/run_sft.py``
    is respected even if ``configs/sft.yaml`` sets a default. Use ``force=`` for values
    that must win (e.g. CLI ``--cuda-visible-devices``).
    """
    env = os.environ.copy()
    if overrides:
        for key, value in overrides.items():
            k = str(key)
            if k in _ENV_SHELL_WINS_GPU and k in os.environ:
                continue
            env[k] = str(value)
    if force:
        env.update({str(k): str(v) for k, v in force.items()})
    return env


def _print_effective_gpu_env(env: dict[str, str] | None) -> None:
    src: dict[str, str] = dict(os.environ) if env is None else env
    parts = [f"{name}={src[name]!r}" if name in src else f"{name}=(unset)" for name in sorted(_ENV_SHELL_WINS_GPU)]
    print("[qwen-finetune] subprocess GPU env:", " ".join(parts))


def run_command(cmd: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> int:
    rendered = shlex.join(cmd)
    print(rendered)
    _print_effective_gpu_env(env)
    if dry_run:
        return 0
    completed = subprocess.run(cmd, env=env, check=False)
    return completed.returncode
