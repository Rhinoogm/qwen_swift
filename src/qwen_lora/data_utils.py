from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .prompting import render_prompt
from .reward_core import CaptionFormatRules

IMAGE_KEYS = ("image", "image_path")
DESCRIPTION_KEYS = ("description",)
CAPTION_KEYS = ("caption",)


@dataclass(frozen=True)
class NormalizedSample:
    sample_id: str
    image_path: str
    description: str
    caption: str
    split: str
    quality_score: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def _get_first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def resolve_image_path(image_value: str, root_dir: str | None = None) -> str:
    path = Path(image_value)
    if root_dir and not path.is_absolute():
        path = Path(root_dir) / path
    return str(path.expanduser().resolve())


def infer_split(identifier: str, val_ratio: float) -> str:
    digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def normalize_record(record: dict[str, Any], *, root_dir: str | None, val_ratio: float) -> NormalizedSample:
    image_value = _get_first(record, IMAGE_KEYS)
    description = _get_first(record, DESCRIPTION_KEYS)
    caption = _get_first(record, CAPTION_KEYS)
    if not isinstance(image_value, str) or not image_value.strip():
        raise ValueError("missing image or image_path")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("missing description")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("missing caption")

    image_path = resolve_image_path(image_value, root_dir=root_dir)
    if not Path(image_path).exists():
        raise ValueError(f"image path does not exist: {image_path}")

    format_rules = CaptionFormatRules()
    errors = format_rules.validate(caption)
    if errors:
        raise ValueError("; ".join(errors))

    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        sample_id = Path(image_path).stem

    split = record.get("split")
    if split is None:
        split = infer_split(sample_id, val_ratio=val_ratio)
    split = str(split).strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split: {split}")

    quality_score = record.get("quality_score", 0.0)
    try:
        quality_score = float(quality_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("quality_score must be numeric") from exc

    return NormalizedSample(
        sample_id=sample_id,
        image_path=image_path,
        description=description.strip(),
        caption=caption.strip(),
        split=split,
        quality_score=quality_score,
    )


def to_sft_record(sample: NormalizedSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "messages": [
            {"role": "user", "content": render_prompt(sample.description)},
            {"role": "assistant", "content": sample.caption},
        ],
        "images": [sample.image_path],
    }


def to_grpo_record(sample: NormalizedSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "messages": [
            {"role": "user", "content": render_prompt(sample.description)},
        ],
        "images": [sample.image_path],
        "reference_caption": sample.caption,
        "quality_score": sample.quality_score,
    }
