from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BUNDLED_DEMO_DATASET_DIR = ROOT / "examples" / "demo_dataset"
BUNDLED_DEMO_ANNOTATIONS = BUNDLED_DEMO_DATASET_DIR / "annotations.jsonl"


@dataclass(frozen=True)
class DemoSample:
    filename: str
    split: str
    sample_id: str
    detector_objects: list[dict[str, Any]]
    autocrop_top1: dict[str, Any]
    teacher_answer: dict[str, Any]


def _load_demo_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with BUNDLED_DEMO_ANNOTATIONS.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


@lru_cache(maxsize=1)
def _cached_demo_records() -> tuple[dict[str, object], ...]:
    return tuple(_load_demo_records())


def demo_records() -> list[dict[str, object]]:
    return [json.loads(json.dumps(record, ensure_ascii=True)) for record in _cached_demo_records()]


def _to_demo_sample(record: dict[str, object]) -> DemoSample:
    return DemoSample(
        filename=Path(str(record["image"])).name,
        split=str(record["split"]),
        sample_id=str(record["sample_id"]),
        detector_objects=list(record["detector_objects"]),
        autocrop_top1=dict(record["autocrop_top1"]),
        teacher_answer=dict(record["teacher_answer"]),
    )


DEMO_SAMPLES = tuple(_to_demo_sample(record) for record in demo_records())


def _copy_file(source: Path, destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    if destination.exists() and not overwrite:
        return
    shutil.copy2(source, destination)


def write_demo_dataset(output_dir: Path, *, overwrite: bool = True) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    images_dir = output_dir / "images"
    annotations_path = output_dir / "annotations.jsonl"

    _copy_file(BUNDLED_DEMO_ANNOTATIONS, annotations_path, overwrite=overwrite)
    source_images_dir = BUNDLED_DEMO_DATASET_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for source_image in sorted(source_images_dir.iterdir()):
        if source_image.is_file():
            _copy_file(source_image, images_dir / source_image.name, overwrite=overwrite)

    return {
        "dataset_dir": output_dir,
        "annotations": annotations_path,
        "images_dir": images_dir,
    }
