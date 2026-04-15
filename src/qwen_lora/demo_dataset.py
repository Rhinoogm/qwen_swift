from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DemoSample:
    filename: str
    description: str
    caption: str
    split: str
    quality_score: float
    sample_id: str


DEMO_SAMPLES = (
    DemoSample(
        filename="dog_001.ppm",
        description="A dog running on grass.",
        caption="A brown dog runs across a grassy field.",
        split="train",
        quality_score=0.98,
        sample_id="dog-001",
    ),
    DemoSample(
        filename="kitchen_001.ppm",
        description="Two people cooking in a kitchen.",
        caption="Two people prepare food together in a kitchen.",
        split="train",
        quality_score=0.95,
        sample_id="kitchen-001",
    ),
    DemoSample(
        filename="street_001.ppm",
        description="A busy city street at night.",
        caption="Cars and pedestrians move along a brightly lit city street at night.",
        split="val",
        quality_score=0.97,
        sample_id="street-001",
    ),
)


def _blank_canvas(width: int, height: int, color: tuple[int, int, int]) -> list[list[tuple[int, int, int]]]:
    return [[color for _ in range(width)] for _ in range(height)]


def _fill_rect(
    canvas: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    height = len(canvas)
    width = len(canvas[0])
    for y in range(max(0, y0), min(height, y1)):
        row = canvas[y]
        for x in range(max(0, x0), min(width, x1)):
            row[x] = color


def _fill_circle(
    canvas: list[list[tuple[int, int, int]]],
    cx: int,
    cy: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    r2 = radius * radius
    height = len(canvas)
    width = len(canvas[0])
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                canvas[y][x] = color


def _write_ppm(path: Path, canvas: list[list[tuple[int, int, int]]]) -> None:
    height = len(canvas)
    width = len(canvas[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write(f"P3\n{width} {height}\n255\n")
        for row in canvas:
            handle.write(" ".join(f"{r} {g} {b}" for r, g, b in row))
            handle.write("\n")


def _build_dog_scene(width: int = 96, height: int = 64) -> list[list[tuple[int, int, int]]]:
    canvas = _blank_canvas(width, height, (150, 210, 255))
    _fill_rect(canvas, 0, 42, width, height, (90, 180, 90))
    _fill_circle(canvas, 55, 38, 9, (140, 92, 56))
    _fill_circle(canvas, 67, 35, 6, (140, 92, 56))
    _fill_rect(canvas, 44, 45, 47, 60, (110, 72, 42))
    _fill_rect(canvas, 56, 45, 59, 60, (110, 72, 42))
    _fill_rect(canvas, 64, 43, 67, 57, (110, 72, 42))
    _fill_rect(canvas, 73, 42, 76, 56, (110, 72, 42))
    _fill_rect(canvas, 39, 33, 43, 38, (110, 72, 42))
    return canvas


def _build_kitchen_scene(width: int = 96, height: int = 64) -> list[list[tuple[int, int, int]]]:
    canvas = _blank_canvas(width, height, (245, 235, 220))
    _fill_rect(canvas, 0, 48, width, height, (190, 170, 145))
    _fill_rect(canvas, 18, 28, 78, 50, (160, 110, 70))
    _fill_rect(canvas, 35, 18, 61, 28, (100, 100, 110))
    _fill_circle(canvas, 28, 23, 6, (225, 190, 160))
    _fill_rect(canvas, 23, 29, 33, 48, (70, 120, 190))
    _fill_circle(canvas, 70, 23, 6, (225, 190, 160))
    _fill_rect(canvas, 65, 29, 75, 48, (190, 90, 90))
    _fill_circle(canvas, 48, 24, 4, (230, 230, 230))
    return canvas


def _build_street_scene(width: int = 96, height: int = 64) -> list[list[tuple[int, int, int]]]:
    canvas = _blank_canvas(width, height, (30, 40, 90))
    _fill_rect(canvas, 0, 40, width, height, (55, 55, 65))
    for x in range(0, width, 18):
        _fill_rect(canvas, x + 6, 48, x + 12, 51, (250, 240, 120))
    _fill_rect(canvas, 8, 18, 22, 40, (60, 70, 120))
    _fill_rect(canvas, 26, 12, 42, 40, (80, 85, 140))
    _fill_rect(canvas, 48, 8, 66, 40, (65, 75, 130))
    _fill_rect(canvas, 70, 15, 88, 40, (75, 80, 145))
    _fill_rect(canvas, 20, 42, 42, 52, (210, 60, 60))
    _fill_rect(canvas, 24, 39, 38, 43, (210, 60, 60))
    _fill_rect(canvas, 62, 43, 66, 57, (210, 210, 210))
    _fill_rect(canvas, 72, 43, 76, 57, (210, 210, 210))
    return canvas


SCENE_BUILDERS = {
    "dog_001.ppm": _build_dog_scene,
    "kitchen_001.ppm": _build_kitchen_scene,
    "street_001.ppm": _build_street_scene,
}


def write_demo_dataset(output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for sample in DEMO_SAMPLES:
        builder = SCENE_BUILDERS[sample.filename]
        _write_ppm(images_dir / sample.filename, builder())

    annotations_path = output_dir / "annotations.jsonl"
    with annotations_path.open("w", encoding="utf-8") as handle:
        for sample in DEMO_SAMPLES:
            record = {
                "image": f"images/{sample.filename}",
                "description": sample.description,
                "caption": sample.caption,
                "split": sample.split,
                "quality_score": sample.quality_score,
                "sample_id": sample.sample_id,
            }
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")

    return {
        "dataset_dir": output_dir,
        "annotations": annotations_path,
        "images_dir": images_dir,
    }


def demo_records() -> list[dict[str, object]]:
    return [
        {
            "image": f"images/{sample.filename}",
            "description": sample.description,
            "caption": sample.caption,
            "split": sample.split,
            "quality_score": sample.quality_score,
            "sample_id": sample.sample_id,
        }
        for sample in DEMO_SAMPLES
    ]
