"""Tests for merge_detector_parquet_autocrop_json (requires pyarrow)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "merge_detector_parquet_autocrop_json.py"


def test_merge_picks_highest_score_autocrop(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "d.parquet"
    pq.write_table(
        __import__("pyarrow").Table.from_pylist(
            [
                {
                    "image": "p/000000553669.jpg",
                    "id_class": 0,
                    "box_cx": 0.5,
                    "box_cy": 0.5,
                    "box_w": 0.2,
                    "box_h": 0.3,
                    "conf": 1.0,
                    "saliency": 0.5,
                }
            ]
        ),
        parquet_path,
    )
    ac = {
        "images": [{"file_name": "000000553669.jpg", "height": 480, "width": 640, "id": 553669}],
        "annotations": [
            {"image_id": 553669, "bbox": [0, 0, 640, 480], "score": 1.0, "id": 1},
            {"image_id": 553669, "bbox": [26, 99, 534, 321], "score": 4.83, "id": 2},
        ],
    }
    json_path = tmp_path / "ac.json"
    json_path.write_text(json.dumps(ac), encoding="utf-8")
    out = tmp_path / "out.jsonl"

    subprocess.run(
        [sys.executable, str(SCRIPT), "--detector-parquet", str(parquet_path), "--auto-crop-json", str(json_path), "--output-jsonl", str(out)],
        cwd=str(ROOT),
        check=True,
    )

    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["autocrop_top1"]["score"] == 4.83
    assert row["detector_objects"][0]["class_name"] == "class_0"
