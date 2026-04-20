#!/usr/bin/env python3
"""Side-by-side GT vs model crop visualization from JSONL + image root.

Supports:
  - predict_dataset.py output (reference_*, predicted_bbox, predicted_reason, prediction)
  - prepared SFT JSONL (reference_* + messages assistant JSON as prediction if no pred fields)

Example:
  python scripts/visualize_crop_predictions.py \\
    --jsonl outputs/predictions/val.jsonl \\
    --image-root /data/coco/val2017 \\
    --output-dir outputs/viz
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.reward_core import inspect_prediction_text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_image_path(row: dict[str, Any], image_root: Path) -> Path | None:
    raw = None
    if isinstance(row.get("images"), list) and row["images"]:
        raw = row["images"][0]
    if raw is None:
        raw = row.get("image") or row.get("image_path")
    if not raw or not isinstance(raw, str):
        return None
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    cand = image_root / p.name
    if cand.is_file():
        return cand.resolve()
    # relative to root
    rel = image_root / p
    if rel.is_file():
        return rel.resolve()
    return None


def parse_pred_from_row(row: dict[str, Any]) -> tuple[dict[str, float] | None, str]:
    pb = row.get("predicted_bbox")
    pr = row.get("predicted_reason")
    if isinstance(pb, dict) and all(k in pb for k in ("x1", "y1", "x2", "y2")):
        box = {k: float(pb[k]) for k in ("x1", "y1", "x2", "y2")}
        reason = str(pr or "").strip()
        return box, reason

    pred_text = row.get("prediction")
    if isinstance(pred_text, str) and pred_text.strip():
        ins = inspect_prediction_text(pred_text)
        if ins.recommendation:
            return ins.recommendation.best_crop.as_dict(), ins.recommendation.reason
        return None, str(ins.reason or "").strip()

    msgs = row.get("messages")
    if isinstance(msgs, list) and len(msgs) >= 2:
        last = msgs[-1]
        if isinstance(last, dict) and last.get("role") == "assistant":
            content = last.get("content", "")
            if isinstance(content, str) and content.strip():
                try:
                    ins = inspect_prediction_text(content)
                    if ins.recommendation:
                        return ins.recommendation.best_crop.as_dict(), ins.recommendation.reason
                except Exception:
                    pass
    return None, ""


def parse_ref_from_row(row: dict[str, Any]) -> tuple[dict[str, float] | None, str]:
    rb = row.get("reference_bbox")
    if isinstance(rb, dict) and all(k in rb for k in ("x1", "y1", "x2", "y2")):
        box = {k: float(rb[k]) for k in ("x1", "y1", "x2", "y2")}
        reason = str(row.get("reference_reason") or "").strip()
        return box, reason

    ta = row.get("teacher_answer")
    if isinstance(ta, dict) and isinstance(ta.get("best_crop"), dict):
        bc = ta["best_crop"]
        box = {k: float(bc[k]) for k in ("x1", "y1", "x2", "y2")}
        reason = str(ta.get("reason") or "").strip()
        return box, reason

    return None, ""


def draw_norm_box(
    draw: Any,
    box: dict[str, float],
    w: int,
    h: int,
    outline: tuple[int, int, int],
    width: int = 3,
) -> None:
    x1 = max(0, min(w, int(box["x1"] * w)))
    y1 = max(0, min(h, int(box["y1"] * h)))
    x2 = max(0, min(w, int(box["x2"] * w)))
    y2 = max(0, min(h, int(box["y2"] * h)))
    if x2 <= x1 or y2 <= y1:
        return
    draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)


def build_panel(
    img_path: Path,
    box: dict[str, float] | None,
    color: tuple[int, int, int],
    max_side: int,
) -> Any:
    from PIL import Image, ImageDraw

    im = Image.open(img_path).convert("RGB")
    ow, oh = im.size
    scale = min(1.0, max_side / max(ow, oh))
    nw, nh = max(1, int(ow * scale)), max(1, int(oh * scale))
    if (nw, nh) != (ow, oh):
        im = im.resize((nw, nh), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(im)
    if box is not None:
        # bbox in JSON is normalized to original image; scale to displayed size
        bx = {k: box[k] for k in ("x1", "y1", "x2", "y2")}
        draw_norm_box(draw, bx, nw, nh, color)
    return im


def compose_row(
    left_im: Any,
    right_im: Any,
    left_caption: str,
    right_caption: str,
    gap: int,
    text_width_chars: int,
    font: Any,
) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    lw, lh = left_im.size
    rw, rh = right_im.size
    h_img = max(lh, rh)
    # pad narrower image vertically to center
    def pad_vert(im: Any, target_h: int) -> Any:
        w, h = im.size
        if h >= target_h:
            return im
        canvas = Image.new("RGB", (w, target_h), (255, 255, 255))
        y0 = (target_h - h) // 2
        canvas.paste(im, (0, y0))
        return canvas

    left_im = pad_vert(left_im, h_img)
    right_im = pad_vert(right_im, h_img)
    lw, _ = left_im.size
    rw, _ = right_im.size

    tw = (lw + rw + gap) // 2
    wrap_l = textwrap.fill(left_caption or "(no ref reason)", width=text_width_chars) if left_caption else "(no ref)"
    wrap_r = textwrap.fill(right_caption or "(no pred reason)", width=text_width_chars) if right_caption else "(no pred)"

    # measure text height with a dummy draw
    dummy = Image.new("RGB", (tw, 10), (255, 255, 255))
    d = ImageDraw.Draw(dummy)
    bbox_l = d.multiline_textbbox((0, 0), wrap_l, font=font)
    bbox_r = d.multiline_textbbox((0, 0), wrap_r, font=font)
    text_h = max(bbox_l[3] - bbox_l[1], bbox_r[3] - bbox_r[1]) + 20

    total_w = lw + gap + rw
    total_h = h_img + text_h + 16
    out = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    out.paste(left_im, (0, 0))
    out.paste(right_im, (lw + gap, 0))
    d = ImageDraw.Draw(out)
    d.multiline_text((4, h_img + 8), wrap_l, fill=(0, 0, 0), font=font)
    d.multiline_text((lw + gap + 4, h_img + 8), wrap_r, fill=(0, 0, 0), font=font)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize GT vs predicted crop boxes and reasons.")
    p.add_argument("--jsonl", required=True, type=Path, help="SFT or predict_dataset JSONL")
    p.add_argument("--image-root", required=True, type=Path, help="Folder to resolve images by basename")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/crop_viz"))
    p.add_argument("--max-side", type=int, default=512, help="Max image side per panel (resize, fast)")
    p.add_argument("--limit", type=int, default=None, help="Max rows to render")
    p.add_argument("--text-width", type=int, default=52, help="Chars per column for caption wrap")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from PIL import ImageFont
    except ImportError as e:
        raise SystemExit("Pillow is required: pip install pillow") from e

    rows = read_jsonl(args.jsonl.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    gt_color = (34, 139, 34)
    pred_color = (220, 20, 60)

    n_ok = 0
    for i, row in enumerate(rows):
        if args.limit is not None and i >= args.limit:
            break
        sid = row.get("sample_id") or f"line{i+1}"
        img_path = resolve_image_path(row, args.image_root.resolve())
        if img_path is None:
            print(f"skip {sid}: image not found", file=sys.stderr)
            continue

        ref_box, ref_reason = parse_ref_from_row(row)
        pred_box, pred_reason = parse_pred_from_row(row)

        if ref_box is None:
            print(f"skip {sid}: no reference_bbox / teacher_answer", file=sys.stderr)
            continue

        left = build_panel(img_path, ref_box, gt_color, args.max_side)
        # If no prediction, still show right panel as copy without box or with message
        right = build_panel(img_path, pred_box, pred_color, args.max_side)
        if pred_box is None:
            pred_reason = pred_reason or "(could not parse prediction)"

        out = compose_row(
            left,
            right,
            f"GT\n{ref_reason}",
            f"Model\n{pred_reason}",
            gap=8,
            text_width_chars=args.text_width,
            font=font,
        )
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(sid))[:120]
        out_path = args.output_dir / f"{safe}.png"
        out.save(out_path, optimize=True)
        n_ok += 1
        print(out_path)

    print(json.dumps({"written": n_ok, "output_dir": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
