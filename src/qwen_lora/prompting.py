from __future__ import annotations

import json
from typing import Any, Sequence

from .reward_core import MAX_REASON_WORDS

PROMPT_TEMPLATE = """<image>You are given detector metadata and an auto-crop proposal for this image.
Infer the best final crop for the image based on the actual image content, detector metadata, and auto-crop proposal.
Do not copy coordinates or wording from any schema example.

Return JSON only with keys "reason" and "best_crop".

Output structure:
{{
  "reason": "<one grounded English sentence describing the rationale first>",
  "best_crop": {{
    "x1": <number>,
    "y1": <number>,
    "x2": <number>,
    "y2": <number>
  }}
}}

Rules:
- reason must be generated first to explain the logical crop choice.
- best_crop must be an object, not a list.
- best_crop must contain normalized x1, y1, x2, y2 values in [0, 1].
- best_crop must satisfy x1 < x2 and y1 < y2.
- Choose the coordinates from the image content and metadata, not from the output structure above.
- reason must describe the crop for this specific image, not a generic template.
- reason must be one grounded English sentence with at most {max_reason_words} words.
- Do not output markdown, code fences, or any extra keys.

detector_objects:
{detector_objects_json}

autocrop_top1:
{autocrop_top1_json}"""

def _render_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True, sort_keys=False)

def render_prompt(detector_objects: Sequence[dict[str, Any]], autocrop_top1: dict[str, Any]) -> str:
    if detector_objects is None:
        raise ValueError("detector_objects must be provided")
    if autocrop_top1 is None:
        raise ValueError("autocrop_top1 must be provided")
    return PROMPT_TEMPLATE.format(
        max_reason_words=MAX_REASON_WORDS,
        detector_objects_json=_render_json(list(detector_objects)),
        autocrop_top1_json=_render_json(autocrop_top1),
    )




