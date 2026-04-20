from __future__ import annotations

import json
from typing import Any, Sequence

from .reward_core import MAX_REASON_WORDS

COMPOSITION_GUIDANCE = """\
Composition guidance:
1. Center composition: the main subject is centered in the crop frame.
2. Rule of thirds: the main subject falls near a one-third intersection of the frame.
3. Leading space: leave space in the direction a moving subject is heading.
4. Headroom: for portraits, leave appropriate space above the subject's head.
5. Frame filling: the subject fills most of the crop frame for emphasis.
6. Negative space: use surrounding empty space to balance and highlight the subject."""

TEACHER_PROMPT_TEMPLATE = """\
<image>You are given detector metadata and an auto-crop proposal for this image.
Infer the best final crop from the actual image content, detector metadata, and auto-crop proposal.
Do not copy coordinates or wording from the schema example.

{composition_guidance}

Return JSON only with keys "best_crop", "reason", and "guidance_id".

Output structure:
{{
  "best_crop": {{
    "x1": <number>,
    "y1": <number>,
    "x2": <number>,
    "y2": <number>
  }},
  "reason": "<one grounded English sentence describing the crop rationale>",
  "guidance_id": <integer 1-6 indicating which composition rule was applied>
}}

Rules:
- best_crop must be an object, not a list.
- best_crop must contain normalized x1, y1, x2, y2 values in [0, 1].
- best_crop must satisfy x1 < x2 and y1 < y2.
- Choose coordinates from the image content and metadata, not from the output structure above.
- reason must describe this specific image, not a generic template.
- reason must be one grounded English sentence with at most {max_reason_words} words.
- guidance_id must be an integer from 1 to 6 corresponding to the composition guidance above.
- Do not output markdown, code fences, or any extra keys.

detector_objects:
{detector_objects_json}

autocrop_top1:
{autocrop_top1_json}"""

STUDENT_PROMPT_TEMPLATE = """\
<image>You are a photo-crop assistant. Analyze the image and recommend the best crop.

{composition_guidance}

Return JSON only with keys "best_crop", "reason", and "guidance_id".

Output structure:
{{
  "best_crop": {{
    "x1": <number>,
    "y1": <number>,
    "x2": <number>,
    "y2": <number>
  }},
  "reason": "<one grounded English sentence describing the crop rationale>",
  "guidance_id": <integer 1-6 indicating which composition rule was applied>
}}

Rules:
- best_crop must be an object, not a list.
- best_crop must contain normalized x1, y1, x2, y2 values in [0, 1].
- best_crop must satisfy x1 < x2 and y1 < y2.
- reason must describe this specific image, not a generic template.
- reason must be one grounded English sentence with at most {max_reason_words} words.
- guidance_id must be an integer from 1 to 6 corresponding to the composition guidance above.
- Do not output markdown, code fences, or any extra keys."""

# Keep the old name as an alias for backward compatibility.
PROMPT_TEMPLATE = TEACHER_PROMPT_TEMPLATE


def _render_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True, sort_keys=False)


def render_prompt(detector_objects: Sequence[dict[str, Any]], autocrop_top1: dict[str, Any]) -> str:
    """Render the teacher prompt (includes detector metadata and autocrop proposal)."""
    if detector_objects is None:
        raise ValueError("detector_objects must be provided")
    if autocrop_top1 is None:
        raise ValueError("autocrop_top1 must be provided")
    return TEACHER_PROMPT_TEMPLATE.format(
        composition_guidance=COMPOSITION_GUIDANCE,
        max_reason_words=MAX_REASON_WORDS,
        detector_objects_json=_render_json(list(detector_objects)),
        autocrop_top1_json=_render_json(autocrop_top1),
    )


def render_student_prompt() -> str:
    """Render the student prompt (image only, no detector metadata or autocrop)."""
    return STUDENT_PROMPT_TEMPLATE.format(
        composition_guidance=COMPOSITION_GUIDANCE,
        max_reason_words=MAX_REASON_WORDS,
    )
