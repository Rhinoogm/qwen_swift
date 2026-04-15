from __future__ import annotations

PROMPT_TEMPLATE = (
    "<image>Given this image and the user description, write one grounded English "
    "caption in at most 40 words. If the description conflicts with the image, "
    "trust the image.\nDescription: {description}"
)

MAX_CAPTION_WORDS = 40


def render_prompt(description: str) -> str:
    description = (description or "").strip()
    if not description:
        raise ValueError("description must be a non-empty string")
    return PROMPT_TEMPLATE.format(description=description)
