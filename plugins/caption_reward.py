from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lora.reward_core import CaptionFormatRules, RougeLScorer, SemanticSimilarityScorer

try:  # pragma: no cover - exercised in training environments where ms-swift is installed
    # GRPO resolves reward names from swift.rewards.orms (ms-swift 4.x+).
    from swift.rewards import ORM, orms
except Exception:  # pragma: no cover - legacy import for older ms-swift
    try:
        from swift.plugin.orm import ORM, orms  # type: ignore[no-redef,assignment]
    except Exception:  # pragma: no cover - keeps local lint/tests lightweight
        class ORM:  # type: ignore[override]
            def __init__(self, args=None, **kwargs):
                pass

        orms = {}


def _references_from_kwargs(kwargs: dict) -> Sequence[str]:
    references = kwargs.get("reference_caption")
    if references is None:
        raise ValueError("reference_caption column is required for caption reward functions")
    return references


class CaptionSemanticReward(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.scorer = SemanticSimilarityScorer()

    def __call__(self, completions, reference_caption=None, **kwargs):
        references = reference_caption or _references_from_kwargs(kwargs)
        return self.scorer.batch_score(completions, references)


class CaptionRougeReward(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.scorer = RougeLScorer()

    def __call__(self, completions, reference_caption=None, **kwargs):
        references = reference_caption or _references_from_kwargs(kwargs)
        return self.scorer.batch_score(completions, references)


class CaptionFormatReward(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.rules = CaptionFormatRules()

    def __call__(self, completions, **kwargs):
        return self.rules.batch_score(completions)


orms["caption_semantic"] = CaptionSemanticReward
orms["caption_rouge"] = CaptionRougeReward
orms["caption_format"] = CaptionFormatReward
