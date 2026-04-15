from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

from .prompting import MAX_CAPTION_WORDS

MARKDOWN_PATTERNS = (
    re.compile(r"^\s*[-*+]\s+", re.MULTILINE),
    re.compile(r"^\s*\d+\.\s+", re.MULTILINE),
    re.compile(r"`"),
    re.compile(r"\[[^\]]+\]\([^)]+\)"),
    re.compile(r"^\s*#+\s+", re.MULTILINE),
)

SPECULATIVE_PATTERNS = (
    re.compile(r"\bmaybe\b", re.IGNORECASE),
    re.compile(r"\bperhaps\b", re.IGNORECASE),
    re.compile(r"\bprobably\b", re.IGNORECASE),
    re.compile(r"\bpossibly\b", re.IGNORECASE),
    re.compile(r"\bit seems\b", re.IGNORECASE),
    re.compile(r"\bit appears\b", re.IGNORECASE),
)


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalize_text(text).lower())


def word_count(text: str) -> int:
    return len(tokenize(text))


def contains_markdown(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in MARKDOWN_PATTERNS)


def contains_speculative_language(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in SPECULATIVE_PATTERNS)


def is_empty(text: str) -> bool:
    return not normalize_text(text)


def is_single_sentence(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned or "\n" in (text or ""):
        return False
    fragments = [frag.strip() for frag in re.split(r"[.!?]+", cleaned) if frag.strip()]
    return len(fragments) <= 1


@dataclass(frozen=True)
class CaptionFormatRules:
    max_words: int = MAX_CAPTION_WORDS

    def validate(self, text: str) -> list[str]:
        errors: list[str] = []
        if is_empty(text):
            errors.append("caption is empty")
        if not is_single_sentence(text):
            errors.append("caption must be a single sentence")
        if word_count(text) > self.max_words:
            errors.append(f"caption exceeds {self.max_words} words")
        if contains_markdown(text):
            errors.append("caption contains markdown or list formatting")
        if contains_speculative_language(text):
            errors.append("caption contains speculative wording")
        return errors

    def score(self, text: str) -> float:
        return 1.0 if not self.validate(text) else 0.0

    def batch_score(self, texts: Sequence[str]) -> list[float]:
        return [self.score(text) for text in texts]


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    dp = [0] * (len(right) + 1)
    for ltok in left:
        prev = 0
        for idx, rtok in enumerate(right, start=1):
            current = dp[idx]
            if ltok == rtok:
                dp[idx] = prev + 1
            else:
                dp[idx] = max(dp[idx], dp[idx - 1])
            prev = current
    return dp[-1]


class RougeLScorer:
    def score(self, prediction: str, reference: str) -> float:
        pred_tokens = tokenize(prediction)
        ref_tokens = tokenize(reference)
        if not pred_tokens or not ref_tokens:
            return 0.0
        lcs = _lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def batch_score(self, predictions: Sequence[str], references: Sequence[str]) -> list[float]:
        return [self.score(pred, ref) for pred, ref in zip(predictions, references)]


class SemanticSimilarityScorer:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        force_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.force_fallback = force_fallback
        self._model = None
        self._backend = None

    def _load_backend(self) -> None:
        if self._backend is not None:
            return
        if self.force_fallback:
            self._backend = "fallback"
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._backend = "sentence_transformers"
        except Exception as exc:  # pragma: no cover - exercised only with optional deps missing
            warnings.warn(
                f"Falling back to lexical semantic scorer because sentence-transformers "
                f"could not be loaded: {exc}",
                RuntimeWarning,
            )
            self._backend = "fallback"

    def _fallback_score(self, prediction: str, reference: str) -> float:
        pred = set(tokenize(prediction))
        ref = set(tokenize(reference))
        if not pred or not ref:
            return 0.0
        overlap = len(pred & ref)
        precision = overlap / len(pred)
        recall = overlap / len(ref)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def score(self, prediction: str, reference: str) -> float:
        self._load_backend()
        if self._backend == "sentence_transformers":
            embeddings = self._model.encode(
                [normalize_text(prediction), normalize_text(reference)],
                normalize_embeddings=True,
            )
            cosine = float(embeddings[0] @ embeddings[1])
            return max(0.0, min(1.0, cosine))
        return self._fallback_score(prediction, reference)

    def batch_score(self, predictions: Sequence[str], references: Sequence[str]) -> list[float]:
        self._load_backend()
        if self._backend == "sentence_transformers":
            normalized_predictions = [normalize_text(item) for item in predictions]
            normalized_references = [normalize_text(item) for item in references]
            pred_embeddings = self._model.encode(normalized_predictions, normalize_embeddings=True)
            ref_embeddings = self._model.encode(normalized_references, normalize_embeddings=True)
            scores = []
            for pred_embedding, ref_embedding in zip(pred_embeddings, ref_embeddings):
                cosine = float(pred_embedding @ ref_embedding)
                scores.append(max(0.0, min(1.0, cosine)))
            return scores
        return [self._fallback_score(pred, ref) for pred, ref in zip(predictions, references)]


class BertScoreScorer:
    def __init__(self, *, force_fallback: bool = False) -> None:
        self.force_fallback = force_fallback
        self._backend = None
        self._semantic_fallback = SemanticSimilarityScorer(force_fallback=True)

    def _load_backend(self) -> None:
        if self._backend is not None:
            return
        if self.force_fallback:
            self._backend = "fallback"
            return
        try:
            import bert_score  # noqa: F401

            self._backend = "bert_score"
        except Exception as exc:  # pragma: no cover - exercised only with optional deps missing
            warnings.warn(
                f"Falling back to lexical BERTScore proxy because bert-score could not "
                f"be loaded: {exc}",
                RuntimeWarning,
            )
            self._backend = "fallback"

    def batch_score(self, predictions: Sequence[str], references: Sequence[str]) -> list[float]:
        self._load_backend()
        if self._backend == "bert_score":
            from bert_score import score as bert_score

            _, _, f1 = bert_score(
                list(predictions),
                list(references),
                lang="en",
                verbose=False,
                rescale_with_baseline=False,
            )
            return [float(item) for item in f1]
        return self._semantic_fallback.batch_score(predictions, references)


def average(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def percent(values: Iterable[float]) -> float:
    return average(values) * 100.0


def empty_output_rate(texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    empty_count = sum(1 for text in texts if is_empty(text))
    return 100.0 * empty_count / len(texts)


def average_word_count(texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    return average(word_count(text) for text in texts)


def acceptance_gate(candidate_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict[str, object]:
    bert_delta = candidate_metrics["bert_score_f1"] - baseline_metrics["bert_score_f1"]
    rouge_delta = candidate_metrics["rouge_l"] - baseline_metrics["rouge_l"]
    checks = {
        "quality_improved": bert_delta >= 0.01 or rouge_delta >= 2.0,
        "format_pass_rate": candidate_metrics["format_pass_rate"] >= 98.0,
        "empty_output_rate": math.isclose(candidate_metrics["empty_output_rate"], 0.0, abs_tol=1e-9),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": {
            "bert_score_f1": bert_delta,
            "rouge_l": rouge_delta,
        },
    }
