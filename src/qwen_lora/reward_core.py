from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

MAX_REASON_WORDS = 40

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

THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


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


def is_single_sentence(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned or "\n" in (text or ""):
        return False
    fragments = [fragment.strip() for fragment in re.split(r"[.!?]+", cleaned) if fragment.strip()]
    return len(fragments) <= 1


def average(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def percent(values: Iterable[float]) -> float:
    return average(values) * 100.0


def extract_json_object_text(text: str) -> str:
    """Extract the JSON object from common model output wrappers.

    Some models still emit reasoning tags, fenced JSON, or a short preamble even
    when prompted for JSON-only output. Rewards and evaluation should judge the
    actual answer object instead of failing on harmless wrappers.
    """
    cleaned = (text or "").strip()
    cleaned = THINK_BLOCK_PATTERN.sub("", cleaned).strip()
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1].strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            cleaned = "\n".join(lines).strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        return cleaned[start : end + 1]
    return cleaned


def _as_float(value: Any, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _round_coord(value: float) -> float:
    return round(float(value), 4)


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def rounded(self) -> "BoundingBox":
        return BoundingBox(
            x1=_round_coord(self.x1),
            y1=_round_coord(self.y1),
            x2=_round_coord(self.x2),
            y2=_round_coord(self.y2),
        )

    def as_dict(self) -> dict[str, float]:
        rounded = self.rounded()
        return {
            "x1": rounded.x1,
            "y1": rounded.y1,
            "x2": rounded.x2,
            "y2": rounded.y2,
        }


@dataclass(frozen=True)
class CropRecommendation:
    best_crop: BoundingBox
    reason: str
    guidance_id: int | None = None

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "best_crop": self.best_crop.as_dict(),
            "reason": normalize_text(self.reason),
        }
        if self.guidance_id is not None:
            d["guidance_id"] = self.guidance_id
        return d


@dataclass(frozen=True)
class PredictionInspection:
    parse_ok: bool
    valid_bbox: bool
    reason: str
    recommendation: CropRecommendation | None
    error: str | None = None


@dataclass(frozen=True)
class PredictionMetrics:
    parse_ok: bool
    valid_bbox: bool
    reason_format_ok: bool
    iou: float
    coord_mae: float
    reason_similarity: float
    guidance_match: bool = False


@dataclass(frozen=True)
class ReasonFormatRules:
    max_words: int = MAX_REASON_WORDS

    def validate(self, text: str) -> list[str]:
        errors: list[str] = []
        if not normalize_text(text):
            errors.append("reason is empty")
        if not is_single_sentence(text):
            errors.append("reason must be a single sentence")
        if word_count(text) > self.max_words:
            errors.append(f"reason exceeds {self.max_words} words")
        if contains_markdown(text):
            errors.append("reason contains markdown or list formatting")
        if contains_speculative_language(text):
            errors.append("reason contains speculative wording")
        return errors

    def score(self, text: str) -> float:
        return 1.0 if not self.validate(text) else 0.0

    def batch_score(self, texts: Sequence[str]) -> list[float]:
        return [self.score(text) for text in texts]


def parse_bbox_mapping(value: Mapping[str, Any], *, round_values: bool = False) -> BoundingBox:
    bbox = BoundingBox(
        x1=_as_float(value.get("x1"), field_name="x1"),
        y1=_as_float(value.get("y1"), field_name="y1"),
        x2=_as_float(value.get("x2"), field_name="x2"),
        y2=_as_float(value.get("y2"), field_name="y2"),
    )
    if not (0.0 <= bbox.x1 < bbox.x2 <= 1.0):
        raise ValueError("best_crop must satisfy 0 <= x1 < x2 <= 1")
    if not (0.0 <= bbox.y1 < bbox.y2 <= 1.0):
        raise ValueError("best_crop must satisfy 0 <= y1 < y2 <= 1")
    return bbox.rounded() if round_values else bbox


def parse_crop_response_json(text: str) -> Mapping[str, Any]:
    json_text = extract_json_object_text(text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"prediction is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("prediction must be a JSON object")
    return payload


def coerce_crop_recommendation(
    payload: Mapping[str, Any],
    *,
    round_values: bool = False,
    validate_reason: bool = False,
) -> CropRecommendation:
    best_crop = payload.get("best_crop")
    reason = payload.get("reason")
    if not isinstance(best_crop, Mapping):
        raise ValueError("best_crop must be an object")
    if not isinstance(reason, str):
        raise ValueError("reason must be a string")
    raw_gid = payload.get("guidance_id")
    guidance_id: int | None = None
    if isinstance(raw_gid, (int, float)) and not isinstance(raw_gid, bool):
        int_gid = int(raw_gid)
        if 1 <= int_gid <= 6:
            guidance_id = int_gid
    recommendation = CropRecommendation(
        best_crop=parse_bbox_mapping(best_crop, round_values=round_values),
        reason=normalize_text(reason),
        guidance_id=guidance_id,
    )
    if validate_reason:
        errors = ReasonFormatRules().validate(recommendation.reason)
        if errors:
            raise ValueError("; ".join(errors))
    return recommendation


def parse_crop_response(text: str, *, validate_reason: bool = False) -> CropRecommendation:
    payload = parse_crop_response_json(text)
    return coerce_crop_recommendation(payload, validate_reason=validate_reason)


def format_crop_recommendation(recommendation: CropRecommendation) -> str:
    bbox = recommendation.best_crop.rounded()
    reason = normalize_text(recommendation.reason)
    escaped_reason = json.dumps(reason, ensure_ascii=True)
    lines = [
        "{",
        '  "best_crop": {',
        f'    "x1": {bbox.x1:.4f},',
        f'    "y1": {bbox.y1:.4f},',
        f'    "x2": {bbox.x2:.4f},',
        f'    "y2": {bbox.y2:.4f}',
        "  },",
        f'  "reason": {escaped_reason}',
    ]
    if recommendation.guidance_id is not None:
        lines[-1] += ","
        lines.append(f'  "guidance_id": {recommendation.guidance_id}')
    lines.append("}")
    return "\n".join(lines)


def inspect_prediction_text(text: str) -> PredictionInspection:
    try:
        payload = parse_crop_response_json(text)
    except ValueError as exc:
        return PredictionInspection(parse_ok=False, valid_bbox=False, reason="", recommendation=None, error=str(exc))

    reason = payload.get("reason") if isinstance(payload.get("reason"), str) else ""
    try:
        recommendation = coerce_crop_recommendation(payload)
    except ValueError as exc:
        return PredictionInspection(
            parse_ok=True,
            valid_bbox=False,
            reason=normalize_text(reason),
            recommendation=None,
            error=str(exc),
        )
    return PredictionInspection(
        parse_ok=True,
        valid_bbox=True,
        reason=normalize_text(reason),
        recommendation=recommendation,
        error=None,
    )


def bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    inter_x1 = max(left.x1, right.x1)
    inter_y1 = max(left.y1, right.y1)
    inter_x2 = min(left.x2, right.x2)
    inter_y2 = min(left.y2, right.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def mean_abs_coord_error(left: BoundingBox, right: BoundingBox) -> float:
    return average(
        [
            abs(left.x1 - right.x1),
            abs(left.y1 - right.y1),
            abs(left.x2 - right.x2),
            abs(left.y2 - right.y2),
        ]
    )


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
                "Falling back to lexical semantic scorer because sentence-transformers "
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
            scores: list[float] = []
            for pred_embedding, ref_embedding in zip(pred_embeddings, ref_embeddings):
                cosine = float(pred_embedding @ ref_embedding)
                scores.append(max(0.0, min(1.0, cosine)))
            return scores
        return [self._fallback_score(pred, ref) for pred, ref in zip(predictions, references)]


def evaluate_prediction(
    prediction_text: str,
    reference_bbox: BoundingBox,
    reference_reason: str,
    *,
    reference_guidance_id: int | None = None,
    semantic_scorer: SemanticSimilarityScorer | None = None,
    reason_rules: ReasonFormatRules | None = None,
) -> PredictionMetrics:
    semantic_scorer = semantic_scorer or SemanticSimilarityScorer()
    reason_rules = reason_rules or ReasonFormatRules()
    inspection = inspect_prediction_text(prediction_text)
    if not inspection.parse_ok:
        return PredictionMetrics(
            parse_ok=False,
            valid_bbox=False,
            reason_format_ok=False,
            iou=0.0,
            coord_mae=1.0,
            reason_similarity=0.0,
            guidance_match=False,
        )
    if not inspection.valid_bbox or inspection.recommendation is None:
        return PredictionMetrics(
            parse_ok=True,
            valid_bbox=False,
            reason_format_ok=False,
            iou=0.0,
            coord_mae=1.0,
            reason_similarity=0.0,
            guidance_match=False,
        )

    reason_format_ok = not reason_rules.validate(inspection.recommendation.reason)
    guidance_match = (
        reference_guidance_id is not None
        and inspection.recommendation.guidance_id is not None
        and inspection.recommendation.guidance_id == reference_guidance_id
    )
    return PredictionMetrics(
        parse_ok=True,
        valid_bbox=True,
        reason_format_ok=reason_format_ok,
        iou=bbox_iou(inspection.recommendation.best_crop, reference_bbox),
        coord_mae=mean_abs_coord_error(inspection.recommendation.best_crop, reference_bbox),
        reason_similarity=semantic_scorer.score(inspection.recommendation.reason, reference_reason),
        guidance_match=guidance_match,
    )


def acceptance_gate(
    candidate_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    autocrop_baseline_metrics: Mapping[str, float],
) -> dict[str, object]:
    checks = {
        "json_parse_rate": candidate_metrics["json_parse_rate"] >= 99.0,
        "valid_bbox_rate": candidate_metrics["valid_bbox_rate"] >= 99.0,
        "beats_autocrop_iou": candidate_metrics["mean_iou_to_teacher"]
        >= autocrop_baseline_metrics["mean_iou_to_teacher"] + 0.03,
        "beats_sft_iou": candidate_metrics["mean_iou_to_teacher"] >= baseline_metrics["mean_iou_to_teacher"],
        "reason_similarity_guardrail": candidate_metrics["reason_semantic_similarity"]
        >= baseline_metrics["reason_semantic_similarity"] - 0.02,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }
