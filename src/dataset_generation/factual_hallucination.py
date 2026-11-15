#!/usr/bin/env python3
"""Schemas, evidence helpers, and validators for factual-hallucination injection.

This module is shared by prompt preparation and dataset generation. It keeps the
factual-hallucination pipeline separate from the general and data-judge schemas
so existing output remains unchanged.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

FACTUAL_JUDGE = "factual_hallucination"

FactualErrorCategory = Literal[
    "customer_identity",
    "order_status",
    "order_amount",
    "order_attribute",
    "product_attribute",
    "numerical_attribute",
    "temporal_attribute",
    "status_attribute",
]

FACTUAL_ERROR_CATEGORIES: tuple[str, ...] = (
    "customer_identity",
    "order_status",
    "order_amount",
    "order_attribute",
    "product_attribute",
    "numerical_attribute",
    "temporal_attribute",
    "status_attribute",
)

# Categories that belong to competing judges. They are deliberately rejected by
# the factual-hallucination pipeline rather than silently re-labelled.
EXCLUDED_ERROR_CATEGORY_TARGETS: dict[str, str] = {
    "tool_selection": "action_accuracy",
    "tool_execution": "action_accuracy",
    "incorrect_tool_arguments": "action_accuracy",
    "wrong_tool_arguments": "action_accuracy",
    "tool_execution_outcome": "action_accuracy",
    "procedural_compliance": "procedure_following",
    "missing_verification": "procedure_following",
    "mandatory_verification": "procedure_following",
    "policy_claim": "procedure_following/data_conflict",
    "customer_intent": "intent_understanding",
    "intent_misunderstanding": "intent_understanding",
    "task_closure": "task_closure",
    "unresolved_task": "task_closure",
    "alternative_guidance": "alternative_guidance",
    "missing_alternative_guidance": "alternative_guidance",
    "missing_data": "missing_data",
    "data_conflict": "data_conflict",
}

EvidenceSourceType = Literal[
    "scenario",
    "merchant_kb",
    "merchant_catalog",
    "documented_fact",
]

RejectionCode = Literal[
    "NO_TRUSTED_EVIDENCE",
    "UNSUPPORTED_ONLY",
    "CONFLICTING_EVIDENCE",
    "EXCLUDED_JUDGE_OVERLAP",
    "COHERENCE_IMPOSSIBLE",
    "SCREENING_REJECTED",
    "POST_GENERATION_REJECTED",
    "OTHER",
]

ScreeningDecision = Literal["ELIGIBLE", "REJECT", "REVIEW"]
EvidenceClassification = Literal[
    "CONTRADICTED",
    "UNSUPPORTED",
    "NOT_VERIFIABLE",
    "SUPPORTED",
]
PostGenerationDisposition = Literal["ACCEPT", "REJECT", "REVIEW"]

# Structured reasons why an evidence/claim pair was not offered to the model.
CandidateExclusionCode = Literal[
    "NO_AGENT_TURN",
    "NO_SUPPORTED_EVIDENCE",
    "IDENTIFIER_NOT_FACTUAL_CLAIM",
    "AMBIGUOUS_EVIDENCE_VALUE",
    "VALUE_QUERIED_NOT_STATED",
    "NO_AGENT_CLAIM_STATING_EVIDENCE",
    "ATTRIBUTE_NOT_EXPLICIT",
    "NEGATED_CLAIM",
    "AMBIGUOUS_VALUE_OCCURRENCE",
    "CONDITIONAL_OR_GENERIC_CLAIM",
    "PRE_EXISTING_INCOHERENT_SENTENCE",
    "EXCLUDED_JUDGE_OVERLAP",
]

# Identifiers (order IDs, account IDs, tracking numbers, ...) are structured
# evidence, but they are not suitable factual-claim targets: a typo in a number
# is not the kind of explicit factual contradiction this Judge targets.
NON_CLAIM_ATTRIBUTE_KINDS = frozenset({"identifier"})

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "were",
    "will",
    "with",
    "you",
    "your",
}


class FactualValidationError(ValueError):
    """Raised when a factual-injection payload violates the pipeline contract."""


def validate_factual_error_category(category: str) -> str:
    """Return a valid factual category or explain which judge owns the error."""
    if category in EXCLUDED_ERROR_CATEGORY_TARGETS:
        owner = EXCLUDED_ERROR_CATEGORY_TARGETS[category]
        raise FactualValidationError(
            f"error_category {category!r} is out of scope for "
            f"{FACTUAL_JUDGE}; it belongs to {owner}"
        )
    if category not in FACTUAL_ERROR_CATEGORIES:
        allowed = ", ".join(FACTUAL_ERROR_CATEGORIES)
        raise FactualValidationError(
            f"Unknown factual error_category {category!r}. Allowed: {allowed}"
        )
    return category


class NonEmptyModel(BaseModel):
    """Base model with strict fields and a helper for non-empty strings."""

    model_config = ConfigDict(extra="forbid", strict=True)


class FactualTurn(NonEmptyModel):
    """One turn in the factual-hallucination conversation."""

    turn_id: int = Field(ge=1)
    speaker: Literal["CUSTOMER", "AGENT", "ACTION"]
    content: str


class EvidenceItem(NonEmptyModel):
    """One explicit, trusted evidence item available to the judge."""

    evidence_id: str
    source_type: EvidenceSourceType
    location: str
    text: str

    @field_validator("evidence_id", "location", "text")
    @classmethod
    def _non_blank_exact(cls, value: str) -> str:
        # Keep the original value byte-for-byte. Exact copies are compared with
        # ``==``; normalization is reserved for matching claims to evidence.
        if not value.strip():
            raise ValueError("evidence fields must not be empty or whitespace")
        return value


class NonTargetRiskAssessment(NonEmptyModel):
    """Explicit checks that the injection does not trip a competing judge."""

    action_accuracy: bool
    procedure_following: bool
    intent_understanding: bool
    task_closure: bool
    alternative_guidance: bool
    missing_data: bool
    data_conflict: bool
    notes: str = ""


class FactualInjectionMetadata(NonEmptyModel):
    """Private metadata describing one evidence-backed factual contradiction."""

    target_judge: Literal["factual_hallucination"]
    already_defective: bool
    injection_status: Literal["injected", "pre_existing"]
    injection_turn_id: int = Field(ge=1)
    scope: Literal["pre_existing", "single_turn"]
    error_category: FactualErrorCategory
    error_classification: Literal["CONTRADICTED"]
    original_agent_claim: str
    injected_claim: str
    evidence_source_type: EvidenceSourceType
    evidence_source_location: str
    evidence_text: str
    contradiction_explanation: str
    scenario_modified: bool
    modified_scenario_fields: dict[str, Any]
    non_target_risk_assessment: NonTargetRiskAssessment
    expected_target_verdict: Literal["FAIL"] = "FAIL"
    # Optional provenance. Older prepared records and model outputs omit these,
    # so they default to None and are filled from the screened candidate during
    # validation when available.
    candidate_id: str | None = None
    screening_decision: Literal["ELIGIBLE"] | None = None

    @field_validator(
        "original_agent_claim",
        "injected_claim",
        "evidence_source_location",
        "evidence_text",
        "contradiction_explanation",
    )
    @classmethod
    def _non_blank_exact(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must not be empty or whitespace")
        return value


class FactualModelOutput(NonEmptyModel):
    """The JSON shape the generation model is asked to return for this judge."""

    injection_metadata: FactualInjectionMetadata
    injected_conversation: list[FactualTurn] = Field(min_length=1)


class FactualJudgeInput(NonEmptyModel):
    """Evaluation input exposed to the Judge. Contains no private annotations."""

    target_judge: Literal["factual_hallucination"]
    target_judge_definition: dict[str, Any]
    conversation: list[FactualTurn] = Field(min_length=1)
    scenario: dict[str, Any]
    sop: Any
    evidence: list[EvidenceItem]


class AutomatedChecks(NonEmptyModel):
    """Automated checks performed before a record can enter the golden set.

    These flags describe deterministic checks only. They do not represent human
    review, LLM semantic verification, or Judge execution.
    """

    exactly_one_injected_claim: bool
    evidence_reference_matches_exact: bool
    same_entity_and_attribute: bool
    explicit_incompatible_value: bool
    structured_contradiction_verified: bool
    evidence_location_non_empty: bool
    contradiction_explanation_non_empty: bool
    scenario_unchanged: bool
    turn_id_valid: bool
    speaker_is_agent: bool
    trusted_evidence_unmodified: bool
    trusted_evidence_integrity: Literal["VERIFIED", "NOT_VERIFIED"]
    original_claim_supported: bool
    injected_claim_contradicted: bool
    single_attribute_value_change: bool
    error_category_valid: bool
    output_structure_valid: bool
    minimal_perturbation: bool
    evidence_classification: EvidenceClassification | None = None


class QualityMetrics(NonEmptyModel):
    """Optional aggregate metrics. None means the metric has not been measured."""

    injected_sample_count: int | None = Field(default=None, ge=0)
    pre_existing_sample_count: int | None = Field(default=None, ge=0)
    rejection_count: int | None = Field(default=None, ge=0)
    failed_generation_count: int | None = Field(default=None, ge=0)
    candidate_filter_exclusion_count: int | None = Field(default=None, ge=0)
    screening_eligible_count: int | None = Field(default=None, ge=0)
    screening_reject_count: int | None = Field(default=None, ge=0)
    screening_review_count: int | None = Field(default=None, ge=0)
    screening_not_run_count: int | None = Field(default=None, ge=0)
    reviewed_sample_count: int | None = Field(default=None, ge=0)
    validation_attempt_count: int | None = Field(default=None, ge=0)
    validation_success_count: int | None = Field(default=None, ge=0)
    validation_success_rate: float | None = Field(default=None, ge=0, le=1)
    total_input_records: int | None = Field(default=None, ge=0)
    generation_attempt_count: int | None = Field(default=None, ge=0)
    failure_reasons: dict[str, int] = Field(default_factory=dict)
    injection_success_rate: float | None = Field(default=None, ge=0, le=1)
    pre_existing_rate: float | None = Field(default=None, ge=0, le=1)
    injection_rejection_rate: float | None = Field(default=None, ge=0, le=1)
    overall_acceptance_rate: float | None = Field(default=None, ge=0, le=1)
    rejection_rate: float | None = Field(default=None, ge=0, le=1)
    failure_rate: float | None = Field(default=None, ge=0, le=1)
    generation_acceptance_rate: float | None = Field(default=None, ge=0, le=1)
    evidence_validity_rate: float | None = Field(default=None, ge=0, le=1)
    single_error_compliance_rate: float | None = Field(default=None, ge=0, le=1)
    minimal_perturbation_rate: float | None = Field(default=None, ge=0, le=1)
    target_judge_detection_rate: float | None = Field(default=None, ge=0, le=1)
    false_positive_rate_clean_samples: float | None = Field(default=None, ge=0, le=1)
    non_target_judge_failure_rate: float | None = Field(default=None, ge=0, le=1)
    evidence_localization_accuracy: float | None = Field(default=None, ge=0, le=1)


class EvidenceVerificationSummary(NonEmptyModel):
    """Entity/attribute/context result recorded during validation."""

    classification: EvidenceClassification
    entity_match: bool
    attribute_match: bool
    context_match: bool
    reason: str


class FactualValidationReport(NonEmptyModel):
    """Per-record validation and independent-evaluation status."""

    automated_checks: AutomatedChecks
    evidence_verification: EvidenceVerificationSummary | None = None
    target_judge_status: Literal["NOT_EVALUATED", "PASS", "FAIL"] = "NOT_EVALUATED"
    non_target_judges_status: Literal["NOT_EVALUATED", "PASS", "FAIL"] = (
        "NOT_EVALUATED"
    )
    independent_evaluation_status: Literal[
        "NOT_EVALUATED",
        "PENDING_HUMAN_REVIEW",
        "HUMAN_REVIEWED",
        "EVIDENCE_VALIDATED",
    ] = "PENDING_HUMAN_REVIEW"
    quality_metrics: QualityMetrics = Field(default_factory=QualityMetrics)
    notes: str = (
        "Automated structural and structured-rule checks only. This is not "
        "semantic proof, human review, or Judge execution. Judges were not "
        "executed. The model's contradiction_explanation and risk flags are "
        "not independent proof of correctness. Human review or independent "
        "evidence validation is required before promotion to the final "
        "Golden Set. judge_evaluations on the golden row, when present, are "
        "the expected label for this verified contradiction. They are not an "
        "independent judge execution, and non_target_risk_assessment flags are "
        "not judge results."
    )


class FactualHallucinationGoldenRecordOutput(NonEmptyModel):
    """Successful factual-hallucination golden record."""

    injection_metadata: FactualInjectionMetadata
    judge_input: FactualJudgeInput
    validation_report: FactualValidationReport


class FactualRejectionRecord(NonEmptyModel):
    """A rejection stored separately from successful golden records."""

    conversation_id: str | int
    target_judge: Literal["factual_hallucination"]
    injection_status: Literal["INJECTION_NOT_POSSIBLE"]
    rejection_code: RejectionCode
    rejection_reason: str
    raw_response: str
    sampled_params: dict[str, Any] = Field(default_factory=dict)


class FactualReviewRecord(NonEmptyModel):
    """An ambiguous sample that must not be generated or accepted."""

    conversation_id: str | int
    target_judge: Literal["factual_hallucination"]
    review_status: Literal["REVIEW"]
    stage: Literal["screening", "post_generation"]
    reason: str
    target_claim: str = ""
    evidence_location: str = ""
    evidence_text: str = ""
    sampled_params: dict[str, Any] = Field(default_factory=dict)


class FactualScreeningRecord(NonEmptyModel):
    """Audit row for one pre-generation screening decision."""

    conversation_id: str | int
    target_judge: Literal["factual_hallucination"]
    stage: Literal["screening"] = "screening"
    decision: ScreeningDecision
    reason: str
    target_claim: str = ""
    evidence_location: str = ""
    evidence_text: str = ""
    candidate_id: str | None = None
    requires_llm: bool = False
    llm_decision: str | None = None
    sampled_params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------


def extract_conversation_turns(conversation: Any) -> list[dict[str, Any]]:
    """Extract turns using the existing one-based turn convention.

    Source AB-CD conversations use ``original`` or ``delexed``; generated or
    fixture conversations may use ``turns``. Explicit turn ids are preserved,
    but they must be strictly sequential starting at 1.
    """

    if isinstance(conversation, Mapping):
        turns = conversation.get("turns")
        if turns is None:
            turns = conversation.get("original", conversation.get("delexed"))
    else:
        turns = conversation
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise FactualValidationError("Conversation has no valid turns list")

    result: list[dict[str, Any]] = []
    for position, turn in enumerate(turns, start=1):
        if isinstance(turn, Mapping):
            speaker = turn.get("speaker")
            content = turn.get("text", turn.get("content"))
            turn_id = turn.get("turn_id", turn.get("turn_count", position))
        elif isinstance(turn, Sequence) and len(turn) == 2:
            speaker, content = turn
            turn_id = position
        else:
            raise FactualValidationError(
                f"Invalid turn at position {position}: expected object or pair"
            )
        if not isinstance(speaker, str) or content is None:
            raise FactualValidationError(
                f"Incomplete turn at position {position}: speaker/content missing"
            )
        try:
            normalized_id = int(turn_id)
        except (TypeError, ValueError) as exc:
            raise FactualValidationError(
                f"Turn at position {position} has invalid turn_id {turn_id!r}"
            ) from exc
        result.append(
            {
                "turn_id": normalized_id,
                "speaker": speaker.upper(),
                "content": str(content),
            }
        )

    expected_ids = list(range(1, len(result) + 1))
    actual_ids = [turn["turn_id"] for turn in result]
    if actual_ids != expected_ids:
        raise FactualValidationError(
            "Conversation turn_ids must be strictly sequential starting at 1; "
            f"got {actual_ids!r}"
        )
    return result


def format_numbered_turns(turns: Sequence[Mapping[str, Any]]) -> str:
    """Render turns with explicit one-based ids for the generation prompt."""

    blocks = []
    for turn in turns:
        blocks.append(
            f"[TURN {turn['turn_id']} | {str(turn['speaker']).upper()}]\n"
            f"{turn['content']}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------


def _scalar_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value if value.strip() else None
    return None


def _flatten_mapping(
    value: Any,
    path: str,
    source_type: EvidenceSourceType,
    output: list[dict[str, Any]],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _flatten_mapping(child, f"{path}.{key}", source_type, output)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _flatten_mapping(child, f"{path}[{index}]", source_type, output)
        return
    text = _scalar_text(value)
    if text is None:
        return
    output.append(
        {
            "evidence_id": path,
            "source_type": source_type,
            "location": path,
            "text": text,
        }
    )


def _flatten_scenario_value(
    value: Any,
    path: str,
    output: list[dict[str, Any]],
) -> None:
    # ``order.products`` is stored as a Python-literal string in AB-CD.
    if path == "scenario.order.products" and isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (Mapping, list)):
            _flatten_mapping(parsed, path, "scenario", output)
            return
    _flatten_mapping(value, path, "scenario", output)


def build_available_evidence(
    scenario: Mapping[str, Any],
    extra_evidence: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build trusted evidence items from scenario fields and optional records.

    Neither the conversation nor SOP is treated as trusted factual evidence.
    ``flow`` and ``subflow`` are routing metadata, not factual attributes.
    """

    evidence: list[dict[str, Any]] = []
    if isinstance(scenario, Mapping):
        for key, value in scenario.items():
            if key in {"flow", "subflow"}:
                continue
            _flatten_scenario_value(value, f"scenario.{key}", evidence)

    if extra_evidence:
        for item in extra_evidence:
            evidence.append(dict(item))

    deduplicated: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in evidence:
        key = (
            str(item["source_type"]),
            str(item["location"]),
            str(item["text"]),
        )
        deduplicated.setdefault(
            key,
            {
                "evidence_id": str(item.get("evidence_id", item["location"])),
                "source_type": key[0],
                "location": key[1],
                "text": key[2],
            },
        )
    return sorted(deduplicated.values(), key=lambda item: item["location"])


def is_supported_evidence_item(
    item: Mapping[str, Any], scenario: Mapping[str, Any]
) -> bool:
    """Return True when a conservative structured rule exists for this location."""

    location = str(item.get("location", ""))
    return _attribute_rule(location, scenario) is not None


def filter_supported_evidence(
    evidence: Sequence[Mapping[str, Any]],
    scenario: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Keep only evidence locations the validator can check without world knowledge."""

    return [dict(item) for item in evidence if is_supported_evidence_item(item, scenario)]


# ---------------------------------------------------------------------------
# Text checks
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_value(text: str, value: str) -> bool:
    """Check whether a value appears as a complete token/phrase in text."""

    normalized_text = _normalize(text)
    normalized_value = _normalize(value)
    if not normalized_value:
        return False
    if normalized_value.isdigit():
        pattern = rf"(?<!\d){re.escape(normalized_value)}(?!\d)"
    else:
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_value)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def _subject_tokens(text: str, evidence_text: str) -> set[str]:
    evidence_tokens = set(re.findall(r"[a-z0-9]+", _normalize(evidence_text)))
    tokens = set(re.findall(r"[a-z0-9]+", _normalize(text)))
    return {
        token
        for token in tokens
        if token not in _STOP_WORDS
        and token not in evidence_tokens
        and not token.isdigit()
    }


def _claims_share_subject(original: str, injected: str, evidence_text: str) -> bool:
    original_subject = _subject_tokens(original, evidence_text)
    injected_subject = _subject_tokens(injected, evidence_text)
    return bool(original_subject & injected_subject)


def _check_claim_present(claim: str, turn_content: str, where: str) -> None:
    if claim not in turn_content:
        raise FactualValidationError(
            f"{where}: claim text {claim!r} is not an exact substring of the turn content"
        )


_QUESTION_WORD_RE = re.compile(
    r"^(?:who|what|when|where|why|how|is|are|was|were|do|does|did|can|could|"
    r"would|will|should|may|might|have|has|had)\b",
    re.IGNORECASE,
)


def _containing_sentence(text: str, snippet: str) -> str:
    """Return the sentence that contains ``snippet`` (or the snippet itself)."""

    start = text.find(snippet)
    if start < 0:
        return snippet
    end = start + len(snippet)
    left = max(
        text.rfind(".", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
    )
    sentence_start = left + 1 if left >= 0 else 0
    right_positions = [
        position
        for marker in ".!?"
        if (position := text.find(marker, end)) >= 0
    ]
    sentence_end = min(right_positions) + 1 if right_positions else len(text)
    return text[sentence_start:sentence_end].strip()


def _is_interrogative(text: str) -> bool:
    """Return True when the text is a question rather than a factual claim."""

    stripped = text.strip()
    if not stripped:
        return False
    return "?" in stripped or _QUESTION_WORD_RE.match(stripped) is not None


def _substituted_turn_content(
    source_content: str, original_claim: str, injected_claim: str
) -> str:
    """Return source content with exactly one original-claim occurrence replaced."""

    if original_claim not in source_content:
        raise FactualValidationError(
            "original_agent_claim must appear exactly once as an exact substring "
            "of the original Agent turn"
        )
    if source_content.count(original_claim) != 1:
        raise FactualValidationError(
            "original_agent_claim is ambiguous: it appears more than once in the "
            "original Agent turn, so a single-claim edit cannot be proven"
        )
    return source_content.replace(original_claim, injected_claim, 1)


def _kind_values(text: str) -> dict[str, set[str]]:
    return {
        "status": _status_groups(text),
        "color": _color_values(text),
        "numeric": _numeric_values(text),
        "date": _date_values(text),
        "email": _email_values(text),
        "phone": _phone_values(text),
        "identifier": _identifier_values(text),
        "name": set(_name_candidates(text)),
    }


def _mask_target_values(text: str, target_kind: str) -> str:
    """Remove target spans that other extractors would otherwise double-count.

    Email local parts look like identifiers, and phone numbers look like both
    numbers and identifiers. Masking only the target span keeps unrelated
    identifiers and numbers in the same sentence subject to their normal checks.
    """

    if target_kind == "email":
        return _EMAIL_RE.sub(" ", text)
    if target_kind == "phone":
        return _PHONE_RE.sub(" ", text)
    return text


def _non_target_values_unchanged(
    original: str, injected: str, target_kind: str
) -> None:
    """Reject extra factual value changes outside the target attribute kind."""

    original_values = _kind_values(_mask_target_values(original, target_kind))
    injected_values = _kind_values(_mask_target_values(injected, target_kind))
    extra_kinds: list[str] = []
    for kind, original_set in original_values.items():
        if kind == target_kind:
            continue
        if original_set != injected_values[kind]:
            extra_kinds.append(kind)
    if extra_kinds:
        raise FactualValidationError(
            "The edit changes extra factual attributes besides the target "
            f"{target_kind}: {', '.join(extra_kinds)}. Exactly one factual "
            "contradiction is required"
        )


# ---------------------------------------------------------------------------
# Structured contradiction rules
# ---------------------------------------------------------------------------
#
# The checks below are deliberately conservative. They only accept an explicit
# replacement value from a recognized attribute domain (status, color, number,
# date, identity, email, phone, or identifier) on the same subject and
# attribute. Merely omitting the evidence value, using a paraphrase, or claiming
# that something is "being handled" is not enough to establish a contradiction.

_NEGATION_RE = re.compile(
    r"\b(?:not|never|no longer|isn t|wasn t|weren t|hasn t|haven t|hadn t|"
    r"doesn t|didn t|cannot|can t|won t|wouldn t|shouldn t|couldn t|mustn t)\b"
)

_STATUS_GROUPS: dict[str, tuple[str, ...]] = {
    "pre_shipment": (
        "pending",
        "processing",
        "preparing",
        "being prepared",
        "order received",
        "received",
        "awaiting shipment",
        "unfulfilled",
        "on hold",
        "in progress",
    ),
    "in_transit": (
        "shipped",
        "in transit",
        "out for delivery",
        "dispatched",
        "on the way",
    ),
    # Do not collapse lifecycle states that are not strict synonyms. In
    # particular, an order marked "completed" or "fulfilled" is not
    # necessarily proof that the shipment was delivered; likewise, returned,
    # refunded, and cancelled are distinct states. Treating them as aliases
    # would create false contradiction labels.
    "delivered": ("delivered",),
    "completed": ("completed", "fulfilled"),
    "cancelled": ("cancelled", "canceled", "voided"),
    "refunded": ("refunded",),
    "returned": ("returned",),
}

_COLOR_SYNONYMS: dict[str, str] = {
    "black": "black",
    "blue": "blue",
    "red": "red",
    "white": "white",
    "green": "green",
    "yellow": "yellow",
    "brown": "brown",
    "gray": "gray",
    "grey": "gray",
    "pink": "pink",
    "purple": "purple",
    "orange": "orange",
    "beige": "beige",
    "navy": "navy",
    "teal": "teal",
    "maroon": "maroon",
    "gold": "gold",
    "silver": "silver",
    "tan": "tan",
    "khaki": "khaki",
}

_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_MONTH_PATTERN = "|".join(_MONTH_NUMBERS)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_FIRST_DATE_RE = re.compile(
    rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)
_DAY_FIRST_DATE_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})\s+(\d{{4}})\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\$|usd\s*)?(-?\d[\d,]*(?:\.\d+)?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\+?\(?\d{1,4}\)?[\s.-]+\d{3,4}[\s.-]+\d{3,4}")
_IDENTIFIER_RE = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9][A-Za-z0-9_-]{3,}\b")
_NAME_RE = re.compile(r"\b[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)+\b")
_GENERIC_NAME_TOKENS = {
    "account",
    "best",
    "customer",
    "dear",
    "hello",
    "hi",
    "my",
    "note",
    "order",
    "our",
    "please",
    "record",
    "regards",
    "thank",
    "thanks",
    "the",
    "this",
    "your",
}

_STATUS_LEAVES = {"shipping_status", "status", "order_status", "delivery_status"}
_COLOR_LEAVES = {"color", "colour"}
_NUMERIC_LEAVES = {
    "amount",
    "amounts",
    "cost",
    "costs",
    "count",
    "order_amount",
    "paid",
    "payment",
    "price",
    "prices",
    "quantity",
    "quantities",
    "refund",
    "subtotal",
    "total",
    "totals",
}
_DATE_LEAVES = {
    "created_at",
    "date",
    "delivery_date",
    "order_date",
    "purchase_date",
    "updated_at",
}
_NAME_LEAVES = {"customer_name", "full_name", "name"}
_EMAIL_LEAVES = {"email", "email_address"}
_PHONE_LEAVES = {"phone", "phone_number", "telephone"}
_IDENTIFIER_LEAVES = {
    "customer_id",
    "id",
    "invoice_id",
    "order_id",
    "tracking_number",
    "transaction_id",
}

_NUMERIC_ANCHORS = (
    "amount",
    "charge",
    "cost",
    "order",
    "paid",
    "payment",
    "price",
    "refund",
    "subtotal",
    "total",
)
_DATE_ANCHORS = (
    "arrival",
    "arrive",
    "arrives",
    "date",
    "deliver",
    "delivered",
    "delivery",
    "order",
    "ordered",
    "placed",
    "purchase",
    "purchased",
)
_NAME_ANCHORS = (
    "account",
    "belongs",
    "customer",
    "holder",
    "name",
    "owner",
    "profile",
    "record",
)
_EMAIL_ANCHORS = ("address", "contact", "email", "mail", "reach", "send")
_PHONE_ANCHORS = ("call", "contact", "number", "phone", "telephone")
_IDENTIFIER_ANCHORS = (
    "account",
    "confirmation",
    "id",
    "invoice",
    "number",
    "order",
    "reference",
    "tracking",
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize(text))


def _has_negation(text: str) -> bool:
    """Return True when a claim contains an explicit negation marker.

    Negated claims are rejected rather than interpreted. This avoids accepting
    "not delivered" as an assertion of "delivered".
    """

    return _NEGATION_RE.search(_normalize(text)) is not None


def _status_groups(text: str) -> set[str]:
    return {
        group
        for group, phrases in _STATUS_GROUPS.items()
        if any(contains_value(text, phrase) for phrase in phrases)
    }


def _single_status_group(text: str) -> str | None:
    groups = _status_groups(text)
    return next(iter(groups)) if len(groups) == 1 else None


def _color_values(text: str) -> set[str]:
    return {
        _COLOR_SYNONYMS[token]
        for token in _tokens(text)
        if token in _COLOR_SYNONYMS
    }


def _single_color(text: str) -> str | None:
    colors = _color_values(text)
    return next(iter(colors)) if len(colors) == 1 else None


def _canonical_number(raw_value: str) -> str | None:
    try:
        number = Decimal(raw_value.replace(",", ""))
    except InvalidOperation:
        return None
    return format(number.normalize(), "f")


def _mask_dates(text: str) -> str:
    """Remove recognized dates so number/identifier extractors do not split them."""

    masked = _ISO_DATE_RE.sub(" ", text)
    masked = _MONTH_FIRST_DATE_RE.sub(" ", masked)
    return _DAY_FIRST_DATE_RE.sub(" ", masked)


def _numeric_values(text: str) -> set[str]:
    values = {
        _canonical_number(match.group(1))
        for match in _NUMBER_RE.finditer(_mask_dates(text))
    }
    return {value for value in values if value is not None}


def _canonical_date(year: int, month: int, day_number: int) -> str | None:
    try:
        return date(year, month, day_number).isoformat()
    except ValueError:
        return None


def _date_values(text: str) -> set[str]:
    values: set[str] = set()
    for year, month, day_number in _ISO_DATE_RE.findall(text):
        value = _canonical_date(int(year), int(month), int(day_number))
        if value:
            values.add(value)
    for month_name, day_number, year in _MONTH_FIRST_DATE_RE.findall(text):
        value = _canonical_date(
            int(year), _MONTH_NUMBERS[month_name.casefold()], int(day_number)
        )
        if value:
            values.add(value)
    for day_number, month_name, year in _DAY_FIRST_DATE_RE.findall(text):
        value = _canonical_date(
            int(year), _MONTH_NUMBERS[month_name.casefold()], int(day_number)
        )
        if value:
            values.add(value)
    return values


def _email_values(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _EMAIL_RE.finditer(text)}


def _phone_values(text: str) -> set[str]:
    return {
        re.sub(r"\D", "", match.group(0)) for match in _PHONE_RE.finditer(text)
    }


def _identifier_values(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in _IDENTIFIER_RE.finditer(_mask_dates(text))
    }


def _name_candidates(text: str) -> dict[str, str]:
    candidates: dict[str, str] = {}
    for match in _NAME_RE.finditer(text):
        candidate = match.group(0)
        if set(_tokens(candidate)) & _GENERIC_NAME_TOKENS:
            continue
        candidates[_normalize(candidate)] = candidate
    return candidates


def _scenario_product_tokens(scenario: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    product = scenario.get("product")
    if isinstance(product, Mapping):
        for key in ("names", "name", "brand", "brands", "product_type"):
            child = product.get(key)
            if isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
                values.extend(str(item) for item in child)
            elif child is not None:
                values.append(str(child))

    order = scenario.get("order")
    if isinstance(order, Mapping):
        products = order.get("products")
        if isinstance(products, str):
            try:
                products = ast.literal_eval(products)
            except (ValueError, SyntaxError):
                products = None
        if isinstance(products, Sequence) and not isinstance(products, (str, bytes)):
            for item in products:
                if isinstance(item, Mapping):
                    for key in ("brand", "product_type", "name"):
                        if item.get(key) is not None:
                            values.append(str(item[key]))

    tokens = {
        token
        for value in values
        for token in _tokens(value)
        if token not in _STOP_WORDS and not token.isdigit() and len(token) > 2
    }
    return tuple(sorted(tokens))


def _location_leaf(location: str) -> str:
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", location)
    return names[-1].casefold() if names else ""


@dataclass(frozen=True)
class AttributeRule:
    """The conservative attribute domain used for contradiction checking."""

    kind: str
    subject_anchors: tuple[str, ...]
    weak_anchors: tuple[str, ...]
    allowed_categories: tuple[str, ...]


@dataclass(frozen=True)
class ContradictionAssessment:
    """Result of deterministic structured contradiction checks."""

    verified: bool
    reason: str
    rule_kind: str | None = None
    same_entity_and_attribute: bool = False
    explicit_incompatible_value: bool = False


def _attribute_rule(
    location: str, scenario: Mapping[str, Any]
) -> AttributeRule | None:
    leaf = _location_leaf(location)
    product_tokens = _scenario_product_tokens(scenario)

    if leaf in _STATUS_LEAVES:
        return AttributeRule(
            kind="status",
            subject_anchors=(
                "delivery",
                "item",
                "order",
                "package",
                "parcel",
                "shipment",
            ),
            weak_anchors=(),
            allowed_categories=(
                "order_status",
                "status_attribute",
                "order_attribute",
            ),
        )
    if leaf in _COLOR_LEAVES:
        return AttributeRule(
            kind="color",
            subject_anchors=("product", "item", "order") + product_tokens,
            weak_anchors=(),
            allowed_categories=("product_attribute", "order_attribute"),
        )
    if leaf in _NUMERIC_LEAVES:
        return AttributeRule(
            kind="numeric",
            subject_anchors=_NUMERIC_ANCHORS + product_tokens,
            weak_anchors=("order", "item"),
            allowed_categories=(
                "order_amount",
                "numerical_attribute",
                "order_attribute",
                "product_attribute",
            ),
        )
    if leaf in _DATE_LEAVES:
        return AttributeRule(
            kind="date",
            subject_anchors=_DATE_ANCHORS + product_tokens,
            weak_anchors=("date", "order"),
            allowed_categories=("temporal_attribute", "order_attribute"),
        )
    if leaf in _NAME_LEAVES and ("personal" in location or "customer" in leaf):
        return AttributeRule(
            kind="name",
            subject_anchors=_NAME_ANCHORS,
            weak_anchors=(),
            allowed_categories=("customer_identity",),
        )
    if leaf in _EMAIL_LEAVES:
        return AttributeRule(
            kind="email",
            subject_anchors=_EMAIL_ANCHORS,
            weak_anchors=("contact",),
            allowed_categories=("customer_identity",),
        )
    if leaf in _PHONE_LEAVES:
        return AttributeRule(
            kind="phone",
            subject_anchors=_PHONE_ANCHORS,
            weak_anchors=("contact",),
            allowed_categories=("customer_identity",),
        )
    if leaf in _IDENTIFIER_LEAVES:
        return AttributeRule(
            kind="identifier",
            subject_anchors=_IDENTIFIER_ANCHORS,
            weak_anchors=("number",),
            allowed_categories=("order_attribute", "customer_identity"),
        )
    return None


def _matched_anchors(text: str, anchors: Sequence[str]) -> set[str]:
    tokens = set(_tokens(text))
    return {anchor for anchor in anchors if anchor in tokens}


def _replacement_values(
    original_values: set[str],
    claim_values: set[str],
    evidence_value: str,
    value_label: str,
) -> tuple[bool, str, str | None]:
    if evidence_value not in original_values:
        return (
            False,
            f"The original Agent claim does not positively assert the "
            f"evidence {value_label} {evidence_value!r}",
            None,
        )
    if evidence_value in claim_values:
        return (
            False,
            f"The candidate claim still asserts the evidence {value_label} "
            f"{evidence_value!r}; it is not a contradiction",
            None,
        )
    removed = original_values - claim_values
    added = claim_values - original_values
    if removed != {evidence_value}:
        return (
            False,
            f"The candidate claim does not isolate exactly one changed "
            f"{value_label}; removed values were {sorted(removed)!r}",
            None,
        )
    if len(added) != 1:
        return (
            False,
            f"The candidate claim is ambiguous: it adds {len(added)} {value_label} "
            f"values ({sorted(added)!r}) instead of exactly one",
            None,
        )
    new_value = next(iter(added))
    if new_value == evidence_value:
        return False, f"The {value_label} value did not change", None
    return True, "", new_value


def _claim_asserts_incompatible_value(
    claim: str,
    evidence: EvidenceItem,
    rule: AttributeRule,
    original_claim: str | None,
) -> tuple[bool, str, str | None]:
    """Check that ``claim`` explicitly asserts one incompatible value."""

    if _has_negation(claim):
        return (
            False,
            "The candidate claim contains a negation, so a positive assertion "
            "cannot be established confidently",
            None,
        )

    if rule.kind == "status":
        evidence_group = _single_status_group(evidence.text)
        if evidence_group is None:
            return (
                False,
                f"Evidence {evidence.text!r} is not a recognized single "
                "shipping/order status; reject or send to human review",
                None,
            )
        claim_group = _single_status_group(claim)
        if claim_group is None:
            return (
                False,
                "The candidate claim does not explicitly assert exactly one "
                "recognized status value; a paraphrase or vague state is not a "
                "verifiable contradiction",
                None,
            )
        if claim_group == evidence_group:
            return (
                False,
                f"The candidate claim asserts the same status "
                f"({claim_group!r}) as the evidence",
                None,
            )
        if original_claim is not None:
            if _has_negation(original_claim):
                return (
                    False,
                    "The original Agent claim contains a negation and cannot be "
                    "used as a positive evidence statement",
                    None,
                )
            if _single_status_group(original_claim) != evidence_group:
                return (
                    False,
                    "The original Agent claim does not positively assert the "
                    f"evidence status {evidence.text!r}",
                    None,
                )
        return True, "", claim_group

    if rule.kind == "color":
        evidence_color = _single_color(evidence.text)
        if evidence_color is None:
            return (
                False,
                f"Evidence {evidence.text!r} is not a recognized single color; "
                "reject or send to human review",
                None,
            )
        claim_color = _single_color(claim)
        if claim_color is None:
            return (
                False,
                "The candidate claim does not explicitly assert exactly one "
                "recognized color value",
                None,
            )
        if claim_color == evidence_color:
            return (
                False,
                f"The candidate claim asserts the same color "
                f"({claim_color!r}) as the evidence",
                None,
            )
        if original_claim is not None and _single_color(original_claim) != evidence_color:
            return (
                False,
                "The original Agent claim does not positively assert the "
                f"evidence color {evidence.text!r}",
                None,
            )
        return True, "", claim_color

    if rule.kind == "numeric":
        evidence_values = _numeric_values(evidence.text)
        if len(evidence_values) != 1:
            return (
                False,
                f"Evidence {evidence.text!r} is not a single unambiguous number",
                None,
            )
        evidence_value = next(iter(evidence_values))
        claim_values = _numeric_values(claim)
        if original_claim is None:
            if len(claim_values) != 1:
                return (
                    False,
                    "A pre-existing numeric defect must assert exactly one "
                    "unambiguous number",
                    None,
                )
            new_value = next(iter(claim_values))
            if new_value == evidence_value:
                return False, "The claim does not contradict the numeric evidence", None
            return True, "", new_value
        if _has_negation(original_claim):
            return (
                False,
                "The original Agent claim contains a negation and cannot be "
                "used as a positive evidence statement",
                None,
            )
        verified, reason, new_value = _replacement_values(
            _numeric_values(original_claim),
            claim_values,
            evidence_value,
            "numeric",
        )
        return verified, reason, new_value

    if rule.kind == "date":
        evidence_values = _date_values(evidence.text)
        if len(evidence_values) != 1:
            return (
                False,
                f"Evidence {evidence.text!r} is not a single unambiguous date",
                None,
            )
        evidence_value = next(iter(evidence_values))
        claim_values = _date_values(claim)
        if original_claim is None:
            if len(claim_values) != 1:
                return (
                    False,
                    "A pre-existing date defect must assert exactly one "
                    "unambiguous date",
                    None,
                )
            new_value = next(iter(claim_values))
            if new_value == evidence_value:
                return False, "The claim does not contradict the date evidence", None
            return True, "", new_value
        if _has_negation(original_claim):
            return (
                False,
                "The original Agent claim contains a negation and cannot be "
                "used as a positive evidence statement",
                None,
            )
        verified, reason, new_value = _replacement_values(
            _date_values(original_claim),
            claim_values,
            evidence_value,
            "date",
        )
        return verified, reason, new_value

    if rule.kind == "name":
        evidence_name = _normalize(evidence.text)
        if not evidence_name:
            return False, "Evidence name is empty after normalization", None
        claim_candidates = _name_candidates(claim)
        if evidence_name in claim_candidates:
            claim_candidates.pop(evidence_name)
        if original_claim is None:
            if len(claim_candidates) != 1:
                return (
                    False,
                    "A pre-existing identity defect must contain exactly one "
                    "unambiguous replacement name",
                    None,
                )
            return True, "", next(iter(claim_candidates.values()))
        original_candidates = _name_candidates(original_claim)
        if evidence_name not in original_candidates:
            return (
                False,
                "The original Agent claim does not explicitly state the "
                f"trusted customer name {evidence.text!r}",
                None,
            )
        added = {
            name: raw
            for name, raw in claim_candidates.items()
            if name not in original_candidates
        }
        if len(added) != 1:
            return (
                False,
                "The candidate claim must introduce exactly one unambiguous "
                "replacement customer name",
                None,
            )
        return True, "", next(iter(added.values()))

    if rule.kind in {"email", "phone", "identifier"}:
        extractor = {
            "email": _email_values,
            "phone": _phone_values,
            "identifier": _identifier_values,
        }[rule.kind]
        evidence_values = extractor(evidence.text)
        if len(evidence_values) != 1:
            return (
                False,
                f"Evidence {evidence.text!r} is not a single unambiguous "
                f"{rule.kind}",
                None,
            )
        evidence_value = next(iter(evidence_values))
        claim_values = extractor(claim)
        if original_claim is None:
            if len(claim_values) != 1:
                return (
                    False,
                    f"A pre-existing {rule.kind} defect must assert exactly one "
                    "unambiguous replacement value",
                    None,
                )
            new_value = next(iter(claim_values))
            if new_value == evidence_value:
                return False, "The claim does not contradict the evidence", None
            return True, "", new_value
        if _has_negation(original_claim):
            return (
                False,
                "The original Agent claim contains a negation and cannot be "
                "used as a positive evidence statement",
                None,
            )
        verified, reason, new_value = _replacement_values(
            extractor(original_claim),
            claim_values,
            evidence_value,
            rule.kind,
        )
        return verified, reason, new_value

    return (
        False,
        f"Attribute kind {rule.kind!r} has no conservative contradiction rule",
        None,
    )


def _assess_structured_contradiction(
    metadata: FactualInjectionMetadata,
    evidence: EvidenceItem,
    scenario: Mapping[str, Any],
) -> ContradictionAssessment:
    rule = _attribute_rule(evidence.location, scenario)
    if rule is None:
        return ContradictionAssessment(
            verified=False,
            reason=(
                f"Evidence location {evidence.location!r} has no supported "
                "structured contradiction rule; reject or send to human review"
            ),
        )
    if rule.kind in NON_CLAIM_ATTRIBUTE_KINDS:
        return ContradictionAssessment(
            verified=False,
            reason=(
                "Identifier evidence "
                f"({evidence.location!r}) is not an eligible factual claim. Order "
                "IDs, account IDs, customer IDs, and tracking numbers must not be "
                "used as factual-hallucination targets."
            ),
        )
    if metadata.error_category not in rule.allowed_categories:
        return ContradictionAssessment(
            verified=False,
            reason=(
                f"error_category {metadata.error_category!r} is not compatible "
                f"with evidence attribute kind {rule.kind!r}"
            ),
        )

    original = metadata.original_agent_claim
    injected = metadata.injected_claim

    if metadata.injection_status == "pre_existing":
        if _normalize(original) != _normalize(injected):
            return ContradictionAssessment(
                verified=False,
                reason=(
                    "Pre-existing defects must keep original_agent_claim and "
                    "injected_claim identical"
                ),
            )
        anchors = _matched_anchors(injected, rule.subject_anchors)
        if not anchors:
            return ContradictionAssessment(
                verified=False,
                reason=(
                    "The pre-existing claim does not name the evidence subject "
                    f"or attribute ({rule.kind})"
                ),
            )
        if not anchors - set(rule.weak_anchors):
            return ContradictionAssessment(
                verified=False,
                reason=(
                    "The pre-existing claim only contains generic subject "
                    "words; the attribute cannot be established confidently"
                ),
            )
        verified, reason, asserted = _claim_asserts_incompatible_value(
            injected, evidence, rule, original_claim=None
        )
        return ContradictionAssessment(
            verified=verified,
            reason=reason or "Pre-existing structured contradiction verified",
            rule_kind=rule.kind,
            same_entity_and_attribute=verified,
            explicit_incompatible_value=verified,
        )

    if _normalize(original) == _normalize(injected):
        return ContradictionAssessment(
            verified=False,
            reason="The injected claim must differ from the original Agent claim",
        )
    if not _claims_share_subject(original, injected, evidence.text):
        return ContradictionAssessment(
            verified=False,
            reason=(
                "The original and injected claims do not appear to refer to the "
                "same factual subject"
            ),
        )

    original_anchors = _matched_anchors(original, rule.subject_anchors)
    injected_anchors = _matched_anchors(injected, rule.subject_anchors)
    shared_anchors = original_anchors & injected_anchors
    if not original_anchors or not injected_anchors:
        return ContradictionAssessment(
            verified=False,
            reason=(
                "The original and injected claims must both name the evidence "
                f"subject or attribute ({rule.kind})"
            ),
        )
    if not shared_anchors - set(rule.weak_anchors):
        return ContradictionAssessment(
            verified=False,
            reason=(
                "The claims only share generic subject words; same entity and "
                "attribute cannot be established confidently"
            ),
        )

    verified, reason, asserted = _claim_asserts_incompatible_value(
        injected, evidence, rule, original_claim=original
    )
    if verified:
        try:
            _non_target_values_unchanged(original, injected, rule.kind)
        except FactualValidationError as exc:
            return ContradictionAssessment(
                verified=False,
                reason=str(exc),
                rule_kind=rule.kind,
                same_entity_and_attribute=True,
                explicit_incompatible_value=True,
            )
    return ContradictionAssessment(
        verified=verified,
        reason=reason or "Structured contradiction verified",
        rule_kind=rule.kind,
        same_entity_and_attribute=verified,
        explicit_incompatible_value=verified,
    )



_CONDITIONAL_CLAIM_PREFIX_RE = re.compile(
    r"^(?:if|when|unless|in case|assuming|provided that|as long as|should)\b",
    re.IGNORECASE,
)

_GENERIC_RULE_MARKERS = (
    "in general",
    "generally",
    "usually",
    "typically",
    "as a rule",
    "as policy",
    "our policy",
)


def _is_conditional_or_generic_claim(
    claim: str, context: str, rule: AttributeRule
) -> tuple[bool, str | None]:
    """Reject hypothetical/generic claims that do not assert the current entity's fact.

    A direct assertion in the claim is not rejected merely because other text in
    the containing sentence is conditional.
    """

    stripped_claim = claim.strip()
    if not stripped_claim:
        return True, "Claim is empty."
    if _CONDITIONAL_CLAIM_PREFIX_RE.match(stripped_claim):
        return (
            True,
            "The candidate claim is conditional or hypothetical, not a direct "
            "assertion about the current entity.",
        )

    claim_norm = _normalize(claim)
    padded_claim = f" {claim_norm} "
    if any(f" {marker} " in padded_claim for marker in _GENERIC_RULE_MARKERS):
        return (
            True,
            "The candidate claim is a generic policy/rule statement rather than "
            "an assertion about the specific entity in this conversation.",
        )

    if rule.kind == "status":
        ctx_norm = _normalize(context)
        prefix = ctx_norm.split(claim_norm, 1)[0] if claim_norm else ctx_norm
        if re.search(r"\bif\b", prefix) or ctx_norm.startswith("if "):
            if not any(
                cue in claim_norm
                for cue in (
                    "your order",
                    "this order",
                    "the order",
                    "your package",
                    "this package",
                    "your shipment",
                    "this shipment",
                    "currently",
                    "right now",
                )
            ):
                return (
                    True,
                    "The claim is framed as a conditional/general status rule "
                    "rather than the current order's actual status.",
                )
    return False, None


def _has_clear_self_contradiction_or_incoherence(text: str) -> str | None:
    """Detect only explicit, unambiguous self-contradictions/incoherence."""

    stripped = text.strip()
    if not stripped:
        return "Source sentence is empty."
    status_groups = _status_groups(stripped)
    if len(status_groups) >= 2 and re.search(r"\b(?:and|but)\b", _normalize(stripped)):
        return (
            "Source sentence explicitly asserts multiple incompatible status "
            f"values ({sorted(status_groups)!r})."
        )
    return None


_PROCEDURAL_SIDE_EFFECT_RE = re.compile(
    r"\b(?:verify your identity|issue a refund|process(?:ing)? a return|"
    r"escalate to|submit a ticket)\b",
    re.IGNORECASE,
)


def _procedural_side_effect(original: str, injected: str) -> str | None:
    """Flag a value edit that also introduces a new procedural action."""

    if _PROCEDURAL_SIDE_EFFECT_RE.search(injected) and not _PROCEDURAL_SIDE_EFFECT_RE.search(
        original
    ):
        return (
            "The injected claim adds a procedural action that was not in the "
            "original statement."
        )
    return None


def _assess_pre_existing_non_target_defects(
    turns: Sequence[Mapping[str, Any]],
    scenario: Mapping[str, Any],
    sop: Any,
    available_evidence: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return (clear_explicit_defects, human_review_flags).

    Only defects explicit and unambiguous from available inputs are clear.
    Absence of ACTION/log evidence is never treated as proof.
    """

    clear: list[str] = []
    review: list[str] = []

    # Clear, explicit contradiction between ACTION outcome and AGENT claim.
    # This is based only on present turns (no invented/missing actions).
    action_outcomes: list[tuple[int, str]] = []
    for turn in turns:
        if str(turn.get("speaker", "")).upper() != "ACTION":
            continue
        text = _normalize(str(turn.get("content", "")))
        if any(token in text for token in ("failed", "error", "declined", "unsuccessful")):
            action_outcomes.append((int(turn.get("turn_id", 0)), "negative"))
        elif any(token in text for token in ("successful", "completed", "processed", "confirmed")):
            action_outcomes.append((int(turn.get("turn_id", 0)), "positive"))

    for action_id, outcome in action_outcomes:
        for turn in turns:
            if str(turn.get("speaker", "")).upper() != "AGENT":
                continue
            turn_id = int(turn.get("turn_id", 0))
            if turn_id <= action_id or turn_id > action_id + 2:
                continue
            text = _normalize(str(turn.get("content", "")))
            if outcome == "negative" and any(
                token in text
                for token in ("successful", "completed", "processed successfully", "confirmed")
            ):
                clear.append(
                    f"Turn {turn_id}: explicit AGENT success claim contradicts a "
                    f"failed ACTION outcome at turn {action_id}."
                )
            if outcome == "positive" and any(
                token in text
                for token in ("failed", "could not", "unable to complete", "unsuccessful")
            ):
                clear.append(
                    f"Turn {turn_id}: explicit AGENT failure claim contradicts a "
                    f"successful ACTION outcome at turn {action_id}."
                )

    # Extremely conservative: do not infer missing actions/steps from absent logs.
    # Flag SOP-backed baseline compliance for human review rather than guessing.
    if sop not in (None, "", {}, []):
        review.append(
            "Baseline non-target compliance cannot be fully proven from available "
            "conversation/SOP/evidence/action-log context; human review recommended."
        )
    return clear, review


def _status_replacement_preserves_sop_branch(
    evidence: EvidenceItem,
    scenario: Mapping[str, Any],
    sop: Any,
) -> tuple[bool, str | None]:
    """Check whether a status substitution can preserve the applicable SOP branch.

    Conservative: do not assume adjacent statuses are safe. If SOP explicitly
    branches on the trusted status phrase, reject unless we can prove safety.
    """

    if sop in (None, "", {}, []):
        return True, None
    sop_text = json.dumps(sop, ensure_ascii=False).casefold()
    evidence_status = _single_status_group(evidence.text)
    if evidence_status is None:
        return False, "Evidence status is not a recognized single status."
    for phrase in _STATUS_GROUPS.get(evidence_status, ()):
        if phrase and phrase in sop_text and re.search(r"\b(?:if|when|only if)\b", sop_text):
            return (
                False,
                "SOP appears to branch on the trusted status; status substitution "
                "may change the applicable SOP branch.",
            )
    return True, None


# ---------------------------------------------------------------------------
# Conservative candidate pre-filtering
# ---------------------------------------------------------------------------
#
# Prompt preparation runs before the LLM. A candidate is offered only when an
# AGENT turn actually states the trusted evidence value in surface text and the
# value belongs to a conservative structured attribute domain. This prevents
# sending prompts that cannot possibly satisfy the deterministic validator (for
# example, an Agent question that never states the price, or an order ID used as
# if it were a factual claim).


@dataclass(frozen=True)
class FactualCandidateExclusion:
    """One evidence/claim pair that was not offered to the generation model."""

    reason_code: CandidateExclusionCode
    reason_detail: str
    evidence_location: str | None = None
    evidence_text: str | None = None
    attribute_kind: str | None = None
    turn_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "reason_detail": self.reason_detail,
            "evidence_location": self.evidence_location,
            "evidence_text": self.evidence_text,
            "attribute_kind": self.attribute_kind,
            "turn_id": self.turn_id,
        }


@dataclass(frozen=True)
class FactualCandidate:
    """An Agent claim that can be minimally contradicted against evidence."""

    candidate_id: str
    turn_id: int
    speaker: str
    attribute_kind: str
    error_category: str
    evidence: dict[str, Any]
    original_agent_claim: str
    claim_context: str

    def as_prompt_item(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "turn_id": self.turn_id,
            "attribute_kind": self.attribute_kind,
            "error_category": self.error_category,
            "evidence_source_type": self.evidence["source_type"],
            "evidence_source_location": self.evidence["location"],
            "evidence_text": self.evidence["text"],
            "suggested_original_claim": self.original_agent_claim,
            "claim_context": self.claim_context,
        }


@dataclass(frozen=True)
class FactualCandidateSelection:
    """Deterministic candidate-filter result for one conversation."""

    eligible: tuple[FactualCandidate, ...] = ()
    exclusions: tuple[FactualCandidateExclusion, ...] = ()

    @property
    def primary_exclusion_code(self) -> str | None:
        priority = {
            "NO_AGENT_TURN": 0,
            "NO_SUPPORTED_EVIDENCE": 1,
            "EXCLUDED_JUDGE_OVERLAP": 2,
            "VALUE_QUERIED_NOT_STATED": 3,
            "NO_AGENT_CLAIM_STATING_EVIDENCE": 4,
            "CONDITIONAL_OR_GENERIC_CLAIM": 5,
            "PRE_EXISTING_INCOHERENT_SENTENCE": 6,
            "ATTRIBUTE_NOT_EXPLICIT": 7,
            "NEGATED_CLAIM": 8,
            "AMBIGUOUS_EVIDENCE_VALUE": 9,
            "AMBIGUOUS_VALUE_OCCURRENCE": 10,
            "IDENTIFIER_NOT_FACTUAL_CLAIM": 11,
        }
        if not self.exclusions:
            return None
        return min(
            self.exclusions, key=lambda item: priority.get(item.reason_code, 99)
        ).reason_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_candidate_count": len(self.eligible),
            "eligible_candidates": [item.as_prompt_item() for item in self.eligible],
            "excluded_candidate_count": len(self.exclusions),
            "excluded_candidates": [item.to_dict() for item in self.exclusions],
            "primary_exclusion_code": self.primary_exclusion_code,
        }


def _phrase_spans(text: str, value: str) -> list[tuple[int, int]]:
    tokens = re.findall(r"[A-Za-z0-9]+", value)
    if not tokens:
        return []
    body = r"[^A-Za-z0-9]+".join(re.escape(token) for token in tokens)
    pattern = re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def _date_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in (_ISO_DATE_RE, _MONTH_FIRST_DATE_RE, _DAY_FIRST_DATE_RE):
        spans.extend((match.start(0), match.end(0)) for match in pattern.finditer(text))
    return spans


def _overlaps(span: tuple[int, int], others: Sequence[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in others)


def _evidence_value_spans(
    text: str, evidence_value: str, rule: AttributeRule
) -> list[tuple[int, int]]:
    """Find every surface span that asserts the canonical evidence value."""

    if rule.kind == "status":
        phrases = _STATUS_GROUPS.get(evidence_value, ())
        spans: list[tuple[int, int]] = []
        for phrase in sorted(phrases, key=len, reverse=True):
            spans.extend(_phrase_spans(text, phrase))
        return spans
    if rule.kind == "color":
        return _phrase_spans(text, evidence_value)
    if rule.kind == "numeric":
        date_spans = _date_spans(text)
        spans = [
            (match.start(1), match.end(1))
            for match in _NUMBER_RE.finditer(text)
            if _canonical_number(match.group(1)) == evidence_value
            and not _overlaps((match.start(1), match.end(1)), date_spans)
        ]
        return spans
    if rule.kind == "date":
        spans = []
        for pattern in (_ISO_DATE_RE, _MONTH_FIRST_DATE_RE, _DAY_FIRST_DATE_RE):
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) == 3:
                    if pattern is _ISO_DATE_RE:
                        canonical = _canonical_date(
                            int(groups[0]), int(groups[1]), int(groups[2])
                        )
                    elif pattern is _MONTH_FIRST_DATE_RE:
                        canonical = _canonical_date(
                            int(groups[2]),
                            _MONTH_NUMBERS[groups[0].casefold()],
                            int(groups[1]),
                        )
                    else:
                        canonical = _canonical_date(
                            int(groups[2]),
                            _MONTH_NUMBERS[groups[1].casefold()],
                            int(groups[0]),
                        )
                    if canonical == evidence_value:
                        spans.append((match.start(0), match.end(0)))
        return spans
    if rule.kind == "name":
        spans = [
            (match.start(0), match.end(0))
            for match in _NAME_RE.finditer(text)
            if _normalize(match.group(0)) == evidence_value
        ]
        return spans or _phrase_spans(text, evidence_value)
    if rule.kind == "email":
        return [
            (match.start(0), match.end(0))
            for match in _EMAIL_RE.finditer(text)
            if match.group(0).casefold() == evidence_value
        ]
    if rule.kind == "phone":
        return [
            (match.start(0), match.end(0))
            for match in _PHONE_RE.finditer(text)
            if re.sub(r"\D", "", match.group(0)) == evidence_value
        ]
    return _phrase_spans(text, evidence_value)


def _status_evidence_phrases(evidence_text: str) -> tuple[str, ...]:
    """Return the exact status phrase(s) present in one evidence value.

    Status groups intentionally treat lifecycle-adjacent labels as equivalent
    for contradiction checking, but candidate selection is stricter: the Agent
    must state the trusted evidence value itself, not a different label that
    happens to share the same group.
    """

    normalized = _normalize(evidence_text)
    for phrases in _STATUS_GROUPS.values():
        matched = tuple(
            phrase for phrase in phrases if contains_value(normalized, phrase)
        )
        if matched:
            return matched
    return ()


def _literal_evidence_value_spans(
    text: str, evidence_text: str, canonical: str, rule: AttributeRule
) -> list[tuple[int, int]]:
    """Find spans where the Agent states the exact evidence value.

    Unlike contradiction detection, this does not expand status values to
    group synonyms such as ``shipped`` / ``in transit`` / ``out for delivery``.
    """

    if rule.kind == "status":
        spans: list[tuple[int, int]] = []
        for phrase in _status_evidence_phrases(evidence_text):
            spans.extend(_phrase_spans(text, phrase))
        return spans
    return _evidence_value_spans(text, canonical, rule)


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max(
        text.rfind(".", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
    )
    sentence_start = left + 1 if left >= 0 else 0
    right_positions = [
        position for marker in ".!?" if (position := text.find(marker, end)) >= 0
    ]
    sentence_end = min(right_positions) + 1 if right_positions else len(text)
    return sentence_start, sentence_end


_DETERMINER_PREFIX_RE = re.compile(
    r"\b(?:the|your|our|this|that|these|those|its|my)\s+$", re.IGNORECASE
)


def _extract_claim_span(
    text: str,
    span: tuple[int, int],
    rule: AttributeRule,
) -> tuple[str, str]:
    """Return ``(suggested_claim, containing_sentence)`` for one value span."""

    start, end = span
    sentence_start, sentence_end = _sentence_bounds(text, start, end)
    sentence = text[sentence_start:sentence_end].strip()

    anchor_start: int | None = None
    for anchor in sorted(rule.subject_anchors, key=len, reverse=True):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(anchor)}(?![A-Za-z0-9])", re.IGNORECASE
        )
        matches = [match for match in pattern.finditer(text, sentence_start, start)]
        if matches:
            candidate_start = matches[-1].start()
            if anchor_start is None or candidate_start > anchor_start:
                anchor_start = candidate_start

    if anchor_start is not None:
        prefix = text[sentence_start:anchor_start]
        determiner = _DETERMINER_PREFIX_RE.search(prefix)
        claim_start = (
            sentence_start + determiner.start() if determiner else anchor_start
        )
    else:
        # No explicit subject anchor before the value. Keep a short window so
        # the suggestion does not swallow unrelated clauses.
        claim_start = max(sentence_start, start - 60)
        while claim_start < start and text[claim_start] not in " ,;":
            claim_start += 1
        if claim_start < start:
            claim_start += 1

    claim = text[claim_start:end].strip(" ,;:-")
    return claim, sentence


def _single_canonical_evidence_value(
    rule: AttributeRule, evidence_text: str
) -> tuple[str | None, str | None]:
    """Return ``(canonical_value, problem)`` for one evidence item."""

    if rule.kind == "status":
        value = _single_status_group(evidence_text)
        if value is None:
            return None, (
                f"Evidence {evidence_text!r} is not one recognized status value"
            )
        return value, None
    if rule.kind == "color":
        value = _single_color(evidence_text)
        if value is None:
            return None, f"Evidence {evidence_text!r} is not one recognized color"
        return value, None
    if rule.kind == "numeric":
        values = _numeric_values(evidence_text)
        if len(values) != 1:
            return None, f"Evidence {evidence_text!r} is not one unambiguous number"
        return next(iter(values)), None
    if rule.kind == "date":
        values = _date_values(evidence_text)
        if len(values) != 1:
            return None, f"Evidence {evidence_text!r} is not one unambiguous date"
        return next(iter(values)), None
    if rule.kind == "name":
        value = _normalize(evidence_text)
        if not value:
            return None, "Evidence name is empty after normalization"
        return value, None
    if rule.kind == "email":
        values = _email_values(evidence_text)
        if len(values) != 1:
            return None, f"Evidence {evidence_text!r} is not one unambiguous email"
        return next(iter(values)), None
    if rule.kind == "phone":
        values = _phone_values(evidence_text)
        if len(values) != 1:
            return None, f"Evidence {evidence_text!r} is not one unambiguous phone"
        return next(iter(values)), None
    return None, f"Attribute kind {rule.kind!r} has no candidate rule"


def _turn_queries_attribute(turn_text: str, rule: AttributeRule) -> bool:
    """Detect an Agent question about the attribute without stating its value."""

    if not _is_interrogative(turn_text):
        return False
    if _matched_anchors(turn_text, rule.subject_anchors):
        return True
    return rule.kind == "numeric" and "how much" in _normalize(turn_text)


def _agent_turns(turns: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [turn for turn in turns if str(turn.get("speaker")).upper() == "AGENT"]


def select_factual_candidates(
    turns: Sequence[Mapping[str, Any]],
    scenario: Mapping[str, Any],
    available_evidence: Sequence[Mapping[str, Any]],
    sop: Any = None,
) -> FactualCandidateSelection:
    """Deterministically select evidence-grounded Agent claims to offer the model.

    The filter is deliberately conservative. It never rewrites the conversation
    and it never invents a claim: a candidate exists only where the Agent turn
    literally states the trusted evidence value on a supported attribute.
    """

    agent_turns = _agent_turns(turns)
    if not agent_turns:
        return FactualCandidateSelection(
            exclusions=(
                FactualCandidateExclusion(
                    reason_code="NO_AGENT_TURN",
                    reason_detail="The conversation contains no AGENT turn.",
                ),
            )
        )

    supported = filter_supported_evidence(available_evidence, scenario)
    if not supported:
        return FactualCandidateSelection(
            exclusions=(
                FactualCandidateExclusion(
                    reason_code="NO_SUPPORTED_EVIDENCE",
                    reason_detail=(
                        "No trusted evidence attribute has a conservative "
                        "structured contradiction rule."
                    ),
                ),
            )
        )

    exclusions: list[FactualCandidateExclusion] = []
    candidates: list[FactualCandidate] = []
    seen: set[tuple[int, str, str]] = set()

    for evidence in supported:
        location = str(evidence.get("location", ""))
        evidence_text = str(evidence.get("text", ""))
        rule = _attribute_rule(location, scenario)
        if rule is None:
            exclusions.append(
                FactualCandidateExclusion(
                    reason_code="NO_SUPPORTED_EVIDENCE",
                    reason_detail=(
                        f"Evidence location {location!r} has no conservative rule."
                    ),
                    evidence_location=location,
                    evidence_text=evidence_text,
                )
            )
            continue
        if rule.kind in NON_CLAIM_ATTRIBUTE_KINDS:
            exclusions.append(
                FactualCandidateExclusion(
                    reason_code="IDENTIFIER_NOT_FACTUAL_CLAIM",
                    reason_detail=(
                        "Identifiers are structured evidence but are not eligible "
                        "factual claims; do not inject typos in IDs."
                    ),
                    evidence_location=location,
                    evidence_text=evidence_text,
                    attribute_kind=rule.kind,
                )
            )
            continue

        canonical, problem = _single_canonical_evidence_value(rule, evidence_text)
        if canonical is None or problem is not None:
            exclusions.append(
                FactualCandidateExclusion(
                    reason_code="AMBIGUOUS_EVIDENCE_VALUE",
                    reason_detail=problem or "Evidence value is ambiguous.",
                    evidence_location=location,
                    evidence_text=evidence_text,
                    attribute_kind=rule.kind,
                )
            )
            continue

        matching: list[tuple[Mapping[str, Any], list[tuple[int, int]]]] = []
        queried_turn: Mapping[str, Any] | None = None
        for turn in agent_turns:
            turn_text = str(turn.get("content", ""))
            spans = _literal_evidence_value_spans(
                turn_text, evidence_text, canonical, rule
            )
            if spans:
                matching.append((turn, spans))
            elif queried_turn is None and _turn_queries_attribute(turn_text, rule):
                queried_turn = turn

        if not matching:
            reason: CandidateExclusionCode = (
                "VALUE_QUERIED_NOT_STATED"
                if queried_turn is not None
                else "NO_AGENT_CLAIM_STATING_EVIDENCE"
            )
            detail = (
                "The Agent asks about the attribute but never states the trusted "
                "value, so there is no original factual claim to contradict."
                if reason == "VALUE_QUERIED_NOT_STATED"
                else "No AGENT turn states the trusted evidence value."
            )
            exclusions.append(
                FactualCandidateExclusion(
                    reason_code=reason,
                    reason_detail=detail,
                    evidence_location=location,
                    evidence_text=evidence_text,
                    attribute_kind=rule.kind,
                    turn_id=int(queried_turn["turn_id"]) if queried_turn else None,
                )
            )
            continue

        for turn, spans in matching:
            turn_id = int(turn["turn_id"])
            dedupe_key = (turn_id, rule.kind, canonical)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            turn_text = str(turn.get("content", ""))
            claim, context = _extract_claim_span(turn_text, spans[0], rule)
            if _is_interrogative(context) or _is_interrogative(claim):
                exclusions.append(
                    FactualCandidateExclusion(
                        reason_code="VALUE_QUERIED_NOT_STATED",
                        reason_detail=(
                            "The value appears only inside a question; a question "
                            "is not an explicit factual claim."
                        ),
                        evidence_location=location,
                        evidence_text=evidence_text,
                        attribute_kind=rule.kind,
                        turn_id=turn_id,
                    )
                )
                continue
            conditional, conditional_reason = _is_conditional_or_generic_claim(
                claim, context, rule
            )
            if conditional:
                exclusions.append(
                    FactualCandidateExclusion(
                        reason_code="CONDITIONAL_OR_GENERIC_CLAIM",
                        reason_detail=conditional_reason
                        or "The claim is conditional or generic.",
                        evidence_location=location,
                        evidence_text=evidence_text,
                        attribute_kind=rule.kind,
                        turn_id=turn_id,
                    )
                )
                continue
            incoherent_reason = _has_clear_self_contradiction_or_incoherence(context)
            if incoherent_reason:
                exclusions.append(
                    FactualCandidateExclusion(
                        reason_code="PRE_EXISTING_INCOHERENT_SENTENCE",
                        reason_detail=incoherent_reason,
                        evidence_location=location,
                        evidence_text=evidence_text,
                        attribute_kind=rule.kind,
                        turn_id=turn_id,
                    )
                )
                continue
            if _has_negation(claim):
                exclusions.append(
                    FactualCandidateExclusion(
                        reason_code="NEGATED_CLAIM",
                        reason_detail=(
                            "The claim contains a negation, so a positive value "
                            "assertion cannot be established."
                        ),
                        evidence_location=location,
                        evidence_text=evidence_text,
                        attribute_kind=rule.kind,
                        turn_id=turn_id,
                    )
                )
                continue
            if rule.kind == "status":
                try:
                    ev_item = EvidenceItem.model_validate(evidence)
                except Exception:
                    ev_item = None
                if ev_item is not None:
                    preserves, branch_reason = _status_replacement_preserves_sop_branch(
                        ev_item, scenario, sop
                    )
                    if not preserves:
                        exclusions.append(
                            FactualCandidateExclusion(
                                reason_code="EXCLUDED_JUDGE_OVERLAP",
                                reason_detail=branch_reason
                                or "Status substitution may change SOP branch.",
                                evidence_location=location,
                                evidence_text=evidence_text,
                                attribute_kind=rule.kind,
                                turn_id=turn_id,
                            )
                        )
                        continue
            anchors = _matched_anchors(claim, rule.subject_anchors)
            if not anchors or not anchors - set(rule.weak_anchors):
                exclusions.append(
                    FactualCandidateExclusion(
                        reason_code="ATTRIBUTE_NOT_EXPLICIT",
                        reason_detail=(
                            "The claim does not explicitly name the evidence "
                            "subject or attribute."
                        ),
                        evidence_location=location,
                        evidence_text=evidence_text,
                        attribute_kind=rule.kind,
                        turn_id=turn_id,
                    )
                )
                continue
            if (
                len(
                    _literal_evidence_value_spans(
                        context, evidence_text, canonical, rule
                    )
                )
                != 1
            ):
                exclusions.append(
                    FactualCandidateExclusion(
                        reason_code="AMBIGUOUS_VALUE_OCCURRENCE",
                        reason_detail=(
                            "The trusted value appears more than once in the claim "
                            "sentence, so a single-claim edit cannot be proven."
                        ),
                        evidence_location=location,
                        evidence_text=evidence_text,
                        attribute_kind=rule.kind,
                        turn_id=turn_id,
                    )
                )
                continue

            candidate = FactualCandidate(
                candidate_id=f"turn{turn_id}:{rule.kind}:{location}",
                turn_id=turn_id,
                speaker="AGENT",
                attribute_kind=rule.kind,
                error_category=rule.allowed_categories[0],
                evidence=dict(evidence),
                original_agent_claim=claim,
                claim_context=context,
            )
            candidates.append(candidate)

    return FactualCandidateSelection(
        eligible=tuple(candidates), exclusions=tuple(exclusions)
    )


def build_candidate_audit_rows(
    selection: FactualCandidateSelection,
    conversation_id: str | int,
    *,
    record_status: Literal["PREPARED", "EXCLUDED"],
    sent_to_llm: bool,
) -> list[dict[str, Any]]:
    """Build structured observability rows for one preparation decision."""

    rows: list[dict[str, Any]] = []
    for candidate in selection.eligible:
        rows.append(
            {
                "conversation_id": conversation_id,
                "record_status": record_status,
                "candidate_id": candidate.candidate_id,
                "turn_id": candidate.turn_id,
                "attribute_kind": candidate.attribute_kind,
                "evidence_location": candidate.evidence["location"],
                "evidence_text": candidate.evidence["text"],
                "filter_status": "ELIGIBLE",
                "reason_code": None,
                "reason_detail": None,
                "sent_to_llm": sent_to_llm,
            }
        )
    for exclusion in selection.exclusions:
        rows.append(
            {
                "conversation_id": conversation_id,
                "record_status": record_status,
                "candidate_id": None,
                "turn_id": exclusion.turn_id,
                "attribute_kind": exclusion.attribute_kind,
                "evidence_location": exclusion.evidence_location,
                "evidence_text": exclusion.evidence_text,
                "filter_status": "EXCLUDED",
                "reason_code": exclusion.reason_code,
                "reason_detail": exclusion.reason_detail,
                "sent_to_llm": False,
            }
        )
    if not rows:
        rows.append(
            {
                "conversation_id": conversation_id,
                "record_status": record_status,
                "candidate_id": None,
                "turn_id": None,
                "attribute_kind": None,
                "evidence_location": None,
                "evidence_text": None,
                "filter_status": "EXCLUDED" if not sent_to_llm else "ELIGIBLE",
                "reason_code": (
                    "NO_SUPPORTED_EVIDENCE" if not sent_to_llm else None
                ),
                "reason_detail": None,
                "sent_to_llm": sent_to_llm,
            }
        )
    return rows


def candidate_from_prompt_item(
    item: Mapping[str, Any],
) -> FactualCandidate | None:
    """Rebuild a ``FactualCandidate`` from the prepared prompt representation."""

    try:
        turn_id = int(item["turn_id"])
        candidate_id = str(item["candidate_id"])
        attribute_kind = str(item["attribute_kind"])
        error_category = str(item["error_category"])
        evidence = {
            "evidence_id": str(
                item.get("evidence_id", item.get("evidence_source_location", ""))
            ),
            "source_type": str(item["evidence_source_type"]),
            "location": str(item["evidence_source_location"]),
            "text": str(item["evidence_text"]),
        }
        original_agent_claim = str(item["suggested_original_claim"])
    except (KeyError, TypeError, ValueError):
        return None
    return FactualCandidate(
        candidate_id=candidate_id,
        turn_id=turn_id,
        speaker="AGENT",
        attribute_kind=attribute_kind,
        error_category=error_category,
        evidence=evidence,
        original_agent_claim=original_agent_claim,
        claim_context=str(item.get("claim_context", original_agent_claim)),
    )


def _exclusion_from_dict(item: Mapping[str, Any]) -> FactualCandidateExclusion | None:
    code = item.get("reason_code")
    if not isinstance(code, str):
        return None
    return FactualCandidateExclusion(
        reason_code=code,  # type: ignore[arg-type]
        reason_detail=str(item.get("reason_detail", "")),
        evidence_location=(
            str(item["evidence_location"])
            if item.get("evidence_location") is not None
            else None
        ),
        evidence_text=(
            str(item["evidence_text"]) if item.get("evidence_text") is not None else None
        ),
        attribute_kind=(
            str(item["attribute_kind"]) if item.get("attribute_kind") is not None else None
        ),
        turn_id=int(item["turn_id"]) if item.get("turn_id") is not None else None,
    )


def selection_from_prompt_record(
    record: Mapping[str, Any],
) -> FactualCandidateSelection:
    """Reconstruct the deterministic candidate selection for a prepared record.

    New prepared records embed ``factual_selection``. Legacy records are
    re-evaluated from their stored source context, so screening also works for
    prompts produced before the screening stage existed.
    """

    stored = record.get("factual_selection")
    if isinstance(stored, Mapping):
        candidates = tuple(
            candidate
            for item in stored.get("eligible_candidates", ())
            if isinstance(item, Mapping)
            if (candidate := candidate_from_prompt_item(item)) is not None
        )
        exclusions = tuple(
            exclusion
            for item in stored.get("excluded_candidates", ())
            if isinstance(item, Mapping)
            if (exclusion := _exclusion_from_dict(item)) is not None
        )
        if candidates or exclusions:
            return FactualCandidateSelection(
                eligible=candidates, exclusions=exclusions
            )

    source_turns = record.get("source_conversation")
    scenario = record.get("source_scenario")
    if not isinstance(source_turns, Sequence) or isinstance(
        source_turns, (str, bytes)
    ):
        return FactualCandidateSelection()
    if not isinstance(scenario, Mapping):
        return FactualCandidateSelection()

    available_evidence = record.get("available_evidence")
    if not isinstance(available_evidence, Sequence) or isinstance(
        available_evidence, (str, bytes)
    ):
        available_evidence = filter_supported_evidence(
            build_available_evidence(scenario), scenario
        )
    return select_factual_candidates(
        [dict(turn) for turn in source_turns if isinstance(turn, Mapping)],
        scenario,
        available_evidence,
        sop=record.get("source_sop"),
    )


# ---------------------------------------------------------------------------
# Pre-generation screening
# ---------------------------------------------------------------------------
#
# Python filters decide whether a candidate exists. An LLM screen must then
# confirm that candidate. ELIGIBLE is returned only when both agree. Unconfirmed
# or uncertain candidates stay REVIEW and must not be sent to injection.


_REVIEW_EXCLUSION_CODES = frozenset(
    {
        "AMBIGUOUS_EVIDENCE_VALUE",
        "AMBIGUOUS_VALUE_OCCURRENCE",
        "ATTRIBUTE_NOT_EXPLICIT",
    }
)

_REJECTION_CODES = frozenset(
    {
        "NO_TRUSTED_EVIDENCE",
        "UNSUPPORTED_ONLY",
        "CONFLICTING_EVIDENCE",
        "EXCLUDED_JUDGE_OVERLAP",
        "COHERENCE_IMPOSSIBLE",
        "SCREENING_REJECTED",
        "POST_GENERATION_REJECTED",
        "OTHER",
    }
)

_AMBIGUOUS_VALIDATION_MARKERS = (
    "cannot be established confidently",
    "send to human review",
    "human review",
    "ambiguous",
    "not a recognized single",
    "paraphrase or vague",
    "only share generic subject",
    "generic subject words",
)


@dataclass(frozen=True)
class FactualScreeningResult:
    """Combined Python-filter and LLM screening decision for one conversation."""

    decision: ScreeningDecision
    reason: str
    target_claim: str | None = None
    evidence_location: str | None = None
    evidence_text: str | None = None
    candidate_id: str | None = None
    requires_llm: bool = False
    llm_decision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "target_claim": self.target_claim,
            "evidence_location": self.evidence_location,
            "evidence_text": self.evidence_text,
            "candidate_id": self.candidate_id,
            "requires_llm": self.requires_llm,
            "llm_decision": self.llm_decision,
        }


def python_screening_stage(selection: FactualCandidateSelection) -> FactualScreeningResult:
    """Apply the existing candidate filter without calling a model.

    A Python-eligible candidate is not generation-eligible until the LLM screen
    confirms it. ``requires_llm`` marks that intermediate state.
    """

    if selection.eligible:
        candidate = selection.eligible[0]
        return FactualScreeningResult(
            decision="REVIEW",
            reason=(
                "Python filters found a candidate. LLM screening must confirm "
                "it before generation."
            ),
            target_claim=candidate.original_agent_claim,
            evidence_location=str(candidate.evidence.get("location", "")),
            evidence_text=str(candidate.evidence.get("text", "")),
            candidate_id=candidate.candidate_id,
            requires_llm=True,
        )

    exclusion = None
    if selection.exclusions:
        exclusion = min(
            selection.exclusions,
            key=lambda item: {
                "NO_AGENT_TURN": 0,
                "NO_SUPPORTED_EVIDENCE": 1,
                "EXCLUDED_JUDGE_OVERLAP": 2,
                "VALUE_QUERIED_NOT_STATED": 3,
                "NO_AGENT_CLAIM_STATING_EVIDENCE": 4,
                "CONDITIONAL_OR_GENERIC_CLAIM": 5,
                "PRE_EXISTING_INCOHERENT_SENTENCE": 6,
                "ATTRIBUTE_NOT_EXPLICIT": 7,
                "NEGATED_CLAIM": 8,
                "AMBIGUOUS_EVIDENCE_VALUE": 9,
                "AMBIGUOUS_VALUE_OCCURRENCE": 10,
                "IDENTIFIER_NOT_FACTUAL_CLAIM": 11,
            }.get(item.reason_code, 99),
        )
    if exclusion is None:
        return FactualScreeningResult(
            decision="REJECT",
            reason="No eligible factual candidate and no trusted evidence claim.",
        )
    decision: ScreeningDecision = (
        "REVIEW" if exclusion.reason_code in _REVIEW_EXCLUSION_CODES else "REJECT"
    )
    return FactualScreeningResult(
        decision=decision,
        reason=exclusion.reason_detail,
        target_claim=None,
        evidence_location=exclusion.evidence_location,
        evidence_text=exclusion.evidence_text,
        requires_llm=False,
    )


def build_screening_messages(
    candidate: FactualCandidate,
    turns: Sequence[Mapping[str, Any]],
    scenario: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build the provider-neutral screening request for one Python candidate."""

    agent_turn = next(
        (dict(turn) for turn in turns if int(turn.get("turn_id", -1)) == candidate.turn_id),
        None,
    )
    payload = {
        "candidate": candidate.as_prompt_item(),
        "agent_turn": agent_turn,
        "scenario": dict(scenario),
    }
    content = (
        "Screen one factual-hallucination candidate before any text is rewritten. "
        "Return JSON only with keys decision, target_claim, evidence_location, "
        "evidence_text, and reason. decision must be ELIGIBLE, REJECT, or REVIEW. "
        "Return ELIGIBLE only when the agent turn asserts the trusted evidence "
        "value for the same entity and attribute, the statement is not a question, "
        "and exactly one value can be contradicted. Return REVIEW when that is "
        "uncertain. Return REJECT when the candidate is clearly ineligible. "
        "Copy target_claim, evidence_location, and evidence_text exactly from the "
        "candidate. Do not invent a claim and do not rewrite the conversation.\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    return [{"role": "user", "content": content}]


def _screening_payload(raw_response: str) -> Mapping[str, Any] | None:
    parsed = _parse_possible_json(raw_response)
    if isinstance(parsed, Mapping):
        return parsed
    return None


def apply_llm_screening(
    stage: FactualScreeningResult,
    raw_response: str,
    selection: FactualCandidateSelection,
) -> FactualScreeningResult:
    """Combine a Python screening stage with one mocked or live model response."""

    if not stage.requires_llm:
        return stage
    candidate = next(
        (item for item in selection.eligible if item.candidate_id == stage.candidate_id),
        selection.eligible[0] if selection.eligible else None,
    )
    if candidate is None:
        return FactualScreeningResult(
            decision="REVIEW",
            reason="LLM screening was requested without a Python candidate.",
            requires_llm=False,
        )

    payload = _screening_payload(raw_response)
    if payload is None:
        return FactualScreeningResult(
            decision="REVIEW",
            reason="LLM screening response was not valid JSON. Generation is blocked.",
            target_claim=candidate.original_agent_claim,
            evidence_location=str(candidate.evidence.get("location", "")),
            evidence_text=str(candidate.evidence.get("text", "")),
            candidate_id=candidate.candidate_id,
            llm_decision=None,
        )

    raw_decision = str(payload.get("decision", "")).strip().upper()
    reason = payload.get("reason")
    reason_text = reason.strip() if isinstance(reason, str) else ""
    llm_claim = payload.get("target_claim")
    llm_evidence = payload.get("evidence_text")
    llm_location = payload.get("evidence_location")
    claim_ok = (
        isinstance(llm_claim, str)
        and llm_claim == candidate.original_agent_claim
    )
    evidence_ok = (
        isinstance(llm_evidence, str)
        and llm_evidence == str(candidate.evidence.get("text", ""))
    )
    location_ok = (
        not isinstance(llm_location, str)
        or not llm_location.strip()
        or llm_location == str(candidate.evidence.get("location", ""))
    )

    if raw_decision == "ELIGIBLE" and claim_ok and evidence_ok and location_ok:
        return FactualScreeningResult(
            decision="ELIGIBLE",
            reason=reason_text or "LLM screening confirmed the Python candidate.",
            target_claim=candidate.original_agent_claim,
            evidence_location=str(candidate.evidence.get("location", "")),
            evidence_text=str(candidate.evidence.get("text", "")),
            candidate_id=candidate.candidate_id,
            llm_decision="ELIGIBLE",
        )
    if raw_decision == "REJECT" and reason_text:
        return FactualScreeningResult(
            decision="REJECT",
            reason=reason_text,
            target_claim=candidate.original_agent_claim,
            evidence_location=str(candidate.evidence.get("location", "")),
            evidence_text=str(candidate.evidence.get("text", "")),
            candidate_id=candidate.candidate_id,
            llm_decision="REJECT",
        )
    if raw_decision == "REVIEW" and reason_text:
        held_reason = reason_text
    elif raw_decision == "ELIGIBLE":
        held_reason = (
            "LLM marked the candidate ELIGIBLE but did not echo the Python "
            "target claim and trusted evidence exactly. Generation is blocked."
        )
    else:
        held_reason = (
            "LLM screening did not return a confident ELIGIBLE or REJECT "
            "decision. Generation is blocked."
        )
    return FactualScreeningResult(
        decision="REVIEW",
        reason=held_reason,
        target_claim=candidate.original_agent_claim,
        evidence_location=str(candidate.evidence.get("location", "")),
        evidence_text=str(candidate.evidence.get("text", "")),
        candidate_id=candidate.candidate_id,
        llm_decision=raw_decision or None,
    )


def screen_factual_candidates(
    selection: FactualCandidateSelection,
    llm_response: str | None = None,
    *,
    llm_invoked: bool = False,
) -> FactualScreeningResult:
    """Screen candidates before generation.

    ``llm_invoked=False`` never returns ELIGIBLE. Callers that skip the model
    therefore cannot generate from an unconfirmed candidate.
    """

    stage = python_screening_stage(selection)
    if not stage.requires_llm:
        return stage
    if not llm_invoked or llm_response is None:
        return FactualScreeningResult(
            decision="REVIEW",
            reason=(
                "LLM screening was not run, so the Python candidate is not "
                "eligible for generation."
            ),
            target_claim=stage.target_claim,
            evidence_location=stage.evidence_location,
            evidence_text=stage.evidence_text,
            candidate_id=stage.candidate_id,
            requires_llm=True,
        )
    return apply_llm_screening(stage, llm_response, selection)


def screened_candidate_constraint(result: FactualScreeningResult) -> str:
    """Instruction appended to the injection prompt after an ELIGIBLE screen."""

    return (
        "\n\nSCREENING DECISION: ELIGIBLE\n"
        "Use only this screened candidate. Do not select a different claim.\n"
        f"candidate_id: {result.candidate_id}\n"
        f"target_claim: {result.target_claim}\n"
        f"evidence_location: {result.evidence_location}\n"
        f"evidence_text: {result.evidence_text}\n"
        "If this exact claim cannot be contradicted by one substitution, "
        "return the rejection JSON."
    )


def build_screening_audit_record(
    result: FactualScreeningResult,
    conversation_id: str | int,
    sampled_params: Mapping[str, Any] | None = None,
) -> FactualScreeningRecord:
    """Build one JSONL audit row for a screening decision."""

    return FactualScreeningRecord(
        conversation_id=conversation_id,
        target_judge=FACTUAL_JUDGE,
        decision=result.decision,
        reason=result.reason,
        target_claim=result.target_claim or "",
        evidence_location=result.evidence_location or "",
        evidence_text=result.evidence_text or "",
        candidate_id=result.candidate_id,
        requires_llm=result.requires_llm,
        llm_decision=result.llm_decision,
        sampled_params=dict(sampled_params or {}),
    )


@dataclass(frozen=True)
class GenerationScreeningGate:
    """Whether a prepared record may be sent to the generation model."""

    ready: bool
    decision: str
    reason: str


def evaluate_generation_gate(
    input_record: Mapping[str, Any], *, allow_unscreened: bool = False
) -> GenerationScreeningGate:
    """Block generation unless the candidate was screened ELIGIBLE.

    A missing screening block always fails closed. ``allow_unscreened`` is kept
    only as a deprecated source-compatible argument; it no longer opens a
    generation path.
    """

    screening = input_record.get("screening")
    if not isinstance(screening, Mapping):
        return GenerationScreeningGate(
            ready=False,
            decision="REVIEW",
            reason=(
                "The record has not passed the pre-generation screening stage; "
                "generation is blocked. Run screen_prompts.py first."
            ),
        )

    decision = str(screening.get("decision", "")).strip().upper()
    if decision == "ELIGIBLE" and isinstance(
        input_record.get("screened_candidate"), Mapping
    ):
        return GenerationScreeningGate(
            ready=True,
            decision="ELIGIBLE",
            reason=str(screening.get("reason", "Screening decision was ELIGIBLE.")),
        )
    if decision == "REJECT":
        return GenerationScreeningGate(
            ready=False,
            decision="REJECT",
            reason=str(screening.get("reason", "Screening rejected the candidate.")),
        )
    if decision == "ELIGIBLE":
        return GenerationScreeningGate(
            ready=False,
            decision="REVIEW",
            reason=(
                "Screening was ELIGIBLE but screened_candidate metadata is "
                "missing, so the target claim cannot be pinned. Generation is "
                "blocked."
            ),
        )
    return GenerationScreeningGate(
        ready=False,
        decision="REVIEW",
        reason=str(
            screening.get(
                "reason",
                "Screening did not return a confident ELIGIBLE decision.",
            )
        ),
    )


def extract_claim_fields(payload: Any) -> tuple[str, str, str]:
    """Best-effort (target claim, evidence location, evidence text) extraction."""

    if not isinstance(payload, Mapping):
        return "", "", ""
    metadata = payload.get("injection_metadata")
    if not isinstance(metadata, Mapping):
        return "", "", ""
    return (
        str(metadata.get("original_agent_claim") or metadata.get("injected_claim") or ""),
        str(metadata.get("evidence_source_location") or ""),
        str(metadata.get("evidence_text") or ""),
    )


# ---------------------------------------------------------------------------
# Evidence verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceVerification:
    """Entity, attribute, and context check against one trusted evidence item.

    ``SUPPORTED`` means the evidence confirms the claim. Injection accepts only
    ``CONTRADICTED``. ``UNSUPPORTED`` means the evidence neither confirms nor
    denies the claim. ``NOT_VERIFIABLE`` means the evidence cannot decide.
    """

    classification: EvidenceClassification
    entity_match: bool
    attribute_match: bool
    context_match: bool
    reason: str


def _evidence_item(evidence: EvidenceItem | Mapping[str, Any]) -> EvidenceItem | None:
    if isinstance(evidence, EvidenceItem):
        return evidence
    try:
        return EvidenceItem.model_validate(evidence)
    except ValidationError:
        return None


def _values_for_kind(text: str, kind: str) -> set[str]:
    """Return the canonical values one extractor recognizes in ``text``."""

    return set(_kind_values(text).get(kind, set()))


def _same_evidence(left: EvidenceItem, right: EvidenceItem) -> bool:
    return (
        left.source_type,
        left.location,
        left.text,
    ) == (
        right.source_type,
        right.location,
        right.text,
    )


def _replacement_format_error(
    kind: str, value: str, evidence_text: str
) -> str | None:
    """Reject a replacement value that is not valid for the target attribute."""

    if kind == "numeric":
        try:
            number = Decimal(value)
        except InvalidOperation:
            return f"Replacement numeric value {value!r} is not a valid number."
        if not number.is_finite():
            return f"Replacement numeric value {value!r} is not finite."
        if number < 0 and not any(
            item.startswith("-") for item in _numeric_values(evidence_text)
        ):
            return (
                f"Replacement numeric value {value!r} is negative while the "
                "trusted evidence is non-negative."
            )
        return None

    if kind == "date":
        try:
            date.fromisoformat(value)
        except ValueError:
            return f"Replacement date value {value!r} is not a valid date."
        return None

    if kind == "email":
        local, separator, domain = value.partition("@")
        if (
            not separator
            or not local
            or not domain
            or ".." in value
            or domain.startswith(".")
            or domain.endswith(".")
            or _EMAIL_RE.fullmatch(value) is None
        ):
            return f"Replacement email value {value!r} is not a valid address."
        return None

    if kind == "phone":
        digits = re.sub(r"\D", "", value)
        if not 7 <= len(digits) <= 15:
            return (
                f"Replacement phone value {value!r} must contain between 7 and "
                "15 digits."
            )
        return None

    if kind == "identifier":
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{3,}", value) is None:
            return f"Replacement identifier value {value!r} has an invalid format."
        return None

    if kind == "name":
        if not _name_candidates(value):
            return f"Replacement name value {value!r} has an invalid format."
        return None

    if kind == "status":
        if value not in _STATUS_GROUPS:
            return f"Replacement status value {value!r} is not recognized."
        return None

    if kind == "color":
        if value not in set(_COLOR_SYNONYMS.values()):
            return f"Replacement color value {value!r} is not recognized."
        return None

    return None


def _validate_replacement_value(
    *,
    kind: str,
    value: str,
    evidence: EvidenceItem,
    all_evidence: Sequence[EvidenceItem | Mapping[str, Any]] | None,
) -> str | None:
    """Check replacement format and collisions with other trusted values."""

    format_error = _replacement_format_error(kind, value, evidence.text)
    if format_error:
        return format_error

    if all_evidence is None:
        return None

    comparable = _normalize(value) if kind == "name" else value
    for raw_item in all_evidence:
        item = _evidence_item(raw_item)
        if item is None or _same_evidence(item, evidence):
            continue
        if comparable in _values_for_kind(item.text, kind):
            return (
                f"Replacement {kind} value {value!r} collides with another "
                f"trusted evidence value at {item.location!r}. It cannot be "
                "used as a controlled contradiction."
            )
    return None


def verify_factual_evidence(
    *,
    original_claim: str,
    injected_claim: str,
    evidence: EvidenceItem | Mapping[str, Any],
    scenario: Mapping[str, Any],
    error_category: str | None = None,
    injection_status: Literal["injected", "pre_existing"] = "injected",
    all_evidence: Sequence[EvidenceItem | Mapping[str, Any]] | None = None,
) -> EvidenceVerification:
    """Classify one claim/evidence pair. Accept only a direct contradiction."""

    item = _evidence_item(evidence)
    if item is None:
        return EvidenceVerification(
            classification="NOT_VERIFIABLE",
            entity_match=False,
            attribute_match=False,
            context_match=False,
            reason="The evidence item is missing or invalid, so the claim cannot be verified.",
        )

    rule = _attribute_rule(item.location, scenario)
    if rule is None or rule.kind in NON_CLAIM_ATTRIBUTE_KINDS:
        return EvidenceVerification(
            classification="NOT_VERIFIABLE",
            entity_match=False,
            attribute_match=False,
            context_match=False,
            reason=(
                f"Evidence location {item.location!r} is not a supported factual "
                "attribute, so the claim is not verifiable."
            ),
        )

    canonical, problem = _single_canonical_evidence_value(rule, item.text)
    if canonical is None or problem is not None:
        return EvidenceVerification(
            classification="NOT_VERIFIABLE",
            entity_match=False,
            attribute_match=False,
            context_match=False,
            reason=problem or "The trusted evidence value is ambiguous.",
        )

    category = error_category or rule.allowed_categories[0]
    attribute_match = category in rule.allowed_categories
    claim_for_anchors = injected_claim if injection_status == "pre_existing" else original_claim
    original_anchors = _matched_anchors(claim_for_anchors, rule.subject_anchors)
    injected_anchors = _matched_anchors(injected_claim, rule.subject_anchors)
    entity_match = bool(
        (original_anchors & injected_anchors) - set(rule.weak_anchors)
    )
    if rule.kind == "email":
        original_local_parts = {
            value.partition("@")[0] for value in _email_values(claim_for_anchors)
        }
        injected_local_parts = {
            value.partition("@")[0] for value in _email_values(injected_claim)
        }
        entity_match = entity_match or bool(
            original_local_parts & injected_local_parts
        )
    conditional, conditional_reason = _is_conditional_or_generic_claim(
        injected_claim, injected_claim, rule
    )
    context_match = (
        entity_match
        and attribute_match
        and not _is_interrogative(original_claim)
        and not _is_interrogative(injected_claim)
        and not conditional
        and (
            injection_status == "pre_existing"
            or _claims_share_subject(original_claim, injected_claim, item.text)
        )
    )
    if not entity_match or not attribute_match or not context_match:
        reasons: list[str] = []
        if not entity_match:
            reasons.append("the claim does not match the evidence entity")
        if not attribute_match:
            reasons.append(
                f"error_category {category!r} does not match attribute {rule.kind}"
            )
        if not context_match:
            reasons.append(
                conditional_reason
                or "the claim context does not match the trusted evidence"
            )
        return EvidenceVerification(
            classification="NOT_VERIFIABLE",
            entity_match=entity_match,
            attribute_match=attribute_match,
            context_match=context_match,
            reason="Not verifiable: " + "; ".join(reasons),
        )

    verified, reason, asserted = _claim_asserts_incompatible_value(
        injected_claim,
        item,
        rule,
        original_claim=None if injection_status == "pre_existing" else original_claim,
    )
    if verified:
        try:
            if injection_status != "pre_existing":
                _non_target_values_unchanged(original_claim, injected_claim, rule.kind)
        except FactualValidationError as exc:
            return EvidenceVerification(
                classification="NOT_VERIFIABLE",
                entity_match=True,
                attribute_match=True,
                context_match=True,
                reason=str(exc),
            )
        if asserted is not None:
            replacement_error = _validate_replacement_value(
                kind=rule.kind,
                value=asserted,
                evidence=item,
                all_evidence=all_evidence,
            )
            if replacement_error:
                return EvidenceVerification(
                    classification="NOT_VERIFIABLE",
                    entity_match=True,
                    attribute_match=True,
                    context_match=True,
                    reason=replacement_error,
                )
        return EvidenceVerification(
            classification="CONTRADICTED",
            entity_match=True,
            attribute_match=True,
            context_match=True,
            reason=reason or "The claim directly contradicts the trusted evidence value.",
        )

    lowered = reason.casefold()
    supported_markers = (
        "still asserts the evidence",
        "same status",
        "same color",
        "did not change",
        "does not contradict",
        "not a contradiction",
    )
    unsupported_markers = (
        "does not explicitly assert exactly one",
        "paraphrase or vague",
        "vague state",
        "adds 0",
    )
    if any(marker in lowered for marker in supported_markers):
        return EvidenceVerification(
            classification="SUPPORTED",
            entity_match=True,
            attribute_match=True,
            context_match=True,
            reason=reason,
        )
    if any(marker in lowered for marker in unsupported_markers):
        return EvidenceVerification(
            classification="UNSUPPORTED",
            entity_match=True,
            attribute_match=True,
            context_match=True,
            reason=reason or "The evidence neither confirms nor denies the claim.",
        )
    return EvidenceVerification(
        classification="NOT_VERIFIABLE",
        entity_match=True,
        attribute_match=True,
        context_match=True,
        reason=reason or "The evidence cannot establish whether the claim is true.",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _get_context_value(context: Mapping[str, Any], key: str) -> Any:
    if key not in context:
        raise FactualValidationError(
            f"Factual validation context is missing required key {key!r}"
        )
    return context[key]


def _validate_evidence_reference(
    metadata: FactualInjectionMetadata,
    available_evidence: Sequence[EvidenceItem],
) -> EvidenceItem:
    """Resolve the cited evidence with exact source, location, and text equality."""

    matches = [
        item
        for item in available_evidence
        if item.source_type == metadata.evidence_source_type
        and item.location == metadata.evidence_source_location
    ]
    if not matches:
        raise FactualValidationError(
            "No trusted evidence item matches "
            f"{metadata.evidence_source_type!r}/"
            f"{metadata.evidence_source_location!r}. "
            "Absence of evidence is not evidence of contradiction."
        )

    distinct_values = {item.text for item in matches}
    if len(distinct_values) > 1:
        raise FactualValidationError(
            "Conflicting trusted evidence exists at "
            f"{metadata.evidence_source_location!r}: "
            f"{sorted(item.text for item in matches)!r}; reject rather than "
            "injecting a factual_hallucination sample"
        )

    canonical = matches[0]
    if metadata.evidence_text != canonical.text:
        raise FactualValidationError(
            "evidence_text must exactly quote the trusted evidence value "
            f"{canonical.text!r}; got {metadata.evidence_text!r}. "
            "Normalized equality is not accepted for copied evidence."
        )
    return canonical


def _validate_claim_structure(
    metadata: FactualInjectionMetadata,
    evidence: EvidenceItem,
) -> None:
    """Check claim-to-conversation structure without judging semantics.

    Semantic contradiction checking is performed separately by
    ``_assess_structured_contradiction``.
    """

    original = metadata.original_agent_claim
    injected = metadata.injected_claim

    if metadata.injection_status == "injected":
        if _normalize(original) == _normalize(injected):
            raise FactualValidationError(
                "The injected claim must differ from the original Agent claim"
            )
        if not contains_value(original, evidence.text):
            raise FactualValidationError(
                "The original Agent claim must state the trusted evidence value "
                f"{evidence.text!r} before it is contradicted"
            )
    else:
        if _normalize(original) != _normalize(injected):
            raise FactualValidationError(
                "Pre-existing defects must keep original_agent_claim and "
                "injected_claim identical because the conversation is unchanged"
            )
        if contains_value(injected, evidence.text):
            raise FactualValidationError(
                "A pre-existing claim that contains the trusted evidence value "
                "does not contradict it"
            )


def _validate_conversation(
    metadata: FactualInjectionMetadata,
    modified_turns: Sequence[FactualTurn],
    source_turns: Sequence[Mapping[str, Any]],
) -> list[int]:
    modified = [turn.model_dump(mode="json") for turn in modified_turns]
    expected_ids = list(range(1, len(modified) + 1))
    actual_ids = [turn["turn_id"] for turn in modified]
    if actual_ids != expected_ids:
        raise FactualValidationError(
            "injected_conversation turn_ids must be strictly sequential "
            f"starting at 1; got {actual_ids!r}"
        )

    if len(modified) != len(source_turns):
        raise FactualValidationError(
            "The injected conversation must preserve the source turn count; "
            f"source has {len(source_turns)}, output has {len(modified)}"
        )

    changed_turns: list[int] = []
    for source, output in zip(source_turns, modified):
        if source.get("turn_id") != output["turn_id"]:
            raise FactualValidationError(
                "Turn ids must match the source conversation at every position"
            )
        if str(source.get("speaker")).upper() != output["speaker"]:
            raise FactualValidationError(
                f"Speaker changed at turn {output['turn_id']}; only Agent text "
                "may be edited"
            )
        if source.get("content") != output["content"]:
            changed_turns.append(output["turn_id"])

    expected_changes = (
        [] if metadata.injection_status == "pre_existing" else [metadata.injection_turn_id]
    )
    if changed_turns != expected_changes:
        raise FactualValidationError(
            "Only the single target AGENT turn may change. "
            f"Expected changed turns {expected_changes!r}, got {changed_turns!r}. "
            "Customer turns, ACTION turns, and unrelated Agent turns must be "
            "preserved exactly."
        )
    return changed_turns


def _verify_trusted_evidence(
    scenario: Mapping[str, Any],
    available_evidence: Sequence[EvidenceItem],
    cited: EvidenceItem,
) -> tuple[bool, Literal["VERIFIED", "NOT_VERIFIED"]]:
    """Compare prepared evidence against rebuilt scenario evidence.

    Scenario-backed items must match the original scenario exactly. Merchant KB
    or catalog items cannot be rebuilt from AB-CD source files in this pilot, so
    those records are marked NOT_VERIFIED even when they were not mutated.
    """

    rebuilt = {
        (item["source_type"], item["location"], item["text"])
        for item in build_available_evidence(scenario)
    }
    unmodified = True
    has_unverified_source = False
    for item in available_evidence:
        key = (item.source_type, item.location, item.text)
        if item.source_type == "scenario":
            if key not in rebuilt:
                unmodified = False
        else:
            has_unverified_source = True
    cited_key = (cited.source_type, cited.location, cited.text)
    if cited_key not in {(item.source_type, item.location, item.text) for item in available_evidence}:
        unmodified = False
    if cited.source_type == "scenario" and cited_key not in rebuilt:
        unmodified = False
    if not unmodified:
        raise FactualValidationError(
            "Trusted evidence does not match the original scenario; the evidence "
            "list or cited values were modified"
        )
    integrity: Literal["VERIFIED", "NOT_VERIFIED"] = (
        "NOT_VERIFIED" if has_unverified_source else "VERIFIED"
    )
    return True, integrity


def _validate_metadata_shape(metadata: FactualInjectionMetadata) -> None:
    if metadata.target_judge != FACTUAL_JUDGE:
        raise FactualValidationError(
            f"target_judge must be {FACTUAL_JUDGE!r}; got {metadata.target_judge!r}"
        )
    if metadata.scenario_modified:
        raise FactualValidationError(
            "scenario_modified must be false for factual_hallucination records"
        )
    if metadata.modified_scenario_fields:
        raise FactualValidationError(
            "modified_scenario_fields must be empty for factual_hallucination records"
        )
    if metadata.already_defective and metadata.injection_status != "pre_existing":
        raise FactualValidationError(
            "already_defective=true requires injection_status='pre_existing'"
        )
    if not metadata.already_defective and metadata.injection_status != "injected":
        raise FactualValidationError(
            "already_defective=false requires injection_status='injected'"
        )
    expected_scope = (
        "pre_existing" if metadata.injection_status == "pre_existing" else "single_turn"
    )
    if metadata.scope != expected_scope:
        raise FactualValidationError(
            f"scope must be {expected_scope!r} for injection_status "
            f"{metadata.injection_status!r}"
        )
    validate_factual_error_category(metadata.error_category)
    if metadata.expected_target_verdict != "FAIL":
        raise FactualValidationError(
            "expected_target_verdict must be 'FAIL' for a successful factual record"
        )

    risk = metadata.non_target_risk_assessment
    risk_flags = {
        "action_accuracy": risk.action_accuracy,
        "procedure_following": risk.procedure_following,
        "intent_understanding": risk.intent_understanding,
        "task_closure": risk.task_closure,
        "alternative_guidance": risk.alternative_guidance,
        "missing_data": risk.missing_data,
        "data_conflict": risk.data_conflict,
    }
    violated = [name for name, flag in risk_flags.items() if flag]
    if violated:
        raise FactualValidationError(
            "Injection overlaps with excluded judge(s): "
            f"{', '.join(violated)}. Reject the candidate instead."
        )


def build_factual_validation_context(
    *,
    conversation_id: str | int,
    source_conversation: Sequence[Mapping[str, Any]],
    scenario: Mapping[str, Any],
    sop: Any,
    judge_definition: Mapping[str, Any],
    available_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble the private context used by the generator for validation."""

    return {
        "conversation_id": conversation_id,
        "target_judge": FACTUAL_JUDGE,
        "source_conversation": [dict(turn) for turn in source_conversation],
        "source_scenario": dict(scenario),
        "source_sop": sop,
        "judge_definition": dict(judge_definition),
        "available_evidence": [dict(item) for item in available_evidence],
    }


def validate_factual_golden_record(
    payload: Any,
    sampled_params: Mapping[str, Any] | None = None,
    input_record: Mapping[str, Any] | None = None,
) -> FactualHallucinationGoldenRecordOutput:
    """Validate one successful factual-hallucination payload and build its record."""

    model_output = FactualModelOutput.model_validate(payload)
    metadata = model_output.injection_metadata
    sampled_params = sampled_params or {}

    if input_record is None:
        raise FactualValidationError(
            "Factual validation requires the prepared source context "
            "(source_conversation, scenario, evidence, and judge definition)"
        )

    source_conversation = _get_context_value(input_record, "source_conversation")
    if not isinstance(source_conversation, Sequence) or isinstance(
        source_conversation, (str, bytes)
    ):
        raise FactualValidationError("source_conversation must be a list of turns")
    source_turns = [dict(turn) for turn in source_conversation]

    scenario = _get_context_value(input_record, "source_scenario")
    if not isinstance(scenario, Mapping):
        raise FactualValidationError("source_scenario must be an object")
    sop = _get_context_value(input_record, "source_sop")
    judge_definition = _get_context_value(input_record, "judge_definition")
    if not isinstance(judge_definition, Mapping):
        raise FactualValidationError("judge_definition must be an object")

    raw_evidence = _get_context_value(input_record, "available_evidence")
    if not isinstance(raw_evidence, Sequence) or isinstance(
        raw_evidence, (str, bytes)
    ):
        raise FactualValidationError("available_evidence must be a list")
    try:
        available_evidence = [
            EvidenceItem.model_validate(item) for item in raw_evidence
        ]
    except ValidationError as exc:
        raise FactualValidationError(
            f"available_evidence contains an invalid item: {exc}"
        ) from exc

    _validate_metadata_shape(metadata)
    screened = input_record.get("screened_candidate")
    if isinstance(screened, Mapping):
        updates: dict[str, Any] = {}
        candidate_id = screened.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            updates["candidate_id"] = candidate_id
        screening = input_record.get("screening")
        if (
            isinstance(screening, Mapping)
            and str(screening.get("decision", "")).upper() == "ELIGIBLE"
        ):
            updates["screening_decision"] = "ELIGIBLE"
        if updates:
            metadata = metadata.model_copy(update=updates)
            model_output = model_output.model_copy(
                update={"injection_metadata": metadata}
            )
    evidence = _validate_evidence_reference(metadata, available_evidence)
    evidence_unmodified, evidence_integrity = _verify_trusted_evidence(
        scenario, available_evidence, evidence
    )
    _validate_claim_structure(metadata, evidence)
    if metadata.injection_status == "injected":
        claim_incoherence = _has_clear_self_contradiction_or_incoherence(
            metadata.injected_claim
        )
        if claim_incoherence:
            raise FactualValidationError(
                "Injected conversation is incoherent: " + claim_incoherence
            )
    assessment = _assess_structured_contradiction(metadata, evidence, scenario)
    if not assessment.verified:
        raise FactualValidationError(
            "Structured contradiction validation failed: " + assessment.reason
        )
    verification = verify_factual_evidence(
        original_claim=metadata.original_agent_claim,
        injected_claim=metadata.injected_claim,
        evidence=evidence,
        scenario=scenario,
        error_category=metadata.error_category,
        injection_status=metadata.injection_status,
        all_evidence=available_evidence,
    )
    if verification.classification != "CONTRADICTED":
        raise FactualValidationError(
            "Evidence verification did not establish a direct contradiction "
            f"(classification {verification.classification}): {verification.reason}"
        )

    if (
        metadata.injection_status == "injected"
        and assessment.rule_kind == "status"
    ):
        preserves, branch_reason = _status_replacement_preserves_sop_branch(
            evidence, scenario, sop
        )
        if not preserves:
            raise FactualValidationError(
                "Status substitution is not allowed: "
                + (branch_reason or "it may change the applicable SOP branch")
            )

    modified_turns = model_output.injected_conversation
    changed_turns = _validate_conversation(metadata, modified_turns, source_turns)

    turn_map = {turn.turn_id: turn for turn in modified_turns}
    target_turn = turn_map.get(metadata.injection_turn_id)
    if target_turn is None:
        raise FactualValidationError(
            f"injection_turn_id {metadata.injection_turn_id} is absent from the "
            "modified conversation"
        )
    if target_turn.speaker != "AGENT":
        raise FactualValidationError(
            f"injection_turn_id {metadata.injection_turn_id} must reference an "
            f"AGENT turn; got {target_turn.speaker}"
        )

    source_turn = source_turns[metadata.injection_turn_id - 1]
    if str(source_turn.get("speaker")).upper() != "AGENT":
        raise FactualValidationError(
            f"Source turn {metadata.injection_turn_id} is not an AGENT turn"
        )

    # A question is not an explicit factual claim. The original statement and
    # the edited statement must both be assertions, not interrogatives. Checking
    # the containing sentence catches edits that only replace a value inside a
    # question (for example "Were the boots $64?" -> "Were the boots $94?").
    if _is_interrogative(
        _containing_sentence(
            str(source_turn.get("content")), metadata.original_agent_claim
        )
    ):
        raise FactualValidationError(
            "The original Agent turn is interrogative, so it does not contain an "
            "explicit factual claim that can be contradicted"
        )
    if _is_interrogative(
        _containing_sentence(target_turn.content, metadata.injected_claim)
    ):
        raise FactualValidationError(
            "The injected Agent turn is interrogative, so the injection does not "
            "assert a factual contradiction"
        )

    if metadata.injection_status == "injected":
        _check_claim_present(
            metadata.original_agent_claim,
            str(source_turn.get("content")),
            "Original Agent turn",
        )
        _check_claim_present(
            metadata.injected_claim,
            target_turn.content,
            "Injected Agent turn",
        )
        expected_content = _substituted_turn_content(
            str(source_turn.get("content")),
            metadata.original_agent_claim,
            metadata.injected_claim,
        )
        if target_turn.content != expected_content:
            raise FactualValidationError(
                "The edited Agent turn is not an exact single-claim substitution "
                "of original_agent_claim with injected_claim. Extra edits cannot "
                "be proven to introduce only one factual contradiction"
            )
        single_injected_claim = len(changed_turns) == 1
        if not single_injected_claim:
            raise FactualValidationError(
                "exactly_one_injected_claim failed: the conversation does not "
                "contain exactly one edited Agent turn"
            )
    else:
        _check_claim_present(
            metadata.injected_claim,
            target_turn.content,
            "Pre-existing Agent turn",
        )
        if target_turn.content != str(source_turn.get("content")):
            raise FactualValidationError(
                "Pre-existing defects must leave the Agent turn unchanged"
            )
        single_injected_claim = False

    if metadata.injection_status == "injected":
        incoherent = _has_clear_self_contradiction_or_incoherence(target_turn.content)
        if incoherent:
            raise FactualValidationError(
                "Injected conversation is incoherent: " + incoherent
            )
        side_effect = _procedural_side_effect(
            metadata.original_agent_claim, metadata.injected_claim
        )
        if side_effect:
            raise FactualValidationError(side_effect)
        screened = input_record.get("screened_candidate")
        if isinstance(screened, Mapping):
            expected_claim = screened.get("target_claim")
            if isinstance(expected_claim, str) and expected_claim:
                if metadata.original_agent_claim != expected_claim:
                    raise FactualValidationError(
                        "Injection did not use the screened target claim "
                        f"{expected_claim!r}"
                    )
            expected_evidence = screened.get("evidence_text")
            if isinstance(expected_evidence, str) and expected_evidence:
                if metadata.evidence_text != expected_evidence:
                    raise FactualValidationError(
                        "Injection evidence does not match the screened evidence "
                        f"{expected_evidence!r}"
                    )

    notes = (
        "Automated structural and structured-rule checks only. This is not "
        "semantic proof, human review, or Judge execution. Judges were not "
        "executed. The model's contradiction_explanation and risk flags are "
        "not independent proof of correctness. Human review or independent "
        "evidence validation is required before promotion to the final "
        "Golden Set. judge_evaluations on the golden row, when present, are "
        "the expected label for this verified contradiction. They are not an "
        "independent judge execution, and non_target_risk_assessment flags are "
        "not judge results."
    )
    if evidence_integrity == "NOT_VERIFIED":
        notes += (
            " Trusted evidence integrity is NOT_VERIFIED because at least one "
            "non-scenario evidence source could not be rebuilt from AB-CD files."
        )
    human_review_flags = input_record.get("human_review_flags") or sampled_params and sampled_params.get("human_review_flags")
    if isinstance(human_review_flags, list) and human_review_flags:
        notes += " Human review flags: " + "; ".join(str(x) for x in human_review_flags)

    checks = AutomatedChecks(
        exactly_one_injected_claim=single_injected_claim,
        evidence_reference_matches_exact=True,
        same_entity_and_attribute=assessment.same_entity_and_attribute,
        explicit_incompatible_value=assessment.explicit_incompatible_value,
        structured_contradiction_verified=assessment.verified,
        evidence_location_non_empty=bool(metadata.evidence_source_location.strip()),
        contradiction_explanation_non_empty=bool(
            metadata.contradiction_explanation.strip()
        ),
        scenario_unchanged=not metadata.scenario_modified
        and not metadata.modified_scenario_fields,
        turn_id_valid=metadata.injection_turn_id in turn_map,
        speaker_is_agent=target_turn.speaker == "AGENT",
        trusted_evidence_unmodified=evidence_unmodified,
        trusted_evidence_integrity=evidence_integrity,
        original_claim_supported=True,
        injected_claim_contradicted=assessment.verified,
        single_attribute_value_change=assessment.explicit_incompatible_value
        and assessment.same_entity_and_attribute,
        error_category_valid=True,
        output_structure_valid=True,
        minimal_perturbation=changed_turns
        == ([] if metadata.injection_status == "pre_existing" else [metadata.injection_turn_id]),
        evidence_classification=verification.classification,
    )

    judge_input = FactualJudgeInput(
        target_judge=FACTUAL_JUDGE,
        target_judge_definition=dict(judge_definition),
        conversation=modified_turns,
        scenario=dict(scenario),
        sop=sop,
        evidence=available_evidence,
    )
    validation_report = FactualValidationReport(
        automated_checks=checks,
        evidence_verification=EvidenceVerificationSummary(
            classification=verification.classification,
            entity_match=verification.entity_match,
            attribute_match=verification.attribute_match,
            context_match=verification.context_match,
            reason=verification.reason,
        ),
        notes=notes,
    )
    return FactualHallucinationGoldenRecordOutput(
        injection_metadata=metadata,
        judge_input=judge_input,
        validation_report=validation_report,
    )


# ---------------------------------------------------------------------------
# Rejection handling and metrics
# ---------------------------------------------------------------------------


_REJECTION_STATUS = "INJECTION_NOT_POSSIBLE"


@dataclass(frozen=True)
class PostGenerationFailureDisposition:
    """How a failed post-generation validation should be routed."""

    disposition: PostGenerationDisposition
    reason_code: str
    reason: str


_AMBIGUOUS_VALIDATION_MARKERS: tuple[str, ...] = (
    "ambiguous",
    "cannot be established confidently",
    "human review",
    "not a recognized single",
    "does not explicitly assert",
    "paraphrase or vague",
    "vague state",
    "only share generic subject",
    "generic subject",
    "no supported structured contradiction rule",
    "not verifiable",
    "cannot be verified",
    "neither confirms nor denies",
)


def classify_post_generation_failure(
    error_text: str,
) -> PostGenerationFailureDisposition:
    """Route a validation failure to REJECT or REVIEW.

    Only failures that indicate genuine ambiguity about the evidence or the
    claim are sent to REVIEW. Structural or evidence-mismatch failures are
    deterministic rejections.
    """

    normalized = error_text.casefold()
    if any(marker in normalized for marker in _AMBIGUOUS_VALIDATION_MARKERS):
        return PostGenerationFailureDisposition(
            disposition="REVIEW",
            reason_code="AMBIGUOUS_VALIDATION",
            reason=(
                "Post-generation validation failed in a way that requires human "
                f"review rather than an automatic rejection: {error_text.strip()}"
            ),
        )
    return PostGenerationFailureDisposition(
        disposition="REJECT",
        reason_code="POST_GENERATION_REJECTED",
        reason=(
            "Post-generation validation failed with a deterministic error: "
            f"{error_text.strip()}"
        ),
    )


def _payload_is_structured_rejection(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if payload.get("injection_status") == _REJECTION_STATUS:
        return True
    metadata = payload.get("injection_metadata")
    return (
        isinstance(metadata, Mapping)
        and metadata.get("injection_status") == _REJECTION_STATUS
    )


def _parse_possible_json(raw_text: str) -> Any | None:
    text = raw_text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end >= start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def is_injection_not_possible(payload: Any, raw_text: str = "") -> bool:
    """Recognize a structured rejection, never a token buried in other text.

    A successful JSON payload that happens to mention
    ``INJECTION_NOT_POSSIBLE`` in an explanation or claim is not a rejection.
    Unparsed text is a rejection only when it is the dedicated status token
    or a JSON object whose ``injection_status`` field has that value.
    """

    if _payload_is_structured_rejection(payload):
        return True
    if isinstance(payload, Mapping):
        return False
    parsed = _parse_possible_json(raw_text)
    if _payload_is_structured_rejection(parsed):
        return True
    if isinstance(parsed, Mapping):
        return False
    return raw_text.strip().upper() == _REJECTION_STATUS


def normalize_rejection(
    payload: Any,
    raw_text: str,
    conversation_id: str | int,
    sampled_params: Mapping[str, Any] | None = None,
) -> FactualRejectionRecord:
    """Build a validated rejection record without forcing a golden sample."""

    code: str = "OTHER"
    reason = "The model returned INJECTION_NOT_POSSIBLE."
    if isinstance(payload, Mapping):
        candidate_code = payload.get("rejection_code")
        if isinstance(candidate_code, str) and candidate_code in {
            "NO_TRUSTED_EVIDENCE",
            "UNSUPPORTED_ONLY",
            "CONFLICTING_EVIDENCE",
            "EXCLUDED_JUDGE_OVERLAP",
            "COHERENCE_IMPOSSIBLE",
            "OTHER",
        }:
            code = candidate_code
        candidate_reason = payload.get("rejection_reason")
        if isinstance(candidate_reason, str) and candidate_reason.strip():
            reason = candidate_reason.strip()
    return FactualRejectionRecord(
        conversation_id=conversation_id,
        target_judge=FACTUAL_JUDGE,
        injection_status="INJECTION_NOT_POSSIBLE",
        rejection_code=code,  # type: ignore[arg-type]
        rejection_reason=reason,
        raw_response=raw_text,
        sampled_params=dict(sampled_params or {}),
    )


FAILURE_REASON_CODES: tuple[str, ...] = (
    "MODEL_CONFIGURATION_ERROR",
    "NETWORK_ERROR",
    "INVALID_JSON",
    "OUTPUT_SCHEMA_INVALID",
    "ORIGINAL_CLAIM_NOT_GROUNDED",
    "MULTIPLE_ATTRIBUTE_CHANGE",
    "SCENARIO_MUTATED",
    "CONVERSATION_STRUCTURE_MUTATED",
    "EVIDENCE_REFERENCE_MISMATCH",
    "CONTRADICTION_NOT_VERIFIED",
    "VALIDATION_ERROR",
    "UNKNOWN",
)


def classify_failure_reason(failure: Mapping[str, Any]) -> str:
    """Return a stable category for one failed generation record.

    The category is derived from deterministic error text, never from an LLM
    self-report. A caller-supplied ``failure_reason_code`` wins when it is a
    known code.
    """

    explicit = failure.get("failure_reason_code")
    if isinstance(explicit, str) and explicit in FAILURE_REASON_CODES:
        return explicit
    if failure.get("model_error"):
        return "MODEL_CONFIGURATION_ERROR"
    if failure.get("network_error"):
        return "NETWORK_ERROR"

    raw_errors = failure.get("validation_errors")
    if isinstance(raw_errors, str):
        errors = [raw_errors]
    elif isinstance(raw_errors, Sequence):
        errors = [str(item) for item in raw_errors]
    else:
        errors = []
    text = " ".join(errors).casefold()
    if not text:
        return "UNKNOWN"

    checks: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "INVALID_JSON",
            (
                "jsondecodeerror",
                "expecting value",
                "expecting property name",
                "unterminated string",
                "invalid control character",
            ),
        ),
        (
            "OUTPUT_SCHEMA_INVALID",
            (
                "validation error",
                "input should be",
                "field required",
                "extra inputs are not permitted",
                "extra_forbidden",
            ),
        ),
        (
            "ORIGINAL_CLAIM_NOT_GROUNDED",
            (
                "original agent claim must state",
                "does not positively assert",
                "not an exact substring",
                "appears more than once",
                "explicit factual claim",
                "interrogative",
            ),
        ),
        (
            "MULTIPLE_ATTRIBUTE_CHANGE",
            (
                "extra factual attributes",
                "single-claim substitution",
                "exactly one injected claim",
                "more than one factual",
            ),
        ),
        (
            "SCENARIO_MUTATED",
            ("scenario_modified", "modified_scenario_fields"),
        ),
        (
            "CONVERSATION_STRUCTURE_MUTATED",
            (
                "only the single target agent turn",
                "speaker changed",
                "preserve the source turn count",
                "turn_ids must be strictly sequential",
                "customer turn",
                "action turn",
            ),
        ),
        (
            "EVIDENCE_REFERENCE_MISMATCH",
            (
                "no trusted evidence item matches",
                "conflicting trusted evidence",
                "exactly quote the trusted evidence",
                "trusted evidence does not match",
                "identifier evidence",
            ),
        ),
        (
            "CONTRADICTION_NOT_VERIFIED",
            (
                "structured contradiction validation failed",
                "not a recognized single",
                "does not explicitly assert",
                "asserts the same",
                "not a contradiction",
            ),
        ),
        ("VALIDATION_ERROR", ("failed validation",)),
    )
    for code, needles in checks:
        if any(needle in text for needle in needles):
            return code
    return "VALIDATION_ERROR"


def compute_quality_metrics(
    successes: Sequence[FactualHallucinationGoldenRecordOutput],
    rejections: Sequence[FactualRejectionRecord] = (),
    *,
    clean_sample_verdicts: Sequence[bool] | None = None,
    failures: Sequence[Mapping[str, Any]] | None = None,
    candidate_filter_exclusions: int | None = None,
    total_input_records: int | None = None,
    screening_counts: Mapping[str, int] | None = None,
    reviewed_sample_count: int | None = None,
) -> QualityMetrics:
    """Compute only metrics supported by actual records; leave others as None.

    Two denominators are deliberately kept separate:

    - ``injection_success_rate`` / ``injection_rejection_rate`` are the legacy
      rates over records that produced a golden success or an explicit
      rejection (they predate failure accounting).
    - ``overall_acceptance_rate`` divides by ``total_input_records``.
    - ``generation_acceptance_rate`` divides by records that actually reached
      generation (accepted + pre-existing + rejected + failed).

    Pre-existing defects are reported separately and are never counted as newly
    injected samples. ``failures=None`` means failure data was not supplied, so
    failure rates stay ``None`` rather than being reported as zero.
    """

    injected_successes = [
        record
        for record in successes
        if record.injection_metadata.injection_status == "injected"
    ]
    pre_existing = [
        record
        for record in successes
        if record.injection_metadata.injection_status == "pre_existing"
    ]
    legacy_total = len(successes) + len(rejections)

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    failure_count = len(failures) if failures is not None else None
    reached_generation = (
        len(injected_successes)
        + len(pre_existing)
        + len(rejections)
        + (failure_count or 0)
        if failures is not None
        else None
    )
    if total_input_records is not None:
        resolved_total = total_input_records
    elif failures is not None and candidate_filter_exclusions is not None:
        resolved_total = (reached_generation or 0) + candidate_filter_exclusions
    else:
        resolved_total = None

    failure_reasons: dict[str, int] = {}
    for failure in failures or ():
        code = classify_failure_reason(failure)
        failure_reasons[code] = failure_reasons.get(code, 0) + 1

    evidence_valid = sum(
        1
        for record in successes
        if record.validation_report.automated_checks.evidence_reference_matches_exact
        and record.validation_report.automated_checks.structured_contradiction_verified
    )
    single_error = sum(
        1
        for record in injected_successes
        if record.validation_report.automated_checks.exactly_one_injected_claim
    )
    minimal = sum(
        1
        for record in successes
        if record.validation_report.automated_checks.minimal_perturbation
    )
    localized = sum(
        1
        for record in successes
        if record.validation_report.automated_checks.turn_id_valid
        and record.validation_report.automated_checks.evidence_location_non_empty
    )

    evaluated_targets = [
        record.validation_report.target_judge_status
        for record in successes
        if record.validation_report.target_judge_status != "NOT_EVALUATED"
    ]
    target_detection = (
        rate(sum(1 for status in evaluated_targets if status == "FAIL"), len(evaluated_targets))
        if evaluated_targets
        else None
    )

    evaluated_non_targets = [
        record.validation_report.non_target_judges_status
        for record in successes
        if record.validation_report.non_target_judges_status != "NOT_EVALUATED"
    ]
    non_target_failure = (
        rate(
            sum(1 for status in evaluated_non_targets if status == "FAIL"),
            len(evaluated_non_targets),
        )
        if evaluated_non_targets
        else None
    )

    false_positive = (
        rate(
            sum(1 for verdict in clean_sample_verdicts if verdict),
            len(clean_sample_verdicts),
        )
        if clean_sample_verdicts
        else None
    )

    normalized_screening: dict[str, int] | None = None
    if screening_counts is not None:
        normalized_screening = {}
        for key, value in screening_counts.items():
            normalized_screening[str(key).strip().upper()] = int(value)

    def screening_value(key: str) -> int | None:
        if normalized_screening is None:
            return None
        return normalized_screening.get(key, 0)

    validation_attempts = (
        len(successes) + (failure_count or 0)
        if failures is not None
        else None
    )

    return QualityMetrics(
        injected_sample_count=len(injected_successes),
        pre_existing_sample_count=len(pre_existing),
        rejection_count=len(rejections),
        failed_generation_count=failure_count,
        candidate_filter_exclusion_count=candidate_filter_exclusions,
        screening_eligible_count=screening_value("ELIGIBLE"),
        screening_reject_count=screening_value("REJECT"),
        screening_review_count=screening_value("REVIEW"),
        screening_not_run_count=screening_value("NOT_RUN"),
        reviewed_sample_count=reviewed_sample_count,
        total_input_records=resolved_total,
        generation_attempt_count=reached_generation,
        validation_attempt_count=validation_attempts,
        validation_success_count=(
            len(successes) if validation_attempts is not None else None
        ),
        validation_success_rate=(
            rate(len(successes), validation_attempts)
            if validation_attempts
            else None
        ),
        failure_reasons=failure_reasons,
        # Legacy rates, retained for existing consumers.
        injection_success_rate=rate(len(injected_successes), legacy_total),
        pre_existing_rate=rate(len(pre_existing), legacy_total),
        injection_rejection_rate=rate(len(rejections), legacy_total),
        # Run-level rates with explicit denominators.
        overall_acceptance_rate=(
            rate(len(injected_successes), resolved_total)
            if resolved_total is not None
            else None
        ),
        rejection_rate=(
            rate(len(rejections), resolved_total)
            if resolved_total is not None
            else None
        ),
        failure_rate=(
            rate(failure_count, resolved_total)
            if resolved_total is not None and failure_count is not None
            else None
        ),
        generation_acceptance_rate=(
            rate(len(injected_successes), reached_generation)
            if reached_generation is not None
            else None
        ),
        evidence_validity_rate=rate(evidence_valid, len(successes)),
        single_error_compliance_rate=rate(single_error, len(injected_successes)),
        minimal_perturbation_rate=rate(minimal, len(successes)),
        target_judge_detection_rate=target_detection,
        false_positive_rate_clean_samples=false_positive,
        non_target_judge_failure_rate=non_target_failure,
        evidence_localization_accuracy=rate(localized, len(successes)),
    )


def compute_run_metrics(
    *,
    accepted_rows: Sequence[Mapping[str, Any]] = (),
    rejections: Sequence[Mapping[str, Any]] = (),
    failures: Sequence[Mapping[str, Any]] | None = None,
    reviews: Sequence[Mapping[str, Any]] = (),
    screening_records: Sequence[Mapping[str, Any]] | None = None,
    candidate_filter_exclusions: int | None = None,
    total_input_records: int | None = None,
) -> QualityMetrics:
    """Compute operational metrics directly from JSONL-shaped run artefacts.

    This is the reporting counterpart to :func:`compute_quality_metrics`. It
    intentionally reports only what the stored rows can prove. Per-record
    automated-check rates that require the private validation report stay
    ``None`` instead of being inferred.
    """

    injected = 0
    pre_existing = 0
    for row in accepted_rows:
        metadata = row.get("injection_metadata")
        status = (
            metadata.get("injection_status") if isinstance(metadata, Mapping) else None
        )
        if status == "injected":
            injected += 1
        elif status == "pre_existing":
            pre_existing += 1

    failure_reasons: dict[str, int] = {}
    for failure in failures or ():
        code = classify_failure_reason(failure)
        failure_reasons[code] = failure_reasons.get(code, 0) + 1

    screening_counts: dict[str, int] | None = None
    if screening_records:
        screening_counts = {}
        for row in screening_records:
            decision = str(row.get("decision", "NOT_RUN")).strip().upper()
            screening_counts[decision] = screening_counts.get(decision, 0) + 1

    accepted_count = len(accepted_rows)
    rejection_count = len(rejections)
    failure_count = len(failures) if failures is not None else None
    review_count = len(reviews)
    screening_rejections = (screening_counts or {}).get("REJECT", 0)
    screening_reviews = (screening_counts or {}).get("REVIEW", 0)
    validation_attempts = (
        accepted_count + failure_count if failure_count is not None else None
    )

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    if total_input_records is None:
        resolved_total = (
            accepted_count
            + rejection_count
            + (failure_count or 0)
            + review_count
            + screening_rejections
            + screening_reviews
        )
    else:
        resolved_total = total_input_records

    reached_generation = accepted_count + rejection_count + (failure_count or 0)
    legacy_total = accepted_count + rejection_count

    return QualityMetrics(
        injected_sample_count=injected,
        pre_existing_sample_count=pre_existing,
        rejection_count=rejection_count,
        failed_generation_count=failure_count,
        candidate_filter_exclusion_count=candidate_filter_exclusions,
        screening_eligible_count=(
            screening_counts.get("ELIGIBLE", 0)
            if screening_counts is not None
            else None
        ),
        screening_reject_count=(
            screening_counts.get("REJECT", 0)
            if screening_counts is not None
            else None
        ),
        screening_review_count=(
            screening_counts.get("REVIEW", 0)
            if screening_counts is not None
            else None
        ),
        screening_not_run_count=(
            screening_counts.get("NOT_RUN", 0)
            if screening_counts is not None
            else None
        ),
        reviewed_sample_count=review_count,
        validation_attempt_count=validation_attempts,
        validation_success_count=(
            accepted_count if validation_attempts is not None else None
        ),
        validation_success_rate=(
            rate(accepted_count, validation_attempts)
            if validation_attempts
            else None
        ),
        total_input_records=resolved_total,
        generation_attempt_count=reached_generation,
        failure_reasons=failure_reasons,
        injection_success_rate=rate(injected, legacy_total),
        pre_existing_rate=rate(pre_existing, legacy_total),
        injection_rejection_rate=rate(rejection_count, legacy_total),
        overall_acceptance_rate=rate(injected, resolved_total),
        rejection_rate=rate(rejection_count, resolved_total),
        failure_rate=(
            rate(failure_count, resolved_total)
            if failure_count is not None
            else None
        ),
        generation_acceptance_rate=rate(injected, reached_generation),
    )
