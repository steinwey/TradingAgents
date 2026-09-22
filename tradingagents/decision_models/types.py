"""Provider-agnostic result types for System One / Jev evaluations.

JevAgent and future graph nodes consume these, never TypeSafe SDK objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JevChoiceResult:
    """Normalized Choice answer: selected label plus full distribution.

    ``probabilities`` values are finite and in ``[0, 1]`` and must sum to 1
    within :data:`~tradingagents.decision_models.jev_client.PROBABILITY_SUM_ABS_TOL`.
    ``confidence``, when present, is in ``[0, 1]``.
    """

    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None


@dataclass(frozen=True)
class JevScoreResult:
    """Normalized Score answer: expected score on the 0-based rubric.

    ``score`` is finite. ``probabilities`` follow the same unit-interval and
    sum-to-one rules as :class:`JevChoiceResult`. Range limits such as
    ``[0, 4]`` are agent-specific and are not applied here.
    """

    score: float
    probabilities: dict[str, float]
    confidence: float | None = None
    legend: dict[str, str] | None = None


@dataclass(frozen=True)
class JevNoulResult:
    """Normalized Noul answer: P(statement is true) in [0, 1]."""

    noul: float


@dataclass(frozen=True)
class JevEvaluateResult:
    """One System One response, split by question type and keyed by question name."""

    choices: dict[str, JevChoiceResult] = field(default_factory=dict)
    scores: dict[str, JevScoreResult] = field(default_factory=dict)
    nouls: dict[str, JevNoulResult] = field(default_factory=dict)
    model: str | None = None
    latency_ms: float | None = None
    raw_answers: dict[str, object] = field(default_factory=dict)
