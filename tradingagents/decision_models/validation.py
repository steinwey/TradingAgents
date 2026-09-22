"""Shared numeric checks for decision responses, independent of SDK types."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from tradingagents.decision_models.errors import JevMalformedResponseError

# Absolute tolerance, shared by generic distributions and agent direction checks.
PROBABILITY_SUM_ABS_TOL = 1e-6


def finite_float(value: Any, *, what: str, coerce: bool = False) -> float:
    """Validate a finite number; SDK normalization may also coerce numeric text."""
    if isinstance(value, bool) or value is None or (
        not coerce and not isinstance(value, (int, float))
    ):
        raise JevMalformedResponseError(f"{what} is not a finite number: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise JevMalformedResponseError(f"{what} is not a finite number: {value!r}") from exc
    if not math.isfinite(number):
        raise JevMalformedResponseError(f"{what} is not a finite number: {value!r}")
    return number


def unit_interval(value: Any, *, what: str, coerce: bool = False) -> float:
    number = finite_float(value, what=what, coerce=coerce)
    if number < 0.0 or number > 1.0:
        raise JevMalformedResponseError(f"{what} must be in [0, 1], got {number}")
    return number


def rubric_score(value: Any, *, what: str, maximum: int) -> float:
    """Validate an agent score without clipping or changing its numeric value."""
    number = finite_float(value, what=what)
    if number < 0.0 or number > maximum:
        raise JevMalformedResponseError(f"{what} must be in [0, {maximum}], got {number}")
    return number


def probability_sum(values: Iterable[float], *, what: str = "probabilities") -> None:
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_ABS_TOL):
        raise JevMalformedResponseError(
            f"{what} must sum to 1 (±{PROBABILITY_SUM_ABS_TOL}), got {total}"
        )
