"""JevAgent: typed market prediction via TypeSafe Jev (System One).

This agent is not a LangGraph node and does not call tools, write reports, or
place orders. It maps a compact :class:`JevMarketState` onto three Jev
questions (direction Choice, signal-strength Score, market-risk Score) and
returns a :class:`JevTradingSignal`.

Wire it into ``TradingAgentsGraph`` in a later change; do not import this
module from ``graph/setup.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Explicit re-exports preserve the original public import paths.
from tradingagents.agents.jev.dates import (
    parse_analysis_date as parse_analysis_date,
    parse_asof_date as parse_asof_date,
)
from tradingagents.agents.jev.market_data import normalize_market_value as normalize_market_value
from tradingagents.agents.jev.models import (
    MARKET_RISK_LEVELS as MARKET_RISK_LEVELS,
    SIGNAL_STRENGTH_LEVELS as SIGNAL_STRENGTH_LEVELS,
    Direction as Direction,
    JevMarketState as JevMarketState,
    JevTradingSignal as JevTradingSignal,
)
from tradingagents.agents.jev.point_in_time import (
    assert_point_in_time as assert_point_in_time,
    observation_dates as observation_dates,
)
from tradingagents.decision_models.errors import (
    JevError,
    JevLookAheadError as JevLookAheadError,
    JevMalformedResponseError,
)
from tradingagents.decision_models.jev_client import JevClient
from tradingagents.decision_models.types import JevChoiceResult, JevEvaluateResult, JevScoreResult
from tradingagents.decision_models.validation import (
    PROBABILITY_SUM_ABS_TOL,
    probability_sum,
    rubric_score,
    unit_interval as _validate_unit_interval,
)

PREDICTION_HORIZONS = ("5m", "30m", "1h", "1d", "5d")  # 允许的预测窗口
DEFAULT_PREDICTION_HORIZON = "1d"

# 发给 Jev Choice 的方向定义：窗口内预期净收益为正 / 说不清 / 为负
DIRECTION_CRITERIA: dict[str, str] = {
    "BULLISH": "Expected net positive return over the prediction horizon.",
    "NEUTRAL": "Expected move is too small or mixed to call a direction.",
    "BEARISH": "Expected net negative return over the prediction horizon.",
}

def _horizon_label(horizon: str) -> str:
    return {
        "5m": "5 minutes",
        "30m": "30 minutes",
        "1h": "1 hour",
        "1d": "1 trading day",
        "5d": "5 trading days",
    }[horizon]


def label_for_score(score: float, levels: tuple[str, ...]) -> str:
    """Map a 0-based expected score onto the nearest rubric label.

    Out-of-range or non-finite scores are malformed; they are not clipped.
    """
    if not levels:
        raise JevError("score rubric is empty")
    number = rubric_score(score, what="score", maximum=len(levels) - 1)
    return levels[int(round(number))]


def build_jev_questions(horizon: str) -> dict[str, Any]:
    window = _horizon_label(horizon)
    pit = (
        "Use only information in the supplied state, which is already truncated "
        "to the analysis timestamp. Do not assume any price, news, or filing "
        "from after that cutoff."
    )
    return {
        "direction": {
            "type": "choice",
            "instructions": (
                f"Given the supplied market state available up to the analysis timestamp, "
                f"classify the expected direction of the asset over the next {window}. {pit}"
            ),
            "criteria": dict(DIRECTION_CRITERIA),
        },
        "signal_strength": {
            "type": "score",
            "instructions": (
                f"Score how strong the directional signal is over the next {window}. "
                f"0 is {SIGNAL_STRENGTH_LEVELS[0]}, "
                f"{len(SIGNAL_STRENGTH_LEVELS) - 1} is {SIGNAL_STRENGTH_LEVELS[-1]}. {pit}"
            ),
            "criteria": list(SIGNAL_STRENGTH_LEVELS),
        },
        "market_risk": {
            "type": "score",
            "instructions": (
                f"Score predictive / market risk over the next {window} "
                "(not portfolio risk). "
                f"0 is {MARKET_RISK_LEVELS[0]}, "
                f"{len(MARKET_RISK_LEVELS) - 1} is {MARKET_RISK_LEVELS[-1]}. {pit}"
            ),
            "criteria": list(MARKET_RISK_LEVELS),
        },
    }


def _validate_direction_choice(
    direction: JevChoiceResult,
) -> tuple[str, dict[str, float], float | None]:
    if not isinstance(direction.probabilities, Mapping) or not direction.probabilities:
        raise JevMalformedResponseError("direction probabilities must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key, raw in direction.probabilities.items():
        label = str(key).upper()
        if label in normalized:
            raise JevMalformedResponseError(
                f"duplicate direction label {label!r} after case normalization"
            )
        normalized[label] = _validate_unit_interval(
            raw, what=f"direction probability {key!r}"
        )
    expected = set(DIRECTION_CRITERIA)
    if set(normalized) != expected:
        raise JevMalformedResponseError(
            "direction probabilities must contain exactly BULLISH, NEUTRAL, and BEARISH"
        )
    probability_sum(normalized.values(), what="direction probabilities")
    choice = str(direction.choice).upper()
    if choice not in expected:
        raise JevMalformedResponseError(f"unsupported direction {direction.choice!r}")
    max_p = max(normalized.values())
    tied = {
        name for name, prob in normalized.items()
        if abs(prob - max_p) <= PROBABILITY_SUM_ABS_TOL
    }
    if choice not in tied:
        raise JevMalformedResponseError(
            f"direction {choice!r} is not a maximum-probability choice"
        )
    confidence = direction.confidence
    if confidence is not None:
        confidence = _validate_unit_interval(confidence, what="direction confidence")
    return choice, normalized, confidence


def _validate_agent_score(result: JevScoreResult, *, name: str, levels: tuple[str, ...]) -> float:
    return rubric_score(result.score, what=name, maximum=len(levels) - 1)


def signal_from_evaluation(
    market_state: JevMarketState,
    horizon: str,
    result: JevEvaluateResult,
) -> JevTradingSignal:
    if "direction" not in result.choices:
        raise JevMalformedResponseError("Jev response is missing the 'direction' choice")
    if "signal_strength" not in result.scores:
        raise JevMalformedResponseError("Jev response is missing the 'signal_strength' score")
    if "market_risk" not in result.scores:
        raise JevMalformedResponseError("Jev response is missing the 'market_risk' score")

    direction = result.choices["direction"]
    label, probabilities, confidence = _validate_direction_choice(direction)
    strength_score = _validate_agent_score(
        result.scores["signal_strength"], name="signal_strength", levels=SIGNAL_STRENGTH_LEVELS
    )
    risk_score = _validate_agent_score(
        result.scores["market_risk"], name="market_risk", levels=MARKET_RISK_LEVELS
    )
    return JevTradingSignal(
        ticker=market_state.ticker,
        timestamp=market_state.analysis_date,
        horizon=horizon,
        direction=label,  # type: ignore[arg-type]
        direction_probabilities=probabilities,
        signal_strength=strength_score,
        signal_strength_label=label_for_score(strength_score, SIGNAL_STRENGTH_LEVELS),
        market_risk=risk_score,
        market_risk_label=label_for_score(risk_score, MARKET_RISK_LEVELS),
        confidence=confidence,
        model=result.model,
        latency_ms=result.latency_ms,
    )


class JevAgent:
    """Thin System One wrapper: market state in, :class:`JevTradingSignal` out."""

    def __init__(
        self,
        client: JevClient | None = None,
        *,
        prediction_horizon: str = DEFAULT_PREDICTION_HORIZON,
        config: Mapping[str, Any] | None = None,
    ):
        if prediction_horizon not in PREDICTION_HORIZONS:
            raise JevError(
                f"unsupported prediction_horizon {prediction_horizon!r}; "
                f"expected one of {PREDICTION_HORIZONS}"
            )
        self.prediction_horizon = prediction_horizon
        self.client = client or JevClient.from_config(config)

    def evaluate(
        self,
        market_state: JevMarketState | Mapping[str, Any],
        *,
        prediction_horizon: str | None = None,
    ) -> JevTradingSignal:
        horizon = prediction_horizon or self.prediction_horizon
        if horizon not in PREDICTION_HORIZONS:
            raise JevError(
                f"unsupported prediction_horizon {horizon!r}; "
                f"expected one of {PREDICTION_HORIZONS}"
            )
        state = (
            market_state
            if isinstance(market_state, JevMarketState)
            else JevMarketState.model_validate(dict(market_state))
        )
        payload = state.to_jev_state(horizon=horizon)
        assert_point_in_time(state, payload=payload)
        result = self.client.evaluate(payload, build_jev_questions(horizon))
        return signal_from_evaluation(state, horizon, result)

    def __call__(self, market_state: JevMarketState | Mapping[str, Any]) -> JevTradingSignal:
        return self.evaluate(market_state)


def create_jev_agent(
    client: JevClient | None = None,
    *,
    prediction_horizon: str = DEFAULT_PREDICTION_HORIZON,
    config: Mapping[str, Any] | None = None,
) -> JevAgent:
    """Factory matching other agents' ``create_*`` style. Not a graph node."""
    return JevAgent(client, prediction_horizon=prediction_horizon, config=config)
