"""Compatibility checks for shared validation and the normalized date scan."""

from datetime import date, datetime

import pandas as pd
import pytest

from tradingagents.agents.jev import jev_agent as agent, point_in_time
from tradingagents.decision_models.errors import JevMalformedResponseError
from tradingagents.decision_models.jev_client import normalize_system_one_response
from tradingagents.decision_models.types import JevChoiceResult


@pytest.mark.unit
@pytest.mark.parametrize("raw", [
    'Date,close\n2026-09-01,120',
    '{"data": [{"Date": "2026-09-01", "close": 120}]}',
    pd.DataFrame({"close": [120]}, index=pd.to_datetime(["2026-09-01"])),
    [{"Date": datetime(2026, 9, 1), "close": 120}],
    {"2026-09-01": [{"close": 120}]},
])
def test_raw_and_normalized_date_scans_agree(raw):
    normalized = agent.normalize_market_value(raw, field="price_data")
    assert agent.observation_dates(raw, field="price_data") == point_in_time._observation_dates(
        normalized, field="price_data", normalized=True,
    )


@pytest.mark.unit
def test_normalized_payload_is_audited_without_reparsing(monkeypatch):
    state = agent.JevMarketState(
        ticker="NVDA", analysis_date="2026-09-01",
        price_data={"data": 'Date,close\n2026-09-02,120'},
    )
    payload = state.to_jev_state(horizon="1d")

    def unexpected_parse(*args, **kwargs):
        pytest.fail("already normalized data was parsed again")

    for name in ("_records_from_csv", "_parse_json_text", "_records_from_frame"):
        monkeypatch.setattr(point_in_time, name, unexpected_parse)
    monkeypatch.setattr(point_in_time, "get_current_date", lambda: "2026-09-03")
    with pytest.raises(agent.JevLookAheadError, match="after analysis_date"):
        agent.assert_point_in_time(state, payload=payload)
    payload["price_data"]["data"][0]["Date"] = "2026-09-01"
    agent.assert_point_in_time(state, payload=payload)


@pytest.mark.unit
def test_standalone_date_scan_still_inherits_outer_date():
    assert agent.observation_dates(
        [{"close": 120}], field="price_data", row_date=date(2026, 9, 1),
    ) == [date(2026, 9, 1)]


@pytest.mark.unit
def test_client_numeric_text_remains_distinct_from_agent_input():
    probabilities = {"BULLISH": "0.7", "NEUTRAL": "0.2", "BEARISH": "0.1"}
    result = normalize_system_one_response({"answers": {"direction": {
        "type": "choice", "choice": "BULLISH",
        "probabilities": probabilities, "confidence": "0.8",
    }}})
    assert agent._validate_direction_choice(result.choices["direction"])[2] == 0.8
    with pytest.raises(JevMalformedResponseError, match="not a finite number"):
        agent._validate_direction_choice(JevChoiceResult(
            choice="BULLISH", probabilities=probabilities, confidence="0.8",
        ))


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_shared_checks_keep_rejecting_invalid_confidence(value):
    answer = {"type": "choice", "choice": "BULLISH", "confidence": value,
              "probabilities": {"BULLISH": 0.7, "NEUTRAL": 0.2, "BEARISH": 0.1}}
    with pytest.raises(JevMalformedResponseError):
        normalize_system_one_response({"answers": {"direction": answer}})
    with pytest.raises(JevMalformedResponseError):
        agent._validate_direction_choice(JevChoiceResult(
            choice="BULLISH", probabilities=answer["probabilities"], confidence=value,
        ))


@pytest.mark.unit
def test_split_modules_keep_public_imports_and_model_schema():
    from tradingagents.agents import jev
    from tradingagents.agents.jev import dates, market_data, models

    assert jev.JevMarketState is agent.JevMarketState is models.JevMarketState
    assert jev.JevTradingSignal is agent.JevTradingSignal is models.JevTradingSignal
    assert agent.normalize_market_value is market_data.normalize_market_value
    assert agent.parse_asof_date is dates.parse_asof_date
    assert agent.parse_analysis_date is dates.parse_analysis_date
    assert agent.assert_point_in_time is point_in_time.assert_point_in_time
    assert agent.observation_dates is point_in_time.observation_dates
    assert models.JevMarketState.model_json_schema()["title"] == "JevMarketState"
    state = agent.JevMarketState(ticker="NVDA", analysis_date="2026-09-01")
    assert models.JevMarketState.model_validate_json(state.model_dump_json()) == state


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, "2", float("nan"), float("inf"), -1, 5])
def test_score_entry_points_keep_the_same_validation(value):
    from tradingagents.decision_models.types import JevScoreResult

    with pytest.raises(JevMalformedResponseError) as label_error:
        agent.label_for_score(value, agent.SIGNAL_STRENGTH_LEVELS)
    with pytest.raises(JevMalformedResponseError) as signal_error:
        agent._validate_agent_score(
            JevScoreResult(score=value, probabilities={}),
            name="score", levels=agent.SIGNAL_STRENGTH_LEVELS,
        )
    assert str(label_error.value) == str(signal_error.value)
