"""Unit tests for JevAgent. Uses a mock JevClient; no live API calls."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tradingagents.agents.jev.jev_agent import (
    DEFAULT_PREDICTION_HORIZON,
    MARKET_RISK_LEVELS,
    SIGNAL_STRENGTH_LEVELS,
    JevAgent,
    JevMarketState,
    create_jev_agent,
    normalize_market_value,
    parse_asof_date,
    signal_from_evaluation,
)
from tradingagents.decision_models.errors import JevError, JevLookAheadError, JevMalformedResponseError, JevNotConfiguredError
from tradingagents.decision_models.types import (
    JevChoiceResult,
    JevEvaluateResult,
    JevScoreResult,
)


def _evaluation(**overrides) -> JevEvaluateResult:
    payload = {
        "choices": {
            "direction": JevChoiceResult(
                choice="BULLISH",
                probabilities={"BULLISH": 0.70, "NEUTRAL": 0.20, "BEARISH": 0.10},
                confidence=0.76,
            )
        },
        "scores": {
            "signal_strength": JevScoreResult(
                score=3.0,
                probabilities={"0": 0.0, "1": 0.05, "2": 0.1, "3": 0.7, "4": 0.15},
                confidence=0.81,
            ),
            "market_risk": JevScoreResult(
                score=2.0,
                probabilities={"0": 0.05, "1": 0.15, "2": 0.6, "3": 0.15, "4": 0.05},
                confidence=0.72,
            ),
        },
        "model": "jev-latest",
        "latency_ms": 18.0,
    }
    payload.update(overrides)
    return JevEvaluateResult(**payload)


def _state(**overrides) -> JevMarketState:
    payload = {
        "ticker": "NVDA",
        "analysis_date": "2026-09-01",
        "price_data": [{"Date": "2026-09-01", "close": 120.0}],
        "returns": {"1d": 0.01},
        "volume": 1_000_000,
        "technical_indicators": {"rsi": 55.0},
    }
    payload.update(overrides)
    return JevMarketState.model_validate(payload)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.evaluate.return_value = _evaluation()
    return client


@pytest.mark.unit
def test_agent_reads_market_state_and_returns_stable_schema(mock_client):
    agent = create_jev_agent(client=mock_client)
    signal = agent(_state())

    assert signal.ticker == "NVDA"
    assert signal.timestamp == "2026-09-01"
    assert signal.horizon == DEFAULT_PREDICTION_HORIZON == "1d"
    assert signal.direction == "BULLISH"
    assert signal.direction_probabilities == {
        "BULLISH": 0.70,
        "NEUTRAL": 0.20,
        "BEARISH": 0.10,
    }
    assert signal.signal_strength == pytest.approx(3.0)
    assert signal.signal_strength_label == "STRONG"
    assert signal.signal_strength_levels == SIGNAL_STRENGTH_LEVELS
    assert signal.market_risk == pytest.approx(2.0)
    assert signal.market_risk_label == "MEDIUM"
    assert signal.market_risk_levels == MARKET_RISK_LEVELS
    assert signal.confidence == pytest.approx(0.76)
    assert signal.model == "jev-latest"
    assert signal.latency_ms == pytest.approx(18.0)

    sent_state, sent_questions = mock_client.evaluate.call_args.args
    assert sent_state["ticker"] == "NVDA"
    assert sent_state["point_in_time_cutoff"] == "2026-09-01"
    assert sent_state["prediction_horizon"] == "1d"
    assert sent_questions["direction"]["type"] == "choice"
    assert set(sent_questions["direction"]["criteria"]) == {"BULLISH", "NEUTRAL", "BEARISH"}
    assert sent_questions["signal_strength"]["criteria"] == list(SIGNAL_STRENGTH_LEVELS)
    assert "next 1 trading day" in sent_questions["direction"]["instructions"]


@pytest.mark.unit
def test_confidence_is_direction_choice_confidence_not_a_second_metric():
    result = _evaluation()
    signal = signal_from_evaluation(_state(), "1d", result)
    assert signal.confidence == result.choices["direction"].confidence
    assert signal.confidence != result.scores["signal_strength"].confidence


@pytest.mark.unit
def test_horizon_override():
    client = MagicMock()
    client.evaluate.return_value = _evaluation()
    agent = JevAgent(client, prediction_horizon="5d")
    signal = agent.evaluate(_state())
    assert signal.horizon == "5d"
    assert "next 5 trading days" in client.evaluate.call_args.args[1]["direction"]["instructions"]


@pytest.mark.unit
def test_unsupported_horizon_fails_loudly():
    with pytest.raises(JevError, match="prediction_horizon"):
        create_jev_agent(client=MagicMock(), prediction_horizon="2w")


@pytest.mark.unit
def test_future_analysis_date_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="after today"):
        agent(_state(analysis_date="2099-01-01"))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_price_bar_after_cutoff_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(
            _state(
                analysis_date="2026-09-01",
                price_data=[
                    {"Date": "2026-09-01", "close": 1},
                    {"Date": "2026-09-02", "close": 2},
                ],
            )
        )
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_csv_price_data_after_cutoff_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(
            _state(
                analysis_date="2026-09-01",
                price_data="Date,close\n2026-09-02,120",
            )
        )
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_market_context_snapshot_after_cutoff_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(
            _state(
                analysis_date="2026-09-01",
                market_context={"date": "2026-09-02", "close": 120},
            )
        )
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_csv_price_data_on_cutoff_is_allowed(mock_client):
    agent = create_jev_agent(client=mock_client)
    agent(_state(price_data="Date,close\n2026-09-01,120"))
    mock_client.evaluate.assert_called_once()


@pytest.mark.unit
def test_unstructured_price_data_string_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="unstructured text"):
        agent(_state(price_data="spot up tomorrow"))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_analyst_report_mentioning_a_future_date_is_not_a_leak(mock_client):
    agent = create_jev_agent(client=mock_client)
    agent(_state(market_report="Catalyst calendar lists 2026-09-02 as an event date."))
    mock_client.evaluate.assert_called_once()
    sent_state = mock_client.evaluate.call_args.args[0]
    assert "look_ahead_warning" in sent_state
    assert "market_report" in sent_state["analyst_reports"]


@pytest.mark.unit
def test_dated_returns_series_after_cutoff_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(_state(returns={"2026-09-01": 0.01, "2026-09-02": 0.02}))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_datetime_bar_after_cutoff_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(
            _state(
                price_data=[{"Date": datetime(2026, 9, 2, 15, 0, 0), "close": 120}],
            )
        )
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_text_reports_are_forwarded_with_an_explicit_warning(mock_client):
    agent = create_jev_agent(client=mock_client)
    agent(_state(market_report="RSI oversold; no dated future claims."))
    sent_state = mock_client.evaluate.call_args.args[0]
    assert "look_ahead_warning" in sent_state
    assert "market_report" in sent_state["analyst_reports"]


@pytest.mark.unit
def test_missing_direction_answer_fails():
    result = _evaluation(choices={})
    with pytest.raises(JevError, match="direction"):
        signal_from_evaluation(_state(), "1d", result)


@pytest.mark.unit
def test_create_jev_agent_loads_project_config(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_JEV_MODEL", "jev-from-project")
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL", raising=False)
    agent = create_jev_agent()
    assert agent.client.model == "jev-from-project"


@pytest.mark.unit
def test_missing_api_key_surfaces_from_agent(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_AI_API_KEY", raising=False)
    agent = create_jev_agent()
    with pytest.raises(JevNotConfiguredError):
        agent(_state())


def _assert_json_safe(payload: dict) -> None:
    json.dumps(payload, allow_nan=False)


@pytest.mark.unit
def test_dataframe_payload_is_json_safe_and_does_not_mutate_caller(mock_client):
    frame = pd.DataFrame(
        {"close": [np.float64(120.0)], "volume": [np.int64(1_000)]},
        index=pd.DatetimeIndex(["2026-09-01"], name="Date"),
    )
    original = frame.copy()
    agent = create_jev_agent(client=mock_client)
    agent(_state(price_data=frame))

    mock_client.evaluate.assert_called_once()
    sent = mock_client.evaluate.call_args.args[0]
    _assert_json_safe(sent)
    assert sent["price_data"][0]["close"] == 120.0
    assert "Date" in sent["price_data"][0]
    assert str(sent["price_data"][0]["Date"]).startswith("2026-09-01")
    pd.testing.assert_frame_equal(frame, original)
    assert list(frame.columns) == ["close", "volume"]


@pytest.mark.unit
def test_datetime_records_payload_is_json_safe(mock_client):
    agent = create_jev_agent(client=mock_client)
    agent(
        _state(
            price_data=[{"Date": datetime(2026, 9, 1, 16, 0, 0), "close": 120.0}],
        )
    )
    sent = mock_client.evaluate.call_args.args[0]
    _assert_json_safe(sent)
    assert sent["price_data"][0]["Date"] == "2026-09-01T16:00:00"


@pytest.mark.unit
def test_same_day_afternoon_bar_is_allowed_on_inclusive_cutoff(mock_client):
    agent = create_jev_agent(client=mock_client)
    agent(
        _state(
            analysis_date="2026-09-01",
            price_data=[{"Date": "2026-09-01T16:00:00", "close": 120}],
        )
    )
    mock_client.evaluate.assert_called_once()


@pytest.mark.unit
def test_future_datetime_index_is_rejected(mock_client):
    frame = pd.DataFrame({"close": [120.0]}, index=pd.DatetimeIndex(["2026-09-02"]))
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(_state(analysis_date="2026-09-01", price_data=frame))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_legal_datetime_index_is_preserved_in_payload(mock_client):
    frame = pd.DataFrame({"close": [118.5]}, index=pd.DatetimeIndex(["2026-09-01"]))
    agent = create_jev_agent(client=mock_client)
    agent(_state(price_data=frame))
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["price_data"][0]["close"] == 118.5
    assert str(sent["price_data"][0]["Date"]).startswith("2026-09-01")
    _assert_json_safe(sent)


@pytest.mark.unit
def test_undated_price_table_is_rejected(mock_client):
    frame = pd.DataFrame({"close": [120.0]})
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="no date column or date index"):
        agent(_state(price_data=frame))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_range_index_is_not_treated_as_dates(mock_client):
    frame = pd.DataFrame({"close": [120.0]}, index=pd.RangeIndex(1, 2, name="Date"))
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="no date column or date index"):
        agent(_state(price_data=frame))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_date_column_and_index_must_agree(mock_client):
    frame = pd.DataFrame(
        {"Date": ["2026-08-01"], "close": [120.0]},
        index=pd.DatetimeIndex(["2026-08-02"]),
    )
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="conflicting date column"):
        agent(_state(price_data=frame))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_matching_date_column_and_index_are_kept(mock_client):
    frame = pd.DataFrame(
        {"Date": ["2026-09-01"], "close": [120.0]},
        index=pd.DatetimeIndex(["2026-09-01"]),
    )
    agent = create_jev_agent(client=mock_client)
    agent(_state(price_data=frame))
    sent = mock_client.evaluate.call_args.args[0]
    row = sent["price_data"][0]
    assert row["Date"] == "2026-09-01"
    assert row["close"] == 120.0
    assert str(row["timestamp"]).startswith("2026-09-01")
    _assert_json_safe(sent)


@pytest.mark.unit
def test_duplicate_dataframe_date_columns_are_rejected(mock_client):
    frame = pd.DataFrame(
        [["2026-09-02", "2026-09-01", 120]],
        columns=["Date", "Date", "close"],
    )
    original = frame.copy()
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="duplicate column"):
        agent(_state(analysis_date="2026-09-01", price_data=frame))
    mock_client.evaluate.assert_not_called()
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.unit
def test_duplicate_dataframe_close_columns_are_rejected(mock_client):
    frame = pd.DataFrame(
        [["2026-09-01", 100, 110]],
        columns=["Date", "close", "close"],
    )
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="duplicate column"):
        agent(_state(price_data=frame))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_duplicate_dataframe_columns_differing_by_case_are_rejected(mock_client):
    frame = pd.DataFrame(
        [["2026-09-01", 100, 110]],
        columns=["Date", "close", "Close"],
    )
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="duplicate column"):
        agent(_state(price_data=frame))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_duplicate_dataframe_columns_differing_by_whitespace_are_rejected(mock_client):
    frame = pd.DataFrame(
        [["2026-09-01", 100, 110]],
        columns=["Date", "close", " close "],
    )
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="duplicate column"):
        agent(_state(price_data=frame))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_unique_dataframe_columns_are_accepted(mock_client):
    frame = pd.DataFrame(
        {"Date": ["2026-09-01"], "open": [118.0], "close": [120.0]},
    )
    original = frame.copy()
    agent = create_jev_agent(client=mock_client)
    agent(_state(price_data=frame))
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["price_data"] == [{"Date": "2026-09-01", "open": 118.0, "close": 120.0}]
    pd.testing.assert_frame_equal(frame, original)
    _assert_json_safe(sent)


@pytest.mark.unit
def test_intraday_index_is_preserved_alongside_date_column(mock_client):
    frame = pd.DataFrame(
        {"Date": ["2026-09-01", "2026-09-01"], "close": [100, 110]},
        index=pd.to_datetime(["2026-09-01 09:30", "2026-09-01 16:00"]),
    )
    original = frame.copy()
    agent = create_jev_agent(client=mock_client)
    agent(_state(analysis_date="2026-09-01", price_data=frame))
    sent = mock_client.evaluate.call_args.args[0]
    rows = sent["price_data"]
    assert rows[0]["Date"] == "2026-09-01"
    assert rows[1]["Date"] == "2026-09-01"
    assert "09:30" in str(rows[0]["timestamp"])
    assert "16:00" in str(rows[1]["timestamp"])
    pd.testing.assert_frame_equal(frame, original)
    _assert_json_safe(sent)


@pytest.mark.unit
def test_index_time_uses_fallback_field_when_timestamp_exists(mock_client):
    frame = pd.DataFrame(
        {"Date": ["2026-09-01"], "timestamp": ["2026-09-01"], "close": [100]},
        index=pd.to_datetime(["2026-09-01 09:30"]),
    )
    agent = create_jev_agent(client=mock_client)
    agent(_state(price_data=frame))
    sent = mock_client.evaluate.call_args.args[0]
    row = sent["price_data"][0]
    assert row["Date"] == "2026-09-01"
    assert row["timestamp"] == "2026-09-01"
    assert "09:30" in str(row["datetime"])
    _assert_json_safe(sent)


@pytest.mark.unit
def test_index_time_field_name_collision_is_rejected(mock_client):
    frame = pd.DataFrame(
        {
            "Date": ["2026-09-01"],
            "timestamp": ["2026-09-01"],
            "datetime": ["2026-09-01"],
            "time": ["2026-09-01"],
            "as_of": ["2026-09-01"],
            "asof": ["2026-09-01"],
            "index_timestamp": ["occupied"],
            "close": [100],
        },
        index=pd.to_datetime(["2026-09-01 09:30"]),
    )
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="overwriting an existing field"):
        agent(_state(price_data=frame))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_date_column_and_index_clock_times_must_agree(mock_client):
    frame = pd.DataFrame(
        {"Date": ["2026-09-01T09:30:00"], "close": [100]},
        index=pd.to_datetime(["2026-09-01 16:00"]),
    )
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="conflicting timestamp"):
        agent(_state(price_data=frame))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_timezone_aware_index_is_preserved(mock_client):
    index = pd.DatetimeIndex(["2026-09-01 09:30:00"], tz="America/New_York")
    frame = pd.DataFrame({"Date": ["2026-09-01"], "close": [100]}, index=index)
    original = frame.copy()
    agent = create_jev_agent(client=mock_client)
    agent(_state(price_data=frame))
    sent = mock_client.evaluate.call_args.args[0]
    stamp = str(sent["price_data"][0]["timestamp"])
    assert "09:30" in stamp
    assert ("-04:00" in stamp or "-05:00" in stamp)
    pd.testing.assert_frame_equal(frame, original)
    _assert_json_safe(sent)


@pytest.mark.unit
def test_same_instant_across_timezones_is_not_a_date_conflict(mock_client):
    frame = pd.DataFrame(
        {"Date": ["2026-09-01T20:30:00-04:00"], "close": [100]},
        index=pd.to_datetime(["2026-09-02T00:30:00+00:00"]),
    )
    original = frame.copy()
    agent = create_jev_agent(client=mock_client)
    agent(_state(analysis_date="2026-09-10", price_data=frame))
    mock_client.evaluate.assert_called_once()
    sent = mock_client.evaluate.call_args.args[0]
    row = sent["price_data"][0]
    assert "-04:00" in str(row["Date"])
    stamp = str(row["timestamp"])
    assert "00:30" in stamp
    assert ("+00:00" in stamp or stamp.endswith("Z"))
    pd.testing.assert_frame_equal(frame, original)
    _assert_json_safe(sent)


@pytest.mark.unit
def test_different_instants_across_timezones_are_rejected(mock_client):
    frame = pd.DataFrame(
        {"Date": ["2026-09-01T20:30:00-04:00"], "close": [100]},
        index=pd.to_datetime(["2026-09-02T01:30:00+00:00"]),
    )
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match=r"conflicting timestamp.*row 0"):
        agent(_state(analysis_date="2026-09-10", price_data=frame))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_aware_and_naive_clock_times_are_rejected(mock_client):
    frame = pd.DataFrame(
        {"Date": ["2026-09-01T09:30:00-04:00"], "close": [100]},
        index=pd.to_datetime(["2026-09-01 09:30:00"]),
    )
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="timezone-aware and naive"):
        agent(_state(analysis_date="2026-09-10", price_data=frame))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_same_offset_clock_times_are_accepted(mock_client):
    frame = pd.DataFrame(
        {"Date": ["2026-09-01T09:30:00-04:00"], "close": [100]},
        index=pd.to_datetime(["2026-09-01T09:30:00-04:00"]),
    )
    agent = create_jev_agent(client=mock_client)
    agent(_state(analysis_date="2026-09-10", price_data=frame))
    mock_client.evaluate.assert_called_once()
    sent = mock_client.evaluate.call_args.args[0]
    assert "-04:00" in str(sent["price_data"][0]["Date"])
    _assert_json_safe(sent)


@pytest.mark.unit
def test_undated_record_list_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="no recognizable date"):
        agent(_state(price_data=[{"close": 120}]))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_nested_json_string_cannot_bypass_future_date_check(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(
            _state(
                analysis_date="2026-09-01",
                price_data={"data": '[{"Date":"2026-09-02","close":120}]'},
            )
        )
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_nested_csv_string_cannot_bypass_future_date_check(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(
            _state(
                analysis_date="2026-09-01",
                price_data={"data": "Date,close\n2026-09-02,120"},
            )
        )
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_category_and_numeric_fields_remain_valid(mock_client):
    agent = create_jev_agent(client=mock_client)
    agent(
        _state(
            returns={"1d": 0.01},
            volume=1_000_000,
            technical_indicators={"rsi": 55},
            market_context={"regime": "risk-on"},
        )
    )
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["returns"] == {"1d": 0.01}
    assert sent["volume"] == 1_000_000
    assert sent["technical_indicators"] == {"rsi": 55}
    assert sent["market_context"] == {"regime": "risk-on"}
    _assert_json_safe(sent)


@pytest.mark.unit
def test_illegal_analysis_date_with_junk_suffix_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="YYYY-MM-DD"):
        agent(_state(analysis_date="2026-09-01junk"))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_illegal_analysis_date_with_time_suffix_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="YYYY-MM-DD"):
        agent(_state(analysis_date="2026-09-01T09:30:00"))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_illegal_market_timestamp_junk_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="not a parseable date"):
        agent(_state(price_data=[{"Date": "2026-09-01junk", "close": 120}]))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_illegal_market_timestamp_with_trailing_garbage_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="not a parseable date"):
        agent(_state(price_data=[{"Date": "2026-09-01T16:00:00junk", "close": 120}]))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_missing_values_become_json_null(mock_client):
    frame = pd.DataFrame(
        {"close": [np.nan]},
        index=pd.DatetimeIndex(["2026-09-01"]),
    )
    agent = create_jev_agent(client=mock_client)
    agent(_state(price_data=frame))
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["price_data"][0]["close"] is None
    _assert_json_safe(sent)


@pytest.mark.unit
def test_quoted_csv_future_date_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    quoted = '"Date","close"\n"2026-09-02",120'
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(_state(analysis_date="2026-09-01", price_data=quoted))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_nested_quoted_csv_future_date_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    quoted = '"Date","close"\n"2026-09-02",120'
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(_state(analysis_date="2026-09-01", price_data={"data": quoted}))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_quoted_csv_on_cutoff_is_sent_as_records(mock_client):
    agent = create_jev_agent(client=mock_client)
    quoted = '"Date","close"\n"2026-09-01",120'
    original = quoted
    agent(_state(analysis_date="2026-09-01", price_data=quoted))
    mock_client.evaluate.assert_called_once()
    sent = mock_client.evaluate.call_args.args[0]
    assert quoted == original
    assert sent["price_data"] == [{"Date": "2026-09-01", "close": "120"}]
    _assert_json_safe(sent)


@pytest.mark.unit
def test_nested_quoted_csv_on_cutoff_is_sent_as_records(mock_client):
    agent = create_jev_agent(client=mock_client)
    quoted = '"Date","close"\n"2026-09-01",120'
    agent(_state(analysis_date="2026-09-01", price_data={"data": quoted}))
    mock_client.evaluate.assert_called_once()
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["price_data"]["data"] == [{"Date": "2026-09-01", "close": "120"}]
    _assert_json_safe(sent)


@pytest.mark.unit
def test_duplicate_date_columns_are_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="duplicate column"):
        agent(_state(price_data="Date,Date,close\n2026-09-02,2026-09-01,120"))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_duplicate_date_columns_differing_by_case_are_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="duplicate column"):
        agent(_state(price_data="Date,DATE,close\n2026-09-02,2026-09-01,120"))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_duplicate_date_columns_differing_by_whitespace_are_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="duplicate column"):
        agent(_state(price_data="Date, Date ,close\n2026-09-02,2026-09-01,120"))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_csv_row_with_extra_fields_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="fields, expected"):
        agent(_state(price_data="Date,close\n2026-09-01,120,extra"))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_csv_row_with_missing_fields_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="fields, expected"):
        agent(_state(price_data="Date,close\n2026-09-01"))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_tsv_price_data_is_sent_as_records(mock_client):
    agent = create_jev_agent(client=mock_client)
    agent(_state(price_data="Date\tclose\n2026-09-01\t120"))
    mock_client.evaluate.assert_called_once()
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["price_data"] == [{"Date": "2026-09-01", "close": "120"}]
    _assert_json_safe(sent)


@pytest.mark.unit
def test_quoted_csv_field_containing_comma_is_sent_as_records(mock_client):
    agent = create_jev_agent(client=mock_client)
    text = 'Date,note,close\n2026-09-01,"hello, world",120'
    agent(_state(price_data=text))
    mock_client.evaluate.assert_called_once()
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["price_data"] == [
        {"Date": "2026-09-01", "note": "hello, world", "close": "120"}
    ]
    _assert_json_safe(sent)


@pytest.mark.unit
def test_plain_csv_payload_is_records_not_raw_text(mock_client):
    agent = create_jev_agent(client=mock_client)
    agent(_state(price_data="Date,close\n2026-09-01,120"))
    sent = mock_client.evaluate.call_args.args[0]
    assert isinstance(sent["price_data"], list)
    assert sent["price_data"] == [{"Date": "2026-09-01", "close": "120"}]
    _assert_json_safe(sent)


@pytest.mark.unit
def test_bom_csv_future_date_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    text = "\ufeffDate,close\n2026-09-02,120"
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(_state(analysis_date="2026-09-01", price_data=text))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_nested_bom_csv_future_date_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    text = "\ufeffDate,close\n2026-09-02,120"
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(_state(analysis_date="2026-09-01", price_data={"data": text}))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_bom_csv_on_cutoff_is_sent_as_records(mock_client):
    agent = create_jev_agent(client=mock_client)
    text = "\ufeffDate,close\n2026-09-01,120"
    original = text
    agent(_state(analysis_date="2026-09-01", price_data=text))
    mock_client.evaluate.assert_called_once()
    sent = mock_client.evaluate.call_args.args[0]
    assert text == original
    assert sent["price_data"] == [{"Date": "2026-09-01", "close": "120"}]
    _assert_json_safe(sent)


@pytest.mark.unit
def test_nested_bom_csv_on_cutoff_is_sent_as_records(mock_client):
    agent = create_jev_agent(client=mock_client)
    text = "\ufeffDate,close\n2026-09-01,120"
    agent(_state(analysis_date="2026-09-01", price_data={"data": text}))
    mock_client.evaluate.assert_called_once()
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["price_data"]["data"] == [{"Date": "2026-09-01", "close": "120"}]
    _assert_json_safe(sent)


@pytest.mark.unit
def test_unclosed_csv_quote_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="CSV could not be parsed"):
        agent(_state(price_data='Date,close\n2026-09-01,"120'))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_nested_unclosed_csv_quote_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="CSV could not be parsed"):
        agent(_state(price_data={"data": 'Date,close\n2026-09-01,"120'}))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_quoted_multiline_csv_field_is_sent_as_records(mock_client):
    agent = create_jev_agent(client=mock_client)
    text = 'Date,note,close\n2026-09-01,"hello\nworld",120'
    agent(_state(price_data=text))
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["price_data"] == [
        {"Date": "2026-09-01", "note": "hello\nworld", "close": "120"}
    ]
    _assert_json_safe(sent)


@pytest.mark.unit
def test_mixed_dated_and_undated_rows_are_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    rows = [
        {"Date": "2026-09-01", "close": 100},
        {"close": 999},
    ]
    original = [dict(row) for row in rows]
    with pytest.raises(JevLookAheadError, match=r"price_data\[1\].*no recognizable date"):
        agent(_state(price_data=rows))
    mock_client.evaluate.assert_not_called()
    assert rows == original


@pytest.mark.unit
def test_empty_date_field_on_a_row_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match=r"price_data\[1\].*not a parseable date"):
        agent(
            _state(
                price_data=[
                    {"Date": "2026-09-01", "close": 100},
                    {"Date": "", "close": 999},
                ]
            )
        )
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_mixed_dated_undated_and_none_rows_are_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    rows = [
        {"Date": "2026-09-01", "close": 100},
        {"close": 999},
        None,
    ]
    original = [None if row is None else dict(row) for row in rows]
    with pytest.raises(JevLookAheadError, match=r"price_data\[\d+\]"):
        agent(_state(analysis_date="2026-09-01", price_data=rows))
    mock_client.evaluate.assert_not_called()
    assert rows == original


@pytest.mark.unit
@pytest.mark.parametrize("extra", [42, False, True, "not-a-bar"])
def test_mixed_scalar_cannot_bypass_undated_row_check(mock_client, extra):
    agent = create_jev_agent(client=mock_client)
    rows = [
        {"Date": "2026-09-01", "close": 100},
        {"close": 999},
        extra,
    ]
    with pytest.raises(JevLookAheadError, match=r"price_data\[\d+\]"):
        agent(_state(analysis_date="2026-09-01", price_data=rows))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_none_row_among_dated_records_is_located(mock_client):
    agent = create_jev_agent(client=mock_client)
    rows = [
        {"Date": "2026-09-01", "close": 100},
        {"Date": "2026-08-31", "close": 99},
        None,
    ]
    with pytest.raises(JevLookAheadError, match=r"price_data\[2\] is null"):
        agent(_state(analysis_date="2026-09-01", price_data=rows))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_all_dated_records_are_accepted(mock_client):
    agent = create_jev_agent(client=mock_client)
    agent(
        _state(
            price_data=[
                {"Date": "2026-09-01", "close": 100},
                {"Date": "2026-08-31", "close": 99},
            ]
        )
    )
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["price_data"][0]["Date"] == "2026-09-01"
    _assert_json_safe(sent)


@pytest.mark.unit
def test_date_keyed_nested_records_inherit_outer_date(mock_client):
    agent = create_jev_agent(client=mock_client)
    agent(
        _state(
            analysis_date="2026-09-01",
            price_data={"2026-09-01": [{"close": 100}, {"close": 101}]},
        )
    )
    mock_client.evaluate.assert_called_once()
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["price_data"]["2026-09-01"] == [{"close": 100}, {"close": 101}]
    _assert_json_safe(sent)


@pytest.mark.unit
def test_date_keyed_nested_future_records_are_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(
            _state(
                analysis_date="2026-09-01",
                price_data={"2026-09-02": [{"close": 100}]},
            )
        )
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_numpy_datetime64_units_normalize_to_iso():
    for unit in ("D", "s", "ms", "us", "ns"):
        raw = (
            np.datetime64("2026-09-02", unit)
            if unit == "D"
            else np.datetime64("2026-09-02T10:00:00", unit)
        )
        out = normalize_market_value(raw, field="price_data")
        assert isinstance(out, str), unit
        assert not isinstance(out, (int, np.integer)), unit
        assert out.startswith("2026-09-02"), (unit, out)
        assert parse_asof_date(out) == date(2026, 9, 2)
        json.dumps(out, allow_nan=False)


@pytest.mark.unit
def test_future_numpy_datetime64_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    stamp = np.datetime64("2026-09-02T10:00:00", "ns")
    with pytest.raises(JevLookAheadError, match="after analysis_date"):
        agent(
            _state(
                analysis_date="2026-09-01",
                price_data=[{"Date": stamp, "close": 120}],
            )
        )
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_legal_numpy_datetime64_passes_through_agent(mock_client):
    agent = create_jev_agent(client=mock_client)
    stamp = np.datetime64("2026-09-01T10:00:00", "ns")
    original = stamp
    agent(_state(analysis_date="2026-09-01", price_data=[{"Date": stamp, "close": 120}]))
    sent = mock_client.evaluate.call_args.args[0]
    assert isinstance(sent["price_data"][0]["Date"], str)
    assert sent["price_data"][0]["Date"].startswith("2026-09-01")
    assert parse_asof_date(sent["price_data"][0]["Date"]) == date(2026, 9, 1)
    assert original is stamp
    _assert_json_safe(sent)


@pytest.mark.unit
def test_numpy_nat_in_required_date_field_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="not a parseable date"):
        agent(_state(price_data=[{"Date": np.datetime64("NaT", "ns"), "close": 120}]))
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_numpy_nat_in_non_date_value_becomes_null(mock_client):
    agent = create_jev_agent(client=mock_client)
    agent(
        _state(
            price_data=[
                {"Date": "2026-09-01", "close": np.datetime64("NaT", "ns")},
            ]
        )
    )
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["price_data"][0]["close"] is None
    _assert_json_safe(sent)


@pytest.mark.unit
def test_numpy_datetime64_mapping_key_is_iso(mock_client):
    agent = create_jev_agent(client=mock_client)
    key = np.datetime64("2026-09-01T10:00:00", "ns")
    payload = {key: 0.01}
    original_keys = list(payload.keys())
    agent(_state(returns=payload))
    sent = mock_client.evaluate.call_args.args[0]
    sent_key = next(iter(sent["returns"]))
    assert isinstance(sent_key, str)
    assert sent_key.startswith("2026-09-01")
    assert original_keys == [key]
    _assert_json_safe(sent)


@pytest.mark.unit
def test_out_of_range_numpy_datetime64_is_rejected(mock_client):
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevLookAheadError, match="outside the supported date range"):
        agent(
            _state(
                price_data=[{"Date": np.datetime64("10000-01-01", "D"), "close": 120}]
            )
        )
    mock_client.evaluate.assert_not_called()


@pytest.mark.unit
def test_numpy_numeric_scalar_behavior_is_unchanged(mock_client):
    agent = create_jev_agent(client=mock_client)
    agent(
        _state(
            volume=np.int64(1_000_000),
            technical_indicators={"rsi": np.float64(55.5)},
        )
    )
    sent = mock_client.evaluate.call_args.args[0]
    assert sent["volume"] == 1_000_000
    assert sent["technical_indicators"]["rsi"] == pytest.approx(55.5)
    _assert_json_safe(sent)


@pytest.mark.unit
def test_signal_from_evaluation_rejects_out_of_range_probability():
    result = _evaluation(
        choices={
            "direction": JevChoiceResult(
                choice="BULLISH",
                probabilities={"BULLISH": 2.0, "NEUTRAL": 0.0, "BEARISH": -1.0},
            )
        }
    )
    with pytest.raises(JevMalformedResponseError, match="\\[0, 1\\]"):
        signal_from_evaluation(_state(), "1d", result)


@pytest.mark.unit
def test_signal_from_evaluation_rejects_missing_neutral():
    result = _evaluation(
        choices={
            "direction": JevChoiceResult(
                choice="BULLISH",
                probabilities={"BULLISH": 0.7, "BEARISH": 0.3},
            )
        }
    )
    with pytest.raises(JevMalformedResponseError, match="exactly BULLISH, NEUTRAL, and BEARISH"):
        signal_from_evaluation(_state(), "1d", result)


@pytest.mark.unit
def test_signal_from_evaluation_rejects_extra_direction_label():
    result = _evaluation(
        choices={
            "direction": JevChoiceResult(
                choice="BULLISH",
                probabilities={
                    "BULLISH": 0.5,
                    "NEUTRAL": 0.2,
                    "BEARISH": 0.2,
                    "FLAT": 0.1,
                },
            )
        }
    )
    with pytest.raises(JevMalformedResponseError, match="exactly BULLISH, NEUTRAL, and BEARISH"):
        signal_from_evaluation(_state(), "1d", result)


@pytest.mark.unit
def test_signal_from_evaluation_rejects_duplicate_direction_labels():
    result = _evaluation(
        choices={
            "direction": JevChoiceResult(
                choice="BULLISH",
                probabilities={"BULLISH": 0.4, "bullish": 0.3, "NEUTRAL": 0.2, "BEARISH": 0.1},
            )
        }
    )
    with pytest.raises(JevMalformedResponseError, match="duplicate direction label"):
        signal_from_evaluation(_state(), "1d", result)


@pytest.mark.unit
def test_signal_from_evaluation_rejects_choice_not_at_maximum():
    result = _evaluation(
        choices={
            "direction": JevChoiceResult(
                choice="BEARISH",
                probabilities={"BULLISH": 0.70, "NEUTRAL": 0.20, "BEARISH": 0.10},
            )
        }
    )
    with pytest.raises(JevMalformedResponseError, match="maximum-probability"):
        signal_from_evaluation(_state(), "1d", result)


@pytest.mark.unit
def test_signal_from_evaluation_allows_tied_maximum_choice():
    result = _evaluation(
        choices={
            "direction": JevChoiceResult(
                choice="NEUTRAL",
                probabilities={"BULLISH": 0.4, "NEUTRAL": 0.4, "BEARISH": 0.2},
                confidence=0.5,
            )
        }
    )
    signal = signal_from_evaluation(_state(), "1d", result)
    assert signal.direction == "NEUTRAL"


@pytest.mark.unit
def test_signal_from_evaluation_rejects_confidence_out_of_range():
    result = _evaluation(
        choices={
            "direction": JevChoiceResult(
                choice="BULLISH",
                probabilities={"BULLISH": 0.70, "NEUTRAL": 0.20, "BEARISH": 0.10},
                confidence=4.0,
            )
        }
    )
    with pytest.raises(JevMalformedResponseError, match="confidence"):
        signal_from_evaluation(_state(), "1d", result)


@pytest.mark.unit
def test_signal_from_evaluation_rejects_signal_strength_out_of_range():
    result = _evaluation(
        scores={
            "signal_strength": JevScoreResult(
                score=99.0,
                probabilities={"0": 1.0},
            ),
            "market_risk": JevScoreResult(score=2.0, probabilities={"0": 1.0}),
        }
    )
    with pytest.raises(JevMalformedResponseError, match="signal_strength"):
        signal_from_evaluation(_state(), "1d", result)


@pytest.mark.unit
def test_signal_from_evaluation_rejects_market_risk_below_zero():
    result = _evaluation(
        scores={
            "signal_strength": JevScoreResult(score=3.0, probabilities={"0": 1.0}),
            "market_risk": JevScoreResult(score=-5.0, probabilities={"0": 1.0}),
        }
    )
    with pytest.raises(JevMalformedResponseError, match="market_risk"):
        signal_from_evaluation(_state(), "1d", result)


@pytest.mark.unit
def test_signal_from_evaluation_rejects_nan_score():
    result = _evaluation(
        scores={
            "signal_strength": JevScoreResult(score=float("nan"), probabilities={"0": 1.0}),
            "market_risk": JevScoreResult(score=2.0, probabilities={"0": 1.0}),
        }
    )
    with pytest.raises(JevMalformedResponseError, match="finite"):
        signal_from_evaluation(_state(), "1d", result)


@pytest.mark.unit
def test_signal_from_evaluation_rejects_infinite_score():
    result = _evaluation(
        scores={
            "signal_strength": JevScoreResult(score=3.0, probabilities={"0": 1.0}),
            "market_risk": JevScoreResult(score=float("inf"), probabilities={"0": 1.0}),
        }
    )
    with pytest.raises(JevMalformedResponseError, match="finite"):
        signal_from_evaluation(_state(), "1d", result)


@pytest.mark.unit
def test_signal_from_evaluation_accepts_boundary_scores_and_optional_confidence():
    result = _evaluation(
        choices={
            "direction": JevChoiceResult(
                choice="BULLISH",
                probabilities={"BULLISH": 1.0, "NEUTRAL": 0.0, "BEARISH": 0.0},
                confidence=None,
            )
        },
        scores={
            "signal_strength": JevScoreResult(score=0.0, probabilities={"0": 1.0}),
            "market_risk": JevScoreResult(score=4.0, probabilities={"4": 1.0}),
        },
    )
    signal = signal_from_evaluation(_state(), "1d", result)
    assert signal.signal_strength == 0.0
    assert signal.signal_strength_label == "VERY_WEAK"
    assert signal.market_risk == 4.0
    assert signal.market_risk_label == "VERY_HIGH"
    assert signal.confidence is None


@pytest.mark.unit
def test_signal_from_evaluation_accepts_fractional_scores():
    result = _evaluation(
        scores={
            "signal_strength": JevScoreResult(score=3.7, probabilities={"0": 1.0}),
            "market_risk": JevScoreResult(score=2.1, probabilities={"0": 1.0}),
        }
    )
    signal = signal_from_evaluation(_state(), "1d", result)
    assert signal.signal_strength == pytest.approx(3.7)
    assert signal.signal_strength_label == "VERY_STRONG"
    assert signal.market_risk == pytest.approx(2.1)
    assert signal.market_risk_label == "MEDIUM"


@pytest.mark.unit
def test_agent_rejects_malformed_client_scores_before_evaluate_returns_to_caller(mock_client):
    mock_client.evaluate.return_value = _evaluation(
        scores={
            "signal_strength": JevScoreResult(score=99.0, probabilities={"0": 1.0}),
            "market_risk": JevScoreResult(score=2.0, probabilities={"0": 1.0}),
        }
    )
    agent = create_jev_agent(client=mock_client)
    with pytest.raises(JevMalformedResponseError, match="signal_strength"):
        agent(_state())


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        os.environ.get("TYPESAFE_API_KEY")
        and os.environ.get("TYPESAFE_API_KEY") != "placeholder"
    ),
    reason="TYPESAFE_API_KEY not set (or placeholder); skipping live Jev call",
)
def test_live_jev_smoke():
    """Optional smoke: one real System One call when a key is present."""
    agent = create_jev_agent()
    signal = agent(
        JevMarketState(
            ticker="NVDA",
            analysis_date="2026-09-01",
            price_data=[{"Date": "2026-09-01", "close": 120.0, "volume": 1000}],
            technical_indicators={"rsi": 55.0, "close_50_sma": 118.0},
        )
    )
    assert signal.direction in {"BULLISH", "NEUTRAL", "BEARISH"}
    assert set(signal.direction_probabilities) >= {"BULLISH", "NEUTRAL", "BEARISH"}
    assert signal.horizon == "1d"
    assert signal.model
