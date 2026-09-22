"""Unit tests for JevDecisionAgent. Mock LLM only; no live API, no graph."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.jev.jev_decision import (
    JevDecisionAgent,
    create_jev_decision_agent,
)
from tradingagents.agents.jev.models import (
    MARKET_RISK_LEVELS,
    SIGNAL_STRENGTH_LEVELS,
    JevTradingSignal,
)
from tradingagents.agents.schemas import TraderAction, TraderProposal
from tradingagents.decision_models.errors import JevTimeoutError
from tradingagents.portfolio import PortfolioContext


def _signal(**overrides) -> JevTradingSignal:
    payload = {
        "ticker": "NVDA",
        "timestamp": "2026-09-01",
        "horizon": "1d",
        "direction": "BULLISH",
        "direction_probabilities": {"BULLISH": 0.70, "NEUTRAL": 0.20, "BEARISH": 0.10},
        "signal_strength": 3.0,
        "signal_strength_label": SIGNAL_STRENGTH_LEVELS[3],
        "market_risk": 2.0,
        "market_risk_label": MARKET_RISK_LEVELS[2],
        "confidence": 0.76,
    }
    payload.update(overrides)
    return JevTradingSignal.model_validate(payload)


def _llm(proposal: TraderProposal | None = None, *, captured: dict | None = None) -> MagicMock:
    if proposal is None:
        proposal = TraderProposal(action=TraderAction.BUY, reasoning="Jev is bullish and strong.")
    structured = MagicMock()

    def _invoke(messages):
        if captured is not None:
            captured["messages"] = messages
        return proposal

    structured.invoke.side_effect = _invoke
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    llm.invoke.side_effect = AssertionError("free-text fallback must not run for live decisions")
    return llm


@pytest.mark.unit
def test_structured_buy_is_returned():
    captured: dict = {}
    agent = create_jev_decision_agent(_llm(captured=captured))
    decision = agent.decide(_signal())
    assert decision.action is TraderAction.BUY
    assert decision.review is False
    assert decision.signal is not None
    assert decision.held_quantity is None
    assert decision.as_trader_proposal().action is TraderAction.BUY
    prompt = captured["messages"][1]["content"]
    assert "NVDA" in prompt
    assert "BULLISH" in prompt
    assert "not provided" in captured["messages"][1]["content"] or "not assume a flat book" in prompt


@pytest.mark.unit
def test_prompt_includes_portfolio_when_supplied():
    captured: dict = {}
    agent = create_jev_decision_agent(_llm(captured=captured))
    book = PortfolioContext.model_validate(
        {"cash": 10000.0, "currency": "USD", "positions": [{"ticker": "NVDA", "quantity": 10}]}
    )
    decision = agent.decide(_signal(), portfolio=book)
    assert decision.held_quantity == 10.0
    user = captured["messages"][1]["content"]
    assert "Current position in NVDA" in user
    assert "10" in user


@pytest.mark.unit
def test_skip_does_not_call_llm():
    llm = _llm()
    agent = JevDecisionAgent(llm)
    decision = agent.skip(
        JevTimeoutError("timed out"),
        ticker="NVDA",
        timestamp="2026-09-01",
        horizon="1d",
    )
    assert decision.action is TraderAction.HOLD
    assert decision.review is True
    assert decision.signal is None
    llm.with_structured_output.return_value.invoke.assert_not_called()
    assert "No Buy or Sell is invented" in decision.reasoning


@pytest.mark.unit
def test_structured_failure_is_hold_review_never_buy():
    structured = MagicMock()
    structured.invoke.side_effect = ValueError("bad JSON")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    agent = create_jev_decision_agent(llm)
    decision = agent.decide(_signal())
    assert decision.action is TraderAction.HOLD
    assert decision.review is True
    assert decision.proposal.action is TraderAction.HOLD


@pytest.mark.unit
def test_missing_structured_output_is_hold_review():
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
    agent = create_jev_decision_agent(llm)
    decision = agent.decide(_signal())
    assert decision.action is TraderAction.HOLD
    assert decision.review is True


@pytest.mark.unit
def test_uses_create_llm_client_when_no_llm_injected(monkeypatch):
    captured: dict = {}
    proposal = TraderProposal(action=TraderAction.SELL, reasoning="Bearish Jev signal.")
    mock_llm = _llm(proposal, captured=captured)
    mock_client = MagicMock()
    mock_client.get_llm.return_value = mock_llm

    def fake_create_llm_client(*, provider, model, base_url, **kwargs):
        captured["client_args"] = {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            **kwargs,
        }
        return mock_client

    monkeypatch.setattr(
        "tradingagents.agents.jev.jev_decision.create_llm_client",
        fake_create_llm_client,
    )
    agent = create_jev_decision_agent(
        config={
            "llm_provider": "openai",
            "quick_think_llm": "gpt-4o-mini",
            "backend_url": None,
            "openai_reasoning_effort": "low",
        }
    )
    decision = agent.decide(_signal())
    assert decision.action is TraderAction.SELL
    assert captured["client_args"]["provider"] == "openai"
    assert captured["client_args"]["model"] == "gpt-4o-mini"
    assert captured["client_args"]["reasoning_effort"] == "low"
    mock_client.get_llm.assert_called_once()


@pytest.mark.unit
def test_callable_matches_decide():
    agent = create_jev_decision_agent(_llm())
    signal = _signal()
    assert agent(signal).action is agent.decide(signal).action


@pytest.mark.unit
def test_system_prompt_forbids_tools_and_reinventing_jev():
    captured: dict = {}
    agent = create_jev_decision_agent(_llm(captured=captured))
    agent.decide(_signal())
    system = captured["messages"][0]["content"]
    assert "Do not call external tools" in system
    assert "do not invent a fourth score" in system
    assert "re-predict" in system
