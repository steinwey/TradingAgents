"""JevDecisionAgent: LLM maps a Jev prediction onto Buy/Hold/Sell.

This agent is not a LangGraph node. It does not call Jev, scan market dates,
or place orders. It uses :func:`create_llm_client` (ChatGPT / other providers)
and the same structured-output binding as the graph Trader.

Jev is System One prediction. This agent is the decision layer. On Jev or LLM
failure use :meth:`JevDecisionAgent.skip` — Hold + review, never an invented Buy.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradingagents.agents.jev.models import JevTradingSignal
from tradingagents.agents.schemas import TraderAction, TraderProposal
from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.agents.utils.structured import NO_EXTERNAL_TOOLS, bind_structured
from tradingagents.llm_clients import create_llm_client
from tradingagents.portfolio import PortfolioContext

logger = logging.getLogger(__name__)

_AGENT_NAME = "JevDecision"


class JevDecision(BaseModel):
    """Live-trading intent from Jev + LLM. Not a broker order."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    timestamp: str
    horizon: str
    action: TraderAction
    review: bool = Field(
        description="True only when Jev or the LLM failed; a model Hold is a real no-trade.",
    )
    reasoning: str
    proposal: TraderProposal
    held_quantity: float | None = Field(
        default=None,
        description="Signed units in the supplied book; None means the book was not given.",
    )
    signal: JevTradingSignal | None = None

    def as_trader_proposal(self) -> TraderProposal:
        return self.proposal


class JevDecisionAgent:
    """LLM decision: :class:`JevTradingSignal` (+ optional book) → trade intent."""

    def __init__(
        self,
        llm: Any | None = None,
        *,
        client: Any | None = None,
        config: Mapping[str, Any] | None = None,
    ):
        if llm is not None:
            self.llm = llm
        elif client is not None:
            self.llm = client.get_llm()
        else:
            self.llm = _llm_from_config(config)
        self._structured = bind_structured(self.llm, TraderProposal, _AGENT_NAME)

    def decide(
        self,
        signal: JevTradingSignal | Mapping[str, Any],
        portfolio: PortfolioContext | Mapping[str, Any] | None = None,
    ) -> JevDecision:
        parsed = (
            signal
            if isinstance(signal, JevTradingSignal)
            else JevTradingSignal.model_validate(dict(signal))
        )
        book = _parse_portfolio(portfolio)
        held = _held_quantity(book, parsed.ticker)
        messages = _decision_messages(parsed, book)
        try:
            proposal = self._invoke_proposal(messages)
        except Exception as exc:
            logger.warning("%s: LLM decision failed (%s); skipping without a Buy", _AGENT_NAME, exc)
            return self.skip(
                exc,
                ticker=parsed.ticker,
                timestamp=parsed.timestamp,
                horizon=parsed.horizon,
            )
        return JevDecision(
            ticker=parsed.ticker,
            timestamp=parsed.timestamp,
            horizon=parsed.horizon,
            action=proposal.action,
            review=False,
            reasoning=proposal.reasoning,
            proposal=proposal,
            held_quantity=held,
            signal=parsed,
        )

    def skip(
        self,
        error: BaseException,
        *,
        ticker: str,
        timestamp: str = "",
        horizon: str = "",
    ) -> JevDecision:
        """Map a Jev or LLM failure to Hold + review. Does not call the LLM."""
        reasoning = (
            f"No usable decision for {ticker}: {error}. "
            "No Buy or Sell is invented from a failed prediction or LLM call."
        )
        proposal = TraderProposal(
            action=TraderAction.HOLD,
            reasoning=reasoning,
            entry_price=None,
            stop_loss=None,
            position_sizing=None,
        )
        return JevDecision(
            ticker=ticker,
            timestamp=timestamp,
            horizon=horizon,
            action=TraderAction.HOLD,
            review=True,
            reasoning=reasoning,
            proposal=proposal,
            held_quantity=None,
            signal=None,
        )

    def __call__(
        self,
        signal: JevTradingSignal | Mapping[str, Any],
        portfolio: PortfolioContext | Mapping[str, Any] | None = None,
    ) -> JevDecision:
        return self.decide(signal, portfolio)

    def _invoke_proposal(self, messages: list[dict[str, str]]) -> TraderProposal:
        if self._structured is None:
            raise RuntimeError(
                f"{_AGENT_NAME}: provider does not support structured output; "
                "refusing to parse free text into a live order"
            )
        result = self._structured.invoke(messages)
        if result is None:
            raise ValueError("structured output returned no parsed result")
        return _as_proposal(result)


def create_jev_decision_agent(
    llm: Any | None = None,
    *,
    client: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> JevDecisionAgent:
    """Factory matching other agents' ``create_*`` style. Not a graph node."""
    return JevDecisionAgent(llm, client=client, config=config)


def _llm_from_config(config: Mapping[str, Any] | None) -> Any:
    from tradingagents.default_config import DEFAULT_CONFIG

    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(dict(config))
    client = create_llm_client(
        provider=cfg["llm_provider"],
        model=cfg["quick_think_llm"],
        base_url=cfg.get("backend_url"),
        **_client_kwargs(cfg),
    )
    return client.get_llm()


def _client_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Provider knobs forwarded to :func:`create_llm_client`, same keys as the graph."""
    kwargs: dict[str, Any] = {}
    provider = str(config.get("llm_provider") or "").lower()
    if provider == "google" and config.get("google_thinking_level"):
        kwargs["thinking_level"] = config["google_thinking_level"]
    elif provider == "openai" and config.get("openai_reasoning_effort"):
        kwargs["reasoning_effort"] = config["openai_reasoning_effort"]
    elif provider == "anthropic" and config.get("anthropic_effort"):
        kwargs["effort"] = config["anthropic_effort"]
    if config.get("temperature") not in (None, ""):
        kwargs["temperature"] = float(config["temperature"])
    if config.get("llm_max_retries") not in (None, ""):
        kwargs["max_retries"] = int(config["llm_max_retries"])
    if config.get("max_tokens") not in (None, ""):
        key = "max_output_tokens" if provider == "google" else "max_tokens"
        kwargs[key] = int(config["max_tokens"])
    return kwargs


def _parse_portfolio(
    portfolio: PortfolioContext | Mapping[str, Any] | None,
) -> PortfolioContext | None:
    if portfolio is None:
        return None
    if isinstance(portfolio, PortfolioContext):
        return portfolio
    return PortfolioContext.model_validate(dict(portfolio))


def _held_quantity(book: PortfolioContext | None, ticker: str) -> float | None:
    if book is None:
        return None
    held = book.position_in(ticker)
    return 0.0 if held is None else float(held.quantity)


def _as_proposal(result: Any) -> TraderProposal:
    if isinstance(result, TraderProposal):
        return result
    if isinstance(result, BaseModel):
        return TraderProposal.model_validate(result.model_dump())
    if isinstance(result, Mapping):
        return TraderProposal.model_validate(dict(result))
    raise TypeError(f"structured output is not a TraderProposal: {type(result)!r}")


def _decision_messages(
    signal: JevTradingSignal,
    book: PortfolioContext | None,
) -> list[dict[str, str]]:
    if book is None:
        portfolio_block = (
            "Portfolio context: not provided. You do not know the caller's current "
            "holdings or cash, so do not assume a flat book; give a directional "
            "Buy/Hold/Sell the caller can apply to their own position."
        )
    else:
        portfolio_block = book.render(signal.ticker)

    signal_payload = json.dumps(signal.model_dump(mode="json"), indent=2, sort_keys=True)
    return [
        {
            "role": "system",
            "content": (
                "You are a live-trading decision agent. Jev (System One) already "
                "produced a market prediction; you do not re-predict direction, "
                "signal strength, or market risk, and you do not invent a fourth score. "
                "Turn that prediction and the optional portfolio into one transaction: "
                "Buy, Hold, or Sell. "
                "Prefer Hold when direction is NEUTRAL, signal strength is weak "
                "(below STRONG / 3), or market risk is high (above MEDIUM / 2). "
                "If a book is supplied: do not add to an already aligned long unless "
                "the case is clearly strong; do not open a short unless the book is "
                "flat or short and the bearish case is clearly strong; a long plus "
                "a bearish prediction is an exit (Sell). "
                "Omit entry_price and stop_loss unless an absolute price is present "
                "in this prompt; never fabricate a level. "
                + NO_EXTERNAL_TOOLS
                + get_language_instruction()
            ),
        },
        {
            "role": "user",
            "content": (
                f"Jev trading signal for {signal.ticker} "
                f"(as-of {signal.timestamp}, horizon {signal.horizon}):\n"
                f"{signal_payload}\n\n"
                f"{portfolio_block}\n\n"
                "Decide the transaction.\n\n"
                "## Output\n\n"
                "- **Action**: exactly one of Buy / Hold / Sell\n"
                "- **Reasoning**: why, anchored in the Jev fields and the book\n"
                "- **Entry Price**, **Stop Loss**, **Position Sizing**: only when you "
                "can state them from evidence in this prompt"
            ),
        },
    ]
