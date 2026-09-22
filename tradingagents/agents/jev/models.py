"""Market inputs, prediction outputs and scoring rubrics for JEV."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tradingagents.agents.jev.market_data import normalize_market_value

Direction = Literal["BULLISH", "NEUTRAL", "BEARISH"]  # 预测方向，不是下单动作

# 方向证据有多清楚；Score 0=VERY_WEAK … 4=VERY_STRONG，不是涨跌幅度
SIGNAL_STRENGTH_LEVELS = ("VERY_WEAK", "WEAK", "MEDIUM", "STRONG", "VERY_STRONG")
# 预测环境有多险；Score 0=VERY_LOW … 4=VERY_HIGH，不是组合仓位风险
MARKET_RISK_LEVELS = ("VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH")

class JevMarketState(BaseModel):
    """Compact, caller-prepared snapshot. Do not dump the full LangGraph state.

    Market-data fields (``price_data``, ``returns``, ``volume``,
    ``technical_indicators``, ``market_context``) must use a supported
    structured format so :func:`assert_point_in_time` can parse dates. They are
    normalized to JSON-safe values before validation and send. Free-text
    analyst reports are forwarded with ``look_ahead_warning`` and are not
    scanned for event dates mentioned in prose.

    ``analysis_date`` is an inclusive calendar date ``YYYY-MM-DD`` (no time
    suffix). The cutoff includes that entire day.
    """

    model_config = ConfigDict(extra="forbid")

    ticker: str
    analysis_date: str = Field(description="Inclusive as-of date YYYY-MM-DD")
    price_data: Any | None = None
    returns: Any | None = None
    volume: Any | None = None
    technical_indicators: Any | None = None
    market_context: Any | None = None
    market_report: str | None = None
    fundamentals_report: str | None = None
    news_report: str | None = None
    sentiment_report: str | None = None

    def to_jev_state(self, *, horizon: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ticker": self.ticker,
            "analysis_date": self.analysis_date,
            "point_in_time_cutoff": self.analysis_date,
            "prediction_horizon": horizon,
        }
        optional = {
            "price_data": self.price_data,
            "returns": self.returns,
            "volume": self.volume,
            "technical_indicators": self.technical_indicators,
            "market_context": self.market_context,
        }
        for key, value in optional.items():
            if value is not None:
                payload[key] = normalize_market_value(value, field=key)
        reports = {
            "market_report": self.market_report,
            "fundamentals_report": self.fundamentals_report,
            "news_report": self.news_report,
            "sentiment_report": self.sentiment_report,
        }
        attached = {k: v for k, v in reports.items() if v}
        if attached:
            payload["analyst_reports"] = attached
            payload["look_ahead_warning"] = (
                "analyst_reports are free text and are not date-validated; "
                "the caller must ensure they contain only information available "
                f"on or before {self.analysis_date}."
            )
        return payload


class JevTradingSignal(BaseModel):
    """Market prediction for a later decision agent. Not a portfolio action."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    timestamp: str
    horizon: str
    direction: Direction
    direction_probabilities: dict[str, float]
    signal_strength: float
    signal_strength_label: str
    signal_strength_levels: tuple[str, ...] = SIGNAL_STRENGTH_LEVELS
    market_risk: float
    market_risk_label: str
    market_risk_levels: tuple[str, ...] = MARKET_RISK_LEVELS
    confidence: float | None = Field(
        default=None,
        description="Jev Choice confidence for direction; not a second invented metric.",
    )
    model: str | None = None
    latency_ms: float | None = None
