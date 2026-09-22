"""Observation-date scanning and inclusive daily cutoffs for JEV market data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import TYPE_CHECKING, Any

from tradingagents.agents.jev.dates import _is_date_field_name, parse_analysis_date, parse_asof_date
from tradingagents.agents.jev.market_data import (
    _looks_like_json_text,
    _parse_json_text,
    _records_from_csv,
    _records_from_frame,
    normalize_market_value,
)
from tradingagents.dataflows.utils import get_current_date
from tradingagents.decision_models.errors import JevLookAheadError

if TYPE_CHECKING:
    from tradingagents.agents.jev.models import JevMarketState

# 必须验日期、防止用到 analysis_date 之后数据的结构化行情字段。
# 分析师报告不在此列：只附 look_ahead_warning，不扫正文里的日期。
_MARKET_DATA_FIELDS = (
    "price_data",  # K 线 / OHLCV
    "returns",  # 收益率
    "volume",  # 成交量
    "technical_indicators",  # 技术指标
    "market_context",  # 市场环境快照
)

def _unsupported_market_format(field: str, value: Any) -> JevLookAheadError:
    return JevLookAheadError(
        f"{field} has unsupported market-data format {type(value).__name__}; "
        "pass dated records (list of dicts with a Date field), a YYYY-MM-DD-keyed "
        "mapping, a Date-column CSV string, numeric scalars, or omit it"
    )


def observation_dates(
    value: Any,
    *,
    field: str,
    allow_labels: bool = False,
    row_date: date | None = None,
) -> list[date]:
    """Audit raw input, preserving the standalone helper's supported formats."""
    return _observation_dates(
        value, field=field, allow_labels=allow_labels, row_date=row_date,
        normalized=False,
    )


def _observation_dates(
    value: Any,
    *,
    field: str,
    allow_labels: bool = False,
    row_date: date | None = None,
    normalized: bool,
) -> list[date]:
    """Scan dates; normalized trees need no CSV/JSON/frame conversion.

    ``allow_labels`` permits category/label strings inside mappings (for example
    ``{"regime": "risk-on"}``). It does not skip JSON, CSV, or date strings.

    ``row_date`` is inherited from an outer mapping key that itself parsed as a
    market timestamp, so date-keyed records need not repeat a Date field.
    """
    if value is None or isinstance(value, bool):
        return [] if row_date is None else [row_date]
    if isinstance(value, (int, float)):
        return [] if row_date is None else [row_date]
    if isinstance(value, str):
        if not value.strip():
            return [] if row_date is None else [row_date]
        if not normalized:
            csv_records = _records_from_csv(value, field=field)
            if csv_records is not None:
                return _observation_dates(
                    csv_records, field=field, allow_labels=False,
                    row_date=row_date, normalized=normalized
                )
            if _looks_like_json_text(value):
                return _observation_dates(
                    _parse_json_text(value, field=field),
                    field=field,
                    allow_labels=False,
                    row_date=row_date,
                    normalized=normalized,
                )
        parsed = parse_asof_date(value)
        if parsed is not None:
            return [parsed]
        if allow_labels:
            return [] if row_date is None else [row_date]
        raise JevLookAheadError(
            f"{field} is unstructured text and cannot be point-in-time validated; "
            "pass dated records, a Date-column CSV, or omit it"
        )

    if not normalized:
        parsed_ts = parse_asof_date(value)
        if (
            parsed_ts is not None
            and not isinstance(value, (str, Mapping))
            and not isinstance(value, Sequence)
        ):
            return [parsed_ts]

        frame_records = _records_from_frame(value, field=field)
        if frame_records is not None:
            return _observation_dates(
                frame_records, field=field, allow_labels=False,
                row_date=row_date, normalized=normalized
            )

    if isinstance(value, Mapping):
        dates: list[date] = []
        for key, nested in value.items():
            key_date = parse_asof_date(key)
            if key_date is not None:
                dates.append(key_date)
                dates.extend(
                    _observation_dates(
                        nested,
                        field=f"{field}.{key}",
                        allow_labels=True,
                        row_date=key_date,
                        normalized=normalized,
                    )
                )
                continue
            if _is_date_field_name(str(key)):
                nested_date = parse_asof_date(nested)
                if nested_date is None:
                    raise JevLookAheadError(
                        f"{field} date field {key!r} is not a parseable date: {nested!r}"
                    )
                dates.append(nested_date)
            else:
                dates.extend(
                    _observation_dates(
                        nested,
                        field=f"{field}.{key}",
                        allow_labels=True,
                        row_date=row_date,
                        normalized=normalized,
                    )
                )
        if not dates and row_date is not None:
            return [row_date]
        return dates

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        dates = []
        for i, item in enumerate(value):
            if item is None:
                raise JevLookAheadError(
                    f"{field}[{i}] is null and cannot be point-in-time validated"
                )
            item_dates = _observation_dates(
                item,
                field=f"{field}[{i}]",
                allow_labels=False,
                row_date=row_date,
                normalized=normalized,
            )
            if not item_dates:
                raise JevLookAheadError(
                    f"{field}[{i}] has no recognizable date field and cannot be "
                    "point-in-time validated"
                )
            dates.extend(item_dates)
        return dates

    raise _unsupported_market_format(field, value)


def assert_point_in_time(
    market_state: JevMarketState,
    *,
    payload: Mapping[str, Any] | None = None,
) -> None:
    """Reject a cutoff in the future, or market-data rows dated after the cutoff.

    Dates are compared as ``datetime.date`` values. The cutoff is the calendar
    day of ``analysis_date`` and includes that entire day. Analyst report
    strings are not scanned: a future event mentioned in prose is not treated
    as a leaked bar.

    When ``payload`` is provided it must be the normalized :meth:`JevMarketState.to_jev_state`
    dict that will be sent. Validation then reads market-data fields from that
    same object.
    """
    cutoff = parse_analysis_date(market_state.analysis_date)
    today = parse_analysis_date(get_current_date())
    if cutoff > today:
        raise JevLookAheadError(
            f"analysis_date {cutoff.isoformat()} is after today ({today.isoformat()}); "
            "refusing look-ahead"
        )

    leaked: list[str] = []
    for name in _MARKET_DATA_FIELDS:
        if payload is not None:
            blob = payload.get(name)
        else:
            raw = getattr(market_state, name)
            blob = None if raw is None else normalize_market_value(raw, field=name)
        for obs in _observation_dates(blob, field=name, normalized=True):
            if obs > cutoff:
                leaked.append(obs.isoformat())
    if leaked:
        raise JevLookAheadError(
            f"market state contains dates after analysis_date {cutoff.isoformat()}: "
            f"{sorted(set(leaked))}"
        )
