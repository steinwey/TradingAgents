"""Date parsing and timestamp comparison, independent of market-data formats."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any

from tradingagents.decision_models.errors import JevLookAheadError

# 日期列/键名白名单（忽略大小写）。只有这些名字会被当成时间戳来解析。
# as_of / asof 都要写：casefold 不管下划线，两种拼法来自不同数据源。
_DATE_FIELD_NAMES = frozenset({
    "date",
    "timestamp",
    "time",
    "datetime",
    "as_of",  # as-of date，常见于 API / JSON
    "asof",  # 无下划线写法（如部分接口、pandas merge_asof）
})

# 截止日只认日历 YYYY-MM-DD：覆盖整天，不支持盘中截止（带时刻会假装更精确）。
_ANALYSIS_DATE_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")

# 行情时间戳更宽（外部数据源），但必须整串匹配，禁止 2026-09-01junk 这种前缀切片。
# 解析后仍落到 date，和 cutoff 只比日历日。
_MARKET_DATE_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")  # YYYY-MM-DD
_MARKET_SLASH_DATE_RE = re.compile(r"\A\d{4}/\d{2}/\d{2}\Z")  # YYYY/MM/DD
_MARKET_DATETIME_RE = re.compile(  # YYYY-MM-DD[T ]HH:MM[:SS[.frac]][±HH:MM]
    r"\A\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:[+-]\d{2}:\d{2})?\Z"
)

def parse_analysis_date(value: Any) -> date:
    """Parse a strict calendar ``YYYY-MM-DD``. Reject time suffixes and junk."""
    if not isinstance(value, str):
        raise JevLookAheadError(f"analysis_date must be YYYY-MM-DD, got {value!r}")
    text = value.strip()
    if not _ANALYSIS_DATE_RE.fullmatch(text):
        raise JevLookAheadError(
            f"analysis_date must be YYYY-MM-DD (date only, no time suffix), got {value!r}"
        )
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise JevLookAheadError(
            f"analysis_date must be YYYY-MM-DD, got {value!r}"
        ) from exc


def _parse_market_date_string(text: str) -> date | None:
    """Parse a complete supported market timestamp. Never slice a prefix."""
    text = text.strip()
    if not text:
        return None
    if text.endswith("Z") and "T" in text:
        text = text[:-1] + "+00:00"
    if _MARKET_SLASH_DATE_RE.fullmatch(text):
        try:
            return datetime.strptime(text, "%Y/%m/%d").date()
        except ValueError:
            return None
    if _MARKET_DATE_RE.fullmatch(text):
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None
    if _MARKET_DATETIME_RE.fullmatch(text):
        try:
            return datetime.fromisoformat(text).date()
        except ValueError:
            return None
    return None


def _is_numpy_datetime64(value: Any) -> bool:
    return type(value).__name__ == "datetime64"


def _is_numpy_nat(value: Any) -> bool:
    if not _is_numpy_datetime64(value):
        return False
    try:
        import numpy as np

        return bool(np.isnat(value))
    except (ImportError, TypeError, ValueError):
        return str(value) == "NaT"


_NUMPY_DATETIME64_UNITS = frozenset({"D", "s", "ms", "us", "ns"})


def _numpy_datetime64_to_iso(value: Any, *, field: str) -> str:
    """Convert numpy.datetime64 to a parseable ISO string. Never emit an integer."""
    try:
        import numpy as np

        unit = np.datetime_data(value.dtype)[0]
    except (ImportError, TypeError, ValueError, AttributeError) as exc:
        raise JevLookAheadError(
            f"{field} numpy.datetime64 {value!r} could not be converted to a calendar date"
        ) from exc
    if unit not in _NUMPY_DATETIME64_UNITS:
        raise JevLookAheadError(
            f"{field} numpy.datetime64 unit {unit!r} is not supported; "
            "expected one of D, s, ms, us, ns"
        )
    try:
        converted = value.astype("datetime64[us]") if unit == "ns" else value
        inner = converted.item()
    except (OverflowError, ValueError, OSError, TypeError) as exc:
        raise JevLookAheadError(
            f"{field} numpy.datetime64 {value!r} is outside the supported date range"
        ) from exc
    if isinstance(inner, datetime):
        return inner.isoformat()
    if isinstance(inner, date):
        return inner.isoformat()
    raise JevLookAheadError(
        f"{field} numpy.datetime64 {value!r} is outside the supported date range"
    )


def parse_asof_date(value: Any) -> date | None:
    """Parse a supported market timestamp to ``datetime.date``, or return None.

    Strings must match a complete supported form. Prefixes such as
    ``2026-09-01junk`` are not accepted.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if (
        callable(to_pydatetime)
        and not isinstance(value, str)
        and not isinstance(value, Mapping)
        and not isinstance(value, Sequence)
    ):
        try:
            converted = to_pydatetime()
        except (TypeError, ValueError):
            converted = None
        if isinstance(converted, datetime):
            return converted.date()
        if isinstance(converted, date):
            return converted
    if isinstance(value, str):
        return _parse_market_date_string(value)
    if _is_numpy_datetime64(value):
        if _is_numpy_nat(value):
            return None
        return _parse_market_date_string(_numpy_datetime64_to_iso(value, field="datetime64"))
    return None


def _is_date_field_name(name: str) -> bool:
    return name.casefold() in _DATE_FIELD_NAMES


def _is_plain_date_string(value: str) -> bool:
    return parse_asof_date(value) is not None


def _as_clock_datetime(value: Any) -> datetime | None:
    """Return a datetime when ``value`` carries a clock time, else None.

    Timezone information is preserved. Date-only values (``date``,
    ``YYYY-MM-DD``, ``datetime64[D]``) return None.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z") and "T" in text:
            text = text[:-1] + "+00:00"
        if _MARKET_DATETIME_RE.fullmatch(text):
            try:
                return datetime.fromisoformat(text)
            except ValueError:
                return None
        return None
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if (
        callable(to_pydatetime)
        and not isinstance(value, (str, Mapping))
        and not isinstance(value, Sequence)
    ):
        try:
            converted = to_pydatetime()
        except (TypeError, ValueError):
            converted = None
        if isinstance(converted, datetime):
            return converted
        return None
    if _is_numpy_datetime64(value) and not _is_numpy_nat(value):
        try:
            import numpy as np

            unit = np.datetime_data(value.dtype)[0]
        except (ImportError, TypeError, ValueError, AttributeError):
            return None
        if unit == "D":
            return None
        try:
            return datetime.fromisoformat(_numpy_datetime64_to_iso(value, field="datetime64"))
        except (JevLookAheadError, ValueError):
            return None
    return None


def _is_timezone_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _assert_date_column_agrees_with_index(
    col_value: Any,
    idx_label: Any,
    *,
    field: str,
    col: str,
    row_i: int,
) -> None:
    """Require a date column and a date index to describe the same observation.

    - Both timezone-aware clocks: compare UTC instants only. Calendar dates
      from each zone are not compared first, so the same instant with
      different local dates is not a conflict.
    - Both naive clocks: compare the naive datetime values.
    - Mixed aware/naive clocks: reject; the timezone is not inferred.
    - At least one side is date-only: compare calendar dates. A date-only
      value must match the other timestamp's **local** calendar date (the
      date from that timestamp's own offset, not the UTC date).
    """
    col_dt = _as_clock_datetime(col_value)
    idx_dt = _as_clock_datetime(idx_label)
    if col_dt is not None and idx_dt is not None:
        col_aware = _is_timezone_aware(col_dt)
        idx_aware = _is_timezone_aware(idx_dt)
        if col_aware != idx_aware:
            raise JevLookAheadError(
                f"{field} has conflicting timestamp column {col!r} and date index "
                f"at row {row_i}: timezone-aware and naive datetimes cannot be "
                f"compared (column={col_value!r} index={idx_label!r})"
            )
        if col_aware and idx_aware:
            if col_dt.astimezone(timezone.utc) != idx_dt.astimezone(timezone.utc):
                raise JevLookAheadError(
                    f"{field} has conflicting timestamp column {col!r} and date index "
                    f"at row {row_i}: column={col_value!r} index={idx_label!r}"
                )
            return
        if col_dt.replace(tzinfo=None) != idx_dt.replace(tzinfo=None):
            raise JevLookAheadError(
                f"{field} has conflicting timestamp column {col!r} and date index "
                f"at row {row_i}: column={col_value!r} index={idx_label!r}"
            )
        return

    col_date = parse_asof_date(col_value)
    if col_date is None:
        raise JevLookAheadError(
            f"{field} date field {col!r} is not a parseable date: {col_value!r}"
        )
    idx_date = parse_asof_date(idx_label)
    if idx_date is None:
        raise JevLookAheadError(
            f"{field} date index at row {row_i} is not a parseable date: {idx_label!r}"
        )
    if col_date != idx_date:
        raise JevLookAheadError(
            f"{field} has conflicting date column {col!r} and date index "
            f"at row {row_i}: column={col_date.isoformat()} "
            f"index={idx_date.isoformat()}"
        )


def _datetime_to_iso(value: datetime) -> str:
    """Serialize a datetime (including pandas Timestamp) to a parseable ISO string."""
    to_py = getattr(value, "to_pydatetime", None)
    if callable(to_py):
        try:
            converted = to_py()
        except (TypeError, ValueError):
            converted = None
        if isinstance(converted, datetime):
            return converted.isoformat()
    return value.isoformat()
