"""Normalize CSV, JSON, DataFrame and scalar inputs into JSON-safe market data."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from io import StringIO
from typing import Any

from tradingagents.agents.jev.dates import (
    _assert_date_column_agrees_with_index,
    _datetime_to_iso,
    _is_date_field_name,
    _is_numpy_datetime64,
    _is_numpy_nat,
    _numpy_datetime64_to_iso,
    parse_asof_date,
)
from tradingagents.decision_models.errors import JevError, JevLookAheadError

# Supported structured market-data shapes. Validation and the Jev payload use
# the same normalized copy (caller objects are not mutated):
# - None / bool / int / finite float
# - pandas/numpy scalars in that range; missing (NaN / NaT / pd.NA) → JSON null
# - date / datetime / pandas Timestamp / numpy datetime64 → ISO string
# - numpy.datetime64 units D, s, ms, us, ns only; out-of-range years error
# - YYYY-MM-DD, YYYY/MM/DD, or a complete ISO / "YYYY-MM-DD HH:MM[:SS]" timestamp
# - mapping of date or category key → value (returns={"1d": 0.01})
# - mapping with a date field (single bar / snapshot)
# - mapping with category/label strings (technical_indicators={"rsi": 55})
# - sequence of dated records (each row is checked on its own; None/scalars
#   cannot disable the per-row date requirement)
# - date-keyed mappings may pass that date down into nested records
# - CSV/TSV text with a Date/timestamp column → list of records (audited and sent).
#   A leading UTF-8 BOM is stripped. Parsing uses csv strict mode.
# - JSON object/array text (parsed, then audited)
# - pandas DataFrame → list of records. Duplicate column names (strip + casefold)
#   are rejected before to_dict. A DatetimeIndex / all-date index is kept as a
#   Date (or index-name) column when no date column exists. When a date column
#   and a date index both exist, they must agree under the rules in
#   :func:`_assert_date_column_agrees_with_index` (UTC instants when both are
#   timezone-aware clocks; local calendar date when one side is date-only).
#   The index timestamp is stored under an unused time field.
#   RangeIndex is never a date index. A table with neither a date column nor a
#   date index is rejected; "no date found" is not treated as valid.
# Anything else is rejected. Output is JSON-serializable
# (json.dumps(..., allow_nan=False)).


_UTF8_BOM = "\ufeff"


def _strip_leading_bom(text: str) -> str:
    if text.startswith(_UTF8_BOM):
        return text[len(_UTF8_BOM) :]
    return text


def _prepare_csv_sample(value: str) -> str | None:
    sample = _strip_leading_bom(value.strip()).strip()
    sample = _strip_leading_bom(sample).strip()
    if not sample or ("\n" not in sample and "\r" not in sample):
        return None
    return sample


def _csv_header_fields(header_line: str, delimiter: str) -> list[str] | None:
    try:
        rows = list(csv.reader([header_line], delimiter=delimiter, strict=True))
    except csv.Error:
        return None
    if not rows:
        return None
    return [_strip_leading_bom(cell).strip() for cell in rows[0]]


def _csv_delimiter(header_line: str) -> str | None:
    """Pick comma or tab from a csv-parsed header that contains a date column."""
    best: str | None = None
    best_width = 0
    for delim in (",", "\t"):
        fields = _csv_header_fields(header_line, delim)
        if fields is None:
            continue
        if not any(_is_date_field_name(name) for name in fields):
            continue
        if len(fields) > best_width:
            best_width = len(fields)
            best = delim
    return best


def _header_has_date_column_loosely(header_line: str) -> bool:
    """True when a delimited header looks like a date table even if csv.Error."""
    line = _strip_leading_bom(header_line).strip()
    for delim in (",", "\t"):
        for raw in line.split(delim):
            cell = _strip_leading_bom(raw.strip().strip('"')).strip()
            if _is_date_field_name(cell):
                return True
    return False


def _records_from_csv(value: str, *, field: str) -> list[dict[str, str]] | None:
    """Parse Date-column CSV/TSV into records, or return None if it is not a table.

    Uses :mod:`csv` with ``strict=True`` so quoted headers, in-field delimiters,
    and a leading UTF-8 BOM are handled. Duplicate column names (strip +
    casefold) and non-empty ragged rows raise. Identified tables that fail to
    parse raise rather than falling back to labels. Does not mutate ``value``.
    """
    sample = _prepare_csv_sample(value)
    if sample is None:
        return None
    header_line = sample.splitlines()[0]
    delimiter = _csv_delimiter(header_line)
    if delimiter is None:
        if _header_has_date_column_loosely(header_line):
            raise JevLookAheadError(f"{field} CSV could not be parsed")
        return None
    try:
        rows = list(
            csv.reader(StringIO(sample, newline=""), delimiter=delimiter, strict=True)
        )
    except csv.Error as exc:
        raise JevLookAheadError(f"{field} CSV could not be parsed") from exc
    if not rows:
        raise JevLookAheadError(f"{field} CSV has no header row")
    headers = [_strip_leading_bom(cell).strip() for cell in rows[0]]
    if not any(_is_date_field_name(name) for name in headers):
        raise JevLookAheadError(
            f"{field} CSV has no Date/timestamp column and cannot be point-in-time validated"
        )
    seen: dict[str, str] = {}
    for name in headers:
        key = name.casefold()
        if key in seen:
            raise JevLookAheadError(
                f"{field} CSV has duplicate column name {name!r} "
                f"(conflicts with {seen[key]!r})"
            )
        seen[key] = name
    width = len(headers)
    records: list[dict[str, str]] = []
    for row_i, row in enumerate(rows[1:], start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != width:
            raise JevLookAheadError(
                f"{field} CSV row {row_i} has {len(row)} fields, expected {width}"
            )
        records.append(dict(zip(headers, row, strict=True)))
    return records


def _looks_like_json_text(value: str) -> bool:
    text = value.strip()
    return bool(text) and text[0] in "{["


def _parse_json_text(value: str, *, field: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise JevLookAheadError(
            f"{field} contains a JSON-like string that is not valid JSON"
        ) from exc


def _is_dataframe_like(value: Any) -> bool:
    return callable(getattr(value, "to_dict", None)) and getattr(value, "columns", None) is not None


def _is_range_index(index: Any) -> bool:
    return type(index).__name__ == "RangeIndex"


def _is_named_date_index_type(index: Any) -> bool:
    return type(index).__name__ == "DatetimeIndex"


def _dates_from_index(index: Any) -> list[date] | None:
    """Return dates if ``index`` is a date index; None if it is not one.

    A :class:`~pandas.RangeIndex` is never a date index. Any other index is a
    date index only when every label parses as a market timestamp.
    """
    if index is None or _is_range_index(index):
        return None
    if len(index) == 0:
        return [] if _is_named_date_index_type(index) else None
    dates: list[date] = []
    for label in index:
        parsed = parse_asof_date(label)
        if parsed is None:
            return None
        dates.append(parsed)
    return dates


def _date_columns(columns: Any) -> list[str]:
    return [str(name) for name in columns if _is_date_field_name(str(name))]


def _field_identity(name: Any) -> str:
    return str(name).strip().casefold()


def _assert_unique_frame_columns(columns: Any, *, field: str) -> None:
    seen: dict[str, str] = {}
    for raw in columns:
        name = str(raw)
        key = _field_identity(name)
        if key in seen:
            raise JevLookAheadError(
                f"{field} table has duplicate column name {name!r} "
                f"(conflicts with {seen[key]!r})"
            )
        seen[key] = name


_INDEX_TIME_FIELD_CANDIDATES = (
    "timestamp",
    "datetime",
    "time",
    "as_of",
    "asof",
    "index_timestamp",
)


def _choose_index_time_field(
    index: Any, row: Mapping[str, Any], *, field: str, row_i: int
) -> str:
    occupied = {_field_identity(key) for key in row}
    candidates: list[str] = []
    index_name = getattr(index, "name", None)
    if index_name is not None and str(index_name).strip():
        candidates.append(str(index_name).strip())
    candidates.extend(_INDEX_TIME_FIELD_CANDIDATES)
    seen: set[str] = set()
    for name in candidates:
        ident = _field_identity(name)
        if ident in seen:
            continue
        seen.add(ident)
        if ident not in occupied:
            return name
    raise JevLookAheadError(
        f"{field}[{row_i}] cannot preserve date index without overwriting an existing field"
    )


def _index_date_column_name(index: Any) -> str:
    name = getattr(index, "name", None)
    if name is not None and _is_date_field_name(str(name)):
        return str(name)
    return "Date"


def _records_from_frame(value: Any, *, field: str) -> list[Any] | None:
    """Copy a DataFrame-like object to records, preserving a date index.

    Does not mutate ``value``. Returns None when ``value`` is not frame-like.
    """
    if not _is_dataframe_like(value):
        return None
    columns = getattr(value, "columns", None)
    if columns is None:
        return None
    _assert_unique_frame_columns(columns, field=field)
    to_dict = value.to_dict
    try:
        records = to_dict(orient="records")
    except TypeError:
        return None
    if not isinstance(records, list):
        return None

    rows = [dict(row) if isinstance(row, Mapping) else row for row in records]
    index = getattr(value, "index", None)
    date_cols = _date_columns(columns)
    index_dates = _dates_from_index(index)

    if index_dates is None and not date_cols:
        raise JevLookAheadError(
            f"{field} table has no date column or date index and cannot be "
            "point-in-time validated"
        )

    if index_dates is not None and date_cols:
        if len(index_dates) != len(rows):
            raise JevLookAheadError(
                f"{field} date index length {len(index_dates)} does not match "
                f"{len(rows)} rows"
            )
        for i, (row, idx_label) in enumerate(zip(rows, index, strict=True)):
            if not isinstance(row, Mapping):
                raise JevLookAheadError(
                    f"{field}[{i}] is not a mapping after DataFrame conversion"
                )
            copied = dict(row)
            for col in date_cols:
                col_value = copied.get(col)
                _assert_date_column_agrees_with_index(
                    col_value, idx_label, field=field, col=col, row_i=i
                )
            time_field = _choose_index_time_field(index, copied, field=field, row_i=i)
            copied[time_field] = idx_label
            rows[i] = copied
        return rows

    if index_dates is not None:
        col_name = _index_date_column_name(index)
        if len(index_dates) != len(rows):
            raise JevLookAheadError(
                f"{field} date index length {len(index_dates)} does not match "
                f"{len(rows)} rows"
            )
        for row, label in zip(rows, index, strict=True):
            if not isinstance(row, dict):
                raise JevLookAheadError(
                    f"{field} row is not a mapping after DataFrame conversion"
                )
            row[col_name] = label
        return rows

    return rows


def _is_scalar_missing(value: Any) -> bool:
    if value is None:
        return True
    if _is_numpy_nat(value):
        return True
    if isinstance(value, (Mapping, list, tuple, str, bytes, bytearray)):
        return False
    if _is_dataframe_like(value):
        return False
    shape = getattr(value, "shape", None)
    if shape not in (None, ()):
        return False
    try:
        import pandas as pd

        result = pd.isna(value)
    except (ImportError, ValueError, TypeError):
        result = None
    if isinstance(result, bool):
        return result
    item = getattr(result, "item", None)
    if result is not None and callable(item) and getattr(result, "shape", None) in (None, ()):
        try:
            return bool(item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _is_numpy_or_pandas_scalar(value: Any) -> bool:
    module = getattr(type(value), "__module__", "") or ""
    if not (module.startswith("numpy") or module.startswith("pandas")):
        return False
    if _is_dataframe_like(value):
        return False
    if isinstance(value, (Mapping, list, tuple, str, bytes, bytearray)):
        return False
    if type(value).__name__ in {
        "Series",
        "Index",
        "DatetimeIndex",
        "RangeIndex",
        "PeriodIndex",
        "TimedeltaIndex",
        "MultiIndex",
        "DataFrame",
    }:
        return False
    return callable(getattr(value, "item", None))


def _unsupported_json_type(field: str, value: Any) -> JevError:
    return JevError(
        f"{field} has unsupported type {type(value).__name__} that cannot be "
        "normalized to JSON; pass dated records, a Date-column CSV, numeric "
        "scalars, or omit it"
    )


def _normalize_mapping_key(key: Any, *, field: str) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, datetime):
        return _datetime_to_iso(key)
    if isinstance(key, date):
        return key.isoformat()
    if _is_numpy_datetime64(key):
        if _is_numpy_nat(key):
            raise _unsupported_json_type(f"{field} key", key)
        return _numpy_datetime64_to_iso(key, field=f"{field} key")
    if isinstance(key, bool) or key is None:
        raise _unsupported_json_type(f"{field} key", key)
    if isinstance(key, (int, float)):
        if isinstance(key, float) and (math.isnan(key) or math.isinf(key)):
            raise JevError(f"{field} mapping key is not a finite number: {key!r}")
        return str(key)
    parsed = parse_asof_date(key)
    iso = getattr(key, "isoformat", None)
    if callable(iso):
        try:
            text = iso()
        except (TypeError, ValueError):
            text = None
        if isinstance(text, str) and text:
            return text
    if parsed is not None:
        return parsed.isoformat()
    raise _unsupported_json_type(f"{field} key", key)


def normalize_market_value(value: Any, *, field: str) -> Any:
    """Return a JSON-safe deep copy of ``value``. Does not mutate the input."""
    if _is_scalar_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, datetime):
        return _datetime_to_iso(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        if _looks_like_json_text(value):
            return normalize_market_value(_parse_json_text(value, field=field), field=field)
        csv_records = _records_from_csv(value, field=field)
        if csv_records is not None:
            return [
                normalize_market_value(row, field=f"{field}[{i}]")
                for i, row in enumerate(csv_records)
            ]
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            raise JevError(f"{field} contains a non-finite number that is not valid JSON")
        return float(value)

    to_pydatetime = getattr(value, "to_pydatetime", None)
    if (
        callable(to_pydatetime)
        and not isinstance(value, Mapping)
        and not isinstance(value, Sequence)
    ):
        try:
            converted = to_pydatetime()
        except (TypeError, ValueError):
            converted = None
        if isinstance(converted, datetime):
            return _datetime_to_iso(converted)
        if isinstance(converted, date):
            return converted.isoformat()

    if _is_numpy_datetime64(value):
        return _numpy_datetime64_to_iso(value, field=field)

    if _is_numpy_or_pandas_scalar(value):
        try:
            inner = value.item()
        except (ValueError, TypeError) as exc:
            raise _unsupported_json_type(field, value) from exc
        if inner is value:
            raise _unsupported_json_type(field, value)
        return normalize_market_value(inner, field=field)

    frame_records = _records_from_frame(value, field=field)
    if frame_records is not None:
        return [
            normalize_market_value(item, field=f"{field}[{i}]")
            for i, item in enumerate(frame_records)
        ]

    if isinstance(value, Mapping):
        return {
            _normalize_mapping_key(key, field=field): normalize_market_value(
                nested, field=f"{field}.{key}"
            )
            for key, nested in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            normalize_market_value(item, field=f"{field}[{i}]")
            for i, item in enumerate(value)
        ]

    raise _unsupported_json_type(field, value)
