"""TypeSafe Jev (System One) client.

Jev is not a generative LLM. It evaluates a JSON/text state against typed
questions (Choice / Score / Noul) and returns structured answers. This module
is therefore a decision backend, not an ``llm_clients`` provider.

Official interface (https://docs.typesafe.ai/sdk/python):

- Package: ``typesafe-sdk`` (optional extra: ``pip install tradingagents[jev]``)
- Auth: ``TYPESAFE_API_KEY`` (community alias ``TYPESAFE_AI_API_KEY``)
- Default model: ``jev-latest``
- Default timeout: 10s
- Call: ``TypeSafeClient.system_one(state=..., questions=...)``

Business code must not import ``typesafe_sdk`` types. Use :class:`JevClient`
and the dataclasses in :mod:`tradingagents.decision_models.types`.
"""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Mapping
from typing import Any

from tradingagents.decision_models.errors import (
    JevError,
    JevMalformedResponseError,
    JevNotConfiguredError,
    JevProviderError,
    JevTimeoutError,
)
from tradingagents.decision_models.types import (
    JevChoiceResult,
    JevEvaluateResult,
    JevNoulResult,
    JevScoreResult,
)
from tradingagents.decision_models.validation import (
    PROBABILITY_SUM_ABS_TOL as PROBABILITY_SUM_ABS_TOL,  # Compatibility export.
    finite_float,
    probability_sum,
    unit_interval,
)

# Official SDK env names: https://docs.typesafe.ai/sdk/python/api/constants
_API_KEY_ENV = "TYPESAFE_API_KEY"
_API_KEY_ALIAS_ENV = "TYPESAFE_AI_API_KEY"
_DEFAULT_MODEL = "jev-latest"
_DEFAULT_TIMEOUT = 10.0


def resolve_typesafe_api_key(explicit: str | None = None) -> str | None:
    """Return the TypeSafe API key, preferring an explicit value then official env."""
    if explicit and explicit.strip():
        return explicit.strip()
    for name in (_API_KEY_ENV, _API_KEY_ALIAS_ENV):
        raw = os.environ.get(name)
        if raw and raw.strip() and raw.strip() != "placeholder":
            return raw.strip()
    return None


def _env_nonempty(*names: str) -> str | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return None


def resolve_jev_model(
    explicit: str | None = None,
    *,
    config_model: str | None = None,
    config_explicit: bool = False,
) -> str:
    """Resolve the Jev model id.

    Precedence:
    1. Constructor ``model=`` / overlay ``{"jev_model": ...}``
    2. ``TRADINGAGENTS_JEV_MODEL`` (project overlay already in DEFAULT_CONFIG)
    3. ``TYPESAFE_DEFAULT_MODEL`` (official SDK env)
    4. Built-in ``jev-latest``

    The DEFAULT_CONFIG sentinel ``jev-latest`` must not hide (3).
    """
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    if config_explicit and config_model is not None and str(config_model).strip():
        return str(config_model).strip()
    project = _env_nonempty("TRADINGAGENTS_JEV_MODEL")
    if project:
        return project
    sdk_default = _env_nonempty("TYPESAFE_DEFAULT_MODEL")
    if sdk_default:
        return sdk_default
    if config_model is not None and str(config_model).strip():
        return str(config_model).strip()
    return _DEFAULT_MODEL


def resolve_jev_timeout(
    explicit: float | None = None,
    *,
    config_timeout: Any = None,
    config_explicit: bool = False,
) -> float:
    if explicit is not None:
        return float(explicit)
    if config_explicit and config_timeout is not None and config_timeout != "":
        return float(config_timeout)
    env = _env_nonempty("TRADINGAGENTS_JEV_TIMEOUT")
    if env:
        return float(env)
    if config_timeout is not None and config_timeout != "":
        return float(config_timeout)
    return _DEFAULT_TIMEOUT


def resolve_jev_base_url(
    explicit: str | None = None,
    *,
    config_url: str | None = None,
    config_explicit: bool = False,
) -> str | None:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    if config_explicit and config_url is not None and str(config_url).strip():
        return str(config_url).strip()
    project = _env_nonempty("TRADINGAGENTS_JEV_BASE_URL")
    if project:
        return project
    sdk = _env_nonempty("TYPESAFE_BASE_URL")
    if sdk:
        return sdk
    if config_url is not None and str(config_url).strip():
        return str(config_url).strip()
    return None


def _project_config_copy() -> dict[str, Any]:
    from tradingagents.default_config import DEFAULT_CONFIG

    return copy.deepcopy(DEFAULT_CONFIG)


def _attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_finite_float(value: Any, *, what: str) -> float:
    return finite_float(value, what=what, coerce=True)


def _as_unit_interval(value: Any, *, what: str) -> float:
    return unit_interval(value, what=what, coerce=True)


def _as_str_float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise JevMalformedResponseError("probabilities must be a non-empty mapping")
    out: dict[str, float] = {}
    for key, raw in value.items():
        out[str(key)] = _as_unit_interval(raw, what=f"probability for {key!r}")
    probability_sum(out.values())
    return out


def _optional_confidence(value: Any) -> float | None:
    if value is None:
        return None
    return _as_unit_interval(value, what="confidence")


def _answer_type(answer: Any) -> str:
    raw = _attr_or_key(answer, "type")
    if raw:
        return str(raw)
    if _attr_or_key(answer, "choice") is not None:
        return "choice"
    if _attr_or_key(answer, "score") is not None:
        return "score"
    if _attr_or_key(answer, "noul") is not None:
        return "noul"
    raise JevMalformedResponseError(f"cannot determine answer type: {answer!r}")


def normalize_system_one_response(
    response: Any, *, latency_ms: float | None = None
) -> JevEvaluateResult:
    """Convert a TypeSafe ``SystemOneResponse`` (or equivalent dict) to internal types."""
    answers = _attr_or_key(response, "answers")
    if not isinstance(answers, Mapping) or not answers:
        raise JevMalformedResponseError("System One response has no answers")

    choices: dict[str, JevChoiceResult] = {}
    scores: dict[str, JevScoreResult] = {}
    nouls: dict[str, JevNoulResult] = {}

    for name, answer in answers.items():
        kind = _answer_type(answer)
        if kind == "choice":
            choice = _attr_or_key(answer, "choice")
            if not choice:
                raise JevMalformedResponseError(f"choice answer {name!r} is missing 'choice'")
            confidence = _attr_or_key(answer, "confidence")
            choices[str(name)] = JevChoiceResult(
                choice=str(choice),
                probabilities=_as_str_float_map(_attr_or_key(answer, "probabilities")),
                confidence=_optional_confidence(confidence),
            )
        elif kind == "score":
            score = _attr_or_key(answer, "score")
            if score is None:
                raise JevMalformedResponseError(f"score answer {name!r} is missing 'score'")
            confidence = _attr_or_key(answer, "confidence")
            legend_raw = _attr_or_key(answer, "legend") or {}
            legend = (
                {str(k): str(v) for k, v in legend_raw.items()}
                if isinstance(legend_raw, Mapping)
                else None
            )
            scores[str(name)] = JevScoreResult(
                score=_as_finite_float(score, what=f"score {name!r}"),
                probabilities=_as_str_float_map(_attr_or_key(answer, "probabilities")),
                confidence=_optional_confidence(confidence),
                legend=legend,
            )
        elif kind == "noul":
            noul = _attr_or_key(answer, "noul")
            if noul is None:
                raise JevMalformedResponseError(f"noul answer {name!r} is missing 'noul'")
            nouls[str(name)] = JevNoulResult(noul=_as_unit_interval(noul, what=f"noul {name!r}"))
        else:
            raise JevMalformedResponseError(f"unsupported answer type {kind!r} for {name!r}")

    model = _attr_or_key(response, "model")
    return JevEvaluateResult(
        choices=choices,
        scores=scores,
        nouls=nouls,
        model=None if model is None else str(model),
        latency_ms=latency_ms,
        raw_answers=dict(answers),
    )


class JevClient:
    """Thin adapter around TypeSafe ``TypeSafeClient.system_one``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        base_url: str | None = None,
        config: Mapping[str, Any] | None = None,
    ):
        cfg = _project_config_copy()
        overlay = dict(config or {})
        if overlay:
            cfg.update(overlay)
        overlay_keys = set(overlay)

        self.api_key = resolve_typesafe_api_key(api_key)
        self.model = resolve_jev_model(
            model,
            config_model=cfg.get("jev_model"),
            config_explicit="jev_model" in overlay_keys,
        )
        self.timeout = resolve_jev_timeout(
            timeout,
            config_timeout=cfg.get("jev_timeout"),
            config_explicit="jev_timeout" in overlay_keys,
        )
        self.base_url = resolve_jev_base_url(
            base_url,
            config_url=cfg.get("jev_base_url"),
            config_explicit="jev_base_url" in overlay_keys,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> JevClient:
        """Build from DEFAULT_CONFIG, with an optional overlay that is not written back."""
        return cls(config=config)

    def evaluate(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
    ) -> JevEvaluateResult:
        """Send ``state`` + typed ``questions`` to Jev and return normalized answers.

        ``questions`` uses the documented dictionary form (``type`` / ``instructions``
        / ``criteria``) so callers never construct SDK objects.
        """
        if not questions:
            raise JevError("questions must be a non-empty mapping")
        if not self.api_key:
            raise JevNotConfiguredError(
                "TypeSafe API key is not set. Export TYPESAFE_API_KEY "
                "(or the alias TYPESAFE_AI_API_KEY)."
            )
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as exc:
            raise JevNotConfiguredError(
                "The TypeSafe SDK is not installed. "
                "Install typesafe-sdk or run: pip install 'tradingagents[jev]'"
            ) from exc

        call_model = model or self.model
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "model": call_model,
            "timeout": self.timeout,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url

        started = time.perf_counter()
        try:
            with TypeSafeClient(**kwargs) as client:
                response = client.system_one(
                    state=state,
                    questions=dict(questions),
                    model=call_model,
                    timeout=self.timeout,
                )
        except JevError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            return normalize_system_one_response(response, latency_ms=latency_ms)
        except JevError:
            raise
        except Exception as exc:
            raise JevMalformedResponseError(f"failed to normalize Jev response: {exc}") from exc


def _mro_names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def _wrap_provider_error(exc: BaseException) -> JevError:
    """Map official typesafe_sdk exceptions onto JevError subclasses.

    Names are matched on the real MRO so a stand-in hierarchy in tests (and the
    optional SDK, when installed) both work. Validation failures are malformed
    responses, not provider outages.
    """
    names = _mro_names(exc)
    message = str(exc) or type(exc).__name__

    if "TypeSafeAPIResponseValidationError" in names:
        return JevMalformedResponseError(message)
    if isinstance(exc, TimeoutError) or "TypeSafeAPITimeoutError" in names:
        return JevTimeoutError(message)
    if "TypeSafeAuthenticationError" in names:
        return JevNotConfiguredError(message)
    if names & {
        "TypeSafeError",
        "TypeSafeAPIError",
        "TypeSafeAPIConnectionError",
        "TypeSafeBadRequestError",
        "TypeSafePermissionDeniedError",
        "TypeSafeNotFoundError",
        "TypeSafeUnprocessableEntityError",
        "TypeSafeRateLimitError",
        "TypeSafeInternalServerError",
    }:
        return JevProviderError(message)
    return JevProviderError(message)
