"""Unit tests for JevClient normalization and configuration.

No live TypeSafe calls: evaluate() uses stand-in SDK exceptions, and missing-key
paths never reach the network.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from tradingagents.decision_models.errors import (
    JevMalformedResponseError,
    JevNotConfiguredError,
    JevProviderError,
    JevTimeoutError,
)
from tradingagents.decision_models.jev_client import (
    JevClient,
    PROBABILITY_SUM_ABS_TOL,
    normalize_system_one_response,
    resolve_jev_model,
    resolve_typesafe_api_key,
)
from tradingagents.default_config import DEFAULT_CONFIG


def _choice_answer(**overrides):
    payload = {
        "type": "choice",
        "choice": "BULLISH",
        "confidence": 0.76,
        "probabilities": {"BULLISH": 0.70, "NEUTRAL": 0.20, "BEARISH": 0.10},
    }
    payload.update(overrides)
    return payload


def _score_answer(score, label="MEDIUM"):
    return {
        "type": "score",
        "score": score,
        "confidence": 0.8,
        "legend": {"0": "LOW", "1": label},
        "probabilities": {"0": 0.1, "1": 0.9},
    }


# Stand-ins matching official typesafe_sdk exception names and inheritance:
# https://docs.typesafe.ai/sdk/python/api/exceptions
class TypeSafeError(Exception):
    pass


class TypeSafeAPIError(TypeSafeError):
    def __init__(self, message="api error", status=None):
        super().__init__(message)
        self.status = status


class TypeSafeAPITimeoutError(TypeSafeError, TimeoutError):
    def __init__(self, message="timed out", timeout=10.0):
        super().__init__(message)
        self.timeout = timeout


class TypeSafeAuthenticationError(TypeSafeAPIError):
    pass


class TypeSafeAPIResponseValidationError(TypeSafeAPIError):
    def __init__(self, message="invalid body", field_path="answers.direction.confidence"):
        super().__init__(message, status=200)
        self.field_path = field_path


class TypeSafeRateLimitError(TypeSafeAPIError):
    def __init__(self, message="429"):
        super().__init__(message, status=429)


class _FakeTypeSafeClient:
    """Minimal context-manager client; not a MagicMock."""

    error: BaseException | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def system_one(self, **kwargs):
        if self.error is not None:
            raise self.error
        return {
            "model": "jev-latest",
            "answers": {"q": {"type": "noul", "noul": 0.5}},
        }


def _install_sdk(monkeypatch, error: BaseException):
    client_cls = type("TypeSafeClient", (_FakeTypeSafeClient,), {"error": error})
    mod = types.ModuleType("typesafe_sdk")
    mod.TypeSafeClient = client_cls
    monkeypatch.setitem(sys.modules, "typesafe_sdk", mod)


@pytest.mark.unit
def test_default_config_keeps_jev_off_the_llm_provider():
    assert DEFAULT_CONFIG["llm_provider"] != "jev"
    assert DEFAULT_CONFIG["jev_model"] == "jev-latest"
    assert DEFAULT_CONFIG["jev_timeout"] == 10.0


@pytest.mark.unit
def test_resolve_key_prefers_official_env(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "official")
    monkeypatch.setenv("TYPESAFE_AI_API_KEY", "alias")
    assert resolve_typesafe_api_key() == "official"


@pytest.mark.unit
def test_resolve_key_falls_back_to_alias(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_AI_API_KEY", "alias")
    assert resolve_typesafe_api_key() == "alias"


@pytest.mark.unit
def test_resolve_key_ignores_placeholder(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "placeholder")
    monkeypatch.delenv("TYPESAFE_AI_API_KEY", raising=False)
    assert resolve_typesafe_api_key() is None


@pytest.mark.unit
def test_project_model_env_beats_sdk_default(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_JEV_MODEL", "jev-project")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-from-sdk")
    assert resolve_jev_model() == "jev-project"
    assert JevClient.from_config(None).model == "jev-project"


@pytest.mark.unit
def test_sdk_default_model_used_when_project_env_unset(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_JEV_MODEL", raising=False)
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-from-sdk")
    assert JevClient.from_config(None).model == "jev-from-sdk"
    assert JevClient().model == "jev-from-sdk"


@pytest.mark.unit
def test_explicit_config_overlay_beats_both_model_envs(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_JEV_MODEL", "jev-project")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-from-sdk")
    client = JevClient.from_config({"jev_model": "jev-overlay"})
    assert client.model == "jev-overlay"


@pytest.mark.unit
def test_from_config_none_loads_timeout_and_does_not_mutate_defaults(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_JEV_TIMEOUT", "15.5")
    original_timeout = DEFAULT_CONFIG["jev_timeout"]
    original_model = DEFAULT_CONFIG["jev_model"]
    client = JevClient.from_config(None)
    assert client.timeout == 15.5
    JevClient.from_config({"jev_model": "must-not-leak", "jev_timeout": 99})
    assert DEFAULT_CONFIG["jev_timeout"] == original_timeout
    assert DEFAULT_CONFIG["jev_model"] == original_model


@pytest.mark.unit
def test_normalize_choice_and_scores():
    response = {
        "model": "jev-1.13.0",
        "answers": {
            "direction": _choice_answer(),
            "signal_strength": _score_answer(3.7, "STRONG"),
            "market_risk": _score_answer(2.1, "MEDIUM"),
        },
    }
    result = normalize_system_one_response(response, latency_ms=12.5)
    assert result.model == "jev-1.13.0"
    assert result.latency_ms == 12.5
    assert result.choices["direction"].choice == "BULLISH"
    assert result.choices["direction"].probabilities["BULLISH"] == pytest.approx(0.70)
    assert result.choices["direction"].confidence == pytest.approx(0.76)
    assert result.scores["signal_strength"].score == pytest.approx(3.7)
    assert result.scores["market_risk"].score == pytest.approx(2.1)


@pytest.mark.unit
def test_normalize_accepts_sdk_style_objects():
    answer = SimpleNamespace(
        type="choice",
        choice="NEUTRAL",
        confidence=0.5,
        probabilities={"BULLISH": 0.2, "NEUTRAL": 0.6, "BEARISH": 0.2},
    )
    response = SimpleNamespace(model="jev-latest", answers={"direction": answer})
    result = normalize_system_one_response(response)
    assert result.choices["direction"].choice == "NEUTRAL"


@pytest.mark.unit
def test_normalize_rejects_missing_probabilities():
    with pytest.raises(JevMalformedResponseError, match="probabilities"):
        normalize_system_one_response(
            {"model": "jev-latest", "answers": {"direction": _choice_answer(probabilities=None)}}
        )


@pytest.mark.unit
def test_normalize_rejects_empty_answers():
    with pytest.raises(JevMalformedResponseError, match="no answers"):
        normalize_system_one_response({"model": "jev-latest", "answers": {}})


@pytest.mark.unit
def test_normalize_rejects_probability_above_one():
    with pytest.raises(JevMalformedResponseError, match="\\[0, 1\\]"):
        normalize_system_one_response(
            {
                "answers": {
                    "direction": _choice_answer(
                        probabilities={"BULLISH": 2.0, "NEUTRAL": 0.0, "BEARISH": 0.0}
                    )
                }
            }
        )


@pytest.mark.unit
def test_normalize_rejects_negative_probability():
    with pytest.raises(JevMalformedResponseError, match="\\[0, 1\\]"):
        normalize_system_one_response(
            {
                "answers": {
                    "direction": _choice_answer(
                        probabilities={"BULLISH": 0.7, "NEUTRAL": 0.4, "BEARISH": -0.1}
                    )
                }
            }
        )


@pytest.mark.unit
def test_normalize_rejects_boolean_probability():
    with pytest.raises(JevMalformedResponseError, match="finite"):
        normalize_system_one_response(
            {
                "answers": {
                    "direction": _choice_answer(
                        probabilities={"BULLISH": True, "NEUTRAL": 0.0, "BEARISH": 0.0}
                    )
                }
            }
        )


@pytest.mark.unit
def test_normalize_rejects_probability_sum_outside_tolerance():
    with pytest.raises(JevMalformedResponseError, match="sum to 1"):
        normalize_system_one_response(
            {
                "answers": {
                    "direction": _choice_answer(
                        probabilities={"BULLISH": 0.5, "NEUTRAL": 0.5, "BEARISH": 0.5}
                    )
                }
            }
        )


@pytest.mark.unit
def test_normalize_accepts_probability_sum_within_tolerance():
    result = normalize_system_one_response(
        {
            "answers": {
                "direction": _choice_answer(
                    probabilities={
                        "BULLISH": 0.70,
                        "NEUTRAL": 0.20,
                        "BEARISH": 0.10 + PROBABILITY_SUM_ABS_TOL / 2,
                    }
                )
            }
        }
    )
    assert result.choices["direction"].choice == "BULLISH"


@pytest.mark.unit
def test_normalize_rejects_confidence_out_of_range():
    with pytest.raises(JevMalformedResponseError, match="confidence"):
        normalize_system_one_response(
            {"answers": {"direction": _choice_answer(confidence=4)}}
        )


@pytest.mark.unit
def test_normalize_rejects_boolean_confidence():
    with pytest.raises(JevMalformedResponseError, match="confidence"):
        normalize_system_one_response(
            {"answers": {"direction": _choice_answer(confidence=True)}}
        )


@pytest.mark.unit
def test_normalize_rejects_non_finite_score():
    with pytest.raises(JevMalformedResponseError, match="finite"):
        normalize_system_one_response(
            {"answers": {"signal_strength": _score_answer(float("nan"))}}
        )


@pytest.mark.unit
def test_normalize_rejects_infinite_score():
    with pytest.raises(JevMalformedResponseError, match="finite"):
        normalize_system_one_response(
            {"answers": {"market_risk": _score_answer(float("inf"))}}
        )


@pytest.mark.unit
def test_normalize_rejects_boolean_score():
    with pytest.raises(JevMalformedResponseError, match="finite"):
        normalize_system_one_response(
            {"answers": {"signal_strength": _score_answer(True)}}
        )


@pytest.mark.unit
def test_normalize_rejects_noul_out_of_range():
    with pytest.raises(JevMalformedResponseError, match="noul"):
        normalize_system_one_response({"answers": {"q": {"type": "noul", "noul": 1.5}}})


@pytest.mark.unit
def test_normalize_accepts_optional_missing_confidence():
    result = normalize_system_one_response(
        {"answers": {"direction": _choice_answer(confidence=None)}}
    )
    assert result.choices["direction"].confidence is None


@pytest.mark.unit
def test_normalize_accepts_boundary_unit_values():
    result = normalize_system_one_response(
        {
            "answers": {
                "direction": _choice_answer(
                    confidence=0.0,
                    probabilities={"BULLISH": 1.0, "NEUTRAL": 0.0, "BEARISH": 0.0},
                ),
                "p": {"type": "noul", "noul": 1.0},
                "strength": _score_answer(0.0),
            }
        }
    )
    assert result.choices["direction"].confidence == 0.0
    assert result.nouls["p"].noul == 1.0
    assert result.scores["strength"].score == 0.0


@pytest.mark.unit
def test_evaluate_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_AI_API_KEY", raising=False)
    client = JevClient()
    with pytest.raises(JevNotConfiguredError, match="TYPESAFE_API_KEY"):
        client.evaluate({"ticker": "NVDA"}, {"direction": {"type": "noul"}})


@pytest.mark.unit
def test_evaluate_raises_when_sdk_missing(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    client = JevClient()
    with pytest.raises(JevNotConfiguredError, match="typesafe-sdk"):
        client.evaluate({"ticker": "NVDA"}, {"q": {"type": "noul"}})


@pytest.mark.unit
def test_evaluate_wraps_timeout(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    _install_sdk(monkeypatch, TypeSafeAPITimeoutError("slow", timeout=10.0))
    client = JevClient()
    with pytest.raises(JevTimeoutError, match="slow"):
        client.evaluate({"ticker": "NVDA"}, {"q": {"type": "noul"}})


@pytest.mark.unit
def test_evaluate_wraps_validation_error_as_malformed(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    _install_sdk(
        monkeypatch,
        TypeSafeAPIResponseValidationError(
            "missing confidence", field_path="answers.q.confidence"
        ),
    )
    client = JevClient()
    with pytest.raises(JevMalformedResponseError, match="missing confidence"):
        client.evaluate({"ticker": "NVDA"}, {"q": {"type": "noul"}})


@pytest.mark.unit
def test_evaluate_wraps_auth_error_as_not_configured(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    _install_sdk(monkeypatch, TypeSafeAuthenticationError("401"))
    client = JevClient()
    with pytest.raises(JevNotConfiguredError, match="401"):
        client.evaluate({"ticker": "NVDA"}, {"q": {"type": "noul"}})


@pytest.mark.unit
def test_evaluate_wraps_rate_limit_as_provider_error(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    _install_sdk(monkeypatch, TypeSafeRateLimitError("429"))
    client = JevClient()
    with pytest.raises(JevProviderError, match="429"):
        client.evaluate({"ticker": "NVDA"}, {"q": {"type": "noul"}})
