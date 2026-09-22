"""Errors raised by decision backends (Jev / System One).

These are independent of the LLM client layer and of data-vendor errors.
JevAgent does not fall back to an LLM or invent a BUY/HOLD/SELL on failure.
"""


class JevError(Exception):
    """Base for Jev backend and agent failures."""


class JevNotConfiguredError(JevError, ValueError):
    """API key missing/invalid, or the optional TypeSafe SDK is not installed."""


class JevTimeoutError(JevError, TimeoutError):
    """The System One request exceeded its configured timeout."""


class JevProviderError(JevError):
    """The TypeSafe API rejected the request or the network call failed."""


class JevMalformedResponseError(JevError):
    """A response was missing required fields (choice, probabilities, score, ...)."""


class JevLookAheadError(JevError, ValueError):
    """Input market state includes information dated after the analysis cutoff."""
