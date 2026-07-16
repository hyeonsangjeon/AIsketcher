"""Exceptions raised by AIsketcher.

The package keeps its exception hierarchy small so applications can either catch
``AIsketcherError`` or react to a specific, actionable failure.
"""

from __future__ import annotations


class AIsketcherError(Exception):
    """Base class for all package errors."""


class ValidationError(AIsketcherError, ValueError):
    """Raised when a public API value is invalid."""


class UnsupportedCapabilityError(AIsketcherError):
    """Raised when a backend cannot honor a resolved recipe."""


class OptionalDependencyError(AIsketcherError, ImportError):
    """Raised when an explicitly requested optional feature is not installed."""


class ModelUnavailableError(AIsketcherError):
    """Raised when a pinned model is unavailable in the local cache."""


class GenerationError(AIsketcherError):
    """Raised when a backend produces an invalid or numerically unstable result."""


class UnsafeModelError(AIsketcherError):
    """Raised when an unpinned or unsafe model reference is requested."""


class ReplayError(AIsketcherError):
    """Raised when an exported study cannot be replayed."""


class IntegrityError(ReplayError):
    """Raised when an artifact does not match the hash in its manifest."""


class RemovedFeatureError(AIsketcherError):
    """Raised for legacy features intentionally removed from the package."""
