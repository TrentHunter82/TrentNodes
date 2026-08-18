"""Hermes Agent integration helpers for the H3 Prompt Director."""

from .assets import (
    AssetIntegrityError,
    verified_manifest_snapshot,
    verify_staged_assets,
)
from .client import (
    MISSING_API_KEY_GUIDANCE,
    HermesAuthenticationError,
    HermesClientError,
    HermesRunCancelledError,
    HermesRunFailedError,
    HermesRunTimeoutError,
    HermesRunsClient,
    HermesUnsupportedRuntimeError,
    RunResult,
)
from .contract import freeze_request_authority

__all__ = [
    "AssetIntegrityError",
    "MISSING_API_KEY_GUIDANCE",
    "HermesAuthenticationError",
    "HermesClientError",
    "HermesRunCancelledError",
    "HermesRunFailedError",
    "HermesRunTimeoutError",
    "HermesRunsClient",
    "HermesUnsupportedRuntimeError",
    "RunResult",
    "freeze_request_authority",
    "verified_manifest_snapshot",
    "verify_staged_assets",
]
