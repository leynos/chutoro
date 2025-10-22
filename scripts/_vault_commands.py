"""Utilities for verifying and provisioning Vault AppRole credentials."""

from __future__ import annotations

import json
from typing import Any, Mapping


class VaultBootstrapError(RuntimeError):
    """Raised when Vault initialisation commands cannot be completed."""


def _ensure_approle_auth_enabled(payload: str) -> Mapping[str, Any]:
    """Ensure the AppRole auth method is enabled on the Vault server."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        msg = "Failed to decode Vault response whilst verifying AppRole auth"
        raise VaultBootstrapError(msg) from exc

    auth_methods = data.get("data", {})
    if "approle" not in auth_methods:
        msg = "AppRole authentication is not enabled for the target Vault"
        raise VaultBootstrapError(msg)

    return auth_methods


def _generate_secret_id(payload: str) -> str:
    """Extract a secret-id from a Vault response payload."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        msg = "Failed to decode Vault response whilst generating a secret-id"
        raise VaultBootstrapError(msg) from exc

    secret_id = data.get("data", {}).get("secret_id")
    if not secret_id:
        msg = "Vault response missing secret_id field"
        raise VaultBootstrapError(msg)

    return secret_id
