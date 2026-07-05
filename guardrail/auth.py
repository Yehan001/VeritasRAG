"""
auth.py
Minimal per-tenant API key store.

This is deliberately simple (in-memory dict, plaintext keys) — appropriate
for an FYP demo/prototype. For anything beyond that: hash the stored keys,
back this with a database, and rotate keys instead of hardcoding them.
"""
import secrets
import threading
from typing import Dict, Optional


class AuthStore:
    def __init__(self) -> None:
        self._keys: Dict[str, str] = {}  # tenant_id -> api_key
        self._lock = threading.Lock()

    def register_tenant(self, tenant_id: str, api_key: Optional[str] = None) -> str:
        """Register a tenant and return its API key (generates one if not given)."""
        key = api_key or secrets.token_urlsafe(24)
        with self._lock:
            self._keys[tenant_id] = key
        return key

    def validate(self, tenant_id: str, api_key: str) -> bool:
        with self._lock:
            expected = self._keys.get(tenant_id)
        return expected is not None and secrets.compare_digest(expected, api_key or "")

    def is_registered(self, tenant_id: str) -> bool:
        with self._lock:
            return tenant_id in self._keys


# One shared auth store for the whole API process.
_SHARED_AUTH_STORE = AuthStore()