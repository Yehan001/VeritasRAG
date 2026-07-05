"""
kb_registry.py
In-memory, thread-safe store holding one fitted KBRelevanceChecker + KBProfile
per tenant (institution). Each tenant's KB is hashed so re-uploading the same
document is a no-op instead of re-fitting the relevance index again.

This is intentionally a plain in-memory dict — fine for a single-process API
and an FYP demo. Swap for Redis/DB-backed storage if this needs to survive
process restarts or run across multiple workers.
"""
import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from .kb_profiler import profile_kb
from .kb_relevance import KBRelevanceChecker
from .schema import KBProfile


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class TenantKB:
    tenant_id: str
    kb_hash: str
    kb_text: str
    kb_profile: KBProfile
    relevance_checker: KBRelevanceChecker
    relevance_threshold: float
    grounding_threshold: float


class KBRegistry:
    def __init__(self) -> None:
        self._store: Dict[str, TenantKB] = {}
        # One lock guarding the whole registry. Simple and safe for the
        # traffic level this API is expected to see; if /upload-kb ever
        # becomes a hot path, switch to a per-tenant lock instead.
        self._lock = threading.Lock()

    def upload_kb(
        self,
        tenant_id: str,
        kb_text: str,
        relevance_threshold: float = 0.25,
        grounding_threshold: float = 0.35,
    ) -> TenantKB:
        new_hash = _hash_text(kb_text)

        with self._lock:
            existing = self._store.get(tenant_id)
            if existing is not None and existing.kb_hash == new_hash:
                # Same document re-uploaded — nothing to refit.
                return existing

            start = time.perf_counter()
            profile = profile_kb(kb_text)
            checker = KBRelevanceChecker(
                relevance_threshold=relevance_threshold,
                grounding_threshold=grounding_threshold,
            )
            checker.fit(kb_text)
            latency = time.perf_counter() - start
            print(f"[KBRegistry.upload_kb] tenant={tenant_id} fit_latency={latency:.4f} seconds")

            tenant_kb = TenantKB(
                tenant_id=tenant_id,
                kb_hash=new_hash,
                kb_text=kb_text,
                kb_profile=profile,
                relevance_checker=checker,
                relevance_threshold=relevance_threshold,
                grounding_threshold=grounding_threshold,
            )
            self._store[tenant_id] = tenant_kb
            return tenant_kb

    def get(self, tenant_id: str) -> Optional[TenantKB]:
        with self._lock:
            return self._store.get(tenant_id)

    def tenant_count(self) -> int:
        with self._lock:
            return len(self._store)


# One shared registry for the whole API process, imported directly by api.py.
_SHARED_KB_REGISTRY = KBRegistry()