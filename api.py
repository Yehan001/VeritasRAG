"""
api.py
FastAPI entry point for the multi-tenant KB-aware guardrail.

Endpoints:
    POST /register-tenant  -> create a tenant, returns its API key
    POST /upload-kb        -> upload/replace a tenant's knowledge base
    POST /check            -> run the full guardrail pipeline on a question
    GET  /health            -> server status + loaded tenant count

Run:
    uvicorn api:app --reload
"""
import time
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from guardrail.audit_logger import append_audit_log
from guardrail.auth import _SHARED_AUTH_STORE
from guardrail.hf_safety import _SHARED_SAFETY_ENGINE
from guardrail.input_filter import GuardrailSettings, KBAwareInputFilter
from guardrail.kb_registry import _SHARED_KB_REGISTRY

app = FastAPI(title="KB-Aware Guardrail API")


# ---------------------------------------------------------------------------
# Startup: warm up the 3 HF models ONCE for the whole process.
# ---------------------------------------------------------------------------
@app.on_event("startup")
def warmup_shared_engine() -> None:
    start = time.perf_counter()
    _SHARED_SAFETY_ENGINE.warmup()
    latency = time.perf_counter() - start
    print(f"[api.startup] shared_safety_engine_warmup latency={latency:.4f} seconds")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
def require_api_key(tenant_id: str, x_api_key: str = Header(...)) -> str:
    if not _SHARED_AUTH_STORE.validate(tenant_id, x_api_key):
        raise HTTPException(status_code=403, detail="Invalid tenant_id or API key")
    return tenant_id


# ---------------------------------------------------------------------------
# /register-tenant
# ---------------------------------------------------------------------------
class RegisterTenantRequest(BaseModel):
    tenant_id: str


@app.post("/register-tenant")
def register_tenant(req: RegisterTenantRequest):
    if _SHARED_AUTH_STORE.is_registered(req.tenant_id):
        raise HTTPException(status_code=409, detail="tenant_id already registered")
    api_key = _SHARED_AUTH_STORE.register_tenant(req.tenant_id)
    return {"tenant_id": req.tenant_id, "api_key": api_key}


# ---------------------------------------------------------------------------
# /upload-kb
# ---------------------------------------------------------------------------
@app.post("/upload-kb")
def upload_kb(
    tenant_id: str,
    file: UploadFile = File(...),
    x_api_key: str = Header(...),
):
    if not _SHARED_AUTH_STORE.validate(tenant_id, x_api_key):
        raise HTTPException(status_code=403, detail="Invalid tenant_id or API key")

    try:
        raw_bytes = file.file.read()
        from pathlib import Path
        import io

        suffix = Path(file.filename or "").suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw_bytes))
            kb_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix == ".docx":
            from docx import Document

            doc = Document(io.BytesIO(raw_bytes))
            kb_text = "\n".join(p.text for p in doc.paragraphs)
        else:
            kb_text = raw_bytes.decode("utf-8", errors="ignore")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {exc}")

    tenant_kb = _SHARED_KB_REGISTRY.upload_kb(tenant_id=tenant_id, kb_text=kb_text)

    return {
        "tenant_id": tenant_kb.tenant_id,
        "kb_hash": tenant_kb.kb_hash,
        "domain": tenant_kb.kb_profile.domain,
        "confidence": tenant_kb.kb_profile.confidence,
        "risk_level": tenant_kb.kb_profile.risk_level,
    }


# ---------------------------------------------------------------------------
# /check
# ---------------------------------------------------------------------------
class CheckRequest(BaseModel):
    tenant_id: str
    question: str
    user_role: str = "public_user"
    strict_mode: bool = False
    kb_authoritative_mode: bool = True


@app.post("/check")
def check(req: CheckRequest, x_api_key: str = Header(...)):
    if not _SHARED_AUTH_STORE.validate(req.tenant_id, x_api_key):
        raise HTTPException(status_code=403, detail="Invalid tenant_id or API key")

    tenant_kb = _SHARED_KB_REGISTRY.get(req.tenant_id)
    if tenant_kb is None:
        raise HTTPException(status_code=404, detail="No KB uploaded for this tenant_id yet")

    settings = GuardrailSettings(
        strict_mode=req.strict_mode,
        kb_authoritative_mode=req.kb_authoritative_mode,
        relevance_threshold=tenant_kb.relevance_threshold,
        grounding_threshold=tenant_kb.grounding_threshold,
        user_role=req.user_role,
    )

    # Reuses the already-fitted KB profile/relevance index (from kb_registry)
    # and the already-warmed shared safety engine — nothing is recomputed
    # per request.
    guardrail = KBAwareInputFilter(
        kb_text=tenant_kb.kb_text,
        settings=settings,
        safety_engine=_SHARED_SAFETY_ENGINE,
        kb_profile=tenant_kb.kb_profile,
        relevance_checker=tenant_kb.relevance_checker,
    )

    result = guardrail.check(req.question, user_role=req.user_role, tenant_id=req.tenant_id)

    append_audit_log(result.to_dict(), path="guardrail_audit_log.jsonl")

    return result.to_api_response().to_dict()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "tenant_count": _SHARED_KB_REGISTRY.tenant_count(),
    }