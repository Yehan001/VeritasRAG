"""
api.py
Simplified FastAPI entry point for the safety-only guardrail.

Endpoints:
    POST /check -> run the guardrail pipeline on a question
    GET  /health -> server status
"""
import time

from fastapi import FastAPI
from pydantic import BaseModel

from guardrail.audit_logger import append_audit_log
from guardrail.hf_safety import _SHARED_SAFETY_ENGINE
from guardrail.input_filter import GuardrailSettings, SafetyInputFilter

app = FastAPI(title="Safety-Only Guardrail API")


@app.on_event("startup")
def warmup_shared_engine() -> None:
    start = time.perf_counter()
    _SHARED_SAFETY_ENGINE.warmup()
    latency = time.perf_counter() - start
    print(f"[api.startup] shared_safety_engine_warmup latency={latency:.4f} seconds")


class CheckRequest(BaseModel):
    question: str
    user_role: str = "public_user"
    strict_mode: bool = False
    enable_hf_models: bool = False


@app.post("/check")
def check(req: CheckRequest):
    settings = GuardrailSettings(
        strict_mode=req.strict_mode,
        user_role=req.user_role,
        enable_hf_models=req.enable_hf_models,
    )

    guardrail = SafetyInputFilter("", settings=settings, safety_engine=_SHARED_SAFETY_ENGINE)
    result = guardrail.check(req.question, user_role=req.user_role)

    append_audit_log(result.to_dict(), path="guardrail_audit_log.jsonl")
    return result.to_api_response().to_dict()


@app.get("/health")
def health():
    return {"status": "ok"}
