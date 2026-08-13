from fastapi import FastAPI
from pydantic import BaseModel

from guardrail import GuardrailSettings, InputGuardrail

app = FastAPI(title="Input Filtering Guardrail API")


class CheckRequest(BaseModel):
    text: str
    enable_hf_models: bool = True
    enable_sentence_transformer: bool = True
    use_presidio: bool = True


@app.post("/check")
def check(req: CheckRequest):
    guardrail = InputGuardrail(GuardrailSettings(
        enable_hf_models=req.enable_hf_models,
        enable_sentence_transformer=req.enable_sentence_transformer,
        use_presidio=req.use_presidio,
        enable_audit_log=True,
    ))
    return guardrail.check(req.text).to_dict()
