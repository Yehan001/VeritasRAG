import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def append_audit_log(result_dict: Dict[str, Any], path: Optional[str] = "guardrail_audit_log.jsonl") -> None:
    """Append one structured decision event to a JSONL audit log.

    The log is intentionally local and lightweight. It records only a preview of the
    original input to reduce privacy exposure. Use this during demonstrations and tests.
    """
    if not path:
        return
    p = Path(path)
    safe = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tenant_id": result_dict.get("tenant_id"),
        "decision": result_dict.get("decision"),
        "passed": result_dict.get("passed"),
        "risk_level": result_dict.get("risk_level"),
        "reason": result_dict.get("reason"),
        "input_preview": (result_dict.get("original_input") or "")[:160],
        "pii_masked_preview": (result_dict.get("pii_masked_input") or "")[:160],
        "kb_domain": (result_dict.get("kb_profile") or {}).get("domain"),
        "kb_relevance_score": (result_dict.get("relevance") or {}).get("score"),
        "safety_label": (result_dict.get("safety") or {}).get("label"),
        "safety_backend": (result_dict.get("safety") or {}).get("backend"),
        "user_role": result_dict.get("user_role"),
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")