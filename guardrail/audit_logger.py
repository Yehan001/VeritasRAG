import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import GuardrailResult


class AuditLogger:
    def __init__(self, path: str = "guardrail_audit_log.jsonl", enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled

    def write(self, result: GuardrailResult) -> Optional[str]:
        if not self.enabled:
            return None
        audit_id = str(uuid.uuid4())
        data = result.to_dict()
        data["audit_id"] = audit_id
        data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True) if self.path.parent != Path('.') else None
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return audit_id
