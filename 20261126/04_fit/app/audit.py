"""안전한 실행 로그 (07). 무엇이 일어났는지만 남기고, 민감한 내용은 남기지 않는다.

남기는 것: 시각, 성공 여부, 호출 횟수, 사용한 Tool 이름, 소요 시간, 질문 길이
남기지 않는 것: API 키, 질문 원문, 답변 원문, 오류의 내부 상세
"""
import json
from datetime import datetime

from app.config import AUDIT_LOG


def write(*, ok: bool, question: str, model_calls: int, tool_names: list[str], elapsed: float, error_type: str | None = None):
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "ok": ok,
        "question_length": len(question),
        "model_calls": model_calls,
        "tools": tool_names,
        "elapsed_sec": round(elapsed, 2),
        "error_type": error_type,   # 예외 클래스 이름만
    }
    AUDIT_LOG.parent.mkdir(exist_ok=True)
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
