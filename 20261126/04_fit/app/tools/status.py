"""Tool 2: lookup_status — 평범한 Tool. 읽기 전용 상태 조회 (05)."""
import json

from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from app import repository
from app.config import ALLOWED_CATEGORIES
from app.errors import UserInputError
from app.tools.validation import check_report_id


class StatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # 그냥 int로 두면 Pydantic이 True → 1, "063" → 63으로 조용히 바꿔 버린다.
    # Strict 타입으로 받아 bool은 여기서 거절하고, 문자열은 check_report_id에서 엄격히 검사한다.
    report_id: StrictInt | StrictStr = Field(description="보고서 번호 (양의 정수)")


@tool("lookup_status", args_schema=StatusInput)
def lookup_status(report_id: int | str) -> str:
    """보고서 번호로 인덱싱 상태를 조회한다.
    index_status(ready/pending/failed)와 vector_fresh(검색에 쓸 수 있는 최신 벡터인지)를 함께 돌려준다."""
    try:
        report_id = check_report_id(report_id)
    except UserInputError as e:
        return json.dumps({"status": "invalid_input", "message": str(e)}, ensure_ascii=False)

    report = repository.find_report(report_id)
    # 없는 보고서와 범위 밖 보고서를 같은 응답으로: 범위 밖 보고서의 '존재 여부'도 알려 주지 않는다
    if report is None or report["category"] not in ALLOWED_CATEGORIES:
        return json.dumps({"status": "not_found_or_forbidden", "report_id": report_id}, ensure_ascii=False)

    return json.dumps({
        "status": "ok",
        "report_id": report_id,
        "title": report["title"],
        "category": report["category"],
        "index_status": report["index_status"],
        "revision": report["revision"],
        "indexed_revision": report["indexed_revision"],
        "vector_fresh": repository.is_vector_fresh(report),
    }, ensure_ascii=False)
