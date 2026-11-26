"""Tool 1: search_reports — 안에 AI(E5 임베딩)가 들어 있는 Tool (04).

이 Tool은 찾은 문서만 돌려준다. 답변은 Agent의 LLM이 한 번만 쓴다.
(수업 자료는 Tool 안에서 LLM이 한 번 더 답변을 쓴다. README '설계 차이' 참고)
"""
import json

from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from app import repository
from app.errors import UserInputError
from app.tools.validation import check_category


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")   # 설명서에 없는 인자를 넣으면 거절
    query: str = Field(min_length=1, max_length=200, description="보고서에서 찾을 내용 (자연어)")
    category: str = Field(default="ALL", description="ALL 또는 EXPERIMENT, DAILY_WORK, RESEARCH_PLAN, DATA_PROCESSING, PROMPT, MEETING")


@tool("search_reports", args_schema=SearchInput)
def search_reports(query: str, category: str = "ALL") -> str:
    """보고서 내용을 의미로 검색한다. 보고서의 내용·설명·근거가 필요할 때 사용한다.
    결과의 source_id만 [RPT-번호]로 인용할 수 있다. sources가 비어 있으면 근거가 없다는 뜻이다."""
    try:
        categories = check_category(category)
    except UserInputError as e:
        return json.dumps({"status": "invalid_input", "message": str(e)}, ensure_ascii=False)

    sources = repository.search_reports(query, categories)
    if not sources:
        return json.dumps({"status": "no_answer", "sources": [],
                           "message": "기준 이상으로 관련된 보고서가 없습니다."}, ensure_ascii=False)
    return json.dumps({"status": "ok", "sources": sources}, ensure_ascii=False)
