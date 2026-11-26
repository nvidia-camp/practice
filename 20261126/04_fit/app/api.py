"""⑤ 입구(HTTP): run.py와 같은 agent.ask()를 웹 요청으로 받는다 (08).

실행 (04_fit 폴더에서):
    uvicorn app.api:app --port 8008
확인:
    http://127.0.0.1:8008/docs 에서 POST /api/agent/ask 실행
로컬 교육용입니다. 사용자 인증이 없으므로 외부에 공개하지 않습니다.
"""
import re

from fastapi import FastAPI, HTTPException
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from pydantic import BaseModel, Field

from app.agent import ask

app = FastAPI(title="Report Agent (04_fit)")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)   # 잘못된 입력은 FastAPI가 422로 막는다


@app.post("/api/agent/ask")
def ask_endpoint(body: AskRequest):
    try:
        run = ask(body.question.strip())
    except (ToolCallLimitExceededError, ModelCallLimitExceededError):
        raise HTTPException(503, "호출 상한을 넘어 중단했습니다. 질문을 나눠 주세요.")
    except Exception:
        raise HTTPException(503, "일시적인 오류입니다.")   # 내부 상세는 응답에 넣지 않는다

    # 출처 검증: 답변이 [RPT-번호]로 인용한 것은 모두 search_reports가 실제로 준 근거여야 한다.
    # (lookup_status로 조회한 번호는 근거가 아니라 '상태 언급'이므로 인용으로 인정하지 않는다)
    cited_ids = set(re.findall(r"RPT-\d+", run["answer"]))
    unverified = cited_ids - run["evidence_ids"]
    if unverified:
        raise HTTPException(502, f"검증되지 않은 출처 인용: {sorted(unverified)}")

    return {
        "status": "completed",
        "answer": run["answer"],
        "tool_trace": run["tool_names"],
        "model_calls": run["model_calls"],
        "cited_ids": sorted(cited_ids),
        "verified_evidence_ids": sorted(run["evidence_ids"]),
        "status_checked_ids": sorted(run["status_ids"]),
    }
