"""④ Agent: 모델 + Tool + 시스템 프롬프트 + 호출 제한을 묶고, 실행 결과를 정리한다.

create_agent가 대신 해 주는 일 (직접 짜면 appendix_raw_loop.py):
  1. 질문과 Tool 설명서를 LLM에게 보낸다
  2. LLM이 tool_calls를 돌려주면 해당 Tool 함수를 실행한다
  3. 결과를 ToolMessage(같은 tool_call_id)로 다시 LLM에게 보낸다
  4. LLM이 더 이상 Tool을 부르지 않으면 그 답을 최종 답변으로 끝낸다
"""
import json
import time

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI

from app import audit
from app.config import LLM_MODEL, MAX_MODEL_CALLS, MAX_TOOL_CALLS
from app.tools import TOOLS

SYSTEM_PROMPT = """당신은 사내 보고서 도우미입니다.
- 보고서 내용은 search_reports로, 보고서 상태는 lookup_status로 확인하세요.
- 내용 설명은 search_reports 결과의 sources에 있는 것만 사용하고, 근거는 그 source_id로 [RPT-번호] 형식으로 인용하세요.
- lookup_status로 확인한 상태는 "보고서 63번은 pending"처럼 번호만 언급하고 [RPT-번호]로 인용하지 마세요.
- sources가 비어 있으면 근거가 없다고 답하고 지어내지 마세요.
- invalid_input이면 이유를 사용자에게 알리고, not_found_or_forbidden이면 찾을 수 없다고만 답하세요."""

agent = create_agent(
    model=ChatOpenAI(model=LLM_MODEL, timeout=45),
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS, exit_behavior="error"),
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="error"),
    ],
)


def summarize(messages: list) -> dict:
    """메시지 목록에서 06번 관찰 포인트를 뽑는다: 호출 수, 근거 ID, 최종 답변."""
    model_calls = 0
    tool_names = []
    evidence_ids = set()   # search_reports가 실제로 돌려준 근거
    status_ids = set()     # lookup_status로 조회한 보고서

    for message in messages:
        if message.type == "ai":
            model_calls += 1
        elif message.type == "tool":
            tool_names.append(message.name)
            result = json.loads(message.content)
            for source in result.get("sources", []):
                evidence_ids.add(source["source_id"])
            if message.name == "lookup_status" and result.get("status") == "ok":
                status_ids.add(f"RPT-{result['report_id']}")

    final = messages[-1]   # 마지막 메시지가 tool_calls 없는 AIMessage여야 정상 종료
    if final.type != "ai" or final.tool_calls:
        raise RuntimeError("Agent가 최종 답변을 만들지 못했습니다.")

    return {
        "answer": final.content,
        "model_calls": model_calls,
        "tool_names": tool_names,
        "evidence_ids": evidence_ids,
        "status_ids": status_ids,
        "messages": messages,
    }


def ask(question: str) -> dict:
    """질문 하나를 Agent에 넣고 정리된 실행 결과를 돌려준다. 성공·실패 모두 감사 로그를 남긴다."""
    started = time.time()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        summary = summarize(result["messages"])
    except Exception as e:
        audit.write(ok=False, question=question, model_calls=0, tool_names=[],
                    elapsed=time.time() - started, error_type=type(e).__name__)
        raise
    audit.write(ok=True, question=question, model_calls=summary["model_calls"],
                tool_names=summary["tool_names"], elapsed=time.time() - started)
    return summary
