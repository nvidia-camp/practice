"""부록: LangChain 없이 OpenAI API만으로 같은 흐름을 직접 짠 버전.

app/agent.py의 create_agent가 내부에서 해 주는 일이 바로 아래 while 반복문이다.
실행 (04_fit 폴더에서):  python appendix_raw_loop.py
"""
import json

from openai import OpenAI

from app import repository
from app.config import ALLOWED_CATEGORIES, LLM_MODEL, MAX_TOOL_CALLS

client = OpenAI()

# 1) Tool 설명서(JSON Schema)를 손으로 작성 — @tool이 자동으로 만들어 주던 것
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_reports",
        "description": "보고서 내용을 검색한다. 보고서의 내용·설명·근거가 필요할 때 사용한다.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "검색어"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "lookup_status",
        "description": "보고서 번호로 인덱싱 상태(ready / pending / failed)를 조회한다.",
        "parameters": {"type": "object",
                       "properties": {"report_id": {"type": "integer", "description": "보고서 번호"}},
                       "required": ["report_id"]}}},
]


# 2) Tool 이름 → 실제 실행할 Python 함수
#    (검증·범위 제한은 생략한 최소 버전. 본체는 app/tools/ 참고)
def run_tool(name: str, args: dict) -> str:
    if name == "search_reports":
        sources = repository.search_reports(args["query"], ALLOWED_CATEGORIES)
        return json.dumps({"sources": sources}, ensure_ascii=False)
    if name == "lookup_status":
        report = repository.find_report(int(args["report_id"]))
        if report is None:
            return json.dumps({"status": "not_found"})
        return json.dumps({"report_id": report["report_id"], "index_status": report["index_status"]})
    return json.dumps({"error": f"모르는 Tool: {name}"})


question = "RAG 근거 인용 정확도 향상과 관련된 보고서의 내용을 설명하고, 보고서 63번의 인덱싱 상태도 알려줘."
messages = [
    {"role": "system", "content": "Tool 결과만 근거로 답하고 [RPT-번호]로 인용하세요."},
    {"role": "user", "content": question},
]
tool_call_count = 0

# 3) 반복: LLM 호출 → tool_calls 있으면 실행하고 결과를 붙여 다시 호출 → 없으면 끝
while True:
    response = client.chat.completions.create(model=LLM_MODEL, messages=messages, tools=TOOL_SCHEMAS)
    ai_message = response.choices[0].message

    if not ai_message.tool_calls:   # Tool 요청이 없으면 최종 답변
        print(f"\n[AI 최종 답변]\n{ai_message.content}")
        break

    messages.append(ai_message)     # LLM의 요청서도 대화 기록에 남겨야 한다
    for call in ai_message.tool_calls:
        args = json.loads(call.function.arguments)
        print(f"\n[AI → Tool 요청] id={call.id}  {call.function.name}({args})")

        result = run_tool(call.function.name, args)
        print(f"[Tool 실행 결과] {result[:200]}")

        # 결과를 같은 tool_call_id로 묶어 돌려준다 — 이것이 ToolMessage
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    tool_call_count += len(ai_message.tool_calls)
    if tool_call_count >= MAX_TOOL_CALLS:   # 무한 반복 방지
        print("Tool 호출 상한에 도달해 중단합니다.")
        break
