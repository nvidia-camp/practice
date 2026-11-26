"""입구(터미널): 질문 하나를 Agent에 넣고 메시지 흐름 전체와 요약을 출력한다.

실행 (04_fit 폴더에서):
    python run.py
    python run.py "출처 정밀도를 높인 실험이 있었어?"
"""
import sys

from app.agent import ask

DEFAULT_QUESTION = "RAG 근거 인용 정확도 향상과 관련된 보고서의 내용을 설명하고, 보고서 63번의 인덱싱 상태도 알려줘."

question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
run = ask(question)

for message in run["messages"]:
    kind = message.type   # human / ai / tool

    if kind == "human":
        print(f"\n[Human] {message.content}")

    elif kind == "ai" and message.tool_calls:
        print("\n[AI → Tool 요청]  (LLM은 요청서만 쓴다. 실행은 아직 안 됨)")
        for call in message.tool_calls:
            print(f"   id={call['id']}  {call['name']}({call['args']})")

    elif kind == "tool":
        print(f"\n[Tool 실행 결과]  (우리 코드가 실행)  tool_call_id={message.tool_call_id}")
        print(f"   {message.name} → {message.content[:300]}")

    elif kind == "ai":
        print(f"\n[AI 최종 답변]\n{message.content}")

print(f"\n[요약] 모델 호출 {run['model_calls']}회 · Tool 실행 {len(run['tool_names'])}회"
      f" · 근거 {sorted(run['evidence_ids']) or '없음'} · 상태 조회 {sorted(run['status_ids']) or '없음'}")
