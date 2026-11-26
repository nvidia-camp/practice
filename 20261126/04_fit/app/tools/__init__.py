"""③ Tool 등록부. 새 Tool을 추가하려면 파일을 하나 만들고 여기 TOOLS에 한 줄 추가한다.

@tool을 붙이면 LangChain이 이름·인자 Schema·docstring으로 JSON 설명서를 만들어 LLM에게 보낸다.
LLM은 코드를 볼 수 없고 이 설명서만 보고 언제 어떤 인자로 부를지 정한다.
"""
from app.tools.search import search_reports
from app.tools.status import lookup_status

TOOLS = [search_reports, lookup_status]
