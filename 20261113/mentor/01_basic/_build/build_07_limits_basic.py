"""20261113 실습 07 · LLM 한계·검증 (기본필수) → 출력 저장된 Colab 노트북.

사용: python build_07_limits_basic.py <support.py 경로> <출력 ipynb> <실행용 임시 폴더>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbkit import Notebook, execute, setup_source, download_source

SUPPORT, OUT, WORKDIR = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
nb = Notebook()
md, code = nb.md, nb.code

CLASSIFY = '''def classify(case):
    """교육용 규칙: 형식 누락 → 근거 누락 → 원본과 불일치 순서로 검사"""
    if "sources" not in case:
        return "format"
    elif not case["sources"]:
        return "evidence"
    elif case["answer"] != case["source"]:
        return "factual_mismatch"
    return "ok"'''

md("""
# 실습 07 · LLM 한계·검증 (기본필수)

LLM 답변이 그럴듯해 보여도 문제가 있을 수 있습니다. 답변 3개를 **형식 · 근거 · 사실** 세 가지 검사로 나눠 분류합니다.

| 검사 | 확인하는 것 |
|---|---|
| format (형식) | 응답에 필수 필드(`sources`)가 **있는가** |
| evidence (근거) | 출처가 **비어 있지 않은가** |
| factual_mismatch (사실) | 답변이 원본 내용과 **같은가** |

### 사용 방법
1. **결과는 이미 셀 아래에 저장되어 있습니다.** 읽기만 해도 흐름을 따라갈 수 있어요.
2. 직접 돌려 보려면 `파일 → Drive에 사본 저장`으로 내 사본을 만드세요.
3. 맨 위 **⚙️ 준비 셀을 먼저 한 번 실행**한 뒤, 보고 싶은 구간의 셀을 실행하세요. 각 구간은 준비 셀만 있으면 따로 실행됩니다.
4. **✏️ 표시 셀**은 오른쪽 입력 칸의 값을 바꾸고 실행해 보세요. 코드의 값도 함께 바뀝니다.

> ⚠️ 수업용으로 미리 설계한 사례와 단순 규칙입니다. 실제 LLM 사실성 자동 판정기가 아닙니다.
""")

code(setup_source("20261113/mentor/01_basic/common/support.py",
                  SUPPORT.read_text(encoding="utf-8"), '''
from IPython.display import Markdown, display

# 실습 사례 (main.py와 같은 값): 사람이 미리 설계한 답변 3개
CASES = [
    {"id": "F01", "question": "파일을 어디에 보관하나요?",
     "source": "실습 파일은 개인 폴더에 보관합니다.",
     "answer": "실습 파일은 공용 서버 폴더에 보관합니다.", "sources": ["K01"]},
    {"id": "F02", "question": "서버는 어디에서 실행하나요?",
     "source": "프로젝트 폴더에서 서버를 실행합니다.",
     "answer": "프로젝트 폴더에서 서버를 실행합니다."},
    {"id": "F03", "question": "질문할 때 무엇을 함께 적나요?",
     "source": "오류 메시지와 시도한 작업을 함께 적습니다.",
     "answer": "오류 메시지와 시도한 작업을 함께 적습니다.", "sources": []},
]

print("준비 완료 ✅  사례", len(CASES), "개")
'''), form=True)

# ======================================================================
md("""
## 0. 사례 보기
`sources` 칸을 잘 보세요. F02는 **필드 자체가 없고**(표에서 NaN), F03은 **빈 목록**입니다.
""")
code('pd.DataFrame(CASES)[["id", "question", "source", "answer", "sources"]]')

# ======================================================================
md("""
## B01. 분류하기

### B01-1. 사례 하나씩 세 가지 검사
""")
code('''
for case in CASES:
    has_field = "sources" in case
    has_evidence = has_field and bool(case["sources"])
    same_as_source = case["answer"] == case["source"]
    print(f"{case['id']}: sources 필드 {has_field!s:5} | 출처 있음 {has_evidence!s:5} | 원본과 같음 {same_as_source}")
''')

md("""
### B01-2. 규칙 함수 (`main.py`와 같은 순서)
검사 **순서**가 중요합니다. 필드가 없으면 출처를 볼 수 없고, 출처가 없으면 사실 비교 이전에 근거 문제입니다.
""")
code(CLASSIFY + '''

issues = [{"id": case["id"], "kind": classify(case)} for case in CASES]
pd.DataFrame(issues)
''')
code('''
expected = [("F01", "factual_mismatch"), ("F02", "format"), ("F03", "evidence")]
assert [(r["id"], r["kind"]) for r in issues] == expected
print("B01 통과 ✅", expected)
''')

md("""
### B01-3. 주의: 형식 통과 ≠ 사실 통과
F01은 형식과 근거는 모두 갖췄지만 **내용이 틀렸습니다.** 반대로 F02, F03은 내용은 맞지만 형식이나 근거가 없습니다. 세 검사는 **서로 독립**입니다.
""")
code('''
for case in CASES:
    print(case["id"], "| 형식 OK" if "sources" in case else "| 형식 ✗",
          "| 내용 OK" if case["answer"] == case["source"] else "| 내용 ✗")
''')

md("### ✏️ B01 직접 해 보기 — 내 답변을 검사해 보세요")
code('''
#@title ✏️ 답변과 출처 상태를 바꿔 보세요
answer = "실습 파일은 개인 폴더에 보관합니다."  #@param {type:"string"}
sources_state = "출처 있음 (K01)"  #@param ["출처 있음 (K01)", "빈 목록", "필드 없음"]

''' + CLASSIFY + '''

my_case = {"answer": answer, "source": "실습 파일은 개인 폴더에 보관합니다."}
if sources_state == "출처 있음 (K01)":
    my_case["sources"] = ["K01"]
elif sources_state == "빈 목록":
    my_case["sources"] = []
result = classify(my_case)
print("원본:", my_case["source"])
print("답변:", answer)
print("판정:", "✅ ok" if result == "ok" else f"❌ {result}")
''')

# ======================================================================
md("## B02. 세 검사의 차이")
code('''
explanations = {
    "format": "JSON/응답에 필수 필드가 존재하는지 보는 구조 검증",
    "evidence": "인용·출처 정보가 있는지와 적절한 근거인지 구분",
    "factual_mismatch": "주장과 승인된 정답/원본이 모순되는지 사람·검증 규칙으로 확인",
}
pd.DataFrame(explanations.items(), columns=["검사", "설명"])
''')
md("""
> 문자열이 완전히 같은지만 보는 규칙은 실제 주장 검증을 대신하지 못합니다. 표현만 다르고 뜻이 같은 답변도 `factual_mismatch`가 됩니다. 위 ✏️ 셀에서 "실습 파일은 개인 폴더에 보관해요."로 바꿔 확인해 보세요.
""")

# ======================================================================
md("""
## 저장. 결과 저장 (`main.py`와 같은 결과)
`outputs/review.json`과 `outputs/review.md`를 저장합니다. 이 셀은 준비 셀만 있으면 혼자 실행됩니다.
""")
code('''
issues = []
for case in CASES:
    if "sources" not in case:
        kind = "format"
    elif not case["sources"]:
        kind = "evidence"
    elif case["answer"] != case["source"]:
        kind = "factual_mismatch"
    else:
        kind = "ok"
    issues.append({"id": case["id"], "kind": kind})
explanations = {
    "format": "JSON/응답에 필수 필드가 존재하는지 보는 구조 검증",
    "evidence": "인용·출처 정보가 있는지와 적절한 근거인지 구분",
    "factual_mismatch": "주장과 승인된 정답/원본이 모순되는지 사람·검증 규칙으로 확인",
}
report = {"classification": issues, "definitions": explanations, "mitigation": {},
          "checks": [], "agent_case": {}, "trust_rules": [],
          "disclaimer": "예상 문제 유형을 분류한 수업용 규칙이며 실제 LLM 사실성 자동 판정이 아님"}
write_json(OUT_DIR / "review.json", report)
lines = ["# LLM 답변 실패·검증 결과", "", "이 결과는 사전 정의된 수업용 예시를 사용했습니다.", ""]
lines += [f"- {row['id']}: {row['kind']}" for row in issues]
lines += ["", "## 플랫폼 연결", "FastAPI: 입력·출력 계약 / PostgreSQL: 원본 / RAG: 검색 근거 / Agent: 도구 권한(후속)"]
write_md(OUT_DIR / "review.md", "\\n".join(lines) + "\\n")
print("[07]", [(r["id"], r["kind"]) for r in issues])
''')
code('display(Markdown((OUT_DIR / "review.md").read_text(encoding="utf-8")))')
code(download_source("review.json", "review.md"))

md("""
---
### 정리
- 답변 문제는 **형식(필드 누락) · 근거(출처 없음) · 사실(원본과 다름)** 으로 나눠 본다.
- 세 검사는 독립이다. 형식이 맞아도 내용이 틀릴 수 있고, 내용이 맞아도 근거가 없을 수 있다.
- 단순 문자열 비교는 실제 사실성 검증을 대신하지 못한다 → 후속 RAG(근거 검색)·검증 단계로 이어진다.
""")

nb.save(OUT)
sys.exit(0 if execute(OUT, WORKDIR) else 1)
