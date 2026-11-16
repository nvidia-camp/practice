"""20261113 실습 01 · AI 서비스 구성요소 연결 (기본필수) → 출력 저장된 Colab 노트북.

사용: python build_01_ecosystem_basic.py <support.py 경로> <출력 ipynb> <실행용 임시 폴더>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbkit import Notebook, execute, setup_source, download_source

SUPPORT, OUT, WORKDIR = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
nb = Notebook()
md, code = nb.md, nb.code

ROLES = '''roles = {
    "FastAPI": "HTTP 요청을 검증하고 함수·응답을 연결한다",
    "PostgreSQL": "원본 지식·분류를 관계형 테이블에 저장한다",
    "LLM": "문맥과 입력을 바탕으로 다음 Token을 생성한다",
    "Response": "호출한 Client에 JSON 결과를 전달한다",
}'''
EDGES = '''edges = [
    ("Client", "FastAPI", "HTTP 요청"),
    ("FastAPI", "PostgreSQL", "원본 지식 조회·수정"),
    ("PostgreSQL", "FastAPI", "원본 데이터"),
    ("FastAPI", "LLM", "입력·문맥(후속 구현)"),
    ("LLM", "Response", "생성 결과(후속 구현)"),
    ("Response", "Client", "JSON"),
]'''

md("""
# 실습 01 · AI 서비스 구성요소 연결 (기본필수)

11/10~12에 만든 지식 관리 서비스(FastAPI + PostgreSQL)에 앞으로 LLM이 붙으면 **각 구성요소가 무슨 일을 하고 어떤 순서로 연결되는지**를 코드로 정리하고 그림으로 그립니다. LLM을 실제로 호출하지는 않습니다.

### 사용 방법
1. **결과는 이미 셀 아래에 저장되어 있습니다.** 읽기만 해도 흐름을 따라갈 수 있어요.
2. 직접 돌려 보려면 `파일 → Drive에 사본 저장`으로 내 사본을 만드세요.
3. 맨 위 **⚙️ 준비 셀을 먼저 한 번 실행**한 뒤, 보고 싶은 구간의 셀을 실행하세요. 각 구간은 준비 셀만 있으면 따로 실행됩니다.
4. **✏️ 표시 셀**은 오른쪽 입력 칸의 값을 바꾸고 실행해 보세요. 코드의 값도 함께 바뀝니다.

| 구간 | 내용 |
|---|---|
| B01 | 구성요소 4개의 역할 |
| B02 | 요청 → 응답 연결 순서 (구현됨 / 후속 구분) |
| B03 | Mermaid 그림 만들기 |
| 저장 | 결과 저장 (`main.py`와 같은 결과) |

> 그림 렌더링은 공개 서비스 mermaid.ink 이미지를 사용하므로 인터넷 연결이 필요합니다.
""")

code(setup_source("20261113/mentor/01_basic/common/support.py",
                  SUPPORT.read_text(encoding="utf-8"), '''
import base64
from IPython.display import Image, display

def show_mermaid(mermaid_code):
    """Mermaid 텍스트를 그림으로 보여 줌 (mermaid.ink 이미지, 인터넷 필요)"""
    encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("ascii")
    display(Image(url="https://mermaid.ink/img/" + encoded))

print("준비 완료 ✅")
'''), form=True)

# ======================================================================
md("""
## B01. 구성요소 4개의 역할
역할을 섞지 않는 것이 핵심입니다. **DB는 다음 Token을 계산하지 않고, LLM은 영구 저장소를 대신하지 않습니다.**
""")
code(ROLES + "\n\npd.DataFrame(roles.items(), columns=[\"구성요소\", \"역할\"])")
code('''
assert set(roles) == {"FastAPI", "PostgreSQL", "LLM", "Response"}
assert all(text.strip() for text in roles.values())
print("B01 통과 ✅ 역할", len(roles), "개 작성")
''')

# ======================================================================
md("""
## B02. 요청 → 응답 연결 순서
`(출발, 도착, 설명)` 튜플 하나가 화살표 하나입니다. LLM이 들어간 연결은 **아직 만들지 않은 후속 단계**라서 설명에 `(후속 구현)`을 붙입니다.
""")
code(EDGES + "\nlen(edges)")
code('''
table = pd.DataFrame(edges, columns=["출발", "도착", "설명"])
table["상태"] = ["후속" if "후속" in label else "구현됨" for label in table["설명"]]
table
''')
md("요청은 Client에서 시작해 Client로 돌아와야 합니다.")
code('''
assert edges[0][0] == "Client" and edges[-1][1] == "Client"
future = [f"{a}→{b}" for a, b, label in edges if "후속" in label]
print("후속 단계로 표시된 연결:", future)
assert all("LLM" in f for f in future)
print("B02 통과 ✅")
''')

# ======================================================================
md("""
## B03. Mermaid 그림 만들기
Mermaid는 **글자로 쓰면 그림이 되는 표기법**입니다. 노드 이름에 `N0`, `N1` 같은 ID를 붙이고, 노드 줄과 화살표 줄을 차례로 만듭니다.

### B03-1. 노드 모으기와 ID 붙이기
""")
code(EDGES + '''
node_names = sorted({n for a, b, _ in edges for n in (a, b)})
node_names
''')
code('''
node_id = {name: f"N{i}" for i, name in enumerate(node_names)}
node_id
''')

md("### B03-2. Mermaid 줄 만들기")
code('''
lines = ["flowchart LR"]                                                   # 왼쪽 → 오른쪽 그림
lines += [f'  {node_id[name]}["{name}"]' for name in node_names]            # 노드 줄
lines += [f'  {node_id[a]} -->|"{label}"| {node_id[b]}' for a, b, label in edges]  # 화살표 줄
print("\\n".join(lines))
''')

md("### B03-3. 그림으로 보기")
code('show_mermaid("\\n".join(lines))')

md("### ✏️ B03 직접 해 보기 — 연결을 하나 추가해 보세요")
code('''
#@title ✏️ 새 연결을 추가해 보세요 (예: 후속 수업의 Retriever)
source = "FastAPI"  #@param {type:"string"}
target = "Retriever"  #@param {type:"string"}
label = "검색 요청(후속 구현)"  #@param {type:"string"}

''' + EDGES + '''
my_edges = edges + [(source, target, label)]
names = sorted({n for a, b, _ in my_edges for n in (a, b)})
ids = {name: f"N{i}" for i, name in enumerate(names)}
my_lines = ["flowchart LR"]
my_lines += [f'  {ids[n]}["{n}"]' for n in names]
my_lines += [f'  {ids[a]} -->|"{t}"| {ids[b]}' for a, b, t in my_edges]
show_mermaid("\\n".join(my_lines))
''')

# ======================================================================
md("""
## 저장. 결과 저장 (`main.py`와 같은 결과)
`outputs/architecture.md`(Mermaid 그림)와 `outputs/architecture.json`(역할·연결 요약)을 저장합니다. 이 셀은 준비 셀만 있으면 혼자 실행됩니다.
""")
code(ROLES + "\n" + EDGES + '''
responsibility = {"storage": "DB는 원본 저장", "generation": "LLM은 언어 생성"}
trust_boundary = {"scope": "후속 구현 개념도"}

node_names = sorted({n for a, b, _ in edges for n in (a, b)})
node_id = {name: f"N{i}" for i, name in enumerate(node_names)}
lines = ["flowchart LR"]
lines += [f'  {node_id[name]}["{name}"]' for name in node_names]
lines += [f'  {node_id[a]} -->|"{label}"| {node_id[b]}' for a, b, label in edges]
write_md(OUT_DIR / "architecture.md", "# AI 지식 플랫폼 개념도\\n\\n```mermaid\\n" + "\\n".join(lines) + "\\n```\\n")
write_json(OUT_DIR / "architecture.json", {
    "roles": roles, "edges": [list(x) for x in edges],
    "responsibility": responsibility, "trust_boundary": trust_boundary,
})
print("[01] architecture.md / architecture.json 생성 완료")
''')
code('print((OUT_DIR / "architecture.md").read_text(encoding="utf-8"))')
code(download_source("architecture.md", "architecture.json"))

md("""
---
### 정리
- **FastAPI**는 요청·응답 연결, **PostgreSQL**은 원본 저장, **LLM**은 다음 Token 생성, **Response**는 결과 전달.
- 오늘 실제로 동작하는 것은 Client ↔ FastAPI ↔ PostgreSQL까지이고, **LLM 연결은 후속 단계**다.
- Mermaid는 그림을 그리는 텍스트 표기법이지 API 호출이 아니다.
""")

nb.save(OUT)
sys.exit(0 if execute(OUT, WORKDIR) else 1)
