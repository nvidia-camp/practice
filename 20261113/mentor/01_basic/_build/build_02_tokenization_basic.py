"""20261113 실습 02 · Token·ID·Context (기본필수) → 출력 저장된 Colab 노트북.

사용: python build_02_tokenization_basic.py <support.py 경로> <출력 ipynb> <실행용 임시 폴더>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbkit import Notebook, execute, setup_source, download_source

SUPPORT, OUT, WORKDIR = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

nb = Notebook()
md, code = nb.md, nb.code

# ======================================================================
md("""
# 실습 02 · Token·ID·Context 분석 (기본필수)

문장이 LLM에 들어가기 전에 **숫자(Token ID)로 바뀌는 과정**을 셀 하나씩 확인합니다.

### 사용 방법
1. **결과는 이미 셀 아래에 저장되어 있습니다.** 읽기만 해도 흐름을 따라갈 수 있어요.
2. 직접 돌려 보려면 `파일 → Drive에 사본 저장`으로 내 사본을 만드세요.
3. 맨 위 **⚙️ 준비 셀을 먼저 한 번 실행**한 뒤, 보고 싶은 구간의 셀을 실행하세요. 각 구간은 준비 셀만 있으면 따로 실행됩니다.
4. **✏️ 표시 셀**은 오른쪽 입력 칸의 값을 바꾸고 실행해 보세요.

| 구간 | 내용 |
|---|---|
| 0 | 실습 데이터 보기 |
| 1 | Tokenizer(단어 사전) |
| B01 | 문장 → Token·ID, 개수 세기 |
| B02 | 길이 맞추기(Padding)와 Mask |
| 실험 | 잘림(Truncation)과 모르는 단어(UNK) |
| B03 | Context 예산 계산 |
| B04 | 결과 저장 (`main.py`와 같은 결과) |
""")

code(setup_source("20261113/mentor/01_basic/common/support.py",
                  SUPPORT.read_text(encoding="utf-8"), '''
texts = [item["content"] for item in DATA]
tok = ToyTokenizer(texts + ["AI", "FastAPI와 PostgreSQL", "print(x) // code"])

def token_table(record):
    """encode 결과를 위치별 표로 보여 줌: token / id / mask"""
    return pd.DataFrame([record["tokens"], record["input_ids"], record["attention_mask"]],
                        index=["token", "id", "mask"])

print("준비 완료 ✅  사전 크기:", len(tok.vocab))
'''), form=True)

# ======================================================================
md("""
## 0. 실습 데이터 보기
지식 문장 3건(A, B, C)입니다. 11/12에 DB에 넣었던 지식 데이터와 같은 형태입니다. Tokenizer에는 `content`만 넣습니다.
""")
code("pd.DataFrame(DATA)")

# ======================================================================
md("""
## 1. Tokenizer(단어 사전)

### 1-1. 문장을 조각으로 자르기
한글 덩어리 / 영어 / 숫자 / 문장부호 단위로 자릅니다.
""")
code('words("실습 파일은 개인 폴더에 보관합니다.")')
code('words("FastAPI와 PostgreSQL")')

md("""
### 1-2. 사전 보기
조각마다 번호가 붙어 있습니다. 앞의 0~3번은 **특수 Token**입니다.
`[PAD]` 빈칸 · `[UNK]` 모르는 단어 · `[BOS]` 문장 시작 · `[EOS]` 문장 끝
""")
code('pd.DataFrame(tok.vocab.items(), columns=["token", "id"]).set_index("id").T')

# ======================================================================
md("""
## B01. 문장 → Token·ID

### B01-1. 문장 A 하나를 변환
`encode()` 결과는 딕셔너리입니다.
""")
code('''
record = tok.encode(DATA[0]["content"])
record
''')

md("""
### B01-2. 표로 보기
위치마다 **token(조각) → id(번호) → mask(1 = 진짜 Token)** 입니다. 맨 앞뒤에 `[BOS]`, `[EOS]`가 붙은 것을 확인하세요.
""")
code("token_table(record)")

md("""
### B01-3. 단어 수 vs Token 수
보이는 단어는 6개인데 Token은 8개입니다. 특수 Token 2개가 더해지기 때문입니다.
""")
code('''
sentence = DATA[0]["content"]
print("단어 수  :", len(words(sentence)))
print("Token 수 :", tok.encode(sentence)["valid_length"])
''')

md("### B01-4. A/B/C 전체 처리 (`main.py`와 같은 코드)")
code('''
rows = []
for item in DATA:
    record = tok.encode(item["content"])
    rows.append({"id": item["id"], "text": item["content"],
                 "tokens": record["tokens"], "ids": record["input_ids"],
                 "count": record["valid_length"], "special_count": 2})

pd.DataFrame(rows)[["id", "count", "tokens", "ids"]]
''')
code('''
counts = [(r["id"], r["count"]) for r in rows]
assert counts == [("A", 8), ("B", 7), ("C", 9)], counts
print("B01 통과 ✅", counts)
''')

md("### ✏️ B01 직접 해 보기")
code('''
#@title ✏️ 문장을 바꿔 보세요
sentence = "실습 파일은 개인 폴더에 보관합니다."  #@param {type:"string"}

my = tok.encode(sentence)
print("Token 수:", my["valid_length"], "  (사전에 없는 단어는 [UNK] = 1)")
token_table(my)
''')

# ======================================================================
md("""
## B02. 길이 맞추기(Padding)와 Mask
여러 문장을 한 번에 처리하려면 **길이가 같아야** 합니다. 짧은 문장 뒤에 `[PAD]`(id 0)를 채웁니다.

### B02-1. 따로 변환하면 길이가 다르다
""")
code('''
print("AI                  :", len(tok.encode("AI")["input_ids"]))
print("FastAPI와 PostgreSQL:", len(tok.encode("FastAPI와 PostgreSQL")["input_ids"]))
''')

md("### B02-2. `batch()`로 함께 변환하면 길이가 맞춰진다")
code('''
batch = tok.batch(["AI", "FastAPI와 PostgreSQL"])
token_table(batch[0])
''')
code("token_table(batch[1])")

md("""
### B02-3. 전체 길이 vs 실제 Token 수
**가장 헷갈리는 부분입니다.** PAD를 채워도 실제 Token 수(`valid_length`)는 그대로이고, mask가 0인 칸이 PAD입니다.
""")
code('''
pd.DataFrame([{"문장": ["AI", "FastAPI와 PostgreSQL"][i],
               "전체 길이": len(x["input_ids"]),
               "실제 Token": x["valid_length"],
               "PAD 수": x["attention_mask"].count(0)} for i, x in enumerate(batch)])
''')
code('''
assert len(batch[0]["input_ids"]) == len(batch[1]["input_ids"])
assert all(len(x["input_ids"]) == len(x["attention_mask"]) for x in batch)
print("B02 통과 ✅ 맞춰진 길이 =", len(batch[0]["input_ids"]))
''')

md("### ✏️ B02 직접 해 보기")
code('''
#@title ✏️ 길이가 다른 두 문장을 넣어 보세요
first = "AI"  #@param {type:"string"}
second = "실습 파일은 개인 폴더에 보관합니다."  #@param {type:"string"}

pair = tok.batch([first, second])
display(token_table(pair[0]))
display(token_table(pair[1]))
''')

# ======================================================================
md("""
## 실험. 잘림(Truncation)과 모르는 단어(UNK)
과제에는 없지만 실제 Tokenizer에서도 똑같이 일어납니다. `max_length`보다 길면 뒷부분이 잘려 **모델이 아예 보지 못합니다.**
""")
code('''
#@title ✏️ 잘라 낼 길이를 바꿔 보세요
sentence = "오류 메시지와 시도한 작업을 함께 적습니다."  #@param {type:"string"}
max_length = 5  #@param {type:"slider", min:2, max:12, step:1}

cut = tok.encode(sentence, max_length=max_length)
print("원래 Token 수:", cut["original_length"], " → 잘렸나?", cut["truncated"])
token_table(cut)
''')

# ======================================================================
md("""
## B03. Context 예산 계산
LLM이 한 번에 다루는 Token 한도(Context Window) 안에 **입력 + 답변 공간**이 모두 들어가야 합니다.

| 항목 | 의미 |
|---|---|
| system | 시스템 지시문 |
| history | 이전 대화 |
| evidence | 찾아온 근거 문서 (11/20 RAG) |
| user | 이번 질문 |
| 출력 예약 | 답변을 쓸 공간 |
""")
code('''
parts = {"system": 300, "history": 1200, "evidence": 2000, "user": 150}
budget = context_budget(5000, parts, 850)
budget
''')
code('''
print(f"입력 {budget['input_used']} + 출력 예약 {budget['output_reserved']} "
      f"= {budget['input_used'] + budget['output_reserved']} / 한도 {budget['window']}")
print("남는 공간:", budget["remaining"])
assert budget["remaining"] == 500 and budget["fits"]
print("B03 통과 ✅")
''')

md("### ✏️ B03 직접 해 보기 — 근거 문서를 늘리면?")
code('''
#@title ✏️ 숫자를 바꿔 보세요
window = 5000  #@param {type:"integer"}
system = 300  #@param {type:"integer"}
history = 1200  #@param {type:"integer"}
evidence = 2000  #@param {type:"slider", min:0, max:5000, step:100}
user = 150  #@param {type:"integer"}
output_reserved = 850  #@param {type:"integer"}

b = context_budget(window, {"system": system, "history": history, "evidence": evidence, "user": user}, output_reserved)
print("남는 공간:", b["remaining"])
print("✅ 들어갑니다" if b["fits"] else "❌ 넘칩니다 — history나 evidence를 줄여야 합니다")
''')

# ======================================================================
md("""
## B04. 결과 저장 (`main.py`와 같은 결과)
B01~B03 결과를 모아 `outputs/token_report.json`으로 저장합니다. 이 셀은 준비 셀만 있으면 혼자 실행됩니다.
""")
code('''
rows = []
for item in DATA:
    record = tok.encode(item["content"])
    rows.append({"id": item["id"], "text": item["content"],
                 "tokens": record["tokens"], "ids": record["input_ids"],
                 "count": record["valid_length"], "special_count": 2})
batch = tok.batch(["AI", "FastAPI와 PostgreSQL"])
budget = context_budget(5000, {"system": 300, "history": 1200, "evidence": 2000, "user": 150}, 850)

result = {"tokenizer_kind": "교육용 regex·고정 어휘표 (사전학습 모델이 아님)",
          "records": rows, "batch": batch, "budget": budget, "truncation": {},
          "batch_report": [], "language_comparison": [],
          "overflow_budget": None, "policy": "예산과 잘림의 차이를 설명"}
write_json(OUT_DIR / "token_report.json", result)

print("[02]", [(x["id"], x["count"]) for x in rows], "remaining", budget["remaining"])
print("저장:", OUT_DIR / "token_report.json")
''')
code(download_source("token_report.json"))

md("""
---
### 정리
- **B01** 문장 → Token → ID. Token 수에는 `[BOS]`/`[EOS]`가 포함된다.
- **B02** 여러 문장을 묶으면 PAD로 길이를 맞춘다. mask 0 = PAD, 실제 Token 수는 `valid_length`.
- **실험** 한도를 넘으면 뒷부분이 잘려 모델이 보지 못한다.
- **B03** Context Window에는 입력뿐 아니라 **답변 공간**도 남겨야 한다.
""")

nb.save(OUT)
ok = execute(OUT, WORKDIR)
sys.exit(0 if ok else 1)
