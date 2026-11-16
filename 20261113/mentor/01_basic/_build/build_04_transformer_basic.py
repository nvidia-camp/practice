"""20261113 실습 04 · Transformer 구조·Shape (기본필수) → 출력 저장된 Colab 노트북.

사용: python build_04_transformer_basic.py <support.py 경로> <출력 ipynb> <실행용 임시 폴더>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbkit import Notebook, execute, setup_source, download_source

SUPPORT, OUT, WORKDIR = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
nb = Notebook()
md, code = nb.md, nb.code

md("""
# 실습 04 · Transformer 구조·Shape (기본필수)

Token ID가 Transformer를 지나면서 **데이터 모양(shape)이 어떻게 바뀌는지** 따라갑니다.

`Token IDs (2, 5)` → `Embedding (2, 5, 8)` → `Encoder 출력 (2, 5, 8)`

| 기호 | 뜻 | 이 실습 값 |
|---|---|---|
| B | 한 번에 처리하는 문장 수 (batch) | 2 |
| T | 문장 길이 (Token 수) | 5 |
| D | Token 하나를 나타내는 숫자 개수 (d_model) | 8 |

### 사용 방법
1. **결과는 이미 셀 아래에 저장되어 있습니다.** 읽기만 해도 흐름을 따라갈 수 있어요.
2. 직접 돌려 보려면 `파일 → Drive에 사본 저장`으로 내 사본을 만드세요.
3. 맨 위 **⚙️ 준비 셀을 먼저 한 번 실행**한 뒤, 보고 싶은 구간의 셀을 실행하세요. 각 구간은 준비 셀만 있으면 따로 실행됩니다.
4. **✏️ 표시 셀**은 오른쪽 슬라이더의 값을 바꾸고 실행해 보세요. 코드의 값도 함께 바뀝니다.

> ⚠️ 가중치는 학습되지 않은 무작위 값입니다. 숫자 자체보다 **shape**에 집중하세요.
""")

code(setup_source("20261113/mentor/01_basic/common/support.py",
                  SUPPORT.read_text(encoding="utf-8"), '''
from torch import nn

torch.manual_seed(42)
torch.set_num_threads(1)

# 실습 입력 (main.py와 같은 값): 문장 2개 × Token 5개, 0 = [PAD]
token_ids = torch.tensor([[2, 4, 5, 3, 0], [2, 6, 7, 3, 0]], dtype=torch.long)

def table(t, row_name="pos", digits=3):
    """2차원 텐서를 표로 보여 줌"""
    return pd.DataFrame(t.detach().numpy()).round(digits).rename_axis(row_name)

print("준비 완료 ✅  token_ids shape:", list(token_ids.shape))
'''), form=True)

# ======================================================================
md("""
## 0. 입력 Token ID
문장 2개, 각 5칸입니다. 맨 끝의 `0`은 길이를 맞추려고 채운 `[PAD]`입니다. (02 실습의 Padding)
""")
code('''
pd.DataFrame(token_ids.numpy(), index=["문장1", "문장2"],
             columns=[f"t{i}" for i in range(token_ids.shape[1])])
''')
code('print("token_ids shape (B, T):", list(token_ids.shape))')

# ======================================================================
md("""
## B01. Embedding: ID → 벡터

`nn.Embedding`은 **ID 번호로 행을 찾아 오는 표**입니다. 16개 단어 × 8개 숫자 표에서 ID 행을 꺼내 옵니다.

### B01-1. Embedding 표 만들기
""")
code('''
torch.manual_seed(42)                     # main.py와 같은 무작위 값
embedding = nn.Embedding(num_embeddings=16, embedding_dim=8)
print("Embedding 표 shape (단어 수, D):", list(embedding.weight.shape))
''')
code('table(embedding.weight, "id").head(8)   # 표의 앞 8행')

md("### B01-2. ID를 넣으면 벡터가 나온다")
code('''
x = embedding(token_ids)
print("token_ids shape:", list(token_ids.shape), " → x shape:", list(x.shape))
''')
code('table(x[0], "문장1 위치")   # 문장1의 Token 5개 × 숫자 8개')

md("""
### B01-3. 같은 ID는 같은 벡터
두 문장의 첫 Token은 모두 ID 2(`[BOS]`)입니다. 꺼내 온 벡터도 같고, Embedding 표의 2번 행과도 같습니다.
""")
code('''
same = torch.equal(x[0, 0], x[1, 0]) and torch.equal(x[0, 0], embedding.weight[2])
print("문장1 첫 Token == 문장2 첫 Token == 표의 2번 행:", same)
''')
code('''
assert list(token_ids.shape) == [2, 5] and list(x.shape) == [2, 5, 8] and same
print("B01 통과 ✅", list(token_ids.shape), "=>", list(x.shape))
''')

# ======================================================================
md("""
## B02. Encoder Layer: 모양은 그대로, 내용은 섞인다

Encoder Layer는 Attention(Token끼리 정보 주고받기)과 FFN(특징 변환)으로 이루어져 있습니다. **입력과 출력의 shape이 같아서** 여러 층을 쌓을 수 있습니다.

### B02-1. Layer 만들기
""")
code('''
torch.manual_seed(42)
embedding = nn.Embedding(16, 8)
x = embedding(token_ids)

layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16,
                                   batch_first=True, dropout=0.0)
layer.eval()
print("d_model=8, nhead=2 → Head 하나가 맡는 차원:", 8 // 2)
print("학습할 파라미터 수:", sum(p.numel() for p in layer.parameters()))
''')

md("""
### B02-2. Padding Mask
`True`인 칸(= PAD)은 Attention에서 **Key로 참고하지 않도록** 막습니다.
""")
code('''
pad_mask = (token_ids == 0)
pd.DataFrame(pad_mask.numpy(), index=["문장1", "문장2"], columns=[f"t{i}" for i in range(5)])
''')

md("### B02-3. Encoder 통과")
code('''
with torch.no_grad():
    y = layer(x, src_key_padding_mask=pad_mask)
print("x shape:", list(x.shape), " → y shape:", list(y.shape))
''')
code('table(y[0], "문장1 위치")   # 같은 모양, 다른 값')

md("""
### B02-4. 주의: PAD 위치의 출력은 0이 아니다
Padding Mask는 PAD를 **참고 대상(Key)** 에서 빼는 것이지, PAD 위치의 **출력**을 0으로 만들지 않습니다. 그래서 뒤 단계에서 PAD 위치 결과는 따로 무시해야 합니다.
""")
code('''
print("문장1 PAD 위치(t4) 출력 절댓값 합:", round(y[0, 4].abs().sum().item(), 4))
''')
code('''
assert x.shape == y.shape
print("B02 통과 ✅", list(x.shape), "=>", list(y.shape))
''')

md("""
### ✏️ B02 직접 해 보기 — 크기를 바꿔 보세요
D가 nhead로 나누어떨어지지 않으면 Layer를 만들 수 없습니다. 예: D=8, nhead=3.
""")
code('''
#@title ✏️ B, T, D, nhead를 바꿔 보세요
B = 2  #@param {type:"slider", min:1, max:6, step:1}
T = 5  #@param {type:"slider", min:1, max:12, step:1}
D = 8  #@param {type:"slider", min:2, max:16, step:2}
nhead = 2  #@param {type:"slider", min:1, max:4, step:1}

ids = torch.randint(0, 16, (B, T))
try:
    my_layer = nn.TransformerEncoderLayer(d_model=D, nhead=nhead, dim_feedforward=16,
                                          batch_first=True, dropout=0.0).eval()
    with torch.no_grad():
        my_x = nn.Embedding(16, D)(ids)
        my_y = my_layer(my_x)
    print("Token IDs", list(ids.shape), "→ Embedding", list(my_x.shape), "→ Encoder", list(my_y.shape))
    print("✅ 입력과 출력 shape이 같습니다")
except AssertionError as e:
    print(f"❌ D={D}는 nhead={nhead}로 나누어떨어지지 않아 Layer를 만들 수 없습니다")
''')

# ======================================================================
md("""
## 저장. 결과 저장 (`main.py`와 같은 결과)
모든 shape과 Mask를 `outputs/transformer_report.json`으로 저장합니다. 이 셀은 준비 셀만 있으면 혼자 실행됩니다.
""")
code('''
torch.manual_seed(42)
embedding = nn.Embedding(num_embeddings=16, embedding_dim=8)
x = embedding(token_ids)
id_shape, emb_shape = list(token_ids.shape), list(x.shape)

layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16,
                                   batch_first=True, dropout=0.0)
layer.eval()
pad_mask = (token_ids == 0)
with torch.no_grad():
    y = layer(x, src_key_padding_mask=pad_mask)
assert x.shape == y.shape
out_shape = list(y.shape)

write_json(OUT_DIR / "transformer_report.json", {
    "ids": id_shape, "embedding": emb_shape, "encoder_output": out_shape,
    "pad_key_mask": pad_mask.tolist(), "padding_positions": [],
    "padding_output_abs_sum": None, "padding_note": "PAD Mask는 Causal Mask와 별개",
    "components": {"attention": "토큰 간 정보", "ffn": "특징 변환", "shape": out_shape},
    "bad_head_rejected": None, "extra_shapes": {},
    "architecture_note": "이 코드는 Encoder Layer의 Shape 실습",
})
print("[04]", id_shape, "=>", emb_shape, "=>", out_shape)
''')
code(download_source("transformer_report.json"))

md("""
---
### 정리
- **Embedding**: ID 하나 → 숫자 D개 벡터. `(B, T)` → `(B, T, D)`. 같은 ID는 같은 벡터.
- **Encoder Layer**: shape `(B, T, D)`를 유지한 채 Token끼리 정보를 섞는다. D는 nhead로 나누어떨어져야 한다.
- **Padding Mask**는 PAD를 참고 대상에서 뺄 뿐, PAD 위치 출력을 0으로 만들지 않는다.
""")

nb.save(OUT)
sys.exit(0 if execute(OUT, WORKDIR) else 1)
