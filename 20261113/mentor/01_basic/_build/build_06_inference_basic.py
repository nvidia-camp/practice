"""20261113 실습 06 · 학습·추론·Sampling (기본필수) → 출력 저장된 Colab 노트북.

사용: python build_06_inference_basic.py <support.py 경로> <출력 ipynb> <실행용 임시 폴더>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbkit import Notebook, execute, setup_source, download_source

SUPPORT, OUT, WORKDIR = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
nb = Notebook()
md, code = nb.md, nb.code

md("""
# 실습 06 · 학습·추론·Sampling (기본필수)

LLM은 마지막에 다음 Token 후보마다 **점수(Logit)** 를 냅니다. 이 점수를 **확률**로 바꾸고, 그중 하나를 **고르는** 과정을 따라갑니다.

`Logit [2, 1, 0]` → `Softmax 확률` → `Temperature로 분포 조절` → `Greedy 선택`

### 사용 방법
1. **결과는 이미 셀 아래에 저장되어 있습니다.** 읽기만 해도 흐름을 따라갈 수 있어요.
2. 직접 돌려 보려면 `파일 → Drive에 사본 저장`으로 내 사본을 만드세요.
3. 맨 위 **⚙️ 준비 셀을 먼저 한 번 실행**한 뒤, 보고 싶은 구간의 셀을 실행하세요. 각 구간은 준비 셀만 있으면 따로 실행됩니다.
4. **✏️ 표시 셀**은 오른쪽 슬라이더의 값을 바꾸고 실행해 보세요. 코드의 값도 함께 바뀝니다.
""")

code(setup_source("20261113/mentor/01_basic/common/support.py",
                  SUPPORT.read_text(encoding="utf-8"), '''
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
torch.set_num_threads(1)

# 실습 입력 (main.py와 같은 값): 다음 Token 후보 3개의 점수
logits = torch.tensor([2.0, 1.0, 0.0])
labels = ["출력A", "출력B", "출력C"]
PLOT_LABELS = ["A", "B", "C"]          # 그래프용 (Colab 한글 폰트 없음)

def bar(probs, title, ax=None):
    """후보별 확률 막대그래프"""
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    p = probs.tolist()
    ax.bar(PLOT_LABELS, p, color=["C0", "C1", "C2"])
    for i, val in enumerate(p):
        ax.text(i, val + 0.02, f"{val:.3f}", ha="center")
    ax.set_ylim(0, 1.1); ax.set_ylabel("probability"); ax.set_title(title)
    if show:
        plt.show()

print("준비 완료 ✅  logits:", logits.tolist())
'''), form=True)

# ======================================================================
md("""
## 0. 후보와 점수(Logit)
점수가 클수록 그 후보가 다음 Token일 가능성이 높다는 뜻입니다. 하지만 **아직 확률은 아닙니다.**
""")
code('pd.DataFrame({"후보": labels, "Logit": logits.tolist()})')

# ======================================================================
md("""
## B01. Softmax: 점수 → 확률

$$p_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$$

### B01-1. 한 단계씩 계산
""")
code('''
exp = logits.exp()
print("e^Logit:", exp.round(decimals=4).tolist())
''')
code('''
total = exp.sum()
print("합계   :", round(total.item(), 4))
print("나누기 :", (exp / total).round(decimals=4).tolist())
''')

md("### B01-2. `torch.softmax`로 계산 (`main.py`와 같은 코드)")
code('''
prob = torch.softmax(logits, dim=0)
pd.DataFrame({"후보": labels, "Logit": logits.tolist(), "확률": prob.round(decimals=3).tolist()})
''')
code('bar(prob, "Softmax (T = 1.0)")')
code('''
assert torch.isclose(prob.sum(), torch.tensor(1.0), atol=1e-6)
assert torch.allclose(prob, logits.exp() / logits.exp().sum())
print("B01 통과 ✅ 확률 합 =", round(prob.sum().item(), 6), " 예상 [0.665, 0.245, 0.090]")
''')

# ======================================================================
md("""
## B02. Temperature: 분포를 뾰족하게 / 평평하게

Logit을 T로 나눈 뒤 Softmax를 합니다.
- **T < 1**: 점수 차이가 커져서 1등에 집중 (더 일정한 답)
- **T > 1**: 점수 차이가 줄어서 평평해짐 (더 다양한 답)
""")
code('''
for t in (0.5, 1.0, 2.0):
    print(f"T={t}: Logit/T = {(logits / t).tolist()}")
''')
code('''
temperature_results = {str(t): torch.softmax(logits / t, dim=0).tolist() for t in (0.5, 1.0, 2.0)}
pd.DataFrame(temperature_results, index=labels).round(3)
''')
code('''
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
for ax, t in zip(axes, (0.5, 1.0, 2.0)):
    bar(torch.softmax(logits / t, dim=0), f"T = {t}", ax=ax)
plt.tight_layout(); plt.show()
''')

md("### ✏️ B02 직접 해 보기 — Temperature를 움직여 보세요")
code('''
#@title ✏️ Temperature를 바꿔 보세요
temperature = 1.0  #@param {type:"slider", min:0.1, max:5, step:0.1}

p = torch.softmax(logits / temperature, dim=0)
print(f"T = {temperature}: 1등(출력A) 확률 {p[0].item():.3f}, 꼴등(출력C) 확률 {p[2].item():.3f}")
bar(p, f"T = {temperature}")
''')

# ======================================================================
md("""
## B03. 다음 Token 고르기와 학습/추론 차이

### B03-1. Greedy: 확률이 가장 큰 후보
""")
code('''
prob = torch.softmax(logits, dim=0)
selected_index = int(torch.argmax(prob).item())
selected_token = labels[selected_index]
print("선택:", selected_index, selected_token)
assert selected_token == "출력A"
print("B03 통과 ✅")
''')

md("""
### B03-2. 실험: Sampling은 확률대로 뽑는다
Greedy는 항상 같은 답을 고르지만, Sampling은 확률에 따라 뽑기 때문에 **실행할 때마다 달라질 수 있습니다.** 시드(seed)를 고정하면 같은 결과가 나옵니다. (응용·심화 내용 미리 보기)
""")
code('''
#@title ✏️ 시드와 Temperature를 바꿔 100번 뽑아 보세요
seed = 42  #@param {type:"integer"}
temperature = 1.0  #@param {type:"slider", min:0.1, max:5, step:0.1}

g = torch.Generator().manual_seed(seed)
draws = torch.multinomial(torch.softmax(logits / temperature, dim=0), 100, replacement=True, generator=g)
counts = torch.bincount(draws, minlength=3).tolist()
print("100번 중 뽑힌 횟수:", dict(zip(labels, counts)))
''')

md("### B03-3. 학습 vs 추론")
code('''
training_vs_inference = {
    "training": "정답 Token과 예측을 비교해 loss/backward/optimizer.step으로 파라미터 갱신",
    "inference": "가중치를 갱신하지 않고 현재 문맥에서 다음 Token을 선택",
    "autoregressive": "선택한 Token을 문맥에 추가하고 EOS/길이 제한까지 반복",
}
pd.DataFrame(training_vs_inference.items(), columns=["구분", "설명"])
''')

# ======================================================================
md("""
## 저장. 결과 저장 (`main.py`와 같은 결과)
확률·Temperature·선택·학습/추론 차이를 `outputs/inference_report.json`으로 저장합니다. 이 셀은 준비 셀만 있으면 혼자 실행됩니다.
""")
code('''
prob = torch.softmax(logits, dim=0)
assert torch.isclose(prob.sum(), torch.tensor(1.0), atol=1e-6)
temperature_results = {str(t): torch.softmax(logits / t, dim=0).tolist() for t in (0.5, 1.0, 2.0)}
selected_index = int(torch.argmax(prob).item())
selected_token = labels[selected_index]
training_vs_inference = {
    "training": "정답 Token과 예측을 비교해 loss/backward/optimizer.step으로 파라미터 갱신",
    "inference": "가중치를 갱신하지 않고 현재 문맥에서 다음 Token을 선택",
    "autoregressive": "선택한 Token을 문맥에 추가하고 EOS/길이 제한까지 반복",
}

write_json(OUT_DIR / "inference_report.json", {
    "logits": logits.tolist(), "probs": prob.tolist(), "temperature": temperature_results,
    "greedy": {"index": selected_index, "token": selected_token},
    "sampling": {"index": None, "token": None},
    "top_k": {}, "top_p": {}, "training_vs_inference": training_vs_inference,
    "training_step": {}, "kv_cache": {"stores": "이전 K/V 재사용", "not": "학습이나 장기 기억이 아님"},
})
print("[06]", selected_token, "probs", prob.round(decimals=3).tolist())
''')
code(download_source("inference_report.json"))

md("""
---
### 정리
- **Logit**은 점수, **Softmax**를 거쳐야 확률(합 = 1). `[2,1,0]` → `[0.665, 0.245, 0.090]`.
- **Temperature**가 낮으면 1등에 집중, 높으면 평평해진다.
- **Greedy**는 가장 큰 확률(출력A)을 고르고, **Sampling**은 확률대로 뽑아 시드에 따라 달라진다.
- **학습**은 가중치를 바꾸고, **추론**은 바꾸지 않고 다음 Token만 고른다.
""")

nb.save(OUT)
sys.exit(0 if execute(OUT, WORKDIR) else 1)
