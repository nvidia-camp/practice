"""Colab용 실습 노트북 생성·검증 도구.

생성 스크립트에서:
    from nbkit import Notebook, execute, setup_source, download_source

검증 CLI:
    python nbkit.py compare <노트북 outputs 폴더> <main.py outputs 폴더> <파일명...>
    python nbkit.py images  <ipynb> <이미지 저장 폴더>
    python nbkit.py summary <ipynb>
"""
import base64
import filecmp
import json
import os
import sys
from pathlib import Path

COLAB_METADATA = {
    "colab": {"provenance": [], "toc_visible": True},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


# ── 셀 소스 ──────────────────────────────────────────────────────────
def setup_source(origin, support_src, extra):
    """⚙️ 준비 셀: 원본 공통 코드(그대로) + 노트북 전용 준비 코드.

    origin: 저장소 기준 원본 경로 문자열 (예: "20261113/mentor/01_basic/common/support.py")
    support_src: 원본 파일 내용 (스크립트가 파일에서 읽어 전달)
    extra: 실습 상수·도우미 함수 등 노트북 전용 코드
    """
    return f'''#@title ⚙️ 먼저 실행하세요 — 실습 공통 코드 준비
# ── 원본: {origin} (그대로 포함) ──
{support_src.strip()}

# ── 노트북 전용 준비 ──
import pandas as pd
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

{extra.strip()}
'''


def download_source(*names):
    """📥 결과 파일 내려받기 셀 (Colab에서만 동작)."""
    files = ", ".join(repr(n) for n in names)
    return f'''#@title 📥 결과 파일 내려받기 (Colab)
if IN_COLAB:
    from google.colab import files
    for name in [{files}]:
        files.download(str(OUT_DIR / name))
else:
    print("Colab에서 실행하면 결과 파일이 내려받아집니다:", [{files}])
'''


# ── 노트북 작성 ──────────────────────────────────────────────────────
class Notebook:
    def __init__(self):
        self.cells = []

    def md(self, text):
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})

    def code(self, text, form=False):
        """form=True: Colab에서 코드를 접고 #@title만 보이게 함 (준비 셀용)."""
        meta = {"cellView": "form"} if form else {}
        self.cells.append({"cell_type": "code", "execution_count": None, "metadata": meta,
                           "outputs": [], "source": text.strip("\n")})

    def save(self, path):
        for i, c in enumerate(self.cells):
            c["id"] = f"c{i:03d}"
        nb = {"cells": self.cells, "metadata": COLAB_METADATA, "nbformat": 4, "nbformat_minor": 5}
        Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def execute(path, workdir):
    """workdir(빈 폴더 권장)에서 노트북을 실행하고 출력을 저장한다.

    오류 셀이 있거나 출력에 로컬 경로가 남으면 False.
    """
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    try:
        NotebookClient(nb, timeout=180, kernel_name="python3",
                       resources={"metadata": {"path": str(workdir)}}).execute()
    except Exception as e:  # 실패해도 어느 셀인지 보이도록 저장
        print("execution stopped:", type(e).__name__, str(e).splitlines()[0][:200])
    for c in nb.cells:
        c.metadata.pop("execution", None)
    nb.metadata = COLAB_METADATA | {"language_info": nb.metadata.get("language_info", {"name": "python"})}
    nbformat.write(nb, path)

    text = Path(path).read_text(encoding="utf-8")
    home = os.path.expanduser("~")
    leaks = [s for s in {home, home.replace("\\", "\\\\"), str(workdir)} if s and s in text]
    errors = [c.get("id") for c in nb.cells if c.cell_type == "code"
              and any(o.get("output_type") == "error" for o in c.outputs)]
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"executed {Path(path).name}: {len(nb.cells)} cells ({n_code} code), "
          f"errors={errors or 0}, path_leaks={leaks}")
    return not errors and not leaks


# ── 검증 CLI ─────────────────────────────────────────────────────────
def _text(s):
    return "".join(s) if isinstance(s, list) else s


def compare(nb_out, lab_out, names):
    ok = True
    for name in names:
        a, b = Path(nb_out) / name, Path(lab_out) / name
        if not a.exists() or not b.exists():
            print(f"MISSING  {name}  (notebook={a.exists()}, main.py={b.exists()})"); ok = False
        elif filecmp.cmp(a, b, shallow=False):
            print(f"SAME     {name}")
        else:
            print(f"DIFF     {name}"); ok = False
    return ok


def images(ipynb, out_dir):
    nb = json.loads(Path(ipynb).read_text(encoding="utf-8"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n = 0
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            png = o.get("data", {}).get("image/png")
            if png:
                p = Path(out_dir) / f"{c.get('id', n)}_{n}.png"
                p.write_bytes(base64.b64decode(_text(png))); n += 1
                print(p)
    print("images:", n)


def summary(ipynb):
    nb = json.loads(Path(ipynb).read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        first = _text(c["source"]).splitlines()[0][:45]
        kinds = []
        for o in c["outputs"]:
            if o["output_type"] == "stream":
                kinds.append(o["name"] + ":" + _text(o["text"]).strip().splitlines()[0][:50])
            elif o["output_type"] == "error":
                kinds.append("ERROR:" + o.get("ename", ""))
            else:
                kinds.append("+".join(k for k in o.get("data", {}) if k != "text/plain") or "text")
        print(f"{c.get('id')} | {first:45} | {kinds or ['(no output)']}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["help"]
    if cmd == "compare" and len(args) >= 3:
        sys.exit(0 if compare(args[0], args[1], args[2:]) else 1)
    elif cmd == "images" and len(args) == 2:
        images(*args)
    elif cmd == "summary" and len(args) == 1:
        summary(*args)
    else:
        print(__doc__); sys.exit(2)
