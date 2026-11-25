"""실습 07 보고서 CLI가 저장된 평가 결과를 변환하는지 검증합니다."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report.py"


@pytest.fixture
def basic_result():
    return {
        "exercise": 7,
        "level": "01_basic",
        "dataset_version": "d28-eval-v1",
        "mode": "live",
        "comparisons": [
            {
                "question_id": "Q01",
                "baseline_recall": 1.0,
                "rerank_recall": 1.0,
                "delta_recall": 0.0,
                "diagnosis": "inspect_answer_grounding",
            },
        ],
        "comparison_kind": "offline_version_comparison_not_online_AB",
    }


def run_report(tmp_path, result, *options):
    source = tmp_path / "result.json"
    source.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(source), *options],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=10,
    )


def test_basic_result_from_another_directory(tmp_path, basic_result):
    process = run_report(tmp_path, basic_result)

    assert process.returncode == 0, process.stderr
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "난이도 `01_basic`" in markdown
    assert "데이터셋 버전 `d28-eval-v1`" in markdown
    assert "실행 모드 `live`" in markdown
    assert "오프라인 버전 비교" in markdown
    assert "| Q01 | 1.0 | 1.0 | 0.0 | `inspect_answer_grounding`" in markdown
    assert "답변의 주장을 Context 및 출처 원문과 대조" in markdown
    assert "변형별 검색 지표" not in markdown


def test_advanced_result_preserves_null_and_regression(tmp_path, basic_result):
    basic_result["level"] = "03_advanced"
    basic_result["mode"] = "offline_fixture"
    basic_result["comparisons"][0]["matrix"] = {
        "rerank_only": {"k": 3, "recall_at_k": 1.0, "rr_at_k": 1.0, "ndcg_at_k": 1.0},
    }
    basic_result["comparisons"][0]["regression"] = {
        "hold_release": True,
        "error_count": 1,
        "reason": "회귀/오류 발견",
    }
    basic_result["comparisons"].append({
        "question_id": "Q06",
        "baseline_recall": None,
        "rerank_recall": None,
        "delta_recall": None,
        "diagnosis": "no_answer_case",
    })
    output = tmp_path / "reports" / "custom.md"

    process = run_report(tmp_path, basic_result, "--output", str(output))

    assert process.returncode == 0, process.stderr
    markdown = output.read_text(encoding="utf-8")
    assert "교육용 고정 데이터" in markdown
    assert "| Q06 | — | — | — | `no_answer_case`" in markdown
    assert "| Q01 | `rerank_only` | 3 | 1.0 | 1.0 | 1.0 |" in markdown
    assert "| Q01 | 보류 | 1 | 회귀/오류 발견 |" in markdown


def test_failed_input_preserves_existing_report(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("기존 보고서", encoding="utf-8")

    process = run_report(tmp_path, {"exercise": 7, "status": "execution_failed"})

    assert process.returncode != 0
    assert "comparisons" in process.stderr
    assert report.read_text(encoding="utf-8") == "기존 보고서"


def test_live_writes_fresh_result_and_displays_answer(tmp_path, basic_result):
    exercise = tmp_path / "exercise"
    scripts = exercise / "scripts"
    scripts.mkdir(parents=True)
    isolated_script = scripts / "report.py"
    isolated_script.write_bytes(SCRIPT.read_bytes())
    output_dir = exercise / "outputs"
    output_dir.mkdir()
    result_path = output_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    basic_result["comparisons"][0]["generations"] = {
        "baseline": {
            "status": "answered",
            "answer": "예약을 취소할 수 있습니다.",
            "sources": [{"citation_id": 5, "chunk_id": "C05", "source_locator": "시설 안내 2절"}],
        },
    }
    fixture_json = json.dumps(basic_result, ensure_ascii=False)
    (exercise / "main.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "def run_evaluation(live):\n"
        "    assert live is True\n"
        f"    raw = {fixture_json!r}\n"
        "    path = Path(__file__).parent / 'outputs' / 'result.json'\n"
        "    path.write_text(raw, encoding='utf-8')\n"
        "    return json.loads(raw)\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.run(
        [sys.executable, str(isolated_script), "--live"], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=10,
    )

    assert process.returncode == 0, process.stderr
    report_path = output_dir / "report.md"
    assert str(result_path) in process.stdout
    assert str(report_path) in process.stdout
    assert json.loads(result_path.read_text(encoding="utf-8")) == basic_result
    markdown = report_path.read_text(encoding="utf-8")
    assert "## Live 답변과 출처" in markdown
    assert "예약을 취소할 수 있습니다." in markdown
    assert "출처 5 · `C05` · 시설 안내 2절" in markdown
