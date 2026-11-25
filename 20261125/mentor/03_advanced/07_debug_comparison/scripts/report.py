"""실습 07을 실제 실행하거나 저장된 결과를 Markdown으로 변환합니다."""

import argparse
import json
from pathlib import Path
import sys


EXERCISE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXERCISE_ROOT / "outputs" / "result.json"

DIAGNOSIS_GUIDE = {
    "system_failure": "실행 상태와 오류 로그를 확인합니다.",
    "no_answer_case": "Gold 문서가 없는 질문입니다. 답변 보류가 적절했는지 확인합니다.",
    "candidate_miss": "검색 후보에 Gold 문서가 없습니다. 검색 단계부터 확인합니다.",
    "ranking_miss": "순위 상위 2개에 Gold 문서가 없습니다. 정렬 결과를 확인합니다.",
    "context_miss": "생성에 전달한 Context에 Gold 문서가 없습니다. Context 구성을 확인합니다.",
    "generation_or_validation": "답변 생성과 출력 검증 단계를 확인합니다.",
    "inspect_answer_grounding": "답변의 주장을 Context 및 출처 원문과 대조합니다.",
}


def format_score(value: float | None) -> str:
    """평가 대상 제외(null)와 실제 0점을 구분해서 표시합니다."""
    if value is None:
        return "—"
    return str(value)


def render_report(result: dict) -> str:
    """평가 코드가 기록한 점수와 진단을 Markdown으로 옮깁니다."""
    if not isinstance(result, dict) or result.get("exercise") != 7:
        raise ValueError("실습 07의 결과 JSON이 필요합니다.")

    comparisons = result["comparisons"]
    if not isinstance(comparisons, list):
        raise ValueError("comparisons는 질문별 비교 결과의 목록이어야 합니다.")

    level = result["level"]
    version = result["dataset_version"]
    mode = result["mode"]
    comparison_kind = result["comparison_kind"]
    lines = [
        "# Day28 실습 07 평가 보고서",
        "",
        f"- 난이도 `{level}`",
        f"- 데이터셋 버전 `{version}`",
        f"- 실행 모드 `{mode}`",
        f"- 비교 유형 `{comparison_kind}`",
    ]
    if comparison_kind == "offline_version_comparison_not_online_AB":
        lines.append("- 동일 평가 질문을 사용한 오프라인 버전 비교입니다.")
    if mode == "offline_fixture":
        lines.append("- 교육용 고정 데이터에서 얻은 결과입니다.")

    lines.extend(["", "## 질문별 Recall 비교", ""])
    if not comparisons:
        lines.append("기록된 비교 결과가 없습니다.")
        return "\n".join(lines) + "\n"

    lines.extend([
        "변화량은 Rerank Recall에서 Baseline Recall을 뺀 값입니다. "
        "`—`는 JSON의 `null`을 나타냅니다. 진단은 Baseline 실행 기준입니다.",
        "",
        "| 질문 ID | Baseline Recall | Rerank Recall | 변화량 | Baseline 진단 |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    variant_rows = []
    regression_rows = []
    generation_details = []
    for row in comparisons:
        question_id = row["question_id"]
        baseline = format_score(row["baseline_recall"])
        rerank = format_score(row["rerank_recall"])
        delta = format_score(row["delta_recall"])
        diagnosis = row["diagnosis"]
        guide = DIAGNOSIS_GUIDE.get(diagnosis, "진단 코드의 정의를 확인합니다.")
        lines.append(
            f"| {question_id} | {baseline} | {rerank} | {delta} | `{diagnosis}` · {guide} |"
        )

        # 고급 결과에는 실제로 실행한 변형의 지표와 회귀 판정이 들어 있습니다.
        for variant, metrics in row.get("matrix", {}).items():
            k = metrics["k"]
            recall = format_score(metrics["recall_at_k"])
            rr = format_score(metrics["rr_at_k"])
            ndcg = format_score(metrics["ndcg_at_k"])
            variant_rows.append(f"| {question_id} | `{variant}` | {k} | {recall} | {rr} | {ndcg} |")

        if "regression" in row:
            regression = row["regression"]
            decision = "보류 조건 없음"
            if regression["hold_release"]:
                decision = "보류"
            error_count = regression["error_count"]
            reason = regression["reason"]
            regression_rows.append(f"| {question_id} | {decision} | {error_count} | {reason} |")

        for variant, generation in row.get("generations", {}).items():
            status = generation["status"]
            generation_details.extend([f"### {question_id} · {variant}", "", f"상태 `{status}`", ""])
            for answer_line in generation["answer"].splitlines():
                generation_details.append(f"    {answer_line}")
            generation_details.append("")
            sources = generation["sources"]
            if sources:
                for source in sources:
                    citation_id = source["citation_id"]
                    chunk_id = source["chunk_id"]
                    locator = source["source_locator"]
                    generation_details.append(f"- 출처 {citation_id} · `{chunk_id}` · {locator}")
            else:
                generation_details.append("반환된 출처가 없습니다.")
            generation_details.append("")

    if variant_rows:
        lines.extend([
            "", "## 변형별 검색 지표", "",
            "| 질문 ID | 변형 | k | Recall@k | RR@k | nDCG@k |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ])
        lines.extend(variant_rows)

    if regression_rows:
        lines.extend([
            "", "## 회귀 검사", "",
            "| 질문 ID | 배포 보류 판정 | 기록된 오류 수 | 사유 |",
            "| --- | --- | ---: | --- |",
        ])
        lines.extend(regression_rows)

    if generation_details:
        lines.extend(["", "## Live 답변과 출처", ""])
        lines.extend(generation_details)

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="실습 07의 평가 보고서를 생성합니다.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="결과 JSON (기본값 outputs/result.json)")
    source.add_argument("--live", action="store_true", help="CUDA·PostgreSQL·LLM으로 실습 07 실행 (API 비용 발생 가능)")
    parser.add_argument("--output", type=Path, help="Markdown 경로 (기본값은 입력 파일 옆 report.md)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_name("report.md")
    try:
        if input_path.resolve() == output_path.resolve():
            raise ValueError("입력과 출력에 서로 다른 파일을 지정하세요.")

        if args.live:
            sys.path.insert(0, str(EXERCISE_ROOT))
            from main import run_evaluation

            print("Live 평가를 시작합니다. PostgreSQL에서 검색하고 LLM 답변을 생성합니다.", flush=True)
            try:
                result = run_evaluation(live=True)
            except Exception as exc:
                stage = getattr(exc, "stage", "evaluation")
                code = getattr(exc, "code", type(exc).__name__)
                failure_path = input_path.with_name("failure.json")
                message = f"Live 평가 실패 ({stage}, {code})\n실패 기록 {failure_path}\n"
                parser.exit(1, message)
        else:
            raw_json = input_path.read_text(encoding="utf-8-sig")
            result = json.loads(raw_json)
        markdown = render_report(result)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"보고서 생성 실패 ({exc})\n")

    print(f"평가 결과 {input_path.resolve()}")
    print(f"보고서 {output_path.resolve()}")


if __name__ == "__main__":
    main()
