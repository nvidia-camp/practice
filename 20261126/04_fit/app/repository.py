"""② 데이터: 보고서 검색과 상태 조회.

이 파일은 LLM이나 Agent를 전혀 모른다. "데이터를 어디서 어떻게 가져오는가"만 책임진다.
실제 PostgreSQL로 바꿀 때는 이 파일의 함수 내부만 고치면 된다.
"""
import json

from app import embedding
from app.config import DATA_FILE, MIN_SIMILARITY, TOP_K
from app.errors import ServiceError

try:
    with open(DATA_FILE, encoding="utf-8") as f:
        REPORTS = json.load(f)
except (OSError, ValueError) as e:
    raise ServiceError("보고서 데이터를 읽지 못했습니다.") from e


def is_vector_fresh(report: dict) -> bool:
    """벡터를 믿어도 되는가: ready이고, 지금 버전(revision)으로 임베딩했어야 한다."""
    return report["index_status"] == "ready" and report["indexed_revision"] == report["revision"]


# 검색 대상: 벡터가 최신인 보고서만 (수업 SQL의 index_status='ready' AND indexed_revision=revision 조건)
SEARCHABLE = [r for r in REPORTS if is_vector_fresh(r)]
SEARCHABLE_VECS = embedding.encode_passages([r["title"] + " " + r["content"] for r in SEARCHABLE])


def search_reports(query: str, categories: set[str]) -> list[dict]:
    """의미가 비슷한 보고서를 찾는다. 기준(MIN_SIMILARITY) 미달은 버린다."""
    scores = embedding.cosine_scores(embedding.encode_query(query), SEARCHABLE_VECS)

    candidates = []
    for report, score in zip(SEARCHABLE, scores):
        if report["category"] in categories and score >= MIN_SIMILARITY:
            candidates.append((score, report))
    candidates.sort(key=lambda pair: pair[0], reverse=True)

    results = []
    for score, report in candidates[:TOP_K]:
        results.append({
            "source_id": f"RPT-{report['report_id']}",
            "title": report["title"],
            "content": report["content"],
            "similarity": round(score, 3),
        })
    return results


def find_report(report_id: int) -> dict | None:
    for report in REPORTS:
        if report["report_id"] == report_id:
            return report
    return None
