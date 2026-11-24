"""파이프라인 실행 결과의 중간 추적(Trace) 검증 (checks.py)"""


def validate_trace(result):
    """원 질문·검색용 질문·후보 ID·최종 Context ID가 올바르게 기록되었는지 검증합니다."""
    t = result['trace']  # 파이프라인 실행 중 기록된 단계별 추적 정보(trace) 추출
    return {
        # 원 질문(original_question)과 재작성된 검색 질문(search_query)이 둘 다 비어 있지 않고 따로 보관됐는지 확인
        'questions_separate': bool(t['original_question'] and t['search_query']),
        # 1차 후보 ID 목록(candidate_ids)과 최종 Context ID 목록(included_ids)이 리스트 형태로 기록됐는지 확인
        'ids_recorded': isinstance(t['candidate_ids'], list) and isinstance(t['included_ids'], list),
        # 최종 Context에 들어간 ID가 반드시 1차 검색 후보 ID의 부분집합인지 확인 (없는 문서가 끼어들지 않았는지 검사)
        'included_from_candidates': set(t['included_ids']).issubset(t['candidate_ids']),
    }
