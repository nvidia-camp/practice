"""파이프라인 입력 요청 스키마 구성 (schemas/request.py)"""


def prepare_pipeline_request(case):
    """원 질문과 대화 이력을 보존한 채 파이프라인 입력 딕셔너리를 구성합니다."""
    return {
        'query': case['question'],        # 사용자가 입력한 원 질문 (체인 내부에서 검색용 질문으로 재작성됨)
        'history': case['history'],       # 이전 대화 이력 (모호한 지시어 복원에 사용)
        'category': case['category'],     # 검색 필터링에 사용할 문서 카테고리
        'candidate_k': 5,                 # 1차 검색(Retriever)에서 가져올 후보 문서 수
        'context_k': 2,                   # 재정렬(Rerank) 후 최종 Context에 넣을 상위 문서 수
        'use_rewrite': True,              # 질문 재작성(Rewrite) 단계 활성화
        'use_multiquery': False,          # 다중 질문(Multi-query) 확장 비활성화
        'use_rerank': True,               # 후보 재정렬(Rerank) 단계 활성화
    }
