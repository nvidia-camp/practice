"""파이프라인 실행 시간 및 출처 연결 감사 (services/audit.py)"""


def audit_pipeline(result):
    """Stage별 실행 시간과 출처 연결을 기록합니다. (답변의 사실 여부 판단은 별도 검토)"""
    t = result['trace']  # 파이프라인 실행 중 기록된 단계별 추적 정보(trace) 추출
    return {
        'mode': t['source_mode'],                                  # 실행 모드 기록 (offline_fixture / live 등)
        'stage_times': sorted(t['timing_ms']),                     # 각 Stage의 소요 시간 키 목록을 정렬해 기록
        'output_source_ids': [r['citation_id'] for r in result['sources']],  # 최종 응답에 포함된 출처(citation_id) 목록 추출
        'manual_grounding_review': True,                           # 출처 ID 구조 검증과 답변 내용의 사실 확인(Grounding)은 별도 수동 검토 필요
    }
