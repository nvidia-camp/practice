"""Day27 실습 06 · LangChain Pipeline 통합 (기본필수)"""
from pathlib import Path  # 파일/폴더 경로를 OS에 상관없이 다루기 위한 모듈
import sys                # 파이썬 모듈 검색 경로(sys.path)를 수정하기 위한 모듈

HERE = Path(__file__).resolve().parent                                 # 현재 main.py가 있는 폴더 경로
ROLE_ROOT = next(p for p in HERE.parents if (p / 'coursekit').is_dir())  # 상위 폴더 중 coursekit/이 있는 역할 루트(mentor/) 탐색
sys.path.insert(0, str(ROLE_ROOT))                                     # coursekit 모듈을 import할 수 있도록 검색 경로 맨 앞에 추가

import argparse                             # 터미널 실행 옵션(--offline 등)을 파싱하는 모듈
from coursekit.workshops import STAGE_RUNNERS  # 각 실습 번호별 실행기(6번은 LangChain 파이프라인 실행)
from coursekit.io import save_report           # 실행 결과를 outputs/*.json 파일로 저장하는 함수


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


def wire_langchain_pipeline(dependencies, offline):
    """Rewrite -> Retriever -> Rerank -> Context 순의 LangChain Runnable 체인을 조립합니다."""
    from coursekit.pipeline import create_rag_pipeline  # LCEL 기반 RAG 파이프라인 조립 함수 불러오기
    # 주의: 여기서는 Runnable 체인만 조립해서 반환하며, 실제 실행은 호출 측의 chain.invoke(req)에서 시작됨
    return create_rag_pipeline(*dependencies, offline=offline)


def main():
    parser = argparse.ArgumentParser(description='Day27 실습 06 · 06_langchain_pipeline')  # CLI 인자 파서 생성
    parser.add_argument(
        '--offline',
        action='store_true',  # --offline 플래그를 주면 True, 안 주면 False
        help='교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.',
    )
    args = parser.parse_args()  # 터미널에서 전달된 인자 파싱

    # 6번 실습 러너에 [요청 준비 함수, 파이프라인 조립 함수]를 넘겨 전체 흐름 실행
    report = STAGE_RUNNERS[6](args.offline, [prepare_pipeline_request, wire_langchain_pipeline])

    # --offline이면 fixture.json, 기본 실행이면 live.json으로 결과 저장
    filename = 'day27_06_' + ('fixture' if args.offline else 'live') + '.json'
    path = save_report(HERE, filename, report)

    print('실습 06 결과:', path)  # 저장된 JSON 파일 경로 출력
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')


if __name__ == '__main__':  # 이 파일을 직접 실행했을 때만 main() 호출 (import 시 자동 실행 방지)
    main()
