"""Day27 실습 06 · LangChain Pipeline 통합 (응용 — main.py 진입점)"""
from pathlib import Path  # 파일/폴더 경로를 OS에 상관없이 다루기 위한 모듈
import sys                # 파이썬 모듈 검색 경로(sys.path)를 수정하기 위한 모듈

HERE = Path(__file__).resolve().parent                                 # 현재 main.py가 있는 폴더 경로
ROLE_ROOT = next(p for p in HERE.parents if (p / 'coursekit').is_dir())  # 상위 폴더 중 coursekit/이 있는 역할 루트(mentor/) 탐색
sys.path.insert(0, str(ROLE_ROOT))                                     # coursekit 모듈을 import할 수 있도록 검색 경로 맨 앞에 추가

import argparse                             # 터미널 실행 옵션(--offline 등)을 파싱하는 모듈
from coursekit.workshops import STAGE_RUNNERS  # 각 실습 번호별 실행기(6번은 LangChain 파이프라인 실행)
from coursekit.io import save_report           # 실행 결과를 outputs/*.json 파일로 저장하는 함수

from steps import prepare_pipeline_request  # 요청 딕셔너리 생성 함수 (steps.py로 분리)
from steps import wire_langchain_pipeline   # LCEL 파이프라인 조립 함수 (steps.py로 분리)
from checks import validate_trace           # 실행 후 Trace 검증 함수 (checks.py로 분리)


def main():
    parser = argparse.ArgumentParser(description='Day27 실습 06 · 06_langchain_pipeline')  # CLI 인자 파서 생성
    parser.add_argument(
        '--offline',
        action='store_true',  # --offline 플래그를 주면 True, 안 주면 False
        help='교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.',
    )
    args = parser.parse_args()  # 터미널에서 전달된 인자 파싱

    # 6번 실습 러너에 [요청 준비, 파이프라인 조립, Trace 검증] 3개 함수를 넘겨 실행
    report = STAGE_RUNNERS[6](
        args.offline,
        [prepare_pipeline_request, wire_langchain_pipeline, validate_trace],
    )

    # --offline이면 fixture.json, 기본 실행이면 live.json으로 결과 저장
    filename = 'day27_06_' + ('fixture' if args.offline else 'live') + '.json'
    path = save_report(HERE, filename, report)

    print('실습 06 결과:', path)  # 저장된 JSON 파일 경로 출력
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')


if __name__ == '__main__':  # 이 파일을 직접 실행했을 때만 main() 호출 (import 시 자동 실행 방지)
    main()
