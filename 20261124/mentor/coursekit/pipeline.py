"""Day27 2-step RAG 파이프라인 엔진: Rewrite -> Retrieve -> Rerank -> Context -> Generation."""
from time import perf_counter  # 각 Stage 실행 시간(ms)을 정밀 측정하기 위한 타이머
from coursekit.documents import context_from_documents, source_index, docs_to_ids  # 문서->Context 변환, 출처 인덱스, ID 추출 유틸
from coursekit.multi_query import unique_queries, search_multiple                  # 다중 질문 중복 제거 및 통합 검색 유틸
from coursekit.llm import offline_answer, validate_generation, NO_ANSWER           # 오프라인 고정 답변, 생성 결과 검증, 기본 무응답 상수


class StageError(RuntimeError):
    """어느 단계(stage)에서 어떤 예외 코드(code)가 발생했는지 감싸는 전용 예외."""

    def __init__(self, stage, code):
        self.stage = stage                      # 실패한 파이프라인 단계명 (예: 'rewrite', 'generation')
        self.code = code                        # 예외 클래스명 또는 에러 코드 (예: 'OpenAIModelNotFoundError')
        super().__init__(f'{stage}:{code}')     # "단계명:에러코드" 형식으로 메시지 통일


def create_rag_pipeline(rewrite_fn, search_fn, rank_fn, generation_fn,
                        *, multiquery_fn=None, offline=False):
    """외부에서 주입받은 함수들(mock 또는 실제 모델)을 LCEL Runnable 체인으로 조립합니다."""
    from langchain_core.runnables import RunnableLambda  # 일반 파이썬 함수를 LangChain Runnable로 감싸는 래퍼

    def run(name, fn):
        """각 Stage 함수를 감싸 실행 시간 측정과 예외 표준화(StageError)를 공통 적용합니다."""
        def wrapped(state):
            start = perf_counter()  # Stage 시작 시각 기록
            try:
                result = fn(state)  # 실제 Stage 함수 실행
            except StageError:
                raise               # 이미 StageError면 그대로 전파
            except Exception as exc:
                # DB 접속정보·API 응답 등 민감한 원문 대신 단계명과 예외 타입만 외부로 노출
                raise StageError(name, type(exc).__name__) from exc
            finally:
                # 성공/실패와 무관하게 이 Stage의 소요 시간(ms)을 state['timing_ms']에 기록
                state.setdefault('timing_ms', {})[name] = round((perf_counter() - start) * 1000, 3)
            # 마지막 Generation 단계처럼 trace가 포함된 최종 응답이라면 timing_ms 스냅샷을 복사해 반영
            if isinstance(result, dict) and 'trace' in result:
                result['trace']['timing_ms'] = dict(state['timing_ms'])
            return result
        return RunnableLambda(wrapped)  # LCEL의 | 연산자로 연결할 수 있도록 RunnableLambda로 반환

    def start(req):
        """0단계: 요청 스키마를 검증하고 파이프라인 전체가 공유할 상태 딕셔너리(state)를 초기화합니다."""
        from coursekit.api import RagRequest
        trusted_role = req.get('trusted_role', 'learner')  # 서버가 신뢰하는 사용자 권한 (기본값 'learner')
        # 사용자 입력 필드만 Pydantic 모델(RagRequest)로 검증 후 dict로 변환 (권한 필드는 별도 분리)
        req = RagRequest.model_validate({k: v for k, v in req.items() if k != 'trusted_role'}).model_dump()
        # 파이프라인 각 Stage가 값을 채워 넣을 초기 상태(state) 딕셔너리 반환
        return dict(
            req,
            trusted_role=trusted_role,                         # 검색 권한 필터에 쓸 신뢰 역할
            search_query=req['query'],                         # 초기 검색 질문은 원 질문으로 설정 (Rewrite에서 갱신됨)
            queries=[],                                        # Multi-query 단계에서 확정될 최종 검색 질문 목록
            candidates=[],                                     # Retrieval 단계에서 가져온 1차 후보 Document 목록
            ranked=[],                                         # Rerank 단계에서 재정렬된 Document 목록
            included=[],                                       # Context 단계에서 최종 선택된 Document 목록
            context='',                                        # LLM에 넘길 최종 Context 문자열
            status='running',                                  # 파이프라인 진행 상태 ('running' / 'clarification_required' 등)
            timing_ms={},                                      # Stage별 소요 시간 기록 딕셔너리
            source_mode='offline_fixture' if offline else 'live',  # 실행 모드 표시
        )

    def rewrite(s):
        """1단계(Rewrite): 대화 이력(history)을 참고해 모호한 질문을 독립된 검색 질문으로 재작성합니다."""
        if not s['use_rewrite']:
            return s                                           # Rewrite 비활성화면 상태 그대로 통과
        plan = rewrite_fn(s['query'], s['history'])            # 원 질문과 이력으로 재작성 실행
        if plan.status == 'clarification_required':            # 질문이 너무 모호해 추가 확인이 필요한 경우
            s['status'] = 'clarification_required'             # 이후 검색/재정렬 단계를 건너뛰도록 상태 변경
            s['reason'] = plan.reason                          # 부족한 정보 사유 기록
        else:
            s['search_query'] = plan.search_query              # 재작성된 검색용 질문 저장 (원 질문 s['query']는 보존)
        return s

    def queries(s):
        """2단계(Multi-query): 필요 시 추가 검색 질문을 만들고 중복을 제거해 최대 3개로 제한합니다."""
        if s['status'] != 'running':
            return s                                           # 이전 단계에서 중단됐으면 건너뜀
        additional = multiquery_fn(s['search_query']) if s['use_multiquery'] else []  # 옵션이 켜진 경우에만 추가 질문 생성
        s['queries'] = unique_queries(s['search_query'], additional, max_queries=3)   # 기본 질문 + 추가 질문 중복 제거(최대 3개)
        return s

    def retrieve(s):
        """3단계(Retrieval): 확정된 질문 목록으로 카테고리·권한 필터를 적용해 후보 문서를 검색합니다."""
        if s['status'] != 'running':
            return s                                           # 중단 상태면 건너뜀
        data = search_multiple(
            s['queries'], search_fn,
            category=s['category'],                            # 문서 카테고리 필터
            role=s['trusted_role'],                            # 접근 권한 필터
            candidate_k=s['candidate_k'],                      # 1차 후보 최대 개수
            max_queries=3,                                     # 검색 호출 횟수 상한
        )
        s['candidates'] = data['documents']                    # 중복 제거된 후보 Document 목록 저장
        s['candidate_ids'] = docs_to_ids(data['documents'])    # 후보 문서들의 chunk_id 목록 기록
        s['query_count'] = data['query_count']                 # 실제 수행한 검색 요청 수 기록
        return s

    def rerank(s):
        """4단계(Rerank): 1차 후보 문서들을 질문과의 관련성 기준으로 다시 순위 매깁니다."""
        if s['status'] != 'running':
            return s                                           # 중단 상태면 건너뜀
        if s['use_rerank'] and s['candidates']:                # Rerank가 켜져 있고 후보 문서가 있을 때만 실행
            s['ranked'] = rank_fn(s['search_query'], s['candidates'], min(len(s['candidates']), 20))
        else:
            s['ranked'] = s['candidates'][:]                   # 아니면 1차 검색 순서를 그대로 복사
        s['ranked_ids'] = docs_to_ids(s['ranked'])             # 재정렬 후 chunk_id 순서 기록
        return s

    def context(s):
        """5단계(Context): 상위 context_k개 문서를 글자 수 제한(650자) 안에서 Context 텍스트로 묶습니다."""
        if s['status'] != 'running':
            return s                                           # 중단 상태면 건너뜀
        s['context'], s['included'] = context_from_documents(
            s['ranked'], context_k=s['context_k'], max_chars=650
        )                                                      # 최종 컨텍스트 문자열과 실제 포함된 문서 목록 반환
        s['included_ids'] = docs_to_ids(s['included'])         # 최종 Context에 들어간 chunk_id 목록 기록
        return s

    def generate(s):
        """6단계(Generation): 최종 Context를 바탕으로 답변을 생성하고 출처 ID를 검증해 응답을 반환합니다."""
        if s['status'] == 'clarification_required':            # Rewrite에서 추가 질문 필요로 판정된 경우
            return {
                'status': 'clarification_required',
                'result': {
                    'status': 'insufficient_context',
                    'answer': NO_ANSWER,
                    'source_ids': [],
                    'missing_information': s['reason'],
                },
                'sources': [],
                'trace': make_trace(s),
            }
        if not s['context']:                                   # 검색된 Context가 비어 있으면 근거 부족 응답 반환
            result = {
                'status': 'insufficient_context',
                'answer': NO_ANSWER,
                'source_ids': [],
                'missing_information': '현재 검색 범위에 답할 Context가 부족합니다.',
            }
        elif offline:                                          # 오프라인 모드: 실제 LLM 대신 고정 답변 생성
            result = offline_answer(s['query'], s['context'], s['included'])
        else:                                                  # 라이브 모드: 원 질문 + 검색 질문 + Context로 LLM 호출 후 구조 검증
            query = s['query'] + '\n[검색용 독립 질문] ' + s['search_query']
            result = validate_generation(
                generation_fn(query, s['context']),
                {d.metadata['citation_id'] for d in s['included']},  # Context에 포함된 출처 ID만 허용
            )
        index = source_index(s['included'])                    # {citation_id: 출처 메타정보} 매핑 테이블 생성
        if not set(result['source_ids']).issubset(index):      # 답변이 Context에 없는 출처 ID를 인용하면 즉시 에러
            raise StageError('generation', 'source_not_in_context')
        sources = [index[cid] for cid in result['source_ids']] # 최종 인용된 출처 메타정보 목록 구성
        return {
            'status': result['status'],
            'result': result,
            'sources': sources,
            'trace': make_trace(s),
        }

    def make_trace(s):
        """실행 추적(Trace) 요약본을 만듭니다. (비밀키·전체 문서 본문은 제외)"""
        return {
            'original_question': s['query'],                   # 사용자가 처음 입력한 원 질문
            'search_query': s['search_query'],                 # Rewrite를 거친 검색용 질문
            'queries': s['queries'],                           # 실제 검색에 사용한 질문 목록
            'candidate_ids': s.get('candidate_ids', []),       # 1차 검색 후보 ID 목록
            'ranked_ids': s.get('ranked_ids', []),             # Rerank 후 정렬된 ID 목록
            'included_ids': s.get('included_ids', []),         # 최종 Context에 포함된 ID 목록
            'query_count': s.get('query_count', 0),            # 검색 호출 횟수
            'timing_ms': s.get('timing_ms', {}),               # 각 Stage별 소요 시간(ms)
            'source_mode': s['source_mode'],                   # 실행 모드 (offline_fixture / live)
        }

    # LCEL의 | 파이프 연산자로 7개 Stage를 순서대로 연결 (여기서는 조립만 하고, .invoke() 호출 시 실제 실행)
    chain = (
        RunnableLambda(start)
        | run('rewrite', rewrite)
        | run('multiquery', queries)
        | run('retrieval', retrieve)
        | run('rerank', rerank)
        | run('context', context)
        | run('generation', generate)
    )
    return chain
