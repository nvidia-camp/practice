"""LCEL 기반 LangChain 파이프라인 그래프 조립 (chains/graph.py)"""


def wire_langchain_pipeline(dependencies, offline):
    """Rewrite -> Retriever -> Rerank -> Context 순의 LangChain Runnable 체인을 조립합니다."""
    from coursekit.pipeline import create_rag_pipeline  # LCEL 기반 RAG 파이프라인 조립 함수 불러오기
    # 주의: 여기서는 Runnable 체인만 조립해서 반환하며, 실제 실행은 호출 측의 chain.invoke(req)에서 시작됨
    return create_rag_pipeline(*dependencies, offline=offline)
