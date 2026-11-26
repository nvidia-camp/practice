"""E5 임베딩 + 코사인 유사도. 수업의 'E5(GPU) + pgvector'를 메모리에서 하는 버전.

- 임베딩: 문장을 384개 숫자(벡터)로 바꾼다. 뜻이 비슷한 문장은 비슷한 벡터가 된다.
- E5 규칙: 문서에는 "passage: ", 질문에는 "query: "를 앞에 붙인다.
- 실제 수업에서는 문서 벡터를 PostgreSQL(pgvector)에 저장해 두고 DB 안에서 유사도를 계산한다.
  여기서는 문서가 12건뿐이라 메모리에 두고 Python으로 계산한다.
"""
import torch
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)


def encode_passages(texts: list[str]) -> torch.Tensor:
    """문서 여러 개를 벡터로. normalize=True라서 벡터끼리 곱하면 곧 코사인 유사도다."""
    return _model.encode([f"passage: {t}" for t in texts], convert_to_tensor=True, normalize_embeddings=True)


def encode_query(text: str) -> torch.Tensor:
    return _model.encode(f"query: {text}", convert_to_tensor=True, normalize_embeddings=True)


def cosine_scores(query_vec: torch.Tensor, passage_vecs: torch.Tensor) -> list[float]:
    """질문 벡터와 각 문서 벡터의 코사인 유사도 (1에 가까울수록 비슷)."""
    return (passage_vecs @ query_vec).tolist()
