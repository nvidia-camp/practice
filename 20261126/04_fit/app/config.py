"""① 설정: 04_fit/.env와 앱 정책 값을 한곳에 모은다."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent   # 04_fit 폴더
load_dotenv(BASE_DIR / ".env")

# --- LLM ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("04_fit/.env에 OPENAI_API_KEY를 넣어 주세요.")

# --- 데이터 / 검색 ---
DATA_FILE = BASE_DIR / "data" / "reports.json"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
MIN_SIMILARITY = 0.82   # 이보다 덜 비슷하면 근거로 쓰지 않는다. E5 점수는 0.78~0.92에 몰려 있어 실측으로 정했다
TOP_K = 3

# --- 앱 정책 (사용자 인증이 아니라 "이 서버가 다루는 범위") ---
ALL_CATEGORIES = ["EXPERIMENT", "DAILY_WORK", "RESEARCH_PLAN", "DATA_PROCESSING", "PROMPT", "MEETING"]
ALLOWED_CATEGORIES = set(ALL_CATEGORIES)   # HR처럼 목록 밖 카테고리는 존재 자체를 숨긴다

# --- 안전장치 ---
MAX_TOOL_CALLS = 4    # 질문 하나에 Tool 실행 최대 횟수
MAX_MODEL_CALLS = 4   # 질문 하나에 LLM 호출 최대 횟수

AUDIT_LOG = BASE_DIR / "logs" / "audit.jsonl"
