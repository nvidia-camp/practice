"""Responses API 연결 테스트. 실제 호출 요금이 발생할 수 있습니다."""
from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI

# 이 파일이 역할 폴더 또는 그 아래 실습 폴더에 있어도 .env를 찾습니다.
env_file = next((p / ".env" for p in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents) if (p / ".env").is_file()), None)
if env_file is None:
    raise SystemExit(".env 파일을 찾을 수 없습니다. mentee 또는 mentor 폴더에 만들어 주세요.")

load_dotenv(env_file)
print(f"설정 파일: {env_file}")
print("실제 API를 호출하므로 요금이 발생할 수 있습니다.")

client = OpenAI()
response = client.responses.create(
    model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17"),
    input="우주에서 카페를 연다면 어떤 특별한 메뉴를 팔 수 있을까? 재미있는 메뉴 하나를 짧게 소개해줘.",
    max_output_tokens=300,
)

print("응답:", response.output_text)
print("응답 ID:", response.id)
