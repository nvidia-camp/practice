# 실습 05 · Validation Gate 검증 보고서

## 1. 개요 및 목적
- **대상 단계:** Day 24 파이프라인 5단계 (`05_validation`, `03_advanced`)
- **역할:** 청킹(`04_chunking`)이 완료된 데이터를 메타데이터 매핑(`06_metadata`) 및 벡터 DB 적재(`07_indexing`)로 넘기기 전, 스키마·원문 정합성·중복 여부를 검사해 통과 청크와 오류 청크를 분리하는 품질 관문(Validation Gate)입니다.
- **핵심 원칙:** 실패한 청크를 조용히 삭제하거나 정상 데이터로 덮어쓰지 않고, 실패 사유(`reason`)와 `chunk_id`를 기록해 격리(`Quarantine`)합니다.

---

## 2. 검증 대상 데이터 (입력 범위)
본 실행(`python main.py`)은 실제 원문 파일(`data/raw/`)이나 DB를 연결하지 않고, `tasks.py`의 `regression()` 함수에 정의된 **테스트용 Fixture 데이터(`DOC-TEST`)**를 사용했습니다.

- **기준 텍스트 (`clean`):** 길이 20자 문자열 (`"01234567890123456789"`)
- **정상 청크 (`good`, 2건):**
  - `C1`: `char_start=0`, `char_end=10`, `body="0123456789"`
  - `C2`: `char_start=10`, `char_end=20`, `body="0123456789"`
- **오류 포함 청크 (`broken`, 4건):** 정상 `C1`, `C2`에 아래 의도적 오류 2건을 추가
  - 본문 변조된 `C1` (`body="corrupted"`)
  - ID가 중복된 `C2`

---

## 3. 검증 파이프라인 동작 흐름 (`tasks.py`)
1. **`schema_check`**: 필수 문자열 필드(`chunk_id`, `doc_id`, `source_version`, `unit_id`, `body`) 및 위치 타입(`char_start`, `char_end`의 `int` 여부) 검사
2. **`text_match`**: `0 <= char_start < char_end <= len(clean)` 범위 확인 및 `body == clean[char_start:char_end]` 일치 여부 검사 (Citation 역추적 보장)
3. **`duplicated_ids`**: 의도된 구간 겹침(Overlap)과 구분하여 동일 `chunk_id` 중복 생성 탐지
4. **`coverage_stats`**: `clean_text` 기준 고유 문자 위치 덮임률(`coverage`), 누락 구간(`gaps`), 중복 길이(`repeat_extra`) 계산
5. **`gate`**: 통과 목록(`passed`)과 격리 목록(`quarantine`) 분리 및 사유 기록
6. **`regression`**: 오류 입력 격리 후 정상 입력 재실행 검증

---

## 4. 실행 결과
`python main.py` 실행 결과, `outputs/validation_gate.json`에 아래 결과가 기록되었으며 `expected/offline_reference_validation_gate.json`과 일치합니다.

```json
{
  "failure_reasons": [
    { "chunk_id": "C1", "reason": "text_mismatch" },
    { "chunk_id": "C2", "reason": "duplicate_id" }
  ],
  "fixed_passed": 2
}
```

- **오류 격리:** 변조된 청크(`C1` → `text_mismatch`)와 중복 ID(`C2` → `duplicate_id`) 총 2건이 정상 격리됨
- **회귀 통과:** 정상 청크 2건(`good`)을 재실행했을 때 `passed` 2건, `coverage == 1.0`으로 통과함

---

## 5. 한계점 및 유의사항
1. **Fixture 기반 검증:** 본 결과는 20자 테스트 데이터(`DOC-TEST`) 기준이며, 실제 교육센터 원문(PDF/MD/HTML) 파싱 결과나 E5 임베딩·pgvector DB 연동(`--live-db`)을 수행한 실측 결과가 아닙니다.
2. **Coverage 100%의 의미:** `coverage == 1.0`은 `clean_text`의 모든 문자를 청크가 덮었다는 의미일 뿐, 앞 단계(Parsing)에서 원본 PDF의 모든 내용이 손실 없이 추출되었음을 보장하지 않습니다.
3. **자동 테스트 환경:** 현재 실행 환경에서는 `pytest` 모듈이 설치되어 있지 않아 `python -m pytest -q tests`는 실행하지 못했습니다.
