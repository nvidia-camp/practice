"""두 Tool이 같이 쓰는 입력 검증. 잘못된 값은 데이터 조회 '전에' 막는다."""
from app.config import ALL_CATEGORIES, ALLOWED_CATEGORIES
from app.errors import UserInputError


def check_report_id(value) -> int:
    """양의 정수만 허용. 추측으로 고쳐 주지 않는다 (True, "063", 3.5, -1 모두 거절)."""
    if isinstance(value, bool):   # bool은 int의 하위 타입이라 먼저 막는다 (True == 1)
        raise UserInputError("report_id는 양의 정수여야 합니다.")
    if isinstance(value, str):
        if not value.isdecimal() or value.startswith("0"):
            raise UserInputError("report_id는 앞자리 0 없는 양의 정수여야 합니다.")
        value = int(value)
    if not isinstance(value, int) or value < 1:
        raise UserInputError("report_id는 1 이상의 정수여야 합니다.")
    return value


def check_category(value: str) -> set[str]:
    """'ALL'이면 허용 범위 전체, 아니면 허용된 카테고리 하나."""
    if value == "ALL":
        return set(ALLOWED_CATEGORIES)
    if value not in ALL_CATEGORIES or value not in ALLOWED_CATEGORIES:
        raise UserInputError(f"category는 ALL 또는 {ALL_CATEGORIES} 중 하나여야 합니다.")
    return {value}
