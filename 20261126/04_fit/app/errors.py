"""오류 두 종류를 구분한다 (07 신뢰성).

- UserInputError: 사용자·LLM이 잘못된 값을 넣었다. 이유를 그대로 알려 줘서 다시 시도하게 한다.
- ServiceError:   우리 쪽(데이터, 모델) 문제다. 내부 상세는 숨기고 "일시적 오류"로만 알린다.
"""


class UserInputError(ValueError):
    pass


class ServiceError(RuntimeError):
    pass
