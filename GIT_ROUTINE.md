# 매일 쓰는 git 루틴

## 0. 구조 이해하기 (이것만 알면 됩니다)

```
 강사님 repo (GitHub)              내 repo (GitHub, public)
 ┌──────────────────┐             ┌──────────────────┐
 │ upstream         │             │ origin           │
 │ 읽기만 함(fetch) │             │ 올리기(push)     │
 └────────┬─────────┘             └────────▲─────────┘
          │ git fetch upstream             │ git push
          ▼                                │
 ┌─────────────────────────────────────────┴────────┐
 │ 내 PC (c:\workspace\practice_assignment)         │
 │                                                  │
 │  upstream/main  ← 강사님 repo의 "복사본 사진"     │
 │  main           ← 내 작업 브랜치                  │
 │                                                  │
 │  git restore --source=upstream/main -- 폴더       │
 │     → 사진에서 원하는 폴더만 꺼내서 내 main에 둠   │
 └──────────────────────────────────────────────────┘
```

| 이름 | 의미 |
|---|---|
| `upstream` | 강사님 repo. **push는 막아 두었습니다**(실수로 올릴 일 없음). |
| `origin` | 내 repo. 내 작업은 여기로만 올립니다. |
| `upstream/main` | `git fetch` 할 때마다 갱신되는 강사님 `main`의 로컬 사본. 직접 수정하지 않습니다. |
| `main` | 내 브랜치. 강사님 커밋 기록과는 별개로, **내가 가져온 폴더와 내 풀이만** 들어 있습니다. |

> 왜 강사님 기록을 통째로 merge 하지 않나요?
> 매일 모든 폴더가 한꺼번에 들어오고, 강사님이 예전 폴더를 수정·삭제하면 내가 푼 파일과 충돌이 납니다.
> 폴더 단위로 골라 오면 **내가 원할 때, 원하는 것만** 가져올 수 있습니다.

### 파일의 3단계

```
작업 폴더(수정)  --git add-->  스테이징(커밋 대기)  --git commit-->  로컬 기록  --git push-->  GitHub
```

모르겠으면 언제든 `git status` 를 치세요. 지금 어느 단계에 뭐가 있는지 알려 줍니다.

---

## 1. 최초 1회 설정

### 1-1. 내 이름/이메일 등록

public repo에는 커밋 이메일이 그대로 공개됩니다. GitHub의 noreply 이메일을 쓰는 것을 추천합니다.
(GitHub → Settings → Emails → "Keep my email addresses private" 체크 → `숫자+아이디@users.noreply.github.com` 확인)

```powershell
git config --global user.name "내GitHub아이디"
git config --global user.email "12345678+아이디@users.noreply.github.com"
```

### 1-2. 첫 커밋 (README, LICENSE, .gitignore, 이 문서)

```powershell
git status                 # 4개 파일이 "Changes to be committed"에 있는지 확인
git add README.md GIT_ROUTINE.md
git commit -m "chore: 실습 기록 저장소 초기 설정"
```

### 1-3. 내 repo 만들고 연결

GitHub에서 새 repo를 만들 때 **README / .gitignore / license 를 모두 체크 해제**한 빈 repo로 만드세요.
(체크하면 GitHub 쪽에 다른 첫 커밋이 생겨서 push가 거절됩니다.)

```powershell
git remote add origin https://github.com/내아이디/내repo이름.git
git remote -v              # upstream(강사님), origin(나) 두 개가 보이면 성공
git push -u origin main    # -u: 앞으로 git push 만 쳐도 origin/main으로 감
```

---

## 2. 매일 루틴

### ① 강사님 repo 새 소식 받기

```powershell
git fetch upstream
```

`fetch` 는 **받아오기만** 하고 내 파일은 전혀 건드리지 않습니다. 안심하고 쓰세요.

### ② 무엇이 새로 올라왔는지 확인

fetch 결과에 이런 줄이 보이면 새 자료가 있는 것입니다. (아무것도 안 나오면 새 자료 없음)

```
   8b0016b..653d53a  main       -> upstream/main
```

```powershell
# 이번 fetch로 새로 들어온 강사님 커밋과 바뀐 파일 (위의 범위를 그대로 붙여넣기)
git log --oneline --stat 8b0016b..653d53a

# 같은 뜻의 단축형: "직전 fetch 시점 .. 지금"
# (처음 한 번은 'only has 1 entries' 오류가 날 수 있음 → 위 방식 사용)
git log --oneline --stat upstream/main@{1}..upstream/main

# 강사님 repo의 최상위 폴더 목록
git ls-tree --name-only upstream/main

# 특정 날짜 폴더 안 들여다보기
git ls-tree --name-only upstream/main:20261112
```

### ③ 오늘 폴더만 골라서 가져오기 (피킹)

```powershell
git restore --source=upstream/main --staged --worktree -- 20261112/Day19_01
```

- `--source=upstream/main` : 강사님 최신 사본에서
- `--staged --worktree` : 작업 폴더에도 놓고, 커밋 대기 상태로도 올림
- `-- 20261112/Day19_01` : 이 경로만

정답(mentor)을 아직 보고 싶지 않다면 `mentee` 폴더만 가져와도 됩니다.

```powershell
git restore --source=upstream/main --staged --worktree -- 20261112/Day19_01/mentee
```

**원본 상태를 먼저 커밋**해 두세요. 그래야 나중에 "내가 뭘 바꿨는지"가 diff로 깔끔하게 보입니다.

```powershell
git status
git commit -m "chore: 20261112 오전 실습 자료 가져오기"
```

### ④ 실습하기

코드를 수정하다가 중간중간 확인:

```powershell
git status                 # 어떤 파일이 바뀌었나
git diff                   # 무엇을 바꿨나 (원본 대비)
```

### ⑤ 내 풀이 커밋

작은 단위(문제 하나, 기능 하나)로 자주 커밋하는 것이 좋습니다.

```powershell
git add 20261112/Day19_01/mentee/01_basic/01_FastAPI
git commit -m "feat: 1112 basic FastAPI CRUD 구현"
```

커밋 메시지 머리말 예시: `feat:` 기능 구현, `fix:` 오류 수정, `docs:` 문서/메모, `chore:` 자료 가져오기·설정

### ⑥ 내 repo에 올리기

```powershell
git push
```

### ⑦ 하루 정리

```powershell
git log --oneline -10      # 오늘 쌓은 커밋 확인
```

---

## 3. 상황별 대처

### 오후에 같은 날짜 폴더에 새 실습(Day19_02)이 올라왔다

새 하위 폴더만 콕 집어 가져오면 됩니다.

```powershell
git fetch upstream
git ls-tree --name-only upstream/main:20261112
git restore --source=upstream/main --staged --worktree -- 20261112/Day19_02
```

> ⚠️ 이미 풀고 있는 폴더(`20261112/Day19_01`)를 통째로 다시 restore 하면 **내 수정이 강사님 원본으로 덮어써집니다.**
> 날짜 폴더 전체가 아니라 **새로 생긴 하위 폴더만** 지정하세요. (커밋해 둔 내용은 `git restore --source=HEAD -- 경로` 로 되돌릴 수 있습니다.)

### 강사님이 이미 가져온 파일을 수정했다

```powershell
# 강사님 최신본과 내 파일의 차이 보기 (내 수정 + 강사님 수정이 섞여 보임)
git diff HEAD upstream/main -- 20261112/Day19_01/README.md
```

- 내가 **안 건드린 파일**이면 그 파일만 다시 가져오기:
  `git restore --source=upstream/main --staged --worktree -- 그파일경로`
- 내가 **수정한 파일**이면 diff를 보고 필요한 부분만 직접 반영하세요.

### 커밋 전 수정을 버리고 싶다

```powershell
git restore 파일경로                # 마지막 커밋 상태로 되돌림 (수정 내용 사라짐, 주의)
git restore --staged 파일경로       # add 만 취소 (수정 내용은 남음)
```

### 방금 한 커밋 메시지를 고치고 싶다 (push 전에만)

```powershell
git commit --amend -m "새 메시지"
```

### 이미 push 한 커밋을 되돌리고 싶다

```powershell
git revert 커밋해시                 # 되돌리는 새 커밋을 만듦 (기록을 지우지 않아 안전)
```

### 실수로 upstream에 push 하려 했다

`fatal: 'DISABLED_push_to_upstream' does not appear to be a git repository` 가 뜨면 정상입니다.
upstream push를 막아 두었기 때문입니다. `git push origin main` 으로 올리세요.

---

## 4. 익숙해지면 해볼 것: 날짜별 브랜치

```powershell
git switch -c day/20261112          # 오늘 작업용 브랜치 만들고 이동
# ... 가져오기, 실습, 커밋 ...
git push -u origin day/20261112     # 브랜치째로 올려 보기 (GitHub에서 PR 연습 가능)
git switch main
git merge day/20261112              # 다 끝나면 main에 합치기
git push
```

브랜치, merge, PR 흐름을 자연스럽게 연습할 수 있습니다.

---

## 치트시트

| 하고 싶은 것 | 명령 |
|---|---|
| 지금 상태 보기 | `git status` |
| 강사님 새 자료 받기 | `git fetch upstream` |
| 새로 올라온 것 보기 | `git log --oneline --stat upstream/main@{1}..upstream/main` |
| 폴더 골라 가져오기 | `git restore --source=upstream/main -SW -- 폴더` |
| 내가 바꾼 것 보기 | `git diff` |
| 커밋 대기에 올리기 | `git add 경로` |
| 커밋 | `git commit -m "메시지"` |
| 내 repo에 올리기 | `git push` |
| 기록 보기 | `git log --oneline --graph -10` |
| 원격 저장소 확인 | `git remote -v` |
