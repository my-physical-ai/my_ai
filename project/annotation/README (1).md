# LeRobot Annotate Mini

수업용 경량 서브태스크 주석 도구. 공식 `lerobot-annotate`와 같은 결과물(`meta/subtasks.parquet` + `subtask_index` 열)을 만들지만, **서버 설치가 필요 없습니다.**

## 개요

| | 공식 lerobot-annotate | 이 도구 (Mini) |
|---|---|---|
| 실행 | FastAPI 서버 + 브라우저 | **HTML 더블클릭** + 스크립트 1개 |
| 설치 | venv + pip + uvicorn | pandas/pyarrow만 (학습 PC엔 이미 있음) |
| 대상 | 범용 | 수업 실습 (에피소드 10~50개) |

## 구조

```
lerobot-annotate-mini/
├── annotate_ui.html        # 자막 편집기 (브라우저에서 바로 열기)
├── apply_annotations.py    # JSON → parquet 반영 스크립트
├── requirements.txt
└── README.md
```

## 사용법

### 🅰️ 자동 초안 + 검수 모드 (권장 · find_boundaries_multi.py 연동)

**"로봇 신호가 초안, 사람이 검수"** — VLM 없이 그리퍼 release·속도·변화점으로 초안을 만들고, UI에서 영상 보며 확인·수정합니다.

```bash
# 1) 자동 경계 감지 → boundaries.json 초안 생성
python find_boundaries_multi.py \
    --repo_id $HF_USER/tower_long --n_subtasks 6 \
    --tasks "Pick block1" "Place block1" "Pick block2" \
            "Stack block2" "Pick block3" "Stack block3" \
    --method hybrid --out boundaries.json --report
```

2) `annotate_ui.html` 열기 → 🎞️ 영상 선택 → **📂 JSON 불러오기에 boundaries.json 지정**
   - 자동으로 **검수 모드** 진입 (헤더에 🔍 칩 표시)
   - `episode_index`를 바꾸면 해당 에피소드 초안이 로드됨 (수정사항은 자동 보존)
   - 영상 보며 경계가 어긋난 곳만 수정 (스크립트가 ⚠️ 표시한 에피소드 위주로)
3) **⬇ boundaries.json (검수 완료본)** 내보내기 → `boundaries_reviewed.json`

```bash
# 4) 검수 완료본으로 서브태스크 분할 진행
python split_by_subtask.py --src_repo_id $HF_USER/tower_long \
    --dst_repo_id $HF_USER/tower_long_split --boundaries boundaries_reviewed.json
```

### 🅱️ 수동 모드 (초안 없이 처음부터)

1. `annotate_ui.html` 열기 → 영상 선택 → `I`/`O`/라벨로 직접 구간 추가
2. **⬇ annotations JSON** 내보내기 (에피소드별)
3. JSON들을 한 폴더에 모아 parquet 반영:

```bash
python apply_annotations.py \
    --dataset_root ~/.cache/huggingface/lerobot/data/rollout_dagger_pink_task_v3_r3 \
    --annotations_dir ./annotations \
    --output_dir ./exports/..._annotated
```

단축키: `Space` 재생/정지 · `I` 시작 · `O` 끝 · `Enter` 추가 · `←/→` 0.1초 이동

## 주의사항

- 라벨 어휘는 **추론 조향에 쓸 표현과 통일** (예: "slowly", "wide grip") — 팀별 어휘 통일표를 먼저 만들 것
- `[start, end)` 규칙: 끝 시각 프레임은 다음 구간에 속함
- 출력 폴더가 이미 있으면 중단됨 (덮어쓰기 방지) — 새 이름을 쓰거나 기존 폴더 삭제 후 실행
