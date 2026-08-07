# boundaries.json 경계에 따라 LeRobot 데이터셋을 "서브태스크 = 1 에피소드"로 분할하는 스크립트
# [2026-08-07 추가] 최초 작성 — annotate_ui 검수 완료본(boundaries_reviewed.json)과 연동
#
# 사용 예:
#   python split_by_subtask.py \
#       --src_repo_id $HF_USER/tower_long \
#       --dst_repo_id $HF_USER/tower_long_split \
#       --boundaries boundaries_reviewed.json \
#       [--src_root ...] [--dst_root ...] [--min_seconds 0.4] [--push_to_hub]
#
# 동작:
#   1) 원본 데이터셋을 LeRobot 라이브러리로 로드
#   2) 에피소드마다 boundaries의 [start, end) 구간대로 프레임을 잘라
#   3) 각 구간을 "task = 서브태스크 문장"인 새 에피소드로 재기록 (영상 재인코딩 포함)
#   → 결과: single_task가 서브태스크 단위로 바뀐 학습용 데이터셋 (π0.5식 분할 학습)
#
# 주의: lerobot(v0.6 기준)이 설치된 학습 PC에서 실행. 영상 재인코딩 때문에 시간이 걸림.

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np


def load_boundaries(path: Path) -> dict[int, list[dict]]:
    """boundaries.json을 읽어 {episode_index: [{start,end,task}...]} 로 반환한다."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, list[dict]] = {}
    for k, v in data.items():
        if not str(k).isdigit():          # "default" 키는 개별 에피소드가 아님
            continue
        out[int(k)] = [
            {"start": float(s["start"]), "end": float(s["end"]), "task": str(s.get("task") or s.get("label"))}
            for s in v
        ]
    if not out:
        sys.exit("[오류] boundaries에 에피소드 항목이 없습니다.")
    return out


def to_recordable_frame(sample: dict, keys: list[str]) -> dict:
    """LeRobotDataset이 돌려준 샘플을 add_frame이 받는 형태로 변환한다.

    - torch.Tensor → numpy
    - 이미지: CHW float[0,1] → HWC uint8
    """
    import torch  # lerobot 의존성에 포함

    frame: dict = {}
    for k in keys:
        v = sample[k]
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if k.startswith("observation.images") and isinstance(v, np.ndarray):
            if v.ndim == 3 and v.shape[0] in (1, 3):          # CHW → HWC
                v = np.transpose(v, (1, 2, 0))
            if v.dtype != np.uint8:                            # float[0,1] → uint8
                v = (np.clip(v, 0.0, 1.0) * 255.0).astype(np.uint8)
        frame[k] = v
    return frame


def add_frame_compat(dst, frame: dict, task: str) -> None:
    """lerobot 버전에 따라 add_frame 시그니처가 달라 호환 처리한다."""
    sig = inspect.signature(dst.add_frame)
    if "task" in sig.parameters:
        dst.add_frame(frame, task=task)
    else:                                  # 구버전: frame dict에 task 포함
        frame = dict(frame)
        frame["task"] = task
        dst.add_frame(frame)


def save_episode_compat(dst, task: str) -> None:
    """save_episode 시그니처 호환 처리 (일부 버전은 task 인자를 받는다)."""
    try:
        dst.save_episode()
    except TypeError:
        dst.save_episode(task=task)


def main() -> None:
    """원본 데이터셋을 boundaries 구간대로 잘라 새 데이터셋으로 재기록한다."""
    p = argparse.ArgumentParser(description="boundaries.json 기준 서브태스크 분할")
    p.add_argument("--src_repo_id", required=True, help="원본 데이터셋 repo_id")
    p.add_argument("--dst_repo_id", required=True, help="생성할 분할 데이터셋 repo_id")
    p.add_argument("--boundaries", type=Path, required=True, help="boundaries(_reviewed).json 경로")
    p.add_argument("--src_root", type=Path, default=None, help="원본 로컬 루트 (기본: HF 캐시)")
    p.add_argument("--dst_root", type=Path, default=None, help="출력 로컬 루트 (기본: HF 캐시)")
    p.add_argument("--min_seconds", type=float, default=0.4, help="이보다 짧은 구간은 건너뜀")
    p.add_argument("--push_to_hub", action="store_true", help="완료 후 Hub 업로드")
    args = p.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        sys.exit("[오류] lerobot이 설치되어 있지 않습니다. 학습 PC(lerobot 환경)에서 실행하세요.")

    boundaries = load_boundaries(args.boundaries)
    print(f"1) 경계 로드: 에피소드 {len(boundaries)}개")

    print("2) 원본 데이터셋 로드")
    src = LeRobotDataset(args.src_repo_id, root=args.src_root)
    fps = src.fps

    # 자동 생성 열은 제외하고, 관측/행동/영상 feature만 새 데이터셋 스키마로 사용
    auto_cols = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    features = {
        k: v for k, v in src.features.items()
        if k not in auto_cols and not k.startswith("next.") and not k.startswith("subtask")
    }
    keys = list(features.keys())

    print("3) 분할 데이터셋 생성")
    dst = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        fps=fps,
        root=args.dst_root,
        features=features,
        robot_type=getattr(src.meta, "robot_type", None),
        use_videos=True,
    )

    # 에피소드/타임스탬프 열만 원시로 뽑아 인덱스 계산 (버전 무관하게 동작)
    ep_col = np.asarray(src.hf_dataset["episode_index"]).reshape(-1)
    ts_col = np.asarray(src.hf_dataset["timestamp"]).reshape(-1)

    print("4) 구간별 재기록 시작 (영상 재인코딩 포함 — 시간이 걸립니다)")
    new_eps, skipped = 0, 0
    for ep in sorted(boundaries):
        idxs = np.where(ep_col == ep)[0]
        if idxs.size == 0:
            print(f"  [주의] episode {ep} 프레임 없음 → 건너뜀")
            continue
        for seg in boundaries[ep]:
            if seg["end"] - seg["start"] < args.min_seconds:
                skipped += 1
                continue
            sel = idxs[(ts_col[idxs] >= seg["start"]) & (ts_col[idxs] < seg["end"])]
            if sel.size == 0:
                skipped += 1
                continue
            for gi in sel:
                sample = src[int(gi)]
                add_frame_compat(dst, to_recordable_frame(sample, keys), task=seg["task"])
            save_episode_compat(dst, task=seg["task"])
            new_eps += 1
            print(f"  ep{ep} [{seg['start']:.2f}–{seg['end']:.2f}s] → 새 에피소드 {new_eps}: \"{seg['task']}\" ({sel.size}프레임)")

    print("\n===== 완료 =====")
    print(f"새 에피소드 {new_eps}개 생성, 짧아서 건너뜀 {skipped}개")
    print(f"학습 시: --dataset.repo_id={args.dst_repo_id}"
          + (f" --dataset.root={args.dst_root}" if args.dst_root else ""))

    if args.push_to_hub:
        print("5) Hub 업로드")
        dst.push_to_hub()


if __name__ == "__main__":
    main()
