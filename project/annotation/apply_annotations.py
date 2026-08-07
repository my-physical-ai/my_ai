# annotate_ui.html에서 내보낸 구간 JSON을 LeRobot 데이터셋 parquet에 반영하는 스크립트
# [2026-08-01 추가] 최초 작성 — subtasks.parquet 생성 + data parquet에 subtask_index 열 추가
#
# 사용 예:
#   python apply_annotations.py \
#       --dataset_root ~/.cache/huggingface/lerobot/data/rollout_dagger_pink_task_v3_r3 \
#       --annotations_dir ./annotations \
#       --output_dir ./exports/rollout_dagger_pink_task_v3_r3_annotated \
#       [--copy_videos]
#
# 동작 (공식 lerobot-annotate의 Export와 동일한 결과 구조):
#   1) annotations_dir의 annotations_ep*.json 을 모두 읽음
#   2) 라벨 목록 → meta/subtasks.parquet (subtask_index ↔ subtask 조견표)
#   3) data/**.parquet 복사본에 subtask_index 열 추가
#      (episode_index 일치 + timestamp ∈ [start, end) → 해당 번호, 그 외 -1)
#   4) meta/ 는 그대로 복사, videos/ 는 심링크(기본) 또는 --copy_videos 시 복사

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def load_annotations(annotations_dir: Path) -> dict[int, list[dict]]:
    """주석 JSON들을 읽어 {episode_index: [segment, ...]} 로 반환한다.

    Args:
        annotations_dir: annotations_ep*.json 파일들이 있는 폴더

    Returns:
        에피소드 번호 → 구간 리스트(dict: start, end, label) 매핑
    """
    result: dict[int, list[dict]] = {}
    files = sorted(annotations_dir.glob("*.json"))
    if not files:
        sys.exit(f"[오류] {annotations_dir} 에 JSON 파일이 없습니다.")
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        ep = int(data["episode_index"])
        segs = [
            {"start": float(s["start"]), "end": float(s["end"]), "label": str(s["label"]).strip()}
            for s in data.get("segments", [])
        ]
        result.setdefault(ep, []).extend(segs)
        print(f"  읽음: {f.name}  (episode {ep}, 구간 {len(segs)}개)")
    return result


# [2026-08-07 추가] annotate_ui 검수 완료본(boundaries_reviewed.json)도 직접 입력 지원
def load_boundaries_file(path: Path) -> dict[int, list[dict]]:
    """boundaries(_reviewed).json을 annotations 형식으로 변환해 반환한다 (task→label)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, list[dict]] = {}
    for k, v in data.items():
        if not str(k).isdigit():          # "default" 키 제외
            continue
        result[int(k)] = [
            {"start": float(s["start"]), "end": float(s["end"]),
             "label": str(s.get("task") or s.get("label")).strip()}
            for s in v
        ]
        print(f"  읽음: episode {k} (구간 {len(result[int(k)])}개)")
    if not result:
        sys.exit("[오류] boundaries 파일에 에피소드 항목이 없습니다.")
    return result


def build_subtask_table(annotations: dict[int, list[dict]]) -> tuple[pd.DataFrame, dict[str, int]]:
    """전체 라벨에서 고유 서브태스크 조견표를 만든다 (등장 순서 기준 번호 부여).

    Returns:
        (subtasks DataFrame[subtask_index, subtask], 라벨→번호 dict)
    """
    label_to_idx: dict[str, int] = {}
    for ep in sorted(annotations):
        for seg in annotations[ep]:
            if seg["label"] not in label_to_idx:
                label_to_idx[seg["label"]] = len(label_to_idx)
    df = pd.DataFrame(
        {"subtask_index": list(label_to_idx.values()), "subtask": list(label_to_idx.keys())}
    )
    return df, label_to_idx


def assign_subtask_index(
    df: pd.DataFrame, annotations: dict[int, list[dict]], label_to_idx: dict[str, int]
) -> pd.Series:
    """프레임 표의 각 행에 subtask_index를 배정한다 (해당 없으면 -1).

    LeRobot의 timestamp 열은 에피소드 기준(각 에피소드가 0초부터 시작)이라
    UI에서 저장한 에피소드 기준 초와 그대로 비교하면 된다.
    """
    idx = pd.Series(-1, index=df.index, dtype="int64")
    if "episode_index" not in df.columns or "timestamp" not in df.columns:
        sys.exit("[오류] parquet에 episode_index/timestamp 열이 없습니다. LeRobot 데이터셋이 맞는지 확인하세요.")
    for ep, segs in annotations.items():
        ep_mask = df["episode_index"] == ep
        if not ep_mask.any():
            print(f"  [주의] episode {ep} 가 이 파일에는 없음 (다른 chunk에 있을 수 있음)")
            continue
        for seg in segs:
            m = ep_mask & (df["timestamp"] >= seg["start"]) & (df["timestamp"] < seg["end"])
            idx.loc[m] = label_to_idx[seg["label"]]
    return idx


def export_dataset(
    dataset_root: Path, output_dir: Path, annotations: dict[int, list[dict]], copy_videos: bool
) -> None:
    """주석이 반영된 새 데이터셋 폴더를 생성한다 (원본은 변경하지 않음)."""
    if output_dir.exists():
        sys.exit(f"[오류] 출력 폴더가 이미 있습니다: {output_dir}  (덮어쓰기 방지)")
    subtasks_df, label_to_idx = build_subtask_table(annotations)

    # --- data/: parquet 복사본에 subtask_index 열 추가 ---
    data_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not data_files:
        sys.exit(f"[오류] {dataset_root}/data 아래에 parquet이 없습니다.")
    total_tagged = 0
    for src in data_files:
        rel = src.relative_to(dataset_root)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(src)
        df["subtask_index"] = assign_subtask_index(df, annotations, label_to_idx)
        tagged = int((df["subtask_index"] >= 0).sum())
        total_tagged += tagged
        df.to_parquet(dst, index=False)
        print(f"  기록: {rel}  (전체 {len(df)}행 중 라벨된 프레임 {tagged}행)")

    # --- meta/: 그대로 복사 + subtasks.parquet 신설 ---
    shutil.copytree(dataset_root / "meta", output_dir / "meta")
    subtasks_df.to_parquet(output_dir / "meta" / "subtasks.parquet", index=False)
    (output_dir / "meta" / "lerobot_annotations.json").write_text(
        json.dumps({str(k): v for k, v in annotations.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- videos/: 심링크(기본) 또는 복사 ---
    videos_src = dataset_root / "videos"
    if videos_src.exists():
        if copy_videos:
            shutil.copytree(videos_src, output_dir / "videos")
            print("  영상: 실제 복사 완료 (다른 PC로 이동 가능)")
        else:
            (output_dir / "videos").symlink_to(videos_src.resolve())
            print("  영상: 심링크 연결 (원본 삭제/이동 시 깨짐 — 공유 시 --copy_videos 사용)")

    # --- 요약 ---
    print("\n===== 완료 =====")
    print(f"출력: {output_dir}")
    print(f"서브태스크 {len(subtasks_df)}종:")
    for _, row in subtasks_df.iterrows():
        print(f"  [{row['subtask_index']}] {row['subtask']}")
    print(f"라벨된 프레임 합계: {total_tagged}행")
    print("학습 시: lerobot-train --dataset.root=" + str(output_dir))


def main() -> None:
    """명령행 인자를 받아 전체 파이프라인을 실행한다."""
    p = argparse.ArgumentParser(description="annotate_ui JSON → LeRobot parquet 반영")
    p.add_argument("--dataset_root", type=Path, required=True, help="원본 데이터셋 루트 (data/, meta/, videos/ 포함)")
    # [2026-08-07 수정] 입력을 두 형식 중 하나로: annotations 폴더 또는 boundaries 파일
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--annotations_dir", type=Path, help="annotations_ep*.json 폴더 (수동 모드)")
    g.add_argument("--boundaries", type=Path, help="boundaries(_reviewed).json 파일 (검수 모드)")
    p.add_argument("--output_dir", type=Path, required=True, help="주석 반영된 새 데이터셋을 만들 위치")
    p.add_argument("--copy_videos", action="store_true", help="영상을 심링크 대신 실제 복사 (팀 공유/이동 시)")
    args = p.parse_args()

    print("1) 주석 읽기")
    if args.annotations_dir:
        annotations = load_annotations(args.annotations_dir)
    else:
        annotations = load_boundaries_file(args.boundaries)
    print("2) 데이터셋 내보내기")
    export_dataset(args.dataset_root, args.output_dir, annotations, args.copy_videos)


if __name__ == "__main__":
    main()
