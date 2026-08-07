# find_boundaries_multi.py — 그리퍼 + 속도 + 변화점을 결합한 서브태스크 경계 감지 (통합 최종본)
# 2026-08-07 · ZETABANK Physical AI Master Course
#
# ┌─ 왜 이 파일이 필요한가 ─────────────────────────────────────────────┐
# │ 문제: 그리퍼가 안 열고닫는 행위(든 채 이동·정렬·회전)는                │
# │       그리퍼 신호만으론 경계를 못 찾음.                                │
# │                                                                      │
# │ 해결: 로봇이 이미 가진 "다른 신호"들을 결합 (영상 VLM 대신!)          │
# │   ① 그리퍼 전환  → 파지/놓기 경계 (가장 명확)                         │
# │   ② 속도 최소점  → 멈칫하는 순간 = 행위 전환 (그리퍼 안 변해도)       │
# │   ③ 변화점(PELT) → 움직임 패턴이 바뀌는 지점 (통계적)                 │
# │                                                                      │
# │ 학계 표준: 그리퍼로 grasp/release 먼저 나누고, 부족하면 힘/속도로     │
# │ 세분화 (Zero-Velocity-Crossing, Change Point Detection)              │
# └──────────────────────────────────────────────────────────────────────┘
#   근거: 접촉조작 분할(arXiv 2605.17601) 그리퍼+힘+ZVC 결합
#         OKAMI(2410.11792) 속도 변화점으로 서브골 키프레임
#         PELT/ruptures(2509.15607) 궤적 국면 자동 분할
#         CycleVLA(2601.02295) 그리퍼 multi-threshold voting
#
# 사용법 (그리퍼로 부족한 6단계 탑쌓기):
#   python find_boundaries_multi.py \
#       --repo_id $HF_USER/tower_long \
#       --n_subtasks 6 \
#       --tasks "Pick block1" "Place block1" "Pick block2" "Stack block2" "Pick block3" "Stack block3" \
#       --method hybrid \
#       --out boundaries.json
#
#   --method: gripper(그리퍼만) / velocity(속도만) / changepoint(패턴변화) / hybrid(그리퍼+속도, 추천)

import argparse
import json

import numpy as np

# LeRobot 0.6 기준 (구버전: lerobot.common.datasets.lerobot_dataset)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ─────────────────────────────────────────────────────────────────────
# 신호 추출
# ─────────────────────────────────────────────────────────────────────
def get_state_traj(ds, f0, f1):
    """에피소드의 관절 상태 궤적 (프레임, 관절차원)."""
    return np.array([np.asarray(ds[i]["observation.state"], dtype=float)
                     for i in range(f0, f1)])


def gripper_events(grip, close_th=0.4, open_th=0.6, min_hold=8):
    """그리퍼 전환(grasp/release) 프레임. (find_boundaries_gripper와 동일 로직)"""
    lo, hi = grip.min(), grip.max()
    g = (grip - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(grip)
    events, state, cand = [], "open", None
    for t, v in enumerate(g):
        if state == "open" and v < close_th:
            if cand is None or cand[1] != "closed":
                cand = (t, "closed")
            elif t - cand[0] >= min_hold:
                events.append(("grasp", cand[0])); state, cand = "closed", None
        elif state == "closed" and v > open_th:
            if cand is None or cand[1] != "open":
                cand = (t, "open")
            elif t - cand[0] >= min_hold:
                events.append(("release", cand[0])); state, cand = "open", None
        else:
            cand = None
    return [t for _, t in events]


def velocity_minima(traj, n_want, fps, min_gap_s=3):
    """속도 최소점(멈칫) n_want개를 경계 후보로.
    그리퍼가 안 변해도 '행위 사이 감속'을 잡아냄 (Zero-Velocity-Crossing 계열)."""
    from scipy.signal import argrelmin
    vel = np.concatenate([[0], np.linalg.norm(np.diff(traj, axis=0), axis=1)])
    k = max(5, fps // 2)
    vel_s = np.convolve(vel, np.ones(k) / k, mode="same")
    mins = argrelmin(vel_s, order=fps)[0]
    mins = sorted(mins, key=lambda i: vel_s[i])   # 속도 낮은 순
    chosen = []
    for m in mins:
        if all(abs(m - c) > fps * min_gap_s for c in chosen):
            chosen.append(int(m))
        if len(chosen) == n_want:
            break
    return sorted(chosen)


def changepoint_boundaries(traj, n_want, fps):
    """움직임 패턴 변화점(PELT/Dynp)으로 경계 n_want개.
    ruptures 미설치 시 속도 방식으로 자동 폴백."""
    try:
        import ruptures as rpt
        # Binseg: Dynp보다 빠르면서 개수 지정 가능
        algo = rpt.Binseg(model="rbf", min_size=fps * 2).fit(traj)
        bkps = algo.predict(n_bkps=n_want)
        return sorted(bkps[:-1])   # 마지막 끝점 제외
    except Exception as e:
        print(f"  (ruptures 사용 불가 → 속도 방식으로 폴백: {e})")
        return velocity_minima(traj, n_want, fps)


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="그리퍼+속도+변화점 결합 경계 감지")
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--n_subtasks", type=int, required=True, help="나눌 서브태스크 개수")
    ap.add_argument("--tasks", nargs="+", required=True, help="서브태스크 텍스트 (개수 일치)")
    ap.add_argument("--method", default="hybrid",
                    choices=["gripper", "velocity", "changepoint", "hybrid"])
    ap.add_argument("--gripper_dim", type=int, default=-1)
    ap.add_argument("--out", default="boundaries.json")
    args = ap.parse_args()

    if len(args.tasks) != args.n_subtasks:
        raise ValueError(f"tasks 개수({len(args.tasks)})가 n_subtasks({args.n_subtasks})와 다름")

    ds = LeRobotDataset(args.repo_id)
    fps = ds.fps
    n_bnd = args.n_subtasks - 1   # 필요한 경계 수
    print(f"데이터셋: {args.repo_id} | 에피소드 {ds.num_episodes}개 | fps={fps}")
    print(f"방법: {args.method} | 서브태스크 {args.n_subtasks}개 → 경계 {n_bnd}개 필요\n")

    out, skipped = {}, []
    for ep in range(ds.num_episodes):
        f0 = ds.episode_data_index["from"][ep].item()
        f1 = ds.episode_data_index["to"][ep].item()
        traj = get_state_traj(ds, f0, f1)
        grip = traj[:, args.gripper_dim]
        n_frames = len(traj)

        # ── 방법별 경계 후보 계산 ──
        if args.method == "gripper":
            cand = gripper_events(grip)
        elif args.method == "velocity":
            cand = velocity_minima(traj, n_bnd, fps)
        elif args.method == "changepoint":
            cand = changepoint_boundaries(traj, n_bnd, fps)
        else:  # hybrid: 그리퍼 경계를 우선 쓰고, 부족하면 속도로 채움
            g_ev = gripper_events(grip)
            if len(g_ev) >= n_bnd:
                cand = sorted(g_ev)[:n_bnd]   # 그리퍼로 충분
            else:
                # 그리퍼 경계 + 속도 최소점으로 부족분 보충
                v_ev = velocity_minima(traj, n_bnd, fps)
                merged = sorted(set(g_ev) | set(v_ev))
                # 너무 붙은 것 제거하며 n_bnd개로
                cand, last = [], -fps * 3
                for c in merged:
                    if c - last > fps * 3:
                        cand.append(c); last = c
                cand = cand[:n_bnd]

        if len(cand) < n_bnd:
            print(f"⚠️ ep{ep}: 경계 {len(cand)}개만 감지(필요 {n_bnd}) → 건너뜀")
            skipped.append(ep)
            continue

        # ── 경계 → 세그먼트로 변환 ──
        cuts = [0] + sorted(cand[:n_bnd]) + [n_frames - 1]
        segs = []
        for i, task in enumerate(args.tasks):
            segs.append({"start": round(cuts[i] / fps, 2),
                         "end": round(cuts[i + 1] / fps, 2),
                         "task": task})
        out[str(ep)] = segs
        pretty = " / ".join(f"{s['task'][:10]}[{s['start']}~{s['end']}s]" for s in segs)
        print(f"✅ ep{ep}: {pretty}")

    if out:
        out["default"] = next(iter(out.values()))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n💾 저장: {args.out} (성공 {len(out) - 1}개, 수동확인 {len(skipped)}개)")
    print(f"다음: python split_by_subtask.py --src_repo_id {args.repo_id} "
          f"--dst_repo_id <새이름> --boundaries {args.out}")
    if skipped:
        print(f"수동 확인 대상: {skipped} (시각화로 확인 후 boundaries.json 직접 수정)")


if __name__ == "__main__":
    main()
