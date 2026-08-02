#!/usr/bin/env python3
# SO-101 ACT 학습 상태 종합 평가 스크립트 v2 (글로벌 표준 반영판)
# 작성일: 2026-08-02
# 작성자: 빅맨 / ZETA Satellite Robotics
# 변경: v1 → v2 글로벌 사이트 심층 분석 반영 (7건 추가/수정)
#
# ═══════════════════════════════════════════════════════════════════════
# 🆕 v2 변경 사항 (글로벌 분석 근거)
# ═══════════════════════════════════════════════════════════════════════
#
# [추가 1] 평가 11: Validation Action MSE — ground-truth 대조 ⭐⭐⭐
#   근거: TOTO 벤치마크(arXiv 2306.00942) "검증 시연에 대한 action MSE,
#         실세계 보상은 검증 오차 감소와 함께 증가"
#         Policy Comparison(2025): val MSE 1.61E-3 vs 1.35E-3
#         → 실제 성공률 56% vs 92% 차이 검증
#
# [추가 2] 평가 11-B: CI-MSE (Critical Interval MSE) ⭐⭐⭐
#   근거: arXiv 2606.29898 (2026.06) "raw MSE는 성공률과 상관 -0.61,
#         태스크 결정 구간만 계산하는 CI-MSE는 -0.87"
#         → grasp 순간 ± 15프레임 구간의 MSE 별도 계산
#
# [수정 3] 평가 10 → 평가 12: 체크포인트 Top-3 최적 후보 추천 ⭐⭐⭐
#   v1 오류: "최종점 분산 최대" 선택 → 불안정 모델도 분산이 큼
#   글로벌 표준: "Do NOT use the final checkpoint. Select the
#   checkpoint with the lowest validation loss" (SVRC 2026)
#   robomimic(arXiv 2108.03298): 단일 지표는 최선 대비 10~100% 저하
#   → v2: Val MSE(주) + CI-MSE(부) + 분산 적정성(보조) 복합 점수 랭킹
#
# [수정 4] temporal ensembling 판별 로직
#   v1 오류: n_action_steps=1 을 무조건 버그 처리
#   공식 문서: temporal_ensemble_coeff 설정 시 n_action_steps=1 필수
#   → v2: coeff 설정 여부로 정상/버그 구분
#
# [추가 5] 평가 13: 추론 지연(latency) 측정
#   근거: VOTE(arXiv 2507.05116) "100회 쿼리로 latency/throughput 측정"
#         30fps 실시간 제어 → select_action 33ms 이내 필수
#
# [추가 6] 평가 3: 정규화 통계 검증
#   근거: SVRC "학습-배포 정규화 불일치는 1~3cm 오프타겟의 대표 원인"
#
# [추가 7] 평가 6에 Jerk(3차 미분) 지표
#   근거: arXiv 2603.11383 "jerk가 궤적 부드러움 표준 지표"
#
# ═══════════════════════════════════════════════════════════════════════
# 📊 평가 구성 (13개)
# ═══════════════════════════════════════════════════════════════════════
#
# 평가 1:  체크포인트 메타 정보
# 평가 2:  ACT 핵심 설정 검증 (temporal ensemble 판별 포함) [v2 수정]
# 평가 3:  정규화 통계 검증 [v2 신규]
# 평가 4:  Vision 인코더 반응
# 평가 5:  State 인코더 반응 (팔 자세 10가지)
# 평가 6:  Chunk 품질 (boundary + jerk) [v2 확장]
# 평가 7:  Vision 다양성 → Mode averaging 진단
# 평가 8:  Observation 일관성 (재현성)
# 평가 9:  State 다양성 → Mode collapse 진단
# 평가 10: Action range 안전성
# 평가 11: ⭐⭐⭐ Validation MSE + CI-MSE (ground-truth 대조) [v2 신규]
# 평가 12: ⭐⭐⭐ 체크포인트 Top-3 최적 후보 추천 [v2 전면 개편]
# 평가 13: 추론 지연 측정 [v2 신규]
#
# 사용법:
#   conda activate lerobot2
#   python eval_train_state_act_so101_v2.py
# ═══════════════════════════════════════════════════════════════════════

import os
import sys
import json
import glob
import time
import warnings
from datetime import datetime

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
warnings.filterwarnings("ignore")

LOG_DIR = os.path.expanduser("~/lerobot_outputs/eval_logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(
    LOG_DIR,
    f"act_eval_so101_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


class Tee:
    """터미널 출력 + 파일 동시 저장"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Tee(LOG_FILE)

import torch
import numpy as np

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


# ═══════════════════════════════════════════════════════════════════════
# ⚙️  설정 (여기만 수정)
# ═══════════════════════════════════════════════════════════════════════
CKPT_DIR    = "/home/zeta/lerobot_outputs/so101_doll_box_act_v3"
CKPT        = os.path.join(CKPT_DIR, "checkpoints/last/pretrained_model")
DATASET_DIR = "/home/zeta/lerobot_datasets/doll_box_trainmix_v1"

CAM_TOP   = "observation.images.top"
CAM_WRIST = "observation.images.wrist"

JOINT_NAMES = ["pan", "lift", "elbow", "wrist_f", "wrist_r", "grip"]
ACTION_SAFE_RANGE = [
    (-110.0, 110.0), (-195.0, 10.0), (-10.0, 175.0),
    (-110.0, 110.0), (-165.0, 165.0), (-35.0, 40.0),
]

VISION_THRESHOLD = 0.1
STATE_THRESHOLD = 0.5
EXPECTED_CHUNK_SIZE = 100
EXPECTED_N_ACTION_STEPS = 50
FPS = 30                          # SO-101 표준 수집 FPS
CI_HALF_WINDOW = 15               # CI-MSE: grasp ± 15프레임 (약 1초)
VAL_EPISODE_RATIO = 0.2           # Validation: 마지막 20% 에피소드
VAL_SAMPLES_PER_EP = 8            # 에피소드당 검증 샘플 수
LATENCY_TRIALS = 100              # 추론 지연 측정 횟수 (VOTE 논문 표준)

print("=" * 70)
print("  🦾 SO-101 ACT 학습 상태 종합 평가 v2 (글로벌 표준 반영판)")
print(f"  체크포인트: {CKPT_DIR.split('/')[-1]}")
print(f"  데이터셋:   {DATASET_DIR.split('/')[-1]}")
print(f"  v2 핵심:    Validation MSE + CI-MSE + Top-3 후보 추천")
print(f"  로그 저장:  {LOG_FILE}")
print("=" * 70)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸리티
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_camera_keys(ckpt_path: str) -> list:
    """체크포인트 config.json에서 카메라 키 추출 (rename 불일치 사전 방지)."""
    config_file = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_file):
        print(f"  ⚠️  config.json 없음 → 기본 카메라 키 사용")
        return [CAM_TOP, CAM_WRIST]
    with open(config_file) as f:
        cfg = json.load(f)
    input_features = cfg.get("input_features", {})
    cam_keys = [k for k in input_features.keys() if "images" in k]
    if not cam_keys:
        cam_keys = [CAM_TOP, CAM_WRIST]
    print(f"  ℹ️  감지된 카메라 키: {cam_keys}")
    return cam_keys


def detect_state_dim(ckpt_path: str) -> int:
    """체크포인트에서 state 차원 추출 (단팔=6, 양팔=12)."""
    config_file = os.path.join(ckpt_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file) as f:
            cfg = json.load(f)
        state_feat = cfg.get("input_features", {}).get("observation.state", {})
        shape = state_feat.get("shape", [6])
        return int(shape[0]) if shape else 6
    return 6


def rand_img(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(1, 3, 480, 640).cuda()


def make_meaningful_test_imgs() -> list:
    """실제 데이터셋 프레임 5장 + 극단 케이스 5장."""
    imgs = []
    real_imgs = sorted(glob.glob(f"{DATASET_DIR}/**/*.jpg", recursive=True))[:5]
    if not real_imgs:
        real_imgs = sorted(glob.glob(f"{DATASET_DIR}/**/*.png", recursive=True))[:5]
    if real_imgs:
        from torchvision import transforms
        from PIL import Image
        to_tensor = transforms.Compose([
            transforms.Resize((480, 640)), transforms.ToTensor(),
        ])
        for path in real_imgs:
            try:
                img = Image.open(path).convert("RGB")
                imgs.append(to_tensor(img).unsqueeze(0).cuda())
                if len(imgs) >= 5:
                    break
            except Exception:
                pass
        if imgs:
            print(f"  ℹ️  실제 데이터셋 이미지 {len(imgs)}장 로드 완료")
    imgs.append(torch.ones(1, 3, 480, 640).cuda())
    imgs.append(torch.zeros(1, 3, 480, 640).cuda())
    r = torch.zeros(1, 3, 480, 640).cuda(); r[0, 0] = 1.0; imgs.append(r)
    g = torch.zeros(1, 3, 480, 640).cuda(); g[0, 1] = 1.0; imgs.append(g)
    b = torch.zeros(1, 3, 480, 640).cuda(); b[0, 2] = 1.0; imgs.append(b)
    return imgs[:10]


def reset_policy_state(p):
    """ACT 내부 action queue 초기화 (평가 간 간섭 방지)."""
    if hasattr(p, "reset") and callable(getattr(p, "reset", None)):
        try:
            p.reset()
            return
        except Exception:
            pass
    if hasattr(p, "_action_queue"):
        p._action_queue.clear()


def get_action(policy, preprocess, obs_imgs: dict, state=None):
    if state is None:
        state = torch.zeros(1, STATE_DIM).cuda()
    reset_policy_state(policy)
    batch = {**obs_imgs, "observation.state": state}
    batch = preprocess(batch)
    with torch.no_grad():
        return policy.select_action(batch).cpu().numpy()[0]


def get_chunk(policy, preprocess, obs_imgs: dict, state=None):
    if state is None:
        state = torch.zeros(1, STATE_DIM).cuda()
    reset_policy_state(policy)
    batch = {**obs_imgs, "observation.state": state}
    batch = preprocess(batch)
    with torch.no_grad():
        return policy.predict_action_chunk(batch)[0].cpu().numpy()


def grade_vision(responding: int, total: int) -> tuple:
    if responding == 0:
        return "F", "❌ Vision 완전 미반응 (인코더 학습 실패)"
    elif responding <= int(total * 0.3):
        return "D", "⚠️  Vision 극초기 반응 (불충분)"
    elif responding <= int(total * 0.6):
        return "C", "⚠️  Vision 부분 반응"
    elif responding <= int(total * 0.8):
        return "B", "✅ Vision 상당 부분 반응"
    else:
        return "A", "✅ Vision 충분히 반응"


def grade_mode_avg(final_std: float) -> tuple:
    if final_std < 0.5:
        return "F", "❌ Mode averaging 심각 (모든 이미지에 같은 궤적)"
    elif final_std < 1.0:
        return "D", "⚠️  Mode averaging 강함"
    elif final_std < 2.0:
        return "C", "⚠️  모드 분리 약함"
    elif final_std < 5.0:
        return "B", "✅ 모드 분리 양호"
    else:
        return "A", "✅ 모드 분리 뚜렷"


def grade_mode_collapse(state_std: float, chunk_range: float) -> tuple:
    if state_std < 0.5 and chunk_range < 3.0:
        return "F", "❌ Mode collapse 심각"
    elif state_std < 1.0 and chunk_range < 5.0:
        return "D", "⚠️  Mode collapse 일부"
    elif state_std < 2.0:
        return "C", "⚠️  공간 적응력 약함"
    elif state_std < 5.0:
        return "B", "✅ 공간 적응력 양호"
    else:
        return "A", "✅ 공간 적응력 뛰어남"


def grade_val_mse(mse: float) -> tuple:
    """Validation MSE 등급 (정규화 공간 기준).

    [근거] Policy Comparison(2025): 1.35E-3(성공 92%) vs 1.61E-3(56%)
           정규화 action 공간에서 1E-3 대가 배포 가능 수준.
    """
    if mse < 1.5e-3:
        return "A", "✅ 배포 수준 (참고: 1.35E-3 → 성공률 92% 사례)"
    elif mse < 3e-3:
        return "B", "✅ 양호 (실전 테스트 가치 있음)"
    elif mse < 8e-3:
        return "C", "⚠️  보통 (추가 학습 권장)"
    elif mse < 2e-2:
        return "D", "⚠️  높음 (수렴 미달 의심)"
    else:
        return "F", "❌ 매우 높음 (학습 실패 수준)"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 1: 체크포인트 메타 정보
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 1: 체크포인트 메타 정보")
print("─" * 70)

scores = {}

step_file = os.path.join(
    CKPT_DIR, "checkpoints/last/training_state/training_step.json"
)
if os.path.exists(step_file):
    with open(step_file) as f:
        actual_step = json.load(f).get("training_step", 0)
    print(f"  실제 학습 완료 스텝: {actual_step:,}")
    scores["actual_step"] = actual_step
else:
    print("  ⚠️  training_step.json 없음")
    scores["actual_step"] = 0

ckpt_root = os.path.join(CKPT_DIR, "checkpoints")
if os.path.exists(ckpt_root):
    ckpts = sorted(
        [d for d in os.listdir(ckpt_root) if d.isdigit()], key=lambda x: int(x)
    )
    print(f"  저장된 체크포인트 ({len(ckpts)}개): "
          f"{ckpts[:5]}{'...' if len(ckpts) > 5 else ''}")
    scores["num_checkpoints"] = len(ckpts)
else:
    ckpts = []
    scores["num_checkpoints"] = 0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 로딩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 모델 로딩 중...")
if not os.path.exists(CKPT):
    print(f"  ❌ 체크포인트 경로 없음: {CKPT}")
    sys.exit(1)

cam_keys = detect_camera_keys(CKPT)
STATE_DIM = detect_state_dim(CKPT)
print(f"  ℹ️  state 차원: {STATE_DIM}")

policy = ACTPolicy.from_pretrained(CKPT).to("cuda").eval()
preprocess, _ = make_pre_post_processors(
    policy.config, CKPT,
    preprocessor_overrides={"device_processor": {"device": "cuda"}},
)
print("  ✅ 완료\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 2: ACT 핵심 설정 검증 [v2: temporal ensemble 판별 수정]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("▶ 평가 2: ACT 핵심 설정 검증")
print("  근거: ACT 원본 (Zhao 2023) + LeRobot 공식 문서")
print("  [v2] temporal_ensemble_coeff 설정 시 n_action_steps=1이 정상")
print("─" * 70)

cfg = policy.config
chunk_size = getattr(cfg, "chunk_size", None)
n_action_steps = getattr(cfg, "n_action_steps", None)
te_coeff = getattr(cfg, "temporal_ensemble_coeff", None)
dim_model = getattr(cfg, "dim_model", None)
use_vae = getattr(cfg, "use_vae", None)
kl_weight = getattr(cfg, "kl_weight", None)

print(f"  chunk_size:              {chunk_size}  (ACT 원본: 100)")
print(f"  n_action_steps:          {n_action_steps}")
print(f"  temporal_ensemble_coeff: {te_coeff}  (0.01이면 앙상블 모드)")
print(f"  dim_model:               {dim_model}  (원본: 512)")
print(f"  use_vae:                 {use_vae}  (원본: True)")
print(f"  kl_weight:               {kl_weight}  (원본: 10)")

config_issues = []
te_mode = te_coeff is not None

# [v2 수정] temporal ensembling 여부에 따라 n_action_steps 판별
if te_mode:
    if n_action_steps == 1:
        print(f"\n  ✅ Temporal Ensembling 모드 — n_action_steps=1 정상")
        print(f"     (공식: coeff 설정 시 매 스텝 추론으로 앙상블 형성)")
    else:
        config_issues.append(
            f"temporal_ensemble_coeff 설정인데 n_action_steps={n_action_steps} "
            f"(반드시 1이어야 함)"
        )
else:
    if n_action_steps == 1:
        config_issues.append(
            "n_action_steps=1 + 앙상블 미설정 = 알려진 버그! → "
            "sed -i 's/\"n_action_steps\": 1/\"n_action_steps\": 50/' "
            "config.json"
        )
    elif n_action_steps != EXPECTED_N_ACTION_STEPS:
        config_issues.append(
            f"n_action_steps({n_action_steps}) ≠ 세션 표준({EXPECTED_N_ACTION_STEPS})"
        )

if dim_model != 512:
    config_issues.append(f"dim_model({dim_model}) ≠ 512")
if use_vae is False:
    config_issues.append("use_vae=False (학습 안정성 저하 가능)")

if not config_issues:
    config_grade = "A"
    print(f"\n  ✅ 모든 설정 정상  등급: A")
else:
    config_grade = ["B", "C", "D"][min(len(config_issues) - 1, 2)]
    print(f"\n  ⚠️  설정 이슈 {len(config_issues)}건:")
    for issue in config_issues:
        print(f"     - {issue}")

scores["config_grade"] = config_grade
scores["n_action_steps"] = n_action_steps
scores["te_mode"] = te_mode

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 3: 정규화 통계 검증 [v2 신규]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 3: 정규화 통계 검증 [v2 신규]")
print("  근거: SVRC 2026 '학습-배포 정규화(mean/std) 불일치는")
print("        1~3cm 오프타겟 grasp의 대표 원인'")
print("─" * 70)

norm_grade = "A"
try:
    # 체크포인트에 저장된 정규화 통계 로드 시도
    stats_candidates = [
        os.path.join(CKPT, "policy_preprocessor_step_registry.json"),
        os.path.join(CKPT, "config.json"),
    ]
    # state_dict에서 normalize 버퍼 확인 (LeRobot 표준)
    norm_keys = [k for k in policy.state_dict().keys()
                 if "normalize" in k.lower() or "buffer" in k.lower()]
    has_stats = len(norm_keys) > 0

    if has_stats:
        print(f"  ✅ 정규화 버퍼 {len(norm_keys)}개 확인")
        # action 정규화 통계의 이상 여부 (0 std는 위험 신호)
        zero_std_found = False
        for k in norm_keys:
            v = policy.state_dict()[k]
            if "std" in k.lower() and torch.any(v == 0):
                zero_std_found = True
                print(f"  ❌ {k}: std=0 채널 존재 (division 위험)")
        if zero_std_found:
            norm_grade = "D"
        else:
            print(f"  ✅ std=0 채널 없음 — 정규화 통계 건전")
    else:
        # processor 파일 존재로 대체 확인
        found = any(os.path.exists(p) for p in stats_candidates)
        if found:
            print(f"  ✅ preprocessor 설정 파일 존재 (통계는 파이프라인 내장)")
        else:
            norm_grade = "C"
            print(f"  ⚠️  정규화 통계를 직접 확인 못함 — 배포 시 주의")

    # 데이터셋 통계 파일과의 대조 (meta/info.json 존재 확인)
    ds_info = os.path.join(DATASET_DIR, "meta/info.json")
    if os.path.exists(ds_info):
        print(f"  ✅ 데이터셋 meta/info.json 존재 — 재계산 가능 상태")
    else:
        print(f"  ⚠️  데이터셋 meta 없음: {ds_info}")
except Exception as e:
    norm_grade = "C"
    print(f"  ⚠️  검증 실패: {str(e)[:50]}")

scores["norm_grade"] = norm_grade

# ── 공통 기준 관측 ──
BASE_IMG = rand_img(0)
BASE_STATE = torch.zeros(1, STATE_DIM).cuda()

def make_obs(img):
    return {k: img for k in cam_keys}

base_obs = make_obs(BASE_IMG)
base_act = get_action(policy, preprocess, base_obs, BASE_STATE)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 4: Vision 인코더 반응
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 4: Vision 인코더 반응")
print("─" * 70)

test_imgs = make_meaningful_test_imgs()
vision_diffs = []
for i, img in enumerate(test_imgs):
    a = get_action(policy, preprocess, make_obs(img))
    diff = np.abs(a - base_act).max()
    vision_diffs.append(diff)
    print(f"  img_{i+1:02d}: Δmax={diff:8.3f}°  "
          f"{'✅' if diff > VISION_THRESHOLD else '❌'}")

vision_responding = sum(1 for d in vision_diffs if d > VISION_THRESHOLD)
vision_grade, vision_msg = grade_vision(vision_responding, len(test_imgs))
print(f"\n  Vision 반응: {vision_responding}/{len(test_imgs)}"
      f"  등급: {vision_grade}  {vision_msg}")

scores.update({
    "vision_grade": vision_grade,
    "vision_responding": vision_responding,
    "vision_mean": float(np.mean(vision_diffs)),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 5: State 인코더 반응 (팔 자세 10가지)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 5: State 인코더 반응 (SO-101 팔 자세 10가지)")
print("─" * 70)

state_cases_6dof = [
    ("홈",          [0.,   -90.,  90.,   0.,  0.,  0.]),
    ("뻗기",        [45.,  -45.,  45.,   0.,  0.,  0.]),
    ("접기",        [0.,  -160., 160.,   0.,  0.,  0.]),
    ("좌회전",      [-90., -90.,  90.,   0.,  0.,  0.]),
    ("우회전",      [90.,  -90.,  90.,   0.,  0.,  0.]),
    ("그리퍼열림",  [0.,   -90.,  90.,   0.,  0., 25.]),
    ("그리퍼닫힘",  [0.,   -90.,  90.,   0.,  0., -25.]),
    ("낮은자세",    [0.,  -170., 160.,  90.,  0.,  0.]),
    ("높은자세",    [0.,   -30.,  30., -30.,  0.,  0.]),
    ("수집자세",    [2.5,  -88.,  82.,  63., -2.,  0.7]),
]

def expand_state(vals: list) -> list:
    if STATE_DIM == len(vals):
        return vals
    if STATE_DIM == len(vals) * 2:
        return vals + vals
    return (vals + [0.0] * STATE_DIM)[:STATE_DIM]

state_diffs = []
for name, vals in state_cases_6dof:
    st = torch.tensor([expand_state(vals)]).float().cuda()
    a = get_action(policy, preprocess, base_obs, st)
    d = np.abs(a - base_act).max()
    state_diffs.append(d)
    print(f"  {name:10s}: Δmax={d:8.3f}°  "
          f"{'✅' if d > STATE_THRESHOLD else '❌'}")

state_responding = sum(1 for d in state_diffs if d > STATE_THRESHOLD)
state_grade = ("A" if state_responding >= 8 else "B" if state_responding >= 5
               else "C" if state_responding >= 3 else "D")
print(f"\n  반응: {state_responding}/10  등급: {state_grade}")

scores.update({
    "state_grade": state_grade,
    "state_responding": state_responding,
    "state_mean": float(np.mean(state_diffs)),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 6: Chunk 품질 (boundary smoothness + jerk) [v2 확장]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 6: Chunk 품질 — Smoothness + Jerk [v2 확장]")
print("  근거: arXiv 2603.11642 (boundary) + arXiv 2603.11383 (jerk 표준)")
print("─" * 70)

chunk = get_chunk(policy, preprocess, base_obs, BASE_STATE)
dt = 1.0 / FPS

# 1차: 속도 (step diff)
vel = np.diff(chunk, axis=0) / dt                    # [T-1, D] °/s
# 2차: 가속도
acc = np.diff(vel, axis=0) / dt                      # [T-2, D] °/s²
# 3차: jerk [v2 신규]
jerk = np.diff(acc, axis=0) / dt                     # [T-3, D] °/s³

max_step = np.abs(np.diff(chunk, axis=0)).max()
mean_jerk = np.abs(jerk).mean()
max_jerk = np.abs(jerk).max()

print(f"  청크 길이:         {len(chunk)}  (기대: {EXPECTED_CHUNK_SIZE})")
print(f"  Step간 최대 변화:  {max_step:.3f}°")
print(f"  평균 |jerk|:       {mean_jerk:,.0f} °/s³")
print(f"  최대 |jerk|:       {max_jerk:,.0f} °/s³")

# jerk 기준: 관절 공간에서 30fps 시연 데이터의 일반 수준 ~1e4-1e5 °/s³
if max_step < 2.0 and mean_jerk < 5e4:
    chunk_grade = "A"
    print("  ✅ 부드러운 궤적 (jerk 낮음)")
elif max_step < 5.0 and mean_jerk < 2e5:
    chunk_grade = "B"
    print("  ✅ 허용 가능 범위")
elif max_step < 10.0:
    chunk_grade = "C"
    print("  ⚠️  다소 급격 — 실전에서 EMA 스무딩 고려")
else:
    chunk_grade = "D"
    print("  ❌ 급격한 변화 — jitter 위험")

scores.update({
    "chunk_grade": chunk_grade,
    "max_step_diff": float(max_step),
    "mean_jerk": float(mean_jerk),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 7: Mode Averaging 진단 (Vision 다양성)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 7: ⭐ Mode Averaging 진단 (Vision 다양성)")
print("─" * 70)

chunks_per_img = []
for img in test_imgs[:5]:
    chunks_per_img.append(get_chunk(policy, preprocess, make_obs(img), BASE_STATE))
chunks_img_arr = np.array(chunks_per_img)

final_std = np.std(chunks_img_arr[:, -1, :], axis=0).mean()
mode_grade, mode_msg = grade_mode_avg(final_std)
print(f"  최종점 분산 (5장): {final_std:.3f}°")
print(f"  등급: {mode_grade}  {mode_msg}")

scores.update({"mode_avg_grade": mode_grade, "final_std": float(final_std)})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 8: Observation 일관성 (재현성)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 8: Observation 일관성 (재현성)")
print("─" * 70)

repeat_actions = [get_action(policy, preprocess, base_obs, BASE_STATE)
                  for _ in range(10)]
repeat_std = np.std(np.array(repeat_actions), axis=0).mean()
rep_grade = ("A" if repeat_std < 0.1 else "B" if repeat_std < 0.5
             else "C" if repeat_std < 2.0 else "D")
print(f"  10회 반복 표준편차: {repeat_std:.4f}°  등급: {rep_grade}")

scores.update({"consistency_grade": rep_grade,
               "consistency_mean": float(repeat_std)})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 9: Mode COLLAPSE 진단 (State 다양성)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 9: ⭐⭐ Mode COLLAPSE 진단 (State 다양성)")
print("─" * 70)

test_states_for_collapse = [
    expand_state([0.,   -90.,  90.,   0.,  0.,  0.]),
    expand_state([45.,  -45.,  45.,   0.,  0.,  0.]),
    expand_state([-45., -120., 130.,  30.,  0., 10.]),
    expand_state([60.,  -60.,  60., -20.,  0., -20.]),
    expand_state([2.5,  -88.,  82.,  63., -2.,  0.7]),
]
chunks_per_state = []
for vals in test_states_for_collapse:
    st = torch.tensor([vals]).float().cuda()
    chunks_per_state.append(
        get_chunk(policy, preprocess, make_obs(test_imgs[0]), st))
chunks_state_arr = np.array(chunks_per_state)

state_final_std = np.std(chunks_state_arr[:, -1, :], axis=0).mean()
chunk_ranges = [(np.max(c, axis=0) - np.min(c, axis=0)).mean()
                for c in chunks_per_img]
avg_chunk_range = np.mean(chunk_ranges)

collapse_grade, collapse_msg = grade_mode_collapse(
    state_final_std, avg_chunk_range)
print(f"  State 변이 최종점 분산: {state_final_std:.3f}°")
print(f"  Chunk 내부 움직임:      {avg_chunk_range:.3f}°")
print(f"  등급: {collapse_grade}  {collapse_msg}")

scores.update({
    "collapse_grade": collapse_grade,
    "state_final_std": float(state_final_std),
    "avg_chunk_range": float(avg_chunk_range),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 10: Action Range 안전성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 10: Action Range 안전성")
print("─" * 70)

action_dim = chunks_state_arr.shape[-1]
all_actions = chunks_state_arr.reshape(-1, action_dim)
total_oor = 0
for j in range(min(action_dim, 12)):
    name = JOINT_NAMES[j % 6] + ("_R" if j >= 6 else "")
    lo, hi = ACTION_SAFE_RANGE[j % 6]
    vals = all_actions[:, j]
    oor_cnt = np.sum((vals < lo) | (vals > hi))
    total_oor += oor_cnt
    print(f"  {name:<10} | min {vals.min():>8.2f} | max {vals.max():>8.2f} | "
          f"{'✅' if oor_cnt == 0 else f'❌ {oor_cnt}건 초과'}")

range_grade = ("A" if total_oor == 0
               else "B" if total_oor < len(all_actions) * 0.01
               else "C" if total_oor < len(all_actions) * 0.05 else "D")
print(f"\n  범위 초과 총 {total_oor}건  등급: {range_grade}")
scores.update({"range_grade": range_grade, "total_oor": int(total_oor)})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 11: ⭐⭐⭐ Validation MSE + CI-MSE (ground-truth 대조) [v2 신규]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "★" * 70)
print("▶ 평가 11: ⭐⭐⭐ Validation MSE + CI-MSE (ground-truth 대조)")
print("  근거 1: TOTO(arXiv 2306.00942) — 검증 시연 action MSE가 표준")
print("  근거 2: CI-MSE(arXiv 2606.29898) — grasp 결정 구간만 계산하면")
print("          성공률 상관 -0.61 → -0.87로 개선")
print("  방법: 마지막 20% 에피소드를 검증셋으로, 실제 (관측→정답) 쌍 대조")
print("★" * 70)


def load_validation_samples():
    """데이터셋에서 검증 샘플 로드: (이미지, state, gt_chunk, is_critical).

    [근거] CI-MSE 논문 — grasp 순간 ± CI_HALF_WINDOW 프레임을
           critical 샘플로 표시. 나머지는 일반 샘플.
    """
    import pandas as pd
    from torchvision import transforms
    from PIL import Image
    import io

    data_files = sorted(glob.glob(f"{DATASET_DIR}/data/chunk-*/file-*.parquet"))
    if not data_files:
        return []

    all_data = pd.concat([pd.read_parquet(f) for f in data_files])
    ep_ids = sorted(all_data['episode_index'].unique())
    n_val = max(1, int(len(ep_ids) * VAL_EPISODE_RATIO))
    val_eps = ep_ids[-n_val:]   # 마지막 20%를 검증셋 (시간순 분리)
    print(f"  📦 전체 {len(ep_ids)} 에피소드 중 검증셋 {n_val}개 "
          f"(episode {val_eps[0]}~{val_eps[-1]})")

    to_tensor = transforms.Compose([
        transforms.Resize((480, 640)), transforms.ToTensor()])

    def decode_img(cell):
        """LeRobotDataset v3: 이미지가 dict {'bytes':..., 'path':''}로 저장됨."""
        if isinstance(cell, dict) and cell.get('bytes'):
            return Image.open(io.BytesIO(cell['bytes'])).convert("RGB")
        return None

    samples = []
    img_cols = [c for c in all_data.columns if "images" in c]
    for ep_idx in val_eps:
        ep = all_data[all_data['episode_index'] == ep_idx].reset_index(drop=True)
        actions = np.stack(ep['action'].values)
        states = np.stack(ep['observation.state'].values)
        T = len(ep)

        # grasp 순간 탐지 (grip이 처음 크게 닫히는 프레임)
        grips = actions[:, 5]
        closing = np.where(grips < grips.min() * 0.5)[0]
        grasp_t = int(closing[0]) if len(closing) > 0 else T // 2

        # 샘플 프레임 선택: critical(4개) + 일반(4개)
        crit_frames = [max(0, grasp_t - CI_HALF_WINDOW), grasp_t,
                       min(T - chunk_size - 1, grasp_t + CI_HALF_WINDOW // 2),
                       max(0, grasp_t - CI_HALF_WINDOW // 2)]
        norm_frames = list(np.linspace(0, max(0, T - chunk_size - 1),
                                       VAL_SAMPLES_PER_EP // 2, dtype=int))
        for t, is_crit in ([(f, True) for f in crit_frames] +
                           [(f, False) for f in norm_frames]):
            t = int(min(t, T - 2))
            gt_end = min(t + chunk_size, T)
            gt_chunk = actions[t:gt_end]
            if len(gt_chunk) < 5:
                continue
            imgs = {}
            ok = True
            for col in img_cols:
                pil = decode_img(ep[col].iloc[t])
                if pil is None:
                    ok = False
                    break
                imgs[col] = to_tensor(pil).unsqueeze(0)
            if not ok or not imgs:
                continue
            samples.append({
                "imgs": imgs, "state": states[t],
                "gt_chunk": gt_chunk, "critical": is_crit,
                "grasp_dist": abs(t - grasp_t),
            })
    return samples


def compute_val_mse(pol, prep, samples):
    """정규화 공간 MSE 계산 (전체 + critical interval).

    [근거] LeRobot ACT 학습 loss는 정규화 공간 L1. 비교 가능하도록
           예측/정답 모두 데이터셋 통계로 정규화 후 MSE 계산.
           통계 접근이 어려우면 원시 공간 MSE를 채널 std로 나눠 근사.
    """
    all_mse, crit_mse = [], []
    # 근사 정규화: 검증 샘플 정답 action의 채널별 std 사용
    gt_all = np.concatenate([s["gt_chunk"] for s in samples], axis=0)
    ch_std = gt_all.std(axis=0) + 1e-6

    for s in samples:
        obs = {k: v.cuda() for k, v in s["imgs"].items()}
        st = torch.tensor([expand_state(list(s["state"][:6]))
                           if STATE_DIM == 6 else list(s["state"])
                           ]).float().cuda()
        try:
            pred = get_chunk(pol, prep, obs, st)
        except Exception:
            continue
        L = min(len(pred), len(s["gt_chunk"]))
        diff = (pred[:L] - s["gt_chunk"][:L]) / ch_std   # 채널 정규화
        mse = float((diff ** 2).mean())
        all_mse.append(mse)
        if s["critical"]:
            crit_mse.append(mse)
    return (np.mean(all_mse) if all_mse else float("nan"),
            np.mean(crit_mse) if crit_mse else float("nan"),
            len(all_mse))


val_samples = []
try:
    val_samples = load_validation_samples()
except Exception as e:
    print(f"  ⚠️  검증 샘플 로드 실패: {str(e)[:60]}")

if val_samples:
    val_mse, ci_mse, n_used = compute_val_mse(policy, preprocess, val_samples)
    # 정규화 근사 공간이므로 스케일 보정 (1e-3 대와 비교하려면 /1000)
    print(f"\n  검증 샘플 수:        {n_used}")
    print(f"  📊 Validation MSE (전체):     {val_mse:.5f}")
    print(f"  📊 CI-MSE (grasp 결정 구간):  {ci_mse:.5f}")
    ratio = ci_mse / val_mse if val_mse > 0 else float("nan")
    print(f"  📊 CI/전체 비율:              {ratio:.2f}")
    print(f"     → 1.5 이상이면 grasp 구간에서 오차 집중 (정밀 교정 필요)")

    # 정규화 근사 공간 등급 (채널 std 정규화 기준: <0.05 우수)
    if val_mse < 0.05:
        vmse_grade = "A"; print(f"\n  ✅ 검증 오차 우수 — 배포 가치 높음")
    elif val_mse < 0.15:
        vmse_grade = "B"; print(f"\n  ✅ 검증 오차 양호")
    elif val_mse < 0.4:
        vmse_grade = "C"; print(f"\n  ⚠️  검증 오차 보통 — 추가 학습 권장")
    else:
        vmse_grade = "D"; print(f"\n  ❌ 검증 오차 큼 — 수렴 미달/과적합 의심")

    scores.update({
        "val_mse_grade": vmse_grade,
        "val_mse": float(val_mse),
        "ci_mse": float(ci_mse),
        "ci_ratio": float(ratio),
    })
else:
    print("  ⚠️  검증 샘플 없음 — 평가 11/12는 분산 지표로 대체됨")
    scores["val_mse_grade"] = "-"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 12: ⭐⭐⭐ 체크포인트 Top-3 최적 후보 추천 [v2 전면 개편]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "★" * 70)
print("▶ 평가 12: ⭐⭐⭐ 체크포인트 Top-3 최적 후보 추천 [v2 개편]")
print("  근거: SVRC 'Do NOT use the final checkpoint —")
print("        lowest validation loss 체크포인트를 선택하라'")
print("        robomimic — 단일 지표는 최선 대비 10~100% 저하 → 복합 점수")
print("  방법: Val MSE(가중 5) + CI-MSE(가중 3) + 분산 적정성(가중 2)")
print("★" * 70)

ckpt_results = []
if len(ckpts) >= 2:
    ckpts_to_eval = ckpts[-8:] if len(ckpts) > 8 else ckpts
    # 검증 샘플 부분집합 (속도)
    sub_samples = val_samples[:24] if val_samples else []

    print(f"\n  평가 대상: 마지막 {len(ckpts_to_eval)}개 체크포인트\n")
    print(f"  {'체크포인트':>10} | {'ValMSE':>9} | {'CI-MSE':>9} | "
          f"{'분산':>7} | {'복합점수':>8}")
    print(f"  {'-' * 58}")

    for ckpt_name in ckpts_to_eval:
        ckpt_path = os.path.join(
            CKPT_DIR, "checkpoints", ckpt_name, "pretrained_model")
        if not os.path.exists(ckpt_path):
            continue
        try:
            p = ACTPolicy.from_pretrained(ckpt_path).to("cuda").eval()
            pp, _ = make_pre_post_processors(
                p.config, ckpt_path,
                preprocessor_overrides={"device_processor": {"device": "cuda"}},
            )

            # (1) Validation MSE + CI-MSE
            if sub_samples:
                v_mse, c_mse, _ = compute_val_mse(p, pp, sub_samples)
            else:
                v_mse, c_mse = float("nan"), float("nan")

            # (2) 분산 적정성 (mode averaging 회피, 보조 지표)
            local_finals = []
            for img in test_imgs[:3]:
                reset_policy_state(p)
                batch = {**make_obs(img), "observation.state": BASE_STATE}
                batch = pp(batch)
                with torch.no_grad():
                    c = p.predict_action_chunk(batch)[0].cpu().numpy()
                local_finals.append(c[-1])
            local_std = np.std(np.array(local_finals), axis=0).mean()

            # (3) 복합 점수 — 낮을수록 좋음
            #     ValMSE(x5) + CI-MSE(x3) - 분산보너스(x2, 적정범위만)
            div_bonus = 0.0
            if 0.5 <= local_std <= 15.0:  # averaging 아님 + 폭주 아님
                div_bonus = 0.02
            if not np.isnan(v_mse):
                composite = v_mse * 5 + c_mse * 3 - div_bonus * 2
            else:
                composite = -local_std  # 폴백: 분산만으로 (v1 방식)

            ckpt_results.append({
                "name": ckpt_name, "val_mse": v_mse,
                "ci_mse": c_mse, "std": local_std,
                "composite": composite,
            })
            print(f"  step {ckpt_name:>6} | {v_mse:>9.5f} | {c_mse:>9.5f} | "
                  f"{local_std:>7.2f} | {composite:>8.4f}")

            del p, pp
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  step {ckpt_name:>6} | 로딩 실패: {str(e)[:30]}")

    if ckpt_results:
        ranked = sorted(ckpt_results, key=lambda x: x["composite"])
        print(f"\n  🏆 최적 모델 후보 Top-3 (복합 점수 낮은 순):")
        medals = ["🥇 1위", "🥈 2위", "🥉 3위"]
        for i, r in enumerate(ranked[:3]):
            print(f"\n  {medals[i]}: step {r['name']}")
            print(f"     Val MSE: {r['val_mse']:.5f} | CI-MSE: {r['ci_mse']:.5f}"
                  f" | 분산: {r['std']:.2f}°")
            print(f"     경로: {CKPT_DIR}/checkpoints/{r['name']}/pretrained_model")

        best = ranked[0]
        is_last = best["name"] == ckpts[-1]
        if not is_last:
            print(f"\n  💡 주목: 최종 체크포인트({ckpts[-1]})가 최적이 아님!")
            print(f"     글로벌 표준대로 step {best['name']} 사용 권장")
        scores["best_ckpt"] = best["name"]
        scores["top3"] = [r["name"] for r in ranked[:3]]
else:
    print("  ⚠️  체크포인트 부족 — 후보 추천 생략")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 13: 추론 지연 측정 [v2 신규]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 13: 추론 지연 측정 [v2 신규]")
print(f"  근거: VOTE(arXiv 2507.05116) — {LATENCY_TRIALS}회 쿼리 표준")
print(f"        {FPS}fps 제어 → select_action {1000//FPS}ms 이내 필수")
print("─" * 70)

# 워밍업 (CUDA graph/캐시 안정화)
for _ in range(5):
    _ = get_action(policy, preprocess, base_obs, BASE_STATE)
torch.cuda.synchronize()

lat_select = []
for _ in range(LATENCY_TRIALS):
    t0 = time.perf_counter()
    _ = get_action(policy, preprocess, base_obs, BASE_STATE)
    torch.cuda.synchronize()
    lat_select.append((time.perf_counter() - t0) * 1000)

lat_chunk = []
for _ in range(20):
    t0 = time.perf_counter()
    _ = get_chunk(policy, preprocess, base_obs, BASE_STATE)
    torch.cuda.synchronize()
    lat_chunk.append((time.perf_counter() - t0) * 1000)

p50 = np.percentile(lat_select, 50)
p95 = np.percentile(lat_select, 95)
budget = 1000 / FPS

print(f"  select_action:  p50 {p50:.1f}ms | p95 {p95:.1f}ms "
      f"(예산: {budget:.0f}ms)")
print(f"  chunk 생성:     평균 {np.mean(lat_chunk):.1f}ms")

if p95 < budget * 0.5:
    lat_grade = "A"; print(f"  ✅ 실시간 여유 충분 (p95 < 예산 50%)")
elif p95 < budget:
    lat_grade = "B"; print(f"  ✅ 실시간 가능 (p95 < 예산)")
elif p50 < budget:
    lat_grade = "C"; print(f"  ⚠️  p95 초과 — 간헐적 프레임 드랍 가능")
else:
    lat_grade = "D"; print(f"  ❌ 예산 초과 — 실시간 제어 불가")

scores.update({"latency_grade": lat_grade, "lat_p95": float(p95)})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 최종 종합 평가
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  📊 최종 종합 평가 (SO-101 ACT v2)")
print("=" * 70)

grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0, "-": 2}

# [v2] Validation MSE가 최고 가중치 (글로벌 표준: 실성능 최상 예측 지표)
total = (
    grade_map.get(scores.get("val_mse_grade",  "-"), 2) * 6 +  # ⭐⭐⭐ GT 대조
    grade_map.get(scores.get("collapse_grade", "F"), 0) * 5 +  # ⭐⭐ 공간 적응
    grade_map.get(scores.get("mode_avg_grade", "F"), 0) * 4 +
    grade_map.get(scores.get("config_grade",   "F"), 0) * 3 +
    grade_map.get(scores.get("vision_grade",   "F"), 0) * 3 +
    grade_map.get(scores.get("state_grade",    "F"), 0) * 3 +
    grade_map.get(scores.get("range_grade",    "F"), 0) * 3 +
    grade_map.get(scores.get("latency_grade",  "F"), 0) * 2 +
    grade_map.get(scores.get("chunk_grade",    "F"), 0) * 2 +
    grade_map.get(scores.get("norm_grade",     "F"), 0) * 2 +
    grade_map.get(scores.get("consistency_grade", "F"), 0) * 1
)
max_total = 4 * (6 + 5 + 4 + 3 + 3 + 3 + 3 + 2 + 2 + 2 + 1)
pct = total / max_total * 100

print(f"""
  ┌────────────────────────────────────────────────────────────┐
  │  ⭐⭐⭐ Validation MSE (GT 대조):     {scores.get('val_mse_grade','-'):>2}   가중치 x6  │
  │  ⭐⭐  Mode Collapse (공간 적응):    {scores.get('collapse_grade','?'):>2}   가중치 x5  │
  │  ⭐   Mode Averaging (Vision):      {scores.get('mode_avg_grade','?'):>2}   가중치 x4  │
  │  ACT 설정 (ensemble 판별 포함):     {scores.get('config_grade','?'):>2}   가중치 x3  │
  │  Vision 인코더:                      {scores.get('vision_grade','?'):>2}   가중치 x3  │
  │  State 인코더:                       {scores.get('state_grade','?'):>2}   가중치 x3  │
  │  Action Range 안전성:                {scores.get('range_grade','?'):>2}   가중치 x3  │
  │  추론 지연 (실시간성):               {scores.get('latency_grade','?'):>2}   가중치 x2  │
  │  Chunk 품질 (jerk 포함):             {scores.get('chunk_grade','?'):>2}   가중치 x2  │
  │  정규화 통계:                        {scores.get('norm_grade','?'):>2}   가중치 x2  │
  │  Observation 일관성:                 {scores.get('consistency_grade','?'):>2}   가중치 x1  │
  ├────────────────────────────────────────────────────────────┤
  │  종합 점수: {total}/{max_total}  ({pct:.0f}%)                                    │
  └────────────────────────────────────────────────────────────┘""")

print(f"""
  ── 🏆 최적 모델 후보 (평가 12) ──────────────────────────
     Top-3: {scores.get('top3', ['?'])}
     1위 배포 경로:
     {CKPT_DIR}/checkpoints/{scores.get('best_ckpt', 'last')}/pretrained_model

  ── 📊 핵심 숫자 정리 ────────────────────────────────────
     학습 스텝:             {scores.get('actual_step', 0):,}
     Validation MSE:        {scores.get('val_mse', float('nan')):.5f} ({scores.get('val_mse_grade','-')})
     CI-MSE (grasp 구간):   {scores.get('ci_mse', float('nan')):.5f}
     CI/전체 비율:          {scores.get('ci_ratio', float('nan')):.2f} (1.5+ = grasp 오차 집중)
     공간 적응력 분산:       {scores.get('state_final_std', 0):.3f}° ({scores.get('collapse_grade','?')})
     Vision 최종점 분산:     {scores.get('final_std', 0):.3f}° ({scores.get('mode_avg_grade','?')})
     평균 |jerk|:           {scores.get('mean_jerk', 0):,.0f} °/s³
     추론 p95 지연:          {scores.get('lat_p95', 0):.1f}ms (예산 {1000//FPS}ms)
     Action 범위 초과:       {scores.get('total_oor', 0)}건

  ── 📚 v2 신규 반영 글로벌 레퍼런스 ────────────────────
     • TOTO 벤치마크: arXiv 2306.00942 (검증 action MSE 표준)
     • CI-MSE: arXiv 2606.29898 (2026.06, 상관 -0.61→-0.87)
     • Policy Comparison: 2025 (val MSE와 성공률 검증 사례)
     • robomimic: arXiv 2108.03298 (단일 지표 함정)
     • SVRC 2026 (final checkpoint 사용 금지 원칙)
     • VOTE: arXiv 2507.05116 (latency 측정 표준)
     • SIMPLER: arXiv 2405.05941 (val MSE 한계 — sim 평가 보완 필요)

  ── ⚠️  오프라인 평가의 한계 (정직한 고지) ──────────────
     SIMPLER 연구: "val MSE는 실세계 성능의 완벽한 proxy가 아님"
     → 이 스크립트는 배포 '후보 선별'용. 최종 판정은 반드시
       실기 10회+ 성공률 측정으로 (Policy Comparison: 50 trials 권장)
""")

print("=" * 70)
print(f"\n  📄 로그 파일: {LOG_FILE}")
print("=" * 70)
