"""
quest3_hand_interaction.py
Quest 3 손으로 물체를 만지면 색이 변하는 인터랙션 (완성본)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
검증으로 확정된 핵심 사실 (2026-07-25):

1. 접속: https://vuer.ai?ws=wss://<PC_IP>:8012  (반드시 wss + 인증서)
   또는 https://<PC_IP>:8012/?ws=wss://<PC_IP>:8012
   → Enter VR(Virtual Reality) 버튼 클릭 → 컨트롤러 내려놓고 양손 들기

2. HTTPS 필수: WebXR 핸드트래킹은 보안 컨텍스트에서만 동작
   → openssl로 cert.pem/key.pem 생성 후 Vuer에 전달

3. 손 데이터 구조: left/right = 25관절 × 16 (4×4 행렬)
   - msgpack ExtType(code=0)으로 전송됨
   - 관절 위치 = 행렬의 offset [12, 13, 14] = (x, y, z)
   - 좌표는 월드 좌표계

4. 워밍업 지연: 접속 직후 몇 초는 len=1 빈 값(0x00)이 옴.
   손이 안정적으로 잡히면 실제 25관절 데이터가 흐름. → 정상 동작

5. 실측 손 위치 범위:
   - 오른손: X≈+0.2, 왼손: X≈-0.14
   - Y(높이) ≈ 1.0 ~ 1.15  (허리~가슴 높이)
   - Z(앞뒤) ≈ -0.3 ~ -0.5  (몸 앞쪽)
   → 물체를 이 범위에 배치해야 손이 실제로 닿음
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

원리:
    1. 매 프레임 손 관절(25개) 위치를 받음
    2. 검지끝·엄지끝·손목 등 주요 관절과 각 물체의 거리 계산
    3. 가까우면 색깔 변경(하얀색)
    4. 멀어지면 원래 색으로 복귀
"""

import os
import socket
import asyncio
import math
import numpy as np
from vuer import Vuer
from vuer.schemas import Scene, Sphere, Box, AmbientLight, DirectionalLight, Hands


# ==================================================
# 네트워크 / 인증서
# ==================================================
def get_local_ip():
    """이 PC의 로컬 IP 주소 자동 탐지"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


PC_IP = get_local_ip()

# 인증서 확인 (없으면 안내)
CERT = "./cert.pem"
KEY = "./key.pem"
HAS_CERT = os.path.exists(CERT) and os.path.exists(KEY)

print("=" * 64)
print("✋ Quest 3 손 인터랙션 (완성본)")
print("=" * 64)
if HAS_CERT:
    print(f"🌐 Quest 3 접속 주소:")
    print(f"   https://vuer.ai?ws=wss://{PC_IP}:8012")
    print(f"   (또는) https://{PC_IP}:8012/?ws=wss://{PC_IP}:8012")
    print("")
    print("📋 접속 순서:")
    print("   1) 위 주소로 접속 → 인증서 경고 시 '고급→계속 진행'")
    print("   2) 우측 하단 'Virtual Reality' 버튼 클릭")
    print("   3) 컨트롤러 내려놓고 양손을 눈앞에 들기")
    print("   4) 손가락 펴고 천천히 움직이면 트래킹 시작")
else:
    print("⚠️  인증서(cert.pem/key.pem)가 없습니다!")
    print("   WebXR 손 트래킹은 HTTPS 필수. 아래 명령으로 생성하세요:")
    print(f"   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\")
    print(f"     -keyout key.pem -out cert.pem -subj '/CN={PC_IP}'")
print("=" * 64)
print("")


# ==================================================
# Vuer 앱 (HTTPS/wss)
# ==================================================
if HAS_CERT:
    app = Vuer(host="0.0.0.0", port=8012, cors="*", cert=CERT, key=KEY)
else:
    app = Vuer(host="0.0.0.0", port=8012, cors="*")


# ==================================================
# 물체 정보 (실측 손 위치 범위에 맞춰 배치)
#   Y ≈ 1.0~1.2, X ≈ ±0.3, Z ≈ -0.3~-0.5
# ==================================================
OBJECTS = {
    "red_ball": {
        "position": [0.0, 1.1, -0.4],
        "size": 0.08,
        "type": "sphere",
        "color_normal": "red",
        "color_touched": "white",
    },
    "green_box": {
        "position": [0.25, 1.1, -0.4],
        "size": 0.12,
        "type": "box",
        "color_normal": "green",
        "color_touched": "white",
    },
    "blue_box": {
        "position": [-0.25, 1.1, -0.4],
        "size": 0.12,
        "type": "box",
        "color_normal": "blue",
        "color_touched": "white",
    },
    "yellow_ball": {
        "position": [0.0, 1.3, -0.45],
        "size": 0.07,
        "type": "sphere",
        "color_normal": "yellow",
        "color_touched": "white",
    },
    "purple_ball": {
        "position": [0.0, 0.9, -0.4],
        "size": 0.07,
        "type": "sphere",
        "color_normal": "magenta",
        "color_touched": "white",
    },
}

# 접촉 임계값 (m) — 손 관절과 물체 중심 거리
TOUCH_DISTANCE = 0.12  # 12cm 이내면 접촉


# ==================================================
# 손 데이터 디코딩 (검증 확정 로직)
# ==================================================
def decode_hand(ext):
    """
    Vuer HAND_MOVE의 left/right(ExtType) → (25, 16) numpy 행렬로 복원.
    워밍업 중 빈 값(len<=2)이면 None 반환.
    """
    if ext is None:
        return None

    # 이미 numpy/list로 복원된 경우
    if isinstance(ext, np.ndarray):
        arr = ext.astype("float32").ravel()
    elif isinstance(ext, (list, tuple)):
        if len(ext) < 16:
            return None
        arr = np.array(ext, dtype="float32").ravel()
    else:
        # msgpack ExtType — .data 바이트에서 float32 파싱
        raw = getattr(ext, "data", None)
        if raw is None or not hasattr(raw, "__len__") or len(raw) <= 2:
            return None  # 워밍업 빈 값(0x00)
        arr = np.frombuffer(bytes(raw), dtype="<f4")

    if arr.size < 16 or arr.size % 16 != 0:
        return None
    return arr.reshape(-1, 16)  # (관절수, 16)


def joint_xyz(matrices, idx):
    """관절 idx의 (x, y, z) 위치. 행렬 offset [12,13,14]"""
    if idx >= matrices.shape[0]:
        return None
    m = matrices[idx]
    return [float(m[12]), float(m[13]), float(m[14])]


def get_hand_points(value):
    """
    HAND_MOVE value에서 양손의 주요 관절 위치 리스트 추출.
    주요 관절: 손목(0), 엄지끝(4), 검지끝(9), 중지끝(14)
    """
    points = []
    KEY_JOINTS = [0, 4, 9, 14]  # 손목, 엄지끝, 검지끝, 중지끝
    for side in ("right", "left"):
        mats = decode_hand(value.get(side))
        if mats is None:
            continue
        for j in KEY_JOINTS:
            p = joint_xyz(mats, j)
            if p is not None:
                points.append(p)
    return points


# ==================================================
# 유틸리티
# ==================================================
def distance(p1, p2):
    """두 점 사이 유클리드 거리"""
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )


def create_object(name, info, is_touched=False):
    """물체(Sphere/Box) 생성. key로 업데이트 대상 지정"""
    color = info["color_touched"] if is_touched else info["color_normal"]
    if info["type"] == "sphere":
        return Sphere(
            args=[info["size"], 32, 32],
            position=info["position"],
            material=dict(color=color),
            key=name,
        )
    else:  # box
        return Box(
            args=[info["size"], info["size"], info["size"]],
            position=info["position"],
            material=dict(color=color),
            key=name,
        )


# ==================================================
# 상태
# ==================================================
touched_objects = set()
tracking_started = [False]  # 첫 유효 데이터 수신 여부


# ==================================================
# 손 이벤트 핸들러 (핵심)
#   ★★★ 반드시 @app.spawn 보다 먼저 등록해야 함! ★★★
#   @app.spawn(start=True)는 데코레이터 시점에 서버를 즉시
#   실행하므로, 그 아래의 add_handler는 등록되지 않는다.
#   (실험으로 검증: 핸들러가 spawn 아래에 있으면 HAND=0)
# ==================================================
@app.add_handler("HAND_MOVE")
async def on_hand(event, session):
    global touched_objects

    value = event.value
    if not isinstance(value, dict):
        return

    # 양손 주요 관절 위치 추출 (워밍업 빈 값은 자동 skip)
    hand_points = get_hand_points(value)
    if not hand_points:
        return  # 아직 유효 데이터 없음

    if not tracking_started[0]:
        tracking_started[0] = True
        print("🎯 손 트래킹 시작! (25관절 수신)")

    # 각 물체와 손 관절 거리 확인
    new_touched = set()
    for name, info in OBJECTS.items():
        obj_pos = info["position"]
        for hp in hand_points:
            if distance(hp, obj_pos) < TOUCH_DISTANCE:
                new_touched.add(name)
                break

    # 상태 변화 감지
    just_touched = new_touched - touched_objects
    just_released = touched_objects - new_touched

    # 새로 만진 물체 → 하얀색
    for name in just_touched:
        print(f"✋ 접촉 시작: {name}")
        session.update @ create_object(name, OBJECTS[name], is_touched=True)

    # 뗀 물체 → 원래 색
    for name in just_released:
        print(f"👋 접촉 해제: {name}")
        session.update @ create_object(name, OBJECTS[name], is_touched=False)

    touched_objects = new_touched


# ==================================================
# VR 진입 감지 (선택)
# ==================================================
camera_first = [False]


@app.add_handler("CAMERA_MOVE")
async def on_camera(event, session):
    if not camera_first[0]:
        camera_first[0] = True
        print("📷 VR 카메라 활성화!")


# ==================================================
# 메인 (씬 로드 + 상태 출력)
# ==================================================
@app.spawn(start=True)
async def main(session):
    print("✅ 브라우저 접속됨! 씬 로드 중...")

    session.set @ Scene(
        AmbientLight(intensity=1.0, key="ambient"),
        DirectionalLight(intensity=1.5, position=[3, 5, 3], key="sun"),
        Hands(fps=30, stream=True, key="hands", left=True, right=True),
        *[create_object(name, info) for name, info in OBJECTS.items()],
        up=[0, 1, 0],
    )

    print("👀 5개 물체 표시 (손 높이 y≈0.9~1.3에 배치)")
    print("✋ 'Virtual Reality' 진입 후 손을 뻗어 물체를 만지세요")
    print(f"   - 접촉 감지 거리: {TOUCH_DISTANCE * 100:.0f}cm")
    print("   - 물체 근처로 손 → 하얀색 / 손 떼면 → 원래 색")
    print("   ⏳ 접속 직후 몇 초는 트래킹 워밍업(빈 값)입니다. 손을 계속 보여주세요.")
    print("")

    counter = 0
    while True:
        await asyncio.sleep(10)
        counter += 10
        status = "트래킹 ON" if tracking_started[0] else "손 대기 중(워밍업)"
        print(f"⏱️ {counter}초 | {status} | 접촉: {len(touched_objects)}개")


if __name__ == "__main__":
    print("⏳ 서버 시작...")
    app.run()
