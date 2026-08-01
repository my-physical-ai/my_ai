"""
quest3_hand_sound.py
Quest 3 손 인터랙션 + 사운드 피드백 (물체마다 다른 음계!)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
quest3_hand_interaction.py 에 사운드를 추가한 버전.

사운드 설계 (실로폰 방식):
    - 물체마다 다른 음: 도(빨강) 레(초록) 미(파랑) 솔(노랑) 라(보라)
    - 접촉(touch)  → 그 물체의 음을 밝게 (높은 옥타브, 0.18초)
    - 해제(release)→ 같은 음을 낮게 짧게 (한 옥타브 아래, 0.10초)
    - 재생 위치: PC 스피커 (서버가 접촉을 판정하므로 지연 최소)

구현 방식:
    - 외부 패키지 불필요: numpy(이미 설치됨) + 표준 wave 모듈로
      시작 시 WAV 파일을 미리 생성 → aplay/paplay 로 비동기 재생
    - 재생은 subprocess.Popen 이라 메인 루프를 막지 않음(논블로킹)

핵심 주의사항 (이전 실습에서 확정):
    1. 모든 @app.add_handler 는 @app.spawn 보다 반드시 위에!
    2. 접속은 https://vuer.ai?ws=wss://<PC_IP>:8012&grid=False
    3. wss(HTTPS) 필수 — cert.pem/key.pem 같은 폴더에
    4. 종료는 Ctrl+C (Ctrl+Z 금지)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import math
import wave
import shutil
import socket
import asyncio
import tempfile
import subprocess
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
CERT = "./cert.pem"
KEY = "./key.pem"
HAS_CERT = os.path.exists(CERT) and os.path.exists(KEY)


# ==================================================
# 사운드 엔진 (numpy → WAV 생성 → aplay/paplay 재생)
# ==================================================
SAMPLE_RATE = 44100
SOUND_DIR = os.path.join(tempfile.gettempdir(), "quest3_sounds")

# 재생기 자동 선택 (Ubuntu 기본: paplay(PulseAudio) 우선, 없으면 aplay(ALSA))
PLAYER = None
for cand in (["paplay"], ["aplay", "-q"]):
    if shutil.which(cand[0]):
        PLAYER = cand
        break


def make_tone_wav(path, freq, duration=0.18, volume=0.5):
    """
    지정 주파수의 짧은 '띵' 사운드를 WAV 파일로 생성.
    - 사인파 + 페이드아웃(딱딱 끊기는 소리 방지)
    - 배음 약간 추가(실로폰 느낌)
    """
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n) / SAMPLE_RATE
    # 기본음 + 2배음(약하게) → 맑은 실로폰 톤
    tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    # 지수 페이드아웃 (타악기처럼 '띵-' 하고 사라짐)
    envelope = np.exp(-t * 18)
    samples = (tone * envelope * volume * 32767).astype(np.int16)

    with wave.open(path, "wb") as w:
        w.setnchannels(1)          # 모노
        w.setsampwidth(2)          # 16bit
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())


def play_sound(path):
    """WAV 비동기 재생 (메인 루프를 막지 않음)"""
    if PLAYER is None:
        return
    try:
        subprocess.Popen(
            PLAYER + [path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


# ==================================================
# 물체 정보 (위치 + 색 + 음계)
#   음계: 도(C5) 레(D5) 미(E5) 솔(G5) 라(A5) — 실로폰!
# ==================================================
OBJECTS = {
    "red_ball": {
        "position": [0.0, 1.1, -0.4],
        "size": 0.08,
        "type": "sphere",
        "color_normal": "red",
        "color_touched": "white",
        "note": "도",
        "freq": 523.25,   # C5
    },
    "green_box": {
        "position": [0.25, 1.1, -0.4],
        "size": 0.12,
        "type": "box",
        "color_normal": "green",
        "color_touched": "white",
        "note": "레",
        "freq": 587.33,   # D5
    },
    "blue_box": {
        "position": [-0.25, 1.1, -0.4],
        "size": 0.12,
        "type": "box",
        "color_normal": "blue",
        "color_touched": "white",
        "note": "미",
        "freq": 659.25,   # E5
    },
    "yellow_ball": {
        "position": [0.0, 1.3, -0.45],
        "size": 0.07,
        "type": "sphere",
        "color_normal": "yellow",
        "color_touched": "white",
        "note": "솔",
        "freq": 783.99,   # G5
    },
    "purple_ball": {
        "position": [0.0, 0.9, -0.4],
        "size": 0.07,
        "type": "sphere",
        "color_normal": "magenta",
        "color_touched": "white",
        "note": "라",
        "freq": 880.00,   # A5
    },
}

TOUCH_DISTANCE = 0.12  # 접촉 임계값 12cm

# 사운드 파일 경로 (시작 시 생성)
SOUNDS = {}  # {"red_ball": {"touch": path, "release": path}, ...}


def prepare_sounds():
    """모든 물체의 접촉/해제 사운드를 미리 WAV로 생성"""
    os.makedirs(SOUND_DIR, exist_ok=True)
    for name, info in OBJECTS.items():
        touch_path = os.path.join(SOUND_DIR, f"{name}_touch.wav")
        release_path = os.path.join(SOUND_DIR, f"{name}_release.wav")
        # 접촉: 그 물체의 음을 밝게
        make_tone_wav(touch_path, info["freq"], duration=0.18, volume=0.55)
        # 해제: 한 옥타브 아래, 더 짧고 작게
        make_tone_wav(release_path, info["freq"] / 2, duration=0.10, volume=0.30)
        SOUNDS[name] = {"touch": touch_path, "release": release_path}


# ==================================================
# 시작 배너
# ==================================================
print("=" * 64)
print("🔔 Quest 3 손 인터랙션 + 사운드 (실로폰 모드)")
print("=" * 64)
if HAS_CERT:
    print(f"🌐 Quest 3 접속 주소:")
    print(f"   https://vuer.ai?ws=wss://{PC_IP}:8012&grid=False")
else:
    print("⚠️  인증서(cert.pem/key.pem)가 없습니다! 아래로 생성:")
    print(f"   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\")
    print(f"     -keyout key.pem -out cert.pem -subj '/CN={PC_IP}'")
if PLAYER:
    print(f"🔊 사운드 재생기: {PLAYER[0]} (PC 스피커에서 재생)")
else:
    print("🔇 aplay/paplay 없음 — 사운드 없이 색 변경만 동작합니다")
print("🎵 음계 배치: 빨강=도 초록=레 파랑=미 노랑=솔 보라=라")
print("=" * 64)
print("")

prepare_sounds()

# ==================================================
# Vuer 앱 (HTTPS/wss)
# ==================================================
if HAS_CERT:
    app = Vuer(host="0.0.0.0", port=8012, cors="*", cert=CERT, key=KEY)
else:
    app = Vuer(host="0.0.0.0", port=8012, cors="*")


# ==================================================
# 손 데이터 디코딩 (검증 확정 로직)
# ==================================================
def decode_hand(ext):
    """msgpack ExtType → (25, 16) numpy. 워밍업 빈 값이면 None"""
    if ext is None:
        return None
    if isinstance(ext, np.ndarray):
        arr = ext.astype("float32").ravel()
    elif isinstance(ext, (list, tuple)):
        if len(ext) < 16:
            return None
        arr = np.array(ext, dtype="float32").ravel()
    else:
        raw = getattr(ext, "data", None)
        if raw is None or not hasattr(raw, "__len__") or len(raw) <= 2:
            return None  # 워밍업 빈 값(0x00)
        arr = np.frombuffer(bytes(raw), dtype="<f4")
    if arr.size < 16 or arr.size % 16 != 0:
        return None
    return arr.reshape(-1, 16)


def joint_xyz(mats, idx):
    """관절 idx의 (x, y, z). 4x4 행렬 offset [12, 13, 14]"""
    if idx >= mats.shape[0]:
        return None
    m = mats[idx]
    return [float(m[12]), float(m[13]), float(m[14])]


def get_hand_points(value):
    """양손의 주요 관절(손목0, 엄지끝4, 검지끝9, 중지끝14) 위치 추출"""
    points = []
    for side in ("right", "left"):
        mats = decode_hand(value.get(side))
        if mats is None:
            continue
        for j in (0, 4, 9, 14):
            p = joint_xyz(mats, j)
            if p is not None:
                points.append(p)
    return points


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )


def create_object(name, info, is_touched=False):
    color = info["color_touched"] if is_touched else info["color_normal"]
    if info["type"] == "sphere":
        return Sphere(
            args=[info["size"], 32, 32],
            position=info["position"],
            material=dict(color=color),
            key=name,
        )
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
tracking_started = [False]
camera_first = [False]


# ==================================================
# 손 이벤트 핸들러
#   ★★★ 반드시 @app.spawn 보다 먼저 등록! ★★★
# ==================================================
@app.add_handler("HAND_MOVE")
async def on_hand(event, session):
    global touched_objects

    value = event.value
    if not isinstance(value, dict):
        return

    hand_points = get_hand_points(value)
    if not hand_points:
        return

    if not tracking_started[0]:
        tracking_started[0] = True
        print("🎯 손 트래킹 시작! (25관절 수신)")

    # 접촉 판정
    new_touched = set()
    for name, info in OBJECTS.items():
        for hp in hand_points:
            if distance(hp, info["position"]) < TOUCH_DISTANCE:
                new_touched.add(name)
                break

    just_touched = new_touched - touched_objects
    just_released = touched_objects - new_touched

    # 접촉: 색 변경 + 그 물체의 음 재생 🔔
    for name in just_touched:
        info = OBJECTS[name]
        print(f"✋ 접촉: {name} → ♪ {info['note']} ({info['freq']:.0f}Hz)")
        session.update @ create_object(name, info, is_touched=True)
        play_sound(SOUNDS[name]["touch"])

    # 해제: 원래 색 + 낮은 음 짧게
    for name in just_released:
        info = OBJECTS[name]
        print(f"👋 해제: {name}")
        session.update @ create_object(name, info, is_touched=False)
        play_sound(SOUNDS[name]["release"])

    touched_objects = new_touched


@app.add_handler("CAMERA_MOVE")
async def on_camera(event, session):
    if not camera_first[0]:
        camera_first[0] = True
        print("📷 VR 카메라 활성화!")


# ==================================================
# 메인 — 반드시 모든 핸들러 아래, 파일 마지막 데코레이터!
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

    print("👀 5개 물체 = 5개 음계 (실로폰처럼 연주해 보세요!)")
    print("   빨강=도  초록=레  파랑=미  노랑=솔  보라=라")
    print("✋ 'Virtual Reality' 진입 → 컨트롤러 내려놓고 양손 들기")
    print("   ⏳ 접속 직후 몇 초는 워밍업(빈 값)입니다. 손을 계속 보여주세요.")
    print("")

    counter = 0
    while True:
        await asyncio.sleep(10)
        counter += 10
        status = "트래킹 ON" if tracking_started[0] else "손 대기 중"
        print(f"⏱️ {counter}초 | {status} | 접촉: {len(touched_objects)}개")


if __name__ == "__main__":
    print("⏳ 서버 시작...")
    app.run()
