"""
quest3_bts_reaction.py
BTS 배경음악 + 실제 콘서트 환호 리액션 버튼 4개! 🎤

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
동작:
    - 서버 시작과 함께 bts.mp4 가 배경음악으로 반복 재생 (PC 스피커)
    - 눈앞의 응원 버튼 4개를 치면 → 멕시코 콘서트 리액션 영상에서
      잘라낸 '진짜 환호 소리'가 배경음악 위에 터집니다!
        🔴 환호1 (도입 함성)      🟠 환호2 (중반 환호)
        🔵 환호3 (떼창 대합창)    🟣 환호4 (최고 폭발)

준비물 (모두 이 스크립트와 같은 폴더 ~/quest3_v310 에):
    - bts.mp4                  ← 배경음악
    - cheers/cheer1.wav ~ cheer4.wav   ← 환호 클립 4개 (제공됨)
    - cert.pem / key.pem       ← HTTPS 인증서

핵심 주의사항 (검증 완료):
    1. 모든 @app.add_handler 는 @app.spawn 보다 반드시 위에!
    2. 접속: https://vuer.ai?ws=wss://<PC_IP>:8012&grid=False
       (먼저 https://<PC_IP>:8012 에서 인증서 승인!)
    3. 종료는 Ctrl+C (배경음악도 함께 자동 종료)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import math
import atexit
import shutil
import socket
import asyncio
import subprocess
import numpy as np
from vuer import Vuer
from vuer.schemas import Scene, Box, AmbientLight, DirectionalLight, Hands


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

BGM_FILE = "./bts.mp4"
BGM_VOLUME = 40                  # 배경음악 볼륨(0~100), 환호가 잘 들리게 중간


# ==================================================
# 배경음악(BGM)
# ==================================================
_bgm_proc = [None]


def find_bgm_player():
    """mp4 반복 재생 가능한 플레이어 자동 탐색"""
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-loglevel", "quiet",
                "-loop", "0", "-volume", str(BGM_VOLUME)]
    if shutil.which("mpv"):
        return ["mpv", "--no-video", "--really-quiet",
                "--loop=inf", f"--volume={BGM_VOLUME}"]
    if shutil.which("cvlc"):
        return ["cvlc", "--quiet", "--loop", "--no-video"]
    return None


def start_bgm():
    if not os.path.exists(BGM_FILE):
        print(f"🔇 배경음악 파일 없음: {BGM_FILE}")
        return False
    player = find_bgm_player()
    if player is None:
        print("🔇 mp4 재생기 없음 — 설치: sudo apt install ffmpeg")
        return False
    try:
        _bgm_proc[0] = subprocess.Popen(
            player + [BGM_FILE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"🎵 배경음악 시작: {BGM_FILE} ({player[0]}, 반복)")
        return True
    except Exception as e:
        print(f"🔇 배경음악 실패: {e}")
        return False


def stop_bgm():
    p = _bgm_proc[0]
    if p is not None and p.poll() is None:
        p.terminate()


atexit.register(stop_bgm)


# ==================================================
# 환호 클립 (리액션 영상에서 잘라낸 실제 소리)
# ==================================================
CHEER_DIR = "./cheers"

SFX_PLAYER = None
for cand in (["paplay"], ["aplay", "-q"]):
    if shutil.which(cand[0]):
        SFX_PLAYER = cand
        break


def play_sfx(path):
    """환호 클립 비동기 재생 (배경음악 위에 겹침)"""
    if SFX_PLAYER is None or not os.path.exists(path):
        return
    try:
        subprocess.Popen(
            SFX_PLAYER + [path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


# ==================================================
# 응원 버튼 4개 — 실제 환호 클립 연결
# ==================================================
BUTTONS = {
    "btn_cheer1": {
        "position": [-0.45, 1.05, -0.45],
        "color": "red",
        "label": "🔴 환호1 (도입 함성)",
        "wav": os.path.join(CHEER_DIR, "cheer1.wav"),
    },
    "btn_cheer2": {
        "position": [-0.15, 1.05, -0.5],
        "color": "orange",
        "label": "🟠 환호2 (중반 환호)",
        "wav": os.path.join(CHEER_DIR, "cheer2.wav"),
    },
    "btn_cheer3": {
        "position": [0.15, 1.05, -0.5],
        "color": "deepskyblue",
        "label": "🔵 환호3 (떼창 대합창)",
        "wav": os.path.join(CHEER_DIR, "cheer3.wav"),
    },
    "btn_cheer4": {
        "position": [0.45, 1.05, -0.45],
        "color": "mediumpurple",
        "label": "🟣 환호4 (최고 폭발)",
        "wav": os.path.join(CHEER_DIR, "cheer4.wav"),
    },
}

BUTTON_SIZE = [0.2, 0.2, 0.06]
FLASH_COLOR = "white"
TOUCH_DISTANCE = 0.15


# ==================================================
# 시작 배너 + 클립 확인
# ==================================================
print("=" * 64)
print("🎤 BTS 콘서트 리액션 모드 — 진짜 환호로 응원하세요!")
print("=" * 64)
if HAS_CERT:
    print(f"🌐 Quest 3 접속 주소:")
    print(f"   https://vuer.ai?ws=wss://{PC_IP}:8012&grid=False")
    print(f"   (첫 접속 전 https://{PC_IP}:8012 에서 인증서 승인!)")
else:
    print("⚠️  인증서(cert.pem/key.pem)가 없습니다! 아래로 생성:")
    print(f"   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\")
    print(f"     -keyout key.pem -out cert.pem -subj '/CN={PC_IP}'")
print("")

# 환호 클립 존재 확인
missing = [info["wav"] for info in BUTTONS.values() if not os.path.exists(info["wav"])]
if missing:
    print("⚠️  환호 클립이 없습니다! cheers 폴더를 이 위치에 복사하세요:")
    for m in missing:
        print(f"   없음: {m}")
else:
    print("✅ 환호 클립 4개 확인 완료 (cheers/cheer1~4.wav)")

start_bgm()

# 사운드 자가 테스트: 시작하자마자 환호1이 한 번 터져야 정상
if SFX_PLAYER and not missing:
    play_sfx(BUTTONS["btn_cheer1"]["wav"])
    print(f"🔉 방금 '와아~' 환호가 났어야 합니다! ({SFX_PLAYER[0]})")
elif SFX_PLAYER is None:
    print("🔇 paplay/aplay 없음 — 소리 없이 색 변경만 동작")
print("")
print("🕹️  버튼 (왼쪽부터): 🔴환호1  🟠환호2  🔵환호3  🟣환호4")
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
            return None
        arr = np.frombuffer(bytes(raw), dtype="<f4")
    if arr.size < 16 or arr.size % 16 != 0:
        return None
    return arr.reshape(-1, 16)


def joint_xyz(mats, idx):
    if idx >= mats.shape[0]:
        return None
    m = mats[idx]
    return [float(m[12]), float(m[13]), float(m[14])]


def get_hand_points(value):
    """양손 주요 관절(손목0, 엄지끝4, 검지끝9, 중지끝14) 위치"""
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


def create_button(name, color):
    info = BUTTONS[name]
    return Box(
        args=BUTTON_SIZE,
        position=info["position"],
        material=dict(color=color),
        key=name,
    )


# ==================================================
# 상태
# ==================================================
touched_now = set()
hit_counts = {name: 0 for name in BUTTONS}
tracking_started = [False]
camera_first = [False]


async def flash_button(session, name):
    """타격한 버튼 하얗게 번쩍 → 0.15초 후 원래 색"""
    session.update @ create_button(name, FLASH_COLOR)
    await asyncio.sleep(0.15)
    session.update @ create_button(name, BUTTONS[name]["color"])


# ==================================================
# 손 이벤트 핸들러
#   ★★★ 반드시 @app.spawn 보다 먼저 등록! ★★★
# ==================================================
@app.add_handler("HAND_MOVE")
async def on_hand(event, session):
    global touched_now

    value = event.value
    if not isinstance(value, dict):
        return

    hand_points = get_hand_points(value)
    if not hand_points:
        return

    if not tracking_started[0]:
        tracking_started[0] = True
        print("🎯 손 트래킹 시작! 버튼을 쳐서 응원하세요 🎤")

    new_touched = set()
    for name, info in BUTTONS.items():
        for hp in hand_points:
            if distance(hp, info["position"]) < TOUCH_DISTANCE:
                new_touched.add(name)
                break

    hits = new_touched - touched_now
    touched_now = new_touched

    for name in hits:
        info = BUTTONS[name]
        hit_counts[name] += 1
        print(f"{info['label']}  x{hit_counts[name]}")
        play_sfx(info["wav"])
        asyncio.create_task(flash_button(session, name))


@app.add_handler("CAMERA_MOVE")
async def on_camera(event, session):
    if not camera_first[0]:
        camera_first[0] = True
        print("📷 VR 카메라 활성화!")


# ==================================================
# 메인 — 반드시 모든 핸들러 아래, 파일 마지막!
# ==================================================
@app.spawn(start=True)
async def main(session):
    print("✅ 브라우저 접속됨! 응원 버튼 세팅 중...")

    session.set @ Scene(
        AmbientLight(intensity=1.0, key="ambient"),
        DirectionalLight(intensity=1.5, position=[3, 5, 3], key="sun"),
        Hands(fps=30, stream=True, key="hands", left=True, right=True),
        *[create_button(name, info["color"]) for name, info in BUTTONS.items()],
        up=[0, 1, 0],
    )

    print("🕹️  버튼 4개 배치 완료 (허리 높이)")
    print("   🔴환호1  🟠환호2  🔵환호3  🟣환호4")
    print("✋ 'Virtual Reality' 진입 → 컨트롤러 내려놓고 → 마음껏 응원!")
    print("   ⏳ 접속 직후 몇 초는 워밍업입니다. 손을 계속 보여주세요.")
    print("")

    counter = 0
    while True:
        await asyncio.sleep(10)
        counter += 10
        status = "트래킹 ON" if tracking_started[0] else "손 대기 중"
        total = sum(hit_counts.values())
        print(f"⏱️ {counter}초 | {status} | 총 응원 {total}회")


if __name__ == "__main__":
    print("⏳ 서버 시작...")
    app.run()
