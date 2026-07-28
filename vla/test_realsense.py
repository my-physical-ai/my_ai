#!/usr/bin/env python3
# test_realsense.py - pyrealsense2 설치 및 카메라 인식 테스트
# 제타위성로보틱스 Physical AI 교육 프로그램
# 실행: python test_realsense.py

import sys

# --- 1. pyrealsense2 임포트 확인 ---
try:
    import pyrealsense2 as rs
    print(f"[1] ✅ pyrealsense2 임포트 성공 (version: {rs.__version__})")
except ModuleNotFoundError:
    print("[1] ❌ pyrealsense2가 설치되지 않았습니다.")
    print("    설치: pip install pyrealsense2")
    sys.exit(1)

# --- 2. 연결된 카메라 검색 ---
ctx = rs.context()
devices = ctx.query_devices()
print(f"\n[2] 발견된 RealSense 개수: {len(devices)}")

if len(devices) == 0:
    print("    ⚠️  카메라를 찾지 못했습니다. 확인 사항:")
    print("      - USB 케이블이 USB 3.0 포트(파란색)에 꽂혀 있는가")
    print("      - 케이블이 데이터 전송용인가 (충전 전용 아님)")
    print("      - lsusb 에서 8086:0b3a 가 보이는가")
    sys.exit(1)

# --- 3. 각 카메라 상세 정보 ---
print("\n[3] 카메라 상세 정보")
print("-" * 50)
for i, dev in enumerate(devices):
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    fw = dev.get_info(rs.camera_info.firmware_version)
    usb = dev.get_info(rs.camera_info.usb_type_descriptor)
    print(f"  카메라 #{i}")
    print(f"    이름      : {name}")
    print(f"    시리얼    : {serial}   ← LeRobot 설정에 쓸 번호!")
    print(f"    펌웨어    : {fw}")
    print(f"    USB 타입  : {usb}   (3.x 여야 깊이 스트림 정상)")
    print("-" * 50)

# --- 4. 실제 스트림 열기 테스트 (한 프레임만) ---
print("\n[4] 스트림 열기 테스트 (컬러 + 깊이 각 1프레임)")
try:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    # 워밍업 (첫 프레임은 버림)
    for _ in range(5):
        frames = pipeline.wait_for_frames()

    color = frames.get_color_frame()
    depth = frames.get_depth_frame()

    print(f"    ✅ 컬러 프레임: {color.get_width()}x{color.get_height()}")
    print(f"    ✅ 깊이 프레임: {depth.get_width()}x{depth.get_height()}")

    # 중앙 픽셀의 거리 측정 (재미있는 확인)
    w, h = depth.get_width(), depth.get_height()
    dist = depth.get_distance(w // 2, h // 2)
    print(f"    📏 화면 중앙까지 거리: {dist:.3f} m")

    pipeline.stop()
    print("\n🎉 모든 테스트 통과! 카메라가 정상 작동합니다.")
except Exception as e:
    print(f"    ❌ 스트림 열기 실패: {e}")
    print("    → USB 3.0 포트 확인, 다른 프로그램이 카메라를 점유 중인지 확인")
    sys.exit(1)
