#!/usr/bin/env python3
# 로컬 HTTPS 서버 - WebXR(패스스루 AR)은 HTTPS가 필수라서 필요합니다.
# 인증서가 없으면 자동 생성합니다.
import http.server, ssl, socket, os, subprocess, sys

PORT = 8443
CERT, KEY = "cert.pem", "key.pem"

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return "127.0.0.1"

def ensure_cert(ip):
    if os.path.exists(CERT) and os.path.exists(KEY):
        print("기존 인증서 재사용:", CERT, KEY); return
    print("인증서 생성 중... (openssl)")
    # openssl로 자체서명 인증서 생성 (IP를 SAN에 포함)
    san = f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"
    cmd = ["openssl","req","-x509","-newkey","rsa:2048","-nodes",
           "-keyout",KEY,"-out",CERT,"-days","365",
           "-subj","/CN=robot-ar","-addext",san]
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        print("인증서 생성 완료")
    except Exception as e:
        print("⚠ openssl 인증서 생성 실패:", e)
        print("  openssl이 설치되어 있는지 확인하세요.")
        sys.exit(1)

ip = get_ip()
ensure_cert(ip)

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(CERT, KEY)

httpd = http.server.HTTPServer(("0.0.0.0", PORT), http.server.SimpleHTTPRequestHandler)
httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

print("="*54)
print("  리틀 도쿄 AR 서버 시작됨")
print("="*54)
print(f"  PC에서 보기:   https://localhost:{PORT}")
print(f"  Quest에서 보기: https://{ip}:{PORT}")
print("="*54)
print("  Quest 3 사용법:")
print(f"   1) Quest 브라우저에서 https://{ip}:{PORT} 접속")
print("   2) '연결이 비공개가 아님' 경고 → 고급 → 계속 진행")
print("   3) 화면의 'AR 시작' 버튼 누르기")
print("   4) 내 방에 로봇이 나타남! 아래 버튼으로 조종")
print("="*54)
print("  (Ctrl+C 로 종료)")
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\n서버 종료")
