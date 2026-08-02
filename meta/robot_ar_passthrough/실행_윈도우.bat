@echo off
chcp 65001 >nul
echo 패스스루 AR 서버를 시작합니다...
cd /d "%~dp0"
python server.py 2>nul || py server.py
pause
