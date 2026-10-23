@echo off
REM ============================================================
REM  Allow LAN devices (phone / tablet) to reach the Dushan
REM  Mahjong web server on TCP/UDP port 8770.
REM
REM  HOW TO RUN:  right-click this file -> "Run as administrator"
REM ============================================================

net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Administrator rights are required. Requesting elevation...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

echo Removing stale rules (if any)...
netsh advfirewall firewall delete rule name="DushanMahjong 8770 TCP" >nul 2>&1
netsh advfirewall firewall delete rule name="DushanMahjong 8770 UDP" >nul 2>&1

echo Adding inbound allow rules for port 8770 ...
netsh advfirewall firewall add rule name="DushanMahjong 8770 TCP" dir=in action=allow protocol=TCP localport=8770
netsh advfirewall firewall add rule name="DushanMahjong 8770 UDP" dir=in action=allow protocol=UDP localport=8770

echo.
echo Done. On your phone (same Wi-Fi), open:  http://^<this-PC-ip^>:8770
echo The PC's LAN address is printed when the server starts.
echo.
pause
