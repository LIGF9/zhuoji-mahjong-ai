@echo off
REM ============================================================================
REM  庄记麻将 M2 训练 —— 一键恢复总控（宿主侧手动运行）
REM
REM  何时用：
REM    沙箱巡检结束后，总控与训练常被连带清理，导致训练停摆。
REM    在**真实的 Windows 终端**（非沙箱）双击本文件即可恢复：
REM    autopilot 会读 reports\night_autopilot_state.json，
REM    从 current 段的 dyn_segs 配置 [latest ckpt] 无缝接力，**不会重头再来**。
REM
REM  注意：本窗口需保持开启；关掉窗口 = 训练停止。
REM        进度看 reports\_m1_auto6rrrr.out，巡检看 reports\_automation_watch.log
REM ============================================================================
setlocal
set PY=C:\Users\LIGF\.workbuddy\binaries\python\envs\zhuoji\Scripts\python.exe
cd /d "%~dp0.."

echo ============================================
echo  M2 训练总控恢复（自动接力 current 段）
echo  工作目录: %CD%
echo ============================================
echo.

"%PY%" -u scripts\m2_autopilot_guard.py --workers 12 --max-hours 20 --poll-s 60

echo.
echo 守护已退出（退出码 %ERRORLEVEL%）。
pause >nul
