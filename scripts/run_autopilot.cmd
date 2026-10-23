@echo off
REM ============================================================================
REM  M1 训练托管 —— 一键恢复入口
REM
REM  什么时候用：
REM    托管进程意外中断（例如关掉了 WorkBuddy 客户端，子进程随之被清理）。
REM    双击本文件即可续跑 —— autopilot 会自己读 models/_ckpts/ 里最新存档
REM    与 reports/m1_sN_history.json，算出还剩多少轮，**不会重头再来**。
REM
REM  注意：本窗口要一直开着。关掉窗口 = 训练停止。
REM        进度看 reports\_m1_s1.out；结论看 reports\AUTOPILOT_REPORT.md
REM ============================================================================
setlocal
set PY=C:\Users\LIGF\.workbuddy\binaries\python\envs\zhuoji\Scripts\python.exe
cd /d "%~dp0.."

echo ============================================
echo  M1 训练托管恢复（S1 剩余 -^> S2 -^> S3）
echo  工作目录: %CD%
echo ============================================
echo.

"%PY%" scripts\m1_autopilot.py --from s1 --to s3 --workers 12

echo.
echo 托管已退出（退出码 %ERRORLEVEL%）。
echo   训练进度: reports\_m1_s1.out
echo   阶段报告: reports\AUTOPILOT_REPORT.md
echo.
pause >nul
