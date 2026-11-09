@echo off
title Installing StoryStudio
echo Installing StoryStudio. Leave this window open until it says StoryStudio is installed.
echo The first time takes a while (large downloads).
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
echo.
pause
