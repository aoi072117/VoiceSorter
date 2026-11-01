@echo off
chcp 65001 >nul
cd /d "%~dp0"
title VoiceSorter
"%~dp0python\python.exe" -m voicesorter serve
if errorlevel 1 pause
