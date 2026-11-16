@echo off
rem music [all|sounds|arrange|check|player] [names] [--midi file.mid]  (see README)
"%~dp0.venv\Scripts\python.exe" "%~dp0music.py" %*
