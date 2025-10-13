@echo off
REM Cleanup script for Windows - Prepare repository for GitHub

echo 🧹 Cleaning up repository for GitHub...
echo.

REM Remove data files
echo → Removing data files...
del /F /Q *.csv 2>nul
rmdir /S /Q data\images 2>nul
if not exist data mkdir data

REM Remove model files
echo → Removing model files...
rmdir /S /Q models 2>nul
mkdir models

REM Remove logs
echo → Removing logs...
del /F /Q *.log 2>nul

REM Remove Python cache
echo → Removing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
del /S /Q *.pyc 2>nul
del /S /Q *.pyo 2>nul

REM Remove any existing venv
echo → Removing old virtual environments...
rmdir /S /Q venv 2>nul
rmdir /S /Q env 2>nul
rmdir /S /Q .venv 2>nul
rmdir /S /Q ENV 2>nul

REM Remove temporary files
echo → Removing temporary files...
rmdir /S /Q temp 2>nul
rmdir /S /Q tmp 2>nul
rmdir /S /Q backup 2>nul
del /F /Q *.tmp 2>nul
del /F /Q *.temp 2>nul
del /F /Q *.bak 2>nul

REM Remove system files
echo → Removing system files...
del /S /Q Thumbs.db 2>nul
del /S /Q .DS_Store 2>nul

echo.
echo ✅ Cleanup complete!
echo.
echo 📁 Current directory contents:
dir
echo.
echo Next steps:
echo 1. Create venv: python -m venv venv
echo 2. Activate: venv\Scripts\activate
echo 3. Install deps: pip install -r requirements.txt
echo 4. Initialize git: git init
echo 5. Add files: git add .
echo 6. Commit: git commit -m "Initial commit"
echo.
pause
