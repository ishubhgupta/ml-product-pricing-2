@echo off
REM Setup script for Python 3.10 virtual environment
REM Run this to create a fresh Python 3.10 environment

echo ========================================
echo Python 3.10 Environment Setup
echo ========================================
echo.

REM Check if Python 3.10 is available
py -3.10 --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python 3.10 not found!
    echo.
    echo Please install Python 3.10 from:
    echo https://www.python.org/downloads/release/python-31011/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    echo or install to C:\Python310
    echo.
    pause
    exit /b 1
)

echo Found Python 3.10
py -3.10 --version
echo.

REM Remove old virtual environment
if exist venv (
    echo Removing old virtual environment...
    rmdir /s /q venv
    echo Old environment removed.
    echo.
)

REM Create new virtual environment
echo Creating new virtual environment with Python 3.10...
py -3.10 -m venv venv

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created successfully.
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo Installing project dependencies...
echo This may take a few minutes...
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Python version in virtual environment:
python --version
echo.
echo To activate this environment in the future, run:
echo   venv\Scripts\activate
echo.
echo To verify the setup, run:
echo   python test_setup.py
echo.
echo To start training, run:
echo   python train.py --config config.yaml
echo.
pause
