@echo off
REM Quick start script for ML Product Pricing (Windows)

echo ==========================================
echo ML Product Pricing - Quick Start
echo ==========================================

REM Step 1: Create virtual environment
echo.
echo Step 1: Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Step 2: Activate virtual environment
echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

REM Step 3: Install dependencies
echo.
echo Step 3: Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Step 4: Run training (without optimization and images for quick start)
echo.
echo ==========================================
echo Starting Training (Quick Mode)
echo ==========================================
python train.py --config config.yaml

REM Step 5: Generate predictions
echo.
echo ==========================================
echo Generating Predictions
echo ==========================================
python predict.py --config config.yaml

echo.
echo ==========================================
echo Quick Start Completed!
echo ==========================================
echo Check test_out.csv for predictions

pause
