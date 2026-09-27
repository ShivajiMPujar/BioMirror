@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  BioMirror — Windows Setup & Run Script
REM  Compatible: Windows 10/11, Python 3.10-3.12, CPU-only
REM  Usage: Double-click setup_windows.bat OR run from cmd/PowerShell
REM ═══════════════════════════════════════════════════════════════════

title BioMirror Setup

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║         BioMirror — Physics-Informed Digital Twin        ║
echo  ║         Windows Setup Script                             ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

REM ── 1. Check Python ─────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Download from https://python.org
    echo         Install Python 3.11, check "Add to PATH"
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER% found

REM ── 2. Create virtual environment ──────────────────────────────────
if not exist "venv\" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [OK] venv created
) else (
    echo [OK] venv already exists
)

REM ── 3. Activate venv ────────────────────────────────────────────────
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM ── 4. Upgrade pip ──────────────────────────────────────────────────
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded

REM ── 5. Install PyTorch CPU (no CUDA required) ───────────────────────
echo [INFO] Installing PyTorch (CPU build, ~180 MB)...
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu ^
    --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 (
    echo [WARN] PyTorch install failed. Continuing without it.
    echo        Feature engineering, evaluation, and reversal engine still work.
) else (
    echo [OK] PyTorch CPU installed
)

REM ── 6. Install remaining dependencies ───────────────────────────────
echo [INFO] Installing project dependencies...
pip install numpy pandas scikit-learn scipy matplotlib ^
    fastapi uvicorn[standard] python-multipart websockets ^
    python-jose[cryptography] passlib[bcrypt] bcrypt==4.0.1 pydantic==2.7.1 ^
    pydantic-settings httpx sqlalchemy aiosqlite optuna anyio --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)
echo [OK] All dependencies installed

REM ── 7. Copy dataset to working directory ────────────────────────────
if not exist "data\diabetes_lifestyle_dataset_500.csv" (
    echo [WARN] Dataset CSV not found at data\diabetes_lifestyle_dataset_500.csv
) else (
    echo [OK] Dataset found
)

REM ── 8. Run test suite ────────────────────────────────────────────────
echo.
echo [INFO] Running test suite (48 tests)...
python ai\tests.py
if errorlevel 1 (
    echo [WARN] Some tests failed. Check output above.
) else (
    echo [OK] All tests passed
)

REM ── 9. Quick pipeline run ────────────────────────────────────────────
echo.
echo [INFO] Running quick evaluation pipeline...
python ai\train.py --quick
echo [OK] Pipeline complete

REM ── 10. Start backend ────────────────────────────────────────────────
echo.
echo  ════════════════════════════════════════
echo   SETUP COMPLETE — Choose what to run:
echo  ════════════════════════════════════════
echo.
echo   [1] Start Backend API (http://localhost:8000)
echo   [2] Open App (frontend\biomirror_app.html)
echo   [3] Run Full Training Pipeline
echo   [4] Run Evaluation + Plots
echo   [5] Run Federated Learning Simulation
echo   [6] Exit
echo.
set /p choice="Enter choice (1-6): "

if "%choice%"=="1" (
    echo Starting FastAPI backend on http://localhost:8000
    echo API docs: http://localhost:8000/docs
    echo Press Ctrl+C to stop
    python -m uvicorn backend.backend_api:app --host 0.0.0.0 --port 8000 --reload
)
if "%choice%"=="2" (
    echo Opening app in default browser...
    start frontend\biomirror_app.html
)
if "%choice%"=="3" (
    echo Running full training pipeline (requires PyTorch)...
    python ai\train.py --full --epochs 50
)
if "%choice%"=="4" (
    echo Running model evaluation + generating plots...
    python ai\evaluation_metrics.py --data data\diabetes_lifestyle_dataset_500.csv --output outputs
    echo Plots saved to biomirror_evaluation/plots/
)
if "%choice%"=="5" (
    echo Running Federated Learning simulation...
    python ai\federated_learning.py --mode simulate --rounds 20
)
if "%choice%"=="6" (
    echo Goodbye!
)

pause
