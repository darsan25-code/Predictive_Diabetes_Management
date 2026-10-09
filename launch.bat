@echo off
setlocal
echo ==========================================================
echo   T1D Digital Twin Research Platform
echo   Research prototype - Not a medical device
echo ==========================================================
echo.

python --version >NUL 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add to PATH.
    pause & exit /b 1
)
node --version >NUL 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Please install Node.js 18+ and add to PATH.
    pause & exit /b 1
)

if not exist "data\processed\patient_synthetic_000.parquet" (
    echo [INFO] No processed data found. Generating synthetic demo dataset...
    python experiments\run_pipeline.py
    if %errorlevel% neq 0 (echo [WARN] Pipeline returned non-zero.) else (echo [OK] Synthetic data generated.)
) else (
    echo [OK] Processed data found.
)

if not exist "frontend\node_modules" (
    echo [INFO] Installing frontend dependencies...
    cd frontend && npm install && cd ..
)

echo.
echo Starting FastAPI backend on http://127.0.0.1:8000 ...
start "T1D Backend (FastAPI)" cmd /k "cd /d "%~dp0" && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 3 /nobreak > NUL

echo Starting React frontend on http://localhost:5173 ...
start "T1D Frontend (React)" cmd /k "cd /d "%~dp0frontend" && npm run dev"
timeout /t 5 /nobreak > NUL

echo.
echo ==========================================================
echo   Application ready!
echo   Frontend:    http://localhost:5173
echo   API Docs:    http://127.0.0.1:8000/docs
echo   API Health:  http://127.0.0.1:8000/api/health
echo ==========================================================
echo.
echo NOTE: Research prototype. Not a medical device.
echo       Not for clinical decisions. All data is synthetic.
echo.
endlocal
