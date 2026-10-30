
@echo off
REM Khoi dong CatVTON API bang Conda CUDA environment
cd /d "%~dp0"

if not exist ".env" (
    echo Chua co file .env.
    echo Hay sao chep .env.example thanh .env va dien CATVTON_API_KEY.
    exit /b 1
)

call conda run --no-capture-output -n catvton311 python api_server.py

pause
