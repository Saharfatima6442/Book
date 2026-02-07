@echo off
echo Starting Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

echo Waiting for Docker to start...
timeout /t 10 /nobreak >nul

echo Checking Docker status...
docker info

if %ERRORLEVEL% EQU 0 (
    echo Docker is running. Building and starting the application...
    cd /d "C:\Users\Saeed\OneDrive\Desktop\Book"
    docker-compose up --build
) else (
    echo Docker is not running. Please start Docker Desktop manually and then run this script again.
    pause
)