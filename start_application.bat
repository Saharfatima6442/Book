@echo off
echo Starting AI Book Application...
echo.

echo Setting up backend...
cd /d "C:\Users\Saeed\OneDrive\Desktop\Book\backend"
echo Installing backend dependencies...
pip install -r requirements.txt > nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Backend dependencies installed successfully.
) else (
    echo Warning: Issue with installing backend dependencies
)

echo.
echo Starting backend server on port 8000...
start /min cmd /c "cd /d \"C:\Users\Saeed\OneDrive\Desktop\Book\backend\" && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

echo.
echo Setting up frontend...
cd /d "C:\Users\Saeed\OneDrive\Desktop\Book\AI-Book"
echo Installing frontend dependencies...
npm install > nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Frontend dependencies installed successfully.
) else (
    echo Warning: Issue with installing frontend dependencies
)

echo.
echo Starting frontend server on port 3000...
start /min cmd /c "cd /d \"C:\Users\Saeed\OneDrive\Desktop\Book\AI-Book\" && npm start"

echo.
echo.
echo The application is now starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit this window.
pause > nul