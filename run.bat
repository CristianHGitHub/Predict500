@echo off
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Running Bull or Bear Predictor...
python bull_or_bear_predictor.py

echo.
if exist plots (
    echo Opening results directory...
    start "" "plots"
) else (
    echo No plots directory found. Check for errors in script execution.
)

echo.
echo Process completed! Press any key to exit...
pause >nul