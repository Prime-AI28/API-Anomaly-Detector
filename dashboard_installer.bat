@echo off
echo Installing Python 3.10...

:: Check if running with administrative privileges
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"

:: If not running as admin, relaunch as admin
if %errorlevel% NEQ 0 (
    echo Script requires administrative privileges. Relaunching as admin...
    powershell -Command "Start-Process '%0' -Verb RunAs"
    exit /B
)

:: Download and install Python 3.10
echo Downloading Python 3.10 installer...
curl -o python-3.10.0-amd64.exe https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe

echo Installing Python 3.10...
python-3.10.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1

:: Check Python installation
echo Checking Python installation...
python --version

:: Create a virtual environment
echo Creating virtual environment...
python -m venv myenv

:: Activate the virtual environment
echo Activating virtual environment...
call myenv\Scripts\activate

:: Install dependencies from requirements.txt
echo Installing dependencies...
pip install -r requirements.txt

:: Run streamlit app
echo Running Streamlit app...
streamlit run streamlit_app.py

:: Deactivate the virtual environment
echo Deactivating virtual environment...
deactivate

echo Process complete.
