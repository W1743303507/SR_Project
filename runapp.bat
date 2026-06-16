@echo off
cd /d D:\SR_Project

start "SR Streamlit Server" /min cmd /k "call D:\Miniconda\Scripts\activate.bat sr_project && python -m streamlit run app\streamlit_app.py --server.headless=true --server.fileWatcherType=none --server.address=127.0.0.1 --server.port=8505"

timeout /t 5 /nobreak >nul
start "" http://127.0.0.1:8505

exit