@echo off
echo ========================================
echo   DeepSeek Desktop - MODO OFFLINE
echo ========================================
echo.
echo Activar entorno virtual...
call venv\Scripts\activate

echo.
echo Iniciando aplicacion en modo offline...
echo Si no tienes internet, usa solo este archivo.
echo.

python main.py

pause