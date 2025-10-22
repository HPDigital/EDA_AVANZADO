@echo off
REM Script de inicio para CAF Dashboard con Tkinter (Windows)

echo ========================================
echo CAF Dashboard - Version Tkinter
echo ========================================
echo.

REM Activar entorno virtual si existe
if exist venv\Scripts\activate.bat (
    echo Activando entorno virtual...
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    echo Activando entorno virtual...
    call .venv\Scripts\activate.bat
) else (
    echo Advertencia: No se encontro entorno virtual
)

REM Verificar instalacion de Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python no esta instalado o no esta en el PATH
    pause
    exit /b 1
)

echo.
echo Iniciando CAF Dashboard...
echo.

REM Ejecutar aplicacion
python app_tkinter.py

if errorlevel 1 (
    echo.
    echo Error al ejecutar la aplicacion
    echo Verifique que las dependencias esten instaladas:
    echo   pip install -r requirements-tkinter.txt
    echo.
    pause
)

