@echo off
echo ========================================
echo   CAF Dashboard - Instalacion Windows
echo ========================================
echo.

echo Instalando dependencias principales...
pip install streamlit>=1.36.0
pip install pandas>=2.2.0
pip install numpy>=1.26.0
pip install plotly>=5.22.0
pip install scipy>=1.13.0
pip install scikit-learn>=1.5.0

echo.
echo Instalando dependencias opcionales...
pip install statsmodels>=0.14.0
pip install pingouin>=0.5.3
pip install dcor>=0.5.3
pip install umap-learn>=0.5.3
pip install networkx>=3.0
pip install ruptures>=1.1.5
pip install prophet>=1.1.4
pip install shap>=0.42.0
pip install prince>=0.6.0
pip install folium>=0.15.0

echo.
echo Intentando instalar hdbscan (opcional)...
pip install hdbscan>=0.8.28
if %errorlevel% neq 0 (
    echo.
    echo WARNING: hdbscan no se pudo instalar.
    echo Para instalarlo, necesitas Microsoft Visual C++ Build Tools.
    echo Descargalos desde: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    echo El dashboard funcionara sin hdbscan, pero algunas funciones de clustering avanzado no estaran disponibles.
    echo.
) else (
    echo hdbscan instalado exitosamente!
)

echo.
echo ========================================
echo   Instalacion completada!
echo ========================================
echo.
echo Para ejecutar el dashboard:
echo   streamlit run app.py
echo.
pause
