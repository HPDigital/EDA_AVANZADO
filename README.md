CAF Dashboard (Tkinter)

Descripción
- Aplicación de escritorio para Análisis Exploratorio de Datos (EDA) desarrollada con Tkinter.
- Visualización con Matplotlib/Seaborn (sin Streamlit ni Plotly).
- Ingesta inteligente multi‑archivo, detección de tipos, transformaciones y módulos de análisis: univariado, bivariado, multivariado, series temporales y ML básico.

Requisitos
- Python 3.12+ (Tkinter incluido en la instalación estándar)
- Dependencias: ver `requirements-tkinter.txt`

Instalación rápida
- Crear y activar entorno virtual:
  - Windows: `python -m venv venv && venv\Scripts\activate`
  - Linux/Mac: `python -m venv venv && source venv/bin/activate`
- Instalar dependencias: `pip install -r requirements-tkinter.txt`

Ejecución
- Windows: `start_tkinter.bat` o `python app_tkinter.py`
- Linux/Mac: `python app_tkinter.py`

Estructura (resumen)
- `app_tkinter.py` — Aplicación principal Tkinter
- `tkinter_ui/` — Componentes de UI (exploración, univariado, bivariado, multivariado, series, ML, reportes)
- `utils/` — Ingesta, validación, transformaciones y utilidades
- `config.py` — Configuración centralizada
- `data/` — `raw/` y `processed/` (carpetas de datos)

Datos
- Coloca CSV en `data/raw/` o selecciona un directorio desde la UI.
- La ingesta inteligente detecta separadores, codificación, encabezados, tipos y columnas temporales.

Notas
- Este repositorio está enfocado en la versión Tkinter. Scripts y módulos Streamlit/Plotly no son necesarios para ejecutar la app de escritorio.
- Directorios locales como `venv/`, `.venv/`, `logs/`, `.cache/` están ignorados para Git.
- Algunas dependencias opcionales (p. ej., `prophet`, `shap`) pueden requerir compiladores o Microsoft C++ Build Tools en Windows.
