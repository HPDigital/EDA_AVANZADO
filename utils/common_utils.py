from __future__ import annotations
"""
Utilidades comunes para el CAF Dashboard
Incluye funciones compartidas, constantes y configuraciones
"""

import pandas as pd
import base64
import numpy as np
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Detectar si Streamlit está disponible
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    class st:
        """Dummy Streamlit para compatibilidad"""
        @staticmethod
        def columns(num): return [DummyColumn() for _ in range(num)]
        @staticmethod
        def metric(label, value): pass
        @staticmethod
        def write(msg): pass
        @staticmethod
        def info(msg): print(f"Info: {msg}")
        @staticmethod
        def warning(msg): print(f"Warning: {msg}")
        @staticmethod
        def error(msg): print(f"Error: {msg}")
        @staticmethod
        def success(msg): print(f"Success: {msg}")
        @staticmethod
        def progress(val): return DummyProgress()
        @staticmethod
        def expander(title, expanded=False): return DummyExpander()
        @staticmethod
        def code(text): print(text)
        @staticmethod
        def markdown(text, unsafe_allow_html=False): pass

class DummyColumn:
    def __enter__(self): return self
    def __exit__(self, *args): pass

class DummyProgress:
    def progress(self, val): pass

class DummyExpander:
    def __enter__(self): return self
    def __exit__(self, *args): pass

# Constantes globales
DEFAULT_CHART_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'plot',
        'height': 500,
        'width': 700,
        'scale': 1
    }
}

DEFAULT_THEME = "plotly"

# Palabras clave para detección de tipos de variables
TIME_KEYWORDS = [
    'date', 'fecha', 'time', 'tiempo', 'year', 'año', 'année', 
    'period', 'período', 'période', 'référence', 'reference',
    'réf', 'ref', 'début', 'debut', 'begin', 'fin', 'end'
]

QUANTITATIVE_KEYWORDS = [
    'count', 'total', 'sum', 'amount', 'value', 'score', 'rate', 'ratio',
    'percent', 'percentage', 'pct', 'num', 'number', 'quantity', 'size',
    'length', 'width', 'height', 'weight', 'price', 'cost', 'revenue',
    'income', 'expense', 'budget', 'profit', 'loss', 'margin', 'growth',
    'change', 'difference', 'variation', 'deviation', 'standard', 'mean',
    'average', 'median', 'mode', 'min', 'max', 'range', 'variance',
    'year', 'año', 'année', 'age', 'edad', 'id', 'code', 'numero', 'num',
    'rank', 'position', 'ordre', 'order'
]

# Funciones de utilidad comunes
def safe_float(value: Any) -> float:
    """Convierte un valor a float de forma segura"""
    try:
        return float(value) if value is not None and str(value).strip() != '' else None
    except (ValueError, TypeError):
        return None

def safe_int(value: Any) -> int:
    """Convierte un valor a int de forma segura"""
    try:
        return int(value) if value is not None and str(value).strip() != '' else None
    except (ValueError, TypeError):
        return None

def normalize_column_name(column_name: str) -> str:
    """Normaliza el nombre de una columna para mejor detección"""
    import re
    # Reemplazar guiones bajos y guiones con espacios
    normalized = column_name.lower().replace('_', ' ').replace('-', ' ')
    # Limpiar espacios múltiples
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def detect_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Detecta la calidad general de los datos"""
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isna().sum().sum()
    missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0
    
    if missing_percentage < 5:
        quality = "Excelente"
    elif missing_percentage < 15:
        quality = "Buena"
    elif missing_percentage < 30:
        quality = "Regular"
    else:
        quality = "Pobre"
    
    return {
        'quality': quality,
        'missing_percentage': missing_percentage,
        'total_cells': total_cells,
        'missing_cells': missing_cells
    }

def create_summary_metrics(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
    """Crea métricas de resumen para el dataset"""
    quality = detect_data_quality(df)
    
    return {
        'total_observations': len(df),
        'total_variables': len(df.columns),
        'numeric_variables': len(numeric_cols),
        'categorical_variables': len(categorical_cols),
        'data_quality': quality['quality'],
        'missing_percentage': quality['missing_percentage'],
        'completeness_score': 100 - quality['missing_percentage']
    }

def validate_data_for_analysis(df: pd.DataFrame, min_observations: int = 10) -> Tuple[bool, str]:
    """Valida si los datos son adecuados para análisis"""
    if len(df) < min_observations:
        return False, f"Se necesitan al menos {min_observations} observaciones para análisis"
    
    if len(df.columns) == 0:
        return False, "El dataset no tiene columnas"
    
    if df.isna().all().all():
        return False, "El dataset está completamente vacío"
    
    return True, "Datos válidos para análisis"

def get_column_type_confidence(series: pd.Series, column_name: str) -> Tuple[str, float]:
    """Determina el tipo de columna con nivel de confianza"""
    normalized_name = normalize_column_name(column_name)
    
    # Verificar si es temporal
    for keyword in TIME_KEYWORDS:
        if keyword in normalized_name:
            return 'temporal', 0.8
    
    # Verificar si es cuantitativo
    if pd.api.types.is_numeric_dtype(series):
        for keyword in QUANTITATIVE_KEYWORDS:
            if keyword in normalized_name:
                return 'quantitative', 0.9
        return 'quantitative', 0.7
    
    # Por defecto, cualitativo
    return 'qualitative', 0.8

def create_plotly_figure(title: str, x_title: str = None, y_title: str = None, theme: str = None) -> Any:
    """Crea una figura de Plotly con configuración estándar"""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template=theme or DEFAULT_THEME,
        font=dict(size=12),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

def display_metric_cards(metrics: Dict[str, Any], columns: int = 4):
    """Muestra métricas en tarjetas usando columnas de Streamlit"""
    cols = st.columns(columns)
    metric_items = list(metrics.items())
    
    for i, (key, value) in enumerate(metric_items):
        with cols[i % columns]:
            if isinstance(value, (int, float)):
                st.metric(key, f"{value:.2f}" if isinstance(value, float) else value)
            else:
                st.metric(key, str(value))

def create_correlation_heatmap(df: pd.DataFrame, method: str = 'pearson') -> Any:
    """Crea un heatmap de correlación"""
    import plotly.graph_objects as go
    
    corr_matrix = df.corr(method=method)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f"Matriz de Correlación ({method.title()})",
        template=DEFAULT_THEME,
        width=600,
        height=600
    )
    
    return fig

def generate_insights_summary(findings: List[str], alerts: List[str], recommendations: List[str]) -> Dict[str, Any]:
    """Genera un resumen de insights para reportes"""
    return {
        'findings': findings,
        'alerts': alerts,
        'recommendations': recommendations,
        'total_insights': len(findings) + len(alerts) + len(recommendations)
    }

def export_data_to_csv(df: pd.DataFrame, filename: str) -> bytes:
    """Exporta un DataFrame a CSV como bytes"""
    return df.to_csv(index=False).encode('utf-8')

def export_data_to_excel(df: pd.DataFrame, filename: str) -> bytes:
    """Exporta un DataFrame a Excel como bytes"""
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()

def create_download_button(data: bytes, filename: str, mime_type: str, button_text: str = "Descargar"):
    """Crea un botón de descarga en Streamlit"""
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{button_text}</a>'
    return st.markdown(href, unsafe_allow_html=True)

def log_analysis_step(step_name: str, success: bool = True, details: str = ""):
    """Registra un paso del análisis para debugging"""
    status = "✅" if success else "❌"
    st.write(f"{status} {step_name}")
    if details:
        st.write(f"   {details}")

def create_progress_bar(total_steps: int, current_step: int, step_name: str):
    """Crea una barra de progreso para análisis largos"""
    progress = current_step / total_steps
    st.progress(progress)
    st.write(f"Paso {current_step}/{total_steps}: {step_name}")

def validate_streamlit_widget_key(key: str, widget_type: str) -> str:
    """Valida y formatea claves de widgets de Streamlit"""
    import re
    # Limpiar la clave para que sea válida
    clean_key = re.sub(r'[^a-zA-Z0-9_]', '_', key)
    return f"{widget_type}_{clean_key}"

def create_expandable_section(title: str, content_func, expanded: bool = False):
    """Crea una sección expandible con contenido"""
    with st.expander(title, expanded=expanded):
        content_func()

def display_error_with_details(error: Exception, context: str = ""):
    """Muestra un error con detalles para debugging"""
    st.error(f"❌ Error en {context}: {str(error)}")
    with st.expander("Detalles del error"):
        import traceback
        st.code(traceback.format_exc())

def create_info_box(message: str, icon: str = "ℹ️"):
    """Crea una caja de información estilizada"""
    st.info(f"{icon} {message}")

def create_warning_box(message: str, icon: str = "⚠️"):
    """Crea una caja de advertencia estilizada"""
    st.warning(f"{icon} {message}")

def create_success_box(message: str, icon: str = "✅"):
    """Crea una caja de éxito estilizada"""
    st.success(f"{icon} {message}")

def create_error_box(message: str, icon: str = "❌"):
    """Crea una caja de error estilizada"""
    st.error(f"{icon} {message}")

# Configuración de logging
import logging

def setup_logging():
    """Configura el sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/caf_dashboard.log'),
            logging.StreamHandler()
        ]
    )

def get_logger(name: str) -> logging.Logger:
    """Obtiene un logger configurado"""
    return logging.getLogger(name)
