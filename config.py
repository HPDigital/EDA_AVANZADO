"""
Configuración centralizada para el dashboard CAF
"""
from pathlib import Path

# Rutas
PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_DATA_DIR = str((PROJECT_ROOT / "data" / "raw").resolve())

# Archivos de datos
CAF_AGE_FILE = "rsa_ppa_s_agg_age.csv"
CAF_FAM_FILE = "rsa_ppa_s_agg_sitfam.csv"

# Configuración de visualización
THEME_TEMPLATE = "plotly"

# Configuración de Plotly para evitar advertencias
PLOTLY_CONFIG = {
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

# Configuración de análisis
ANALYSIS_CONFIG = {
    'min_observations': 10,
    'max_variables_display': 20,
    'correlation_threshold': 0.8,
    'outlier_threshold': 1.5,
    'confidence_level': 0.95,
    'max_clusters': 10,
    'test_size_default': 0.2,
    'cv_folds_default': 5
}

# Configuración de reportes
REPORT_CONFIG = {
    'max_chart_width': 800,
    'max_chart_height': 600,
    'default_chart_format': 'PNG',
    'include_metadata': True,
    'include_recommendations': True,
    'max_recommendations': 10
}

# Configuración de performance
PERFORMANCE_CONFIG = {
    'enable_caching': True,
    'cache_ttl': 3600,  # 1 hora
    'max_dataframe_size': 1000000,  # 1M filas
    'chunk_size': 10000,
    'enable_parallel_processing': True
}

# Configuración de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/caf_dashboard.log',
    'max_size': 10485760,  # 10MB
    'backup_count': 5
}

# Configuración de mantenimiento
MAINTENANCE_CONFIG = {
    'enabled': True,
    'auto_cleanup': True,
    'backup_retention_days': 30,
    'log_retention_days': 7,
    'performance_monitoring': True,
    'alert_thresholds': {
        'disk_usage_percent': 90,
        'memory_usage_percent': 80,
        'cpu_usage_percent': 90
    }
}

# Configuración de seguridad
SECURITY_CONFIG = {
    'enable_authentication': False,
    'session_timeout': 3600,  # 1 hora
    'max_upload_size': 100 * 1024 * 1024,  # 100MB
    'allowed_file_types': ['.csv', '.xlsx', '.json'],
    'rate_limiting': {
        'enabled': True,
        'requests_per_minute': 60
    }
}

# Mapeos de datos
PRESTATION_MAP = {
    "RSA seul": "RSA", "RSA SEUL": "RSA",
    "PA seule": "PA", "PA SEULE": "PA",
    "Prime d'activité": "PA", "PRIME D'ACTIVITE": "PA",
    "RSA et PA": "RSA_ET_PA", "RSA ET PA": "RSA_ET_PA"
}

AGE_MAP = {
    "Moins de 25 ans": "<25",
    "Entre 25 et 29 ans": "25-29",
    "Entre 30 et 34 ans": "30-34",
    "Entre 35 et 39 ans": "35-39",
    "Entre 40 et 44 ans": "40-44",
    "Entre 45 et 49 ans": "45-49",
    "Entre 50 et 54 ans": "50-54",
    "Entre 55 et 59 ans": "55-59",
    "Entre 60 et 64 ans": "60-64",
    "65 ans ou plus": "65+",
    "Age indéterminé": "Indéterminé"
}

# Verificación de dependencias opcionales
def check_dependencies():
    """Retorna dict con dependencias disponibles"""
    deps = {}
    
    try:
        import statsmodels
        deps['statsmodels'] = True
    except ImportError:
        deps['statsmodels'] = False
    
    try:
        import dcor
        deps['dcor'] = True
    except ImportError:
        deps['dcor'] = False
    
    try:
        import umap
        deps['umap'] = True
    except ImportError:
        deps['umap'] = False
    
    try:
        import hdbscan
        deps['hdbscan'] = True
    except ImportError:
        deps['hdbscan'] = False
    
    try:
        import pingouin
        deps['pingouin'] = True
    except ImportError:
        deps['pingouin'] = False
    
    try:
        from prophet import Prophet
        deps['prophet'] = True
    except ImportError:
        deps['prophet'] = False
    
    try:
        import ruptures
        deps['ruptures'] = True
    except ImportError:
        deps['ruptures'] = False
    
    try:
        import folium
        deps['folium'] = True
    except ImportError:
        deps['folium'] = False
    
    try:
        import shap
        deps['shap'] = True
    except ImportError:
        deps['shap'] = False
    
    return deps

DEPENDENCIES = check_dependencies()