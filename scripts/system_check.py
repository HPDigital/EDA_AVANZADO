#!/usr/bin/env python3
"""
Script de verificación del sistema CAF Dashboard
Verifica la instalación, configuración y funcionamiento de todos los módulos
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

def print_header(title):
    """Imprime un encabezado formateado"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_success(message):
    """Imprime un mensaje de éxito"""
    print(f"✅ {message}")

def print_error(message):
    """Imprime un mensaje de error"""
    print(f"❌ {message}")

def print_warning(message):
    """Imprime un mensaje de advertencia"""
    print(f"⚠️  {message}")

def print_info(message):
    """Imprime un mensaje informativo"""
    print(f"ℹ️  {message}")

def check_python_version():
    """Verifica la versión de Python"""
    print_header("VERIFICACIÓN DE PYTHON")
    
    version = sys.version_info
    print_info(f"Versión de Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print_success("Versión de Python compatible")
        return True
    else:
        print_error("Se requiere Python 3.8 o superior")
        return False

def check_dependencies():
    """Verifica las dependencias principales"""
    print_header("VERIFICACIÓN DE DEPENDENCIAS")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'scipy',
        'sklearn',
        'statsmodels'
    ]
    
    optional_packages = [
        'prophet',
        'ruptures',
        'networkx',
        'umap',
        'hdbscan'
    ]
    
    all_good = True
    
    # Verificar paquetes requeridos
    print_info("Verificando paquetes requeridos...")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_success(f"{package} - OK")
        except ImportError as e:
            print_error(f"{package} - FALTA: {e}")
            all_good = False
    
    # Verificar paquetes opcionales
    print_info("Verificando paquetes opcionales...")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print_success(f"{package} - OK")
        except ImportError:
            print_warning(f"{package} - No disponible (opcional)")

    return all_good

def check_project_structure():
    """Verifica la estructura del proyecto"""
    print_header("VERIFICACIÓN DE ESTRUCTURA DEL PROYECTO")
    
    required_files = [
        'app.py',
        'config.py',
        'requirements.txt',
        'utils/__init__.py',
        'utils/common_utils.py',
        'utils/performance_utils.py',
        'utils/validation_utils.py',
        'utils/smart_ingestion.py',
        'utils/advanced_analysis.py',
        'utils/bivariate_analysis.py',
        'utils/multivariate_analysis.py',
        'modules/__init__.py',
        'modules/machine_learning.py',
        'modules/time_series.py',
        'modules/reports.py'
    ]
    
    all_good = True
    
    for file_path in required_files:
        full_path = root_dir / file_path
        if full_path.exists():
            print_success(f"{file_path} - OK")
        else:
            print_error(f"{file_path} - FALTA")
            all_good = False
    
    return all_good

def check_imports():
    """Verifica que todos los módulos se puedan importar"""
    print_header("VERIFICACIÓN DE IMPORTS")
    
    modules_to_test = [
        'config',
        'utils.common_utils',
        'utils.performance_utils',
        'utils.validation_utils',
        'utils.smart_ingestion',
        'utils.advanced_analysis',
        'utils.bivariate_analysis',
        'utils.multivariate_analysis',
        'modules.machine_learning',
        'modules.time_series',
        'modules.reports'
    ]
    
    all_good = True
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print_success(f"{module_name} - OK")
        except Exception as e:
            print_error(f"{module_name} - ERROR: {e}")
            all_good = False
    
    return all_good

def check_data_directory():
    """Verifica el directorio de datos"""
    print_header("VERIFICACIÓN DE DATOS")
    
    data_dir = root_dir / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    # Crear directorios si no existen
    data_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    
    print_success("Directorios de datos creados/verificados")
    
    # Verificar archivos de datos
    csv_files = list(raw_dir.glob('*.csv'))
    if csv_files:
        print_success(f"Encontrados {len(csv_files)} archivos CSV en data/raw/")
        for file in csv_files:
            print_info(f"  - {file.name}")
    else:
        print_warning("No se encontraron archivos CSV en data/raw/")
    
    return True

def check_logs_directory():
    """Verifica el directorio de logs"""
    print_header("VERIFICACIÓN DE LOGS")
    
    logs_dir = root_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    print_success("Directorio de logs creado/verificado")
    
    # Verificar permisos de escritura
    test_file = logs_dir / 'test.log'
    try:
        with open(test_file, 'w') as f:
            f.write("Test log entry\n")
        test_file.unlink()  # Eliminar archivo de prueba
        print_success("Permisos de escritura en logs/ - OK")
        return True
    except Exception as e:
        print_error(f"Error de escritura en logs/: {e}")
        return False

def check_configuration():
    """Verifica la configuración"""
    print_header("VERIFICACIÓN DE CONFIGURACIÓN")
    
    try:
        from config import (
            THEME_TEMPLATE, PLOTLY_CONFIG, ANALYSIS_CONFIG,
            REPORT_CONFIG, PERFORMANCE_CONFIG, LOGGING_CONFIG
        )
        
        print_success("Configuración cargada correctamente")
        
        # Verificar configuraciones específicas
        configs = {
            'THEME_TEMPLATE': THEME_TEMPLATE,
            'PLOTLY_CONFIG': PLOTLY_CONFIG,
            'ANALYSIS_CONFIG': ANALYSIS_CONFIG,
            'REPORT_CONFIG': REPORT_CONFIG,
            'PERFORMANCE_CONFIG': PERFORMANCE_CONFIG,
            'LOGGING_CONFIG': LOGGING_CONFIG
        }
        
        for name, config in configs.items():
            if config:
                print_success(f"{name} - OK")
            else:
                print_warning(f"{name} - Vacío")
        
        return True
        
    except Exception as e:
        print_error(f"Error cargando configuración: {e}")
        return False

def check_utilities():
    """Verifica las utilidades"""
    print_header("VERIFICACIÓN DE UTILIDADES")
    
    try:
        from utils.common_utils import (
            safe_float, safe_int, normalize_column_name,
            detect_data_quality, create_summary_metrics
        )
        
        from utils.performance_utils import (
            memory_optimize_dataframe, chunked_processing,
            safe_divide, safe_percentage
        )
        
        from utils.validation_utils import (
            validate_dataframe_structure, validate_data_quality,
            run_data_validation_suite
        )
        
        print_success("Utilidades comunes - OK")
        print_success("Utilidades de performance - OK")
        print_success("Utilidades de validación - OK")
        
        return True
        
    except Exception as e:
        print_error(f"Error cargando utilidades: {e}")
        return False

def check_analysis_modules():
    """Verifica los módulos de análisis"""
    print_header("VERIFICACIÓN DE MÓDULOS DE ANÁLISIS")
    
    modules = [
        ('utils.smart_ingestion', 'SmartCSVIngestion'),
        ('utils.advanced_analysis', 'display_advanced_numeric_analysis'),
        ('utils.bivariate_analysis', 'display_numeric_numeric_analysis'),
        ('utils.multivariate_analysis', 'display_advanced_multivariate_analysis'),
        ('modules.machine_learning', 'display_machine_learning_analysis'),
        ('modules.time_series', 'display_time_series_analysis'),
        ('modules.reports', 'display_reports_analysis')
    ]
    
    all_good = True
    
    for module_name, function_name in modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, function_name):
                print_success(f"{module_name}.{function_name} - OK")
            else:
                print_error(f"{module_name}.{function_name} - NO ENCONTRADO")
                all_good = False
        except Exception as e:
            print_error(f"{module_name} - ERROR: {e}")
            all_good = False
    
    return all_good

def run_basic_tests():
    """Ejecuta tests básicos"""
    print_header("TESTS BÁSICOS")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Test de DataFrame
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'text': ['hello', 'world', 'test', 'data', 'analysis']
        })
        
        print_success("Creación de DataFrame - OK")
        
        # Test de utilidades
        from utils.common_utils import safe_float, safe_int, detect_data_quality
        
        assert safe_float('123.45') == 123.45
        assert safe_int('123') == 123
        assert safe_float('invalid') is None
        
        print_success("Utilidades básicas - OK")
        
        # Test de detección de calidad
        quality = detect_data_quality(df)
        assert 'quality' in quality
        assert 'missing_percentage' in quality
        
        print_success("Detección de calidad - OK")
        
        # Test de validación
        from utils.validation_utils import validate_dataframe_structure
        
        is_valid, errors = validate_dataframe_structure(df)
        assert is_valid
        assert len(errors) == 0
        
        print_success("Validación de DataFrame - OK")
        
        return True
        
    except Exception as e:
        print_error(f"Error en tests básicos: {e}")
        traceback.print_exc()
        return False

def main():
    """Función principal de verificación"""
    print_header("VERIFICACIÓN DEL SISTEMA CAF DASHBOARD")
    print_info("Iniciando verificación completa del sistema...")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Imports", check_imports),
        ("Data Directory", check_data_directory),
        ("Logs Directory", check_logs_directory),
        ("Configuration", check_configuration),
        ("Utilities", check_utilities),
        ("Analysis Modules", check_analysis_modules),
        ("Basic Tests", run_basic_tests)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_error(f"Error en {check_name}: {e}")
            results.append((check_name, False))
    
    # Resumen final
    print_header("RESUMEN FINAL")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print_info(f"Verificaciones pasadas: {passed}/{total}")
    
    for check_name, result in results:
        if result:
            print_success(f"{check_name} - PASÓ")
        else:
            print_error(f"{check_name} - FALLÓ")
    
    if passed == total:
        print_success("🎉 ¡Todas las verificaciones pasaron! El sistema está listo.")
        return 0
    else:
        print_error(f"❌ {total - passed} verificaciones fallaron. Revisar los errores arriba.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
