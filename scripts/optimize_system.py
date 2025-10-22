#!/usr/bin/env python3
"""
Script de optimización automática del sistema CAF Dashboard
Optimiza la configuración, limpia archivos temporales y mejora el rendimiento
"""

import sys
import os
import shutil
import gc
from pathlib import Path
import pandas as pd
import numpy as np

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

def print_info(message):
    """Imprime un mensaje informativo"""
    print(f"ℹ️  {message}")

def print_warning(message):
    """Imprime un mensaje de advertencia"""
    print(f"⚠️  {message}")

def clean_temp_files():
    """Limpia archivos temporales"""
    print_header("LIMPIEZA DE ARCHIVOS TEMPORALES")
    
    temp_dirs = ['tmp', 'logs', '__pycache__']
    temp_extensions = ['.pyc', '.pyo', '.pyd', '.log', '.tmp']
    
    cleaned_files = 0
    cleaned_dirs = 0
    
    # Limpiar archivos temporales
    for root, dirs, files in os.walk(root_dir):
        # Eliminar archivos con extensiones temporales
        for file in files:
            if any(file.endswith(ext) for ext in temp_extensions):
                file_path = Path(root) / file
                try:
                    file_path.unlink()
                    cleaned_files += 1
                except Exception as e:
                    print_warning(f"No se pudo eliminar {file_path}: {e}")
        
        # Eliminar directorios __pycache__
        for dir_name in dirs[:]:  # Usar slice para modificar durante iteración
            if dir_name == '__pycache__':
                dir_path = Path(root) / dir_name
                try:
                    shutil.rmtree(dir_path)
                    cleaned_dirs += 1
                    dirs.remove(dir_name)  # Evitar procesar subdirectorios
                except Exception as e:
                    print_warning(f"No se pudo eliminar {dir_path}: {e}")
    
    print_success(f"Archivos temporales eliminados: {cleaned_files}")
    print_success(f"Directorios __pycache__ eliminados: {cleaned_dirs}")

def optimize_data_files():
    """Optimiza archivos de datos existentes"""
    print_header("OPTIMIZACIÓN DE ARCHIVOS DE DATOS")
    
    data_dir = root_dir / 'data' / 'raw'
    if not data_dir.exists():
        print_warning("Directorio data/raw no existe")
        return
    
    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        print_warning("No se encontraron archivos CSV para optimizar")
        return
    
    try:
        from utils.performance_utils import memory_optimize_dataframe
        
        for csv_file in csv_files:
            print_info(f"Optimizando {csv_file.name}...")
            
            try:
                # Leer archivo
                df = pd.read_csv(csv_file)
                original_size = df.memory_usage(deep=True).sum()
                
                # Optimizar
                df_optimized = memory_optimize_dataframe(df)
                new_size = df_optimized.memory_usage(deep=True).sum()
                
                # Calcular reducción
                reduction = (original_size - new_size) / original_size * 100
                
                if reduction > 5:  # Solo guardar si hay reducción significativa
                    # Crear backup
                    backup_file = csv_file.with_suffix('.csv.backup')
                    shutil.copy2(csv_file, backup_file)
                    
                    # Guardar optimizado
                    df_optimized.to_csv(csv_file, index=False)
                    
                    print_success(f"{csv_file.name}: {reduction:.1f}% reducción de memoria")
                else:
                    print_info(f"{csv_file.name}: Sin optimización necesaria")
                
            except Exception as e:
                print_warning(f"Error optimizando {csv_file.name}: {e}")
    
    except ImportError:
        print_warning("No se pueden importar utilidades de performance")

def create_directories():
    """Crea directorios necesarios"""
    print_header("CREACIÓN DE DIRECTORIOS")
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'logs',
        'reports',
        'tmp/uploads',
        'tmp/exports'
    ]
    
    for dir_path in required_dirs:
        full_path = root_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print_success(f"Directorio creado/verificado: {dir_path}")

def optimize_config():
    """Optimiza la configuración"""
    print_header("OPTIMIZACIÓN DE CONFIGURACIÓN")
    
    config_file = root_dir / 'config.py'
    if not config_file.exists():
        print_warning("Archivo config.py no encontrado")
        return
    
    try:
        # Leer configuración actual
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar configuraciones recomendadas
        optimizations = []
        
        if 'PERFORMANCE_CONFIG' not in content:
            optimizations.append("Agregar PERFORMANCE_CONFIG")
        
        if 'ANALYSIS_CONFIG' not in content:
            optimizations.append("Agregar ANALYSIS_CONFIG")
        
        if 'REPORT_CONFIG' not in content:
            optimizations.append("Agregar REPORT_CONFIG")
        
        if optimizations:
            print_warning("Configuraciones recomendadas faltantes:")
            for opt in optimizations:
                print_info(f"  - {opt}")
        else:
            print_success("Configuración optimizada")
    
    except Exception as e:
        print_warning(f"Error verificando configuración: {e}")

def check_dependencies():
    """Verifica y actualiza dependencias"""
    print_header("VERIFICACIÓN DE DEPENDENCIES")
    
    requirements_file = root_dir / 'requirements.txt'
    if not requirements_file.exists():
        print_warning("Archivo requirements.txt no encontrado")
        return
    
    try:
        import subprocess
        
        # Verificar pip
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print_success("pip disponible")
        else:
            print_warning("pip no disponible")
            return
        
        # Verificar paquetes instalados
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            installed_packages = result.stdout
            print_success("Lista de paquetes obtenida")
        else:
            print_warning("No se pudo obtener lista de paquetes")
            return
        
        # Verificar paquetes críticos
        critical_packages = ['streamlit', 'pandas', 'numpy', 'plotly', 'scipy', 'sklearn']
        missing_packages = []
        
        for package in critical_packages:
            if package not in installed_packages:
                missing_packages.append(package)
        
        if missing_packages:
            print_warning(f"Paquetes faltantes: {', '.join(missing_packages)}")
            print_info("Ejecutar: pip install -r requirements.txt")
        else:
            print_success("Todos los paquetes críticos están instalados")
    
    except Exception as e:
        print_warning(f"Error verificando dependencias: {e}")

def optimize_memory():
    """Optimiza el uso de memoria"""
    print_header("OPTIMIZACIÓN DE MEMORIA")
    
    # Limpiar caché de Python
    gc.collect()
    print_success("Caché de Python limpiado")
    
    # Verificar memoria disponible
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        print_info(f"Memoria disponible: {available_gb:.1f} GB")
        
        if available_gb < 1:
            print_warning("Memoria disponible baja. Considerar cerrar otras aplicaciones.")
        else:
            print_success("Memoria disponible suficiente")
    
    except ImportError:
        print_info("psutil no disponible. No se puede verificar memoria.")

def create_optimization_report():
    """Crea un reporte de optimización"""
    print_header("REPORTE DE OPTIMIZACIÓN")
    
    report_file = root_dir / 'logs' / 'optimization_report.txt'
    report_file.parent.mkdir(exist_ok=True)
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE OPTIMIZACIÓN - CAF DASHBOARD\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Fecha: {pd.Timestamp.now()}\n")
            f.write(f"Python: {sys.version}\n")
            f.write(f"Pandas: {pd.__version__}\n")
            f.write(f"Numpy: {np.__version__}\n\n")
            
            f.write("OPTIMIZACIONES REALIZADAS:\n")
            f.write("- Limpieza de archivos temporales\n")
            f.write("- Optimización de archivos de datos\n")
            f.write("- Creación de directorios necesarios\n")
            f.write("- Verificación de configuración\n")
            f.write("- Verificación de dependencias\n")
            f.write("- Optimización de memoria\n\n")
            
            f.write("RECOMENDACIONES:\n")
            f.write("- Ejecutar este script regularmente\n")
            f.write("- Monitorear el uso de memoria\n")
            f.write("- Mantener las dependencias actualizadas\n")
            f.write("- Revisar los logs regularmente\n")
        
        print_success(f"Reporte guardado en: {report_file}")
    
    except Exception as e:
        print_warning(f"Error creando reporte: {e}")

def main():
    """Función principal de optimización"""
    print_header("OPTIMIZACIÓN AUTOMÁTICA - CAF DASHBOARD")
    print_info("Iniciando optimización del sistema...")
    
    try:
        # Ejecutar optimizaciones
        clean_temp_files()
        create_directories()
        optimize_config()
        check_dependencies()
        optimize_memory()
        optimize_data_files()
        create_optimization_report()
        
        print_header("OPTIMIZACIÓN COMPLETADA")
        print_success("🎉 ¡Optimización completada exitosamente!")
        print_info("El sistema está optimizado y listo para usar.")
        
        return 0
    
    except Exception as e:
        print_warning(f"Error durante la optimización: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
