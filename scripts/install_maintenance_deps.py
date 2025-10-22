#!/usr/bin/env python3
"""
Instalador de dependencias para el sistema de mantenimiento
Instala las dependencias necesarias para todos los scripts de mantenimiento
"""

import sys
import subprocess
import os
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

def install_package(package_name, description=""):
    """Instala un paquete de Python"""
    print(f"📦 Instalando {package_name}...")
    if description:
        print(f"   {description}")
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"   ✅ {package_name} instalado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error instalando {package_name}: {e.stderr}")
        return False

def check_package(package_name):
    """Verifica si un paquete está instalado"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """Función principal del instalador"""
    print("🔧 INSTALADOR DE DEPENDENCIAS DE MANTENIMIENTO")
    print("=" * 60)
    
    # Dependencias necesarias para el sistema de mantenimiento
    maintenance_dependencies = [
        {
            'package': 'psutil',
            'description': 'Información del sistema y procesos',
            'required': True
        },
        {
            'package': 'requests',
            'description': 'Peticiones HTTP para actualizaciones',
            'required': True
        },
        {
            'package': 'schedule',
            'description': 'Programación de tareas',
            'required': False
        },
        {
            'package': 'croniter',
            'description': 'Parsing de expresiones cron',
            'required': False
        },
        {
            'package': 'colorama',
            'description': 'Colores en terminal',
            'required': False
        },
        {
            'package': 'tqdm',
            'description': 'Barras de progreso',
            'required': False
        }
    ]
    
    print("🔍 Verificando dependencias existentes...")
    
    # Verificar dependencias existentes
    existing_packages = []
    missing_packages = []
    
    for dep in maintenance_dependencies:
        if check_package(dep['package']):
            existing_packages.append(dep['package'])
            print(f"✅ {dep['package']} ya está instalado")
        else:
            missing_packages.append(dep)
            print(f"❌ {dep['package']} no está instalado")
    
    if not missing_packages:
        print("\n🎉 ¡Todas las dependencias ya están instaladas!")
        return 0
    
    print(f"\n📋 Dependencias faltantes: {len(missing_packages)}")
    
    # Instalar dependencias faltantes
    print("\n🚀 Instalando dependencias faltantes...")
    
    successful_installs = 0
    failed_installs = 0
    
    for dep in missing_packages:
        success = install_package(dep['package'], dep['description'])
        if success:
            successful_installs += 1
        else:
            failed_installs += 1
            if dep['required']:
                print(f"⚠️ {dep['package']} es requerido pero falló la instalación")
    
    # Resumen de instalación
    print(f"\n📊 RESUMEN DE INSTALACIÓN")
    print("=" * 40)
    print(f"✅ Instalaciones exitosas: {successful_installs}")
    print(f"❌ Instalaciones fallidas: {failed_installs}")
    print(f"📦 Total de paquetes: {len(missing_packages)}")
    
    # Verificar instalaciones críticas
    critical_failed = []
    for dep in missing_packages:
        if dep['required'] and not check_package(dep['package']):
            critical_failed.append(dep['package'])
    
    if critical_failed:
        print(f"\n🚨 DEPENDENCIAS CRÍTICAS FALLIDAS:")
        for package in critical_failed:
            print(f"   ❌ {package}")
        print("\n💡 Soluciones sugeridas:")
        print("   1. Verificar conexión a internet")
        print("   2. Actualizar pip: python -m pip install --upgrade pip")
        print("   3. Instalar manualmente: pip install <package_name>")
        return 1
    else:
        print("\n🎉 ¡Instalación completada exitosamente!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
