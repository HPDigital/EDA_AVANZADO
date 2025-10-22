#!/usr/bin/env python3
"""
Script de deployment para CAF Dashboard
Prepara y despliega el sistema en diferentes entornos
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

class DeploymentSystem:
    """Sistema de deployment para CAF Dashboard"""
    
    def __init__(self):
        self.root_dir = root_dir
        self.deployment_dir = self.root_dir / 'deployment'
        self.environments = {
            'development': {
                'port': 8501,
                'host': 'localhost',
                'debug': True,
                'log_level': 'DEBUG'
            },
            'staging': {
                'port': 8502,
                'host': '0.0.0.0',
                'debug': False,
                'log_level': 'INFO'
            },
            'production': {
                'port': 8503,
                'host': '0.0.0.0',
                'debug': False,
                'log_level': 'WARNING'
            }
        }
    
    def prepare_deployment(self, environment='development'):
        """Prepara el sistema para deployment"""
        print(f"🚀 Preparando deployment para entorno: {environment}")
        
        if environment not in self.environments:
            print(f"❌ Entorno no válido: {environment}")
            return False
        
        try:
            # Crear directorio de deployment
            env_dir = self.deployment_dir / environment
            env_dir.mkdir(parents=True, exist_ok=True)
            
            # Copiar archivos necesarios
            self._copy_application_files(env_dir)
            
            # Crear archivos de configuración específicos del entorno
            self._create_environment_config(env_dir, environment)
            
            # Crear scripts de inicio
            self._create_startup_scripts(env_dir, environment)
            
            # Crear Dockerfile si es necesario
            self._create_dockerfile(env_dir, environment)
            
            # Crear docker-compose si es necesario
            self._create_docker_compose(env_dir, environment)
            
            print(f"✅ Deployment preparado en: {env_dir}")
            return True
        
        except Exception as e:
            print(f"❌ Error preparando deployment: {e}")
            return False
    
    def _copy_application_files(self, env_dir):
        """Copia archivos de la aplicación"""
        print("  📁 Copiando archivos de la aplicación...")
        
        # Archivos principales
        main_files = [
            'app.py',
            'config.py',
            'requirements.txt',
            'requirements-windows.txt',
            'README.md',
            'MAINTENANCE.md'
        ]
        
        for file_name in main_files:
            src_file = self.root_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, env_dir / file_name)
                print(f"    ✅ {file_name}")
        
        # Directorios
        dirs_to_copy = ['utils', 'modules', 'scripts']
        for dir_name in dirs_to_copy:
            src_dir = self.root_dir / dir_name
            if src_dir.exists():
                dst_dir = env_dir / dir_name
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                print(f"    ✅ {dir_name}/")
        
        # Crear directorios necesarios
        required_dirs = ['data', 'logs', 'reports', 'tmp']
        for dir_name in required_dirs:
            (env_dir / dir_name).mkdir(exist_ok=True)
            print(f"    ✅ {dir_name}/")
    
    def _create_environment_config(self, env_dir, environment):
        """Crea configuración específica del entorno"""
        print(f"  ⚙️ Creando configuración para {environment}...")
        
        config = self.environments[environment]
        
        # Crear config.py específico del entorno
        config_content = f'''"""
Configuración específica para entorno {environment}
Generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Configuración de Streamlit
STREAMLIT_CONFIG = {{
    'server.port': {config['port']},
    'server.address': '{config['host']}',
    'browser.gatherUsageStats': False,
    'theme.base': 'light',
    'theme.primaryColor': '#1f77b4',
    'theme.backgroundColor': '#ffffff',
    'theme.secondaryBackgroundColor': '#f0f2f6',
    'theme.textColor': '#262730'
}}

# Configuración de logging
LOGGING_CONFIG = {{
    'level': '{config['log_level']}',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/caf_dashboard_{environment}.log',
    'max_size': 10485760,  # 10MB
    'backup_count': 5
}}

# Configuración de performance
PERFORMANCE_CONFIG = {{
    'enable_caching': True,
    'cache_ttl': 3600,  # 1 hora
    'max_dataframe_size': 1000000,  # 1M filas
    'chunk_size': 10000,
    'enable_parallel_processing': True
}}

# Configuración de análisis
ANALYSIS_CONFIG = {{
    'min_observations': 10,
    'max_variables_display': 20,
    'correlation_threshold': 0.8,
    'outlier_threshold': 1.5,
    'confidence_level': 0.95,
    'max_clusters': 10,
    'test_size_default': 0.2,
    'cv_folds_default': 5
}}

# Configuración de reportes
REPORT_CONFIG = {{
    'max_chart_width': 800,
    'max_chart_height': 600,
    'default_chart_format': 'PNG',
    'include_metadata': True,
    'include_recommendations': True,
    'max_recommendations': 10
}}

# Configuración de datos
DATA_CONFIG = {{
    'data_dir': 'data',
    'raw_data_dir': 'data/raw',
    'processed_data_dir': 'data/processed',
    'backup_dir': 'backups',
    'reports_dir': 'reports'
}}

# Configuración de seguridad
SECURITY_CONFIG = {{
    'enable_authentication': {str(config.get('enable_auth', False)).lower()},
    'session_timeout': 3600,  # 1 hora
    'max_upload_size': 100 * 1024 * 1024,  # 100MB
    'allowed_file_types': ['.csv', '.xlsx', '.json']
}}
'''
        
        with open(env_dir / 'config_env.py', 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"    ✅ config_env.py")
    
    def _create_startup_scripts(self, env_dir, environment):
        """Crea scripts de inicio para el entorno"""
        print(f"  🚀 Creando scripts de inicio para {environment}...")
        
        # Script de inicio para Windows
        windows_script = f'''@echo off
echo Iniciando CAF Dashboard - Entorno {environment}
echo ================================================

REM Verificar si el entorno virtual existe
if not exist "venv" (
    echo Creando entorno virtual...
    python -m venv venv
)

REM Activar entorno virtual
call venv\\Scripts\\activate

REM Instalar dependencias
echo Instalando dependencias...
pip install -r requirements-windows.txt

REM Crear directorios necesarios
if not exist "data\\raw" mkdir data\\raw
if not exist "data\\processed" mkdir data\\processed
if not exist "logs" mkdir logs
if not exist "reports" mkdir reports
if not exist "tmp" mkdir tmp

REM Iniciar aplicación
echo Iniciando aplicación en puerto {self.environments[environment]['port']}...
streamlit run app.py --server.port {self.environments[environment]['port']} --server.address {self.environments[environment]['host']}

pause
'''
        
        with open(env_dir / 'start_windows.bat', 'w', encoding='utf-8') as f:
            f.write(windows_script)
        
        # Script de inicio para Linux/Mac
        unix_script = f'''#!/bin/bash
echo "Iniciando CAF Dashboard - Entorno {environment}"
echo "================================================"

# Verificar si el entorno virtual existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias
echo "Instalando dependencias..."
pip install -r requirements.txt

# Crear directorios necesarios
mkdir -p data/raw data/processed logs reports tmp

# Iniciar aplicación
echo "Iniciando aplicación en puerto {self.environments[environment]['port']}..."
streamlit run app.py --server.port {self.environments[environment]['port']} --server.address {self.environments[environment]['host']}
'''
        
        with open(env_dir / 'start_unix.sh', 'w', encoding='utf-8') as f:
            f.write(unix_script)
        
        # Hacer ejecutable en Unix
        if os.name != 'nt':
            os.chmod(env_dir / 'start_unix.sh', 0o755)
        
        print(f"    ✅ start_windows.bat")
        print(f"    ✅ start_unix.sh")
    
    def _create_dockerfile(self, env_dir, environment):
        """Crea Dockerfile para el entorno"""
        print(f"  🐳 Creando Dockerfile para {environment}...")
        
        dockerfile_content = f'''FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY requirements.txt .
COPY requirements-windows.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p data/raw data/processed logs reports tmp

# Exponer puerto
EXPOSE {self.environments[environment]['port']}

# Comando por defecto
CMD ["streamlit", "run", "app.py", "--server.port", "{self.environments[environment]['port']}", "--server.address", "{self.environments[environment]['host']}"]
'''
        
        with open(env_dir / 'Dockerfile', 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        
        print(f"    ✅ Dockerfile")
    
    def _create_docker_compose(self, env_dir, environment):
        """Crea docker-compose.yml para el entorno"""
        print(f"  🐳 Creando docker-compose.yml para {environment}...")
        
        compose_content = f'''version: '3.8'

services:
  caf-dashboard:
    build: .
    ports:
      - "{self.environments[environment]['port']}:{self.environments[environment]['port']}"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./reports:/app/reports
    environment:
      - STREAMLIT_SERVER_PORT={self.environments[environment]['port']}
      - STREAMLIT_SERVER_ADDRESS={self.environments[environment]['host']}
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{self.environments[environment]['port']}"]
      interval: 30s
      timeout: 10s
      retries: 3
'''
        
        with open(env_dir / 'docker-compose.yml', 'w', encoding='utf-8') as f:
            f.write(compose_content)
        
        print(f"    ✅ docker-compose.yml")
    
    def deploy_environment(self, environment='development'):
        """Despliega el entorno especificado"""
        print(f"🚀 Desplegando entorno: {environment}")
        
        env_dir = self.deployment_dir / environment
        
        if not env_dir.exists():
            print(f"❌ Entorno {environment} no está preparado. Ejecutar prepare_deployment primero.")
            return False
        
        try:
            # Cambiar al directorio del entorno
            os.chdir(env_dir)
            
            # Verificar si Docker está disponible
            if self._check_docker():
                print("🐳 Docker disponible. Usando contenedor...")
                self._deploy_with_docker(environment)
            else:
                print("🐍 Docker no disponible. Usando Python nativo...")
                self._deploy_with_python(environment)
            
            return True
        
        except Exception as e:
            print(f"❌ Error desplegando entorno: {e}")
            return False
    
    def _check_docker(self):
        """Verifica si Docker está disponible"""
        try:
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _deploy_with_docker(self, environment):
        """Despliega usando Docker"""
        print("  🐳 Construyendo imagen Docker...")
        subprocess.run(['docker', 'build', '-t', f'caf-dashboard-{environment}', '.'], check=True)
        
        print("  🐳 Iniciando contenedor...")
        subprocess.run([
            'docker', 'run', '-d',
            '--name', f'caf-dashboard-{environment}',
            '-p', f'{self.environments[environment]["port"]}:{self.environments[environment]["port"]}',
            f'caf-dashboard-{environment}'
        ], check=True)
        
        print(f"  ✅ Aplicación desplegada en puerto {self.environments[environment]['port']}")
    
    def _deploy_with_python(self, environment):
        """Despliega usando Python nativo"""
        print("  🐍 Creando entorno virtual...")
        subprocess.run(['python', '-m', 'venv', 'venv'], check=True)
        
        if os.name == 'nt':
            activate_script = 'venv\\Scripts\\activate'
            pip_script = 'venv\\Scripts\\pip'
        else:
            activate_script = 'venv/bin/activate'
            pip_script = 'venv/bin/pip'
        
        print("  🐍 Instalando dependencias...")
        subprocess.run([pip_script, 'install', '-r', 'requirements.txt'], check=True)
        
        print("  🐍 Iniciando aplicación...")
        subprocess.run([
            'streamlit', 'run', 'app.py',
            '--server.port', str(self.environments[environment]['port']),
            '--server.address', self.environments[environment]['host']
        ])
    
    def list_environments(self):
        """Lista entornos disponibles"""
        print("\n🌍 ENTORNOS DISPONIBLES")
        print("=" * 40)
        
        for env_name, config in self.environments.items():
            status = "✅ Preparado" if (self.deployment_dir / env_name).exists() else "❌ No preparado"
            print(f"{env_name:12} | Puerto: {config['port']:4} | {status}")
    
    def cleanup_environment(self, environment):
        """Limpia un entorno específico"""
        print(f"🧹 Limpiando entorno: {environment}")
        
        env_dir = self.deployment_dir / environment
        if env_dir.exists():
            shutil.rmtree(env_dir)
            print(f"✅ Entorno {environment} limpiado")
        else:
            print(f"⚠️ Entorno {environment} no existe")

def main():
    """Función principal del sistema de deployment"""
    print("🚀 SISTEMA DE DEPLOYMENT - CAF DASHBOARD")
    print("=" * 50)
    
    deployment = DeploymentSystem()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'prepare':
            environment = sys.argv[2] if len(sys.argv) > 2 else 'development'
            deployment.prepare_deployment(environment)
        
        elif command == 'deploy':
            environment = sys.argv[2] if len(sys.argv) > 2 else 'development'
            deployment.deploy_environment(environment)
        
        elif command == 'list':
            deployment.list_environments()
        
        elif command == 'cleanup':
            environment = sys.argv[2] if len(sys.argv) > 2 else 'development'
            deployment.cleanup_environment(environment)
        
        else:
            print(f"❌ Comando no válido: {command}")
            return 1
    
    else:
        # Modo interactivo
        while True:
            print("\nOpciones disponibles:")
            print("1. Preparar entorno")
            print("2. Desplegar entorno")
            print("3. Listar entornos")
            print("4. Limpiar entorno")
            print("5. Salir")
            
            try:
                choice = input("\nSelecciona una opción (1-5): ").strip()
                
                if choice == '1':
                    environment = input("Entorno (development/staging/production): ").strip()
                    deployment.prepare_deployment(environment)
                
                elif choice == '2':
                    environment = input("Entorno (development/staging/production): ").strip()
                    deployment.deploy_environment(environment)
                
                elif choice == '3':
                    deployment.list_environments()
                
                elif choice == '4':
                    environment = input("Entorno a limpiar: ").strip()
                    deployment.cleanup_environment(environment)
                
                elif choice == '5':
                    print("👋 ¡Hasta luego!")
                    break
                
                else:
                    print("❌ Opción inválida")
            
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
