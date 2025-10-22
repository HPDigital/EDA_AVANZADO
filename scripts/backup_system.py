#!/usr/bin/env python3
"""
Sistema de backup automático para CAF Dashboard
Crea copias de seguridad de datos, configuraciones y código
"""

import sys
import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

class BackupSystem:
    """Sistema de backup automático"""
    
    def __init__(self, backup_dir=None):
        self.backup_dir = backup_dir or (root_dir / 'backups')
        self.backup_dir.mkdir(exist_ok=True)
        self.max_backups = 10  # Mantener solo los últimos 10 backups
    
    def create_backup(self, backup_name=None, include_data=True, include_code=True, include_config=True):
        """Crea un backup completo del sistema"""
        if not backup_name:
            backup_name = f"caf_dashboard_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        print(f"📦 Creando backup: {backup_name}")
        
        try:
            # Backup de datos
            if include_data:
                self._backup_data(backup_path)
            
            # Backup de código
            if include_code:
                self._backup_code(backup_path)
            
            # Backup de configuración
            if include_config:
                self._backup_config(backup_path)
            
            # Crear archivo de metadatos
            self._create_metadata(backup_path, include_data, include_code, include_config)
            
            # Comprimir backup
            self._compress_backup(backup_path)
            
            # Limpiar backups antiguos
            self._cleanup_old_backups()
            
            print(f"✅ Backup creado exitosamente: {backup_path}.zip")
            return str(backup_path) + '.zip'
        
        except Exception as e:
            print(f"❌ Error creando backup: {e}")
            # Limpiar en caso de error
            if backup_path.exists():
                shutil.rmtree(backup_path)
            return None
    
    def _backup_data(self, backup_path):
        """Backup de datos"""
        print("  📊 Respaldando datos...")
        
        data_dir = root_dir / 'data'
        if data_dir.exists():
            backup_data_dir = backup_path / 'data'
            shutil.copytree(data_dir, backup_data_dir)
            print(f"    ✅ Datos respaldados: {backup_data_dir}")
        else:
            print("    ⚠️ Directorio de datos no encontrado")
    
    def _backup_code(self, backup_path):
        """Backup de código"""
        print("  💻 Respaldando código...")
        
        # Archivos de código importantes
        code_files = [
            'app.py',
            'config.py',
            'requirements.txt',
            'requirements-windows.txt',
            'README.md',
            'MAINTENANCE.md'
        ]
        
        # Directorios de código
        code_dirs = [
            'utils',
            'modules',
            'scripts'
        ]
        
        # Copiar archivos individuales
        for file_name in code_files:
            file_path = root_dir / file_name
            if file_path.exists():
                shutil.copy2(file_path, backup_path / file_name)
                print(f"    ✅ {file_name}")
        
        # Copiar directorios
        for dir_name in code_dirs:
            dir_path = root_dir / dir_name
            if dir_path.exists():
                backup_code_dir = backup_path / dir_name
                shutil.copytree(dir_path, backup_code_dir)
                print(f"    ✅ {dir_name}/")
    
    def _backup_config(self, backup_path):
        """Backup de configuración"""
        print("  ⚙️ Respaldando configuración...")
        
        config_files = [
            'config.py',
            '.env',
            'streamlit_config.toml'
        ]
        
        for file_name in config_files:
            file_path = root_dir / file_name
            if file_path.exists():
                shutil.copy2(file_path, backup_path / file_name)
                print(f"    ✅ {file_name}")
        
        # Backup de logs
        logs_dir = root_dir / 'logs'
        if logs_dir.exists():
            backup_logs_dir = backup_path / 'logs'
            shutil.copytree(logs_dir, backup_logs_dir)
            print(f"    ✅ logs/")
    
    def _create_metadata(self, backup_path, include_data, include_code, include_config):
        """Crea archivo de metadatos del backup"""
        metadata = {
            'backup_name': backup_path.name,
            'created_at': datetime.now().isoformat(),
            'includes': {
                'data': include_data,
                'code': include_code,
                'config': include_config
            },
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'root_directory': str(root_dir)
            }
        }
        
        metadata_file = backup_path / 'backup_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"    ✅ Metadatos: {metadata_file}")
    
    def _compress_backup(self, backup_path):
        """Comprime el backup en un archivo ZIP"""
        print("  🗜️ Comprimiendo backup...")
        
        zip_path = backup_path.with_suffix('.zip')
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in backup_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(backup_path)
                    zipf.write(file_path, arcname)
        
        # Eliminar directorio sin comprimir
        shutil.rmtree(backup_path)
        
        # Calcular tamaño
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"    ✅ Backup comprimido: {size_mb:.1f} MB")
    
    def _cleanup_old_backups(self):
        """Elimina backups antiguos manteniendo solo los últimos N"""
        print("  🧹 Limpiando backups antiguos...")
        
        # Obtener todos los archivos ZIP de backup
        backup_files = list(self.backup_dir.glob('*.zip'))
        
        if len(backup_files) <= self.max_backups:
            print(f"    ℹ️ Solo {len(backup_files)} backups, no se necesita limpieza")
            return
        
        # Ordenar por fecha de modificación (más recientes primero)
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Eliminar los más antiguos
        files_to_delete = backup_files[self.max_backups:]
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                print(f"    🗑️ Eliminado: {file_path.name}")
            except Exception as e:
                print(f"    ⚠️ Error eliminando {file_path.name}: {e}")
        
        print(f"    ✅ Limpieza completada. Mantenidos: {self.max_backups} backups")
    
    def list_backups(self):
        """Lista todos los backups disponibles"""
        print("\n📋 BACKUPS DISPONIBLES")
        print("=" * 50)
        
        backup_files = list(self.backup_dir.glob('*.zip'))
        
        if not backup_files:
            print("No hay backups disponibles")
            return []
        
        # Ordenar por fecha de modificación
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        backups_info = []
        for i, backup_file in enumerate(backup_files, 1):
            stat = backup_file.stat()
            size_mb = stat.st_size / (1024 * 1024)
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            
            print(f"{i:2d}. {backup_file.name}")
            print(f"    📅 Fecha: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    📦 Tamaño: {size_mb:.1f} MB")
            print()
            
            backups_info.append({
                'file': backup_file,
                'name': backup_file.name,
                'size_mb': size_mb,
                'modified': modified_time
            })
        
        return backups_info
    
    def restore_backup(self, backup_name):
        """Restaura un backup específico"""
        backup_file = self.backup_dir / backup_name
        
        if not backup_file.exists():
            print(f"❌ Backup no encontrado: {backup_name}")
            return False
        
        print(f"🔄 Restaurando backup: {backup_name}")
        
        try:
            # Crear directorio temporal para extraer
            temp_dir = self.backup_dir / 'temp_restore'
            temp_dir.mkdir(exist_ok=True)
            
            # Extraer backup
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Restaurar archivos
            extracted_dir = temp_dir / backup_name.replace('.zip', '')
            
            # Restaurar datos
            data_backup = extracted_dir / 'data'
            if data_backup.exists():
                data_target = root_dir / 'data'
                if data_target.exists():
                    shutil.rmtree(data_target)
                shutil.copytree(data_backup, data_target)
                print("  ✅ Datos restaurados")
            
            # Restaurar código
            for item in extracted_dir.iterdir():
                if item.is_file() and item.name not in ['backup_metadata.json']:
                    target = root_dir / item.name
                    shutil.copy2(item, target)
                    print(f"  ✅ {item.name}")
            
            # Restaurar directorios de código
            for dir_name in ['utils', 'modules', 'scripts']:
                dir_backup = extracted_dir / dir_name
                if dir_backup.exists():
                    dir_target = root_dir / dir_name
                    if dir_target.exists():
                        shutil.rmtree(dir_target)
                    shutil.copytree(dir_backup, dir_target)
                    print(f"  ✅ {dir_name}/")
            
            # Limpiar directorio temporal
            shutil.rmtree(temp_dir)
            
            print("✅ Backup restaurado exitosamente")
            return True
        
        except Exception as e:
            print(f"❌ Error restaurando backup: {e}")
            return False
    
    def schedule_backup(self, interval_hours=24):
        """Programa backups automáticos"""
        print(f"⏰ Programando backups cada {interval_hours} horas")
        print("💡 Para implementar backups automáticos, usar un programador de tareas del sistema")
        print(f"   Comando: python {__file__} --auto-backup")

def main():
    """Función principal del sistema de backup"""
    print("💾 SISTEMA DE BACKUP - CAF DASHBOARD")
    print("=" * 50)
    
    backup_system = BackupSystem()
    
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        if sys.argv[1] == '--auto-backup':
            # Backup automático
            backup_system.create_backup()
            return 0
        elif sys.argv[1] == '--list':
            # Listar backups
            backup_system.list_backups()
            return 0
    
    # Modo interactivo
    while True:
        print("\nOpciones disponibles:")
        print("1. Crear backup completo")
        print("2. Crear backup de datos solamente")
        print("3. Crear backup de código solamente")
        print("4. Listar backups disponibles")
        print("5. Restaurar backup")
        print("6. Programar backups automáticos")
        print("7. Salir")
        
        try:
            choice = input("\nSelecciona una opción (1-7): ").strip()
            
            if choice == '1':
                backup_system.create_backup()
            
            elif choice == '2':
                backup_system.create_backup(include_code=False, include_config=False)
            
            elif choice == '3':
                backup_system.create_backup(include_data=False, include_config=False)
            
            elif choice == '4':
                backup_system.list_backups()
            
            elif choice == '5':
                backups = backup_system.list_backups()
                if backups:
                    try:
                        backup_num = int(input("Número de backup a restaurar: ")) - 1
                        if 0 <= backup_num < len(backups):
                            backup_system.restore_backup(backups[backup_num]['name'])
                        else:
                            print("❌ Número de backup inválido")
                    except ValueError:
                        print("❌ Número inválido")
            
            elif choice == '6':
                interval = input("Intervalo en horas (default 24): ").strip()
                interval = int(interval) if interval else 24
                backup_system.schedule_backup(interval)
            
            elif choice == '7':
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
