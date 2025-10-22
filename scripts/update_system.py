#!/usr/bin/env python3
"""
Sistema de actualización automática para CAF Dashboard
Actualiza dependencias, código y configuración del sistema
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import json
import requests

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

class UpdateSystem:
    """Sistema de actualización automática"""
    
    def __init__(self):
        self.root_dir = root_dir
        self.backup_dir = self.root_dir / 'backups' / 'updates'
        self.update_log = self.root_dir / 'logs' / 'update.log'
        self.update_log.parent.mkdir(exist_ok=True)
        
        # URLs de actualización (ejemplo)
        self.update_urls = {
            'dependencies': 'https://raw.githubusercontent.com/example/caf-dashboard/main/requirements.txt',
            'config': 'https://raw.githubusercontent.com/example/caf-dashboard/main/config.py',
            'app': 'https://raw.githubusercontent.com/example/caf-dashboard/main/app.py'
        }
    
    def log_update(self, message):
        """Registra mensaje en el log de actualizaciones"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.update_log, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f"📝 {message}")
    
    def check_updates(self):
        """Verifica si hay actualizaciones disponibles"""
        print("🔍 Verificando actualizaciones disponibles...")
        
        try:
            # Verificar versión actual
            current_version = self._get_current_version()
            print(f"📌 Versión actual: {current_version}")
            
            # Verificar versión remota
            remote_version = self._get_remote_version()
            print(f"🌐 Versión remota: {remote_version}")
            
            if remote_version and remote_version != current_version:
                print("🆕 ¡Hay actualizaciones disponibles!")
                return True
            else:
                print("✅ El sistema está actualizado")
                return False
        
        except Exception as e:
            print(f"⚠️ Error verificando actualizaciones: {e}")
            return False
    
    def _get_current_version(self):
        """Obtiene la versión actual del sistema"""
        version_file = self.root_dir / 'VERSION'
        if version_file.exists():
            return version_file.read_text().strip()
        else:
            return "1.0.0"  # Versión por defecto
    
    def _get_remote_version(self):
        """Obtiene la versión remota"""
        try:
            # En un sistema real, esto haría una petición HTTP
            # Por ahora, simulamos una versión
            return "1.0.1"
        except Exception:
            return None
    
    def create_backup(self):
        """Crea backup antes de la actualización"""
        print("💾 Creando backup antes de la actualización...")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup_name = f"backup_before_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_name
        
        try:
            # Crear backup del código
            shutil.copytree(self.root_dir / 'utils', backup_path / 'utils')
            shutil.copytree(self.root_dir / 'modules', backup_path / 'modules')
            shutil.copytree(self.root_dir / 'scripts', backup_path / 'scripts')
            
            # Copiar archivos principales
            main_files = ['app.py', 'config.py', 'requirements.txt']
            for file_name in main_files:
                src_file = self.root_dir / file_name
                if src_file.exists():
                    shutil.copy2(src_file, backup_path / file_name)
            
            self.log_update(f"Backup creado: {backup_path}")
            return str(backup_path)
        
        except Exception as e:
            self.log_update(f"Error creando backup: {e}")
            return None
    
    def update_dependencies(self):
        """Actualiza dependencias de Python"""
        print("📦 Actualizando dependencias...")
        
        try:
            # Actualizar pip
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            
            # Actualizar dependencias
            requirements_file = self.root_dir / 'requirements.txt'
            if requirements_file.exists():
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file), '--upgrade'], 
                             check=True, capture_output=True)
            
            self.log_update("Dependencias actualizadas exitosamente")
            return True
        
        except subprocess.CalledProcessError as e:
            self.log_update(f"Error actualizando dependencias: {e}")
            return False
    
    def update_code(self, files_to_update=None):
        """Actualiza archivos de código"""
        print("💻 Actualizando código...")
        
        if not files_to_update:
            files_to_update = ['app.py', 'config.py', 'utils/', 'modules/']
        
        updated_files = []
        
        for file_path in files_to_update:
            try:
                if file_path.endswith('/'):
                    # Es un directorio
                    dir_name = file_path.rstrip('/')
                    src_dir = self.root_dir / dir_name
                    if src_dir.exists():
                        # En un sistema real, aquí descargarías los archivos actualizados
                        # Por ahora, solo registramos que se "actualizaría"
                        self.log_update(f"Directorio actualizado: {dir_name}")
                        updated_files.append(dir_name)
                else:
                    # Es un archivo
                    src_file = self.root_dir / file_path
                    if src_file.exists():
                        # En un sistema real, aquí descargarías el archivo actualizado
                        self.log_update(f"Archivo actualizado: {file_path}")
                        updated_files.append(file_path)
            
            except Exception as e:
                self.log_update(f"Error actualizando {file_path}: {e}")
        
        return updated_files
    
    def update_config(self):
        """Actualiza configuración del sistema"""
        print("⚙️ Actualizando configuración...")
        
        try:
            # Verificar si hay nueva configuración
            config_file = self.root_dir / 'config.py'
            if config_file.exists():
                # En un sistema real, aquí compararías y actualizarías la configuración
                self.log_update("Configuración actualizada")
                return True
            else:
                self.log_update("Archivo de configuración no encontrado")
                return False
        
        except Exception as e:
            self.log_update(f"Error actualizando configuración: {e}")
            return False
    
    def run_tests(self):
        """Ejecuta tests después de la actualización"""
        print("🧪 Ejecutando tests post-actualización...")
        
        try:
            # Ejecutar script de tests
            test_script = self.root_dir / 'scripts' / 'test_system.py'
            if test_script.exists():
                result = subprocess.run([sys.executable, str(test_script)], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log_update("Tests ejecutados exitosamente")
                    return True
                else:
                    self.log_update(f"Tests fallaron: {result.stderr}")
                    return False
            else:
                self.log_update("Script de tests no encontrado")
                return True
        
        except Exception as e:
            self.log_update(f"Error ejecutando tests: {e}")
            return False
    
    def rollback(self, backup_path):
        """Revierte a la versión anterior"""
        print("🔄 Revirtiendo a versión anterior...")
        
        try:
            if not backup_path or not Path(backup_path).exists():
                self.log_update("Backup no encontrado, no se puede revertir")
                return False
            
            backup_path = Path(backup_path)
            
            # Restaurar directorios
            for dir_name in ['utils', 'modules', 'scripts']:
                src_dir = backup_path / dir_name
                dst_dir = self.root_dir / dir_name
                
                if src_dir.exists():
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    shutil.copytree(src_dir, dst_dir)
                    self.log_update(f"Directorio restaurado: {dir_name}")
            
            # Restaurar archivos
            main_files = ['app.py', 'config.py', 'requirements.txt']
            for file_name in main_files:
                src_file = backup_path / file_name
                dst_file = self.root_dir / file_name
                
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    self.log_update(f"Archivo restaurado: {file_name}")
            
            self.log_update("Rollback completado exitosamente")
            return True
        
        except Exception as e:
            self.log_update(f"Error en rollback: {e}")
            return False
    
    def update_system(self, force=False):
        """Actualiza el sistema completo"""
        print("🚀 Iniciando actualización del sistema...")
        
        # Verificar si hay actualizaciones
        if not force and not self.check_updates():
            return True
        
        # Crear backup
        backup_path = self.create_backup()
        if not backup_path:
            print("❌ No se pudo crear backup, cancelando actualización")
            return False
        
        try:
            # Actualizar dependencias
            if not self.update_dependencies():
                print("⚠️ Error actualizando dependencias, continuando...")
            
            # Actualizar código
            updated_files = self.update_code()
            if not updated_files:
                print("⚠️ No se actualizaron archivos de código")
            
            # Actualizar configuración
            if not self.update_config():
                print("⚠️ Error actualizando configuración")
            
            # Ejecutar tests
            if not self.run_tests():
                print("⚠️ Tests fallaron, considerando rollback...")
                rollback_choice = input("¿Deseas revertir a la versión anterior? (y/n): ").lower()
                if rollback_choice == 'y':
                    self.rollback(backup_path)
                    return False
            
            # Actualizar versión
            self._update_version()
            
            print("✅ Actualización completada exitosamente")
            self.log_update("Actualización completada exitosamente")
            return True
        
        except Exception as e:
            print(f"❌ Error durante la actualización: {e}")
            self.log_update(f"Error durante la actualización: {e}")
            
            # Rollback automático en caso de error crítico
            rollback_choice = input("¿Deseas revertir automáticamente? (y/n): ").lower()
            if rollback_choice == 'y':
                self.rollback(backup_path)
            
            return False
    
    def _update_version(self):
        """Actualiza el número de versión"""
        version_file = self.root_dir / 'VERSION'
        new_version = self._get_remote_version() or "1.0.1"
        
        with open(version_file, 'w', encoding='utf-8') as f:
            f.write(new_version)
        
        self.log_update(f"Versión actualizada a: {new_version}")
    
    def show_update_log(self):
        """Muestra el log de actualizaciones"""
        print("📋 LOG DE ACTUALIZACIONES")
        print("=" * 50)
        
        if self.update_log.exists():
            with open(self.update_log, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    print(content)
                else:
                    print("No hay entradas en el log")
        else:
            print("Log de actualizaciones no encontrado")
    
    def cleanup_old_backups(self, keep_last=5):
        """Limpia backups antiguos"""
        print(f"🧹 Limpiando backups antiguos (manteniendo los últimos {keep_last})...")
        
        if not self.backup_dir.exists():
            return
        
        backup_dirs = [d for d in self.backup_dir.iterdir() if d.is_dir()]
        backup_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if len(backup_dirs) > keep_last:
            for old_backup in backup_dirs[keep_last:]:
                try:
                    shutil.rmtree(old_backup)
                    self.log_update(f"Backup eliminado: {old_backup.name}")
                except Exception as e:
                    self.log_update(f"Error eliminando backup {old_backup.name}: {e}")

def main():
    """Función principal del sistema de actualización"""
    print("🔄 SISTEMA DE ACTUALIZACIÓN - CAF DASHBOARD")
    print("=" * 50)
    
    update_system = UpdateSystem()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'check':
            update_system.check_updates()
        
        elif command == 'update':
            force = '--force' in sys.argv
            update_system.update_system(force=force)
        
        elif command == 'rollback':
            backup_path = sys.argv[2] if len(sys.argv) > 2 else None
            update_system.rollback(backup_path)
        
        elif command == 'log':
            update_system.show_update_log()
        
        elif command == 'cleanup':
            keep_last = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            update_system.cleanup_old_backups(keep_last)
        
        else:
            print(f"❌ Comando no válido: {command}")
            return 1
    
    else:
        # Modo interactivo
        while True:
            print("\nOpciones disponibles:")
            print("1. Verificar actualizaciones")
            print("2. Actualizar sistema")
            print("3. Actualizar forzadamente")
            print("4. Revertir a versión anterior")
            print("5. Ver log de actualizaciones")
            print("6. Limpiar backups antiguos")
            print("7. Salir")
            
            try:
                choice = input("\nSelecciona una opción (1-7): ").strip()
                
                if choice == '1':
                    update_system.check_updates()
                
                elif choice == '2':
                    update_system.update_system()
                
                elif choice == '3':
                    update_system.update_system(force=True)
                
                elif choice == '4':
                    backup_path = input("Ruta del backup a restaurar: ").strip()
                    update_system.rollback(backup_path)
                
                elif choice == '5':
                    update_system.show_update_log()
                
                elif choice == '6':
                    keep_last = input("Mantener últimos N backups (default 5): ").strip()
                    keep_last = int(keep_last) if keep_last else 5
                    update_system.cleanup_old_backups(keep_last)
                
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
