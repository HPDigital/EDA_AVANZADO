#!/usr/bin/env python3
"""
Script maestro de mantenimiento para CAF Dashboard
Ejecuta todos los scripts de mantenimiento y optimización
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

class MaintenanceMaster:
    """Maestro de mantenimiento del sistema"""
    
    def __init__(self):
        self.root_dir = root_dir
        self.scripts_dir = root_dir / 'scripts'
        self.log_file = root_dir / 'logs' / 'maintenance_master.log'
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Definir scripts de mantenimiento
        self.maintenance_scripts = {
            'diagnose': {
                'script': 'diagnose_system.py',
                'description': 'Diagnóstico completo del sistema',
                'critical': True,
                'timeout': 300  # 5 minutos
            },
            'optimize': {
                'script': 'optimize_system.py',
                'description': 'Optimización del sistema',
                'critical': False,
                'timeout': 600  # 10 minutos
            },
            'test': {
                'script': 'test_system.py',
                'description': 'Tests del sistema',
                'critical': True,
                'timeout': 300  # 5 minutos
            },
            'backup': {
                'script': 'backup_system.py',
                'description': 'Backup del sistema',
                'critical': True,
                'timeout': 900  # 15 minutos
            },
            'performance': {
                'script': 'performance_monitor.py',
                'description': 'Monitor de rendimiento',
                'critical': False,
                'timeout': 60  # 1 minuto
            },
            'logs': {
                'script': 'log_monitor.py',
                'description': 'Monitor de logs',
                'critical': False,
                'timeout': 120  # 2 minutos
            },
            'update': {
                'script': 'update_system.py',
                'description': 'Actualización del sistema',
                'critical': False,
                'timeout': 1800  # 30 minutos
            }
        }
    
    def log_message(self, message, level='INFO'):
        """Registra mensaje en el log"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f"📝 {message}")
    
    def run_script(self, script_name, args=None):
        """Ejecuta un script específico"""
        if script_name not in self.maintenance_scripts:
            self.log_message(f"Script no encontrado: {script_name}", 'ERROR')
            return False
        
        script_info = self.maintenance_scripts[script_name]
        script_path = self.scripts_dir / script_info['script']
        
        if not script_path.exists():
            self.log_message(f"Archivo de script no encontrado: {script_path}", 'ERROR')
            return False
        
        self.log_message(f"Ejecutando {script_info['description']}...")
        
        try:
            # Preparar comando
            cmd = [sys.executable, str(script_path)]
            if args:
                cmd.extend(args)
            
            # Ejecutar script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=script_info['timeout'],
                cwd=self.root_dir
            )
            
            if result.returncode == 0:
                self.log_message(f"✅ {script_info['description']} completado exitosamente")
                return True
            else:
                self.log_message(f"❌ {script_info['description']} falló: {result.stderr}", 'ERROR')
                return False
        
        except subprocess.TimeoutExpired:
            self.log_message(f"⏰ {script_info['description']} excedió el tiempo límite", 'ERROR')
            return False
        except Exception as e:
            self.log_message(f"💥 Error ejecutando {script_info['description']}: {e}", 'ERROR')
            return False
    
    def run_maintenance_routine(self, routine='full'):
        """Ejecuta rutina de mantenimiento"""
        print(f"🔧 INICIANDO RUTINA DE MANTENIMIENTO: {routine.upper()}")
        print("=" * 60)
        
        start_time = datetime.now()
        results = {}
        
        # Definir secuencias de mantenimiento
        if routine == 'quick':
            sequence = ['diagnose', 'test']
        elif routine == 'standard':
            sequence = ['diagnose', 'optimize', 'test', 'backup']
        elif routine == 'full':
            sequence = ['diagnose', 'optimize', 'test', 'backup', 'performance', 'logs']
        elif routine == 'emergency':
            sequence = ['diagnose', 'backup', 'test']
        else:
            print(f"❌ Rutina no válida: {routine}")
            return False
        
        # Ejecutar secuencia
        for script_name in sequence:
            print(f"\n🔄 Ejecutando: {self.maintenance_scripts[script_name]['description']}")
            success = self.run_script(script_name)
            results[script_name] = success
            
            # Si es crítico y falló, detener
            if self.maintenance_scripts[script_name]['critical'] and not success:
                self.log_message(f"🚨 Script crítico falló: {script_name}. Deteniendo rutina.", 'ERROR')
                break
        
        # Resumen de resultados
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n📊 RESUMEN DE MANTENIMIENTO")
        print("=" * 40)
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"⏱️ Duración total: {duration:.1f} segundos")
        print(f"✅ Exitosos: {successful}/{total}")
        print(f"❌ Fallidos: {total - successful}/{total}")
        
        # Mostrar detalles
        for script_name, success in results.items():
            status = "✅" if success else "❌"
            description = self.maintenance_scripts[script_name]['description']
            print(f"  {status} {description}")
        
        # Guardar reporte
        self._save_maintenance_report(routine, results, duration)
        
        return successful == total
    
    def run_individual_script(self, script_name, args=None):
        """Ejecuta un script individual"""
        if script_name not in self.maintenance_scripts:
            print(f"❌ Script no encontrado: {script_name}")
            print("Scripts disponibles:")
            for name, info in self.maintenance_scripts.items():
                print(f"  - {name}: {info['description']}")
            return False
        
        print(f"🚀 Ejecutando script individual: {script_name}")
        return self.run_script(script_name, args)
    
    def list_scripts(self):
        """Lista todos los scripts disponibles"""
        print("📋 SCRIPTS DE MANTENIMIENTO DISPONIBLES")
        print("=" * 50)
        
        for name, info in self.maintenance_scripts.items():
            critical = "🚨" if info['critical'] else "ℹ️"
            timeout = info['timeout'] // 60
            print(f"{critical} {name:12} | {info['description']:30} | {timeout:2} min")
    
    def show_status(self):
        """Muestra estado del sistema"""
        print("📊 ESTADO DEL SISTEMA")
        print("=" * 30)
        
        # Verificar archivos de log
        log_dir = self.root_dir / 'logs'
        if log_dir.exists():
            log_files = list(log_dir.glob('*.log'))
            print(f"📄 Archivos de log: {len(log_files)}")
            
            # Mostrar logs recientes
            recent_logs = sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            print("📋 Logs recientes:")
            for log_file in recent_logs:
                size_kb = log_file.stat().st_size / 1024
                mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                print(f"  • {log_file.name} ({size_kb:.1f} KB) - {mod_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("📄 No hay archivos de log")
        
        # Verificar backups
        backup_dir = self.root_dir / 'backups'
        if backup_dir.exists():
            backup_files = list(backup_dir.rglob('*.zip'))
            print(f"💾 Backups disponibles: {len(backup_files)}")
        else:
            print("💾 No hay backups")
        
        # Verificar espacio en disco
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.root_dir)
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            used_percent = (used / total) * 100
            
            print(f"💿 Espacio en disco: {free_gb:.1f} GB libres de {total_gb:.1f} GB ({used_percent:.1f}% usado)")
            
            if used_percent > 90:
                print("⚠️ Espacio en disco bajo")
            elif used_percent > 80:
                print("ℹ️ Espacio en disco moderado")
            else:
                print("✅ Espacio en disco suficiente")
        
        except Exception as e:
            print(f"⚠️ No se pudo verificar espacio en disco: {e}")
    
    def _save_maintenance_report(self, routine, results, duration):
        """Guarda reporte de mantenimiento"""
        report_file = self.root_dir / 'logs' / f'maintenance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'routine': routine,
            'duration_seconds': duration,
            'results': results,
            'summary': {
                'total_scripts': len(results),
                'successful': sum(1 for success in results.values() if success),
                'failed': sum(1 for success in results.values() if not success)
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.log_message(f"📄 Reporte guardado en: {report_file}")
    
    def schedule_maintenance(self, routine='standard', cron_expression=None):
        """Programa mantenimiento automático"""
        print(f"⏰ Programando mantenimiento: {routine}")
        
        if not cron_expression:
            if routine == 'quick':
                cron_expression = "0 */6 * * *"  # Cada 6 horas
            elif routine == 'standard':
                cron_expression = "0 2 * * *"  # Diario a las 2 AM
            elif routine == 'full':
                cron_expression = "0 1 * * 0"  # Semanal los domingos a la 1 AM
            else:
                print("❌ Rutina no válida para programación")
                return False
        
        # Crear script de cron
        cron_script = self.scripts_dir / 'cron_maintenance.sh'
        cron_content = f'''#!/bin/bash
# Mantenimiento automático programado
# Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

cd {self.root_dir}
python scripts/maintenance_master.py run {routine}
'''
        
        with open(cron_script, 'w', encoding='utf-8') as f:
            f.write(cron_content)
        
        # Hacer ejecutable
        os.chmod(cron_script, 0o755)
        
        print(f"📝 Script de cron creado: {cron_script}")
        print(f"⏰ Expresión cron: {cron_expression}")
        print("💡 Para activar, agregar a crontab:")
        print(f"   {cron_expression} {cron_script}")
        
        return True

def main():
    """Función principal del maestro de mantenimiento"""
    print("🔧 MAESTRO DE MANTENIMIENTO - CAF DASHBOARD")
    print("=" * 50)
    
    master = MaintenanceMaster()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'run':
            routine = sys.argv[2] if len(sys.argv) > 2 else 'standard'
            master.run_maintenance_routine(routine)
        
        elif command == 'script':
            script_name = sys.argv[2] if len(sys.argv) > 2 else None
            if script_name:
                args = sys.argv[3:] if len(sys.argv) > 3 else None
                master.run_individual_script(script_name, args)
            else:
                master.list_scripts()
        
        elif command == 'list':
            master.list_scripts()
        
        elif command == 'status':
            master.show_status()
        
        elif command == 'schedule':
            routine = sys.argv[2] if len(sys.argv) > 2 else 'standard'
            master.schedule_maintenance(routine)
        
        else:
            print(f"❌ Comando no válido: {command}")
            return 1
    
    else:
        # Modo interactivo
        while True:
            print("\nOpciones disponibles:")
            print("1. Ejecutar rutina rápida")
            print("2. Ejecutar rutina estándar")
            print("3. Ejecutar rutina completa")
            print("4. Ejecutar script individual")
            print("5. Listar scripts disponibles")
            print("6. Mostrar estado del sistema")
            print("7. Programar mantenimiento")
            print("8. Salir")
            
            try:
                choice = input("\nSelecciona una opción (1-8): ").strip()
                
                if choice == '1':
                    master.run_maintenance_routine('quick')
                
                elif choice == '2':
                    master.run_maintenance_routine('standard')
                
                elif choice == '3':
                    master.run_maintenance_routine('full')
                
                elif choice == '4':
                    master.list_scripts()
                    script_name = input("Nombre del script: ").strip()
                    if script_name:
                        master.run_individual_script(script_name)
                
                elif choice == '5':
                    master.list_scripts()
                
                elif choice == '6':
                    master.show_status()
                
                elif choice == '7':
                    routine = input("Rutina (quick/standard/full): ").strip()
                    master.schedule_maintenance(routine)
                
                elif choice == '8':
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
