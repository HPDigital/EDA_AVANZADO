#!/usr/bin/env python3
"""
Monitor de logs para CAF Dashboard
Monitorea, analiza y alerta sobre eventos en los logs del sistema
"""

import sys
import os
import time
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

class LogMonitor:
    """Monitor de logs del sistema"""
    
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or (root_dir / 'logs')
        self.log_dir.mkdir(exist_ok=True)
        self.patterns = {
            'error': re.compile(r'ERROR|CRITICAL|FATAL', re.IGNORECASE),
            'warning': re.compile(r'WARNING|WARN', re.IGNORECASE),
            'info': re.compile(r'INFO', re.IGNORECASE),
            'debug': re.compile(r'DEBUG', re.IGNORECASE),
            'exception': re.compile(r'Exception|Traceback|Error:', re.IGNORECASE),
            'performance': re.compile(r'performance|slow|timeout|memory', re.IGNORECASE),
            'user_action': re.compile(r'user|login|logout|upload|download', re.IGNORECASE),
            'data_processing': re.compile(r'data|processing|analysis|ingestion', re.IGNORECASE)
        }
        self.alert_thresholds = {
            'error_rate': 0.1,  # 10% de errores
            'warning_rate': 0.2,  # 20% de warnings
            'response_time': 5.0,  # 5 segundos
            'memory_usage': 0.8,  # 80% de memoria
            'disk_usage': 0.9  # 90% de disco
        }
    
    def scan_logs(self, hours=24):
        """Escanea logs de las últimas N horas"""
        print(f"🔍 Escaneando logs de las últimas {hours} horas...")
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        log_files = list(self.log_dir.glob('*.log'))
        
        if not log_files:
            print("⚠️ No se encontraron archivos de log")
            return {}
        
        results = {
            'total_lines': 0,
            'errors': 0,
            'warnings': 0,
            'info': 0,
            'debug': 0,
            'exceptions': 0,
            'performance_issues': 0,
            'user_actions': 0,
            'data_processing': 0,
            'log_files': len(log_files),
            'time_range': f"{cutoff_time.strftime('%Y-%m-%d %H:%M')} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            'details': []
        }
        
        for log_file in log_files:
            file_results = self._analyze_log_file(log_file, cutoff_time)
            self._merge_results(results, file_results)
        
        return results
    
    def _analyze_log_file(self, log_file, cutoff_time):
        """Analiza un archivo de log específico"""
        results = {
            'file': log_file.name,
            'lines': 0,
            'errors': 0,
            'warnings': 0,
            'info': 0,
            'debug': 0,
            'exceptions': 0,
            'performance_issues': 0,
            'user_actions': 0,
            'data_processing': 0,
            'recent_errors': [],
            'recent_warnings': []
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    results['lines'] += 1
                    
                    # Verificar si la línea está dentro del rango de tiempo
                    if not self._is_line_in_time_range(line, cutoff_time):
                        continue
                    
                    # Analizar patrones
                    self._analyze_line(line, results, line_num)
        
        except Exception as e:
            print(f"⚠️ Error leyendo {log_file}: {e}")
        
        return results
    
    def _is_line_in_time_range(self, line, cutoff_time):
        """Verifica si una línea de log está dentro del rango de tiempo"""
        # Buscar timestamp en la línea (formato: YYYY-MM-DD HH:MM:SS)
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if not timestamp_match:
            return True  # Si no hay timestamp, incluir la línea
        
        try:
            line_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
            return line_time >= cutoff_time
        except ValueError:
            return True  # Si no se puede parsear, incluir la línea
    
    def _analyze_line(self, line, results, line_num):
        """Analiza una línea de log específica"""
        line_lower = line.lower()
        
        # Contar por nivel de log
        if self.patterns['error'].search(line):
            results['errors'] += 1
            if len(results['recent_errors']) < 10:
                results['recent_errors'].append(f"Línea {line_num}: {line.strip()}")
        
        elif self.patterns['warning'].search(line):
            results['warnings'] += 1
            if len(results['recent_warnings']) < 10:
                results['recent_warnings'].append(f"Línea {line_num}: {line.strip()}")
        
        elif self.patterns['info'].search(line):
            results['info'] += 1
        
        elif self.patterns['debug'].search(line):
            results['debug'] += 1
        
        # Contar por categoría
        if self.patterns['exception'].search(line):
            results['exceptions'] += 1
        
        if self.patterns['performance'].search(line):
            results['performance_issues'] += 1
        
        if self.patterns['user_action'].search(line):
            results['user_actions'] += 1
        
        if self.patterns['data_processing'].search(line):
            results['data_processing'] += 1
    
    def _merge_results(self, main_results, file_results):
        """Combina resultados de archivos individuales"""
        for key in ['total_lines', 'errors', 'warnings', 'info', 'debug', 
                   'exceptions', 'performance_issues', 'user_actions', 'data_processing']:
            main_results[key] += file_results[key]
        
        # Agregar detalles del archivo
        main_results['details'].append({
            'file': file_results['file'],
            'lines': file_results['lines'],
            'errors': file_results['errors'],
            'warnings': file_results['warnings'],
            'recent_errors': file_results['recent_errors'][:5],  # Solo los primeros 5
            'recent_warnings': file_results['recent_warnings'][:5]
        })
    
    def generate_report(self, hours=24):
        """Genera un reporte de análisis de logs"""
        print(f"\n📊 REPORTE DE ANÁLISIS DE LOGS - Últimas {hours} horas")
        print("=" * 60)
        
        results = self.scan_logs(hours)
        
        if not results['total_lines']:
            print("No hay datos de logs para analizar")
            return results
        
        # Resumen general
        print(f"📁 Archivos analizados: {results['log_files']}")
        print(f"📝 Líneas totales: {results['total_lines']:,}")
        print(f"⏰ Período: {results['time_range']}")
        print()
        
        # Distribución por nivel
        print("📊 DISTRIBUCIÓN POR NIVEL DE LOG:")
        levels = [
            ('ERROR', results['errors'], '❌'),
            ('WARNING', results['warnings'], '⚠️'),
            ('INFO', results['info'], 'ℹ️'),
            ('DEBUG', results['debug'], '🔍')
        ]
        
        for level, count, icon in levels:
            percentage = (count / results['total_lines']) * 100
            print(f"  {icon} {level:8}: {count:6,} ({percentage:5.1f}%)")
        
        print()
        
        # Análisis por categoría
        print("🔍 ANÁLISIS POR CATEGORÍA:")
        categories = [
            ('Excepciones', results['exceptions'], '💥'),
            ('Problemas de rendimiento', results['performance_issues'], '🐌'),
            ('Acciones de usuario', results['user_actions'], '👤'),
            ('Procesamiento de datos', results['data_processing'], '📊')
        ]
        
        for category, count, icon in categories:
            print(f"  {icon} {category:25}: {count:6,}")
        
        print()
        
        # Alertas
        self._check_alerts(results)
        
        # Detalles por archivo
        if results['details']:
            print("📁 DETALLES POR ARCHIVO:")
            for detail in results['details']:
                if detail['lines'] > 0:
                    print(f"  📄 {detail['file']}:")
                    print(f"    Líneas: {detail['lines']:,}")
                    print(f"    Errores: {detail['errors']}")
                    print(f"    Warnings: {detail['warnings']}")
                    
                    if detail['recent_errors']:
                        print(f"    Últimos errores:")
                        for error in detail['recent_errors']:
                            print(f"      • {error}")
                    
                    if detail['recent_warnings']:
                        print(f"    Últimos warnings:")
                        for warning in detail['recent_warnings']:
                            print(f"      • {warning}")
                    print()
        
        return results
    
    def _check_alerts(self, results):
        """Verifica alertas basadas en umbrales"""
        print("🚨 VERIFICACIÓN DE ALERTAS:")
        
        alerts = []
        
        # Verificar tasa de errores
        error_rate = results['errors'] / results['total_lines'] if results['total_lines'] > 0 else 0
        if error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"⚠️ Tasa de errores alta: {error_rate:.1%}")
        
        # Verificar tasa de warnings
        warning_rate = results['warnings'] / results['total_lines'] if results['total_lines'] > 0 else 0
        if warning_rate > self.alert_thresholds['warning_rate']:
            alerts.append(f"⚠️ Tasa de warnings alta: {warning_rate:.1%}")
        
        # Verificar excepciones
        if results['exceptions'] > 10:
            alerts.append(f"💥 Muchas excepciones: {results['exceptions']}")
        
        # Verificar problemas de rendimiento
        if results['performance_issues'] > 5:
            alerts.append(f"🐌 Problemas de rendimiento: {results['performance_issues']}")
        
        if alerts:
            for alert in alerts:
                print(f"  {alert}")
        else:
            print("  ✅ Sin alertas críticas")
        
        print()
    
    def monitor_realtime(self, log_file=None, interval=5):
        """Monitorea logs en tiempo real"""
        if not log_file:
            log_files = list(self.log_dir.glob('*.log'))
            if not log_files:
                print("❌ No se encontraron archivos de log")
                return
            log_file = max(log_files, key=lambda x: x.stat().st_mtime)
        
        print(f"👁️ Monitoreando en tiempo real: {log_file.name}")
        print(f"⏱️ Intervalo: {interval} segundos")
        print("Presiona Ctrl+C para detener...")
        print()
        
        try:
            # Obtener tamaño inicial del archivo
            last_size = log_file.stat().st_size
            
            while True:
                time.sleep(interval)
                
                # Verificar si el archivo ha crecido
                current_size = log_file.stat().st_size
                if current_size > last_size:
                    # Leer nuevas líneas
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(last_size)
                        new_lines = f.readlines()
                    
                    # Analizar nuevas líneas
                    for line in new_lines:
                        self._analyze_realtime_line(line)
                    
                    last_size = current_size
                
        except KeyboardInterrupt:
            print("\n👋 Monitoreo detenido")
    
    def _analyze_realtime_line(self, line):
        """Analiza una línea en tiempo real"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if self.patterns['error'].search(line):
            print(f"❌ [{timestamp}] ERROR: {line.strip()}")
        elif self.patterns['warning'].search(line):
            print(f"⚠️ [{timestamp}] WARNING: {line.strip()}")
        elif self.patterns['exception'].search(line):
            print(f"💥 [{timestamp}] EXCEPTION: {line.strip()}")
        elif self.patterns['performance'].search(line):
            print(f"🐌 [{timestamp}] PERFORMANCE: {line.strip()}")
    
    def export_analysis(self, results, output_file=None):
        """Exporta análisis a archivo JSON"""
        if not output_file:
            output_file = self.log_dir / f'log_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        # Agregar metadatos
        export_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_duration_hours': 24,  # Por defecto
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Análisis exportado a: {output_file}")

def main():
    """Función principal del monitor de logs"""
    print("📊 MONITOR DE LOGS - CAF DASHBOARD")
    print("=" * 50)
    
    monitor = LogMonitor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'scan':
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
            results = monitor.generate_report(hours)
            monitor.export_analysis(results)
        
        elif command == 'monitor':
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            monitor.monitor_realtime(interval=interval)
        
        else:
            print(f"❌ Comando no válido: {command}")
            return 1
    
    else:
        # Modo interactivo
        while True:
            print("\nOpciones disponibles:")
            print("1. Escanear logs (últimas 24 horas)")
            print("2. Escanear logs (últimas N horas)")
            print("3. Monitoreo en tiempo real")
            print("4. Exportar análisis")
            print("5. Salir")
            
            try:
                choice = input("\nSelecciona una opción (1-5): ").strip()
                
                if choice == '1':
                    results = monitor.generate_report(24)
                    monitor.export_analysis(results)
                
                elif choice == '2':
                    hours = int(input("Horas a analizar: "))
                    results = monitor.generate_report(hours)
                    monitor.export_analysis(results)
                
                elif choice == '3':
                    interval = int(input("Intervalo en segundos (default 5): ") or "5")
                    monitor.monitor_realtime(interval=interval)
                
                elif choice == '4':
                    results = monitor.scan_logs(24)
                    monitor.export_analysis(results)
                
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
