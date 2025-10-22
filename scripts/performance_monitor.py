#!/usr/bin/env python3
"""
Monitor de rendimiento para CAF Dashboard
Monitorea el uso de recursos y el rendimiento del sistema
"""

import sys
import os
import time
import psutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

class PerformanceMonitor:
    """Monitor de rendimiento del sistema"""
    
    def __init__(self, log_file=None):
        self.log_file = log_file or (root_dir / 'logs' / 'performance.log')
        self.log_file.parent.mkdir(exist_ok=True)
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = []
        
    def start_monitoring(self, interval=30):
        """Inicia el monitoreo en segundo plano"""
        if self.monitoring:
            print("⚠️ El monitoreo ya está activo")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print(f"✅ Monitoreo iniciado (intervalo: {interval}s)")
    
    def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("⏹️ Monitoreo detenido")
    
    def _monitor_loop(self, interval):
        """Loop principal de monitoreo"""
        while self.monitoring:
            try:
                metric = self._collect_metric()
                self.metrics.append(metric)
                self._log_metric(metric)
                time.sleep(interval)
            except Exception as e:
                print(f"⚠️ Error en monitoreo: {e}")
                time.sleep(interval)
    
    def _collect_metric(self):
        """Recolecta una métrica del sistema"""
        timestamp = datetime.now()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memoria
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available / (1024**3)  # GB
        memory_used = memory.used / (1024**3)  # GB
        
        # Disco
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free = disk.free / (1024**3)  # GB
        
        # Procesos Python
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return {
            'timestamp': timestamp.isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count
            },
            'memory': {
                'percent': memory_percent,
                'available_gb': memory_available,
                'used_gb': memory_used
            },
            'disk': {
                'percent': disk_percent,
                'free_gb': disk_free
            },
            'python_processes': python_processes
        }
    
    def _log_metric(self, metric):
        """Registra una métrica en el archivo de log"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metric) + '\n')
        except Exception as e:
            print(f"⚠️ Error escribiendo log: {e}")
    
    def get_summary(self, hours=1):
        """Obtiene un resumen de las métricas"""
        if not self.metrics:
            return "No hay métricas disponibles"
        
        # Filtrar métricas de las últimas N horas
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics 
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        if not recent_metrics:
            return f"No hay métricas de las últimas {hours} horas"
        
        # Calcular estadísticas
        cpu_values = [m['cpu']['percent'] for m in recent_metrics]
        memory_values = [m['memory']['percent'] for m in recent_metrics]
        disk_values = [m['disk']['percent'] for m in recent_metrics]
        
        summary = {
            'periodo_horas': hours,
            'metricas_totales': len(recent_metrics),
            'cpu': {
                'promedio': sum(cpu_values) / len(cpu_values),
                'maximo': max(cpu_values),
                'minimo': min(cpu_values)
            },
            'memoria': {
                'promedio': sum(memory_values) / len(memory_values),
                'maximo': max(memory_values),
                'minimo': min(memory_values)
            },
            'disco': {
                'promedio': sum(disk_values) / len(disk_values),
                'maximo': max(disk_values),
                'minimo': min(disk_values)
            }
        }
        
        return summary
    
    def generate_report(self, hours=24):
        """Genera un reporte de rendimiento"""
        print(f"\n📊 REPORTE DE RENDIMIENTO - Últimas {hours} horas")
        print("=" * 50)
        
        summary = self.get_summary(hours)
        if isinstance(summary, str):
            print(summary)
            return
        
        print(f"📈 Métricas analizadas: {summary['metricas_totales']}")
        print(f"⏱️  Período: {summary['periodo_horas']} horas")
        print()
        
        # CPU
        cpu = summary['cpu']
        print(f"🖥️  CPU:")
        print(f"   Promedio: {cpu['promedio']:.1f}%")
        print(f"   Máximo: {cpu['maximo']:.1f}%")
        print(f"   Mínimo: {cpu['minimo']:.1f}%")
        print()
        
        # Memoria
        memory = summary['memoria']
        print(f"💾 Memoria:")
        print(f"   Promedio: {memory['promedio']:.1f}%")
        print(f"   Máximo: {memory['maximo']:.1f}%")
        print(f"   Mínimo: {memory['minimo']:.1f}%")
        print()
        
        # Disco
        disk = summary['disco']
        print(f"💿 Disco:")
        print(f"   Promedio: {disk['promedio']:.1f}%")
        print(f"   Máximo: {disk['maximo']:.1f}%")
        print(f"   Mínimo: {disk['minimo']:.1f}%")
        print()
        
        # Alertas
        alerts = []
        if cpu['maximo'] > 90:
            alerts.append("⚠️ CPU alta (>90%)")
        if memory['maximo'] > 90:
            alerts.append("⚠️ Memoria alta (>90%)")
        if disk['maximo'] > 90:
            alerts.append("⚠️ Disco lleno (>90%)")
        
        if alerts:
            print("🚨 ALERTAS:")
            for alert in alerts:
                print(f"   {alert}")
        else:
            print("✅ Sin alertas de rendimiento")
    
    def export_metrics(self, output_file=None):
        """Exporta las métricas a un archivo CSV"""
        if not self.metrics:
            print("No hay métricas para exportar")
            return
        
        output_file = output_file or (root_dir / 'logs' / f'performance_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        # Convertir a DataFrame
        data = []
        for metric in self.metrics:
            data.append({
                'timestamp': metric['timestamp'],
                'cpu_percent': metric['cpu']['percent'],
                'memory_percent': metric['memory']['percent'],
                'memory_available_gb': metric['memory']['available_gb'],
                'memory_used_gb': metric['memory']['used_gb'],
                'disk_percent': metric['disk']['percent'],
                'disk_free_gb': metric['disk']['free_gb']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"✅ Métricas exportadas a: {output_file}")

def main():
    """Función principal del monitor"""
    print("🔍 MONITOR DE RENDIMIENTO - CAF DASHBOARD")
    print("=" * 50)
    
    monitor = PerformanceMonitor()
    
    try:
        # Verificar si psutil está disponible
        import psutil
    except ImportError:
        print("❌ psutil no está instalado. Instalar con: pip install psutil")
        return 1
    
    # Mostrar opciones
    print("\nOpciones disponibles:")
    print("1. Iniciar monitoreo en tiempo real")
    print("2. Generar reporte de las últimas 24 horas")
    print("3. Exportar métricas a CSV")
    print("4. Ver estado actual del sistema")
    print("5. Salir")
    
    while True:
        try:
            choice = input("\nSelecciona una opción (1-5): ").strip()
            
            if choice == '1':
                interval = input("Intervalo de monitoreo en segundos (default 30): ").strip()
                interval = int(interval) if interval else 30
                
                monitor.start_monitoring(interval)
                print("\nPresiona Enter para detener el monitoreo...")
                input()
                monitor.stop_monitoring()
            
            elif choice == '2':
                hours = input("Horas a analizar (default 24): ").strip()
                hours = int(hours) if hours else 24
                monitor.generate_report(hours)
            
            elif choice == '3':
                monitor.export_metrics()
            
            elif choice == '4':
                # Estado actual
                metric = monitor._collect_metric()
                print(f"\n📊 ESTADO ACTUAL DEL SISTEMA")
                print(f"🖥️  CPU: {metric['cpu']['percent']:.1f}%")
                print(f"💾 Memoria: {metric['memory']['percent']:.1f}% ({metric['memory']['used_gb']:.1f}GB usados)")
                print(f"💿 Disco: {metric['disk']['percent']:.1f}% ({metric['disk']['free_gb']:.1f}GB libres)")
                print(f"🐍 Procesos Python: {len(metric['python_processes'])}")
            
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
