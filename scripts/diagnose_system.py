#!/usr/bin/env python3
"""
Sistema de diagnóstico para CAF Dashboard
Diagnostica problemas, verifica dependencias y sugiere soluciones
"""

import sys
import os
import subprocess
import platform
import psutil
from pathlib import Path
from datetime import datetime
import json

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

class SystemDiagnostic:
    """Sistema de diagnóstico del CAF Dashboard"""
    
    def __init__(self):
        self.root_dir = root_dir
        self.diagnostic_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'python_info': {},
            'dependencies': {},
            'files_structure': {},
            'performance': {},
            'issues': [],
            'recommendations': []
        }
    
    def run_full_diagnostic(self):
        """Ejecuta diagnóstico completo del sistema"""
        print("🔍 DIAGNÓSTICO COMPLETO DEL SISTEMA CAF DASHBOARD")
        print("=" * 60)
        
        # Diagnóstico del sistema
        self._diagnose_system()
        
        # Diagnóstico de Python
        self._diagnose_python()
        
        # Diagnóstico de dependencias
        self._diagnose_dependencies()
        
        # Diagnóstico de estructura de archivos
        self._diagnose_file_structure()
        
        # Diagnóstico de rendimiento
        self._diagnose_performance()
        
        # Análisis de problemas
        self._analyze_issues()
        
        # Generar recomendaciones
        self._generate_recommendations()
        
        # Mostrar resultados
        self._display_results()
        
        # Guardar reporte
        self._save_report()
        
        return len(self.diagnostic_results['issues']) == 0
    
    def _diagnose_system(self):
        """Diagnostica información del sistema"""
        print("🖥️ Diagnosticando sistema...")
        
        try:
            self.diagnostic_results['system_info'] = {
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'architecture': platform.architecture(),
                'hostname': platform.node(),
                'python_build': platform.python_build(),
                'python_compiler': platform.python_compiler()
            }
            
            # Información de memoria
            memory = psutil.virtual_memory()
            self.diagnostic_results['system_info']['memory'] = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_percent': memory.percent
            }
            
            # Información de CPU
            self.diagnostic_results['system_info']['cpu'] = {
                'count': psutil.cpu_count(),
                'count_logical': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
            
            # Información de disco
            disk = psutil.disk_usage('/')
            self.diagnostic_results['system_info']['disk'] = {
                'total_gb': round(disk.total / (1024**3), 2),
                'used_gb': round(disk.used / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'used_percent': round((disk.used / disk.total) * 100, 2)
            }
            
            print("  ✅ Información del sistema recopilada")
        
        except Exception as e:
            self.diagnostic_results['issues'].append(f"Error obteniendo información del sistema: {e}")
            print(f"  ❌ Error: {e}")
    
    def _diagnose_python(self):
        """Diagnostica información de Python"""
        print("🐍 Diagnosticando Python...")
        
        try:
            self.diagnostic_results['python_info'] = {
                'version': sys.version,
                'version_info': sys.version_info._asdict(),
                'executable': sys.executable,
                'path': sys.path,
                'platform': sys.platform,
                'implementation': platform.python_implementation(),
                'compiler': platform.python_compiler()
            }
            
            # Verificar si está en entorno virtual
            in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            self.diagnostic_results['python_info']['in_virtual_env'] = in_venv
            self.diagnostic_results['python_info']['virtual_env_path'] = sys.prefix if in_venv else None
            
            print("  ✅ Información de Python recopilada")
        
        except Exception as e:
            self.diagnostic_results['issues'].append(f"Error obteniendo información de Python: {e}")
            print(f"  ❌ Error: {e}")
    
    def _diagnose_dependencies(self):
        """Diagnostica dependencias de Python"""
        print("📦 Diagnosticando dependencias...")
        
        # Lista de dependencias críticas
        critical_deps = [
            'streamlit', 'pandas', 'numpy', 'plotly', 'scipy', 
            'sklearn', 'statsmodels', 'prophet', 'ruptures'
        ]
        
        optional_deps = [
            'hdbscan', 'umap', 'pingouin', 'dcor', 'networkx', 
            'folium', 'pysal', 'psutil'
        ]
        
        self.diagnostic_results['dependencies'] = {
            'critical': {},
            'optional': {},
            'missing_critical': [],
            'missing_optional': []
        }
        
        # Verificar dependencias críticas
        for dep in critical_deps:
            try:
                module = __import__(dep)
                version = getattr(module, '__version__', 'Unknown')
                self.diagnostic_results['dependencies']['critical'][dep] = version
                print(f"  ✅ {dep}: {version}")
            except ImportError:
                self.diagnostic_results['dependencies']['missing_critical'].append(dep)
                self.diagnostic_results['issues'].append(f"Dependencia crítica faltante: {dep}")
                print(f"  ❌ {dep}: No instalado")
        
        # Verificar dependencias opcionales
        for dep in optional_deps:
            try:
                module = __import__(dep)
                version = getattr(module, '__version__', 'Unknown')
                self.diagnostic_results['dependencies']['optional'][dep] = version
                print(f"  ℹ️ {dep}: {version}")
            except ImportError:
                self.diagnostic_results['dependencies']['missing_optional'].append(dep)
                print(f"  ⚠️ {dep}: No instalado (opcional)")
    
    def _diagnose_file_structure(self):
        """Diagnostica estructura de archivos"""
        print("📁 Diagnosticando estructura de archivos...")
        
        required_files = [
            'app.py',
            'config.py',
            'requirements.txt',
            'README.md'
        ]
        
        required_dirs = [
            'utils',
            'modules',
            'scripts',
            'data',
            'logs'
        ]
        
        self.diagnostic_results['files_structure'] = {
            'required_files': {},
            'required_dirs': {},
            'missing_files': [],
            'missing_dirs': []
        }
        
        # Verificar archivos requeridos
        for file_name in required_files:
            file_path = self.root_dir / file_name
            if file_path.exists():
                size = file_path.stat().st_size
                self.diagnostic_results['files_structure']['required_files'][file_name] = {
                    'exists': True,
                    'size_bytes': size,
                    'size_kb': round(size / 1024, 2)
                }
                print(f"  ✅ {file_name}: {size} bytes")
            else:
                self.diagnostic_results['files_structure']['missing_files'].append(file_name)
                self.diagnostic_results['issues'].append(f"Archivo requerido faltante: {file_name}")
                print(f"  ❌ {file_name}: No encontrado")
        
        # Verificar directorios requeridos
        for dir_name in required_dirs:
            dir_path = self.root_dir / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.rglob('*')))
                self.diagnostic_results['files_structure']['required_dirs'][dir_name] = {
                    'exists': True,
                    'file_count': file_count
                }
                print(f"  ✅ {dir_name}/: {file_count} archivos")
            else:
                self.diagnostic_results['files_structure']['missing_dirs'].append(dir_name)
                self.diagnostic_results['issues'].append(f"Directorio requerido faltante: {dir_name}")
                print(f"  ❌ {dir_name}/: No encontrado")
    
    def _diagnose_performance(self):
        """Diagnostica rendimiento del sistema"""
        print("⚡ Diagnosticando rendimiento...")
        
        try:
            # Test de importación de módulos principales
            import_times = {}
            modules_to_test = ['pandas', 'numpy', 'plotly', 'streamlit', 'sklearn']
            
            for module_name in modules_to_test:
                try:
                    start_time = datetime.now()
                    __import__(module_name)
                    end_time = datetime.now()
                    import_time = (end_time - start_time).total_seconds()
                    import_times[module_name] = import_time
                except ImportError:
                    import_times[module_name] = None
            
            self.diagnostic_results['performance'] = {
                'import_times': import_times,
                'memory_usage_mb': psutil.Process().memory_info().rss / (1024**2),
                'cpu_percent': psutil.cpu_percent(interval=1)
            }
            
            # Verificar tiempos de importación
            for module, time_taken in import_times.items():
                if time_taken is not None:
                    if time_taken > 2.0:
                        self.diagnostic_results['issues'].append(f"Importación lenta de {module}: {time_taken:.2f}s")
                        print(f"  ⚠️ {module}: {time_taken:.2f}s (lento)")
                    else:
                        print(f"  ✅ {module}: {time_taken:.2f}s")
                else:
                    print(f"  ❌ {module}: No disponible")
        
        except Exception as e:
            self.diagnostic_results['issues'].append(f"Error en diagnóstico de rendimiento: {e}")
            print(f"  ❌ Error: {e}")
    
    def _analyze_issues(self):
        """Analiza problemas encontrados"""
        print("🔍 Analizando problemas...")
        
        issues = self.diagnostic_results['issues']
        
        # Categorizar problemas
        critical_issues = []
        warning_issues = []
        info_issues = []
        
        for issue in issues:
            if any(keyword in issue.lower() for keyword in ['faltante', 'missing', 'no encontrado', 'no instalado']):
                critical_issues.append(issue)
            elif any(keyword in issue.lower() for keyword in ['lento', 'slow', 'error']):
                warning_issues.append(issue)
            else:
                info_issues.append(issue)
        
        self.diagnostic_results['issue_categories'] = {
            'critical': critical_issues,
            'warning': warning_issues,
            'info': info_issues
        }
        
        print(f"  🚨 Problemas críticos: {len(critical_issues)}")
        print(f"  ⚠️ Advertencias: {len(warning_issues)}")
        print(f"  ℹ️ Informativos: {len(info_issues)}")
    
    def _generate_recommendations(self):
        """Genera recomendaciones basadas en el diagnóstico"""
        print("💡 Generando recomendaciones...")
        
        recommendations = []
        
        # Recomendaciones basadas en dependencias faltantes
        missing_critical = self.diagnostic_results['dependencies'].get('missing_critical', [])
        if missing_critical:
            recommendations.append({
                'type': 'critical',
                'title': 'Instalar dependencias críticas faltantes',
                'description': f"Instalar: {', '.join(missing_critical)}",
                'command': f"pip install {' '.join(missing_critical)}"
            })
        
        # Recomendaciones basadas en archivos faltantes
        missing_files = self.diagnostic_results['files_structure'].get('missing_files', [])
        if missing_files:
            recommendations.append({
                'type': 'critical',
                'title': 'Archivos del sistema faltantes',
                'description': f"Archivos faltantes: {', '.join(missing_files)}",
                'command': "Verificar integridad del proyecto"
            })
        
        # Recomendaciones basadas en rendimiento
        import_times = self.diagnostic_results['performance'].get('import_times', {})
        slow_modules = [mod for mod, time in import_times.items() if time and time > 2.0]
        if slow_modules:
            recommendations.append({
                'type': 'warning',
                'title': 'Módulos con importación lenta',
                'description': f"Módulos lentos: {', '.join(slow_modules)}",
                'command': "Considerar optimización o reinstalación"
            })
        
        # Recomendaciones basadas en memoria
        memory_info = self.diagnostic_results['system_info'].get('memory', {})
        if memory_info.get('used_percent', 0) > 80:
            recommendations.append({
                'type': 'warning',
                'title': 'Uso alto de memoria',
                'description': f"Memoria usada: {memory_info.get('used_percent', 0):.1f}%",
                'command': "Cerrar aplicaciones innecesarias o reiniciar"
            })
        
        # Recomendaciones basadas en disco
        disk_info = self.diagnostic_results['system_info'].get('disk', {})
        if disk_info.get('used_percent', 0) > 90:
            recommendations.append({
                'type': 'critical',
                'title': 'Espacio en disco bajo',
                'description': f"Disco usado: {disk_info.get('used_percent', 0):.1f}%",
                'command': "Liberar espacio en disco"
            })
        
        self.diagnostic_results['recommendations'] = recommendations
        print(f"  ✅ {len(recommendations)} recomendaciones generadas")
    
    def _display_results(self):
        """Muestra resultados del diagnóstico"""
        print("\n📊 RESULTADOS DEL DIAGNÓSTICO")
        print("=" * 60)
        
        # Resumen general
        total_issues = len(self.diagnostic_results['issues'])
        critical_issues = len(self.diagnostic_results['issue_categories']['critical'])
        
        if critical_issues == 0:
            print("✅ Sistema en buen estado")
        elif critical_issues <= 2:
            print("⚠️ Sistema con problemas menores")
        else:
            print("❌ Sistema con problemas críticos")
        
        print(f"📈 Total de problemas: {total_issues}")
        print(f"🚨 Críticos: {critical_issues}")
        
        # Mostrar recomendaciones
        if self.diagnostic_results['recommendations']:
            print("\n💡 RECOMENDACIONES:")
            for i, rec in enumerate(self.diagnostic_results['recommendations'], 1):
                icon = "🚨" if rec['type'] == 'critical' else "⚠️" if rec['type'] == 'warning' else "ℹ️"
                print(f"{i}. {icon} {rec['title']}")
                print(f"   {rec['description']}")
                if rec.get('command'):
                    print(f"   💻 {rec['command']}")
                print()
    
    def _save_report(self):
        """Guarda reporte de diagnóstico"""
        report_file = self.root_dir / 'logs' / f'diagnostic_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.diagnostic_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Reporte guardado en: {report_file}")
    
    def quick_check(self):
        """Verificación rápida del sistema"""
        print("⚡ VERIFICACIÓN RÁPIDA")
        print("=" * 30)
        
        # Verificar Python
        print(f"🐍 Python: {sys.version.split()[0]}")
        
        # Verificar dependencias críticas
        critical_deps = ['streamlit', 'pandas', 'numpy', 'plotly']
        missing_deps = []
        
        for dep in critical_deps:
            try:
                __import__(dep)
                print(f"✅ {dep}")
            except ImportError:
                print(f"❌ {dep}")
                missing_deps.append(dep)
        
        # Verificar archivos principales
        main_files = ['app.py', 'config.py']
        missing_files = []
        
        for file_name in main_files:
            if (self.root_dir / file_name).exists():
                print(f"✅ {file_name}")
            else:
                print(f"❌ {file_name}")
                missing_files.append(file_name)
        
        # Resumen
        if not missing_deps and not missing_files:
            print("\n✅ Sistema listo para usar")
            return True
        else:
            print(f"\n❌ Problemas encontrados:")
            if missing_deps:
                print(f"  - Dependencias faltantes: {', '.join(missing_deps)}")
            if missing_files:
                print(f"  - Archivos faltantes: {', '.join(missing_files)}")
            return False

def main():
    """Función principal del sistema de diagnóstico"""
    print("🔍 SISTEMA DE DIAGNÓSTICO - CAF DASHBOARD")
    print("=" * 50)
    
    diagnostic = SystemDiagnostic()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'quick':
            diagnostic.quick_check()
        
        elif command == 'full':
            diagnostic.run_full_diagnostic()
        
        else:
            print(f"❌ Comando no válido: {command}")
            return 1
    
    else:
        # Modo interactivo
        while True:
            print("\nOpciones disponibles:")
            print("1. Verificación rápida")
            print("2. Diagnóstico completo")
            print("3. Salir")
            
            try:
                choice = input("\nSelecciona una opción (1-3): ").strip()
                
                if choice == '1':
                    diagnostic.quick_check()
                
                elif choice == '2':
                    diagnostic.run_full_diagnostic()
                
                elif choice == '3':
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
