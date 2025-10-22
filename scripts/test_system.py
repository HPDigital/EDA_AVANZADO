#!/usr/bin/env python3
"""
Sistema de testing automatizado para CAF Dashboard
Ejecuta tests unitarios, de integración y de rendimiento
"""

import sys
import os
import unittest
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

class TestDataIngestion(unittest.TestCase):
    """Tests para el sistema de ingesta de datos"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        try:
            from utils.smart_ingestion import SmartCSVIngestion
            self.ingestion = SmartCSVIngestion()
        except ImportError as e:
            self.skipTest(f"No se puede importar SmartCSVIngestion: {e}")
    
    def test_detect_column_type_numeric(self):
        """Test detección de columnas numéricas"""
        series = pd.Series([1, 2, 3, 4, 5])
        col_type, confidence = self.ingestion.detect_column_type(series, "test_numeric")
        self.assertEqual(col_type, "quantitative")
        self.assertGreater(confidence, 0.5)
    
    def test_detect_column_type_categorical(self):
        """Test detección de columnas categóricas"""
        series = pd.Series(["A", "B", "C", "A", "B"])
        col_type, confidence = self.ingestion.detect_column_type(series, "test_categorical")
        self.assertEqual(col_type, "qualitative")
        self.assertGreater(confidence, 0.5)
    
    def test_detect_column_type_datetime(self):
        """Test detección de columnas de fecha"""
        series = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
        col_type, confidence = self.ingestion.detect_column_type(series, "date")
        self.assertEqual(col_type, "datetime")
        self.assertGreater(confidence, 0.5)
    
    def test_normalize_column_name(self):
        """Test normalización de nombres de columnas"""
        test_cases = [
            ("test_column", "test column"),
            ("test-column", "test column"),
            ("test  column", "test column"),
            ("TEST_COLUMN", "test column")
        ]
        
        for input_name, expected in test_cases:
            result = self.ingestion._normalize_column_name(input_name)
            self.assertEqual(result, expected)

class TestDataAnalysis(unittest.TestCase):
    """Tests para funciones de análisis de datos"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        # Crear datos de prueba
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'numeric1': np.random.normal(100, 15, 100),
            'numeric2': np.random.normal(50, 10, 100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D')
        })
    
    def test_correlation_analysis(self):
        """Test análisis de correlación"""
        corr = self.test_data[['numeric1', 'numeric2']].corr()
        self.assertIsInstance(corr, pd.DataFrame)
        self.assertEqual(corr.shape, (2, 2))
    
    def test_descriptive_statistics(self):
        """Test estadísticas descriptivas"""
        stats = self.test_data['numeric1'].describe()
        self.assertIn('mean', stats.index)
        self.assertIn('std', stats.index)
        self.assertIn('min', stats.index)
        self.assertIn('max', stats.index)
    
    def test_missing_data_detection(self):
        """Test detección de datos faltantes"""
        # Agregar algunos valores faltantes
        test_data_with_na = self.test_data.copy()
        test_data_with_na.loc[0:5, 'numeric1'] = np.nan
        
        missing_count = test_data_with_na['numeric1'].isna().sum()
        self.assertEqual(missing_count, 6)

class TestPerformance(unittest.TestCase):
    """Tests de rendimiento"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        # Crear dataset grande para tests de rendimiento
        np.random.seed(42)
        self.large_data = pd.DataFrame({
            f'col_{i}': np.random.normal(0, 1, 10000)
            for i in range(20)
        })
    
    def test_data_loading_performance(self):
        """Test rendimiento de carga de datos"""
        start_time = time.time()
        
        # Simular carga de datos
        data_copy = self.large_data.copy()
        
        end_time = time.time()
        load_time = end_time - start_time
        
        # Debe cargar en menos de 1 segundo
        self.assertLess(load_time, 1.0)
        print(f"Tiempo de carga: {load_time:.3f}s")
    
    def test_correlation_calculation_performance(self):
        """Test rendimiento de cálculo de correlación"""
        start_time = time.time()
        
        corr_matrix = self.large_data.corr()
        
        end_time = time.time()
        calc_time = end_time - start_time
        
        # Debe calcular en menos de 2 segundos
        self.assertLess(calc_time, 2.0)
        print(f"Tiempo de correlación: {calc_time:.3f}s")
    
    def test_memory_usage(self):
        """Test uso de memoria"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Crear datos adicionales
        additional_data = pd.DataFrame({
            f'extra_col_{i}': np.random.normal(0, 1, 5000)
            for i in range(10)
        })
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # El aumento de memoria debe ser razonable (< 100MB)
        self.assertLess(memory_increase, 100)
        print(f"Aumento de memoria: {memory_increase:.1f}MB")

class TestDataValidation(unittest.TestCase):
    """Tests de validación de datos"""
    
    def test_dataframe_structure(self):
        """Test estructura del DataFrame"""
        # Crear DataFrame de prueba
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'C', 'D', 'E'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        # Verificar estructura
        self.assertEqual(len(df), 5)
        self.assertEqual(len(df.columns), 3)
        self.assertFalse(df.empty)
    
    def test_data_types(self):
        """Test tipos de datos"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['A', 'B', 'C'],
            'bool_col': [True, False, True]
        })
        
        self.assertTrue(pd.api.types.is_integer_dtype(df['int_col']))
        self.assertTrue(pd.api.types.is_float_dtype(df['float_col']))
        self.assertTrue(pd.api.types.is_object_dtype(df['str_col']))
        self.assertTrue(pd.api.types.is_bool_dtype(df['bool_col']))
    
    def test_missing_values_handling(self):
        """Test manejo de valores faltantes"""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': ['A', 'B', None, 'D', 'E']
        })
        
        # Verificar detección de valores faltantes
        self.assertEqual(df['col1'].isna().sum(), 1)
        self.assertEqual(df['col2'].isna().sum(), 1)
        
        # Verificar eliminación de valores faltantes
        df_clean = df.dropna()
        self.assertEqual(len(df_clean), 4)

class TestSystemIntegration(unittest.TestCase):
    """Tests de integración del sistema"""
    
    def test_import_all_modules(self):
        """Test importación de todos los módulos"""
        modules_to_test = [
            'utils.smart_ingestion',
            'utils.advanced_analysis',
            'utils.bivariate_analysis',
            'utils.multivariate_analysis',
            'modules.machine_learning',
            'modules.time_series',
            'modules.reports'
        ]
        
        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"No se puede importar {module_name}: {e}")
    
    def test_config_loading(self):
        """Test carga de configuración"""
        try:
            from config import THEME_TEMPLATE, PLOTLY_CONFIG
            self.assertIsNotNone(THEME_TEMPLATE)
            self.assertIsInstance(PLOTLY_CONFIG, dict)
        except ImportError as e:
            self.fail(f"No se puede cargar configuración: {e}")

class TestRunner:
    """Ejecutor de tests con reportes detallados"""
    
    def __init__(self):
        self.results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'execution_time': 0,
            'test_details': []
        }
    
    def run_all_tests(self):
        """Ejecuta todos los tests"""
        print("🧪 EJECUTANDO TESTS DEL SISTEMA CAF DASHBOARD")
        print("=" * 60)
        
        start_time = time.time()
        
        # Crear suite de tests
        test_suite = unittest.TestSuite()
        
        # Agregar todas las clases de test
        test_classes = [
            TestDataIngestion,
            TestDataAnalysis,
            TestPerformance,
            TestDataValidation,
            TestSystemIntegration
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        
        # Ejecutar tests
        runner = unittest.TextTestRunner(verbosity=2, stream=open(os.devnull, 'w'))
        result = runner.run(test_suite)
        
        end_time = time.time()
        
        # Actualizar resultados
        self.results.update({
            'total_tests': result.testsRun,
            'passed': result.testsRun - len(result.failures) - len(result.errors),
            'failed': len(result.failures),
            'errors': len(result.errors),
            'execution_time': end_time - start_time
        })
        
        # Mostrar resumen
        self._print_summary()
        
        # Generar reporte
        self._generate_report()
        
        return result.wasSuccessful()
    
    def _print_summary(self):
        """Imprime resumen de resultados"""
        print(f"\n📊 RESUMEN DE TESTS")
        print(f"Total de tests: {self.results['total_tests']}")
        print(f"✅ Exitosos: {self.results['passed']}")
        print(f"❌ Fallidos: {self.results['failed']}")
        print(f"⚠️ Errores: {self.results['errors']}")
        print(f"⏱️ Tiempo total: {self.results['execution_time']:.2f}s")
        
        success_rate = (self.results['passed'] / self.results['total_tests']) * 100
        print(f"📈 Tasa de éxito: {success_rate:.1f}%")
    
    def _generate_report(self):
        """Genera reporte detallado"""
        report_file = root_dir / 'logs' / 'test_report.json'
        report_file.parent.mkdir(exist_ok=True)
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.results,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'pandas_version': pd.__version__,
                'numpy_version': np.__version__
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Reporte guardado en: {report_file}")

def main():
    """Función principal del sistema de testing"""
    print("🧪 SISTEMA DE TESTING - CAF DASHBOARD")
    print("=" * 50)
    
    # Verificar dependencias
    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"❌ Dependencias faltantes: {e}")
        return 1
    
    # Ejecutar tests
    test_runner = TestRunner()
    success = test_runner.run_all_tests()
    
    if success:
        print("\n🎉 ¡Todos los tests pasaron exitosamente!")
        return 0
    else:
        print("\n⚠️ Algunos tests fallaron. Revisar el reporte para más detalles.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
