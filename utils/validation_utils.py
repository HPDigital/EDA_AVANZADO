"""
Utilidades de validación y testing para CAF Dashboard
Incluye validaciones de datos, tests de integridad y verificaciones de calidad
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def validate_dataframe_structure(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Valida la estructura básica de un DataFrame"""
    errors = []
    
    if df is None:
        errors.append("DataFrame es None")
        return False, errors
    
    if df.empty:
        errors.append("DataFrame está vacío")
        return False, errors
    
    if len(df.columns) == 0:
        errors.append("DataFrame no tiene columnas")
        return False, errors
    
    if len(df) == 0:
        errors.append("DataFrame no tiene filas")
        return False, errors
    
    return True, errors

def validate_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[bool, List[str]]:
    """Valida que las columnas numéricas sean realmente numéricas"""
    errors = []
    
    for col in numeric_cols:
        if col not in df.columns:
            errors.append(f"Columna numérica '{col}' no existe en el DataFrame")
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Columna '{col}' no es de tipo numérico")
            continue
        
        # Verificar si hay valores no numéricos
        non_numeric_count = pd.to_numeric(df[col], errors='coerce').isna().sum()
        if non_numeric_count > 0:
            errors.append(f"Columna '{col}' tiene {non_numeric_count} valores no numéricos")
    
    return len(errors) == 0, errors

def validate_categorical_columns(df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[bool, List[str]]:
    """Valida que las columnas categóricas sean apropiadas"""
    errors = []
    
    for col in categorical_cols:
        if col not in df.columns:
            errors.append(f"Columna categórica '{col}' no existe en el DataFrame")
            continue
        
        # Verificar si tiene demasiados valores únicos (posible error de tipo)
        unique_count = df[col].nunique()
        if unique_count > len(df) * 0.9:
            errors.append(f"Columna '{col}' tiene demasiados valores únicos ({unique_count}), posible error de tipo")
    
    return len(errors) == 0, errors

def validate_time_columns(df: pd.DataFrame, time_cols: List[str]) -> Tuple[bool, List[str]]:
    """Valida que las columnas temporales sean válidas"""
    errors = []
    
    for col in time_cols:
        if col not in df.columns:
            errors.append(f"Columna temporal '{col}' no existe en el DataFrame")
            continue
        
        # Intentar convertir a datetime
        try:
            pd.to_datetime(df[col], errors='raise')
        except Exception as e:
            errors.append(f"Columna temporal '{col}' no se puede convertir a fecha: {str(e)}")
    
    return len(errors) == 0, errors

def validate_data_quality(df: pd.DataFrame, threshold: float = 0.5) -> Tuple[bool, Dict[str, Any]]:
    """Valida la calidad general de los datos"""
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_percentage': (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100),
        'duplicate_rows': df.duplicated().sum(),
        'empty_columns': (df.isna().all()).sum(),
        'quality_score': 0
    }
    
    # Calcular score de calidad
    quality_score = 100
    
    # Penalizar por datos faltantes
    missing_penalty = quality_report['missing_percentage'] * 2
    quality_score -= missing_penalty
    
    # Penalizar por filas duplicadas
    duplicate_penalty = (quality_report['duplicate_rows'] / len(df)) * 100
    quality_score -= duplicate_penalty
    
    # Penalizar por columnas vacías
    empty_penalty = (quality_report['empty_columns'] / len(df.columns)) * 100
    quality_score -= empty_penalty
    
    quality_report['quality_score'] = max(0, quality_score)
    
    is_valid = quality_report['quality_score'] >= (threshold * 100)
    
    return is_valid, quality_report

def validate_analysis_requirements(df: pd.DataFrame, analysis_type: str, 
                                 numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[bool, List[str]]:
    """Valida los requisitos para diferentes tipos de análisis"""
    errors = []
    
    if analysis_type == "univariate":
        if len(numeric_cols) == 0 and len(categorical_cols) == 0:
            errors.append("Se necesitan al menos variables numéricas o categóricas para análisis univariado")
    
    elif analysis_type == "bivariate":
        if len(numeric_cols) < 2 and len(categorical_cols) < 2:
            errors.append("Se necesitan al menos 2 variables para análisis bivariado")
    
    elif analysis_type == "multivariate":
        if len(numeric_cols) < 3:
            errors.append("Se necesitan al menos 3 variables numéricas para análisis multivariado")
    
    elif analysis_type == "regression":
        if len(numeric_cols) < 2:
            errors.append("Se necesitan al menos 2 variables numéricas para regresión")
    
    elif analysis_type == "classification":
        if len(numeric_cols) == 0 or len(categorical_cols) == 0:
            errors.append("Se necesitan variables numéricas y categóricas para clasificación")
    
    elif analysis_type == "clustering":
        if len(numeric_cols) < 2:
            errors.append("Se necesitan al menos 2 variables numéricas para clustering")
    
    elif analysis_type == "time_series":
        time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if len(time_cols) == 0:
            errors.append("Se necesita al menos una columna temporal para análisis de series temporales")
        if len(numeric_cols) == 0:
            errors.append("Se necesitan variables numéricas para análisis de series temporales")
    
    return len(errors) == 0, errors

def validate_correlation_matrix(corr_matrix: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Valida una matriz de correlación"""
    errors = []
    
    if corr_matrix is None:
        errors.append("Matriz de correlación es None")
        return False, errors
    
    if corr_matrix.empty:
        errors.append("Matriz de correlación está vacía")
        return False, errors
    
    # Verificar que sea cuadrada
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        errors.append("Matriz de correlación no es cuadrada")
    
    # Verificar que los valores estén en el rango [-1, 1]
    if not corr_matrix.isna().all().all():
        min_corr = corr_matrix.min().min()
        max_corr = corr_matrix.max().max()
        
        if min_corr < -1.01 or max_corr > 1.01:
            errors.append(f"Valores de correlación fuera del rango [-1, 1]: min={min_corr:.3f}, max={max_corr:.3f}")
    
    return len(errors) == 0, errors

def validate_statistical_test_results(results: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Valida los resultados de tests estadísticos"""
    errors = []
    
    if not isinstance(results, dict):
        errors.append("Los resultados deben ser un diccionario")
        return False, errors
    
    required_keys = ['statistic', 'p_value']
    for key in required_keys:
        if key not in results:
            errors.append(f"Falta la clave '{key}' en los resultados")
    
    if 'p_value' in results:
        p_value = results['p_value']
        if not isinstance(p_value, (int, float)) or p_value < 0 or p_value > 1:
            errors.append(f"Valor p inválido: {p_value}")
    
    return len(errors) == 0, errors

def validate_ml_model_inputs(X: pd.DataFrame, y: pd.Series, model_type: str) -> Tuple[bool, List[str]]:
    """Valida las entradas para modelos de machine learning"""
    errors = []
    
    # Validar X
    if X is None or X.empty:
        errors.append("Matriz de características X está vacía")
    
    if X is not None and X.isna().any().any():
        errors.append("Matriz de características X contiene valores faltantes")
    
    # Validar y
    if y is None or y.empty:
        errors.append("Variable objetivo y está vacía")
    
    if y is not None and y.isna().any():
        errors.append("Variable objetivo y contiene valores faltantes")
    
    # Validar compatibilidad
    if X is not None and y is not None and len(X) != len(y):
        errors.append("X e y tienen diferentes números de muestras")
    
    # Validaciones específicas por tipo de modelo
    if model_type == "classification":
        if y is not None:
            unique_classes = y.nunique()
            if unique_classes < 2:
                errors.append("Clasificación requiere al menos 2 clases")
            elif unique_classes > len(y) * 0.9:
                errors.append("Demasiadas clases únicas para clasificación")
    
    elif model_type == "regression":
        if y is not None and not pd.api.types.is_numeric_dtype(y):
            errors.append("Regresión requiere variable objetivo numérica")
    
    return len(errors) == 0, errors

def validate_plotly_figure(fig) -> Tuple[bool, List[str]]:
    """Valida una figura de Plotly"""
    errors = []
    
    if fig is None:
        errors.append("Figura de Plotly es None")
        return False, errors
    
    try:
        # Verificar que tenga al menos un trace
        if not hasattr(fig, 'data') or len(fig.data) == 0:
            errors.append("Figura no tiene datos (traces)")
        
        # Verificar que tenga layout
        if not hasattr(fig, 'layout'):
            errors.append("Figura no tiene layout")
        
    except Exception as e:
        errors.append(f"Error validando figura: {str(e)}")
    
    return len(errors) == 0, errors

def validate_report_data(report_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Valida los datos de un reporte"""
    errors = []
    
    if not isinstance(report_data, dict):
        errors.append("Datos del reporte deben ser un diccionario")
        return False, errors
    
    required_keys = ['metadata', 'sections']
    for key in required_keys:
        if key not in report_data:
            errors.append(f"Falta la clave '{key}' en los datos del reporte")
    
    if 'metadata' in report_data:
        metadata = report_data['metadata']
        if not isinstance(metadata, dict):
            errors.append("Metadatos deben ser un diccionario")
        else:
            required_metadata = ['title', 'generated_at']
            for key in required_metadata:
                if key not in metadata:
                    errors.append(f"Falta '{key}' en los metadatos")
    
    return len(errors) == 0, errors

def run_data_validation_suite(df: pd.DataFrame, numeric_cols: List[str], 
                             categorical_cols: List[str]) -> Dict[str, Any]:
    """Ejecuta una suite completa de validaciones"""
    validation_results = {
        'overall_valid': True,
        'errors': [],
        'warnings': [],
        'quality_score': 0
    }
    
    # Validar estructura
    is_valid, errors = validate_dataframe_structure(df)
    if not is_valid:
        validation_results['overall_valid'] = False
        validation_results['errors'].extend(errors)
    
    # Validar columnas numéricas
    is_valid, errors = validate_numeric_columns(df, numeric_cols)
    if not is_valid:
        validation_results['overall_valid'] = False
        validation_results['errors'].extend(errors)
    
    # Validar columnas categóricas
    is_valid, errors = validate_categorical_columns(df, categorical_cols)
    if not is_valid:
        validation_results['overall_valid'] = False
        validation_results['errors'].extend(errors)
    
    # Validar calidad de datos
    is_valid, quality_report = validate_data_quality(df)
    validation_results['quality_score'] = quality_report['quality_score']
    if not is_valid:
        validation_results['warnings'].append(f"Calidad de datos baja: {quality_report['quality_score']:.1f}/100")
    
    return validation_results

def create_validation_summary(validation_results: Dict[str, Any]) -> str:
    """Crea un resumen de las validaciones"""
    summary = f"**Resumen de Validación:**\n\n"
    
    if validation_results['overall_valid']:
        summary += "✅ **Estado:** Válido\n"
    else:
        summary += "❌ **Estado:** Inválido\n"
    
    summary += f"📊 **Calidad de Datos:** {validation_results['quality_score']:.1f}/100\n"
    
    if validation_results['errors']:
        summary += f"\n❌ **Errores ({len(validation_results['errors'])}):**\n"
        for error in validation_results['errors']:
            summary += f"- {error}\n"
    
    if validation_results['warnings']:
        summary += f"\n⚠️ **Advertencias ({len(validation_results['warnings'])}):**\n"
        for warning in validation_results['warnings']:
            summary += f"- {warning}\n"
    
    return summary
