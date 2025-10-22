from __future__ import annotations
"""
Utilidades de optimización de performance para CAF Dashboard
Incluye caching, procesamiento en lotes y optimizaciones de memoria
"""

import pandas as pd
import numpy as np
from functools import wraps
import time
import gc
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Detectar si Streamlit está disponible
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    class st:
        """Dummy Streamlit para compatibilidad"""
        @staticmethod
        def write(msg): pass
        @staticmethod
        def warning(msg): print(f"Warning: {msg}")
        @staticmethod
        def progress(val): return DummyProgress()

class DummyProgress:
    def progress(self, val): pass

# Configuración de performance
try:
    from config import PERFORMANCE_CONFIG
except ImportError:
    PERFORMANCE_CONFIG = {
        'enable_caching': True,
        'cache_ttl': 3600,
        'max_dataframe_size': 1000000,
        'chunk_size': 10000,
        'enable_parallel_processing': True
    }

def performance_monitor(func: Callable) -> Callable:
    """Decorador para monitorear el rendimiento de funciones"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if PERFORMANCE_CONFIG.get('enable_caching', True):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Log de performance si es necesario
            if end_time - start_time > 1.0:  # Solo log si toma más de 1 segundo
                st.write(f"⏱️ {func.__name__} ejecutado en {end_time - start_time:.2f} segundos")
            
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

def memory_optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimiza el uso de memoria de un DataFrame"""
    original_memory = df.memory_usage(deep=True).sum()
    
    # Optimizar tipos de datos
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # Para columnas de texto, usar category si tiene pocos valores únicos
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
    
    new_memory = df.memory_usage(deep=True).sum()
    reduction = (original_memory - new_memory) / original_memory * 100
    
    if reduction > 10:  # Solo mostrar si hay reducción significativa
        st.write(f"💾 Memoria optimizada: {reduction:.1f}% de reducción")
    
    return df

def chunked_processing(df: pd.DataFrame, func: Callable, chunk_size: int = None) -> pd.DataFrame:
    """Procesa un DataFrame en chunks para optimizar memoria"""
    if chunk_size is None:
        chunk_size = PERFORMANCE_CONFIG.get('chunk_size', 10000)
    
    if len(df) <= chunk_size:
        return func(df)
    
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        processed_chunk = func(chunk)
        chunks.append(processed_chunk)
        
        # Limpiar memoria
        del chunk
        gc.collect()
    
    return pd.concat(chunks, ignore_index=True)

def safe_divide(a: Any, b: Any, default: float = 0.0) -> float:
    """División segura que evita división por cero"""
    try:
        if b == 0 or pd.isna(b) or pd.isna(a):
            return default
        return float(a) / float(b)
    except (ValueError, TypeError, ZeroDivisionError):
        return default

def safe_percentage(a: Any, b: Any, default: float = 0.0) -> float:
    """Calcula porcentaje de forma segura"""
    return safe_divide(a * 100, b, default)

def batch_correlation_analysis(df: pd.DataFrame, numeric_cols: List[str], 
                              batch_size: int = 50) -> pd.DataFrame:
    """Calcula correlaciones en lotes para datasets grandes"""
    if len(numeric_cols) <= batch_size:
        return df[numeric_cols].corr()
    
    # Dividir en lotes
    batches = [numeric_cols[i:i + batch_size] for i in range(0, len(numeric_cols), batch_size)]
    correlation_results = []
    
    for i, batch1 in enumerate(batches):
        for j, batch2 in enumerate(batches):
            if i <= j:  # Solo calcular la mitad superior de la matriz
                corr_batch = df[batch1 + batch2].corr()
                correlation_results.append(corr_batch)
    
    # Combinar resultados
    return pd.concat(correlation_results, axis=1)

def optimize_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """Optimiza un DataFrame para uso en Streamlit"""
    # Limitar el tamaño si es muy grande
    max_size = PERFORMANCE_CONFIG.get('max_dataframe_size', 1000000)
    if len(df) > max_size:
        st.warning(f"⚠️ Dataset muy grande ({len(df)} filas). Mostrando solo las primeras {max_size} filas.")
        df = df.head(max_size)
    
    # Optimizar memoria
    df = memory_optimize_dataframe(df)
    
    return df

def create_progress_callback(total_steps: int):
    """Crea una función de callback para mostrar progreso"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def callback(step: int, message: str = ""):
        progress = step / total_steps
        progress_bar.progress(progress)
        if message:
            status_text.text(f"Paso {step}/{total_steps}: {message}")
    
    return callback

def memory_usage_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Obtiene información de uso de memoria de un DataFrame"""
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    return {
        'total_memory_mb': total_memory / 1024 / 1024,
        'memory_per_column': memory_usage.to_dict(),
        'memory_optimized': total_memory < 100 * 1024 * 1024  # Menos de 100MB
    }

def clear_memory():
    """Limpia la memoria del sistema"""
    gc.collect()

def parallel_apply(df: pd.DataFrame, func: Callable, axis: int = 0, 
                  n_jobs: int = -1) -> pd.DataFrame:
    """Aplica una función en paralelo a un DataFrame"""
    if not PERFORMANCE_CONFIG.get('enable_parallel_processing', True):
        return df.apply(func, axis=axis)
    
    try:
        from joblib import Parallel, delayed
        
        if axis == 0:
            # Aplicar por columnas
            results = Parallel(n_jobs=n_jobs)(
                delayed(func)(df[col]) for col in df.columns
            )
            return pd.DataFrame(dict(zip(df.columns, results)))
        else:
            # Aplicar por filas
            results = Parallel(n_jobs=n_jobs)(
                delayed(func)(row) for _, row in df.iterrows()
            )
            return pd.DataFrame(results)
    except ImportError:
        # Fallback a apply normal si joblib no está disponible
        return df.apply(func, axis=axis)

def cache_dataframe(func: Callable) -> Callable:
    """Decorador para cachear DataFrames usando Streamlit cache"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if PERFORMANCE_CONFIG.get('enable_caching', True):
            return st.cache_data(ttl=PERFORMANCE_CONFIG.get('cache_ttl', 3600))(func)(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

def validate_dataframe_size(df: pd.DataFrame, max_rows: int = None) -> bool:
    """Valida si el DataFrame es demasiado grande para procesar"""
    if max_rows is None:
        max_rows = PERFORMANCE_CONFIG.get('max_dataframe_size', 1000000)
    
    return len(df) <= max_rows

def create_memory_efficient_correlation(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Crea matriz de correlación de forma eficiente en memoria"""
    if len(numeric_cols) > 100:
        # Para datasets muy grandes, usar muestreo
        sample_size = min(10000, len(df))
        df_sample = df[numeric_cols].sample(n=sample_size, random_state=42)
        return df_sample.corr()
    else:
        return df[numeric_cols].corr()

def optimize_plotly_figure(fig, max_points: int = 10000):
    """Optimiza una figura de Plotly para mejor rendimiento"""
    try:
        import plotly.graph_objects as go
        
        # Si hay demasiados puntos, usar muestreo
        for trace in fig.data:
            if hasattr(trace, 'x') and hasattr(trace, 'y'):
                if len(trace.x) > max_points:
                    # Muestrear puntos
                    indices = np.random.choice(len(trace.x), max_points, replace=False)
                    trace.x = [trace.x[i] for i in sorted(indices)]
                    trace.y = [trace.y[i] for i in sorted(indices)]
        
        return fig
    except:
        return fig

def create_efficient_summary(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Crea un resumen eficiente del DataFrame"""
    summary = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isna().sum().sum(),
        'numeric_summary': {}
    }
    
    if numeric_cols:
        # Calcular estadísticas solo para columnas numéricas
        numeric_df = df[numeric_cols]
        summary['numeric_summary'] = {
            'mean': numeric_df.mean().to_dict(),
            'std': numeric_df.std().to_dict(),
            'min': numeric_df.min().to_dict(),
            'max': numeric_df.max().to_dict()
        }
    
    return summary

def batch_statistical_tests(df: pd.DataFrame, numeric_cols: List[str], 
                           test_func: Callable, batch_size: int = 20) -> Dict[str, Any]:
    """Ejecuta tests estadísticos en lotes para optimizar memoria"""
    results = {}
    
    for i in range(0, len(numeric_cols), batch_size):
        batch_cols = numeric_cols[i:i + batch_size]
        batch_df = df[batch_cols]
        
        for col in batch_cols:
            try:
                results[col] = test_func(batch_df[col])
            except Exception as e:
                results[col] = f"Error: {str(e)}"
        
        # Limpiar memoria
        del batch_df
        gc.collect()
    
    return results
