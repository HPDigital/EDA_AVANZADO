"""
Módulo de Generación de Reportes Avanzados
Incluye reportes completos con análisis estadístico, visualizaciones y exportación
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
import base64
from io import BytesIO
import json
import warnings
warnings.filterwarnings('ignore')

# Importar configuración
try:
    from config import THEME_TEMPLATE, PLOTLY_CONFIG
except ImportError:
    THEME_TEMPLATE = "plotly"
    PLOTLY_CONFIG = {}

# Importar módulos de análisis
try:
    from utils.smart_ingestion import create_data_quality_report
    from utils.advanced_analysis import get_advanced_numeric_summary, get_advanced_categorical_summary
    INGESTION_AVAILABLE = True
except ImportError:
    INGESTION_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import r2_score, accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def display_reports_analysis(df: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    """
    Función principal para mostrar generación de reportes avanzados
    """
    st.header("📊 Generador de Reportes Avanzados")
    
    # Configuración del reporte
    st.subheader("⚙️ Configuración del Reporte")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        report_type = st.selectbox(
            "Tipo de reporte:",
            ["Reporte Completo", "Reporte de Calidad", "Reporte Estadístico", "Reporte de Series Temporales", "Reporte de Machine Learning", "Reporte Ejecutivo"],
            key="report_type"
        )
    
    with col2:
        include_charts = st.checkbox("Incluir gráficos", value=True, key="include_charts")
    
    with col3:
        chart_format = st.selectbox(
            "Formato de gráficos:",
            ["PNG", "SVG", "HTML"],
            key="chart_format"
        )
    
    # Opciones avanzadas
    with st.expander("🔧 Opciones Avanzadas"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_vars_analysis = st.number_input("Máx. variables para análisis detallado", 5, 50, 20, key="max_vars")
            confidence_level = st.slider("Nivel de confianza (%)", 90, 99, 95, key="confidence_level")
        
        with col2:
            include_correlations = st.checkbox("Incluir matriz de correlaciones", value=True, key="include_correlations")
            include_outliers = st.checkbox("Incluir análisis de outliers", value=True, key="include_outliers")
        
        with col3:
            include_ml_insights = st.checkbox("Incluir insights de ML", value=True, key="include_ml_insights")
            include_time_analysis = st.checkbox("Incluir análisis temporal", value=True, key="include_time_analysis")
    
    # Generar reporte
    if st.button("🚀 Generar Reporte", key="generate_report"):
        with st.spinner("Generando reporte completo..."):
            try:
                # Generar reporte basado en el tipo seleccionado
                if report_type == "Reporte Completo":
                    report_data = generate_comprehensive_report(
                        df, numeric_cols, categorical_cols, 
                        include_charts, chart_format, max_vars_analysis,
                        confidence_level, include_correlations, include_outliers,
                        include_ml_insights, include_time_analysis
                    )
                elif report_type == "Reporte de Calidad":
                    report_data = generate_quality_report(df, numeric_cols, categorical_cols)
                elif report_type == "Reporte Estadístico":
                    report_data = generate_statistical_report(df, numeric_cols, categorical_cols, confidence_level)
                elif report_type == "Reporte de Series Temporales":
                    report_data = generate_time_series_report(df, numeric_cols, categorical_cols)
                elif report_type == "Reporte de Machine Learning":
                    report_data = generate_ml_report(df, numeric_cols, categorical_cols)
                elif report_type == "Reporte Ejecutivo":
                    report_data = generate_executive_report(df, numeric_cols, categorical_cols)
                
                # Mostrar reporte
                display_report(report_data, report_type)
                
                # Opciones de exportación
                display_export_options(report_data, report_type, chart_format)
                
            except Exception as e:
                st.error(f"Error generando reporte: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


def generate_comprehensive_report(df: pd.DataFrame, numeric_cols: list, categorical_cols: list, 
                                include_charts: bool, chart_format: str, max_vars: int,
                                confidence_level: float, include_correlations: bool, 
                                include_outliers: bool, include_ml_insights: bool, 
                                include_time_analysis: bool) -> dict:
    """Generar reporte comprensivo completo"""
    
    report_data = {
        'metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'report_type': 'Comprehensive Analysis Report',
            'total_observations': len(df),
            'total_variables': len(numeric_cols) + len(categorical_cols),
            'numeric_variables': len(numeric_cols),
            'categorical_variables': len(categorical_cols),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        },
        'sections': {}
    }
    
    # 1. Resumen Ejecutivo
    report_data['sections']['executive_summary'] = generate_executive_summary(df, numeric_cols, categorical_cols)
    
    # 2. Análisis de Calidad de Datos
    report_data['sections']['data_quality'] = generate_data_quality_section(df, numeric_cols, categorical_cols)
    
    # 3. Análisis Descriptivo
    report_data['sections']['descriptive_analysis'] = generate_descriptive_section(df, numeric_cols, categorical_cols, max_vars)
    
    # 4. Análisis de Correlaciones
    if include_correlations and len(numeric_cols) > 1:
        report_data['sections']['correlation_analysis'] = generate_correlation_section(df, numeric_cols)
    
    # 5. Análisis de Outliers
    if include_outliers and len(numeric_cols) > 0:
        report_data['sections']['outlier_analysis'] = generate_outlier_section(df, numeric_cols)
    
    # 6. Análisis Temporal
    if include_time_analysis:
        report_data['sections']['temporal_analysis'] = generate_temporal_section(df, numeric_cols, categorical_cols)
    
    # 7. Insights de Machine Learning
    if include_ml_insights and len(numeric_cols) > 1:
        report_data['sections']['ml_insights'] = generate_ml_insights_section(df, numeric_cols, categorical_cols)
    
    # 8. Recomendaciones
    report_data['sections']['recommendations'] = generate_recommendations_section(df, numeric_cols, categorical_cols)
    
    return report_data


def generate_executive_summary(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """Generar resumen ejecutivo"""
    
    summary = {
        'title': 'Resumen Ejecutivo',
        'content': {
            'dataset_overview': {
                'total_observations': len(df),
                'total_variables': len(numeric_cols) + len(categorical_cols),
                'numeric_variables': len(numeric_cols),
                'categorical_variables': len(categorical_cols),
                'completeness_rate': ((len(df) - df.isnull().sum().sum()) / (len(df) * len(df.columns))) * 100
            },
            'key_insights': [],
            'data_quality_score': 0,
            'main_findings': []
        }
    }
    
    # Calcular score de calidad
    completeness = summary['content']['dataset_overview']['completeness_rate']
    numeric_ratio = len(numeric_cols) / (len(numeric_cols) + len(categorical_cols)) if (len(numeric_cols) + len(categorical_cols)) > 0 else 0
    
    # Score basado en completitud y balance de tipos
    quality_score = (completeness * 0.7) + (numeric_ratio * 30)
    summary['content']['data_quality_score'] = min(100, max(0, quality_score))
    
    # Insights clave
    if len(numeric_cols) > 0:
        summary['content']['key_insights'].append(f"El dataset contiene {len(numeric_cols)} variables numéricas para análisis estadístico avanzado")
    
    if len(categorical_cols) > 0:
        summary['content']['key_insights'].append(f"Se identificaron {len(categorical_cols)} variables categóricas para análisis de frecuencias")
    
    # Hallazgos principales
    if completeness > 90:
        summary['content']['main_findings'].append("Excelente calidad de datos con alta completitud")
    elif completeness > 75:
        summary['content']['main_findings'].append("Buena calidad de datos con completitud aceptable")
    else:
        summary['content']['main_findings'].append("Calidad de datos requiere atención - completitud baja")
    
    return summary


def generate_data_quality_section(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """Generar sección de calidad de datos"""
    
    quality_section = {
        'title': 'Análisis de Calidad de Datos',
        'content': {
            'overall_quality': {},
            'variable_quality': [],
            'missing_data_patterns': {},
            'data_types_analysis': {},
            'recommendations': []
        }
    }
    
    # Calidad general
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    completeness_rate = ((total_cells - missing_cells) / total_cells) * 100
    
    quality_section['content']['overall_quality'] = {
        'completeness_rate': completeness_rate,
        'missing_data_percentage': (missing_cells / total_cells) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
    }
    
    # Calidad por variable
    for col in df.columns:
        if col.startswith('_'):  # Skip metadata columns
            continue
            
        series = df[col]
        missing_pct = (series.isnull().sum() / len(series)) * 100
        unique_ratio = series.nunique() / len(series)
        
        var_quality = {
            'variable': col,
            'type': str(series.dtype),
            'missing_percentage': missing_pct,
            'unique_ratio': unique_ratio,
            'completeness_score': 100 - missing_pct,
            'quality_issues': []
        }
        
        # Identificar problemas de calidad
        if missing_pct > 20:
            var_quality['quality_issues'].append("Alto porcentaje de valores faltantes")
        if unique_ratio < 0.01 and series.dtype == 'object':
            var_quality['quality_issues'].append("Muy pocos valores únicos - posible ID")
        if missing_pct > 0 and missing_pct < 5:
            var_quality['quality_issues'].append("Valores faltantes esporádicos")
        
        quality_section['content']['variable_quality'].append(var_quality)
    
    # Patrones de datos faltantes
    if missing_cells > 0:
        missing_patterns = df.isnull().sum().sort_values(ascending=False)
        quality_section['content']['missing_data_patterns'] = {
            'variables_with_missing': missing_patterns[missing_patterns > 0].to_dict(),
            'missing_data_matrix_available': len(df) < 1000  # Solo para datasets pequeños
        }
    
    # Análisis de tipos de datos
    type_counts = df.dtypes.value_counts().to_dict()
    quality_section['content']['data_types_analysis'] = {
        'type_distribution': {str(k): int(v) for k, v in type_counts.items()},
        'optimization_opportunities': []
    }
    
    # Recomendaciones
    if completeness_rate < 80:
        quality_section['content']['recommendations'].append("Implementar estrategia de imputación de valores faltantes")
    
    if df.duplicated().sum() > 0:
        quality_section['content']['recommendations'].append("Revisar y eliminar filas duplicadas")
    
    if len([v for v in quality_section['content']['variable_quality'] if len(v['quality_issues']) > 0]) > 0:
        quality_section['content']['recommendations'].append("Revisar variables con problemas de calidad identificados")
    
    return quality_section


def generate_descriptive_section(df: pd.DataFrame, numeric_cols: list, categorical_cols: list, max_vars: int) -> dict:
    """Generar sección de análisis descriptivo"""
    
    descriptive_section = {
        'title': 'Análisis Descriptivo',
        'content': {
            'numeric_summary': {},
            'categorical_summary': {},
            'distribution_analysis': {},
            'key_statistics': {}
        }
    }
    
    # Resumen numérico
    if len(numeric_cols) > 0:
        numeric_data = df[numeric_cols[:max_vars]]
        
        descriptive_section['content']['numeric_summary'] = {
            'count': len(numeric_cols),
            'variables_analyzed': numeric_cols[:max_vars],
            'descriptive_stats': numeric_data.describe().round(3).to_dict(),
            'correlation_matrix': numeric_data.corr().round(3).to_dict() if len(numeric_cols) > 1 else None
        }
        
        # Estadísticas clave
        descriptive_section['content']['key_statistics']['numeric'] = {
            'highest_variance_var': numeric_data.var().idxmax(),
            'highest_variance_value': float(numeric_data.var().max()),
            'most_correlated_pair': get_most_correlated_pair(numeric_data) if len(numeric_cols) > 1 else None,
            'skewness_analysis': get_skewness_analysis(numeric_data)
        }
    
    # Resumen categórico
    if len(categorical_cols) > 0:
        categorical_data = df[categorical_cols[:max_vars]]
        
        cat_summary = {}
        for col in categorical_data.columns:
            series = categorical_data[col].dropna()
            if len(series) > 0:
                cat_summary[col] = {
                    'unique_values': int(series.nunique()),
                    'most_frequent': series.mode().iloc[0] if len(series.mode()) > 0 else None,
                    'frequency': int(series.value_counts().iloc[0]) if len(series.value_counts()) > 0 else 0,
                    'frequency_percentage': float((series.value_counts().iloc[0] / len(series)) * 100) if len(series.value_counts()) > 0 else 0
                }
        
        descriptive_section['content']['categorical_summary'] = {
            'count': len(categorical_cols),
            'variables_analyzed': categorical_cols[:max_vars],
            'summary': cat_summary
        }
        
        # Estadísticas clave categóricas
        descriptive_section['content']['key_statistics']['categorical'] = {
            'highest_cardinality': get_highest_cardinality(categorical_data),
            'most_imbalanced': get_most_imbalanced(categorical_data)
        }
    
    # Análisis de distribución
    descriptive_section['content']['distribution_analysis'] = get_distribution_analysis(df, numeric_cols[:max_vars])
    
    return descriptive_section


def generate_correlation_section(df: pd.DataFrame, numeric_cols: list) -> dict:
    """Generar sección de análisis de correlaciones"""
    
    correlation_section = {
        'title': 'Análisis de Correlaciones',
        'content': {
            'correlation_matrix': {},
            'strong_correlations': [],
            'correlation_insights': [],
            'multicollinearity_analysis': {}
        }
    }
    
    if len(numeric_cols) < 2:
        return correlation_section
    
    # Matriz de correlación
    corr_matrix = df[numeric_cols].corr()
    correlation_section['content']['correlation_matrix'] = corr_matrix.round(3).to_dict()
    
    # Correlaciones fuertes
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # Correlación fuerte
                strong_corrs.append({
                    'variable1': corr_matrix.columns[i],
                    'variable2': corr_matrix.columns[j],
                    'correlation': float(corr_val),
                    'strength': 'Muy fuerte' if abs(corr_val) > 0.9 else 'Fuerte'
                })
    
    correlation_section['content']['strong_correlations'] = strong_corrs
    
    # Insights de correlación
    if len(strong_corrs) > 0:
        correlation_section['content']['correlation_insights'].append(f"Se encontraron {len(strong_corrs)} pares de variables con correlación fuerte")
    
    # Análisis de multicolinealidad
    correlation_section['content']['multicollinearity_analysis'] = analyze_multicollinearity(corr_matrix)
    
    return correlation_section


def generate_outlier_section(df: pd.DataFrame, numeric_cols: list) -> dict:
    """Generar sección de análisis de outliers"""
    
    outlier_section = {
        'title': 'Análisis de Outliers',
        'content': {
            'outlier_detection_methods': ['IQR', 'Z-Score'],
            'outlier_summary': {},
            'outlier_details': {},
            'outlier_impact': {}
        }
    }
    
    outlier_summary = {}
    outlier_details = {}
    
    for col in numeric_cols[:10]:  # Limitar a 10 variables
        series = df[col].dropna()
        if len(series) < 4:
            continue
        
        # Detección IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        # Detección Z-Score
        z_scores = np.abs((series - series.mean()) / series.std())
        z_outliers = series[z_scores > 3]
        
        outlier_summary[col] = {
            'total_values': len(series),
            'iqr_outliers_count': len(iqr_outliers),
            'iqr_outliers_percentage': (len(iqr_outliers) / len(series)) * 100,
            'zscore_outliers_count': len(z_outliers),
            'zscore_outliers_percentage': (len(z_outliers) / len(series)) * 100
        }
        
        outlier_details[col] = {
            'iqr_outliers': iqr_outliers.tolist() if len(iqr_outliers) > 0 else [],
            'zscore_outliers': z_outliers.tolist() if len(z_outliers) > 0 else []
        }
    
    outlier_section['content']['outlier_summary'] = outlier_summary
    outlier_section['content']['outlier_details'] = outlier_details
    
    return outlier_section


def generate_temporal_section(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """Generar sección de análisis temporal"""
    
    temporal_section = {
        'title': 'Análisis Temporal',
        'content': {
            'temporal_variables': [],
            'time_series_analysis': {},
            'trend_analysis': {},
            'seasonality_analysis': {}
        }
    }
    
    # Identificar variables temporales
    time_vars = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['date', 'fecha', 'time', 'tiempo', 'year', 'año']):
            time_vars.append(col)
    
    temporal_section['content']['temporal_variables'] = time_vars
    
    if len(time_vars) > 0 and STATSMODELS_AVAILABLE:
        # Análisis de series temporales para variables numéricas
        for time_var in time_vars[:1]:  # Solo primera variable temporal
            for num_var in numeric_cols[:3]:  # Solo primeras 3 variables numéricas
                try:
                    # Preparar datos temporales
                    temp_df = df[[time_var, num_var]].dropna()
                    if len(temp_df) < 10:
                        continue
                    
                    temp_df[time_var] = pd.to_datetime(temp_df[time_var], errors='coerce')
                    temp_df = temp_df.sort_values(time_var)
                    
                    # Análisis de tendencia
                    slope, r_squared = calculate_trend(temp_df[num_var])
                    
                    temporal_section['content']['trend_analysis'][f"{time_var}_{num_var}"] = {
                        'slope': slope,
                        'r_squared': r_squared,
                        'trend_direction': 'Creciente' if slope > 0 else 'Decreciente' if slope < 0 else 'Estable'
                    }
                    
                except Exception as e:
                    continue
    
    return temporal_section


def generate_ml_insights_section(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """Generar sección de insights de Machine Learning"""
    
    ml_section = {
        'title': 'Insights de Machine Learning',
        'content': {
            'feature_importance': {},
            'clustering_insights': {},
            'dimensionality_analysis': {},
            'predictive_potential': {}
        }
    }
    
    if not SKLEARN_AVAILABLE or len(numeric_cols) < 2:
        ml_section['content']['note'] = "Análisis de ML no disponible - requiere sklearn y al menos 2 variables numéricas"
        return ml_section
    
    try:
        # Preparar datos
        ml_data = df[numeric_cols].dropna()
        if len(ml_data) < 10:
            ml_section['content']['note'] = "Datos insuficientes para análisis de ML"
            return ml_section
        
        # Análisis de importancia de características (si hay variable objetivo)
        if len(numeric_cols) > 2:
            # Usar la última variable como objetivo (ejemplo)
            target_col = numeric_cols[-1]
            feature_cols = numeric_cols[:-1]
            
            X = ml_data[feature_cols]
            y = ml_data[target_col]
            
            # Random Forest para importancia de características
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            feature_importance = dict(zip(feature_cols, rf.feature_importances_))
            ml_section['content']['feature_importance'] = {
                'target_variable': target_col,
                'feature_importance': feature_importance,
                'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        # Análisis de clustering
        if len(ml_data) > 10:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(ml_data)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            ml_section['content']['clustering_insights'] = {
                'n_clusters': 3,
                'cluster_sizes': [int(np.sum(clusters == i)) for i in range(3)],
                'inertia': float(kmeans.inertia_)
            }
        
        # Análisis de dimensionalidad
        if len(numeric_cols) > 2:
            pca = PCA()
            pca.fit(ml_data)
            
            explained_variance_ratio = pca.explained_variance_ratio_.tolist()
            cumulative_variance = np.cumsum(explained_variance_ratio).tolist()
            
            ml_section['content']['dimensionality_analysis'] = {
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance': cumulative_variance,
                'components_95_variance': int(np.argmax(cumulative_variance >= 0.95) + 1),
                'reduction_potential': len(numeric_cols) - int(np.argmax(cumulative_variance >= 0.95) + 1)
            }
    
    except Exception as e:
        ml_section['content']['error'] = f"Error en análisis de ML: {str(e)}"
    
    return ml_section


def generate_recommendations_section(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """Generar sección de recomendaciones"""
    
    recommendations = {
        'title': 'Recomendaciones y Próximos Pasos',
        'content': {
            'data_quality_recommendations': [],
            'analysis_recommendations': [],
            'modeling_recommendations': [],
            'action_items': []
        }
    }
    
    # Recomendaciones de calidad de datos
    completeness_rate = ((len(df) - df.isnull().sum().sum()) / (len(df) * len(df.columns))) * 100
    
    if completeness_rate < 90:
        recommendations['content']['data_quality_recommendations'].append(
            f"Implementar estrategia de imputación de valores faltantes (completitud actual: {completeness_rate:.1f}%)"
        )
    
    if df.duplicated().sum() > 0:
        recommendations['content']['data_quality_recommendations'].append(
            f"Eliminar {df.duplicated().sum()} filas duplicadas identificadas"
        )
    
    # Recomendaciones de análisis
    if len(numeric_cols) > 5:
        recommendations['content']['analysis_recommendations'].append(
            "Considerar análisis de componentes principales (PCA) para reducción de dimensionalidad"
        )
    
    if len(categorical_cols) > 0:
        recommendations['content']['analysis_recommendations'].append(
            "Realizar análisis de frecuencias y asociación entre variables categóricas"
        )
    
    # Recomendaciones de modelado
    if len(numeric_cols) > 2:
        recommendations['content']['modeling_recommendations'].append(
            "Adecuado para modelos de regresión y clustering"
        )
    
    if len(categorical_cols) > 1:
        recommendations['content']['modeling_recommendations'].append(
            "Considerar modelos de clasificación y análisis de asociación"
        )
    
    # Items de acción
    recommendations['content']['action_items'] = [
        "Validar resultados con stakeholders del negocio",
        "Implementar monitoreo continuo de calidad de datos",
        "Documentar decisiones de preprocesamiento",
        "Considerar análisis adicionales según objetivos específicos"
    ]
    
    return recommendations


# Funciones auxiliares
def get_most_correlated_pair(df: pd.DataFrame) -> dict:
    """Obtener el par de variables más correlacionadas"""
    corr_matrix = df.corr()
    
    # Excluir diagonal
    corr_matrix_no_diag = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
    
    # Encontrar máximo absoluto
    max_corr = corr_matrix_no_diag.abs().max().max()
    
    # Encontrar las variables
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) == max_corr:
                return {
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': float(corr_matrix.iloc[i, j])
                }
    
    return None


def get_skewness_analysis(df: pd.DataFrame) -> dict:
    """Análisis de asimetría"""
    skewness = df.skew()
    
    return {
        'most_skewed': skewness.abs().idxmax(),
        'skewness_value': float(skewness.abs().max()),
        'symmetric_variables': skewness[abs(skewness) < 0.5].index.tolist(),
        'highly_skewed': skewness[abs(skewness) > 1].index.tolist()
    }


def get_highest_cardinality(df: pd.DataFrame) -> dict:
    """Variable categórica con mayor cardinalidad"""
    cardinalities = df.nunique()
    max_card_var = cardinalities.idxmax()
    
    return {
        'variable': max_card_var,
        'unique_values': int(cardinalities[max_card_var])
    }


def get_most_imbalanced(df: pd.DataFrame) -> dict:
    """Variable categórica más desbalanceada"""
    max_imbalance = 0
    most_imbalanced_var = None
    
    for col in df.columns:
        value_counts = df[col].value_counts()
        if len(value_counts) > 1:
            imbalance = value_counts.iloc[0] / value_counts.sum()
            if imbalance > max_imbalance:
                max_imbalance = imbalance
                most_imbalanced_var = col
    
    return {
        'variable': most_imbalanced_var,
        'imbalance_ratio': float(max_imbalance) if most_imbalanced_var else 0
    }


def get_distribution_analysis(df: pd.DataFrame, numeric_cols: list) -> dict:
    """Análisis de distribución"""
    if len(numeric_cols) == 0:
        return {}
    
    distributions = {}
    for col in numeric_cols[:5]:  # Limitar a 5 variables
        series = df[col].dropna()
        if len(series) < 4:
            continue
        
        distributions[col] = {
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'skewness': float(series.skew()),
            'kurtosis': float(series.kurtosis()),
            'distribution_type': get_distribution_type(series)
        }
    
    return distributions


def get_distribution_type(series: pd.Series) -> str:
    """Determinar tipo de distribución"""
    skewness = series.skew()
    kurtosis = series.kurtosis()
    
    if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
        return "Normal"
    elif abs(skewness) > 1:
        return "Altamente sesgada"
    elif abs(kurtosis) > 3:
        return "Leptocúrtica (picos altos)"
    elif abs(kurtosis) < -1:
        return "Platicúrtica (picos bajos)"
    else:
        return "Moderadamente sesgada"


def analyze_multicollinearity(corr_matrix: pd.DataFrame) -> dict:
    """Analizar multicolinealidad"""
    # Calcular VIF aproximado (simplificado)
    vif_data = {}
    
    for i, col in enumerate(corr_matrix.columns):
        # VIF aproximado usando R²
        other_cols = [c for c in corr_matrix.columns if c != col]
        if len(other_cols) > 0:
            # R² aproximado usando correlación máxima
            max_corr = corr_matrix[col][other_cols].abs().max()
            vif = 1 / (1 - max_corr**2) if max_corr < 0.99 else float('inf')
            vif_data[col] = float(vif)
    
    high_vif = {k: v for k, v in vif_data.items() if v > 10}
    
    return {
        'vif_scores': vif_data,
        'high_multicollinearity': list(high_vif.keys()),
        'recommendation': "Considerar eliminar variables con VIF > 10" if high_vif else "No hay problemas de multicolinealidad severa"
    }


def calculate_trend(series: pd.Series) -> tuple:
    """Calcular tendencia de una serie"""
    try:
        x = np.arange(len(series))
        y = series.values
        
        # Regresión lineal simple
        slope = np.polyfit(x, y, 1)[0]
        
        # R²
        y_pred = slope * x + np.polyfit(x, y, 1)[1]
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return slope, r_squared
    except:
        return 0, 0


def display_report(report_data: dict, report_type: str):
    """Mostrar el reporte generado"""
    
    st.subheader(f"📊 {report_type}")
    
    # Metadatos del reporte
    metadata = report_data.get('metadata', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Observaciones", f"{metadata.get('total_observations', 0):,}")
    
    with col2:
        st.metric("Variables", f"{metadata.get('total_variables', 0)}")
    
    with col3:
        st.metric("Memoria (MB)", f"{metadata.get('memory_usage_mb', 0):.1f}")
    
    with col4:
        st.metric("Generado", metadata.get('generated_at', 'N/A'))
    
    # Mostrar secciones del reporte
    sections = report_data.get('sections', {})
    
    # Resumen ejecutivo
    if 'executive_summary' in sections:
        display_executive_summary(sections['executive_summary'])
    
    # Calidad de datos
    if 'data_quality' in sections:
        display_data_quality_section(sections['data_quality'])
    
    # Análisis descriptivo
    if 'descriptive_analysis' in sections:
        display_descriptive_section(sections['descriptive_analysis'])
    
    # Análisis de correlaciones
    if 'correlation_analysis' in sections:
        display_correlation_section(sections['correlation_analysis'])
    
    # Análisis de outliers
    if 'outlier_analysis' in sections:
        display_outlier_section(sections['outlier_analysis'])
    
    # Análisis temporal
    if 'temporal_analysis' in sections:
        display_temporal_section(sections['temporal_analysis'])
    
    # Insights de ML
    if 'ml_insights' in sections:
        display_ml_insights_section(sections['ml_insights'])
    
    # Recomendaciones
    if 'recommendations' in sections:
        display_recommendations_section(sections['recommendations'])


def display_executive_summary(section: dict):
    """Mostrar resumen ejecutivo"""
    
    with st.expander("📋 Resumen Ejecutivo", expanded=True):
        
        # Score de calidad
        quality_score = section['content']['data_quality_score']
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write("**Puntuación de Calidad de Datos:**")
            progress_bar = st.progress(quality_score / 100)
            st.caption(f"Score: {quality_score:.1f}/100")
        
        with col2:
            overview = section['content']['dataset_overview']
            st.metric("Completitud", f"{overview['completeness_rate']:.1f}%")
        
        with col3:
            st.metric("Variables", f"{overview['total_variables']}")
        
        # Insights clave
        st.write("**Insights Clave:**")
        for insight in section['content']['key_insights']:
            st.write(f"• {insight}")
        
        # Hallazgos principales
        st.write("**Hallazgos Principales:**")
        for finding in section['content']['main_findings']:
            st.info(f"💡 {finding}")


def display_data_quality_section(section: dict):
    """Mostrar sección de calidad de datos"""
    
    with st.expander("🔍 Análisis de Calidad de Datos"):
        
        # Calidad general
        overall_quality = section['content']['overall_quality']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Completitud", f"{overall_quality['completeness_rate']:.1f}%")
        
        with col2:
            st.metric("Datos Faltantes", f"{overall_quality['missing_data_percentage']:.1f}%")
        
        with col3:
            st.metric("Filas Duplicadas", f"{overall_quality['duplicate_rows']:,}")
        
        with col4:
            st.metric("Duplicados (%)", f"{overall_quality['duplicate_percentage']:.1f}%")
        
        # Tabla de calidad por variable
        st.write("**Calidad por Variable:**")
        
        var_quality = section['content']['variable_quality']
        quality_df = pd.DataFrame(var_quality)
        
        if not quality_df.empty:
            # Crear columnas para mostrar
            display_df = quality_df[['variable', 'type', 'missing_percentage', 'completeness_score']].copy()
            display_df.columns = ['Variable', 'Tipo', 'Faltantes (%)', 'Score Completitud']
            
            # Agregar columna de problemas
            problems = []
            for _, row in quality_df.iterrows():
                if len(row['quality_issues']) > 0:
                    problems.append(", ".join(row['quality_issues']))
                else:
                    problems.append("✅ Sin problemas")
            
            display_df['Problemas'] = problems
            
            st.dataframe(display_df, width='stretch')
        
        # Recomendaciones
        recommendations = section['content']['recommendations']
        if recommendations:
            st.write("**Recomendaciones:**")
            for rec in recommendations:
                st.warning(f"⚠️ {rec}")


def display_descriptive_section(section: dict):
    """Mostrar sección descriptiva"""
    
    with st.expander("📊 Análisis Descriptivo"):
        
        # Resumen numérico
        numeric_summary = section['content']['numeric_summary']
        if numeric_summary:
            st.write("**Variables Numéricas:**")
            
            # Estadísticas descriptivas
            desc_stats = numeric_summary['descriptive_stats']
            if desc_stats:
                desc_df = pd.DataFrame(desc_stats)
                st.dataframe(desc_df, width='stretch')
            
            # Estadísticas clave
            key_stats = section['content']['key_statistics']['numeric']
            if key_stats:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Mayor varianza:** {key_stats['highest_variance_var']} ({key_stats['highest_variance_value']:.3f})")
                
                with col2:
                    if key_stats['most_correlated_pair']:
                        pair = key_stats['most_correlated_pair']
                        st.write(f"**Más correlacionadas:** {pair['var1']} ↔ {pair['var2']} ({pair['correlation']:.3f})")
                
                # Análisis de asimetría
                skewness = key_stats['skewness_analysis']
                st.write(f"**Más asimétrica:** {skewness['most_skewed']} (skewness: {skewness['skewness_value']:.3f})")
        
        # Resumen categórico
        categorical_summary = section['content']['categorical_summary']
        if categorical_summary:
            st.write("**Variables Categóricas:**")
            
            cat_df = pd.DataFrame(categorical_summary['summary']).T
            if not cat_df.empty:
                st.dataframe(cat_df, width='stretch')
            
            # Estadísticas clave categóricas
            key_stats_cat = section['content']['key_statistics']['categorical']
            if key_stats_cat:
                col1, col2 = st.columns(2)
                
                with col1:
                    highest_card = key_stats_cat['highest_cardinality']
                    st.write(f"**Mayor cardinalidad:** {highest_card['variable']} ({highest_card['unique_values']} valores únicos)")
                
                with col2:
                    most_imbalanced = key_stats_cat['most_imbalanced']
                    if most_imbalanced['variable']:
                        st.write(f"**Más desbalanceada:** {most_imbalanced['variable']} ({most_imbalanced['imbalance_ratio']:.1%})")


def display_correlation_section(section: dict):
    """Mostrar sección de correlaciones"""
    
    with st.expander("🔗 Análisis de Correlaciones"):
        
        # Correlaciones fuertes
        strong_corrs = section['content']['strong_correlations']
        if strong_corrs:
            st.write("**Correlaciones Fuertes Detectadas:**")
            
            for corr in strong_corrs:
                strength_color = "🔴" if corr['strength'] == 'Muy fuerte' else "🟡"
                st.write(f"{strength_color} {corr['variable1']} ↔ {corr['variable2']}: {corr['correlation']:.3f}")
        
        # Matriz de correlaciones
        corr_matrix = section['content']['correlation_matrix']
        if corr_matrix:
            st.write("**Matriz de Correlaciones:**")
            
            corr_df = pd.DataFrame(corr_matrix)
            
            # Crear heatmap
            fig = px.imshow(
                corr_df,
                text_auto=True,
                aspect="auto",
                title="Matriz de Correlaciones"
            )
            fig.update_layout(template=THEME_TEMPLATE)
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
        
        # Análisis de multicolinealidad
        multicollinearity = section['content']['multicollinearity_analysis']
        if multicollinearity and multicollinearity['recommendation']:
            st.info(f"💡 {multicollinearity['recommendation']}")


def display_outlier_section(section: dict):
    """Mostrar sección de outliers"""
    
    with st.expander("🎯 Análisis de Outliers"):
        
        outlier_summary = section['content']['outlier_summary']
        
        if outlier_summary:
            # Crear DataFrame de resumen
            summary_data = []
            for var, stats in outlier_summary.items():
                summary_data.append({
                    'Variable': var,
                    'Total Valores': stats['total_values'],
                    'Outliers IQR': stats['iqr_outliers_count'],
                    'Outliers IQR (%)': f"{stats['iqr_outliers_percentage']:.1f}%",
                    'Outliers Z-Score': stats['zscore_outliers_count'],
                    'Outliers Z-Score (%)': f"{stats['zscore_outliers_percentage']:.1f}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, width='stretch')
            
            # Identificar variables con más outliers
            if not summary_df.empty:
                max_iqr_var = summary_df.loc[summary_df['Outliers IQR'].idxmax(), 'Variable']
                max_iqr_pct = summary_df['Outliers IQR (%)'].max()
                
                st.warning(f"⚠️ Variable con más outliers (IQR): {max_iqr_var} ({max_iqr_pct})")


def display_temporal_section(section: dict):
    """Mostrar sección temporal"""
    
    with st.expander("📈 Análisis Temporal"):
        
        temporal_vars = section['content']['temporal_variables']
        if temporal_vars:
            st.write("**Variables Temporales Identificadas:**")
            for var in temporal_vars:
                st.write(f"• {var}")
        
        # Análisis de tendencias
        trend_analysis = section['content']['trend_analysis']
        if trend_analysis:
            st.write("**Análisis de Tendencias:**")
            
            for analysis_key, trend_data in trend_analysis.items():
                direction_emoji = "📈" if trend_data['trend_direction'] == 'Creciente' else "📉" if trend_data['trend_direction'] == 'Decreciente' else "➡️"
                st.write(f"{direction_emoji} {analysis_key}: {trend_data['trend_direction']} (R²: {trend_data['r_squared']:.3f})")


def display_ml_insights_section(section: dict):
    """Mostrar sección de insights de ML"""
    
    with st.expander("🤖 Insights de Machine Learning"):
        
        # Importancia de características
        feature_importance = section['content']['feature_importance']
        if feature_importance:
            st.write("**Importancia de Características:**")
            st.write(f"Variable objetivo: {feature_importance['target_variable']}")
            
            top_features = feature_importance['top_features']
            if top_features:
                features_df = pd.DataFrame(top_features, columns=['Característica', 'Importancia'])
                st.dataframe(features_df, width='stretch')
        
        # Insights de clustering
        clustering = section['content']['clustering_insights']
        if clustering:
            st.write("**Análisis de Clustering (K-Means):**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Número de clusters", clustering['n_clusters'])
            with col2:
                st.metric("Inercia", f"{clustering['inertia']:.2f}")
            
            # Tamaños de clusters
            cluster_sizes = clustering['cluster_sizes']
            st.write("**Distribución de clusters:**")
            for i, size in enumerate(cluster_sizes):
                st.write(f"Cluster {i+1}: {size} observaciones")
        
        # Análisis de dimensionalidad
        dimensionality = section['content']['dimensionality_analysis']
        if dimensionality:
            st.write("**Análisis de Dimensionalidad (PCA):**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Componentes para 95% varianza", dimensionality['components_95_variance'])
            with col2:
                st.metric("Potencial de reducción", f"{dimensionality['reduction_potential']} variables")
            
            # Gráfico de varianza explicada
            explained_var = dimensionality['explained_variance_ratio'][:10]  # Primeras 10 componentes
            cumulative_var = dimensionality['cumulative_variance'][:10]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(range(1, len(explained_var) + 1)),
                y=explained_var,
                name='Varianza Explicada',
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumulative_var) + 1)),
                y=cumulative_var,
                name='Varianza Acumulada',
                yaxis='y2',
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title="Varianza Explicada por Componente",
                xaxis_title="Componente",
                yaxis=dict(title="Varianza Explicada", side="left"),
                yaxis2=dict(title="Varianza Acumulada", side="right", overlaying="y"),
                template=THEME_TEMPLATE
            )
            
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)


def display_recommendations_section(section: dict):
    """Mostrar sección de recomendaciones"""
    
    with st.expander("💡 Recomendaciones y Próximos Pasos", expanded=True):
        
        # Recomendaciones de calidad de datos
        data_quality_recs = section['content']['data_quality_recommendations']
        if data_quality_recs:
            st.write("**🔧 Recomendaciones de Calidad de Datos:**")
            for rec in data_quality_recs:
                st.write(f"• {rec}")
        
        # Recomendaciones de análisis
        analysis_recs = section['content']['analysis_recommendations']
        if analysis_recs:
            st.write("**📊 Recomendaciones de Análisis:**")
            for rec in analysis_recs:
                st.write(f"• {rec}")
        
        # Recomendaciones de modelado
        modeling_recs = section['content']['modeling_recommendations']
        if modeling_recs:
            st.write("**🤖 Recomendaciones de Modelado:**")
            for rec in modeling_recs:
                st.write(f"• {rec}")
        
        # Items de acción
        action_items = section['content']['action_items']
        if action_items:
            st.write("**✅ Items de Acción:**")
            for item in action_items:
                st.write(f"• {item}")


def display_export_options(report_data: dict, report_type: str, chart_format: str):
    """Mostrar opciones de exportación"""
    
    st.subheader("💾 Opciones de Exportación")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Exportar como Markdown
        markdown_content = generate_markdown_report(report_data, report_type)
        
        st.download_button(
            label="📄 Descargar Markdown",
            data=markdown_content,
            file_name=f"reporte_{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            key="download_markdown"
        )
    
    with col2:
        # Exportar como JSON
        json_content = json.dumps(report_data, indent=2, ensure_ascii=False, default=str)
        
        st.download_button(
            label="📋 Descargar JSON",
            data=json_content,
            file_name=f"reporte_{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            key="download_json"
        )
    
    with col3:
        # Exportar como HTML
        html_content = generate_html_report(report_data, report_type)
        
        st.download_button(
            label="🌐 Descargar HTML",
            data=html_content,
            file_name=f"reporte_{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            key="download_html"
        )


def generate_markdown_report(report_data: dict, report_type: str) -> str:
    """Generar reporte en formato Markdown"""
    
    metadata = report_data.get('metadata', {})
    sections = report_data.get('sections', {})
    
    markdown = f"""# {report_type}
**Generado:** {metadata.get('generated_at', 'N/A')}

## Metadatos del Dataset
- **Observaciones:** {metadata.get('total_observations', 0):,}
- **Variables:** {metadata.get('total_variables', 0)}
- **Variables Numéricas:** {metadata.get('numeric_variables', 0)}
- **Variables Categóricas:** {metadata.get('categorical_variables', 0)}
- **Memoria:** {metadata.get('memory_usage_mb', 0):.1f} MB

"""
    
    # Resumen ejecutivo
    if 'executive_summary' in sections:
        exec_summary = sections['executive_summary']
        markdown += f"""## Resumen Ejecutivo

**Puntuación de Calidad:** {exec_summary['content']['data_quality_score']:.1f}/100

### Insights Clave
"""
        for insight in exec_summary['content']['key_insights']:
            markdown += f"- {insight}\n"
        
        markdown += "\n### Hallazgos Principales\n"
        for finding in exec_summary['content']['main_findings']:
            markdown += f"- {finding}\n"
        
        markdown += "\n"
    
    # Calidad de datos
    if 'data_quality' in sections:
        quality = sections['data_quality']
        overall = quality['content']['overall_quality']
        
        markdown += f"""## Análisis de Calidad de Datos

- **Completitud:** {overall['completeness_rate']:.1f}%
- **Datos Faltantes:** {overall['missing_data_percentage']:.1f}%
- **Filas Duplicadas:** {overall['duplicate_rows']:,}

### Recomendaciones de Calidad
"""
        for rec in quality['content']['recommendations']:
            markdown += f"- {rec}\n"
        
        markdown += "\n"
    
    # Recomendaciones
    if 'recommendations' in sections:
        recs = sections['recommendations']
        
        markdown += """## Recomendaciones y Próximos Pasos

### Items de Acción
"""
        for item in recs['content']['action_items']:
            markdown += f"- {item}\n"
        
        markdown += "\n"
    
    return markdown


def generate_html_report(report_data: dict, report_type: str) -> str:
    """Generar reporte en formato HTML"""
    
    metadata = report_data.get('metadata', {})
    sections = report_data.get('sections', {})
    
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_type}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; }}
        h3 {{ color: #7f8c8d; }}
        .metric {{ background: #ecf0f1; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .recommendation {{ background: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
        .insight {{ background: #d1ecf1; padding: 10px; margin: 5px 0; border-left: 4px solid #17a2b8; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{report_type}</h1>
    <p><strong>Generado:</strong> {metadata.get('generated_at', 'N/A')}</p>
    
    <h2>Metadatos del Dataset</h2>
    <div class="metric">
        <strong>Observaciones:</strong> {metadata.get('total_observations', 0):,}<br>
        <strong>Variables:</strong> {metadata.get('total_variables', 0)}<br>
        <strong>Variables Numéricas:</strong> {metadata.get('numeric_variables', 0)}<br>
        <strong>Variables Categóricas:</strong> {metadata.get('categorical_variables', 0)}<br>
        <strong>Memoria:</strong> {metadata.get('memory_usage_mb', 0):.1f} MB
    </div>
"""
    
    # Resumen ejecutivo
    if 'executive_summary' in sections:
        exec_summary = sections['executive_summary']
        quality_score = exec_summary['content']['data_quality_score']
        
        html += f"""
    <h2>Resumen Ejecutivo</h2>
    <div class="metric">
        <strong>Puntuación de Calidad:</strong> {quality_score:.1f}/100
    </div>
    
    <h3>Insights Clave</h3>
"""
        for insight in exec_summary['content']['key_insights']:
            html += f"    <div class='insight'>{insight}</div>\n"
        
        html += "    <h3>Hallazgos Principales</h3>\n"
        for finding in exec_summary['content']['main_findings']:
            html += f"    <div class='insight'>{finding}</div>\n"
    
    # Recomendaciones
    if 'recommendations' in sections:
        recs = sections['recommendations']
        
        html += """
    <h2>Recomendaciones y Próximos Pasos</h2>
    <h3>Items de Acción</h3>
"""
        for item in recs['content']['action_items']:
            html += f"    <div class='recommendation'>{item}</div>\n"
    
    html += """
</body>
</html>
"""
    
    return html


# Funciones para reportes específicos
def generate_quality_report(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """Generar reporte específico de calidad de datos"""
    return {
        'metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'report_type': 'Quality Report',
            'total_observations': len(df)
        },
        'sections': {
            'data_quality': generate_data_quality_section(df, numeric_cols, categorical_cols)
        }
    }


def generate_statistical_report(df: pd.DataFrame, numeric_cols: list, categorical_cols: list, confidence_level: float) -> dict:
    """Generar reporte específico estadístico"""
    return {
        'metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'report_type': 'Statistical Report',
            'total_observations': len(df),
            'confidence_level': confidence_level
        },
        'sections': {
            'descriptive_analysis': generate_descriptive_section(df, numeric_cols, categorical_cols, 20),
            'correlation_analysis': generate_correlation_section(df, numeric_cols),
            'outlier_analysis': generate_outlier_section(df, numeric_cols)
        }
    }


def generate_time_series_report(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """Generar reporte específico de series temporales"""
    return {
        'metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'report_type': 'Time Series Report',
            'total_observations': len(df)
        },
        'sections': {
            'temporal_analysis': generate_temporal_section(df, numeric_cols, categorical_cols),
            'descriptive_analysis': generate_descriptive_section(df, numeric_cols, categorical_cols, 10)
        }
    }


def generate_ml_report(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """Generar reporte específico de Machine Learning"""
    return {
        'metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'report_type': 'Machine Learning Report',
            'total_observations': len(df)
        },
        'sections': {
            'ml_insights': generate_ml_insights_section(df, numeric_cols, categorical_cols),
            'descriptive_analysis': generate_descriptive_section(df, numeric_cols, categorical_cols, 15),
            'correlation_analysis': generate_correlation_section(df, numeric_cols)
        }
    }


def display_advanced_reports_analysis(df: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    """Análisis avanzado de reportes - Función principal"""
    
    st.header("📋 Generador de Reportes Avanzados")
    
    # Configuración del reporte
    st.subheader("⚙️ Configuración del Reporte")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Tipo de Reporte",
            [
                "Reporte Completo",
                "Reporte de Calidad",
                "Reporte Estadístico",
                "Reporte de Series Temporales",
                "Reporte de Machine Learning"
            ],
            key="report_type_selector"
        )
        
        confidence_level = st.slider(
            "Nivel de Confianza (%)",
            min_value=80,
            max_value=99,
            value=95,
            step=1,
            key="confidence_level_slider"
        )
    
    with col2:
        chart_format = st.selectbox(
            "Formato de Gráficos",
            ["PNG", "SVG", "HTML"],
            key="chart_format_selector"
        )
        
        include_recommendations = st.checkbox(
            "Incluir Recomendaciones",
            value=True,
            key="include_recommendations_checkbox"
        )
    
    # Configuraciones adicionales
    with st.expander("🔧 Configuraciones Avanzadas"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_vars_descriptive = st.number_input(
                "Máx. Variables Descriptivas",
                min_value=5,
                max_value=50,
                value=20,
                key="max_vars_descriptive_input"
            )
            
            correlation_threshold = st.slider(
                "Umbral Correlación",
                min_value=0.1,
                max_value=0.9,
                value=0.7,
                step=0.1,
                key="correlation_threshold_slider"
            )
        
        with col2:
            outlier_method = st.selectbox(
                "Método de Outliers",
                ["IQR", "Z-Score", "Ambos"],
                key="outlier_method_selector"
            )
            
            include_ml_insights = st.checkbox(
                "Incluir Insights de ML",
                value=True,
                key="include_ml_insights_checkbox"
            )
    
    # Botón para generar reporte
    if st.button("🚀 Generar Reporte", key="generate_report_button"):
        
        with st.spinner("Generando reporte avanzado..."):
            
            # Generar reporte según el tipo seleccionado
            if report_type == "Reporte Completo":
                report_data = generate_comprehensive_report(
                    df, numeric_cols, categorical_cols, 
                    confidence_level, max_vars_descriptive,
                    correlation_threshold, outlier_method,
                    include_ml_insights, include_recommendations
                )
            
            elif report_type == "Reporte de Calidad":
                report_data = generate_quality_report(df, numeric_cols, categorical_cols)
            
            elif report_type == "Reporte Estadístico":
                report_data = generate_statistical_report(
                    df, numeric_cols, categorical_cols, confidence_level
                )
            
            elif report_type == "Reporte de Series Temporales":
                report_data = generate_time_series_report(df, numeric_cols, categorical_cols)
            
            elif report_type == "Reporte de Machine Learning":
                report_data = generate_ml_report(df, numeric_cols, categorical_cols)
            
            # Mostrar el reporte
            display_report(report_data, report_type)
            
            # Mostrar opciones de exportación
            display_export_options(report_data, report_type, chart_format)
            
            # Mostrar resumen de métricas
            display_report_summary(report_data)
            
            st.success("✅ Reporte generado exitosamente!")


def display_report_summary(report_data: dict):
    """Mostrar resumen de métricas del reporte"""
    
    st.subheader("📊 Resumen de Métricas")
    
    metadata = report_data.get('metadata', {})
    sections = report_data.get('sections', {})
    
    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Tiempo Generación", 
            f"{datetime.now().strftime('%H:%M:%S')}"
        )
    
    with col2:
        total_sections = len(sections)
        st.metric("Secciones", total_sections)
    
    with col3:
        if 'executive_summary' in sections:
            quality_score = sections['executive_summary']['content']['data_quality_score']
            st.metric("Score Calidad", f"{quality_score:.1f}/100")
        else:
            st.metric("Score Calidad", "N/A")
    
    with col4:
        if 'data_quality' in sections:
            completeness = sections['data_quality']['content']['overall_quality']['completeness_rate']
            st.metric("Completitud", f"{completeness:.1f}%")
        else:
            st.metric("Completitud", "N/A")
    
    # Resumen de secciones incluidas
    st.write("**Secciones Incluidas:**")
    
    section_names = {
        'executive_summary': '📋 Resumen Ejecutivo',
        'data_quality': '🔍 Calidad de Datos',
        'descriptive_analysis': '📊 Análisis Descriptivo',
        'correlation_analysis': '🔗 Análisis de Correlaciones',
        'outlier_analysis': '🎯 Análisis de Outliers',
        'temporal_analysis': '📈 Análisis Temporal',
        'ml_insights': '🤖 Insights de ML',
        'recommendations': '💡 Recomendaciones'
    }
    
    included_sections = []
    for section_key, section_name in section_names.items():
        if section_key in sections:
            included_sections.append(section_name)
    
    if included_sections:
        for section in included_sections:
            st.write(f"✅ {section}")
    else:
        st.info("No hay secciones disponibles para mostrar")


def generate_comprehensive_report(
    df: pd.DataFrame, 
    numeric_cols: list, 
    categorical_cols: list,
    confidence_level: float = 95,
    max_vars_descriptive: int = 20,
    correlation_threshold: float = 0.7,
    outlier_method: str = "Ambos",
    include_ml_insights: bool = True,
    include_recommendations: bool = True
) -> dict:
    """Generar reporte completo con todas las secciones"""
    
    # Generar todas las secciones
    sections = {}
    
    # Secciones básicas (siempre incluidas)
    sections['data_quality'] = generate_data_quality_section(df, numeric_cols, categorical_cols)
    sections['descriptive_analysis'] = generate_descriptive_section(df, numeric_cols, categorical_cols, max_vars_descriptive)
    sections['correlation_analysis'] = generate_correlation_section(df, numeric_cols)
    sections['outlier_analysis'] = generate_outlier_section(df, numeric_cols)
    
    # Secciones opcionales
    if include_ml_insights:
        sections['ml_insights'] = generate_ml_insights_section(df, numeric_cols, categorical_cols)
    
    # Análisis temporal si hay variables temporales
    temporal_vars = identify_temporal_variables(df, numeric_cols, categorical_cols)
    if temporal_vars:
        sections['temporal_analysis'] = generate_temporal_section(df, numeric_cols, categorical_cols)
    
    # Generar resumen ejecutivo
    sections['executive_summary'] = generate_executive_summary_section(
        df, numeric_cols, categorical_cols, sections
    )
    
    # Recomendaciones si se solicitan
    if include_recommendations:
        sections['recommendations'] = generate_recommendations_section(
            df, numeric_cols, categorical_cols, sections
        )
    
    # Metadatos
    metadata = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'report_type': 'Comprehensive Report',
        'total_observations': len(df),
        'total_variables': len(numeric_cols) + len(categorical_cols),
        'numeric_variables': len(numeric_cols),
        'categorical_variables': len(categorical_cols),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'confidence_level': confidence_level,
        'max_vars_descriptive': max_vars_descriptive,
        'correlation_threshold': correlation_threshold,
        'outlier_method': outlier_method,
        'include_ml_insights': include_ml_insights,
        'include_recommendations': include_recommendations
    }
    
    return {
        'metadata': metadata,
        'sections': sections
    }


def generate_executive_report(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """Genera un reporte ejecutivo conciso y orientado a la toma de decisiones"""
    
    # Metadatos del reporte
    metadata = {
        'title': 'Reporte Ejecutivo de Análisis de Datos',
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols)
        }
    }
    
    # Análisis de calidad de datos
    quality_analysis = {
        'missing_data': {
            'total_missing': df.isna().sum().sum(),
            'missing_percentage': (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100),
            'columns_with_missing': df.isna().sum()[df.isna().sum() > 0].to_dict()
        },
        'data_types': {
            'numeric': len(numeric_cols),
            'categorical': len(categorical_cols),
            'datetime': len([col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])])
        }
    }
    
    # Análisis estadístico resumido
    statistical_summary = {}
    if numeric_cols:
        numeric_stats = df[numeric_cols].describe()
        statistical_summary['numeric'] = {
            'mean_values': numeric_stats.loc['mean'].to_dict(),
            'std_values': numeric_stats.loc['std'].to_dict(),
            'correlation_matrix': df[numeric_cols].corr().to_dict()
        }
    
    if categorical_cols:
        categorical_stats = {}
        for col in categorical_cols:
            categorical_stats[col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'frequency': df[col].value_counts().head(5).to_dict()
            }
        statistical_summary['categorical'] = categorical_stats
    
    # Análisis de tendencias (si hay datos temporales)
    trend_analysis = {}
    time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if time_cols and numeric_cols:
        time_col = time_cols[0]
        for col in numeric_cols[:3]:  # Analizar solo las primeras 3 variables numéricas
            try:
                # Regresión lineal simple para tendencia
                x = np.arange(len(df))
                y = df[col].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                trend_analysis[col] = {
                    'slope': slope,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'trend': 'creciente' if slope > 0 else 'decreciente' if slope < 0 else 'estable'
                }
            except:
                trend_analysis[col] = {'trend': 'no disponible'}
    
    # Recomendaciones estratégicas
    recommendations = []
    
    # Recomendaciones basadas en calidad de datos
    if quality_analysis['missing_data']['missing_percentage'] > 20:
        recommendations.append("🔴 CRÍTICO: Alto porcentaje de datos faltantes. Implementar estrategia de recolección de datos.")
    elif quality_analysis['missing_data']['missing_percentage'] > 10:
        recommendations.append("🟡 ATENCIÓN: Datos faltantes moderados. Considerar imputación o recolección adicional.")
    else:
        recommendations.append("🟢 EXCELENTE: Calidad de datos alta. Proceder con análisis avanzado.")
    
    # Recomendaciones basadas en correlaciones
    if numeric_cols and len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))
        
        if high_corr_pairs:
            recommendations.append(f"🟡 ATENCIÓN: {len(high_corr_pairs)} pares de variables altamente correlacionadas. Considerar reducción de dimensionalidad.")
    
    # Recomendaciones basadas en tendencias
    if trend_analysis:
        increasing_trends = [var for var, analysis in trend_analysis.items() if analysis.get('trend') == 'creciente']
        decreasing_trends = [var for var, analysis in trend_analysis.items() if analysis.get('trend') == 'decreciente']
        
        if increasing_trends:
            recommendations.append(f"📈 OPORTUNIDAD: Variables con tendencia creciente: {', '.join(increasing_trends)}")
        if decreasing_trends:
            recommendations.append(f"📉 RIESGO: Variables con tendencia decreciente: {', '.join(decreasing_trends)}")
    
    # Recomendaciones de análisis
    if len(numeric_cols) >= 5:
        recommendations.append("🔬 ANÁLISIS: Dataset multivariado. Recomendado análisis de componentes principales y clustering.")
    
    if len(categorical_cols) >= 3:
        recommendations.append("📊 ANÁLISIS: Múltiples variables categóricas. Recomendado análisis de asociación y segmentación.")
    
    # Métricas clave
    key_metrics = {
        'data_quality_score': max(0, 100 - quality_analysis['missing_data']['missing_percentage']),
        'analysis_complexity': 'Alta' if len(numeric_cols) >= 10 else 'Media' if len(numeric_cols) >= 5 else 'Baja',
        'recommendation_priority': 'Alta' if quality_analysis['missing_data']['missing_percentage'] > 20 else 'Media' if quality_analysis['missing_data']['missing_percentage'] > 10 else 'Baja'
    }
    
    # Secciones del reporte
    sections = {
        'executive_summary': {
            'title': 'Resumen Ejecutivo',
            'content': f"""
            Este reporte presenta un análisis ejecutivo del dataset con {len(df)} observaciones y {len(df.columns)} variables.
            La calidad de los datos es {'excelente' if key_metrics['data_quality_score'] > 90 else 'buena' if key_metrics['data_quality_score'] > 80 else 'regular' if key_metrics['data_quality_score'] > 70 else 'pobre'}.
            Se identificaron {len(recommendations)} recomendaciones estratégicas para optimizar el análisis y la toma de decisiones.
            """
        },
        'key_findings': {
            'title': 'Hallazgos Clave',
            'content': {
                'data_quality': quality_analysis,
                'statistical_insights': statistical_summary,
                'trend_analysis': trend_analysis
            }
        },
        'strategic_recommendations': {
            'title': 'Recomendaciones Estratégicas',
            'content': recommendations
        },
        'key_metrics': {
            'title': 'Métricas Clave',
            'content': key_metrics
        }
    }
    
    return {
        'metadata': metadata,
        'sections': sections
    }
