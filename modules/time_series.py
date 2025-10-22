"""
Módulo de Análisis de Series Temporales Avanzado
Incluye análisis estadístico, descomposición, pronósticos y detección de anomalías
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from typing import List
import warnings
warnings.filterwarnings('ignore')

# Importar configuración
try:
    from config import THEME_TEMPLATE, PLOTLY_CONFIG
except ImportError:
    THEME_TEMPLATE = "plotly"
    PLOTLY_CONFIG = {}

# Importaciones para análisis avanzado
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("⚠️ statsmodels no está disponible. Algunas funcionalidades avanzadas estarán limitadas.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("⚠️ Prophet no está disponible. Los pronósticos con Prophet estarán limitados.")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    st.warning("⚠️ ruptures no está disponible. La detección de cambios estructurales estará limitada.")

# Importar funciones de pronósticos
try:
    from modules.time_series_forecasting import (
        display_forecasting_analysis, 
        display_anomaly_detection
    )
    FORECASTING_AVAILABLE = True
except ImportError:
    FORECASTING_AVAILABLE = False
    st.warning("⚠️ Módulo de pronósticos no disponible.")


def display_time_series_analysis(df: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    """
    Función principal para mostrar análisis de series temporales
    """
    st.header("📈 Análisis Avanzado de Series Temporales")
    
    # Configuración general
    st.subheader("⚙️ Configuración General")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Tamaño de prueba para pronósticos (%)", 10, 40, 20, key="ts_test_size")
    
    with col2:
        confidence_level = st.slider("Nivel de confianza (%)", 80, 99, 95, key="ts_confidence")
    
    with col3:
        max_lags = st.slider("Máximo número de lags", 5, 50, 20, key="ts_max_lags")
    
    # Detectar columnas temporales
    time_columns = detect_time_columns(df)
    
    # Mostrar información de diagnóstico
    st.info(f"🔍 Columnas detectadas como temporales: {time_columns}")
    
    if not time_columns:
        st.warning("⚠️ No se detectaron columnas temporales automáticamente.")
        st.info("💡 Mostrando todas las columnas disponibles para selección manual...")
        
        # Mostrar todas las columnas disponibles para selección manual
        all_columns = list(df.columns)
        st.info(f"📋 Columnas disponibles: {all_columns}")
        
        # Permitir selección manual
        time_col = st.selectbox(
            "🗓️ Seleccionar columna temporal manualmente:",
            all_columns,
            key="ts_time_column_manual"
        )
        
        if not time_col:
            st.error("❌ Debes seleccionar una columna temporal para continuar.")
            return
    else:
        # Selección de columna temporal con información adicional
        st.success(f"✅ Se detectaron {len(time_columns)} columna(s) temporal(es)")
        
        # Mostrar información sobre cada columna detectada
        for i, col in enumerate(time_columns):
            sample_values = df[col].dropna().head(3).tolist()
            st.info(f"📅 {col}: {sample_values}")
        
        time_col = st.selectbox(
            "🗓️ Seleccionar columna temporal:",
            time_columns,
            key="ts_time_column"
        )
    
    if not time_col:
        return
    
    # Verificar que la columna seleccionada realmente contiene fechas
    if not verify_date_content(df[time_col]):
        st.error(f"❌ La columna '{time_col}' no contiene fechas válidas.")
        st.info("💡 Los valores de muestra no se pueden convertir a fechas.")
        
        # Mostrar muestra de valores problemáticos
        sample_values = df[time_col].dropna().head(5).tolist()
        st.info(f"📝 Valores de muestra en '{time_col}': {sample_values}")
        
        # Permitir selección manual de otra columna
        st.warning("⚠️ Por favor, selecciona una columna que contenga fechas válidas.")
        return
    
    # Preparar datos temporales
    time_series_data = prepare_time_series_data(df, time_col)
    
    if time_series_data.empty:
        st.error("❌ No se pudieron preparar los datos temporales.")
        return
    
    # Selección de variables para análisis
    analysis_variables = select_analysis_variables(time_series_data, numeric_cols, time_col)
    
    if not analysis_variables:
        st.error("❌ No hay variables numéricas disponibles para análisis.")
        return
    
    # Tabs para diferentes tipos de análisis
    tab_names = ["📊 Exploración", "📈 Descomposición", "🔮 Pronósticos", "🎯 Detección de Anomalías", "🔗 Causalidad", "⚡ Cambios Estructurales", "📋 Resumen Ejecutivo"]
    tabs = st.tabs(tab_names)
    
    with tabs[0]:
        display_exploratory_analysis(time_series_data, time_col, analysis_variables)
    
    with tabs[1]:
        display_decomposition_analysis(time_series_data, time_col, analysis_variables)
    
    with tabs[2]:
        if FORECASTING_AVAILABLE:
            display_forecasting_analysis(time_series_data, time_col, analysis_variables, test_size/100, confidence_level/100)
        else:
            st.error("Módulo de pronósticos no disponible.")
    
    with tabs[3]:
        if FORECASTING_AVAILABLE:
            display_anomaly_detection(time_series_data, time_col, analysis_variables)
        else:
            st.error("Módulo de detección de anomalías no disponible.")
    
    with tabs[4]:
        display_causality_analysis(time_series_data, time_col, analysis_variables, max_lags)
    
    with tabs[5]:
        display_structural_change_detection(time_series_data, time_col, analysis_variables)


def detect_time_columns(df: pd.DataFrame) -> list:
    """Detecta columnas que contienen información temporal"""
    time_columns = []
    
    # Palabras clave prioritarias para fechas (ordenadas por prioridad)
    priority_keywords = ['date référence', 'date_reference', 'date reference', 'référence', 'reference']
    secondary_keywords = ['date', 'fecha', 'time', 'tiempo', 'year', 'año', 'année', 
                         'period', 'período', 'période', 'réf', 'ref', 'début', 'debut', 'begin', 'fin', 'end']
    
    # Primera pasada: buscar palabras clave prioritarias
    for col in df.columns:
        col_lower = col.lower()
        for keyword in priority_keywords:
            if keyword in col_lower:
                time_columns.append(col)
                break
    
    # Si encontramos columnas prioritarias, verificar su contenido
    if time_columns:
        verified_priority = []
        for col in time_columns:
            if verify_date_content(df[col]):
                verified_priority.append(col)
        if verified_priority:
            return verified_priority
    
    # Segunda pasada: buscar palabras clave secundarias
    for col in df.columns:
        col_lower = col.lower()
        for keyword in secondary_keywords:
            if keyword in col_lower:
                time_columns.append(col)
                break
        
        # Verificar tipo de datos
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_columns.append(col)
            continue
        
        # Verificar contenido
        sample_values = df[col].dropna().head(20)
        if len(sample_values) > 0:
            # Intentar convertir a datetime con diferentes formatos
            date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y']
            converted_any = False
            
            for fmt in date_formats:
                try:
                    pd.to_datetime(sample_values, format=fmt, errors='raise')
                    time_columns.append(col)
                    converted_any = True
                    break
                except:
                    continue
            
            if not converted_any:
                # Intentar conversión automática
                try:
                    pd.to_datetime(sample_values, errors='raise')
                    time_columns.append(col)
                    converted_any = True
                except:
                    pass
            
            if not converted_any:
                # Verificar si son años
                try:
                    years = pd.to_numeric(sample_values, errors='coerce')
                    if len(years.dropna()) > 0 and all(1900 <= year <= 2100 for year in years.dropna()):
                        time_columns.append(col)
                except:
                    pass
    
    return list(set(time_columns))


def verify_date_content(series: pd.Series) -> bool:
    """Verifica si una serie contiene realmente fechas válidas"""
    sample_values = series.dropna().head(10)
    if len(sample_values) == 0:
        return False
    
    # Intentar convertir a datetime con diferentes formatos
    date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y']
    
    for fmt in date_formats:
        try:
            pd.to_datetime(sample_values, format=fmt, errors='raise')
            return True
        except:
            continue
    
    # Intentar conversión automática
    try:
        converted = pd.to_datetime(sample_values, errors='coerce')
        if converted.notna().sum() > len(sample_values) * 0.8:  # 80% éxito
            return True
    except:
        pass
    
    return False


def prepare_time_series_data(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Prepara los datos para análisis de series temporales"""
    try:
        # Crear copia del DataFrame
        ts_data = df.copy()
        
        # Verificar que la columna existe
        if time_col not in ts_data.columns:
            st.error(f"❌ La columna temporal '{time_col}' no existe en el DataFrame.")
            st.info(f"Columnas disponibles: {list(ts_data.columns)}")
            return pd.DataFrame()
        
        # Mostrar información de diagnóstico
        st.info(f"🔍 Preparando datos temporales con columna: '{time_col}'")
        st.info(f"📊 Valores únicos en {time_col}: {ts_data[time_col].nunique()}")
        st.info(f"📋 Tipo original: {ts_data[time_col].dtype}")
        
        # Mostrar muestra de valores
        sample_values = ts_data[time_col].dropna().head(5).tolist()
        st.info(f"📝 Muestra de valores: {sample_values}")
        
        # Intentar conversión a datetime con múltiples formatos
        original_length = len(ts_data)
        
        # Probar diferentes formatos de fecha
        date_formats = [
            '%Y-%m-%d',     # 2020-02-01
            '%d/%m/%Y',     # 01/02/2020
            '%Y-%m-%d %H:%M:%S',  # 2020-02-01 00:00:00
            '%d-%m-%Y',     # 01-02-2020
            '%m/%d/%Y',     # 02/01/2020
        ]
        
        converted = False
        for fmt in date_formats:
            try:
                ts_data[time_col] = pd.to_datetime(ts_data[time_col], format=fmt, errors='coerce')
                if ts_data[time_col].notna().sum() > original_length * 0.8:  # 80% éxito
                    st.info(f"✅ Formato de fecha detectado: {fmt}")
                    converted = True
                    break
            except:
                continue
        
        if not converted:
            # Intentar conversión automática como último recurso
            st.warning("⚠️ Usando conversión automática de pandas")
            ts_data[time_col] = pd.to_datetime(ts_data[time_col], errors='coerce')
        
        # Verificar cuántos valores se convirtieron correctamente
        valid_dates = ts_data[time_col].notna().sum()
        st.info(f"✅ Valores convertidos a fecha: {valid_dates}/{original_length}")
        
        # Mostrar rango de fechas detectado
        if valid_dates > 0:
            min_date = ts_data[time_col].min()
            max_date = ts_data[time_col].max()
            st.info(f"📅 Rango de fechas: {min_date.strftime('%Y-%m-%d')} a {max_date.strftime('%Y-%m-%d')}")
        
        # Eliminar filas con fechas inválidas
        ts_data = ts_data.dropna(subset=[time_col])
        
        if ts_data.empty:
            st.error("❌ No quedaron datos válidos después de la conversión a fecha.")
            return pd.DataFrame()
        
        # Ordenar por tiempo
        ts_data = ts_data.sort_values(time_col)
        
        # Establecer índice temporal
        ts_data = ts_data.set_index(time_col)
        
        # Eliminar duplicados en el índice temporal
        duplicates = ts_data.index.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Se encontraron {duplicates} fechas duplicadas. Se mantendrá la primera ocurrencia.")
            ts_data = ts_data[~ts_data.index.duplicated(keep='first')]
        
        st.success(f"✅ Datos temporales preparados exitosamente: {len(ts_data)} observaciones")
        return ts_data
        
    except Exception as e:
        st.error(f"❌ Error preparando datos temporales: {str(e)}")
        st.error(f"Tipo de error: {type(e).__name__}")
        import traceback
        st.error(f"Detalles: {traceback.format_exc()}")
        return pd.DataFrame()


def select_analysis_variables(ts_data: pd.DataFrame, numeric_cols: list, time_col: str) -> list:
    """Permite seleccionar variables para análisis"""
    st.subheader("📋 Selección de Variables")
    
    # Filtrar variables numéricas disponibles (compatibles con tipos pandas actuales)
    available_vars = [
        col for col in numeric_cols
        if col in ts_data.columns and pd.api.types.is_numeric_dtype(ts_data[col])
    ]
    
    if not available_vars:
        return []
    
    # Opciones de selección
    selection_mode = st.radio(
        "Modo de selección:",
        ["Todas las variables", "Selección manual", "Top N variables"],
        key="ts_var_selection_mode"
    )
    
    if selection_mode == "Todas las variables":
        selected_vars = available_vars
        st.info(f"✅ Seleccionadas todas las {len(selected_vars)} variables numéricas")
        
    elif selection_mode == "Selección manual":
        selected_vars = st.multiselect(
            "Seleccionar variables específicas:",
            available_vars,
            default=available_vars[:min(5, len(available_vars))],
            key="ts_manual_vars"
        )
        
    else:  # Top N variables
        n_vars = st.slider("Número de variables", 1, min(10, len(available_vars)), min(5, len(available_vars)), key="ts_top_n")
        
        # Criterio de selección
        criteria = st.selectbox(
            "Criterio de selección:",
            ["Por varianza", "Por rango de valores", "Por completitud de datos"],
            key="ts_selection_criteria"
        )
        
        if criteria == "Por varianza":
            variances = ts_data[available_vars].var().sort_values(ascending=False)
            selected_vars = variances.head(n_vars).index.tolist()
        elif criteria == "Por rango de valores":
            ranges = (ts_data[available_vars].max() - ts_data[available_vars].min()).sort_values(ascending=False)
            selected_vars = ranges.head(n_vars).index.tolist()
        else:  # Por completitud
            completeness = ts_data[available_vars].count().sort_values(ascending=False)
            selected_vars = completeness.head(n_vars).index.tolist()
        
        st.info(f"✅ Seleccionadas {len(selected_vars)} variables: {', '.join(selected_vars)}")
    
    # Mostrar información de las variables seleccionadas
    if selected_vars:
        with st.expander("📊 Información de Variables Seleccionadas"):
            var_info = []
            for var in selected_vars:
                series = ts_data[var].dropna()
                var_info.append({
                    'Variable': var,
                    'Observaciones': len(series),
                    'Valores faltantes': ts_data[var].isna().sum(),
                    'Media': round(series.mean(), 3),
                    'Desv. Std': round(series.std(), 3),
                    'Mín': round(series.min(), 3),
                    'Máx': round(series.max(), 3),
                    'Rango': round(series.max() - series.min(), 3)
                })
            
            var_df = pd.DataFrame(var_info)
            st.dataframe(var_df, width='stretch')
    
    return selected_vars


def display_exploratory_analysis(ts_data: pd.DataFrame, time_col: str, variables: list):
    """Análisis exploratorio de series temporales"""
    st.subheader("📊 Análisis Exploratorio")
    
    if not variables:
        st.warning("No hay variables seleccionadas para análisis.")
        return
    
    # Seleccionar variable principal para análisis detallado
    main_var = st.selectbox("Variable principal para análisis detallado:", variables, key="ts_main_var")
    
    if not main_var:
        return
    
    # Obtener serie temporal
    series = ts_data[main_var].dropna()
    
    if len(series) < 2:
        st.error("La serie temporal debe tener al menos 2 observaciones.")
        return
    
    # Métricas básicas
    st.subheader("📈 Métricas Básicas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Observaciones", len(series))
        st.metric("Período", f"{series.index.min().strftime('%Y-%m-%d')} a {series.index.max().strftime('%Y-%m-%d')}")
    
    with col2:
        st.metric("Media", f"{series.mean():.3f}")
        st.metric("Mediana", f"{series.median():.3f}")
    
    with col3:
        st.metric("Desv. Estándar", f"{series.std():.3f}")
        st.metric("Coef. Variación", f"{series.std()/series.mean()*100:.1f}%")
    
    with col4:
        st.metric("Mínimo", f"{series.min():.3f}")
        st.metric("Máximo", f"{series.max():.3f}")
    
    # Visualización temporal
    st.subheader("📈 Visualización Temporal")
    
    # Opciones de visualización
    viz_options = st.multiselect(
        "Tipo de visualización:",
        ["Línea temporal", "Distribución", "Box plot por período", "Correlograma"],
        default=["Línea temporal"],
        key="ts_viz_options"
    )
    
    if "Línea temporal" in viz_options:
        # Gráfico de línea temporal
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name=main_var,
            line=dict(width=2),
            hovertemplate='<b>%{x}</b><br>Valor: %{y:.3f}<extra></extra>'
        ))
        
        # Agregar media móvil si hay suficientes datos
        if len(series) > 30:
            window = min(30, len(series)//10)
            moving_avg = series.rolling(window=window, center=True).mean()
            
            fig.add_trace(go.Scatter(
                x=moving_avg.index,
                y=moving_avg.values,
                mode='lines',
                name=f'Media móvil ({window})',
                line=dict(dash='dash', width=2),
                hovertemplate='<b>%{x}</b><br>Media móvil: %{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Serie Temporal: {main_var}",
            xaxis_title="Tiempo",
            yaxis_title="Valor",
            template=THEME_TEMPLATE,
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
    
    if "Distribución" in viz_options:
        # Gráfico de distribución
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma
            fig_hist = px.histogram(
                x=series.values,
                nbins=30,
                title=f"Distribución de {main_var}",
                labels={'x': main_var, 'y': 'Frecuencia'}
            )
            fig_hist.update_layout(template=THEME_TEMPLATE)
            st.plotly_chart(fig_hist, width='stretch', config=PLOTLY_CONFIG)
        
        with col2:
            # Q-Q plot
            from scipy.stats import probplot
            qq_data = probplot(series.values, dist="norm")
            
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Datos',
                marker=dict(size=6)
            ))
            
            # Línea de referencia
            fig_qq.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][0] * qq_data[1][0] + qq_data[1][1],
                mode='lines',
                name='Línea teórica',
                line=dict(dash='dash', color='red')
            ))
            
            fig_qq.update_layout(
                title=f"Q-Q Plot: {main_var}",
                xaxis_title="Cuantiles teóricos",
                yaxis_title="Cuantiles muestrales",
                template=THEME_TEMPLATE
            )
            
            st.plotly_chart(fig_qq, width='stretch', config=PLOTLY_CONFIG)
    
    if "Box plot por período" in viz_options:
        # Box plot por período
        period_option = st.selectbox(
            "Agrupar por:",
            ["Año", "Mes", "Trimestre", "Día de la semana"],
            key="ts_period_grouping"
        )
        
        if period_option == "Año":
            period_series = series.groupby(series.index.year)
            period_labels = [str(year) for year in period_series.groups.keys()]
        elif period_option == "Mes":
            period_series = series.groupby(series.index.month)
            period_labels = [f"{month:02d}" for month in period_series.groups.keys()]
        elif period_option == "Trimestre":
            period_series = series.groupby(series.index.quarter)
            period_labels = [f"Q{quarter}" for quarter in period_series.groups.keys()]
        else:  # Día de la semana
            period_series = series.groupby(series.index.dayofweek)
            period_labels = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        
        # Crear datos para box plot
        box_data = []
        for period, group in period_series:
            box_data.append(group.values)
        
        fig_box = go.Figure()
        
        for i, (data, label) in enumerate(zip(box_data, period_labels)):
            fig_box.add_trace(go.Box(
                y=data,
                name=label,
                boxpoints='outliers'
            ))
        
        fig_box.update_layout(
            title=f"Box Plot por {period_option}: {main_var}",
            xaxis_title=period_option,
            yaxis_title=main_var,
            template=THEME_TEMPLATE
        )
        
        st.plotly_chart(fig_box, width='stretch', config=PLOTLY_CONFIG)
    
    if "Correlograma" in viz_options and STATSMODELS_AVAILABLE:
        # Correlograma (ACF y PACF)
        max_lags = min(40, len(series)//4)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ACF
            try:
                from statsmodels.graphics.tsaplots import plot_acf
                import matplotlib.pyplot as plt
                
                fig_acf, ax_acf = plt.subplots(figsize=(8, 4))
                plot_acf(series, lags=max_lags, ax=ax_acf, title=f"ACF: {main_var}")
                ax_acf.grid(True, alpha=0.3)
                st.pyplot(fig_acf)
                plt.close(fig_acf)
                
            except Exception as e:
                st.error(f"Error generando ACF: {str(e)}")
        
        with col2:
            # PACF
            try:
                from statsmodels.graphics.tsaplots import plot_pacf
                
                fig_pacf, ax_pacf = plt.subplots(figsize=(8, 4))
                plot_pacf(series, lags=max_lags, ax=ax_pacf, title=f"PACF: {main_var}")
                ax_pacf.grid(True, alpha=0.3)
                st.pyplot(fig_pacf)
                plt.close(fig_pacf)
                
            except Exception as e:
                st.error(f"Error generando PACF: {str(e)}")
    
    # Análisis de estacionariedad
    st.subheader("🔍 Análisis de Estacionariedad")
    
    if STATSMODELS_AVAILABLE:
        # Prueba ADF
        try:
            adf_result = adfuller(series.dropna())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Estadístico ADF", f"{adf_result[0]:.4f}")
            
            with col2:
                st.metric("p-valor", f"{adf_result[1]:.4f}")
            
            with col3:
                critical_values = adf_result[4]
                st.metric("Valor crítico (5%)", f"{critical_values['5%']:.4f}")
            
            # Interpretación
            if adf_result[1] <= 0.05:
                st.success("✅ La serie es **estacionaria** (p-valor ≤ 0.05)")
            else:
                st.warning("⚠️ La serie **no es estacionaria** (p-valor > 0.05)")
                
                # Sugerencias para hacer estacionaria
                st.info("💡 **Sugerencias para hacer estacionaria:**")
                st.write("- Aplicar diferenciación (primera diferencia)")
                st.write("- Aplicar logaritmo si hay heterocedasticidad")
                st.write("- Remover tendencia")
                
        except Exception as e:
            st.error(f"Error en prueba de estacionariedad: {str(e)}")
    
    else:
        st.warning("⚠️ statsmodels no disponible para análisis de estacionariedad")
    
    # Análisis de tendencia
    st.subheader("📈 Análisis de Tendencia")
    
    # Regresión lineal en el tiempo
    try:
        # Convertir índice a numérico para regresión
        time_numeric = np.arange(len(series))
        
        # Regresión lineal
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, series.values)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Pendiente", f"{slope:.6f}")
            if slope > 0:
                st.success("📈 Tendencia **creciente**")
            elif slope < 0:
                st.warning("📉 Tendencia **decreciente**")
            else:
                st.info("➡️ Sin tendencia")
        
        with col2:
            st.metric("R²", f"{r_value**2:.4f}")
        
        with col3:
            st.metric("p-valor", f"{p_value:.4f}")
        
        # Visualizar tendencia
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name='Datos originales',
            line=dict(width=2)
        ))
        
        # Línea de tendencia
        trend_line = intercept + slope * time_numeric
        fig_trend.add_trace(go.Scatter(
            x=series.index,
            y=trend_line,
            mode='lines',
            name='Tendencia lineal',
            line=dict(dash='dash', color='red', width=2)
        ))
        
        fig_trend.update_layout(
            title=f"Tendencia Lineal: {main_var}",
            xaxis_title="Tiempo",
            yaxis_title="Valor",
            template=THEME_TEMPLATE
        )
        
        st.plotly_chart(fig_trend, width='stretch', config=PLOTLY_CONFIG)
        
    except Exception as e:
        st.error(f"Error en análisis de tendencia: {str(e)}")


def display_decomposition_analysis(ts_data: pd.DataFrame, time_col: str, variables: list):
    """Análisis de descomposición temporal"""
    st.subheader("📈 Descomposición Temporal")
    
    if not variables:
        st.warning("No hay variables seleccionadas para análisis.")
        return
    
    # Seleccionar variable para descomposición
    main_var = st.selectbox("Variable para descomposición:", variables, key="ts_decomp_var")
    
    if not main_var:
        return
    
    # Obtener serie temporal
    series = ts_data[main_var].dropna()
    
    if len(series) < 24:  # Necesitamos al menos 2 años de datos mensuales
        st.error("Se necesitan al menos 24 observaciones para descomposición temporal.")
        return
    
    # Configuración de descomposición
    st.subheader("⚙️ Configuración de Descomposición")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        decomp_type = st.selectbox(
            "Tipo de descomposición:",
            ["Aditiva", "Multiplicativa"],
            key="ts_decomp_type"
        )
    
    with col2:
        decomp_model = st.selectbox(
            "Modelo de descomposición:",
            ["STL", "Clásica", "X-13ARIMA-SEATS"],
            key="ts_decomp_model"
        )
    
    with col3:
        if decomp_model == "STL":
            seasonal_period = st.number_input("Período estacional:", min_value=2, max_value=52, value=12, key="ts_seasonal_period")
        else:
            seasonal_period = 12
    
    # Realizar descomposición
    try:
        if decomp_model == "STL" and STATSMODELS_AVAILABLE:
            # Descomposición STL
            stl_decomp = STL(series, seasonal=seasonal_period).fit()
            
            # Crear DataFrame con componentes
            decomp_df = pd.DataFrame({
                'original': series,
                'trend': stl_decomp.trend,
                'seasonal': stl_decomp.seasonal,
                'residual': stl_decomp.resid
            })
            
        elif STATSMODELS_AVAILABLE:
            # Descomposición clásica
            model_type = 'additive' if decomp_type == "Aditiva" else 'multiplicative'
            decomp_result = seasonal_decompose(series, model=model_type, period=seasonal_period)
            
            # Crear DataFrame con componentes
            decomp_df = pd.DataFrame({
                'original': series,
                'trend': decomp_result.trend,
                'seasonal': decomp_result.seasonal,
                'residual': decomp_result.resid
            })
            
        else:
            st.error("statsmodels no está disponible para descomposición temporal.")
            return
        
        # Mostrar métricas de descomposición
        st.subheader("📊 Métricas de Descomposición")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tendencia", f"{decomp_df['trend'].std():.3f}", delta=None)
        
        with col2:
            st.metric("Estacionalidad", f"{decomp_df['seasonal'].std():.3f}", delta=None)
        
        with col3:
            st.metric("Residuos", f"{decomp_df['residual'].std():.3f}", delta=None)
        
        with col4:
            # Calcular varianza explicada
            total_var = decomp_df['original'].var()
            residual_var = decomp_df['residual'].var()
            explained_var = 1 - (residual_var / total_var)
            st.metric("Varianza explicada", f"{explained_var*100:.1f}%")
        
        # Visualización de descomposición
        st.subheader("📈 Componentes de la Descomposición")
        
        # Crear subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Serie Original', 'Tendencia', 'Estacionalidad', 'Residuos'),
            vertical_spacing=0.08
        )
        
        # Serie original
        fig.add_trace(
            go.Scatter(
                x=decomp_df.index,
                y=decomp_df['original'],
                mode='lines',
                name='Original',
                line=dict(width=2, color='blue')
            ),
            row=1, col=1
        )
        
        # Tendencia
        fig.add_trace(
            go.Scatter(
                x=decomp_df.index,
                y=decomp_df['trend'],
                mode='lines',
                name='Tendencia',
                line=dict(width=2, color='red'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Estacionalidad
        fig.add_trace(
            go.Scatter(
                x=decomp_df.index,
                y=decomp_df['seasonal'],
                mode='lines',
                name='Estacionalidad',
                line=dict(width=2, color='green'),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Residuos
        fig.add_trace(
            go.Scatter(
                x=decomp_df.index,
                y=decomp_df['residual'],
                mode='lines',
                name='Residuos',
                line=dict(width=2, color='orange'),
                showlegend=False
            ),
            row=4, col=1
        )
        
        fig.update_layout(
            title=f"Descomposición Temporal: {main_var}",
            height=800,
            template=THEME_TEMPLATE
        )
        
        fig.update_xaxes(title_text="Tiempo", row=4, col=1)
        fig.update_yaxes(title_text="Valor", row=1, col=1)
        fig.update_yaxes(title_text="Tendencia", row=2, col=1)
        fig.update_yaxes(title_text="Estacionalidad", row=3, col=1)
        fig.update_yaxes(title_text="Residuos", row=4, col=1)
        
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
        
        # Análisis de residuos
        st.subheader("🔍 Análisis de Residuos")
        
        # Estadísticas de residuos
        residuals = decomp_df['residual'].dropna()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Media de residuos", f"{residuals.mean():.6f}")
        
        with col2:
            st.metric("Desv. Estándar", f"{residuals.std():.3f}")
        
        with col3:
            # Prueba de normalidad
            if len(residuals) > 3:
                from scipy.stats import shapiro
                if len(residuals) <= 5000:  # Shapiro solo funciona hasta 5000 observaciones
                    _, p_value = shapiro(residuals)
                    st.metric("Normalidad (Shapiro)", f"{p_value:.4f}")
                else:
                    from scipy.stats import jarque_bera
                    _, p_value = jarque_bera(residuals)
                    st.metric("Normalidad (Jarque-Bera)", f"{p_value:.4f}")
        
        # Visualización de residuos
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma de residuos
            fig_resid_hist = px.histogram(
                x=residuals,
                nbins=30,
                title="Distribución de Residuos",
                labels={'x': 'Residuos', 'y': 'Frecuencia'}
            )
            fig_resid_hist.update_layout(template=THEME_TEMPLATE)
            st.plotly_chart(fig_resid_hist, width='stretch', config=PLOTLY_CONFIG)
        
        with col2:
            # Q-Q plot de residuos
            from scipy.stats import probplot
            qq_data = probplot(residuals, dist="norm")
            
            fig_resid_qq = go.Figure()
            fig_resid_qq.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Residuos',
                marker=dict(size=6)
            ))
            
            # Línea de referencia
            fig_resid_qq.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][0] * qq_data[1][0] + qq_data[1][1],
                mode='lines',
                name='Línea teórica',
                line=dict(dash='dash', color='red')
            ))
            
            fig_resid_qq.update_layout(
                title="Q-Q Plot de Residuos",
                xaxis_title="Cuantiles teóricos",
                yaxis_title="Cuantiles muestrales",
                template=THEME_TEMPLATE
            )
            
            st.plotly_chart(fig_resid_qq, width='stretch', config=PLOTLY_CONFIG)
        
        # Prueba de autocorrelación en residuos
        if STATSMODELS_AVAILABLE and len(residuals) > 10:
            st.subheader("🔗 Autocorrelación en Residuos")
            
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                
                # Prueba de Ljung-Box
                ljung_box_result = acorr_ljungbox(residuals, lags=10, return_df=True)
                
                # Mostrar resultados
                st.write("**Prueba de Ljung-Box (autocorrelación en residuos):**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Estadístico LB", f"{ljung_box_result['lb_stat'].iloc[-1]:.4f}")
                
                with col2:
                    st.metric("p-valor", f"{ljung_box_result['lb_pvalue'].iloc[-1]:.4f}")
                
                with col3:
                    if ljung_box_result['lb_pvalue'].iloc[-1] > 0.05:
                        st.success("✅ Sin autocorrelación")
                    else:
                        st.warning("⚠️ Autocorrelación detectada")
                
                # Gráfico de estadísticos
                fig_lb = go.Figure()
                
                fig_lb.add_trace(go.Scatter(
                    x=ljung_box_result.index,
                    y=ljung_box_result['lb_stat'],
                    mode='lines+markers',
                    name='Estadístico LB',
                    line=dict(width=2)
                ))
                
                fig_lb.update_layout(
                    title="Estadístico de Ljung-Box",
                    xaxis_title="Lags",
                    yaxis_title="Estadístico",
                    template=THEME_TEMPLATE
                )
                
                st.plotly_chart(fig_lb, width='stretch', config=PLOTLY_CONFIG)
                
            except Exception as e:
                st.error(f"Error en prueba de autocorrelación: {str(e)}")
        
        # Exportar componentes
        st.subheader("💾 Exportar Componentes")
        
        if st.button("Descargar componentes de descomposición", key="ts_download_decomp"):
            # Crear CSV
            csv_data = decomp_df.to_csv()
            
            st.download_button(
                label="📥 Descargar CSV",
                data=csv_data,
                file_name=f"decomposicion_{main_var}.csv",
                mime="text/csv",
                key="ts_download_csv"
            )
    
    except Exception as e:
        st.error(f"Error en descomposición temporal: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


def display_causality_analysis(ts_data: pd.DataFrame, time_col: str, variables: list, max_lags: int):
    """Análisis de causalidad de Granger"""
    st.subheader("🔗 Análisis de Causalidad")
    
    if len(variables) < 2:
        st.warning("Se necesitan al menos 2 variables para análisis de causalidad.")
        return
    
    if not STATSMODELS_AVAILABLE:
        st.error("statsmodels no está disponible para análisis de causalidad.")
        return
    
    # Seleccionar variables para análisis
    st.subheader("⚙️ Configuración de Análisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_vars = st.multiselect(
            "Variables para análisis de causalidad:",
            variables,
            default=variables[:min(3, len(variables))],
            key="ts_causality_vars"
        )
    
    with col2:
        if selected_vars:
            test_lags = st.slider("Máximo número de lags:", 1, min(max_lags, 20), min(5, max_lags), key="ts_test_lags")
    
    if len(selected_vars) < 2:
        st.error("Selecciona al menos 2 variables para análisis de causalidad.")
        return
    
    # Preparar datos
    causality_data = ts_data[selected_vars].dropna()
    
    if len(causality_data) < 20:
        st.error("Se necesitan al menos 20 observaciones para análisis de causalidad.")
        return
    
    try:
        # Realizar pruebas de causalidad de Granger
        st.subheader("📊 Resultados de Causalidad de Granger")
        
        causality_results = []
        
        for i, cause_var in enumerate(selected_vars):
            for j, effect_var in enumerate(selected_vars):
                if i != j:  # No analizar causalidad de una variable consigo misma
                    try:
                        # Preparar datos para la prueba
                        y = causality_data[effect_var]
                        x = causality_data[cause_var]
                        
                        # Prueba de causalidad de Granger
                        from statsmodels.tsa.stattools import grangercausalitytests
                        
                        # Crear DataFrame con ambas variables
                        test_data = pd.DataFrame({effect_var: y, cause_var: x})
                        
                        # Realizar prueba
                        gc_result = grangercausalitytests(test_data, maxlag=test_lags, verbose=False)
                        
                        # Extraer p-valores
                        p_values = []
                        for lag in range(1, test_lags + 1):
                            if lag in gc_result:
                                f_stat = gc_result[lag][0]['ssr_ftest'][0]
                                p_value = gc_result[lag][0]['ssr_ftest'][1]
                                p_values.append(p_value)
                        
                        # Usar el p-valor mínimo
                        min_p_value = min(p_values) if p_values else 1.0
                        
                        # Determinar significancia
                        significant = min_p_value < 0.05
                        
                        causality_results.append({
                            'Variable causa': cause_var,
                            'Variable efecto': effect_var,
                            'P-valor mínimo': f"{min_p_value:.4f}",
                            'Significativo (5%)': 'Sí' if significant else 'No',
                            'Interpretación': f"{cause_var} causa {effect_var}" if significant else f"No hay evidencia de que {cause_var} cause {effect_var}"
                        })
                        
                    except Exception as e:
                        st.warning(f"Error en prueba {cause_var} -> {effect_var}: {str(e)}")
        
        # Mostrar resultados
        if causality_results:
            causality_df = pd.DataFrame(causality_results)
            st.dataframe(causality_df, width='stretch')
            
            # Resumen de causalidades significativas
            significant_results = [r for r in causality_results if r['Significativo (5%)'] == 'Sí']
            
            if significant_results:
                st.subheader("✅ Causalidades Significativas Detectadas")
                for result in significant_results:
                    st.success(f"🔗 {result['Interpretación']} (p-valor: {result['P-valor mínimo']})")
            else:
                st.info("ℹ️ No se detectaron causalidades significativas entre las variables analizadas.")
        
        # Matriz de causalidad
        if len(selected_vars) <= 5:  # Solo para un número limitado de variables
            st.subheader("📊 Matriz de Causalidad")
            
            # Crear matriz de p-valores
            n_vars = len(selected_vars)
            causality_matrix = np.ones((n_vars, n_vars))
            
            for result in causality_results:
                cause_idx = selected_vars.index(result['Variable causa'])
                effect_idx = selected_vars.index(result['Variable efecto'])
                p_value = float(result['P-valor mínimo'])
                causality_matrix[effect_idx, cause_idx] = p_value
            
            # Crear heatmap
            fig = go.Figure(data=go.Heatmap(
                z=causality_matrix,
                x=selected_vars,
                y=selected_vars,
                colorscale='RdYlBu_r',
                zmin=0,
                zmax=1,
                text=np.round(causality_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Matriz de P-valores de Causalidad de Granger",
                xaxis_title="Variable Causa",
                yaxis_title="Variable Efecto",
                template=THEME_TEMPLATE
            )
            
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
            
            st.caption("💡 **Interpretación:** Valores bajos (azul) indican evidencia de causalidad. Diagonal principal = 1 (una variable no puede causarse a sí misma).")
    
    except Exception as e:
        st.error(f"Error en análisis de causalidad: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


def display_structural_change_detection(ts_data: pd.DataFrame, time_col: str, variables: list):
    """Detección de cambios estructurales"""
    st.subheader("⚡ Detección de Cambios Estructurales")
    
    if not variables:
        st.warning("No hay variables seleccionadas para análisis.")
        return
    
    if not RUPTURES_AVAILABLE:
        st.warning("⚠️ ruptures no está disponible. La detección de cambios estructurales está limitada.")
        return
    
    # Seleccionar variable para detección
    main_var = st.selectbox("Variable para detección de cambios:", variables, key="ts_change_var")
    
    if not main_var:
        return
    
    # Obtener serie temporal
    series = ts_data[main_var].dropna()
    
    if len(series) < 20:
        st.error("Se necesitan al menos 20 observaciones para detección de cambios estructurales.")
        return
    
    # Configuración
    st.subheader("⚙️ Configuración de Detección")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        change_methods = st.multiselect(
            "Métodos de detección:",
            ["Pelt", "Binary Segmentation", "Bottom-up", "Window-based"],
            default=["Pelt"],
            key="ts_change_methods"
        )
    
    with col2:
        if change_methods:
            min_segment_size = st.slider("Tamaño mínimo de segmento:", 5, min(20, len(series)//4), 10, key="ts_min_segment")
    
    with col3:
        if change_methods:
            max_changes = st.slider("Máximo número de cambios:", 1, 10, 5, key="ts_max_changes")
    
    if not change_methods:
        st.error("Selecciona al menos un método de detección.")
        return
    
    # Detectar cambios estructurales
    change_results = {}
    
    for method in change_methods:
        try:
            with st.spinner(f"Detectando cambios con método {method}..."):
                if method == "Pelt":
                    changes = detect_pelt_changes(series, min_segment_size)
                elif method == "Binary Segmentation":
                    changes = detect_binary_segmentation_changes(series, min_segment_size)
                elif method == "Bottom-up":
                    changes = detect_bottom_up_changes(series, min_segment_size)
                elif method == "Window-based":
                    changes = detect_window_based_changes(series, min_segment_size)
                
                change_results[method] = changes
                
        except Exception as e:
            st.error(f"Error en método {method}: {str(e)}")
    
    # Mostrar resultados
    if change_results:
        display_change_results(series, change_results, main_var)
    else:
        st.error("No se pudieron detectar cambios estructurales con los métodos seleccionados.")


def detect_pelt_changes(series, min_segment_size):
    """Detectar cambios usando PELT"""
    model = "rbf"
    algo = rpt.Pelt(model=model, min_size=min_segment_size)
    algo.fit(series.values.reshape(-1, 1))
    result = algo.predict(pen=10)
    return result[:-1]  # Excluir el último punto


def detect_binary_segmentation_changes(series, min_segment_size):
    """Detectar cambios usando Binary Segmentation"""
    model = "rbf"
    algo = rpt.Binseg(model=model, min_size=min_segment_size)
    algo.fit(series.values.reshape(-1, 1))
    result = algo.predict(n_bkps=5)
    return result[:-1]  # Excluir el último punto


def detect_bottom_up_changes(series, min_segment_size):
    """Detectar cambios usando Bottom-up"""
    model = "rbf"
    algo = rpt.BottomUp(model=model, min_size=min_segment_size)
    algo.fit(series.values.reshape(-1, 1))
    result = algo.predict(n_bkps=5)
    return result[:-1]  # Excluir el último punto


def detect_window_based_changes(series, min_segment_size):
    """Detectar cambios usando Window-based"""
    model = "rbf"
    algo = rpt.Window(width=min_segment_size, model=model, min_size=min_segment_size)
    algo.fit(series.values.reshape(-1, 1))
    result = algo.predict(n_bkps=5)
    return result[:-1]  # Excluir el último punto


def display_change_results(series, change_results, main_var):
    """Mostrar resultados de detección de cambios estructurales"""
    st.subheader("📊 Resultados de Detección de Cambios")
    
    # Estadísticas de cambios detectados
    st.subheader("📈 Estadísticas de Cambios")
    
    change_stats = []
    for method, changes in change_results.items():
        num_changes = len(changes)
        change_stats.append({
            'Método': method,
            'Cambios detectados': num_changes,
            'Posiciones': str(changes) if changes else 'Ninguno'
        })
    
    stats_df = pd.DataFrame(change_stats)
    st.dataframe(stats_df, width='stretch')
    
    # Visualización de cambios
    st.subheader("📈 Visualización de Cambios Estructurales")
    
    fig = go.Figure()
    
    # Serie temporal completa
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines',
        name=main_var,
        line=dict(color='blue', width=2)
    ))
    
    # Cambios detectados por método
    colors = ['red', 'orange', 'green', 'purple']
    
    for i, (method, changes) in enumerate(change_results.items()):
        color = colors[i % len(colors)]
        
        if changes:
            change_dates = series.index[changes]
            change_values = series.iloc[changes]
            
            fig.add_trace(go.Scatter(
                x=change_dates,
                y=change_values,
                mode='markers',
                name=f'Cambios {method}',
                marker=dict(
                    color=color,
                    size=10,
                    symbol='diamond',
                    line=dict(width=2, color='white')
                )
            ))
    
    fig.update_layout(
        title=f"Cambios Estructurales Detectados: {main_var}",
        xaxis_title="Tiempo",
        yaxis_title="Valor",
        template=THEME_TEMPLATE,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
    
    # Análisis de segmentos
    st.subheader("📋 Análisis de Segmentos")
    
    for method, changes in change_results.items():
        if changes:
            with st.expander(f"Segmentos detectados por {method}"):
                # Crear segmentos
                segment_points = [0] + changes + [len(series)]
                segments = []
                
                for i in range(len(segment_points) - 1):
                    start_idx = segment_points[i]
                    end_idx = segment_points[i + 1]
                    
                    segment_data = series.iloc[start_idx:end_idx]
                    
                    segments.append({
                        'Segmento': i + 1,
                        'Inicio': segment_data.index[0],
                        'Fin': segment_data.index[-1],
                        'Duración': len(segment_data),
                        'Media': f"{segment_data.mean():.3f}",
                        'Desv. Std': f"{segment_data.std():.3f}",
                        'Mínimo': f"{segment_data.min():.3f}",
                        'Máximo': f"{segment_data.max():.3f}"
                    })
                
                segment_df = pd.DataFrame(segments)
                st.dataframe(segment_df, width='stretch')
    
    # Exportar resultados
    st.subheader("💾 Exportar Resultados")
    
    if st.button("Descargar cambios estructurales", key="ts_download_changes"):
        # Crear DataFrame con todos los cambios
        all_changes = []
        
        for method, changes in change_results.items():
            if changes:
                for change_idx in changes:
                    change_date = series.index[change_idx]
                    change_value = series.iloc[change_idx]
                    
                    all_changes.append({
                        'Fecha': change_date,
                        'Valor': change_value,
                        'Método': method,
                        'Posición': change_idx
                    })
        
        if all_changes:
            changes_df = pd.DataFrame(all_changes)
            csv_data = changes_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Descargar CSV",
                data=csv_data,
                file_name=f"cambios_estructurales_{main_var}.csv",
                mime="text/csv",
                key="ts_download_csv_changes"
            )
    
    with tabs[6]:  # Resumen Ejecutivo
        st.subheader("📋 Resumen Ejecutivo del Análisis de Series Temporales")
        
        # Generar resumen automático
        summary = generate_timeseries_summary(time_series_data, time_col, analysis_variables)
        
        # Mostrar resumen en columnas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Hallazgos Principales")
            for finding in summary['findings']:
                st.write(f"• {finding}")
        
        with col2:
            st.subheader("⚠️ Alertas y Recomendaciones")
            for alert in summary['alerts']:
                st.write(f"• {alert}")
        
        # Métricas clave
        st.subheader("📊 Métricas Clave")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Período de Análisis", summary['period'])
        with col2:
            st.metric("Variables Analizadas", summary['n_variables'])
        with col3:
            st.metric("Tendencia General", summary['trend'])
        with col4:
            st.metric("Recomendación", summary['recommendation'])


def generate_timeseries_summary(ts_data: pd.DataFrame, time_col: str, variables: List[str]) -> dict:
    """Genera un resumen ejecutivo del análisis de series temporales"""
    findings = []
    alerts = []
    
    n_variables = len(variables)
    n_observations = len(ts_data)
    
    # Análisis del período
    if hasattr(ts_data.index, 'min') and hasattr(ts_data.index, 'max'):
        start_date = ts_data.index.min()
        end_date = ts_data.index.max()
        period_days = (end_date - start_date).days
        period = f"{start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')} ({period_days} días)"
    else:
        period = f"{n_observations} observaciones"
    
    # Análisis de tendencia general
    trend_analysis = []
    for var in variables:
        if var in ts_data.columns:
            try:
                # Regresión lineal simple para tendencia
                x = np.arange(len(ts_data[var]))
                y = ts_data[var].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                if p_value < 0.05:
                    if slope > 0:
                        trend_analysis.append(f"{var}: Tendencia creciente")
                    else:
                        trend_analysis.append(f"{var}: Tendencia decreciente")
                else:
                    trend_analysis.append(f"{var}: Sin tendencia significativa")
            except:
                trend_analysis.append(f"{var}: No se pudo analizar")
    
    # Determinar tendencia general
    if any("creciente" in t for t in trend_analysis):
        trend = "Creciente"
    elif any("decreciente" in t for t in trend_analysis):
        trend = "Decreciente"
    else:
        trend = "Estable"
    
    # Análisis de estacionalidad
    seasonal_vars = []
    for var in variables:
        if var in ts_data.columns:
            try:
                # Test de estacionalidad simple (varianza por mes)
                if hasattr(ts_data.index, 'month'):
                    monthly_var = ts_data[var].groupby(ts_data.index.month).var()
                    if monthly_var.std() > monthly_var.mean() * 0.5:
                        seasonal_vars.append(var)
            except:
                pass
    
    # Análisis de outliers
    outlier_vars = []
    for var in variables:
        if var in ts_data.columns:
            try:
                Q1 = ts_data[var].quantile(0.25)
                Q3 = ts_data[var].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ts_data[(ts_data[var] < Q1 - 1.5*IQR) | (ts_data[var] > Q3 + 1.5*IQR)]
                if len(outliers) > 0:
                    outlier_vars.append(var)
            except:
                pass
    
    # Generar hallazgos
    findings.append(f"Período de análisis: {period}")
    findings.append(f"Variables analizadas: {n_variables}")
    findings.append(f"Tendencia general: {trend.lower()}")
    
    if seasonal_vars:
        findings.append(f"Variables con estacionalidad: {', '.join(seasonal_vars)}")
    
    if outlier_vars:
        findings.append(f"Variables con outliers: {', '.join(outlier_vars)}")
    
    # Análisis de calidad de datos
    missing_pct = ts_data.isna().sum().sum() / (n_variables * n_observations) * 100
    if missing_pct < 5:
        data_quality = "Excelente"
    elif missing_pct < 15:
        data_quality = "Buena"
    elif missing_pct < 30:
        data_quality = "Regular"
    else:
        data_quality = "Pobre"
    
    findings.append(f"Calidad de datos: {data_quality}")
    
    # Generar alertas
    if missing_pct > 15:
        alerts.append(f"Alto porcentaje de valores faltantes ({missing_pct:.1f}%)")
    
    if len(outlier_vars) > n_variables * 0.5:
        alerts.append("Muchas variables con outliers, revisar calidad de datos")
    
    if n_observations < 30:
        alerts.append("Pocas observaciones para análisis temporal robusto")
    
    if not seasonal_vars and n_observations > 365:
        alerts.append("No se detectó estacionalidad, verificar frecuencia de datos")
    
    # Recomendación
    if data_quality == "Excelente" and n_observations >= 100:
        recommendation = "Excelente"
    elif data_quality in ["Excelente", "Buena"] and n_observations >= 50:
        recommendation = "Buena"
    elif data_quality in ["Excelente", "Buena", "Regular"]:
        recommendation = "Revisar"
    else:
        recommendation = "Requiere atención"
    
    return {
        'findings': findings,
        'alerts': alerts,
        'period': period,
        'n_variables': n_variables,
        'trend': trend,
        'recommendation': recommendation
    }
