"""
Funciones adicionales para pronósticos y análisis avanzado de series temporales
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
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
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False


def display_forecasting_analysis(ts_data: pd.DataFrame, time_col: str, variables: list, test_size: float, confidence_level: float):
    """Análisis de pronósticos de series temporales"""
    st.subheader("🔮 Análisis de Pronósticos")
    
    if not variables:
        st.warning("No hay variables seleccionadas para análisis.")
        return
    
    # Seleccionar variable para pronóstico
    main_var = st.selectbox("Variable para pronóstico:", variables, key="ts_forecast_var")
    
    if not main_var:
        return
    
    # Obtener serie temporal
    series = ts_data[main_var].dropna()
    
    if len(series) < 20:
        st.error("Se necesitan al menos 20 observaciones para pronósticos.")
        return
    
    # Configuración de pronósticos
    st.subheader("⚙️ Configuración de Pronósticos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_periods = st.number_input("Períodos a pronosticar:", min_value=1, max_value=50, value=12, key="ts_forecast_periods")
    
    with col2:
        available_models = []
        if STATSMODELS_AVAILABLE:
            available_models.extend(["ARIMA", "SARIMA", "Holt-Winters", "VAR"])
        if PROPHET_AVAILABLE:
            available_models.append("Prophet")
        
        selected_models = st.multiselect(
            "Modelos de pronóstico:",
            available_models,
            default=available_models[:min(3, len(available_models))] if available_models else [],
            key="ts_forecast_models"
        )
    
    with col3:
        if "ARIMA" in selected_models or "SARIMA" in selected_models:
            auto_arima = st.checkbox("Auto-ARIMA (selección automática de parámetros)", value=True, key="ts_auto_arima")
    
    if not selected_models:
        st.error("Selecciona al menos un modelo de pronóstico.")
        return
    
    # Dividir datos en entrenamiento y prueba
    split_point = int(len(series) * (1 - test_size))
    train_series = series[:split_point]
    test_series = series[split_point:]
    
    st.info(f"📊 Datos de entrenamiento: {len(train_series)} observaciones")
    st.info(f"📊 Datos de prueba: {len(test_series)} observaciones")
    
    # Realizar pronósticos
    forecasts = {}
    model_metrics = {}
    
    for model_name in selected_models:
        try:
            with st.spinner(f"Entrenando modelo {model_name}..."):
                if model_name == "ARIMA" and STATSMODELS_AVAILABLE:
                    forecast, metrics = fit_arima_model(train_series, test_series, forecast_periods, auto_arima)
                    forecasts[model_name] = forecast
                    model_metrics[model_name] = metrics
                
                elif model_name == "SARIMA" and STATSMODELS_AVAILABLE:
                    forecast, metrics = fit_sarima_model(train_series, test_series, forecast_periods)
                    forecasts[model_name] = forecast
                    model_metrics[model_name] = metrics
                
                elif model_name == "Holt-Winters" and STATSMODELS_AVAILABLE:
                    forecast, metrics = fit_holt_winters_model(train_series, test_series, forecast_periods)
                    forecasts[model_name] = forecast
                    model_metrics[model_name] = metrics
                
                elif model_name == "Prophet" and PROPHET_AVAILABLE:
                    forecast, metrics = fit_prophet_model(train_series, test_series, forecast_periods, confidence_level)
                    forecasts[model_name] = forecast
                    model_metrics[model_name] = metrics
                
                elif model_name == "VAR" and STATSMODELS_AVAILABLE and len(variables) > 1:
                    forecast, metrics = fit_var_model(ts_data, variables, time_col, forecast_periods, test_size)
                    forecasts[model_name] = forecast
                    model_metrics[model_name] = metrics
        
        except Exception as e:
            st.error(f"Error entrenando modelo {model_name}: {str(e)}")
    
    # Mostrar resultados
    if forecasts:
        display_forecast_results(series, train_series, test_series, forecasts, model_metrics, confidence_level)
    else:
        st.error("No se pudieron generar pronósticos con los modelos seleccionados.")


def fit_arima_model(train_series, test_series, forecast_periods, auto_arima=True):
    """Ajustar modelo ARIMA"""
    if auto_arima:
        # Auto-ARIMA simple (búsqueda básica)
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(0, 4):
            for d in range(0, 3):
                for q in range(0, 4):
                    try:
                        model = ARIMA(train_series, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        order = best_order
    else:
        # Parámetros manuales
        order = (1, 1, 1)
    
    # Entrenar modelo con mejor orden
    model = ARIMA(train_series, order=order)
    fitted_model = model.fit()
    
    # Pronósticos
    forecast = fitted_model.get_forecast(steps=forecast_periods)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # Métricas en datos de prueba
    test_forecast = fitted_model.get_forecast(steps=len(test_series))
    test_pred = test_forecast.predicted_mean
    
    metrics = calculate_forecast_metrics(test_series, test_pred)
    
    return {
        'mean': forecast_mean,
        'ci_lower': forecast_ci.iloc[:, 0],
        'ci_upper': forecast_ci.iloc[:, 1],
        'model': fitted_model
    }, metrics


def fit_sarima_model(train_series, test_series, forecast_periods):
    """Ajustar modelo SARIMA"""
    # Parámetros SARIMA básicos
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)  # Asumiendo datos mensuales
    
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit()
    
    # Pronósticos
    forecast = fitted_model.get_forecast(steps=forecast_periods)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # Métricas en datos de prueba
    test_forecast = fitted_model.get_forecast(steps=len(test_series))
    test_pred = test_forecast.predicted_mean
    
    metrics = calculate_forecast_metrics(test_series, test_pred)
    
    return {
        'mean': forecast_mean,
        'ci_lower': forecast_ci.iloc[:, 0],
        'ci_upper': forecast_ci.iloc[:, 1],
        'model': fitted_model
    }, metrics


def fit_holt_winters_model(train_series, test_series, forecast_periods):
    """Ajustar modelo Holt-Winters"""
    model = ExponentialSmoothing(
        train_series, 
        trend='add', 
        seasonal='add', 
        seasonal_periods=12
    )
    fitted_model = model.fit()
    
    # Pronósticos
    forecast = fitted_model.forecast(steps=forecast_periods)
    
    # Para Holt-Winters, no tenemos intervalos de confianza directos
    forecast_mean = forecast
    forecast_ci_lower = forecast * 0.9  # Aproximación simple
    forecast_ci_upper = forecast * 1.1
    
    # Métricas en datos de prueba
    test_pred = fitted_model.forecast(steps=len(test_series))
    
    metrics = calculate_forecast_metrics(test_series, test_pred)
    
    return {
        'mean': forecast_mean,
        'ci_lower': forecast_ci_lower,
        'ci_upper': forecast_ci_upper,
        'model': fitted_model
    }, metrics


def fit_prophet_model(train_series, test_series, forecast_periods, confidence_level):
    """Ajustar modelo Prophet"""
    # Preparar datos para Prophet
    prophet_data = pd.DataFrame({
        'ds': train_series.index,
        'y': train_series.values
    })
    
    model = Prophet(
        interval_width=confidence_level,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    
    model.fit(prophet_data)
    
    # Crear fechas futuras
    future = model.make_future_dataframe(periods=forecast_periods, freq='D')
    future = future[future['ds'] >= train_series.index.max()]
    
    # Pronósticos
    forecast = model.predict(future)
    
    forecast_mean = forecast['yhat']
    forecast_ci_lower = forecast['yhat_lower']
    forecast_ci_upper = forecast['yhat_upper']
    
    # Métricas en datos de prueba
    test_dates = pd.DataFrame({'ds': test_series.index})
    test_forecast = model.predict(test_dates)
    test_pred = pd.Series(test_forecast['yhat'].values, index=test_series.index)
    
    metrics = calculate_forecast_metrics(test_series, test_pred)
    
    return {
        'mean': forecast_mean,
        'ci_lower': forecast_ci_lower,
        'ci_upper': forecast_ci_upper,
        'model': model
    }, metrics


def fit_var_model(ts_data, variables, time_col, forecast_periods, test_size):
    """Ajustar modelo VAR"""
    # Seleccionar variables para VAR
    var_vars = st.multiselect(
        "Variables para modelo VAR:",
        variables,
        default=variables[:min(3, len(variables))],
        key="ts_var_variables"
    )
    
    if len(var_vars) < 2:
        raise ValueError("VAR requiere al menos 2 variables")
    
    # Preparar datos
    var_data = ts_data[var_vars].dropna()
    
    # Dividir datos
    split_point = int(len(var_data) * (1 - test_size))
    train_data = var_data[:split_point]
    test_data = var_data[split_point:]
    
    # Entrenar modelo VAR
    model = VAR(train_data)
    fitted_model = model.fit(maxlags=5, ic='aic')
    
    # Pronósticos
    forecast = fitted_model.forecast(train_data.values[-fitted_model.k_ar:], steps=forecast_periods)
    
    # Crear DataFrame de pronósticos
    # Construir índice temporal respetando la frecuencia original si es posible
    try:
        inferred_freq = getattr(train_data.index, 'freqstr', None) or pd.infer_freq(train_data.index)
    except Exception:
        inferred_freq = None

    if inferred_freq is None:
        # Fallback: usar la diferencia mediana si es datetimes, o un índice de rango
        try:
            last = train_data.index[-1]
            forecast_index = pd.date_range(start=last, periods=forecast_periods+1, freq='D')[1:]
        except Exception:
            forecast_index = range(len(train_data), len(train_data) + forecast_periods)
    else:
        last = train_data.index[-1]
        forecast_index = pd.date_range(start=last, periods=forecast_periods+1, freq=inferred_freq)[1:]

    forecast_df = pd.DataFrame(
        forecast,
        columns=var_vars,
        index=forecast_index
    )
    
    # Métricas (solo para la primera variable)
    main_var = var_vars[0]
    test_pred = forecast_df[main_var][:len(test_data)]
    
    metrics = calculate_forecast_metrics(test_data[main_var], test_pred)
    
    return {
        'mean': forecast_df,
        'variables': var_vars,
        'model': fitted_model
    }, metrics


def calculate_forecast_metrics(actual, predicted):
    """Calcular métricas de pronóstico"""
    try:
        # Asegurar que las series tengan el mismo índice
        common_index = actual.index.intersection(predicted.index)
        if len(common_index) == 0:
            return {}
        
        actual_aligned = actual.loc[common_index]
        predicted_aligned = predicted.loc[common_index]
        
        mse = mean_squared_error(actual_aligned, predicted_aligned)
        mae = mean_absolute_error(actual_aligned, predicted_aligned)
        rmse = np.sqrt(mse)
        
        try:
            r2 = r2_score(actual_aligned, predicted_aligned)
        except:
            r2 = np.nan
        
        # MAPE (Mean Absolute Percentage Error) seguro
        denom = actual_aligned.replace(0, np.nan)
        mape = float(np.nanmean(np.abs((actual_aligned - predicted_aligned) / denom)) * 100)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse,
            'R²': r2,
            'MAPE': mape
        }
    except Exception as e:
        return {'Error': str(e)}


def display_forecast_results(series, train_series, test_series, forecasts, model_metrics, confidence_level):
    """Mostrar resultados de pronósticos"""
    st.subheader("📊 Resultados de Pronósticos")
    
    # Comparar métricas de modelos
    if model_metrics:
        st.subheader("📈 Comparación de Modelos")
        
        metrics_df = pd.DataFrame(model_metrics).T
        metrics_df = metrics_df.round(4)
        
        # Mostrar métricas
        st.dataframe(metrics_df, width='stretch')
        
        # Identificar mejor modelo por RMSE
        if 'RMSE' in metrics_df.columns:
            best_model = metrics_df['RMSE'].idxmin()
            st.success(f"🏆 **Mejor modelo por RMSE:** {best_model}")
    
    # Visualización de pronósticos
    st.subheader("📈 Visualización de Pronósticos")
    
    fig = go.Figure()
    
    # Datos históricos
    fig.add_trace(go.Scatter(
        x=train_series.index,
        y=train_series.values,
        mode='lines',
        name='Datos de entrenamiento',
        line=dict(color='blue', width=2)
    ))
    
    # Datos de prueba
    fig.add_trace(go.Scatter(
        x=test_series.index,
        y=test_series.values,
        mode='lines',
        name='Datos de prueba',
        line=dict(color='green', width=2)
    ))
    
    # Pronósticos
    colors = ['red', 'purple', 'orange', 'brown', 'pink']
    
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        color = colors[i % len(colors)]
        
        if isinstance(forecast['mean'], pd.DataFrame):
            # Modelo VAR
            main_var = forecast['variables'][0]
            forecast_mean = forecast['mean'][main_var]
        else:
            forecast_mean = forecast['mean']
        
        # Crear índice para pronósticos
        last_date = series.index[-1]
        try:
            inferred_freq = getattr(series.index, 'freqstr', None) or pd.infer_freq(series.index)
        except Exception:
            inferred_freq = None
        if inferred_freq is None:
            # Fallback razonable
            try:
                forecast_dates = pd.date_range(start=last_date, periods=len(forecast_mean)+1, freq='D')[1:]
            except Exception:
                forecast_dates = list(range(len(series), len(series) + len(forecast_mean)))
        else:
            forecast_dates = pd.date_range(start=last_date, periods=len(forecast_mean)+1, freq=inferred_freq)[1:]
        
        # Pronóstico medio
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_mean,
            mode='lines',
            name=f'Pronóstico {model_name}',
            line=dict(color=color, width=2, dash='dash')
        ))
        
        # Intervalos de confianza (si están disponibles)
        if 'ci_lower' in forecast and 'ci_upper' in forecast:
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast['ci_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast['ci_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=f'rgba(255,0,0,0.2)' if color == 'red' else f'rgba(0,0,255,0.2)',
                name=f'IC {confidence_level*100:.0f}% {model_name}',
                showlegend=True
            ))
    
    fig.update_layout(
        title="Pronósticos de Series Temporales",
        xaxis_title="Tiempo",
        yaxis_title="Valor",
        template=THEME_TEMPLATE,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
    
    # Exportar pronósticos
    st.subheader("💾 Exportar Pronósticos")
    
    if st.button("Descargar pronósticos", key="ts_download_forecasts"):
        # Crear DataFrame con pronósticos
        forecast_data = {}
        
        for model_name, forecast in forecasts.items():
            if isinstance(forecast['mean'], pd.DataFrame):
                main_var = forecast['variables'][0]
                forecast_data[f'{model_name}_{main_var}'] = forecast['mean'][main_var]
            else:
                forecast_data[model_name] = forecast['mean']
        
        forecast_df = pd.DataFrame(forecast_data)
        
        # Crear CSV
        csv_data = forecast_df.to_csv()
        
        st.download_button(
            label="📥 Descargar CSV",
            data=csv_data,
            file_name="pronosticos_series_temporales.csv",
            mime="text/csv",
            key="ts_download_csv_forecasts"
        )


def display_anomaly_detection(ts_data: pd.DataFrame, time_col: str, variables: list):
    """Detección de anomalías en series temporales"""
    st.subheader("🎯 Detección de Anomalías")
    
    if not variables:
        st.warning("No hay variables seleccionadas para análisis.")
        return
    
    # Seleccionar variable para detección de anomalías
    main_var = st.selectbox("Variable para detección de anomalías:", variables, key="ts_anomaly_var")
    
    if not main_var:
        return
    
    # Obtener serie temporal
    series = ts_data[main_var].dropna()
    
    if len(series) < 10:
        st.error("Se necesitan al menos 10 observaciones para detección de anomalías.")
        return
    
    # Configuración de detección
    st.subheader("⚙️ Configuración de Detección")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        anomaly_methods = st.multiselect(
            "Métodos de detección:",
            ["Z-Score", "IQR", "Isolation Forest", "DBSCAN", "LOF"],
            default=["Z-Score", "IQR"],
            key="ts_anomaly_methods"
        )
    
    with col2:
        if "Z-Score" in anomaly_methods:
            z_threshold = st.slider("Umbral Z-Score:", 1.0, 5.0, 3.0, 0.1, key="ts_z_threshold")
        
        if "IQR" in anomaly_methods:
            iqr_multiplier = st.slider("Multiplicador IQR:", 1.0, 3.0, 1.5, 0.1, key="ts_iqr_multiplier")
    
    with col3:
        if "Isolation Forest" in anomaly_methods:
            contamination = st.slider("Contaminación (Isolation Forest):", 0.01, 0.5, 0.1, 0.01, key="ts_contamination")
    
    if not anomaly_methods:
        st.error("Selecciona al menos un método de detección.")
        return
    
    # Detectar anomalías
    anomaly_results = {}
    
    for method in anomaly_methods:
        try:
            if method == "Z-Score":
                anomalies = detect_zscore_anomalies(series, z_threshold)
            elif method == "IQR":
                anomalies = detect_iqr_anomalies(series, iqr_multiplier)
            elif method == "Isolation Forest":
                anomalies = detect_isolation_forest_anomalies(series, contamination)
            elif method == "DBSCAN":
                anomalies = detect_dbscan_anomalies(series)
            elif method == "LOF":
                anomalies = detect_lof_anomalies(series)
            
            anomaly_results[method] = anomalies
            
        except Exception as e:
            st.error(f"Error en método {method}: {str(e)}")
    
    # Mostrar resultados
    if anomaly_results:
        display_anomaly_results(series, anomaly_results, main_var)
    else:
        st.error("No se pudieron detectar anomalías con los métodos seleccionados.")


def detect_zscore_anomalies(series, threshold):
    """Detectar anomalías usando Z-Score"""
    z_scores = np.abs((series - series.mean()) / series.std())
    anomalies = z_scores > threshold
    return anomalies


def detect_iqr_anomalies(series, multiplier):
    """Detectar anomalías usando IQR"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    anomalies = (series < lower_bound) | (series > upper_bound)
    return anomalies


def detect_isolation_forest_anomalies(series, contamination):
    """Detectar anomalías usando Isolation Forest"""
    from sklearn.ensemble import IsolationForest
    
    # Preparar datos para Isolation Forest
    X = series.values.reshape(-1, 1)
    
    # Entrenar modelo
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X)
    
    # Convertir a boolean (True = anomalía)
    anomalies = pd.Series(anomaly_labels == -1, index=series.index)
    return anomalies


def detect_dbscan_anomalies(series):
    """Detectar anomalías usando DBSCAN"""
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    
    # Preparar datos
    X = series.values.reshape(-1, 1)
    X_scaled = StandardScaler().fit_transform(X)
    
    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    # Los puntos con label -1 son outliers
    anomalies = pd.Series(cluster_labels == -1, index=series.index)
    return anomalies


def detect_lof_anomalies(series):
    """Detectar anomalías usando Local Outlier Factor"""
    from sklearn.neighbors import LocalOutlierFactor
    
    # Preparar datos
    X = series.values.reshape(-1, 1)
    
    # Aplicar LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    anomaly_labels = lof.fit_predict(X)
    
    # Convertir a boolean (True = anomalía)
    anomalies = pd.Series(anomaly_labels == -1, index=series.index)
    return anomalies


def display_anomaly_results(series, anomaly_results, main_var):
    """Mostrar resultados de detección de anomalías"""
    st.subheader("📊 Resultados de Detección de Anomalías")
    
    # Estadísticas de anomalías
    st.subheader("📈 Estadísticas de Anomalías")
    
    anomaly_stats = []
    for method, anomalies in anomaly_results.items():
        num_anomalies = anomalies.sum()
        percentage = (num_anomalies / len(anomalies)) * 100
        
        anomaly_stats.append({
            'Método': method,
            'Anomalías detectadas': num_anomalies,
            'Porcentaje': f"{percentage:.2f}%"
        })
    
    stats_df = pd.DataFrame(anomaly_stats)
    st.dataframe(stats_df, width='stretch')
    
    # Visualización de anomalías
    st.subheader("📈 Visualización de Anomalías")
    
    fig = go.Figure()
    
    # Serie temporal completa
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines',
        name=main_var,
        line=dict(color='blue', width=2)
    ))
    
    # Anomalías por método
    colors = ['red', 'orange', 'purple', 'green', 'brown']
    
    for i, (method, anomalies) in enumerate(anomaly_results.items()):
        color = colors[i % len(colors)]
        
        if anomalies.any():
            anomaly_points = series[anomalies]
            
            fig.add_trace(go.Scatter(
                x=anomaly_points.index,
                y=anomaly_points.values,
                mode='markers',
                name=f'Anomalías {method}',
                marker=dict(
                    color=color,
                    size=8,
                    symbol='x',
                    line=dict(width=2, color='white')
                )
            ))
    
    fig.update_layout(
        title=f"Detección de Anomalías: {main_var}",
        xaxis_title="Tiempo",
        yaxis_title="Valor",
        template=THEME_TEMPLATE,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
    
    # Tabla de anomalías detectadas
    st.subheader("📋 Detalles de Anomalías Detectadas")
    
    for method, anomalies in anomaly_results.items():
        if anomalies.any():
            with st.expander(f"Anomalías detectadas por {method}"):
                anomaly_data = series[anomalies]
                
                anomaly_df = pd.DataFrame({
                    'Fecha': anomaly_data.index,
                    'Valor': anomaly_data.values,
                    'Método': method
                })
                
                st.dataframe(anomaly_df, width='stretch')
    
    # Exportar resultados
    st.subheader("💾 Exportar Resultados")
    
    if st.button("Descargar anomalías detectadas", key="ts_download_anomalies"):
        # Crear DataFrame con todas las anomalías
        all_anomalies = []
        
        for method, anomalies in anomaly_results.items():
            if anomalies.any():
                anomaly_data = series[anomalies]
                for idx, value in anomaly_data.items():
                    all_anomalies.append({
                        'Fecha': idx,
                        'Valor': value,
                        'Método': method
                    })
        
        if all_anomalies:
            anomaly_df = pd.DataFrame(all_anomalies)
            csv_data = anomaly_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Descargar CSV",
                data=csv_data,
                file_name=f"anomalias_{main_var}.csv",
                mime="text/csv",
                key="ts_download_csv_anomalies"
            )

