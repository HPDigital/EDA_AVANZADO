"""
Frame de análisis de series temporales
"""
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .plot_viewer import MultiPlotViewer
from utils.plot_utils import PlotBuilder


class TimeSeriesAnalysisFrame(ttk.Frame):
    """Frame para análisis de series temporales"""
    
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.current_df = None
        self.create_widgets()
    
    def create_widgets(self):
        left_panel = ttk.Frame(self, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        ttk.Label(left_panel, text="Series Temporales",
                 font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        var_frame = ttk.LabelFrame(left_panel, text="Variable", padding=10)
        var_frame.pack(fill=tk.X, pady=5)
        
        self.var_combo = ttk.Combobox(var_frame, state='readonly')
        self.var_combo.pack(fill=tk.X)
        
        analysis_frame = ttk.LabelFrame(left_panel, text="Análisis", padding=10)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_frame, text="📈 Serie temporal",
                  command=self.plot_timeseries).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="📊 Descomposición",
                  command=self.decompose_series).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="🔮 Predicción simple",
                  command=self.simple_forecast).pack(fill=tk.X, pady=2)
        
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.plot_viewer = MultiPlotViewer(right_panel)
        self.plot_viewer.pack(fill=tk.BOTH, expand=True)
    
    def update_data(self, df, data_results):
        self.current_df = df
        self.time_column = None
        
        if df is not None:
            self.detect_time_column()
            data_columns = [col for col in df.columns if not col.startswith('_')]
            
            numeric_cols = [col for col in data_columns 
                          if pd.api.types.is_numeric_dtype(df[col])]
            self.var_combo['values'] = numeric_cols
            if numeric_cols:
                self.var_combo.current(0)
    
    def detect_time_column(self):
        """Detecta la columna temporal en el DataFrame"""
        if self.current_df is None:
            return
            
        data_columns = [col for col in self.current_df.columns if not col.startswith('_')]
        
        # Detectar columna temporal (datetime/timestamp)
        for col in data_columns:
            type_col = f'_type_{col}'
            if type_col in self.current_df.columns:
                col_type = self.current_df[type_col].iloc[0] if len(self.current_df) > 0 else None
                if col_type in ['datetime', 'time']:
                    self.time_column = col
                    return
        
        # Si no hay metadata, intentar detectar por tipo pandas
        if self.time_column is None:
            for col in data_columns:
                if pd.api.types.is_datetime64_any_dtype(self.current_df[col]):
                    self.time_column = col
                    return
    
    def plot_timeseries(self):
        var_name = self.var_combo.get()
        if not var_name or self.current_df is None:
            messagebox.showwarning("Sin selección", "Seleccione una variable")
            return

        try:
            # Obtener datos numéricos
            df_clean = self.current_df.copy()
            numeric_data = pd.to_numeric(df_clean[var_name], errors='coerce')
            
            # Si hay columna temporal, usarla como índice
            if self.time_column is not None and self.time_column in df_clean.columns:
                # Convertir columna temporal a datetime si no lo es
                time_data = df_clean[self.time_column]
                if not pd.api.types.is_datetime64_any_dtype(time_data):
                    try:
                        time_data = pd.to_datetime(time_data, errors='coerce')
                    except:
                        pass
                
                # Crear DataFrame con tiempo e índice, eliminar NaN
                temp_df = pd.DataFrame({
                    'time': time_data,
                    'value': numeric_data
                }).dropna()
                
                if len(temp_df) == 0:
                    messagebox.showwarning("Sin datos", "No hay datos válidos para graficar")
                    return
                
                # Crear serie con índice temporal
                series = pd.Series(temp_df['value'].values, index=temp_df['time'])
                series.name = var_name
            else:
                # Sin columna temporal, usar índice numérico
                series = numeric_data.dropna()
                series.name = var_name

            # Usar PlotBuilder para serie temporal
            fig = PlotBuilder.create_time_series_plot(
                data=series,
                title=f"Serie Temporal - {var_name}"
            )

            self.plot_viewer.add_plot("Serie", fig, var_name, plot_type='matplotlib')
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def decompose_series(self):
        var_name = self.var_combo.get()
        if not var_name or self.current_df is None:
            return

        try:
            # Obtener datos numéricos
            df_clean = self.current_df.copy()
            numeric_data = pd.to_numeric(df_clean[var_name], errors='coerce')
            
            # Si hay columna temporal, usarla como índice
            if self.time_column is not None and self.time_column in df_clean.columns:
                time_data = df_clean[self.time_column]
                if not pd.api.types.is_datetime64_any_dtype(time_data):
                    try:
                        time_data = pd.to_datetime(time_data, errors='coerce')
                    except:
                        pass
                
                temp_df = pd.DataFrame({
                    'time': time_data,
                    'value': numeric_data
                }).dropna()
                
                if len(temp_df) < 10:
                    messagebox.showwarning("Datos insuficientes",
                                         "Se necesitan al menos 10 observaciones")
                    return
                
                series = pd.Series(temp_df['value'].values, index=temp_df['time'])
                x_values = temp_df['time']
                x_label = self.time_column
            else:
                series = numeric_data.dropna()
                if len(series) < 10:
                    messagebox.showwarning("Datos insuficientes",
                                         "Se necesitan al menos 10 observaciones")
                    return
                x_values = range(len(series))
                x_label = 'Índice'

            # Tendencia (media móvil)
            window = min(5, len(series) // 2)
            trend = pd.Series(series.values).rolling(window=window, center=True).mean()
            residual = series.values - trend

            # Crear subplots con Matplotlib
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

            # Serie original
            ax1.plot(x_values, series.values, 'b-', linewidth=1.5)
            ax1.set_ylabel(var_name, fontsize=11)
            ax1.set_title('Original', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            if self.time_column:
                ax1.tick_params(axis='x', rotation=45)

            # Tendencia
            ax2.plot(x_values, trend.values, 'r-', linewidth=2)
            ax2.set_ylabel('Tendencia', fontsize=11)
            ax2.set_title('Tendencia (Media Móvil)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Residuo
            ax3.plot(x_values, residual, 'g-', linewidth=1)
            ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax3.set_ylabel('Residuo', fontsize=11)
            ax3.set_xlabel(x_label, fontsize=11)
            ax3.set_title('Residuo', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            if self.time_column:
                ax3.tick_params(axis='x', rotation=45)

            plt.suptitle(f"Descomposición - {var_name}", fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()

            self.plot_viewer.add_plot("Descomposición", fig, var_name, plot_type='matplotlib')
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def simple_forecast(self):
        var_name = self.var_combo.get()
        if not var_name or self.current_df is None:
            return

        try:
            from sklearn.linear_model import LinearRegression

            # Obtener datos numéricos
            df_clean = self.current_df.copy()
            numeric_data = pd.to_numeric(df_clean[var_name], errors='coerce')
            
            # Si hay columna temporal, usarla como índice
            if self.time_column is not None and self.time_column in df_clean.columns:
                time_data = df_clean[self.time_column]
                if not pd.api.types.is_datetime64_any_dtype(time_data):
                    try:
                        time_data = pd.to_datetime(time_data, errors='coerce')
                    except:
                        pass
                
                temp_df = pd.DataFrame({
                    'time': time_data,
                    'value': numeric_data
                }).dropna()
                
                if len(temp_df) < 5:
                    messagebox.showwarning("Datos insuficientes", "Se necesitan al menos 5 observaciones")
                    return
                
                series = pd.Series(temp_df['value'].values, index=temp_df['time'])
                x_values = temp_df['time']
                use_time_axis = True
            else:
                series = numeric_data.dropna()
                if len(series) < 5:
                    messagebox.showwarning("Datos insuficientes", "Se necesitan al menos 5 observaciones")
                    return
                x_values = range(len(series))
                use_time_axis = False

            # Preparar datos para regresión
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values

            lr = LinearRegression()
            lr.fit(X, y)

            n_forecast = 5
            X_future = np.arange(len(series), len(series) + n_forecast).reshape(-1, 1)
            y_pred = lr.predict(X_future)

            # Crear gráfico con Matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))

            if use_time_axis:
                # Serie histórica con fechas
                ax.plot(x_values, series.values, 'b-o', label='Histórico', linewidth=2, markersize=4)
                
                # Generar fechas futuras
                time_delta = x_values.iloc[-1] - x_values.iloc[-2] if len(x_values) > 1 else pd.Timedelta(days=1)
                future_dates = pd.date_range(start=x_values.iloc[-1] + time_delta, 
                                            periods=n_forecast, 
                                            freq=time_delta)
                
                # Predicción con fechas futuras
                ax.plot(future_dates, y_pred, 'r--o', label='Predicción', linewidth=2, markersize=6)
                
                # Línea vertical separando histórico de predicción
                ax.axvline(x=x_values.iloc[-1], color='gray', linestyle=':', linewidth=1.5)
                
                ax.set_xlabel(self.time_column, fontsize=12)
                ax.tick_params(axis='x', rotation=45)
            else:
                # Serie histórica con índice numérico
                ax.plot(range(len(series)), series.values, 'b-o', label='Histórico', linewidth=2, markersize=4)
                
                # Predicción
                ax.plot(range(len(series), len(series) + n_forecast), y_pred,
                       'r--o', label='Predicción', linewidth=2, markersize=6)
                
                # Línea vertical separando histórico de predicción
                ax.axvline(x=len(series)-0.5, color='gray', linestyle=':', linewidth=1.5)
                
                ax.set_xlabel('Índice', fontsize=12)

            ax.set_ylabel(var_name, fontsize=12)
            ax.set_title(f'Predicción Simple - {var_name}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            self.plot_viewer.add_plot("Predicción", fig, var_name, plot_type='matplotlib')
        except Exception as e:
            messagebox.showerror("Error", str(e))

