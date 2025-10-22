"""
Frame de análisis univariado con capacidades estadísticas completas
"""
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from .plot_viewer import MultiPlotViewer
from utils.plot_utils import PlotBuilder


class UnivariateAnalysisFrame(ttk.Frame):
    """Frame para análisis univariado completo"""
    
    def __init__(self, parent, main_window):
        """
        Inicializa el frame de análisis univariado
        
        Args:
            parent: Widget padre
            main_window: Instancia de MainWindow
        """
        super().__init__(parent)
        self.main_window = main_window
        self.current_df = None
        self.current_data = None
        self.time_column = None  # Columna temporal detectada
        
        self.create_widgets()
    
    def create_widgets(self):
        """Crea los widgets del frame"""
        # Panel izquierdo (controles)
        left_panel = ttk.Frame(self, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        # Título
        ttk.Label(
            left_panel,
            text="Análisis Univariado",
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=10)
        
        # Selección de tipo de variable
        type_frame = ttk.LabelFrame(left_panel, text="Tipo de Variable", padding=10)
        type_frame.pack(fill=tk.X, pady=5)
        
        self.var_type = tk.StringVar(value="numeric")
        ttk.Radiobutton(
            type_frame,
            text="📊 Numérica",
            variable=self.var_type,
            value="numeric",
            command=self.on_type_changed
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            type_frame,
            text="📑 Categórica",
            variable=self.var_type,
            value="categorical",
            command=self.on_type_changed
        ).pack(anchor=tk.W)
        
        # Selección de variable
        var_frame = ttk.LabelFrame(left_panel, text="Variable a Analizar", padding=10)
        var_frame.pack(fill=tk.X, pady=5)
        
        self.variable_combo = ttk.Combobox(var_frame, state='readonly')
        self.variable_combo.pack(fill=tk.X)
        self.variable_combo.bind('<<ComboboxSelected>>', self.on_variable_selected)
        
        # Métricas básicas
        self.metrics_frame = ttk.LabelFrame(left_panel, text="Métricas Básicas", padding=10)
        self.metrics_frame.pack(fill=tk.X, pady=5)
        
        self.metrics_text = tk.Text(self.metrics_frame, height=8, wrap=tk.WORD,
                                   state=tk.DISABLED, font=('Courier', 9))
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # Botones de análisis
        analysis_frame = ttk.LabelFrame(left_panel, text="Análisis Disponibles", padding=10)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        self.numeric_buttons = []
        self.categorical_buttons = []
        
        # Botones para numéricos
        btn1 = ttk.Button(analysis_frame, text="📈 Serie temporal", 
                         command=self.plot_time_series)
        btn1.pack(fill=tk.X, pady=2)
        self.numeric_buttons.append(btn1)
        
        btn2 = ttk.Button(analysis_frame, text="📊 Distribución",
                         command=self.plot_distribution)
        btn2.pack(fill=tk.X, pady=2)
        self.numeric_buttons.append(btn2)
        
        btn3 = ttk.Button(analysis_frame, text="📉 Estadísticos",
                         command=self.show_extended_stats)
        btn3.pack(fill=tk.X, pady=2)
        self.numeric_buttons.append(btn3)
        
        btn4 = ttk.Button(analysis_frame, text="📊 Tests de normalidad",
                         command=self.run_normality_tests)
        btn4.pack(fill=tk.X, pady=2)
        self.numeric_buttons.append(btn4)
        
        btn5 = ttk.Button(analysis_frame, text="🔍 Detección de outliers",
                         command=self.detect_outliers)
        btn5.pack(fill=tk.X, pady=2)
        self.numeric_buttons.append(btn5)
        
        btn6 = ttk.Button(analysis_frame, text="🔄 Transformaciones",
                         command=self.show_transformations)
        btn6.pack(fill=tk.X, pady=2)
        self.numeric_buttons.append(btn6)
        
        # Botones para categóricos
        btn7 = ttk.Button(analysis_frame, text="📊 Frecuencias",
                         command=self.plot_frequencies)
        btn7.pack(fill=tk.X, pady=2)
        btn7.pack_forget()  # Ocultar inicialmente
        self.categorical_buttons.append(btn7)
        
        btn8 = ttk.Button(analysis_frame, text="📉 Estadísticas",
                         command=self.show_categorical_stats)
        btn8.pack(fill=tk.X, pady=2)
        btn8.pack_forget()
        self.categorical_buttons.append(btn8)
        
        # Panel derecho (visualizaciones)
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visor de gráficos múltiples
        self.plot_viewer = MultiPlotViewer(right_panel)
        self.plot_viewer.pack(fill=tk.BOTH, expand=True)
    
    def update_data(self, df, data_results):
        """
        Actualiza el frame con nuevos datos
        
        Args:
            df: DataFrame con los datos
            data_results: Resultados de la carga de datos
        """
        self.current_df = df
        self.current_data = data_results
        self.time_column = None
        
        if df is not None and isinstance(df, pd.DataFrame):
            self.detect_time_column()
            self.update_variable_list()
        else:
            self.variable_combo['values'] = []
            self.clear_metrics()
    
    def detect_time_column(self):
        """Detecta la columna temporal en el DataFrame"""
        if self.current_df is None:
            return
        
        # Filtrar columnas de metadatos
        data_columns = [col for col in self.current_df.columns if not col.startswith('_')]
        
        # Detectar columna temporal (datetime/timestamp)
        for col in data_columns:
            type_col = f'_type_{col}'
            if type_col in self.current_df.columns:
                col_type = self.current_df[type_col].iloc[0] if len(self.current_df) > 0 else None
                if col_type in ['datetime', 'time']:
                    self.time_column = col
                    break
        
        # También buscar columnas que contengan palabras clave temporales
        if self.time_column is None:
            for col in data_columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['fecha', 'date', 'time', 'tiempo']):
                    # Verificar si es realmente una columna temporal
                    if pd.api.types.is_datetime64_any_dtype(self.current_df[col]):
                        self.time_column = col
                        break
        
        # Si no hay metadata, intentar detectar por tipo pandas
        if self.time_column is None:
            for col in data_columns:
                if pd.api.types.is_datetime64_any_dtype(self.current_df[col]):
                    self.time_column = col
                    break
    
    def on_type_changed(self):
        """Callback cuando cambia el tipo de variable"""
        self.update_variable_list()
        self.toggle_analysis_buttons()
    
    def toggle_analysis_buttons(self):
        """Muestra/oculta botones según el tipo de variable"""
        if self.var_type.get() == "numeric":
            # Mostrar botones numéricos
            for btn in self.numeric_buttons:
                btn.pack(fill=tk.X, pady=2)
            # Ocultar botones categóricos
            for btn in self.categorical_buttons:
                btn.pack_forget()
        else:
            # Ocultar botones numéricos
            for btn in self.numeric_buttons:
                btn.pack_forget()
            # Mostrar botones categóricos
            for btn in self.categorical_buttons:
                btn.pack(fill=tk.X, pady=2)
    
    def update_variable_list(self):
        """Actualiza la lista de variables disponibles"""
        if self.current_df is None:
            return
        
        # Filtrar columnas de metadatos
        data_columns = [col for col in self.current_df.columns if not col.startswith('_')]
        
        if self.var_type.get() == "numeric":
            # Variables numéricas - excluir columnas temporales
            cols = []
            for col in data_columns:
                if pd.api.types.is_numeric_dtype(self.current_df[col]):
                    # Verificar que no sea una columna temporal
                    is_time_col = self.is_temporal_column(col)
                    if not is_time_col:
                        cols.append(col)
        else:
            # Variables categóricas - excluir columnas temporales y numéricas
            cols = []
            for col in data_columns:
                if not pd.api.types.is_numeric_dtype(self.current_df[col]):
                    # Verificar que no sea una columna temporal
                    is_time_col = self.is_temporal_column(col)
                    if not is_time_col:
                        cols.append(col)
        
        self.variable_combo['values'] = cols
        
        if cols:
            self.variable_combo.current(0)
            self.on_variable_selected(None)
        else:
            # Limpiar selección si no hay columnas disponibles
            self.variable_combo.set('')
            self.clear_metrics("No hay variables disponibles para este tipo")
    
    def is_temporal_column(self, col_name):
        """Verifica si una columna es temporal (datetime/timestamp)"""
        # Verificar por metadata
        type_col = f'_type_{col_name}'
        if type_col in self.current_df.columns and len(self.current_df) > 0:
            col_type = self.current_df[type_col].iloc[0]
            if col_type in ['datetime', 'time']:
                return True
        
        # Verificar por tipo pandas
        if pd.api.types.is_datetime64_any_dtype(self.current_df[col_name]):
            return True
        
        # Verificar por nombre (patrones comunes de fechas)
        col_lower = col_name.lower()
        time_keywords = ['date', 'fecha', 'time', 'tiempo', 'timestamp', 'datetime', 
                        'référence', 'reference', 'created', 'updated', 'modified']
        for keyword in time_keywords:
            if keyword in col_lower:
                return True
        
        return False
    
    def on_variable_selected(self, event):
        """Callback cuando se selecciona una variable"""
        var_name = self.variable_combo.get()
        
        if not var_name or self.current_df is None:
            return
        
        self.update_metrics(var_name)
    
    def update_metrics(self, var_name):
        """Actualiza las métricas básicas"""
        if self.current_df is None or not var_name:
            return
        
        series = self.current_df[var_name]
        
        if self.var_type.get() == "numeric":
            self.update_numeric_metrics(series, var_name)
        else:
            self.update_categorical_metrics(series, var_name)
    
    def update_numeric_metrics(self, series, var_name):
        """Actualiza métricas para variables numéricas"""
        clean_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(clean_series) == 0:
            self.clear_metrics("No hay valores numéricos válidos")
            return
        
        metrics = f"N válidos: {len(clean_series)}\n"
        metrics += f"N faltantes: {series.isna().sum()}\n"
        metrics += f"Media: {clean_series.mean():.2f}\n"
        metrics += f"Mediana: {clean_series.median():.2f}\n"
        metrics += f"Desv.Std: {clean_series.std():.2f}\n"
        metrics += f"Mín: {clean_series.min():.2f}\n"
        metrics += f"Máx: {clean_series.max():.2f}\n"
        
        self.set_metrics_text(metrics)
    
    def update_categorical_metrics(self, series, var_name):
        """Actualiza métricas para variables categóricas"""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            self.clear_metrics("No hay valores válidos")
            return
        
        value_counts = clean_series.value_counts()
        
        metrics = f"N válidos: {len(clean_series)}\n"
        metrics += f"N faltantes: {series.isna().sum()}\n"
        metrics += f"Categorías: {clean_series.nunique()}\n"
        metrics += f"Más frecuente: {value_counts.index[0]}\n"
        metrics += f"Frecuencia: {value_counts.iloc[0]}\n"
        
        self.set_metrics_text(metrics)
    
    def set_metrics_text(self, text):
        """Establece el texto de métricas"""
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, text)
        self.metrics_text.config(state=tk.DISABLED)
    
    def clear_metrics(self, message=""):
        """Limpia las métricas"""
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, message)
        self.metrics_text.config(state=tk.DISABLED)
    
    def get_current_series(self):
        """Obtiene la serie actual seleccionada"""
        var_name = self.variable_combo.get()
        
        if not var_name or self.current_df is None:
            messagebox.showwarning("Sin selección", "Seleccione una variable")
            return None, None
        
        series = self.current_df[var_name]
        
        if self.var_type.get() == "numeric":
            series = pd.to_numeric(series, errors='coerce')
        
        return series, var_name
    
    def plot_time_series(self):
        """Grafica la serie temporal"""
        series, var_name = self.get_current_series()
        if series is None:
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
                clean_series = pd.Series(temp_df['value'].values, index=temp_df['time'])
                clean_series.name = var_name
            else:
                # Sin columna temporal, usar índice numérico
                clean_series = numeric_data.dropna()
                clean_series.name = var_name

            if len(clean_series) == 0:
                messagebox.showwarning("Sin datos", "No hay datos válidos para graficar")
                return

            # Crear gráfico con Matplotlib
            fig = PlotBuilder.create_time_series_plot(
                clean_series,
                title=f"Serie Temporal - {var_name}"
            )

            # Añadir media móvil con mejor configuración
            window = min(5, len(clean_series) // 10)  # Ventana más pequeña para mejor suavizado
            if window >= 2:
                ma = clean_series.rolling(window=window, center=True).mean()
                ax = fig.axes[0]
                
                # Calcular media móvil con el mismo índice que la serie original
                ma_clean = ma.dropna()
                
                if len(ma_clean) > 0:
                    # Para series temporales, usar el índice temporal
                    ax.plot(ma_clean.index, ma_clean.values, 'r--', linewidth=3,
                           label=f'Media móvil ({window})', alpha=0.8)
                    
                    # Mejorar la leyenda
                    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

            self.plot_viewer.add_plot("Serie Temporal", fig, var_name, plot_type='matplotlib')

        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráfico:\n{str(e)}")
    
    def plot_distribution(self):
        """Grafica la distribución"""
        series, var_name = self.get_current_series()
        if series is None:
            return

        try:
            clean_series = series.dropna()

            if len(clean_series) == 0:
                messagebox.showwarning("Sin datos", "No hay datos válidos para graficar")
                return

            # Crear figura con subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Histograma
            ax1.hist(clean_series.values, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.set_title('Histograma')
            ax1.set_xlabel(var_name)
            ax1.set_ylabel('Frecuencia')
            ax1.grid(True, alpha=0.3)

            # Box plot
            ax2.boxplot(clean_series.values, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
            ax2.set_title('Box Plot')
            ax2.set_ylabel(var_name)
            ax2.grid(True, alpha=0.3, axis='y')

            fig.suptitle(f'Distribución - {var_name}', fontsize=14)
            plt.tight_layout()

            self.plot_viewer.add_plot("Distribución", fig, var_name, plot_type='matplotlib')

        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráfico:\n{str(e)}")
    
    def show_extended_stats(self):
        """Muestra estadísticos extendidos"""
        series, var_name = self.get_current_series()
        if series is None:
            return
        
        try:
            clean_series = series.dropna()
            
            if len(clean_series) < 3:
                messagebox.showwarning("Datos insuficientes", 
                                     "Se necesitan al menos 3 valores")
                return
            
            # Calcular estadísticos
            q = clean_series.quantile([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
            iqr = q.loc[0.75] - q.loc[0.25]
            cv = (clean_series.std() / clean_series.mean()) * 100 if clean_series.mean() != 0 else np.nan
            
            # Crear figura con estadísticos usando matplotlib
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.axis('off')
            
            # Preparar datos para la tabla
            stats_data = [
                ['Tendencia Central', '', ''],
                ['  Media', f"{clean_series.mean():.4f}", ''],
                ['  Mediana', f"{clean_series.median():.4f}", ''],
                ['  Moda', f"{clean_series.mode().iloc[0] if len(clean_series.mode()) > 0 else 'N/A'}", ''],
                ['', '', ''],
                ['Dispersión', '', ''],
                ['  Desv. Estándar', f"{clean_series.std():.4f}", ''],
                ['  Varianza', f"{clean_series.var():.4f}", ''],
                ['  Rango', f"{clean_series.max() - clean_series.min():.4f}", ''],
                ['  IQR', f"{iqr:.4f}", ''],
                ['  Coef. Variación', f"{cv:.2f}%", ''],
                ['', '', ''],
                ['Valores Extremos', '', ''],
                ['  Mínimo', f"{clean_series.min():.4f}", ''],
                ['  Máximo', f"{clean_series.max():.4f}", ''],
                ['', '', ''],
                ['Cuartiles', '', ''],
                ['  Q1 (25%)', f"{q.loc[0.25]:.4f}", ''],
                ['  Q2 (50%)', f"{q.loc[0.5]:.4f}", ''],
                ['  Q3 (75%)', f"{q.loc[0.75]:.4f}", ''],
                ['', '', ''],
                ['Forma de Distribución', '', ''],
                ['  Asimetría', f"{stats.skew(clean_series, bias=False):.4f}", ''],
                ['  Curtosis', f"{stats.kurtosis(clean_series, bias=False, fisher=True):.4f}", ''],
                ['', '', ''],
                ['Tamaño de Muestra', '', ''],
                ['  N válidos', f"{len(clean_series)}", ''],
                ['  N faltantes', f"{series.isna().sum()}", ''],
                ['  N total', f"{len(series)}", '']
            ]
            
            # Crear tabla
            table = ax.table(
                cellText=stats_data,
                colLabels=['Categoría', 'Valor', ''],
                cellLoc='left',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            # Estilizar tabla
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Colorear encabezados
            for i in range(3):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Colorear categorías principales
            for i, row in enumerate(stats_data):
                if row[0] and not row[0].startswith('  ') and row[0] != '':
                    for j in range(3):
                        table[(i+1, j)].set_facecolor('#E8F5E8')
                        table[(i+1, j)].set_text_props(weight='bold')
            
            plt.title(f'Estadísticos Extendidos - {var_name}', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Mostrar en el visor de gráficos principal
            self.plot_viewer.add_plot(
                "Estadísticos Extendidos",
                fig,
                f"Estadísticos de {var_name}",
                plot_type='matplotlib'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular estadísticos:\n{str(e)}")
    
    def run_normality_tests(self):
        """Ejecuta tests de normalidad"""
        series, var_name = self.get_current_series()
        if series is None:
            return
        
        try:
            clean_series = series.dropna()
            
            if len(clean_series) < 3:
                messagebox.showwarning("Datos insuficientes", 
                                     "Se necesitan al menos 3 valores")
                return
            
            # Ejecutar tests
            test_results = []
            
            # Shapiro-Wilk
            if 3 <= len(clean_series) <= 5000:
                w, p = stats.shapiro(clean_series)
                test_results.append(['Shapiro-Wilk', f"{w:.6f}", f"{p:.6g}", 
                                   'Datos normales' if p > 0.05 else 'Datos NO normales'])
            
            # D'Agostino K²
            if len(clean_series) >= 20:
                k2, p = stats.normaltest(clean_series)
                test_results.append(['D\'Agostino K²', f"{k2:.6f}", f"{p:.6g}", 
                                   'Datos normales' if p > 0.05 else 'Datos NO normales'])
            
            # Anderson-Darling
            ad_result = stats.anderson(clean_series, dist='norm')
            ad_interpretation = "Datos normales" if ad_result.statistic < ad_result.critical_values[2] else "Datos NO normales"
            test_results.append(['Anderson-Darling', f"{ad_result.statistic:.6f}", 
                               f"Crítico: {ad_result.critical_values[2]:.4f}", ad_interpretation])
            
            # Jarque-Bera
            if len(clean_series) >= 7:
                jb, p = stats.jarque_bera(clean_series)
                test_results.append(['Jarque-Bera', f"{jb:.6f}", f"{p:.6g}", 
                                   'Datos normales' if p > 0.05 else 'Datos NO normales'])
            
            # Crear figura con resultados usando matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            
            # Preparar datos para la tabla
            table_data = [['Test', 'Estadístico', 'p-value / Crítico', 'Interpretación']] + test_results
            
            # Crear tabla
            table = ax.table(
                cellText=table_data,
                colLabels=None,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            # Estilizar tabla
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)
            
            # Colorear encabezados
            for i in range(4):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Colorear filas de resultados
            for i in range(1, len(table_data)):
                for j in range(4):
                    if 'NO normales' in table_data[i][3]:
                        table[(i, j)].set_facecolor('#FFCDD2')  # Rojo claro
                    else:
                        table[(i, j)].set_facecolor('#C8E6C9')  # Verde claro
            
            plt.title(f'Tests de Normalidad - {var_name}', fontsize=16, fontweight='bold', pad=20)
            
            # Añadir nota explicativa
            note_text = "Nota: Un p-value > 0.05 sugiere que los datos son consistentes\ncon una distribución normal al nivel de significancia de 5%."
            plt.figtext(0.5, 0.02, note_text, ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            
            # Mostrar en el visor de gráficos principal
            self.plot_viewer.add_plot(
                "Tests de Normalidad",
                fig,
                f"Tests de normalidad para {var_name}",
                plot_type='matplotlib'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en tests de normalidad:\n{str(e)}")
    
    def detect_outliers(self):
        """Detecta valores atípicos"""
        series, var_name = self.get_current_series()
        if series is None:
            return
        
        try:
            clean_series = series.dropna()
            
            if len(clean_series) < 3:
                messagebox.showwarning("Datos insuficientes", 
                                     "Se necesitan al menos 3 valores")
                return
            
            # Calcular outliers
            xvals = clean_series.values
            
            # Método IQR
            q1, q3 = np.percentile(xvals, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers_iqr = xvals[(xvals < lower) | (xvals > upper)]
            
            # Método Z-score
            z_scores = np.abs(stats.zscore(xvals))
            outliers_z = xvals[z_scores > 3]
            
            # Crear figura con resultados usando matplotlib
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Gráfico 1: Box plot con outliers
            ax1.boxplot(xvals, patch_artist=True, 
                                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                                  medianprops=dict(color='red', linewidth=2),
                                  flierprops=dict(marker='o', markerfacecolor='red', 
                                                markeredgecolor='red', markersize=6))
            ax1.set_title(f'Box Plot - {var_name}', fontsize=12, fontweight='bold')
            ax1.set_ylabel(var_name)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Gráfico 2: Scatter plot con outliers resaltados
            normal_mask = (xvals >= lower) & (xvals <= upper)
            outlier_mask = (xvals < lower) | (xvals > upper)
            
            ax2.scatter(range(len(xvals[normal_mask])), xvals[normal_mask], 
                       c='blue', alpha=0.6, s=30, label='Normal')
            ax2.scatter(range(len(xvals[normal_mask]), len(xvals[normal_mask]) + len(xvals[outlier_mask])), 
                       xvals[outlier_mask], c='red', alpha=0.8, s=50, 
                       marker='x', label='Outliers')
            
            # Líneas de límites IQR
            ax2.axhline(y=lower, color='orange', linestyle='--', alpha=0.7, label=f'Límite inferior ({lower:.2f})')
            ax2.axhline(y=upper, color='orange', linestyle='--', alpha=0.7, label=f'Límite superior ({upper:.2f})')
            
            ax2.set_title(f'Detección de Outliers - {var_name}', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Índice')
            ax2.set_ylabel(var_name)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f'Análisis de Outliers - {var_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Crear tabla de resultados
            fig2, ax3 = plt.subplots(figsize=(12, 8))
            ax3.axis('off')
            
            # Preparar datos para la tabla
            table_data = [
                ['Método', 'Valor', 'Resultado'],
                ['Total observaciones', f"{len(xvals)}", ''],
                ['', '', ''],
                ['MÉTODO IQR (1.5 × IQR)', '', ''],
                ['  Q1', f"{q1:.4f}", ''],
                ['  Q3', f"{q3:.4f}", ''],
                ['  IQR', f"{iqr:.4f}", ''],
                ['  Límite inferior', f"{lower:.4f}", ''],
                ['  Límite superior', f"{upper:.4f}", ''],
                ['  Outliers detectados', f"{len(outliers_iqr)}", f"({len(outliers_iqr)/len(xvals)*100:.1f}%)"],
                ['', '', ''],
                ['MÉTODO Z-SCORE (|z| > 3)', '', ''],
                ['  Outliers detectados', f"{len(outliers_z)}", f"({len(outliers_z)/len(xvals)*100:.1f}%)"]
            ]
            
            # Crear tabla
            table = ax3.table(
                cellText=table_data,
                colLabels=None,
                cellLoc='left',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            # Estilizar tabla
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Colorear encabezados
            for i in range(3):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Colorear métodos principales
            for i, row in enumerate(table_data):
                if row[0] and not row[0].startswith('  ') and row[0] != '' and 'MÉTODO' in row[0]:
                    for j in range(3):
                        table[(i+1, j)].set_facecolor('#E8F5E8')
                        table[(i+1, j)].set_text_props(weight='bold')
            
            plt.title(f'Resumen de Detección de Outliers - {var_name}', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Mostrar gráficos en el visor principal
            self.plot_viewer.add_plot(
                "Outliers - Gráficos",
                fig,
                f"Análisis de outliers para {var_name}",
                plot_type='matplotlib'
            )
            
            self.plot_viewer.add_plot(
                "Outliers - Resumen",
                fig2,
                f"Resumen de outliers para {var_name}",
                plot_type='matplotlib'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en detección de outliers:\n{str(e)}")
    
    def show_transformations(self):
        """Muestra transformaciones de la variable"""
        series, var_name = self.get_current_series()
        if series is None:
            return

        try:
            clean_series = series.dropna()
            xvals = clean_series.values

            if len(xvals) < 3:
                messagebox.showwarning("Datos insuficientes",
                                     "Se necesitan al menos 3 valores")
                return

            # Crear figura con subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

            # Original
            ax1.hist(xvals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.set_title('Original')
            ax1.set_xlabel(var_name)
            ax1.set_ylabel('Frecuencia')
            ax1.grid(True, alpha=0.3)

            # Log
            if np.all(xvals > 0):
                log_vals = np.log(xvals)
                ax2.hist(log_vals, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
                ax2.set_title('Transformación Log')
                ax2.set_xlabel(f'log({var_name})')
                ax2.set_ylabel('Frecuencia')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Requiere valores > 0', ha='center', va='center')
                ax2.set_title('Transformación Log')

            # Box-Cox
            if np.all(xvals > 0):
                try:
                    bc_vals, lambda_val = stats.boxcox(xvals)
                    ax3.hist(bc_vals, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
                    ax3.set_title(f'Box-Cox (λ={lambda_val:.2f})')
                    ax3.set_xlabel(f'boxcox({var_name})')
                    ax3.set_ylabel('Frecuencia')
                    ax3.grid(True, alpha=0.3)
                except:
                    ax3.text(0.5, 0.5, 'Error en Box-Cox', ha='center', va='center')
                    ax3.set_title('Box-Cox')
            else:
                ax3.text(0.5, 0.5, 'Requiere valores > 0', ha='center', va='center')
                ax3.set_title('Box-Cox')

            # Sqrt
            if np.all(xvals >= 0):
                sqrt_vals = np.sqrt(xvals)
                ax4.hist(sqrt_vals, bins=30, edgecolor='black', alpha=0.7, color='lightyellow')
                ax4.set_title('Transformación Sqrt')
                ax4.set_xlabel(f'sqrt({var_name})')
                ax4.set_ylabel('Frecuencia')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Requiere valores >= 0', ha='center', va='center')
                ax4.set_title('Transformación Sqrt')

            fig.suptitle(f'Transformaciones - {var_name}', fontsize=14)
            plt.tight_layout()

            self.plot_viewer.add_plot("Transformaciones", fig, var_name, plot_type='matplotlib')

        except Exception as e:
            messagebox.showerror("Error", f"Error al generar transformaciones:\n{str(e)}")
    
    def plot_frequencies(self):
        """Grafica las frecuencias para variable categórica"""
        series, var_name = self.get_current_series()
        if series is None:
            return

        try:
            clean_series = series.dropna()

            # Crear gráfico de barras
            fig = PlotBuilder.create_bar_plot(
                clean_series,
                title=f"Frecuencias - {var_name}",
                top_n=20
            )

            self.plot_viewer.add_plot("Frecuencias", fig, var_name, plot_type='matplotlib')

        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráfico:\n{str(e)}")
    
    def show_categorical_stats(self):
        """Muestra estadísticas para variable categórica"""
        series, var_name = self.get_current_series()
        if series is None:
            return
        
        try:
            clean_series = series.dropna()
            value_counts = clean_series.value_counts()
            
            # Crear figura con tabla de estadísticas categóricas
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.axis('tight')
            ax.axis('off')
            
            # Preparar datos para la tabla
            table_data = [
                ['Categoría', 'Frecuencia', 'Porcentaje (%)']
            ]
            
            total = value_counts.sum()
            for cat, freq in value_counts.items():
                pct = (freq / total) * 100
                table_data.append([str(cat), f"{freq}", f"{pct:.2f}"])
            
            # Agregar fila de total
            table_data.append(['', '', ''])
            table_data.append(['TOTAL', f"{total}", '100.00'])
            
            # Crear tabla
            table = ax.table(
                cellText=table_data,
                colLabels=None,
                cellLoc='left',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            # Estilizar tabla
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Colorear encabezados
            for j in range(3):
                table[(0, j)].set_facecolor('#4CAF50')
                table[(0, j)].set_text_props(weight='bold', color='white')
            
            # Colorear filas alternadas
            for i in range(1, len(table_data) - 2):
                for j in range(3):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#F5F5F5')
                    else:
                        table[(i, j)].set_facecolor('#FFFFFF')
            
            # Colorear fila de total
            for j in range(3):
                table[(len(table_data) - 1, j)].set_facecolor('#FFF3E0')
                table[(len(table_data) - 1, j)].set_text_props(weight='bold')
            
            plt.title(f'Estadísticas Categóricas - {var_name}', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Mostrar en el visor de gráficos principal
            self.plot_viewer.add_plot(
                "Estadísticas Categóricas",
                fig,
                f"Estadísticas categóricas: {var_name}",
                plot_type='matplotlib'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar estadísticas:\n{str(e)}")
