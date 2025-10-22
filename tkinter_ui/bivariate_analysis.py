"""
Frame de análisis bivariado
"""
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from .plot_viewer import MultiPlotViewer
from utils.plot_utils import PlotBuilder


class BivariateAnalysisFrame(ttk.Frame):
    """Frame para análisis bivariado"""
    
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.current_df = None
        self.current_data = None
        
        self.create_widgets()
    
    def create_widgets(self):
        """Crea los widgets del frame"""
        # Panel izquierdo
        left_panel = ttk.Frame(self, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        ttk.Label(left_panel, text="Análisis Bivariado",
                 font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        # Tipo de análisis
        type_frame = ttk.LabelFrame(left_panel, text="Tipo de Análisis", padding=10)
        type_frame.pack(fill=tk.X, pady=5)
        
        self.analysis_type = tk.StringVar(value="numeric_numeric")
        ttk.Radiobutton(type_frame, text="Numérico-Numérico",
                       variable=self.analysis_type, value="numeric_numeric",
                       command=self.on_type_changed).pack(anchor=tk.W)
        ttk.Radiobutton(type_frame, text="Numérico-Categórico",
                       variable=self.analysis_type, value="numeric_categorical",
                       command=self.on_type_changed).pack(anchor=tk.W)
        ttk.Radiobutton(type_frame, text="Categórico-Categórico",
                       variable=self.analysis_type, value="categorical_categorical",
                       command=self.on_type_changed).pack(anchor=tk.W)
        
        # Selección de variables
        var_frame = ttk.LabelFrame(left_panel, text="Variables", padding=10)
        var_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(var_frame, text="Variable X:").pack(anchor=tk.W)
        self.var_x_combo = ttk.Combobox(var_frame, state='readonly')
        self.var_x_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(var_frame, text="Variable Y:").pack(anchor=tk.W, pady=(10,0))
        self.var_y_combo = ttk.Combobox(var_frame, state='readonly')
        self.var_y_combo.pack(fill=tk.X, pady=2)
        
        # Análisis disponibles
        analysis_frame = ttk.LabelFrame(left_panel, text="Análisis", padding=10)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        # Crear botones de análisis (inicialmente ocultos)
        self.analysis_buttons = {}
        
        # Botones para Numérico-Numérico
        self.analysis_buttons['scatter'] = ttk.Button(analysis_frame, text="📊 Gráfico de dispersión",
                  command=self.plot_scatter)
        self.analysis_buttons['correlations'] = ttk.Button(analysis_frame, text="📈 Correlaciones",
                  command=self.show_correlations)
        self.analysis_buttons['regression'] = ttk.Button(analysis_frame, text="📉 Regresión lineal",
                  command=self.show_regression)
        
        # Botones para Numérico-Categórico
        self.analysis_buttons['boxplot'] = ttk.Button(analysis_frame, text="📦 Gráfico de cajas",
                  command=self.plot_boxplot)
        self.analysis_buttons['violin'] = ttk.Button(analysis_frame, text="🎻 Gráfico de violín",
                  command=self.plot_violin)
        
        # Botones para Categórico-Categórico
        self.analysis_buttons['heatmap'] = ttk.Button(analysis_frame, text="🔥 Mapa de calor",
                  command=self.plot_heatmap)
        self.analysis_buttons['stacked'] = ttk.Button(analysis_frame, text="📊 Gráfico apilado",
                  command=self.plot_stacked)
        
        # Botón de tests estadísticos (disponible para todos)
        self.analysis_buttons['tests'] = ttk.Button(analysis_frame, text="🔬 Tests estadísticos",
                  command=self.run_statistical_tests)
        
        # Inicializar con botones para Numérico-Numérico
        self.update_analysis_buttons()
        
        # Panel derecho
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.plot_viewer = MultiPlotViewer(right_panel)
        self.plot_viewer.pack(fill=tk.BOTH, expand=True)
    
    def update_data(self, df, data_results):
        """Actualiza con nuevos datos"""
        self.current_df = df
        self.current_data = data_results
        
        if df is not None:
            self.update_variable_lists()
    
    def on_type_changed(self):
        """Callback cuando cambia el tipo"""
        self.update_variable_lists()
        self.update_analysis_buttons()
    
    def update_variable_lists(self):
        """Actualiza las listas de variables"""
        if self.current_df is None:
            return
        
        data_columns = [col for col in self.current_df.columns if not col.startswith('_')]
        numeric_cols = [col for col in data_columns 
                       if pd.api.types.is_numeric_dtype(self.current_df[col])]
        categorical_cols = [col for col in data_columns 
                          if not pd.api.types.is_numeric_dtype(self.current_df[col])]
        
        atype = self.analysis_type.get()
        
        if atype == "numeric_numeric":
            self.var_x_combo['values'] = numeric_cols
            self.var_y_combo['values'] = numeric_cols
        elif atype == "numeric_categorical":
            self.var_x_combo['values'] = numeric_cols
            self.var_y_combo['values'] = categorical_cols
        else:  # categorical_categorical
            self.var_x_combo['values'] = categorical_cols
            self.var_y_combo['values'] = categorical_cols
        
        if len(self.var_x_combo['values']) > 0:
            self.var_x_combo.current(0)
        if len(self.var_y_combo['values']) > 1:
            self.var_y_combo.current(1)
    
    def get_selected_variables(self):
        """Obtiene las variables seleccionadas"""
        var_x = self.var_x_combo.get()
        var_y = self.var_y_combo.get()
        
        if not var_x or not var_y:
            messagebox.showwarning("Sin selección", "Seleccione ambas variables")
            return None, None, None, None
        
        if var_x == var_y:
            messagebox.showwarning("Variables iguales", "Seleccione variables diferentes")
            return None, None, None, None
        
        return (self.current_df[var_x], self.current_df[var_y], var_x, var_y)
    
    def plot_scatter(self):
        """Grafica diagrama de dispersión"""
        x, y, var_x, var_y = self.get_selected_variables()
        if x is None:
            return

        try:
            df_plot = pd.DataFrame({var_x: x, var_y: y}).dropna()

            if len(df_plot) < 2:
                messagebox.showwarning("Datos insuficientes",
                                     "Se necesitan al menos 2 puntos válidos")
                return

            if self.analysis_type.get() == "numeric_numeric":
                # Scatter plot con línea de tendencia usando Matplotlib
                fig = PlotBuilder.create_regression_plot(
                    x=df_plot[var_x],
                    y=df_plot[var_y],
                    title=f"{var_x} vs {var_y}"
                )
            else:
                # Boxplot por categoría usando Matplotlib
                fig = PlotBuilder.create_box_plot_by_category(
                    data=df_plot,
                    numeric_col=var_x,
                    category_col=var_y,
                    title=f"{var_x} por {var_y}"
                )

            self.plot_viewer.add_plot("Dispersión", fig, f"{var_x} vs {var_y}", plot_type='matplotlib')

        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráfico:\n{str(e)}")
    
    def update_analysis_buttons(self):
        """Actualiza los botones de análisis según el tipo seleccionado"""
        # Ocultar todos los botones primero
        for button in self.analysis_buttons.values():
            button.pack_forget()
        
        # Mostrar botones según el tipo de análisis
        analysis_type = self.analysis_type.get()
        
        if analysis_type == "numeric_numeric":
            # Numérico-Numérico: dispersión, correlaciones, regresión
            self.analysis_buttons['scatter'].pack(fill=tk.X, pady=2)
            self.analysis_buttons['correlations'].pack(fill=tk.X, pady=2)
            self.analysis_buttons['regression'].pack(fill=tk.X, pady=2)
            self.analysis_buttons['tests'].pack(fill=tk.X, pady=2)
            
        elif analysis_type == "numeric_categorical":
            # Numérico-Categórico: boxplot, violín
            self.analysis_buttons['boxplot'].pack(fill=tk.X, pady=2)
            self.analysis_buttons['violin'].pack(fill=tk.X, pady=2)
            self.analysis_buttons['tests'].pack(fill=tk.X, pady=2)
            
        elif analysis_type == "categorical_categorical":
            # Categórico-Categórico: mapa de calor, gráfico apilado
            self.analysis_buttons['heatmap'].pack(fill=tk.X, pady=2)
            self.analysis_buttons['stacked'].pack(fill=tk.X, pady=2)
            self.analysis_buttons['tests'].pack(fill=tk.X, pady=2)
    
    def plot_boxplot(self):
        """Gráfico de cajas para numérico-categórico"""
        x, y, var_x, var_y = self.get_selected_variables()
        if x is None:
            return
        
        try:
            # Crear DataFrame limpio
            df_clean = pd.DataFrame({var_x: x, var_y: y}).dropna()
            
            # Crear gráfico de cajas
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Obtener categorías únicas
            categories = df_clean[var_y].unique()
            
            # Preparar datos para boxplot
            data_for_box = []
            labels = []
            
            for cat in categories:
                cat_data = df_clean[df_clean[var_y] == cat][var_x].dropna()
                if len(cat_data) > 0:
                    data_for_box.append(cat_data)
                    labels.append(str(cat))
            
            if data_for_box:
                box_plot = ax.boxplot(data_for_box, labels=labels, patch_artist=True)
                
                # Colorear las cajas
                colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_box)))
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_xlabel(var_y, fontsize=12)
                ax.set_ylabel(var_x, fontsize=12)
                ax.set_title(f'Gráfico de Cajas: {var_x} por {var_y}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                self.plot_viewer.add_plot("Boxplot", fig, f"Boxplot: {var_x} por {var_y}", plot_type='matplotlib')
            else:
                messagebox.showwarning("Datos insuficientes", "No hay datos válidos para generar el gráfico")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráfico:\n{str(e)}")
    
    def plot_violin(self):
        """Gráfico de violín para numérico-categórico"""
        x, y, var_x, var_y = self.get_selected_variables()
        if x is None:
            return
        
        try:
            # Crear DataFrame limpio
            df_clean = pd.DataFrame({var_x: x, var_y: y}).dropna()
            
            # Crear gráfico de violín
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Obtener categorías únicas
            categories = df_clean[var_y].unique()
            
            # Preparar datos para violin plot
            data_for_violin = []
            labels = []
            
            for cat in categories:
                cat_data = df_clean[df_clean[var_y] == cat][var_x].dropna()
                if len(cat_data) > 0:
                    data_for_violin.append(cat_data)
                    labels.append(str(cat))
            
            if data_for_violin:
                parts = ax.violinplot(data_for_violin, positions=range(len(labels)), showmeans=True, showmedians=True)
                
                # Colorear los violines
                colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_violin)))
                for pc, color in zip(parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45)
                ax.set_xlabel(var_y, fontsize=12)
                ax.set_ylabel(var_x, fontsize=12)
                ax.set_title(f'Gráfico de Violín: {var_x} por {var_y}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                
                self.plot_viewer.add_plot("Violin", fig, f"Violin: {var_x} por {var_y}", plot_type='matplotlib')
            else:
                messagebox.showwarning("Datos insuficientes", "No hay datos válidos para generar el gráfico")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráfico:\n{str(e)}")
    
    def plot_heatmap(self):
        """Mapa de calor para categórico-categórico"""
        x, y, var_x, var_y = self.get_selected_variables()
        if x is None:
            return
        
        try:
            # Crear tabla de contingencia
            df_clean = pd.DataFrame({var_x: x, var_y: y}).dropna()
            contingency_table = pd.crosstab(df_clean[var_x], df_clean[var_y])
            
            # Crear mapa de calor
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(contingency_table.values, cmap='YlOrRd', aspect='auto')
            
            # Configurar ejes
            ax.set_xticks(range(len(contingency_table.columns)))
            ax.set_yticks(range(len(contingency_table.index)))
            ax.set_xticklabels(contingency_table.columns, rotation=45, ha='right')
            ax.set_yticklabels(contingency_table.index)
            
            # Agregar valores en las celdas
            for i in range(len(contingency_table.index)):
                for j in range(len(contingency_table.columns)):
                    ax.text(j, i, int(contingency_table.iloc[i, j]),
                            ha="center", va="center", color="black", fontweight='bold')
            
            ax.set_xlabel(var_y, fontsize=12)
            ax.set_ylabel(var_x, fontsize=12)
            ax.set_title(f'Mapa de Calor: {var_x} vs {var_y}', fontsize=14, fontweight='bold')
            
            # Agregar barra de color
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Frecuencia', rotation=270, labelpad=20)
            
            plt.tight_layout()
            
            self.plot_viewer.add_plot("Heatmap", fig, f"Heatmap: {var_x} vs {var_y}", plot_type='matplotlib')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráfico:\n{str(e)}")
    
    def plot_stacked(self):
        """Gráfico apilado para categórico-categórico"""
        x, y, var_x, var_y = self.get_selected_variables()
        if x is None:
            return
        
        try:
            # Crear tabla de contingencia
            df_clean = pd.DataFrame({var_x: x, var_y: y}).dropna()
            contingency_table = pd.crosstab(df_clean[var_x], df_clean[var_y])
            
            # Convertir a porcentajes
            contingency_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
            
            # Crear gráfico apilado
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Crear gráfico de barras apiladas
            bottom = np.zeros(len(contingency_pct.index))
            colors = plt.cm.Set3(np.linspace(0, 1, len(contingency_pct.columns)))
            
            for i, col in enumerate(contingency_pct.columns):
                ax.bar(contingency_pct.index, contingency_pct[col], bottom=bottom, 
                      label=str(col), color=colors[i], alpha=0.8)
                bottom += contingency_pct[col]
            
            ax.set_xlabel(var_x, fontsize=12)
            ax.set_ylabel('Porcentaje (%)', fontsize=12)
            ax.set_title(f'Gráfico Apilado: {var_x} vs {var_y}', fontsize=14, fontweight='bold')
            ax.legend(title=var_y, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            self.plot_viewer.add_plot("Stacked", fig, f"Stacked: {var_x} vs {var_y}", plot_type='matplotlib')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráfico:\n{str(e)}")
    
    def show_correlations(self):
        """Muestra correlaciones"""
        x, y, var_x, var_y = self.get_selected_variables()
        if x is None:
            return
        
        if self.analysis_type.get() != "numeric_numeric":
            messagebox.showinfo("No disponible", 
                              "Correlaciones solo para variables numéricas")
            return
        
        try:
            x_clean = pd.to_numeric(x, errors='coerce')
            y_clean = pd.to_numeric(y, errors='coerce')
            
            df_clean = pd.DataFrame({var_x: x_clean, var_y: y_clean}).dropna()
            
            if len(df_clean) < 3:
                messagebox.showwarning("Datos insuficientes", 
                                     "Se necesitan al menos 3 puntos")
                return
            
            x_vals = df_clean[var_x].values
            y_vals = df_clean[var_y].values
            
            # Calcular correlaciones
            r_p, p_p = stats.pearsonr(x_vals, y_vals)
            r_s, p_s = stats.spearmanr(x_vals, y_vals)
            r_k, p_k = stats.kendalltau(x_vals, y_vals)
            
            # Intervalo de confianza para Pearson
            n = len(x_vals)
            z = np.arctanh(r_p)
            se = 1.0 / np.sqrt(n - 3)
            zc = stats.norm.ppf(0.975)
            lo_ci = np.tanh(z - zc * se)
            hi_ci = np.tanh(z + zc * se)
            
            # Crear figura con tabla de correlaciones usando matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # Preparar datos para la tabla
            table_data = [
                ['Método', 'Coeficiente', 'p-value', 'IC 95%', 'Interpretación'],
                ['Pearson', f"{r_p:.4f}", f"{p_p:.6g}", f"[{lo_ci:.4f}, {hi_ci:.4f}]", 
                 'Significativa' if p_p < 0.05 else 'No significativa'],
                ['Spearman', f"{r_s:.4f}", f"{p_s:.6g}", 'N/A', 
                 'Significativa' if p_s < 0.05 else 'No significativa'],
                ['Kendall', f"{r_k:.4f}", f"{p_k:.6g}", 'N/A', 
                 'Significativa' if p_k < 0.05 else 'No significativa'],
                ['', '', '', '', ''],
                ['N observaciones', f"{n}", '', '', '']
            ]
            
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
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Colorear encabezados
            for i in range(5):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Colorear filas de métodos
            for i in range(1, 4):
                for j in range(5):
                    table[(i, j)].set_facecolor('#E8F5E8')
                    if j == 4:  # Columna de interpretación
                        color = '#2E7D32' if 'Significativa' in table_data[i][j] else '#D32F2F'
                        table[(i, j)].set_text_props(color=color, weight='bold')
            
            # Colorear fila de observaciones
            for j in range(5):
                table[(5, j)].set_facecolor('#FFF3E0')
                table[(5, j)].set_text_props(weight='bold')
            
            plt.title(f'Análisis de Correlación: {var_x} vs {var_y}', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Mostrar en el visor de gráficos principal
            self.plot_viewer.add_plot(
                "Correlaciones",
                fig,
                f"Correlaciones: {var_x} vs {var_y}",
                plot_type='matplotlib'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular correlaciones:\n{str(e)}")
    
    def show_regression(self):
        """Muestra regresión lineal"""
        x, y, var_x, var_y = self.get_selected_variables()
        if x is None:
            return

        if self.analysis_type.get() != "numeric_numeric":
            messagebox.showinfo("No disponible",
                              "Regresión solo para variables numéricas")
            return

        try:
            x_clean = pd.to_numeric(x, errors='coerce')
            y_clean = pd.to_numeric(y, errors='coerce')

            df_clean = pd.DataFrame({var_x: x_clean, var_y: y_clean}).dropna()

            if len(df_clean) < 3:
                messagebox.showwarning("Datos insuficientes",
                                     "Se necesitan al menos 3 puntos")
                return

            x_vals = df_clean[var_x].values
            y_vals = df_clean[var_y].values

            # Regresión
            lr = stats.linregress(x_vals, y_vals)

            # Predicciones y residuos
            y_pred = lr.intercept + lr.slope * x_vals
            residuals = y_vals - y_pred

            # Crear gráficos con Matplotlib
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Scatter con línea de regresión
            ax1.scatter(x_vals, y_vals, alpha=0.6, edgecolors='k', s=50, label='Datos')
            ax1.plot(x_vals, y_pred, 'r-', linewidth=2, label='Regresión')
            ax1.set_xlabel(var_x, fontsize=12)
            ax1.set_ylabel(var_y, fontsize=12)
            ax1.set_title("Ajuste del modelo", fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Gráfico de residuos
            ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=50, color='coral')
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel("Valores predichos", fontsize=12)
            ax2.set_ylabel("Residuos", fontsize=12)
            ax2.set_title("Residuos vs Predichos", fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            plt.suptitle(f"Regresión Lineal: {var_y} ~ {var_x}", fontsize=16, fontweight='bold')
            plt.tight_layout()

            self.plot_viewer.add_plot("Regresión", fig, "Análisis de regresión", plot_type='matplotlib')
            
            # Crear figura con tabla de estadísticas de regresión
            fig2, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # Preparar datos para la tabla
            table_data = [
                ['Parámetro', 'Valor', 'Interpretación'],
                ['', '', ''],
                ['ECUACIÓN AJUSTADA', '', ''],
                [f'{var_y} = β₀ + β₁ × {var_x}', '', ''],
                [f'{var_y} = {lr.intercept:.4f} + {lr.slope:.4f} × {var_x}', '', ''],
                ['', '', ''],
                ['ESTADÍSTICAS DEL MODELO', '', ''],
                ['R² (coef. determinación)', f"{lr.rvalue**2:.4f}", f"Explica {lr.rvalue**2*100:.1f}% de la varianza"],
                ['R (correlación)', f"{lr.rvalue:.4f}", ''],
                ['Pendiente (β₁)', f"{lr.slope:.4f}", ''],
                ['Intercepto (β₀)', f"{lr.intercept:.4f}", ''],
                ['p-value', f"{lr.pvalue:.6g}", 'Significativo' if lr.pvalue < 0.05 else 'No significativo'],
                ['Error estándar', f"{lr.stderr:.4f}", ''],
                ['', '', ''],
                ['N observaciones', f"{len(x_vals)}", ''],
                ['', '', ''],
                ['CONCLUSIÓN', '', ''],
                ['Modelo', '', f"{'ES' if lr.pvalue < 0.05 else 'NO ES'} significativo (α=0.05)"]
            ]
            
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
            
            # Colorear secciones principales
            for i in [2, 6, 15]:  # Filas de títulos de sección (índices 2, 6, 15)
                for j in range(3):
                    table[(i, j)].set_facecolor('#2196F3')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            
            # Colorear ecuación
            for i in [3, 4]:
                for j in range(3):
                    table[(i, j)].set_facecolor('#E3F2FD')
                    table[(i, j)].set_text_props(weight='bold')
            
            # Colorear estadísticas
            for i in range(7, 13):
                for j in range(3):
                    table[(i, j)].set_facecolor('#E8F5E8')
            
            # Colorear conclusión
            for j in range(3):
                table[(17, j)].set_facecolor('#FFF3E0')
                table[(17, j)].set_text_props(weight='bold')
                if 'ES' in table_data[17][2] and 'NO ES' not in table_data[17][2]:
                    table[(17, 2)].set_text_props(color='#2E7D32', weight='bold')
                else:
                    table[(17, 2)].set_text_props(color='#D32F2F', weight='bold')
            
            plt.title(f'Estadísticas de Regresión Lineal: {var_y} ~ {var_x}', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Mostrar tabla en el visor de gráficos principal
            self.plot_viewer.add_plot(
                "Regresión - Estadísticas",
                fig2,
                f"Estadísticas de regresión: {var_y} ~ {var_x}",
                plot_type='matplotlib'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en regresión:\n{str(e)}")
    
    def run_statistical_tests(self):
        """Ejecuta tests estadísticos"""
        x, y, var_x, var_y = self.get_selected_variables()
        if x is None:
            return
        
        try:
            if self.analysis_type.get() == "numeric_numeric":
                self.run_correlation_tests(x, y, var_x, var_y)
            elif self.analysis_type.get() == "numeric_categorical":
                self.run_anova_test(x, y, var_x, var_y)
            elif self.analysis_type.get() == "categorical_categorical":
                self.run_chi_square_test(x, y, var_x, var_y)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error en tests:\n{str(e)}")
    
    def run_correlation_tests(self, x_numeric, y_numeric, var_x, var_y):
        """Ejecuta tests de correlación para variables numérico-numérico"""
        x_clean = pd.to_numeric(x_numeric, errors='coerce')
        y_clean = pd.to_numeric(y_numeric, errors='coerce')
        
        df_clean = pd.DataFrame({var_x: x_clean, var_y: y_clean}).dropna()
        
        if len(df_clean) < 3:
            messagebox.showwarning("Datos insuficientes", 
                                 "Se necesitan al menos 3 puntos")
            return
        
        x_vals = df_clean[var_x].values
        y_vals = df_clean[var_y].values
        
        # Calcular correlaciones
        r_p, p_p = stats.pearsonr(x_vals, y_vals)
        r_s, p_s = stats.spearmanr(x_vals, y_vals)
        r_k, p_k = stats.kendalltau(x_vals, y_vals)
        
        # Crear figura con tabla de resultados de correlación
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Preparar datos para la tabla
        table_data = [
            ['Método', 'Coeficiente', 'p-value', 'Interpretación'],
            ['Pearson', f"{r_p:.4f}", f"{p_p:.6g}", 
             'Significativa' if p_p < 0.05 else 'No significativa'],
            ['Spearman', f"{r_s:.4f}", f"{p_s:.6g}", 
             'Significativa' if p_s < 0.05 else 'No significativa'],
            ['Kendall', f"{r_k:.4f}", f"{p_k:.6g}", 
             'Significativa' if p_k < 0.05 else 'No significativa'],
            ['', '', '', ''],
            ['N observaciones', f"{len(x_vals)}", '', '']
        ]
        
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
        
        # Colorear filas de métodos
        for i in range(1, 4):
            for j in range(4):
                if 'Significativa' in table_data[i][3]:
                    table[(i, j)].set_facecolor('#C8E6C9')  # Verde claro
                else:
                    table[(i, j)].set_facecolor('#FFCDD2')  # Rojo claro
        
        # Colorear fila de observaciones
        for j in range(4):
            table[(5, j)].set_facecolor('#FFF3E0')
            table[(5, j)].set_text_props(weight='bold')
        
        plt.title(f'Tests de Correlación: {var_x} vs {var_y}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Mostrar en el visor de gráficos principal
        self.plot_viewer.add_plot(
            "Tests Correlación",
            fig,
            f"Tests de correlación: {var_x} vs {var_y}",
            plot_type='matplotlib'
        )
    
    def run_anova_test(self, x_numeric, y_categorical, var_x, var_y):
        """Ejecuta ANOVA"""
        df_clean = pd.DataFrame({var_x: x_numeric, var_y: y_categorical}).dropna()
        
        groups = [group[var_x].values for name, group in df_clean.groupby(var_y)]
        
        if len(groups) < 2:
            messagebox.showwarning("Datos insuficientes", 
                                 "Se necesitan al menos 2 grupos")
            return
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Crear figura con tabla de resultados ANOVA
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Preparar datos para la tabla
        table_data = [
            ['Concepto', 'Descripción', 'Valor'],
            ['', '', ''],
            ['HIPÓTESIS', '', ''],
            ['H₀ (nula)', 'Las medias de todos los grupos son iguales', ''],
            ['H₁ (alternativa)', 'Al menos una media es diferente', ''],
            ['', '', ''],
            ['RESULTADOS', '', ''],
            ['F-estadístico', '', f"{f_stat:.4f}"],
            ['p-value', '', f"{p_value:.6g}"],
            ['Decisión (α=0.05)', '', 'Rechazar H₀' if p_value < 0.05 else 'No rechazar H₀'],
            ['', '', ''],
            ['DATOS', '', ''],
            ['Número de grupos', '', f"{len(groups)}"],
            ['N total', '', f"{len(df_clean)}"],
            ['', '', ''],
            ['CONCLUSIÓN', '', ''],
            ['Interpretación', '', 'Hay diferencias significativas entre los grupos' if p_value < 0.05 else 'No hay evidencia de diferencias entre los grupos']
        ]
        
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
        
        # Colorear secciones principales (títulos)
        section_rows = [2, 6, 11, 15]  # Filas de títulos de sección
        for row_idx in section_rows:
            if row_idx < len(table_data):
                for j in range(3):
                    table[(row_idx, j)].set_facecolor('#2196F3')
                    table[(row_idx, j)].set_text_props(weight='bold', color='white')
        
        # Colorear hipótesis (filas 3, 4)
        for row_idx in [3, 4]:
            if row_idx < len(table_data):
                for j in range(3):
                    table[(row_idx, j)].set_facecolor('#E3F2FD')
        
        # Colorear resultados (filas 7, 8, 9)
        for row_idx in range(7, 10):
            if row_idx < len(table_data):
                for j in range(3):
                    table[(row_idx, j)].set_facecolor('#E8F5E8')
        
        # Colorear decisión (fila 9)
        if 9 < len(table_data):
            if p_value < 0.05:
                table[(9, 2)].set_text_props(color='#2E7D32', weight='bold')
            else:
                table[(9, 2)].set_text_props(color='#D32F2F', weight='bold')
        
        # Colorear datos (filas 12, 13)
        for row_idx in [12, 13]:
            if row_idx < len(table_data):
                for j in range(3):
                    table[(row_idx, j)].set_facecolor('#FFF3E0')
        
        # Colorear conclusión (fila 16)
        if 16 < len(table_data):
            for j in range(3):
                table[(16, j)].set_facecolor('#FFFDE7')
                table[(16, j)].set_text_props(weight='bold')
        
        plt.title(f'Análisis de Varianza (ANOVA): {var_x} por {var_y}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Mostrar en el visor de gráficos principal
        self.plot_viewer.add_plot(
            "ANOVA",
            fig,
            f"Test ANOVA: {var_x} por {var_y}",
            plot_type='matplotlib'
        )
    
    def run_chi_square_test(self, x_cat, y_cat, var_x, var_y):
        """Ejecuta Chi-cuadrado"""
        df_clean = pd.DataFrame({var_x: x_cat, var_y: y_cat}).dropna()
        
        contingency_table = pd.crosstab(df_clean[var_x], df_clean[var_y])
        
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Crear figura con tabla de resultados Chi-cuadrado
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Preparar datos para la tabla
        table_data = [
            ['Concepto', 'Descripción', 'Valor'],
            ['', '', ''],
            ['HIPÓTESIS', '', ''],
            ['H₀ (nula)', 'Las variables son independientes', ''],
            ['H₁ (alternativa)', 'Las variables están asociadas', ''],
            ['', '', ''],
            ['RESULTADOS', '', ''],
            ['Chi²', '', f"{chi2:.4f}"],
            ['p-value', '', f"{p:.6g}"],
            ['Grados de libertad', '', f"{dof}"],
            ['Decisión (α=0.05)', '', 'Rechazar H₀' if p < 0.05 else 'No rechazar H₀'],
            ['', '', ''],
            ['DATOS', '', ''],
            ['Dimensiones tabla', '', f"{contingency_table.shape[0]} × {contingency_table.shape[1]}"],
            ['', '', ''],
            ['CONCLUSIÓN', '', ''],
            ['Interpretación', '', 'Las variables ESTÁN asociadas' if p < 0.05 else 'No hay evidencia de asociación']
        ]
        
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
        
        # Colorear secciones principales (títulos)
        section_rows = [2, 6, 12, 15]  # Filas de títulos de sección
        for row_idx in section_rows:
            if row_idx < len(table_data):
                for j in range(3):
                    table[(row_idx, j)].set_facecolor('#2196F3')
                    table[(row_idx, j)].set_text_props(weight='bold', color='white')
        
        # Colorear hipótesis (filas 3, 4)
        for row_idx in [3, 4]:
            if row_idx < len(table_data):
                for j in range(3):
                    table[(row_idx, j)].set_facecolor('#E3F2FD')
        
        # Colorear resultados (filas 7, 8, 9, 10)
        for row_idx in range(7, 11):
            if row_idx < len(table_data):
                for j in range(3):
                    table[(row_idx, j)].set_facecolor('#E8F5E8')
        
        # Colorear decisión (fila 10)
        if 10 < len(table_data):
            if p < 0.05:
                table[(10, 2)].set_text_props(color='#2E7D32', weight='bold')
            else:
                table[(10, 2)].set_text_props(color='#D32F2F', weight='bold')
        
        # Colorear datos (fila 13)
        if 13 < len(table_data):
            for j in range(3):
                table[(13, j)].set_facecolor('#FFF3E0')
        
        # Colorear conclusión (fila 16)
        if 16 < len(table_data):
            for j in range(3):
                table[(16, j)].set_facecolor('#FFFDE7')
                table[(16, j)].set_text_props(weight='bold')
                if j == 2:  # Solo la columna de interpretación
                    if 'ESTÁN' in table_data[16][2]:
                        table[(16, 2)].set_text_props(color='#2E7D32', weight='bold')
                    else:
                        table[(16, 2)].set_text_props(color='#D32F2F', weight='bold')
        
        plt.title(f'Test Chi-cuadrado de Independencia: {var_x} vs {var_y}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Mostrar en el visor de gráficos principal
        self.plot_viewer.add_plot(
            "Chi-cuadrado",
            fig,
            f"Test Chi-cuadrado: {var_x} vs {var_y}",
            plot_type='matplotlib'
        )
