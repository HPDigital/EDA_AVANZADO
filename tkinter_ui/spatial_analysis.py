"""
Frame de análisis espacial
"""
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from .plot_viewer import MultiPlotViewer


class SpatialAnalysisFrame(ttk.Frame):
    """Frame para análisis espacial"""
    
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.current_df = None
        self.create_widgets()
    
    def create_widgets(self):
        left_panel = ttk.Frame(self, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        ttk.Label(left_panel, text="Análisis Espacial",
                 font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        # Información
        info_label = ttk.Label(left_panel, 
                              text="Análisis geográfico y espacial de datos",
                              wraplength=250)
        info_label.pack(pady=10)
        
        var_frame = ttk.LabelFrame(left_panel, text="Variable", padding=10)
        var_frame.pack(fill=tk.X, pady=5)
        
        self.var_combo = ttk.Combobox(var_frame, state='readonly')
        self.var_combo.pack(fill=tk.X)
        
        ttk.Button(var_frame, text="📊 Distribución espacial",
                  command=self.plot_spatial).pack(fill=tk.X, pady=10)
        
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.plot_viewer = MultiPlotViewer(right_panel)
        self.plot_viewer.pack(fill=tk.BOTH, expand=True)
    
    def update_data(self, df, data_results):
        self.current_df = df
        if df is not None:
            data_columns = [col for col in df.columns if not col.startswith('_')]
            numeric_cols = [col for col in data_columns 
                          if pd.api.types.is_numeric_dtype(df[col])]
            self.var_combo['values'] = numeric_cols
            if numeric_cols:
                self.var_combo.current(0)
    
    def plot_spatial(self):
        var_name = self.var_combo.get()
        if not var_name or self.current_df is None:
            messagebox.showwarning("Sin selección", "Seleccione una variable")
            return

        try:
            # Buscar columna espacial/geográfica
            spatial_col = None
            for col in self.current_df.columns:
                if any(keyword in col.lower() for keyword in ['depart', 'region', 'ciudad', 'city', 'zone']):
                    spatial_col = col
                    break

            if spatial_col:
                df_plot = self.current_df[[spatial_col, var_name]].dropna()
                df_grouped = df_plot.groupby(spatial_col)[var_name].mean().sort_values(ascending=False).head(20)

                # Crear gráfico con Matplotlib
                fig, ax = plt.subplots(figsize=(12, 6))

                ax.bar(range(len(df_grouped)), df_grouped.values, color='steelblue', edgecolor='black')
                ax.set_xticks(range(len(df_grouped)))
                ax.set_xticklabels(df_grouped.index, rotation=45, ha='right')
                ax.set_xlabel(spatial_col, fontsize=12)
                ax.set_ylabel(var_name, fontsize=12)
                ax.set_title(f'Distribución Espacial - {var_name}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

                plt.tight_layout()

                self.plot_viewer.add_plot("Espacial", fig, var_name, plot_type='matplotlib')
            else:
                messagebox.showinfo("Sin datos espaciales",
                                  "No se encontró una columna geográfica")
        except Exception as e:
            messagebox.showerror("Error", str(e))

