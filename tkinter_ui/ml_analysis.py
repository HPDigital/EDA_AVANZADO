"""
Frame de machine learning
"""
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from .plot_viewer import MultiPlotViewer


class MachineLearningFrame(ttk.Frame):
    """Frame para machine learning"""
    
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.current_df = None
        self.create_widgets()
    
    def create_widgets(self):
        left_panel = ttk.Frame(self, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        ttk.Label(left_panel, text="Machine Learning",
                 font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        # Variables
        var_frame = ttk.LabelFrame(left_panel, text="Configuración", padding=10)
        var_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(var_frame, text="Variable objetivo:").pack(anchor=tk.W)
        self.target_combo = ttk.Combobox(var_frame, state='readonly')
        self.target_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(var_frame, text="Modelo:").pack(anchor=tk.W, pady=(10,0))
        self.model_combo = ttk.Combobox(var_frame, state='readonly',
                                       values=["Random Forest", "Linear Regression"])
        self.model_combo.pack(fill=tk.X, pady=2)
        self.model_combo.current(0)
        
        ttk.Button(var_frame, text="🤖 Entrenar modelo",
                  command=self.train_model).pack(fill=tk.X, pady=10)
        
        # Resultados
        results_frame = ttk.LabelFrame(left_panel, text="Resultados", padding=10)
        results_frame.pack(fill=tk.X, pady=5)
        
        self.results_text = tk.Text(results_frame, height=10, wrap=tk.WORD,
                                   state=tk.DISABLED, font=('Courier', 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
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
            self.target_combo['values'] = numeric_cols
            if numeric_cols:
                self.target_combo.current(0)
    
    def train_model(self):
        target = self.target_combo.get()
        model_name = self.model_combo.get()
        
        if not target or self.current_df is None:
            messagebox.showwarning("Sin selección", "Configure el análisis")
            return
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import cross_val_score

            data_columns = [col for col in self.current_df.columns if not col.startswith('_')]
            numeric_cols = [col for col in data_columns
                          if pd.api.types.is_numeric_dtype(self.current_df[col])]
            predictors = [col for col in numeric_cols if col != target]

            if len(predictors) < 1:
                messagebox.showwarning("Datos insuficientes",
                                     "Se necesita al menos 1 variable predictora")
                return

            X = self.current_df[predictors].apply(pd.to_numeric, errors='coerce').fillna(0).values
            y = self.current_df[target].apply(pd.to_numeric, errors='coerce').fillna(0).values

            if model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=0)
            else:
                model = LinearRegression()

            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = model.score(X, y)

            cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)//2), scoring='r2')

            # Resultados
            results = f"""
Modelo: {model_name}
Target: {target}
Predictores: {len(predictors)}

R² (entrenamiento): {r2:.4f}
R² (CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

N observaciones: {len(X)}
"""

            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            self.results_text.config(state=tk.DISABLED)

            # Gráfico con Matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))

            # Scatter de predicciones
            ax.scatter(y, y_pred, alpha=0.6, edgecolors='k', s=50, label='Predicciones')

            # Línea de referencia perfecta
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfecto')

            ax.set_xlabel('Observado', fontsize=12)
            ax.set_ylabel('Predicho', fontsize=12)
            ax.set_title(f'{model_name} - Observado vs Predicho', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)

            # Añadir texto con R²
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            self.plot_viewer.add_plot("Modelo", fig, model_name, plot_type='matplotlib')

        except Exception as e:
            messagebox.showerror("Error", str(e))

