"""
Frame de análisis multivariado
"""
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .plot_viewer import MultiPlotViewer
from utils.plot_utils import PlotBuilder


class MultivariateAnalysisFrame(ttk.Frame):
    """Frame para análisis multivariado"""
    
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.current_df = None
        self.current_data = None
        
        self.create_widgets()
    
    def create_widgets(self):
        """Crea los widgets"""
        # Panel izquierdo
        left_panel = ttk.Frame(self, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        ttk.Label(left_panel, text="Análisis Multivariado",
                 font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        # Selección de variables
        var_frame = ttk.LabelFrame(left_panel, text="Variables", padding=10)
        var_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(var_frame, text="Variables a analizar:").pack(anchor=tk.W)
        
        # Listbox con scrollbar
        list_frame = ttk.Frame(var_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.var_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE,
                                      yscrollcommand=scrollbar.set, height=10)
        self.var_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.var_listbox.yview)
        
        ttk.Button(var_frame, text="Seleccionar todas",
                  command=self.select_all_vars).pack(fill=tk.X, pady=2)
        ttk.Button(var_frame, text="Limpiar selección",
                  command=self.clear_selection).pack(fill=tk.X, pady=2)
        ttk.Button(var_frame, text="🔄 Refrescar lista",
                  command=self.refresh_variable_list).pack(fill=tk.X, pady=2)
        
        # Análisis disponibles
        analysis_frame = ttk.LabelFrame(left_panel, text="Análisis", padding=10)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_frame, text="📊 Matriz de correlación",
                  command=self.plot_correlation_matrix).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="📈 PCA",
                  command=self.run_pca).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="🔬 Clustering",
                  command=self.run_clustering).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="🎯 Outliers multivariados",
                  command=self.detect_multivariate_outliers).pack(fill=tk.X, pady=2)
        
        # Botón de debug
        ttk.Button(analysis_frame, text="🔍 Debug tipos",
                  command=self.debug_column_types).pack(fill=tk.X, pady=2)
        
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
            self.update_variable_list()
    
    def refresh_variable_list(self):
        """Refresca la lista de variables cuando cambian los tipos de datos"""
        if self.current_df is not None:
            self.update_variable_list()
    
    def update_variable_list(self):
        """Actualiza la lista de variables"""
        if self.current_df is None:
            return
        
        self.var_listbox.delete(0, tk.END)
        
        # Filtrar columnas de metadatos
        data_columns = [col for col in self.current_df.columns if not col.startswith('_')]
        
        # Incluir variables numéricas y categóricas (excluir solo temporales)
        available_vars = []
        for col in data_columns:
            # Excluir columnas temporales
            if not self.is_temporal_column(col):
                available_vars.append(col)
        
        # Ordenar: primero numéricas, luego categóricas
        numeric_cols = [col for col in available_vars 
                       if pd.api.types.is_numeric_dtype(self.current_df[col])]
        categorical_cols = [col for col in available_vars 
                          if not pd.api.types.is_numeric_dtype(self.current_df[col])]
        
        # Insertar variables con indicador de tipo
        for col in numeric_cols:
            self.var_listbox.insert(tk.END, f"{col} (num)")
        
        for col in categorical_cols:
            self.var_listbox.insert(tk.END, f"{col} (cat)")
    
    def is_temporal_column(self, col_name):
        """Verifica si una columna es temporal (datetime/timestamp)"""
        # PRIORIDAD 1: Verificar por metadata (correcciones manuales tienen prioridad)
        type_col = f'_type_{col_name}'
        if type_col in self.current_df.columns and len(self.current_df) > 0:
            col_type = self.current_df[type_col].iloc[0]
            # Si la metadata dice que es temporal, entonces es temporal
            if col_type in ['datetime', 'time']:
                return True
            # Si la metadata dice que NO es temporal (cuantitativo/cualitativo), respetar eso
            elif col_type in ['quantitative', 'qualitative']:
                return False
        
        # PRIORIDAD 2: Verificar por tipo pandas
        if pd.api.types.is_datetime64_any_dtype(self.current_df[col_name]):
            return True
        
        # PRIORIDAD 3: Verificar por nombre (solo si no hay metadata o es ambigua)
        col_lower = col_name.lower()
        time_keywords = ['date', 'fecha', 'time', 'tiempo', 'timestamp', 'datetime', 
                        'référence', 'reference', 'created', 'updated', 'modified']
        for keyword in time_keywords:
            if keyword in col_lower:
                return True
        
        return False
    
    def select_all_vars(self):
        """Selecciona todas las variables"""
        self.var_listbox.select_set(0, tk.END)
    
    def clear_selection(self):
        """Limpia la selección"""
        self.var_listbox.selection_clear(0, tk.END)
    
    def get_selected_variables(self):
        """Obtiene las variables seleccionadas"""
        indices = self.var_listbox.curselection()
        
        if len(indices) < 2:
            messagebox.showwarning("Selección insuficiente",
                                 "Seleccione al menos 2 variables")
            return None
        
        # Extraer nombres de variables sin el indicador de tipo
        selected_vars = []
        for i in indices:
            var_with_type = self.var_listbox.get(i)
            # Remover el indicador (num) o (cat)
            var_name = var_with_type.replace(' (num)', '').replace(' (cat)', '')
            selected_vars.append(var_name)
        
        return selected_vars
    
    def plot_correlation_matrix(self):
        """Grafica matriz de correlación"""
        selected_vars = self.get_selected_variables()
        if selected_vars is None:
            return

        try:
            # Separar variables numéricas y categóricas
            numeric_vars = []
            categorical_vars = []
            
            for var in selected_vars:
                if pd.api.types.is_numeric_dtype(self.current_df[var]):
                    numeric_vars.append(var)
                else:
                    categorical_vars.append(var)
            
            if len(numeric_vars) < 2:
                messagebox.showwarning("Variables insuficientes", 
                                     "Para matriz de correlación se necesitan al menos 2 variables numéricas")
                return
            
            # Procesar solo variables numéricas para correlación
            numeric_data = self.current_df[numeric_vars].apply(pd.to_numeric, errors='coerce')
            # Matriz de correlación se calcula dentro del PlotBuilder

            # Crear título informativo
            title = "Matriz de Correlación"
            if categorical_vars:
                title += f" (Solo variables numéricas: {len(numeric_vars)} de {len(selected_vars)} seleccionadas)"

            # Usar PlotBuilder para crear heatmap de correlación
            fig = PlotBuilder.create_correlation_heatmap(
                df=numeric_data,
                title=title
            )

            self.plot_viewer.add_plot("Correlación", fig, "Matriz de correlación", plot_type='matplotlib')

        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráfico:\n{str(e)}")
    
    def run_pca(self):
        """Ejecuta PCA"""
        selected_vars = self.get_selected_variables()
        if selected_vars is None:
            return

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            import pandas as pd

            # Preparar datos mixtos
            X_mixed = self.current_df[selected_vars].copy()
            
            # Procesar variables categóricas con encoding
            encoded_vars = []
            var_names = []
            
            for var in selected_vars:
                if pd.api.types.is_numeric_dtype(X_mixed[var]):
                    # Variable numérica - convertir a numérico
                    numeric_vals = pd.to_numeric(X_mixed[var], errors='coerce')
                    encoded_vars.append(numeric_vals)
                    var_names.append(f"{var} (num)")
                else:
                    # Variable categórica - encoding
                    le = LabelEncoder()
                    encoded_vals = le.fit_transform(X_mixed[var].astype(str))
                    encoded_vars.append(encoded_vals)
                    var_names.append(f"{var} (cat)")

            # Crear DataFrame con todas las variables codificadas
            X = pd.DataFrame(dict(zip(var_names, encoded_vars)))
            X = X.dropna()

            if len(X) < 3:
                messagebox.showwarning("Datos insuficientes",
                                     "Se necesitan al menos 3 observaciones válidas")
                return

            # Estandarizar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # PCA
            n_components = min(len(var_names), len(X), 5)
            pca = PCA(n_components=n_components)
            pcs = pca.fit_transform(X_scaled)

            # Scree plot con Matplotlib
            evr = pca.explained_variance_ratio_
            evr_cum = np.cumsum(evr)

            fig, ax = plt.subplots(figsize=(10, 6))
            x_pos = np.arange(n_components)

            # Barras de varianza explicada
            ax.bar(x_pos, evr, alpha=0.7, color='skyblue', edgecolor='black', label='Varianza explicada')

            # Línea acumulada
            ax2 = ax.twinx()
            ax2.plot(x_pos, evr_cum, 'ro-', linewidth=2, markersize=8, label='Acumulada')
            ax2.set_ylabel('Varianza acumulada', fontsize=12)
            ax2.set_ylim(0, 1.1)

            ax.set_xlabel('Componente Principal', fontsize=12)
            ax.set_ylabel('Varianza explicada', fontsize=12)
            ax.set_title('Scree Plot - PCA (Variables Mixtas)', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"PC{i+1}" for i in range(n_components)])
            ax.grid(True, alpha=0.3, axis='y')

            # Combinar leyendas
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            plt.tight_layout()

            self.plot_viewer.add_plot("PCA Scree", fig, "Varianza explicada", plot_type='matplotlib')

            # Biplot
            if n_components >= 2:
                fig_bi, ax_bi = plt.subplots(figsize=(10, 8))

                ax_bi.scatter(pcs[:, 0], pcs[:, 1], alpha=0.6, edgecolors='k', s=50, c='steelblue')
                ax_bi.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)', fontsize=12)
                ax_bi.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)', fontsize=12)
                ax_bi.set_title(f'Biplot PCA - Variables Mixtas (Varianza total: {sum(evr[:2])*100:.1f}%)',
                               fontsize=14, fontweight='bold')
                ax_bi.axhline(y=0, color='gray', linestyle='--', linewidth=1)
                ax_bi.axvline(x=0, color='gray', linestyle='--', linewidth=1)
                ax_bi.grid(True, alpha=0.3)

                plt.tight_layout()

                self.plot_viewer.add_plot("PCA Biplot", fig_bi, "Componentes principales", plot_type='matplotlib')

        except Exception as e:
            messagebox.showerror("Error", f"Error en PCA:\n{str(e)}")
    
    def run_clustering(self):
        """Ejecuta clustering"""
        selected_vars = self.get_selected_variables()
        if selected_vars is None:
            return

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.decomposition import PCA
            import pandas as pd

            # Preparar datos mixtos
            X_mixed = self.current_df[selected_vars].copy()
            
            # Procesar variables categóricas con encoding
            encoded_vars = []
            var_names = []
            
            for var in selected_vars:
                if pd.api.types.is_numeric_dtype(X_mixed[var]):
                    # Variable numérica - convertir a numérico
                    numeric_vals = pd.to_numeric(X_mixed[var], errors='coerce')
                    encoded_vars.append(numeric_vals)
                    var_names.append(f"{var} (num)")
                else:
                    # Variable categórica - encoding
                    le = LabelEncoder()
                    encoded_vals = le.fit_transform(X_mixed[var].astype(str))
                    encoded_vars.append(encoded_vals)
                    var_names.append(f"{var} (cat)")

            # Crear DataFrame con todas las variables codificadas
            X = pd.DataFrame(dict(zip(var_names, encoded_vars)))
            X = X.dropna()

            if len(X) < 3:
                messagebox.showwarning("Datos insuficientes",
                                     "Se necesitan al menos 3 observaciones válidas")
                return

            # Estandarizar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # K-means con k=3
            kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            # Visualizar en PCA
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X_scaled)

            # Crear gráfico con Matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))

            # Colores para cada cluster
            colors = ['red', 'blue', 'green']
            for cluster_id in range(3):
                mask = labels == cluster_id
                ax.scatter(pcs[mask, 0], pcs[mask, 1],
                          c=colors[cluster_id], label=f'Cluster {cluster_id}',
                          alpha=0.6, edgecolors='k', s=50)

            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            ax.set_title('K-Means Clustering - Variables Mixtas (k=3)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            self.plot_viewer.add_plot("Clustering", fig, "Clusters", plot_type='matplotlib')

        except Exception as e:
            messagebox.showerror("Error", f"Error en clustering:\n{str(e)}")
    
    def detect_multivariate_outliers(self):
        """Detecta outliers multivariados"""
        selected_vars = self.get_selected_variables()
        if selected_vars is None:
            return

        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.decomposition import PCA
            import pandas as pd

            # Preparar datos mixtos
            X_mixed = self.current_df[selected_vars].copy()
            
            # Procesar variables categóricas con encoding
            encoded_vars = []
            var_names = []
            
            for var in selected_vars:
                if pd.api.types.is_numeric_dtype(X_mixed[var]):
                    # Variable numérica - convertir a numérico
                    numeric_vals = pd.to_numeric(X_mixed[var], errors='coerce')
                    encoded_vars.append(numeric_vals)
                    var_names.append(f"{var} (num)")
                else:
                    # Variable categórica - encoding
                    le = LabelEncoder()
                    encoded_vals = le.fit_transform(X_mixed[var].astype(str))
                    encoded_vars.append(encoded_vals)
                    var_names.append(f"{var} (cat)")

            # Crear DataFrame con todas las variables codificadas
            X = pd.DataFrame(dict(zip(var_names, encoded_vars)))
            X = X.dropna()

            if len(X) < 3:
                messagebox.showwarning("Datos insuficientes",
                                     "Se necesitan al menos 3 observaciones válidas")
                return

            # Estandarizar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=0)
            outlier_labels = iso_forest.fit_predict(X_scaled)

            n_outliers = (outlier_labels == -1).sum()

            # Visualizar en PCA
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X_scaled)

            # Crear gráfico con Matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))

            # Separar outliers de normales
            normal_mask = outlier_labels == 1
            outlier_mask = outlier_labels == -1

            ax.scatter(pcs[normal_mask, 0], pcs[normal_mask, 1],
                      c='blue', label='Normal', alpha=0.6, edgecolors='k', s=50)
            ax.scatter(pcs[outlier_mask, 0], pcs[outlier_mask, 1],
                      c='red', label='Outlier', alpha=0.8, edgecolors='k', s=80, marker='X')

            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            ax.set_title(f'Outliers Multivariados - Variables Mixtas (Detectados: {n_outliers})',
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            self.plot_viewer.add_plot("Outliers", fig, "Detección de outliers", plot_type='matplotlib')

            messagebox.showinfo("Outliers detectados",
                              f"Se detectaron {n_outliers} outliers de {len(X)} observaciones")

        except Exception as e:
            messagebox.showerror("Error", f"Error en detección de outliers:\n{str(e)}")
    
    def debug_column_types(self):
        """Método de debug para verificar tipos de columnas"""
        if self.current_df is None:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        
        debug_info = "=== DEBUG TIPOS DE COLUMNAS ===\n\n"
        
        # Filtrar columnas de metadatos
        data_columns = [col for col in self.current_df.columns if not col.startswith('_')]
        
        for col in data_columns:
            debug_info += f"Columna: {col}\n"
            
            # Tipo pandas
            pandas_type = str(self.current_df[col].dtype)
            debug_info += f"  Tipo pandas: {pandas_type}\n"
            
            # Metadata
            type_col = f'_type_{col}'
            if type_col in self.current_df.columns and len(self.current_df) > 0:
                metadata_type = self.current_df[type_col].iloc[0]
                debug_info += f"  Metadata: {metadata_type}\n"
            else:
                debug_info += "  Metadata: No disponible\n"
            
            # Detección temporal
            is_temporal = self.is_temporal_column(col)
            debug_info += f"  Detectada como temporal: {is_temporal}\n"
            
            # Clasificación final
            if is_temporal:
                debug_info += "  Clasificación: EXCLUIDA (temporal)\n"
            elif pd.api.types.is_numeric_dtype(self.current_df[col]):
                debug_info += "  Clasificación: NUMÉRICA\n"
            else:
                debug_info += "  Clasificación: CATEGÓRICA\n"
            
            debug_info += "\n"
        
        # Mostrar en ventana de mensaje
        messagebox.showinfo("Debug - Tipos de Columnas", debug_info[:1000] + "..." if len(debug_info) > 1000 else debug_info)

