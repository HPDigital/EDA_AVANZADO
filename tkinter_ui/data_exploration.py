"""
Frame de exploración de datos
"""
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from .plot_viewer import MultiPlotViewer
# from utils.plot_utils import PlotBuilder  # no usado en este módulo


class DataExplorationFrame(ttk.Frame):
    """Frame para exploración general de datos"""
    
    def __init__(self, parent, main_window):
        """
        Inicializa el frame de exploración
        
        Args:
            parent: Widget padre
            main_window: Instancia de MainWindow
        """
        super().__init__(parent)
        self.main_window = main_window
        self.current_df = None
        self.current_data = None
        self.column_types = {}  # Almacenar tipos de columnas
        self.original_types = {}  # Tipos originales detectados
        
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
            text="Exploración de Datos",
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=10)
        
        # Información general
        info_frame = ttk.LabelFrame(left_panel, text="Información General", padding=10)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD, 
                                state=tk.DISABLED, font=('Courier', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Tipos de datos y corrección
        types_frame = ttk.LabelFrame(left_panel, text="Tipos de Datos", padding=10)
        types_frame.pack(fill=tk.X, pady=5)
        
        # Frame para mostrar tipos detectados
        types_display_frame = ttk.Frame(types_frame)
        types_display_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Scrollbar para la lista de tipos
        types_scroll = ttk.Scrollbar(types_display_frame)
        types_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Lista de tipos de datos
        self.types_listbox = tk.Listbox(types_display_frame, height=6, 
                                       yscrollcommand=types_scroll.set,
                                       font=('Courier', 9))
        self.types_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        types_scroll.config(command=self.types_listbox.yview)
        
        # Frame para controles de tipo
        type_controls_frame = ttk.Frame(types_frame)
        type_controls_frame.pack(fill=tk.X, pady=5)
        
        # Selector de nuevo tipo
        ttk.Label(type_controls_frame, text="Cambiar a:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.type_var = tk.StringVar(value="quantitative")
        type_combo = ttk.Combobox(type_controls_frame, textvariable=self.type_var,
                                 values=["quantitative", "qualitative", "datetime"],
                                 state="readonly", width=12)
        type_combo.pack(side=tk.LEFT, padx=(0, 5))
        
        # Botón para aplicar cambio
        ttk.Button(type_controls_frame, text="Aplicar",
                  command=self.apply_type_change).pack(side=tk.LEFT, padx=(0, 5))
        
        # Botón para recargar tipos
        ttk.Button(type_controls_frame, text="🔄",
                  command=self.reload_types).pack(side=tk.RIGHT)
        
        # Botón de debug para forzar actualización
        ttk.Button(type_controls_frame, text="🔧",
                  command=self.debug_force_update).pack(side=tk.RIGHT, padx=(0, 5))
        
        # Opciones de visualización
        viz_frame = ttk.LabelFrame(left_panel, text="Visualizaciones", padding=10)
        viz_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            viz_frame,
            text="📊 Distribución por fuente",
            command=self.plot_source_distribution
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            viz_frame,
            text="📈 Tipos de datos",
            command=self.plot_data_types
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            viz_frame,
            text="🔍 Valores faltantes",
            command=self.plot_missing_values
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            viz_frame,
            text="📉 Estadísticas descriptivas",
            command=self.show_descriptive_stats
        ).pack(fill=tk.X, pady=2)
        
        # Panel derecho (visualizaciones)
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Notebook para múltiples vistas
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de vista previa
        preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(preview_frame, text="Vista Previa")
        
        # Tabla con scrollbars
        table_frame = ttk.Frame(preview_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        h_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        
        # Treeview para mostrar datos
        self.data_table = ttk.Treeview(
            table_frame,
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
            show='headings'
        )
        
        v_scroll.config(command=self.data_table.yview)
        h_scroll.config(command=self.data_table.xview)
        
        # Empaquetar
        self.data_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Pestaña de gráficos
        self.plot_viewer = MultiPlotViewer(self.notebook)
        self.notebook.add(self.plot_viewer, text="Gráficos")
        
        # Mensaje inicial
        self.show_no_data_message()
    
    def update_data(self, df, data_results):
        """
        Actualiza el frame con nuevos datos
        
        Args:
            df: DataFrame con los datos
            data_results: Resultados de la carga de datos
        """
        self.current_df = df
        self.current_data = data_results
        
        if df is not None and isinstance(df, pd.DataFrame):
            self.update_info()
            self.update_preview()
            self.detect_column_types()
        else:
            self.show_no_data_message()
    
    def update_info(self):
        """Actualiza la información general"""
        if self.current_df is None:
            return
        
        df = self.current_df
        
        # Filtrar columnas de metadatos
        data_columns = [col for col in df.columns if not col.startswith('_')]
        
        info_str = f"Filas: {len(df)}\n"
        info_str += f"Columnas: {len(data_columns)}\n"
        info_str += f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n"
        info_str += "\nTipos de datos:\n"
        
        # Contar tipos
        numeric_cols = [col for col in data_columns 
                       if pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in data_columns 
                          if not pd.api.types.is_numeric_dtype(df[col])]
        
        info_str += f"  Numéricos: {len(numeric_cols)}\n"
        info_str += f"  Categóricos: {len(categorical_cols)}\n"
        info_str += f"\nValores faltantes: {df[data_columns].isnull().sum().sum()}\n"
        
        # Archivos fuente
        if '_source_file' in df.columns:
            n_files = df['_source_file'].nunique()
            info_str += f"Archivos fuente: {n_files}\n"
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_str)
        self.info_text.config(state=tk.DISABLED)
    
    def update_preview(self):
        """Actualiza la vista previa de datos"""
        if self.current_df is None:
            return
        
        df = self.current_df
        
        # Filtrar columnas de metadatos
        data_columns = [col for col in df.columns if not col.startswith('_')]
        
        # Limpiar tabla
        self.data_table.delete(*self.data_table.get_children())
        
        # Configurar columnas (máximo 20 primeras)
        display_columns = data_columns[:20]
        self.data_table['columns'] = display_columns
        
        for col in display_columns:
            self.data_table.heading(col, text=col)
            self.data_table.column(col, width=120, anchor=tk.W)
        
        # Insertar filas (máximo 100 primeras)
        for idx, row in df[display_columns].head(100).iterrows():
            values = [str(v)[:50] for v in row]  # Truncar valores largos
            self.data_table.insert('', tk.END, values=values)
    
    def plot_source_distribution(self):
        """Grafica la distribución por archivo fuente"""
        if self.current_df is None:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return

        if '_source_file' not in self.current_df.columns:
            messagebox.showinfo(
                "No disponible",
                "La información de archivo fuente no está disponible para estos datos"
            )
            return

        try:
            source_counts = self.current_df['_source_file'].value_counts()

            # Crear gráfico con Matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(range(len(source_counts)), source_counts.values, color='skyblue', edgecolor='black')
            ax.set_xticks(range(len(source_counts)))
            ax.set_xticklabels(source_counts.index, rotation=45, ha='right')
            ax.set_title("Distribución de filas por archivo fuente", fontsize=14)
            ax.set_xlabel("Archivo", fontsize=12)
            ax.set_ylabel("Número de filas", fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()

            self.plot_viewer.add_plot(
                "Distribución",
                fig,
                "Filas por archivo fuente",
                plot_type='matplotlib'
            )
            
            # Cambiar a la pestaña de gráficos
            self.notebook.select(1)

        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráfico:\n{str(e)}")
    
    def plot_data_types(self):
        """Grafica la distribución de tipos de datos"""
        if self.current_df is None:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return

        try:
            # Filtrar columnas de metadatos
            data_columns = [col for col in self.current_df.columns
                          if not col.startswith('_')]

            # Clasificar tipos semánticos en lugar de tipos técnicos
            semantic_types = []
            for col in data_columns:
                semantic_type = self.get_semantic_type(col)
                semantic_types.append(semantic_type)

            # Contar tipos semánticos
            type_counts = pd.Series(semantic_types).value_counts()

            # Crear gráfico de pastel con Matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Colores específicos para cada tipo semántico
            color_map = {
                'Cuantitativo': '#4CAF50',      # Verde
                'Cualitativo': '#2196F3',       # Azul
                'Timestamp': '#FF9800'          # Naranja
            }
            
            colors = [color_map.get(tipo, '#9E9E9E') for tipo in type_counts.index]
            
            wedges, texts, autotexts = ax.pie(
                type_counts.values,
                labels=type_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )
            ax.set_title("Distribución de tipos de datos", fontsize=14, fontweight='bold')

            # Mejorar legibilidad
            for text in texts:
                text.set_fontsize(12)
                text.set_fontweight('bold')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)

            plt.tight_layout()

            self.plot_viewer.add_plot(
                "Tipos",
                fig,
                "Tipos de datos",
                plot_type='matplotlib'
            )
            
            # Cambiar a la pestaña de gráficos
            self.notebook.select(1)

        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráfico:\n{str(e)}")
    
    def get_semantic_type(self, col_name):
        """Obtiene el tipo semántico de una columna"""
        # Verificar si es temporal
        if self.is_temporal_column(col_name):
            return 'Timestamp'
        
        # Verificar si es numérica
        if pd.api.types.is_numeric_dtype(self.current_df[col_name]):
            return 'Cuantitativo'
        
        # Todo lo demás es cualitativo (incluyendo object, category, etc.)
        return 'Cualitativo'
    
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
    
    def plot_missing_values(self):
        """Grafica los valores faltantes"""
        if self.current_df is None:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return

        try:
            # Filtrar columnas de metadatos
            data_columns = [col for col in self.current_df.columns
                          if not col.startswith('_')]

            # Calcular valores faltantes
            missing = self.current_df[data_columns].isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)

            if len(missing) == 0:
                messagebox.showinfo("Sin datos faltantes",
                                  "No hay valores faltantes en el dataset")
                return

            # Crear gráfico con Matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(range(len(missing)), missing.values, color='lightcoral', edgecolor='black')
            ax.set_xticks(range(len(missing)))
            ax.set_xticklabels(missing.index, rotation=45, ha='right')
            ax.set_title("Valores faltantes por columna", fontsize=14)
            ax.set_xlabel("Columna", fontsize=12)
            ax.set_ylabel("Número de valores faltantes", fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()

            self.plot_viewer.add_plot(
                "Faltantes",
                fig,
                "Valores faltantes",
                plot_type='matplotlib'
            )
            
            # Cambiar a la pestaña de gráficos
            self.notebook.select(1)

        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráfico:\n{str(e)}")
    
    def show_descriptive_stats(self):
        """Muestra estadísticas descriptivas"""
        if self.current_df is None:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        
        try:
            # Filtrar columnas numéricas
            data_columns = [col for col in self.current_df.columns 
                          if not col.startswith('_')]
            numeric_cols = [col for col in data_columns 
                          if pd.api.types.is_numeric_dtype(self.current_df[col])]
            
            if not numeric_cols:
                messagebox.showinfo("Sin datos numéricos", 
                                  "No hay columnas numéricas para analizar")
                return
            
            # Calcular estadísticas
            stats_df = self.current_df[numeric_cols].describe()
            
            # Crear figura con tabla de estadísticas usando matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # Preparar datos para la tabla
            table_data = []
            for stat_name, row in stats_df.iterrows():
                row_data = [stat_name] + [f"{v:.2f}" for v in row]
                table_data.append(row_data)
            
            # Crear tabla
            table = ax.table(
                cellText=table_data,
                colLabels=['Estadístico'] + list(stats_df.columns),
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            # Estilizar tabla
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Colorear encabezados
            for i in range(len(stats_df.columns) + 1):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Colorear filas alternadas
            for i in range(1, len(table_data) + 1):
                for j in range(len(stats_df.columns) + 1):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
            
            plt.title('Estadísticas Descriptivas', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Mostrar en el visor de gráficos principal
            self.plot_viewer.add_plot(
                "Estadísticas",
                fig,
                "Estadísticas descriptivas",
                plot_type='matplotlib'
            )
            
            # Cambiar a la pestaña de gráficos
            self.notebook.select(1)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular estadísticas:\n{str(e)}")
    
    def show_no_data_message(self):
        """Muestra mensaje cuando no hay datos"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, "No hay datos cargados.\n\nCargue datos desde el menú lateral.")
        self.info_text.config(state=tk.DISABLED)
        
        # Limpiar tabla
        self.data_table.delete(*self.data_table.get_children())
        self.data_table['columns'] = []
        
        # Limpiar tipos de datos
        self.types_listbox.delete(0, tk.END)
        self.column_types = {}
        self.original_types = {}
    
    def detect_column_types(self):
        """Detecta los tipos de datos de las columnas"""
        if self.current_df is None:
            return
        
        from utils.smart_ingestion import SmartCSVIngestion
        
        # Filtrar columnas de metadatos
        data_columns = [col for col in self.current_df.columns if not col.startswith('_')]
        
        # Crear instancia del detector
        detector = SmartCSVIngestion()
        
        # Detectar tipos para cada columna
        self.column_types = {}
        self.original_types = {}
        
        for col in data_columns:
            series = self.current_df[col]
            
            # Primero, intentar leer el tipo desde la metadata si existe
            type_col = f'_type_{col}'
            if type_col in self.current_df.columns and len(self.current_df) > 0:
                existing_type = self.current_df[type_col].iloc[0]
                if existing_type in ['quantitative', 'qualitative', 'datetime', 'time']:
                    # Usar el tipo existente de la metadata
                    detected_type = existing_type
                    confidence = 1.0  # Alta confianza porque viene de metadata
                else:
                    # Detectar automáticamente
                    detected_type, confidence = detector.detect_column_type(series, col)
            else:
                # Detectar automáticamente
                detected_type, confidence = detector.detect_column_type(series, col)
            
            # Usar pandas para verificar si ya es numérico
            if pd.api.types.is_numeric_dtype(series) and detected_type == 'qualitative':
                detected_type = 'quantitative'
                confidence = 0.9
            
            self.column_types[col] = detected_type
            self.original_types[col] = (detected_type, confidence)
        
        self.update_types_display()
    
    def update_types_display(self):
        """Actualiza la visualización de tipos de datos"""
        print(f"DEBUG: update_types_display() - Diccionario tiene {len(self.column_types)} columnas")
        print(f"DEBUG: Contenido del diccionario: {self.column_types}")
        
        self.types_listbox.delete(0, tk.END)
        
        for col, col_type in self.column_types.items():
            # Obtener confianza original
            original_type, original_confidence = self.original_types.get(col, (col_type, 0.0))
            
            # Indicar si fue modificado
            is_modified = col_type != original_type
            modified = " (modificado)" if is_modified else ""
            
            # Usar confianza alta para tipos modificados manualmente
            if is_modified:
                confidence = 1.0  # Alta confianza para cambios manuales
            else:
                confidence = original_confidence
            
            # Formatear línea
            line = f"{col[:25]:<25} | {col_type:<12} | {confidence:.2f}{modified}"
            print(f"DEBUG: Insertando línea: {line}")
            self.types_listbox.insert(tk.END, line)
        
        print(f"DEBUG: Listbox ahora tiene {self.types_listbox.size()} elementos")
    
    def apply_type_change(self):
        """Aplica el cambio de tipo seleccionado"""
        selection = self.types_listbox.curselection()
        if not selection:
            messagebox.showwarning("Sin selección", "Seleccione una columna para cambiar su tipo")
            return
        
        # Obtener columna seleccionada usando el índice
        selected_index = selection[0]
        # Obtener el nombre de la columna directamente desde el diccionario
        column_names = list(self.column_types.keys())
        if selected_index >= len(column_names):
            messagebox.showerror("Error", "Índice de columna inválido")
            return
        
        column_name = column_names[selected_index]
        
        # Obtener nuevo tipo
        new_type = self.type_var.get()
        
        # Debug: imprimir información
        print(f"DEBUG: Cambiando columna '{column_name}' de tipo '{self.column_types.get(column_name, 'unknown')}' a '{new_type}'")
        
        # Validar que la columna existe
        if column_name not in self.current_df.columns:
            messagebox.showerror("Error", f"La columna '{column_name}' no existe")
            return
        
        try:
            # Aplicar conversión de tipo
            if new_type == 'quantitative':
                # Convertir a numérico
                self.current_df[column_name] = pd.to_numeric(
                    self.current_df[column_name], errors='coerce'
                )
                messagebox.showinfo("Éxito", f"Columna '{column_name}' convertida a numérico")
                
            elif new_type == 'datetime':
                # Convertir a datetime
                self.current_df[column_name] = pd.to_datetime(
                    self.current_df[column_name], errors='coerce'
                )
                messagebox.showinfo("Éxito", f"Columna '{column_name}' convertida a fecha")
                
            elif new_type == 'qualitative':
                # Convertir a categórico
                self.current_df[column_name] = self.current_df[column_name].astype('category')
                messagebox.showinfo("Éxito", f"Columna '{column_name}' convertida a categórico")
            
            # Actualizar el tipo en el diccionario local
            old_type = self.column_types.get(column_name, 'unknown')
            self.column_types[column_name] = new_type
            
            print(f"DEBUG: Diccionario actualizado - {column_name}: {new_type}")
            
            # Actualizar la columna de metadata _type_ en el DataFrame
            type_col = f'_type_{column_name}'
            if type_col in self.current_df.columns:
                self.current_df[type_col] = new_type
                print(f"DEBUG: Metadata actualizada - {type_col}: {new_type}")
            
            # Forzar actualización inmediata de la visualización
            print("DEBUG: Llamando a update_types_display()")
            self.update_types_display()
            print("DEBUG: update_types_display() completado")
            
            # Restaurar la selección en la lista actualizada
            try:
                column_names = list(self.column_types.keys())
                if column_name in column_names:
                    new_index = column_names.index(column_name)
                    self.types_listbox.selection_clear(0, tk.END)
                    self.types_listbox.selection_set(new_index)
                    self.types_listbox.see(new_index)
            except:
                pass  # Si falla la selección, no es crítico
            
            # Actualizar vista previa
            self.update_preview()
            
            # Forzar actualización de la interfaz
            self.update_idletasks()
            
            # Notificar a la ventana principal que los datos han cambiado
            # para que otros módulos se actualicen si es necesario
            if hasattr(self.main_window, 'on_data_type_changed'):
                self.main_window.on_data_type_changed(column_name, old_type, new_type)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al convertir tipo:\n{str(e)}")
    
    def reload_types(self):
        """Recarga los tipos de datos detectados automáticamente"""
        if self.current_df is not None:
            self.detect_column_types()
            messagebox.showinfo("Recargado", "Tipos de datos recargados automáticamente")
        else:
            messagebox.showwarning("Sin datos", "No hay datos para recargar")
    
    def debug_force_update(self):
        """Método de debug para forzar actualización de la visualización"""
        print("=== DEBUG FORCE UPDATE ===")
        print(f"Current column_types: {self.column_types}")
        print(f"Current original_types: {self.original_types}")
        print(f"Listbox size before: {self.types_listbox.size()}")
        
        # Forzar actualización
        self.update_types_display()
        
        print(f"Listbox size after: {self.types_listbox.size()}")
        print("=== END DEBUG ===")
        
        messagebox.showinfo("Debug", f"Listbox actualizada. Tamaño: {self.types_listbox.size()}")
