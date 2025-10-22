"""
Pestaña de ingesta de datos integrada en la aplicación principal
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from pathlib import Path
 

from utils.step_by_step_ingestion import StepByStepIngestion


class IngestionTab(ttk.Frame):
    """Pestaña de ingesta de datos integrada"""
    
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.ingestion = StepByStepIngestion()
        self.current_step = 1
        self.selected_files = []
        
        self.create_widgets()
    
    def create_widgets(self):
        """Crea la interfaz de usuario"""
        # Frame principal con scroll
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(main_frame, text="📊 Ingesta de Datos", 
                               font=('Segoe UI', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Crear notebook para los pasos
        self.steps_notebook = ttk.Notebook(main_frame)
        self.steps_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Paso 1: Selección y carga de archivos
        self.step1_frame = self.create_step1_frame()
        self.steps_notebook.add(self.step1_frame, text="1️⃣ Cargar Archivos")
        
        # Paso 2: Previsualización y corrección de tipos
        self.step2_frame = self.create_step2_frame()
        self.steps_notebook.add(self.step2_frame, text="2️⃣ Previsualizar y Corregir")
        
        # Paso 3: Configurar unificación
        self.step3_frame = self.create_step3_frame()
        self.steps_notebook.add(self.step3_frame, text="3️⃣ Configurar Unificación")
        
        # Botones de navegación
        self.create_navigation_buttons(main_frame)
    
    def create_step1_frame(self):
        """Crea el frame para el paso 1"""
        frame = ttk.Frame(self.steps_notebook)
        
        # Instrucciones
        instructions = ttk.Label(frame, 
            text="Selecciona uno o más archivos de datos que deseas cargar.\n"
                 "El sistema detectará automáticamente el formato y tipos de datos.\n"
                 "Puedes trabajar con un solo archivo o múltiples archivos.",
            font=('Segoe UI', 10))
        instructions.pack(pady=10)
        
        # Frame para selección de archivos
        selection_frame = ttk.LabelFrame(frame, text="Selección de Archivos", padding=10)
        selection_frame.pack(fill=tk.X, pady=10)
        
        # Botón para seleccionar archivos
        select_btn = ttk.Button(selection_frame, text="📁 Seleccionar Archivos", 
                               command=self.select_files, style='Accent.TButton')
        select_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botón para cargar archivos
        load_btn = ttk.Button(selection_frame, text="🔄 Cargar Archivos", 
                             command=self.load_files)
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botón para carga directa
        direct_btn = ttk.Button(selection_frame, text="⚡ Cargar y Aplicar", 
                               command=self.load_and_apply_direct, style='Accent.TButton')
        direct_btn.pack(side=tk.LEFT)
        
        # Lista de archivos seleccionados
        files_frame = ttk.LabelFrame(frame, text="Archivos Seleccionados", padding=10)
        files_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Treeview para mostrar archivos
        columns = ('archivo', 'formato', 'estado', 'filas', 'columnas')
        self.files_tree = ttk.Treeview(files_frame, columns=columns, show='headings', height=8)
        
        # Configurar columnas
        self.files_tree.heading('archivo', text='Archivo')
        self.files_tree.heading('formato', text='Formato')
        self.files_tree.heading('estado', text='Estado')
        self.files_tree.heading('filas', text='Filas')
        self.files_tree.heading('columnas', text='Columnas')
        
        self.files_tree.column('archivo', width=200)
        self.files_tree.column('formato', width=80)
        self.files_tree.column('estado', width=100)
        self.files_tree.column('filas', width=60)
        self.files_tree.column('columnas', width=60)
        
        # Scrollbar para el tree
        files_scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.files_tree.yview)
        self.files_tree.config(yscrollcommand=files_scrollbar.set)
        
        self.files_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Área de resultados
        results_frame = ttk.LabelFrame(frame, text="Resultados de Carga", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.step1_results = tk.Text(results_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.step1_results.pack(fill=tk.BOTH, expand=True)
        
        return frame
    
    def create_step2_frame(self):
        """Crea el frame para el paso 2"""
        frame = ttk.Frame(self.steps_notebook)
        
        # Instrucciones
        instructions = ttk.Label(frame, 
            text="Revisa los datos cargados y corrige los tipos de datos si es necesario.\n"
                 "Selecciona un archivo para ver sus datos y tipos detectados.",
            font=('Segoe UI', 10))
        instructions.pack(pady=10)
        
        # Frame superior para selección de archivo
        file_selection_frame = ttk.LabelFrame(frame, text="Seleccionar Archivo", padding=10)
        file_selection_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(file_selection_frame, text="Archivo:").pack(side=tk.LEFT, padx=(0, 10))
        self.file_combo = ttk.Combobox(file_selection_frame, state='readonly', width=40)
        self.file_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.file_combo.bind('<<ComboboxSelected>>', self.on_file_selected)
        
        refresh_btn = ttk.Button(file_selection_frame, text="🔄 Actualizar", 
                                command=self.refresh_file_list)
        refresh_btn.pack(side=tk.LEFT)
        
        # Frame principal con dos paneles
        main_panel = ttk.Frame(frame)
        main_panel.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Panel izquierdo: Previsualización de datos
        preview_frame = ttk.LabelFrame(main_panel, text="Previsualización de Datos", padding=10)
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Treeview para previsualización
        self.preview_tree = ttk.Treeview(preview_frame, height=15)
        self.preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        preview_scrollbar_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_tree.yview)
        preview_scrollbar_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.preview_tree.xview)
        self.preview_tree.config(yscrollcommand=preview_scrollbar_y.set, xscrollcommand=preview_scrollbar_x.set)
        
        preview_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        preview_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Panel derecho: Corrección de tipos
        types_frame = ttk.LabelFrame(main_panel, text="Corrección de Tipos", padding=10)
        types_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Lista de columnas con tipos
        self.types_listbox = tk.Listbox(types_frame, height=10)
        self.types_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Controles de corrección
        correction_frame = ttk.Frame(types_frame)
        correction_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(correction_frame, text="Nuevo tipo:").pack(side=tk.LEFT, padx=(0, 5))
        self.type_combo = ttk.Combobox(correction_frame, values=['quantitative', 'qualitative', 'datetime'], 
                                      state='readonly', width=15)
        self.type_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        apply_btn = ttk.Button(correction_frame, text="Aplicar", command=self.apply_type_correction)
        apply_btn.pack(side=tk.LEFT)
        
        return frame
    
    def create_step3_frame(self):
        """Crea el frame para el paso 3"""
        frame = ttk.Frame(self.steps_notebook)
        
        # Instrucciones
        instructions = ttk.Label(frame, 
            text="Configura cómo se procesarán los archivos:\n"
                 "• Un solo archivo: Se carga directamente sin unificación\n"
                 "• Múltiples archivos: Selecciona las columnas que servirán como unión\n"
                 "  (estas columnas deben contener valores comunes entre los archivos)",
            font=('Segoe UI', 10))
        instructions.pack(pady=10)
        
        # Frame para información de archivo único
        self.single_file_frame = ttk.LabelFrame(frame, text="Archivo Único", padding=10)
        self.single_file_frame.pack(fill=tk.X, pady=10)
        
        self.single_file_info = ttk.Label(self.single_file_frame, 
            text="Se detectó un solo archivo. Se cargará directamente sin unificación.",
            font=('Segoe UI', 10))
        self.single_file_info.pack(pady=5)
        
        # Botones de acción para archivo único
        single_file_actions = ttk.Frame(self.single_file_frame)
        single_file_actions.pack(fill=tk.X, pady=(10, 0))
        
        # Botón para cargar archivo único directamente
        load_single_btn = ttk.Button(single_file_actions, text="⚡ Cargar Archivo Único", 
                                    command=self.load_single_file_direct, style='Accent.TButton')
        load_single_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botón para ir a configuración
        config_btn = ttk.Button(single_file_actions, text="⚙️ Configurar", 
                               command=self.go_to_config)
        config_btn.pack(side=tk.LEFT)
        
        # Frame para configuración de unificación
        self.config_frame = ttk.LabelFrame(frame, text="Configuración de Unificación", padding=10)
        self.config_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Panel superior: Selección de columnas pivote
        pivot_frame = ttk.LabelFrame(self.config_frame, text="Columnas Pivote", padding=10)
        pivot_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Treeview para mostrar archivos y sus columnas pivote
        pivot_columns = ('archivo', 'columna_pivote', 'tipo', 'valores_unicos')
        self.pivot_tree = ttk.Treeview(pivot_frame, columns=pivot_columns, show='headings', height=8)
        
        self.pivot_tree.heading('archivo', text='Archivo')
        self.pivot_tree.heading('columna_pivote', text='Columna Pivote')
        self.pivot_tree.heading('tipo', text='Tipo')
        self.pivot_tree.heading('valores_unicos', text='Valores Únicos')
        
        self.pivot_tree.column('archivo', width=200)
        self.pivot_tree.column('columna_pivote', width=150)
        self.pivot_tree.column('tipo', width=100)
        self.pivot_tree.column('valores_unicos', width=100)
        
        pivot_scrollbar = ttk.Scrollbar(pivot_frame, orient=tk.VERTICAL, command=self.pivot_tree.yview)
        self.pivot_tree.config(yscrollcommand=pivot_scrollbar.set)
        
        self.pivot_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pivot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Controles para agregar/quitar columnas pivote
        controls_frame = ttk.Frame(pivot_frame)
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(controls_frame, text="Archivo:").pack(side=tk.LEFT, padx=(0, 5))
        self.pivot_file_combo = ttk.Combobox(controls_frame, state='readonly', width=20)
        self.pivot_file_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.pivot_file_combo.bind('<<ComboboxSelected>>', self.on_pivot_file_selected)
        
        ttk.Label(controls_frame, text="Columna:").pack(side=tk.LEFT, padx=(0, 5))
        self.pivot_column_combo = ttk.Combobox(controls_frame, state='readonly', width=20)
        self.pivot_column_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        add_pivot_btn = ttk.Button(controls_frame, text="➕ Agregar", 
                                  command=self.add_pivot_column)
        add_pivot_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        remove_pivot_btn = ttk.Button(controls_frame, text="➖ Quitar", 
                                     command=self.remove_pivot_column)
        remove_pivot_btn.pack(side=tk.LEFT)
        
        # Panel inferior: Botones de acción
        action_frame = ttk.Frame(self.config_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        # Botón para unificar
        unify_btn = ttk.Button(action_frame, text="🔗 Unificar Datos", 
                              command=self.unify_data, style='Accent.TButton')
        unify_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botón para cargar en la aplicación
        load_app_btn = ttk.Button(action_frame, text="✅ Cargar en Aplicación", 
                                 command=self.load_to_application, style='Accent.TButton')
        load_app_btn.pack(side=tk.LEFT)
        
        # Botón de carga directa (siempre visible)
        direct_load_btn = ttk.Button(action_frame, text="⚡ Cargar Directo", 
                                    command=self.direct_load_files, style='Accent.TButton')
        direct_load_btn.pack(side=tk.RIGHT)
        
        # Área de resultados
        results_frame = ttk.LabelFrame(self.config_frame, text="Resultados de Unificación", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.step3_results = tk.Text(results_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.step3_results.pack(fill=tk.BOTH, expand=True)
        
        # Botones principales al final del frame
        main_actions_frame = ttk.Frame(frame)
        main_actions_frame.pack(fill=tk.X, pady=20)
        
        # Botón principal de carga
        main_load_btn = ttk.Button(main_actions_frame, text="✅ CARGAR EN APLICACIÓN", 
                                  command=self.load_to_application, style='Accent.TButton')
        main_load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botón de carga directa
        main_direct_btn = ttk.Button(main_actions_frame, text="⚡ CARGA DIRECTA", 
                                    command=self.direct_load_files, style='Accent.TButton')
        main_direct_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botón de unificación
        main_unify_btn = ttk.Button(main_actions_frame, text="🔗 UNIFICAR DATOS", 
                                   command=self.unify_data)
        main_unify_btn.pack(side=tk.LEFT)
        
        return frame
    
    def create_navigation_buttons(self, parent):
        """Crea los botones de navegación"""
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, pady=10)
        
        prev_btn = ttk.Button(nav_frame, text="⬅️ Anterior", command=self.prev_step)
        prev_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        next_btn = ttk.Button(nav_frame, text="Siguiente ➡️", command=self.next_step)
        next_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        reset_btn = ttk.Button(nav_frame, text="🔄 Reiniciar", command=self.reset_ingestion)
        reset_btn.pack(side=tk.RIGHT)
        
        self.nav_buttons = {
            'prev': prev_btn,
            'next': next_btn,
            'reset': reset_btn
        }
    
    def select_files(self):
        """Selecciona archivos para cargar"""
        try:
            from utils.multi_format_loader import MultiFormatLoader
            loader = MultiFormatLoader()
            supported_formats = loader.get_supported_formats()
            
            # Crear lista de tipos de archivo para el diálogo (solo CSV y XLSX)
            filetypes = []
            # Poner "Archivos soportados" primero para que sea el filtro por defecto
            filetypes.append(("Archivos soportados", "*.csv;*.xlsx"))
            filetypes.append(("Todos los archivos", "*.*"))
            # Agregar formatos individuales después
            for ext, description in supported_formats.items():
                filetypes.append((description, f"*{ext}"))
            
            # Usar directorio de datos configurado como inicial
            from config import DEFAULT_DATA_DIR
            initial_dir = DEFAULT_DATA_DIR
            
            files = filedialog.askopenfilenames(
                title="Seleccionar archivos de datos",
                filetypes=filetypes,
                initialdir=initial_dir
            )
            
            if files:
                self.selected_files = list(files)
                self.update_files_tree()
                messagebox.showinfo("Archivos seleccionados", f"Se seleccionaron {len(files)} archivo(s)")
            else:
                messagebox.showinfo("Sin selección", "No se seleccionaron archivos")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al seleccionar archivos:\n{str(e)}")
    
    def update_files_tree(self):
        """Actualiza el tree de archivos seleccionados"""
        # Limpiar tree
        for item in self.files_tree.get_children():
            self.files_tree.delete(item)
        
        # Agregar archivos seleccionados
        for file_path in self.selected_files:
            file_name = Path(file_path).name
            file_format = Path(file_path).suffix.lower()
            self.files_tree.insert('', 'end', values=(file_name, file_format, 'Pendiente', '-', '-'))
    
    def load_files(self):
        """Carga los archivos seleccionados"""
        if not self.selected_files:
            messagebox.showwarning("Sin archivos", "Selecciona al menos un archivo")
            return
        
        # Limpiar resultados anteriores
        self.ingestion.reset()
        
        # Cargar archivos
        results = self.ingestion.step1_ingest_individual_files(self.selected_files)
        
        # Actualizar tree con resultados
        self.update_files_tree_with_results(results)
        
        # Mostrar resultados
        self.step1_results.config(state=tk.NORMAL)
        self.step1_results.delete(1.0, tk.END)
        
        success_count = 0
        for i, result in enumerate(results):
            if result.success:
                success_count += 1
                self.step1_results.insert(tk.END, f"✅ {result.file_name}\n")
                self.step1_results.insert(tk.END, f"   Formato: {result.file_format}\n")
                self.step1_results.insert(tk.END, f"   Filas: {len(result.df)}\n")
                self.step1_results.insert(tk.END, f"   Columnas: {len(result.df.columns)}\n")
                self.step1_results.insert(tk.END, "\n")
            else:
                self.step1_results.insert(tk.END, f"❌ {result.file_name}\n")
                self.step1_results.insert(tk.END, f"   Error: {result.error_message}\n")
                self.step1_results.insert(tk.END, "\n")
        
        self.step1_results.insert(tk.END, f"\nResumen: {success_count}/{len(results)} archivos cargados exitosamente")
        self.step1_results.config(state=tk.DISABLED)
        
        if success_count > 0:
            # Habilitar paso 2
            self.steps_notebook.tab(1, state='normal')
            self.refresh_file_list()
            # Actualizar información de archivo único
            self.refresh_single_file_info()
            messagebox.showinfo("Carga exitosa", f"Se cargaron {success_count} archivos exitosamente")
        else:
            messagebox.showerror("Error", "No se pudo cargar ningún archivo")
    
    def update_files_tree_with_results(self, results):
        """Actualiza el tree con los resultados de carga"""
        # Limpiar tree
        for item in self.files_tree.get_children():
            self.files_tree.delete(item)
        
        # Agregar resultados
        for i, result in enumerate(results):
            file_name = result.file_name
            file_format = result.file_format
            
            if result.success:
                estado = "✅ Cargado"
                filas = str(len(result.df))
                columnas = str(len(result.df.columns))
            else:
                estado = "❌ Error"
                filas = "-"
                columnas = "-"
            
            self.files_tree.insert('', 'end', values=(file_name, file_format, estado, filas, columnas))
    
    def refresh_file_list(self):
        """Actualiza la lista de archivos en el paso 2"""
        self.file_combo['values'] = []
        for i, result in enumerate(self.ingestion.ingested_files):
            if result.success:
                self.file_combo['values'] = list(self.file_combo['values']) + [f"{i}: {result.file_name}"]
        
        if self.file_combo['values']:
            self.file_combo.current(0)
            self.on_file_selected(None)
    
    def on_file_selected(self, event):
        """Maneja la selección de archivo en el paso 2"""
        selection = self.file_combo.get()
        if not selection:
            return
        
        try:
            file_index = int(selection.split(':')[0])
            self.current_file_index = file_index
            self.update_data_preview()
            self.update_column_types_display()
        except (ValueError, IndexError):
            pass
    
    def update_data_preview(self):
        """Actualiza la previsualización de datos"""
        try:
            # Limpiar tree completamente
            for item in self.preview_tree.get_children():
                self.preview_tree.delete(item)
            
            # Limpiar columnas existentes
            self.preview_tree['columns'] = []
            self.preview_tree['show'] = 'tree'
            
            if not hasattr(self, 'current_file_index'):
                return
            
            file_result = self.ingestion.ingested_files[self.current_file_index]
            if not file_result.success:
                return
            
            df = file_result.df
            
            # Debug: Mostrar información del archivo
            print(f"[DEBUG] Mostrando archivo: {file_result.file_name}")
            print(f"[DEBUG] Columnas: {list(df.columns)}")
            print(f"[DEBUG] Primera fila: {df.iloc[0].to_dict()}")
            
            # Configurar columnas del tree
            columns = list(df.columns)
            self.preview_tree['columns'] = columns
            self.preview_tree['show'] = 'headings'
            
            # Configurar encabezados
            for col in columns:
                self.preview_tree.heading(col, text=col)
                self.preview_tree.column(col, width=100)
            
            # Agregar datos (máximo 50 filas)
            for i, row in df.head(50).iterrows():
                values = [str(row[col]) if pd.notna(row[col]) else '' for col in columns]
                self.preview_tree.insert('', 'end', values=values)
                
        except Exception as e:
            print(f"[ERROR] Error en update_data_preview: {str(e)}")
            # Limpiar tree en caso de error
            for item in self.preview_tree.get_children():
                self.preview_tree.delete(item)
            self.preview_tree['columns'] = []
            self.preview_tree['show'] = 'tree'
    
    def update_column_types_display(self):
        """Actualiza la visualización de tipos de columnas"""
        self.types_listbox.delete(0, tk.END)
        
        if not hasattr(self, 'current_file_index'):
            return
        
        file_result = self.ingestion.ingested_files[self.current_file_index]
        if not file_result.success:
            return
        
        for col in file_result.df.columns:
            if col.startswith('_'):
                continue
            
            detected_type = file_result.column_types.get(col, 'unknown')
            corrected_type = file_result.corrected_types.get(col, detected_type)
            
            # Mostrar si fue corregido
            status = " (corregido)" if detected_type != corrected_type else ""
            
            line = f"{col:<25} | {detected_type:<12} -> {corrected_type:<12}{status}"
            self.types_listbox.insert(tk.END, line)
    
    def apply_type_correction(self):
        """Aplica la corrección de tipo seleccionada"""
        if not hasattr(self, 'current_file_index'):
            messagebox.showwarning("Sin selección", "Selecciona un archivo primero")
            return
        
        selection = self.types_listbox.curselection()
        if not selection:
            messagebox.showwarning("Sin selección", "Selecciona una columna")
            return
        
        new_type = self.type_combo.get()
        if not new_type:
            messagebox.showwarning("Sin tipo", "Selecciona un tipo de dato")
            return
        
        # Obtener nombre de la columna
        line = self.types_listbox.get(selection[0])
        col_name = line.split(' | ')[0].strip()
        
        # Aplicar corrección
        success = self.ingestion.step2_correct_column_types(
            self.current_file_index, 
            {col_name: new_type}
        )
        
        if success:
            messagebox.showinfo("Corrección aplicada", f"Tipo de columna '{col_name}' cambiado a '{new_type}'")
            self.update_column_types_display()
            self.update_data_preview()
        else:
            messagebox.showerror("Error", "No se pudo aplicar la corrección")
    
    def refresh_pivot_display(self):
        """Actualiza la visualización de columnas pivote"""
        try:
            # Limpiar tree
            for item in self.pivot_tree.get_children():
                self.pivot_tree.delete(item)
            
            # Contar archivos cargados exitosamente
            successful_files = [r for r in self.ingestion.ingested_files if r.success]
            file_count = len(successful_files)
            
            if file_count == 0:
                # No hay archivos cargados
                self.single_file_frame.pack_forget()
                self.config_frame.pack_forget()
                print("[DEBUG] No hay archivos cargados")
                
            elif file_count == 1:
                # Un solo archivo - mostrar información de archivo único
                self.single_file_frame.pack(fill=tk.X, pady=10)
                self.config_frame.pack_forget()
                
                file_result = successful_files[0]
                self.single_file_info.config(
                    text=f"Archivo único detectado: {file_result.file_name}\n"
                         f"Filas: {len(file_result.df)}, Columnas: {len(file_result.df.columns)}\n"
                         f"Se cargará directamente sin unificación."
                )
                print(f"[DEBUG] Archivo único detectado: {file_result.file_name}")
                
            else:
                # Múltiples archivos - mostrar configuración de unificación
                self.single_file_frame.pack_forget()
                self.config_frame.pack(fill=tk.BOTH, expand=True, pady=10)
                
                # Actualizar combo de archivos
                self.pivot_file_combo['values'] = []
                file_options = []
                
                for i, result in enumerate(self.ingestion.ingested_files):
                    if result.success:
                        file_options.append(f"{i}: {result.file_name}")
                
                self.pivot_file_combo['values'] = file_options
                
                if file_options:
                    self.pivot_file_combo.current(0)
                    self.on_pivot_file_selected(None)
                    print(f"[DEBUG] Pivot display actualizado con {len(file_options)} archivos")
                
        except Exception as e:
            print(f"[ERROR] Error en refresh_pivot_display: {str(e)}")
            messagebox.showerror("Error", f"Error al actualizar visualización de pivotes:\n{str(e)}")
    
    def on_pivot_file_selected(self, event):
        """Maneja la selección de archivo para pivote"""
        selection = self.pivot_file_combo.get()
        if not selection:
            return
        
        try:
            file_index = int(selection.split(':')[0])
            columns = self.ingestion.get_available_columns_for_pivot(file_index)
            self.pivot_column_combo['values'] = columns
            if columns:
                self.pivot_column_combo.current(0)
                print(f"[DEBUG] Archivo {file_index} seleccionado, {len(columns)} columnas disponibles")
            else:
                print(f"[DEBUG] Archivo {file_index} seleccionado, sin columnas disponibles")
        except (ValueError, IndexError) as e:
            print(f"[ERROR] Error en on_pivot_file_selected: {str(e)}")
            messagebox.showerror("Error", f"Error al seleccionar archivo:\n{str(e)}")
    
    def add_pivot_column(self):
        """Agrega una columna pivote"""
        file_selection = self.pivot_file_combo.get()
        column = self.pivot_column_combo.get()
        
        if not file_selection or not column:
            messagebox.showwarning("Sin selección", "Selecciona archivo y columna")
            return
        
        try:
            file_index = int(file_selection.split(':')[0])
            file_result = self.ingestion.ingested_files[file_index]
            
            # Obtener información de la columna
            col_type = file_result.corrected_types.get(column, 'unknown')
            unique_count = file_result.df[column].nunique()
            
            # Agregar al tree
            file_name = file_result.file_name
            self.pivot_tree.insert('', 'end', values=(file_name, column, col_type, str(unique_count)))
            
        except (ValueError, IndexError):
            messagebox.showerror("Error", "Selección inválida")
    
    def remove_pivot_column(self):
        """Quita una columna pivote seleccionada"""
        selection = self.pivot_tree.selection()
        if not selection:
            messagebox.showwarning("Sin selección", "Selecciona una fila para quitar")
            return
        
        for item in selection:
            self.pivot_tree.delete(item)
    
    def unify_data(self):
        """Unifica los datos usando las columnas pivote o carga un archivo único"""
        # Verificar si hay un solo archivo
        successful_files = [r for r in self.ingestion.ingested_files if r.success]
        
        if len(successful_files) == 1:
            # Un solo archivo - cargar directamente
            file_result = successful_files[0]
            unified_df = file_result.df.copy()
            
            # Agregar metadatos del archivo
            unified_df['_source_file'] = file_result.file_name
            unified_df['_file_index'] = 0
            unified_df['_file_format'] = file_result.file_format
            
            # Agregar información de tipos de columnas
            for col, col_type in file_result.corrected_types.items():
                if col in unified_df.columns:
                    unified_df[f'_type_{col}'] = col_type
            
            # Guardar en el sistema de ingesta
            self.ingestion.unified_df = unified_df
            
            # Mostrar resultados
            self.step3_results.config(state=tk.NORMAL)
            self.step3_results.delete(1.0, tk.END)
            
            self.step3_results.insert(tk.END, "✅ Archivo único cargado exitosamente\n")
            self.step3_results.insert(tk.END, f"Archivo: {file_result.file_name}\n")
            self.step3_results.insert(tk.END, f"Filas: {len(unified_df)}\n")
            self.step3_results.insert(tk.END, f"Columnas: {len(unified_df.columns)}\n\n")
            
            # Mostrar columnas
            self.step3_results.insert(tk.END, "Columnas del DataFrame:\n")
            for col in unified_df.columns:
                if not col.startswith('_'):
                    self.step3_results.insert(tk.END, f"  - {col}\n")
            
            messagebox.showinfo("Archivo cargado", "El archivo único ha sido cargado correctamente")
            
        else:
            # Múltiples archivos - unificar usando columnas pivote
            pivot_columns = {}
            for item in self.pivot_tree.get_children():
                values = self.pivot_tree.item(item, 'values')
                file_name = values[0]
                column_name = values[1]
                
                # Encontrar el índice del archivo
                for i, result in enumerate(self.ingestion.ingested_files):
                    if result.file_name == file_name:
                        pivot_columns[i] = column_name
                        break
            
            if not pivot_columns:
                messagebox.showwarning("Sin pivotes", "Selecciona al menos una columna pivote")
                return
            
            # Unificar datos
            unified_df = self.ingestion.step3_unify_with_pivot_columns(pivot_columns)
            
            # Mostrar resultados
            self.step3_results.config(state=tk.NORMAL)
            self.step3_results.delete(1.0, tk.END)
            
            if not unified_df.empty:
                self.step3_results.insert(tk.END, "✅ Datos unificados exitosamente\n")
                self.step3_results.insert(tk.END, f"Filas totales: {len(unified_df)}\n")
                self.step3_results.insert(tk.END, f"Columnas totales: {len(unified_df.columns)}\n")
                self.step3_results.insert(tk.END, f"Archivos unificados: {len(pivot_columns)}\n\n")
                
                # Mostrar columnas
                self.step3_results.insert(tk.END, "Columnas del DataFrame unificado:\n")
                for col in unified_df.columns:
                    if not col.startswith('_'):
                        self.step3_results.insert(tk.END, f"  - {col}\n")
                
                messagebox.showinfo("Unificación exitosa", "Los datos han sido unificados correctamente")
            else:
                self.step3_results.insert(tk.END, "❌ Error al unificar los datos")
                messagebox.showerror("Error", "No se pudieron unificar los datos")
        
        self.step3_results.config(state=tk.DISABLED)
    
    def load_to_application(self):
        """Carga los datos unificados en la aplicación principal"""
        unified_df = self.ingestion.get_unified_dataframe()
        
        if unified_df.empty:
            messagebox.showwarning("Sin datos", "No hay datos unificados para cargar")
            return
        
        # Crear metadata de resultados
        data_results = {
            'success': True,
            'message': f"Datos unificados: {len(unified_df)} filas, {len(unified_df.columns)} columnas",
            'files': [result.file_path for result in self.ingestion.ingested_files if result.success],
            'files_processed': len([r for r in self.ingestion.ingested_files if r.success]),
            'tables': [result.table_info for result in self.ingestion.ingested_files if result.success and result.table_info],
            'unified_df': unified_df
        }
        
        # Notificar a la ventana principal
        self.main_window.on_data_loaded(unified_df, data_results)
        
        messagebox.showinfo("Datos cargados", 
                           f"Datos cargados exitosamente en la aplicación:\n"
                           f"- Filas: {len(unified_df)}\n"
                           f"- Columnas: {len(unified_df.columns)}\n"
                           f"- Archivos: {len(data_results['files'])}")
    
    def direct_load_files(self):
        """Carga archivos directamente sin unificación"""
        try:
            # Obtener archivos exitosos
            successful_files = [r for r in self.ingestion.ingested_files if r.success]
            
            if not successful_files:
                messagebox.showwarning("Sin datos", "No hay archivos cargados para procesar")
                return
            
            if len(successful_files) == 1:
                # Un solo archivo - cargar directamente
                file_result = successful_files[0]
                unified_df = file_result.df.copy()
                
                # Agregar metadatos del archivo
                unified_df['_source_file'] = file_result.file_name
                unified_df['_file_index'] = 0
                unified_df['_file_format'] = file_result.file_format
                
                # Agregar información de tipos de columnas
                # Primero usar tipos corregidos si existen
                for col, col_type in file_result.corrected_types.items():
                    if col in unified_df.columns:
                        unified_df[f'_type_{col}'] = col_type
                
                # Luego agregar tipos detectados automáticamente para columnas sin corrección
                for col in unified_df.columns:
                    if not col.startswith('_'):  # Solo columnas de datos
                        type_col = f'_type_{col}'
                        if type_col not in unified_df.columns:
                            # Buscar tipo detectado en table_info.columns
                            col_type = 'qualitative'  # Por defecto
                            if file_result.table_info and file_result.table_info.columns:
                                for col_info in file_result.table_info.columns:
                                    if col_info.name == col:
                                        col_type = col_info.dtype
                                        break
                            unified_df[type_col] = col_type
                
                # Guardar en el sistema de ingesta
                self.ingestion.unified_df = unified_df
                
                # Crear metadata de resultados
                data_results = {
                    'success': True,
                    'message': f"Archivo único cargado: {len(unified_df)} filas, {len(unified_df.columns)} columnas",
                    'files': [file_result.file_path],
                    'files_processed': 1,
                    'tables': [file_result.table_info] if hasattr(file_result, 'table_info') else [],
                    'unified_df': unified_df
                }
                
                # Notificar a la ventana principal
                self.main_window.on_data_loaded(unified_df, data_results)
                
                messagebox.showinfo("Carga exitosa", 
                                   f"Archivo cargado exitosamente:\n"
                                   f"- Archivo: {file_result.file_name}\n"
                                   f"- Filas: {len(unified_df)}\n"
                                   f"- Columnas: {len(unified_df.columns)}")
                
            else:
                # Múltiples archivos - usar unificación automática
                # Primero ejecutar unificación automática
                self.unify_data()
                
                # Luego cargar en la aplicación
                self.load_to_application()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en carga directa:\n{str(e)}")
    
    def load_and_apply_direct(self):
        """Carga archivos y los aplica directamente a la aplicación"""
        try:
            # Verificar que hay archivos seleccionados
            if not self.selected_files:
                messagebox.showwarning("Sin archivos", "Selecciona al menos un archivo")
                return
            
            # Cargar archivos
            self.load_files()
            
            # Esperar un momento para que se procesen
            import time
            time.sleep(0.5)
            
            # Aplicar directamente
            self.direct_load_files()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en carga y aplicación directa:\n{str(e)}")
    
    def load_single_file_direct(self):
        """Carga un archivo único directamente"""
        try:
            # Obtener archivos exitosos
            successful_files = [r for r in self.ingestion.ingested_files if r.success]
            
            if not successful_files:
                messagebox.showwarning("Sin datos", "No hay archivos cargados para procesar")
                return
            
            if len(successful_files) == 1:
                # Usar el método de carga directa
                self.direct_load_files()
            else:
                messagebox.showinfo("Múltiples archivos", 
                                   f"Se detectaron {len(successful_files)} archivos. "
                                   "Usa 'Carga Directa' para procesar múltiples archivos.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en carga de archivo único:\n{str(e)}")
    
    def go_to_config(self):
        """Va a la sección de configuración"""
        # Expandir el frame de configuración
        self.config_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        # Actualizar la información de archivo único
        self.refresh_single_file_info()
    
    def refresh_single_file_info(self):
        """Actualiza la información del archivo único"""
        try:
            successful_files = [r for r in self.ingestion.ingested_files if r.success]
            
            if len(successful_files) == 1:
                file_result = successful_files[0]
                info_text = (f"Archivo único detectado: {file_result.file_name}\n"
                           f"Filas: {len(file_result.df)}, Columnas: {len(file_result.df.columns)}\n"
                           f"Se cargará directamente sin unificación.")
                self.single_file_info.config(text=info_text)
            else:
                info_text = f"Se detectaron {len(successful_files)} archivos. Configura la unificación abajo."
                self.single_file_info.config(text=info_text)
                
        except Exception:
            self.single_file_info.config(text="Error al obtener información de archivos")
    
    def prev_step(self):
        """Va al paso anterior"""
        current = self.steps_notebook.index(self.steps_notebook.select())
        if current > 0:
            self.steps_notebook.select(current - 1)
    
    def next_step(self):
        """Va al paso siguiente"""
        current = self.steps_notebook.index(self.steps_notebook.select())
        if current < self.steps_notebook.index('end') - 1:
            next_index = current + 1
            self.steps_notebook.select(next_index)
            
            # Si vamos al paso 3, refrescar la visualización de pivotes
            if next_index == 2:
                self.refresh_pivot_display()
    
    def reset_ingestion(self):
        """Reinicia el proceso de ingesta"""
        try:
            # Limpiar sistema de ingesta
            if hasattr(self, 'ingestion'):
                self.ingestion.reset()
                del self.ingestion
            
            # Recrear sistema de ingesta
            from utils.step_by_step_ingestion import StepByStepIngestion
            self.ingestion = StepByStepIngestion()
            
            # Limpiar variables
            self.selected_files = []
            if hasattr(self, 'current_file_index'):
                delattr(self, 'current_file_index')
            
            # Limpiar interfaces
            for item in self.files_tree.get_children():
                self.files_tree.delete(item)
            
            # Limpiar preview tree completamente
            for item in self.preview_tree.get_children():
                self.preview_tree.delete(item)
            # Limpiar columnas del preview tree
            self.preview_tree['columns'] = []
            self.preview_tree['show'] = 'tree'
            
            for item in self.pivot_tree.get_children():
                self.pivot_tree.delete(item)
            
            self.types_listbox.delete(0, tk.END)
            self.file_combo['values'] = []
            self.pivot_file_combo['values'] = []
            self.pivot_column_combo['values'] = []
            
            # Limpiar áreas de texto
            self.step1_results.config(state=tk.NORMAL)
            self.step1_results.delete(1.0, tk.END)
            self.step1_results.config(state=tk.DISABLED)
            
            self.step3_results.config(state=tk.NORMAL)
            self.step3_results.delete(1.0, tk.END)
            self.step3_results.config(state=tk.DISABLED)
            
            # Ocultar frames de configuración
            self.single_file_frame.pack_forget()
            self.config_frame.pack_forget()
            
            # Ir al paso 1
            self.steps_notebook.select(0)
            
            # Forzar garbage collection
            import gc
            gc.collect()
            
            messagebox.showinfo("Reiniciado", "El proceso de ingesta ha sido reiniciado")
            
        except Exception as e:
            print(f"[ERROR] Error en reset_ingestion: {str(e)}")
            messagebox.showerror("Error", f"Error al reiniciar: {str(e)}")
    
    def cleanup_memory(self):
        """Limpia la memoria antes de cerrar"""
        try:
            print("[CLEANUP] Limpiando memoria de pestaña de ingesta...")
            
            # Limpiar sistema de ingesta
            if hasattr(self, 'ingestion'):
                self.ingestion.reset()
                del self.ingestion
            
            # Reinicializar sistema de ingesta
            from utils.step_by_step_ingestion import StepByStepIngestion
            self.ingestion = StepByStepIngestion()
            
            # Limpiar variables
            self.selected_files = []
            if hasattr(self, 'current_file_index'):
                delattr(self, 'current_file_index')
            
            # Limpiar widgets de la interfaz
            if hasattr(self, 'files_tree'):
                for item in self.files_tree.get_children():
                    self.files_tree.delete(item)
            
            if hasattr(self, 'preview_tree'):
                for item in self.preview_tree.get_children():
                    self.preview_tree.delete(item)
                self.preview_tree['columns'] = []
                self.preview_tree['show'] = 'tree'
            
            if hasattr(self, 'pivot_tree'):
                for item in self.pivot_tree.get_children():
                    self.pivot_tree.delete(item)
            
            if hasattr(self, 'types_listbox'):
                self.types_listbox.delete(0, tk.END)
            
            # Limpiar combos
            if hasattr(self, 'file_combo'):
                self.file_combo['values'] = []
            
            if hasattr(self, 'pivot_file_combo'):
                self.pivot_file_combo['values'] = []
            
            if hasattr(self, 'pivot_column_combo'):
                self.pivot_column_combo['values'] = []
            
            # Limpiar áreas de texto
            if hasattr(self, 'step1_results'):
                self.step1_results.config(state=tk.NORMAL)
                self.step1_results.delete(1.0, tk.END)
                self.step1_results.config(state=tk.DISABLED)
            
            if hasattr(self, 'step3_results'):
                self.step3_results.config(state=tk.NORMAL)
                self.step3_results.delete(1.0, tk.END)
                self.step3_results.config(state=tk.DISABLED)
            
            # Limpiar referencias
            self.main_window = None
            
            # Forzar garbage collection múltiple
            import gc
            for _ in range(3):
                gc.collect()
            
            print("[CLEANUP] Memoria de pestaña de ingesta limpiada")
            
        except Exception as e:
            print(f"[ERROR] Error en cleanup_memory: {str(e)}")
