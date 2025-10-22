"""
Interfaz de usuario para el sistema de ingesta paso a paso
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
 
from pathlib import Path

from utils.step_by_step_ingestion import StepByStepIngestion


class StepByStepIngestionUI(ttk.Frame):
    """Interfaz para ingesta paso a paso"""
    
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.ingestion = StepByStepIngestion()
        self.current_step = 1
        self.selected_files = []
        
        self.create_widgets()
    
    def create_widgets(self):
        """Crea la interfaz de usuario"""
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(main_frame, text="Ingesta de Datos Paso a Paso", 
                               font=('Segoe UI', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Crear notebook para los pasos
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Paso 1: Selección y carga de archivos
        self.step1_frame = self.create_step1_frame()
        self.notebook.add(self.step1_frame, text="Paso 1: Cargar Archivos")
        
        # Paso 2: Corrección de tipos
        self.step2_frame = self.create_step2_frame()
        self.notebook.add(self.step2_frame, text="Paso 2: Corregir Tipos")
        
        # Paso 3: Unificación
        self.step3_frame = self.create_step3_frame()
        self.notebook.add(self.step3_frame, text="Paso 3: Unificar Datos")
        
        # Botones de navegación
        self.create_navigation_buttons(main_frame)
    
    def create_step1_frame(self):
        """Crea el frame para el paso 1"""
        frame = ttk.Frame(self.notebook)
        
        # Instrucciones
        instructions = ttk.Label(frame, 
            text="Selecciona los archivos de datos que deseas cargar.\n"
                 "El sistema detectará automáticamente el formato y tipos de datos.",
            font=('Segoe UI', 10))
        instructions.pack(pady=10)
        
        # Botón para seleccionar archivos
        select_btn = ttk.Button(frame, text="📁 Seleccionar Archivos", 
                               command=self.select_files)
        select_btn.pack(pady=10)
        
        # Lista de archivos seleccionados
        files_frame = ttk.LabelFrame(frame, text="Archivos Seleccionados", padding=10)
        files_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.files_listbox = tk.Listbox(files_frame, height=8)
        self.files_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Botón para cargar archivos
        load_btn = ttk.Button(frame, text="🔄 Cargar Archivos", 
                             command=self.load_files)
        load_btn.pack(pady=10)
        
        # Área de resultados
        self.step1_results = tk.Text(frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.step1_results.pack(fill=tk.BOTH, expand=True, pady=10)
        
        return frame
    
    def create_step2_frame(self):
        """Crea el frame para el paso 2"""
        frame = ttk.Frame(self.notebook)
        
        # Instrucciones
        instructions = ttk.Label(frame, 
            text="Revisa y corrige los tipos de datos detectados automáticamente.\n"
                 "Selecciona un archivo y ajusta los tipos de sus columnas.",
            font=('Segoe UI', 10))
        instructions.pack(pady=10)
        
        # Frame para selección de archivo
        file_selection_frame = ttk.Frame(frame)
        file_selection_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(file_selection_frame, text="Archivo:").pack(side=tk.LEFT, padx=(0, 10))
        self.file_combo = ttk.Combobox(file_selection_frame, state='readonly', width=30)
        self.file_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.file_combo.bind('<<ComboboxSelected>>', self.on_file_selected)
        
        refresh_btn = ttk.Button(file_selection_frame, text="🔄 Actualizar", 
                                command=self.refresh_file_list)
        refresh_btn.pack(side=tk.LEFT)
        
        # Frame para corrección de tipos
        types_frame = ttk.LabelFrame(frame, text="Tipos de Columnas", padding=10)
        types_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Lista de columnas con tipos
        self.types_listbox = tk.Listbox(types_frame, height=15)
        self.types_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar para la lista
        types_scrollbar = ttk.Scrollbar(types_frame, orient=tk.VERTICAL, command=self.types_listbox.yview)
        types_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.types_listbox.config(yscrollcommand=types_scrollbar.set)
        
        # Frame para controles de corrección
        controls_frame = ttk.Frame(types_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        ttk.Label(controls_frame, text="Nuevo tipo:").pack(pady=(0, 5))
        self.type_combo = ttk.Combobox(controls_frame, values=['quantitative', 'qualitative', 'datetime'], 
                                      state='readonly', width=15)
        self.type_combo.pack(pady=(0, 10))
        
        apply_btn = ttk.Button(controls_frame, text="Aplicar", command=self.apply_type_correction)
        apply_btn.pack(pady=(0, 10))
        
        # Botón para continuar al paso 3
        continue_btn = ttk.Button(frame, text="➡️ Continuar al Paso 3", 
                                 command=self.go_to_step3)
        continue_btn.pack(pady=10)
        
        return frame
    
    def create_step3_frame(self):
        """Crea el frame para el paso 3"""
        frame = ttk.Frame(self.notebook)
        
        # Instrucciones
        instructions = ttk.Label(frame, 
            text="Selecciona las columnas que servirán como pivote para unificar los archivos.\n"
                 "Estas columnas deben contener valores comunes entre los archivos.",
            font=('Segoe UI', 10))
        instructions.pack(pady=10)
        
        # Frame para selección de columnas pivote
        pivot_frame = ttk.LabelFrame(frame, text="Columnas Pivote", padding=10)
        pivot_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Lista de archivos con sus columnas
        self.pivot_tree = ttk.Treeview(pivot_frame, columns=('file', 'pivot_column'), show='tree headings', height=10)
        self.pivot_tree.heading('#0', text='Archivo')
        self.pivot_tree.heading('file', text='Archivo')
        self.pivot_tree.heading('pivot_column', text='Columna Pivote')
        
        self.pivot_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar para el tree
        pivot_scrollbar = ttk.Scrollbar(pivot_frame, orient=tk.VERTICAL, command=self.pivot_tree.yview)
        pivot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.pivot_tree.config(yscrollcommand=pivot_scrollbar.set)
        
        # Frame para controles de selección
        pivot_controls_frame = ttk.Frame(frame)
        pivot_controls_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(pivot_controls_frame, text="Archivo:").pack(side=tk.LEFT, padx=(0, 10))
        self.pivot_file_combo = ttk.Combobox(pivot_controls_frame, state='readonly', width=20)
        self.pivot_file_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.pivot_file_combo.bind('<<ComboboxSelected>>', self.on_pivot_file_selected)
        
        ttk.Label(pivot_controls_frame, text="Columna:").pack(side=tk.LEFT, padx=(0, 10))
        self.pivot_column_combo = ttk.Combobox(pivot_controls_frame, state='readonly', width=20)
        self.pivot_column_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        add_pivot_btn = ttk.Button(pivot_controls_frame, text="➕ Agregar Pivote", 
                                  command=self.add_pivot_column)
        add_pivot_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        remove_pivot_btn = ttk.Button(pivot_controls_frame, text="➖ Quitar Pivote", 
                                     command=self.remove_pivot_column)
        remove_pivot_btn.pack(side=tk.LEFT)
        
        # Botón para unificar
        unify_btn = ttk.Button(frame, text="🔗 Unificar Datos", 
                              command=self.unify_data)
        unify_btn.pack(pady=10)
        
        # Área de resultados
        self.step3_results = tk.Text(frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.step3_results.pack(fill=tk.BOTH, expand=True, pady=10)
        
        return frame
    
    def create_navigation_buttons(self, parent):
        """Crea los botones de navegación"""
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, pady=10)
        
        prev_btn = ttk.Button(nav_frame, text="⬅️ Anterior", command=self.prev_step)
        prev_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        next_btn = ttk.Button(nav_frame, text="Siguiente ➡️", command=self.next_step)
        next_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        finish_btn = ttk.Button(nav_frame, text="✅ Finalizar", command=self.finish_ingestion)
        finish_btn.pack(side=tk.RIGHT)
        
        self.nav_buttons = {
            'prev': prev_btn,
            'next': next_btn,
            'finish': finish_btn
        }
    
    def select_files(self):
        """Selecciona archivos para cargar"""
        from utils.multi_format_loader import MultiFormatLoader
        loader = MultiFormatLoader()
        supported_formats = loader.get_supported_formats()
        
        # Crear lista de tipos de archivo para el diálogo
        filetypes = []
        for ext, description in supported_formats.items():
            filetypes.append((description, f"*{ext}"))
        filetypes.append(("Todos los archivos soportados", "*.csv;*.tsv;*.xlsx;*.xls;*.json;*.parquet;*.pickle;*.feather;*.hdf5;*.xml;*.html;*.txt"))
        filetypes.append(("Todos los archivos", "*.*"))
        
        files = filedialog.askopenfilenames(
            title="Seleccionar archivos de datos",
            filetypes=filetypes
        )
        
        if files:
            self.selected_files = list(files)
            self.update_files_list()
    
    def update_files_list(self):
        """Actualiza la lista de archivos seleccionados"""
        self.files_listbox.delete(0, tk.END)
        for file_path in self.selected_files:
            self.files_listbox.insert(tk.END, Path(file_path).name)
    
    def load_files(self):
        """Carga los archivos seleccionados"""
        if not self.selected_files:
            messagebox.showwarning("Sin archivos", "Selecciona al menos un archivo")
            return
        
        # Limpiar resultados anteriores
        self.ingestion.reset()
        
        # Cargar archivos
        results = self.ingestion.step1_ingest_individual_files(self.selected_files)
        
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
            # Actualizar lista de archivos en paso 2
            self.refresh_file_list()
            # Habilitar paso 2
            self.notebook.tab(1, state='normal')
            messagebox.showinfo("Carga exitosa", f"Se cargaron {success_count} archivos exitosamente")
        else:
            messagebox.showerror("Error", "No se pudo cargar ningún archivo")
    
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
            self.update_column_types_display()
        except (ValueError, IndexError):
            pass
    
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
        else:
            messagebox.showerror("Error", "No se pudo aplicar la corrección")
    
    def go_to_step3(self):
        """Va al paso 3"""
        self.notebook.select(2)
        self.refresh_pivot_display()
    
    def refresh_pivot_display(self):
        """Actualiza la visualización de columnas pivote"""
        # Limpiar tree
        for item in self.pivot_tree.get_children():
            self.pivot_tree.delete(item)
        
        # Actualizar combo de archivos
        self.pivot_file_combo['values'] = []
        for i, result in enumerate(self.ingestion.ingested_files):
            if result.success:
                self.pivot_file_combo['values'] = list(self.pivot_file_combo['values']) + [f"{i}: {result.file_name}"]
        
        if self.pivot_file_combo['values']:
            self.pivot_file_combo.current(0)
            self.on_pivot_file_selected(None)
    
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
        except (ValueError, IndexError):
            pass
    
    def add_pivot_column(self):
        """Agrega una columna pivote"""
        file_selection = self.pivot_file_combo.get()
        column = self.pivot_column_combo.get()
        
        if not file_selection or not column:
            messagebox.showwarning("Sin selección", "Selecciona archivo y columna")
            return
        
        try:
            file_index = int(file_selection.split(':')[0])
            
            # Agregar al tree
            file_name = self.ingestion.ingested_files[file_index].file_name
            self.pivot_tree.insert('', 'end', text=file_name, values=(file_name, column))
            
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
        """Unifica los datos usando las columnas pivote"""
        # Obtener columnas pivote del tree
        pivot_columns = {}
        for item in self.pivot_tree.get_children():
            values = self.pivot_tree.item(item, 'values')
            file_name = values[0]
            
            # Encontrar el índice del archivo
            for i, result in enumerate(self.ingestion.ingested_files):
                if result.file_name == file_name:
                    pivot_columns[i] = values[1]
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
    
    def finish_ingestion(self):
        """Finaliza la ingesta y pasa los datos a la aplicación principal"""
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
        
        messagebox.showinfo("Ingesta completada", 
                           f"Datos cargados exitosamente:\n"
                           f"- Filas: {len(unified_df)}\n"
                           f"- Columnas: {len(unified_df.columns)}\n"
                           f"- Archivos: {len(data_results['files'])}")
    
    def prev_step(self):
        """Va al paso anterior"""
        current = self.notebook.index(self.notebook.select())
        if current > 0:
            self.notebook.select(current - 1)
    
    def next_step(self):
        """Va al paso siguiente"""
        current = self.notebook.index(self.notebook.select())
        if current < self.notebook.index('end') - 1:
            self.notebook.select(current + 1)
