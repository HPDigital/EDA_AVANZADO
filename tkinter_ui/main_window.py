"""
Ventana principal de la aplicación Tkinter (UTF-8)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

from .data_exploration import DataExplorationFrame
from .univariate_analysis import UnivariateAnalysisFrame
from .bivariate_analysis import BivariateAnalysisFrame
from .multivariate_analysis import MultivariateAnalysisFrame
from .timeseries_analysis import TimeSeriesAnalysisFrame
from .spatial_analysis import SpatialAnalysisFrame
from .ml_analysis import MachineLearningFrame
from .reports import ReportsFrame


class MainWindow:
    """Ventana principal de la aplicación"""

    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.current_df = None
        self.current_data = None

        # Interfaz
        self.create_menu()
        self.create_header()
        self.create_sidebar()

        # Separador visual
        ttk.Separator(self.root, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=2)

        self.create_main_content()
        self.create_statusbar()

    # --------- Menú / Header ---------
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Seleccionar directorio...", command=self.app.select_data_directory)
        file_menu.add_command(label="Seleccionar archivos...", command=self.app.select_data_files)
        file_menu.add_command(label="Ingesta paso a paso...", command=self.app.open_step_by_step_ingestion)
        file_menu.add_command(label="Cargar datos", command=self.app.load_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exportar datos actuales...", command=self.export_current_data)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.app.quit_app)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Análisis", menu=analysis_menu)
        analysis_menu.add_command(label="Exploración de datos", command=lambda: self.show_tab(0))
        analysis_menu.add_command(label="Univariado", command=lambda: self.show_tab(1))
        analysis_menu.add_command(label="Bivariado", command=lambda: self.show_tab(2))
        analysis_menu.add_command(label="Multivariado", command=lambda: self.show_tab(3))
        analysis_menu.add_command(label="Series temporales", command=lambda: self.show_tab(4))
        analysis_menu.add_command(label="Análisis espacial", command=lambda: self.show_tab(5))
        analysis_menu.add_command(label="Machine Learning", command=lambda: self.show_tab(6))
        analysis_menu.add_command(label="Reportes", command=lambda: self.show_tab(7))

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Herramientas", menu=tools_menu)
        tools_menu.add_command(label="Limpiar caché", command=self.clear_cache)
        tools_menu.add_command(label="Verificar dependencias", command=self.check_dependencies)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ayuda", menu=help_menu)
        help_menu.add_command(label="Documentación", command=self.show_documentation)
        help_menu.add_command(label="Acerca de...", command=self.app.show_about)

    def create_header(self):
        header = ttk.Frame(self.root, style="Title.TFrame", height=60)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)

        ttk.Label(header, text="CAF Dashboard - EDA Avanzado", style="Title.TLabel").pack(
            side=tk.LEFT, padx=20, pady=10
        )
        ttk.Button(header, text="Recargar datos", command=self.open_ingestion_tab).pack(
            side=tk.RIGHT, padx=20, pady=10
        )

    # --------- Sidebar ---------
    def create_sidebar(self):
        container = ttk.Frame(self.root, width=300)
        container.pack(side=tk.LEFT, fill=tk.Y)
        container.pack_propagate(False)
        container.configure(width=300)

        canvas = tk.Canvas(container, bg="#f0f0f0")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.sidebar_frame = ttk.Frame(canvas)
        self.sidebar_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.sidebar_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.create_sidebar_content()

    def create_sidebar_content(self):
        data_section = ttk.LabelFrame(self.sidebar_frame, text="Fuente de datos", padding=10)
        data_section.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(data_section, text="Modo de ingesta:").pack(anchor=tk.W)
        self.ingestion_mode_var = tk.StringVar(value="automatic")
        ttk.Radiobutton(
            data_section, text="Automática (recomendado)", variable=self.ingestion_mode_var,
            value="automatic", command=lambda: self.app.toggle_ingestion_mode("automatic")
        ).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(
            data_section, text="Manual (CAF específico)", variable=self.ingestion_mode_var,
            value="manual", command=lambda: self.app.toggle_ingestion_mode("manual")
        ).pack(anchor=tk.W, pady=2)

        ttk.Label(data_section, text="Directorio:").pack(anchor=tk.W, pady=(10, 2))
        self.data_dir_label = ttk.Label(data_section, text=self.app.data_dir, wraplength=250, foreground="#666")
        self.data_dir_label.pack(anchor=tk.W)

        info_section = ttk.LabelFrame(self.sidebar_frame, text="Información", padding=10)
        info_section.pack(fill=tk.X, padx=10, pady=10)
        self.info_section = info_section
        self.info_text = tk.Text(info_section, height=8, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.update_info_text("No hay datos cargados")

        dep_section = ttk.LabelFrame(self.sidebar_frame, text="Dependencias", padding=10)
        dep_section.pack(fill=tk.X, padx=10, pady=10)
        dep_text = tk.Text(dep_section, height=10, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 8))
        dep_text.pack(fill=tk.BOTH, expand=True)
        from config import DEPENDENCIES
        dep_text.config(state=tk.NORMAL)
        for dep, available in DEPENDENCIES.items():
            icon = "✔" if available else "✗"
            dep_text.insert(tk.END, f"{icon} {dep}\n")
        dep_text.config(state=tk.DISABLED)

    # --------- Contenido principal ---------
    def create_main_content(self):
        content = ttk.Frame(self.root)
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(content)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.create_tabs()
        self.add_ingestion_tab()

    def create_tabs(self):
        self.exploration_frame = DataExplorationFrame(self.notebook, self)
        self.notebook.add(self.exploration_frame, text="Exploración")

        self.univariate_frame = UnivariateAnalysisFrame(self.notebook, self)
        self.notebook.add(self.univariate_frame, text="Univariado")

        self.bivariate_frame = BivariateAnalysisFrame(self.notebook, self)
        self.notebook.add(self.bivariate_frame, text="Bivariado")

        self.multivariate_frame = MultivariateAnalysisFrame(self.notebook, self)
        self.notebook.add(self.multivariate_frame, text="Multivariado")

        self.timeseries_frame = TimeSeriesAnalysisFrame(self.notebook, self)
        self.notebook.add(self.timeseries_frame, text="Series temporales")

    def add_ingestion_tab(self):
        from tkinter_ui.ingestion_tab import IngestionTab
        self.ingestion_frame = IngestionTab(self.notebook, self)
        self.notebook.insert(0, self.ingestion_frame, text="Ingesta")

        self.spatial_frame = SpatialAnalysisFrame(self.notebook, self)
        self.notebook.add(self.spatial_frame, text="Análisis espacial")

        self.ml_frame = MachineLearningFrame(self.notebook, self)
        self.notebook.add(self.ml_frame, text="Machine Learning")

        self.reports_frame = ReportsFrame(self.notebook, self)
        self.notebook.add(self.reports_frame, text="Reportes")

    def create_statusbar(self):
        self.statusbar = ttk.Label(self.root, text="Listo", relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    # --------- Callbacks y utilidades ---------
    def open_ingestion_tab(self):
        self.notebook.select(0)

    def on_data_loaded(self, df, data_results):
        self.cleanup_previous_data()
        self.current_df = df
        self.current_data = data_results
        if isinstance(df, pd.DataFrame):
            info_str = (
                f"Filas: {len(df)}\n"
                f"Columnas: {len(df.columns)}\n"
                f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n"
            )
            if isinstance(data_results, dict) and "files_processed" in data_results:
                info_str += f"Archivos: {data_results['files_processed']}\n"
            self.update_info_text(info_str)
        # Propagar a pestañas
        self.exploration_frame.update_data(df, data_results)
        self.univariate_frame.update_data(df, data_results)
        self.bivariate_frame.update_data(df, data_results)
        self.multivariate_frame.update_data(df, data_results)
        self.timeseries_frame.update_data(df, data_results)
        self.spatial_frame.update_data(df, data_results)
        self.ml_frame.update_data(df, data_results)
        self.reports_frame.update_data(df, data_results)
        self.update_status("Datos cargados correctamente")

    def on_data_type_changed(self, column_name, old_type, new_type):
        if hasattr(self, "multivariate_frame"):
            self.multivariate_frame.refresh_variable_list()
        if hasattr(self, "univariate_frame"):
            self.univariate_frame.update_variable_list()
        if hasattr(self, "bivariate_frame"):
            self.bivariate_frame.update_variable_lists()
        if hasattr(self, "timeseries_frame"):
            self.timeseries_frame.detect_time_column()
        self.update_status(f"Tipo de columna '{column_name}' cambiado de {old_type} a {new_type}")

    def update_info_text(self, text):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, text)
        self.info_text.config(state=tk.DISABLED)

    def update_data_dir_label(self, directory):
        self.data_dir_label.config(text=directory)

    def update_status(self, message):
        self.statusbar.config(text=message)

    def show_tab(self, index):
        self.notebook.select(index)

    def export_current_data(self):
        if self.current_df is not None:
            self.app.export_data(self.current_df, "caf_data")
        else:
            messagebox.showwarning("Sin datos", "No hay datos cargados para exportar")

    def clear_cache(self):
        messagebox.showinfo("Caché", "Funcionalidad de limpieza de caché")

    def check_dependencies(self):
        from config import DEPENDENCIES
        available = sum(1 for v in DEPENDENCIES.values() if v)
        total = len(DEPENDENCIES)
        msg = f"Dependencias disponibles: {available}/{total}\n\n"
        for dep, avail in DEPENDENCIES.items():
            status = "✔" if avail else "✗"
            msg += f"{status} {dep}\n"
        messagebox.showinfo("Dependencias", msg)

    def show_documentation(self):
        messagebox.showinfo(
            "Documentación",
            "La documentación completa está disponible en:\n\n"
            "- README.md\n"
            "- SETUP.md",
        )

    def show_about(self):
        about_text = (
            "CAF Dashboard - EDA Avanzado\n"
            "Versión 2.0\n\n"
            "Aplicación de análisis exploratorio de datos\n"
            "con interfaz moderna y capacidades estadísticas completas.\n\n"
            "Desarrollado para el análisis de datos CAF\n"
            "con soporte para múltiples formatos y análisis avanzados.\n\n"
            "© 2025 CAF Dashboard Team"
        )
        messagebox.showinfo("Acerca de CAF Dashboard", about_text)

    def select_data_directory(self):
        from tkinter import filedialog
        directory = filedialog.askdirectory(
            title="Seleccionar directorio de datos",
            initialdir=self.app.data_dir if hasattr(self.app, "data_dir") else ".",
        )
        if directory:
            self.app.data_dir = directory
            self.update_data_dir_label(directory)
            messagebox.showinfo("Directorio seleccionado", f"Directorio de datos actualizado:\n{directory}")

    def cleanup_previous_data(self):
        try:
            if hasattr(self, "current_df") and self.current_df is not None:
                del self.current_df
                self.current_df = None
            if hasattr(self, "current_data"):
                del self.current_data
                self.current_data = None
            import gc
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error en cleanup_previous_data: {e}")

    def cleanup_all_memory(self):
        try:
            self.cleanup_previous_data()
            if hasattr(self, "ingestion_frame"):
                if hasattr(self.ingestion_frame, "cleanup_memory"):
                    self.ingestion_frame.cleanup_memory()
                del self.ingestion_frame
                self.ingestion_frame = None
            frames_to_clean = [
                "exploration_frame",
                "univariate_frame",
                "bivariate_frame",
                "multivariate_frame",
                "timeseries_frame",
                "spatial_frame",
                "ml_frame",
            ]
            for name in frames_to_clean:
                if hasattr(self, name):
                    frame = getattr(self, name)
                    if hasattr(frame, "cleanup_memory"):
                        frame.cleanup_memory()
                    del frame
            import gc
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error en cleanup_all_memory: {e}")

