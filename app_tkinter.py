"""
CAF Dashboard - Aplicación principal con Tkinter
Análisis estadístico completo con interfaz desktop moderna
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Configurar matplotlib para Tkinter ANTES de importar otros módulos
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Importar módulos de configuración
from config import (
    DEFAULT_DATA_DIR, THEME_TEMPLATE, 
    CAF_AGE_FILE, CAF_FAM_FILE, DEPENDENCIES
)

# Importar utilidades
from utils.data_loader import CAFDataLoader
from utils.smart_ingestion import SmartCSVIngestion

# Importar ventanas de análisis
from tkinter_ui.main_window import MainWindow

warnings.filterwarnings("ignore")


class CAFDashboardApp:
    """Aplicación principal del dashboard CAF con Tkinter"""
    
    def __init__(self):
        """Inicializa la aplicación"""
        self.root = tk.Tk()
        self.root.title("CAF Dashboard - EDA Avanzado")
        self.root.geometry("1600x900")
        self.root.minsize(1200, 700)
        
        # Configurar estilo
        self.setup_style()
        
        # Variables de aplicación
        self.data_dir = DEFAULT_DATA_DIR
        self.ingestion_mode = "automatic"  # automatic o manual
        self.current_data = None
        self.unified_df = None
        
        # Crear ventana principal
        self.main_window = MainWindow(self.root, self)
        
        # Cargar datos al inicio si existen
        self.check_initial_data()
    
    def setup_style(self):
        """Configura el estilo visual de la aplicación"""
        style = ttk.Style()
        
        # Usar tema moderno
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'vista' in available_themes:
            style.theme_use('vista')
        
        # Colores personalizados
        bg_color = '#f0f0f0'
        fg_color = '#333333'
        accent_color = '#4a90e2'
        
        # Configurar estilos
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=fg_color)
        style.configure('TButton', padding=6)
        style.configure('Accent.TButton', foreground='white', background=accent_color)
        
        # Frame de título
        style.configure('Title.TFrame', background='#2c3e50')
        style.configure('Title.TLabel', background='#2c3e50', foreground='white', 
                       font=('Segoe UI', 16, 'bold'))
        
        # Notebook (tabs)
        style.configure('TNotebook', background=bg_color)
        style.configure('TNotebook.Tab', padding=[12, 6])
    
    def check_initial_data(self):
        """Verifica si hay datos disponibles al inicio"""
        if os.path.exists(self.data_dir):
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            if csv_files:
                # Mostrar diálogo para cargar datos
                response = messagebox.askyesno(
                    "Datos disponibles",
                    f"Se encontraron {len(csv_files)} archivo(s) CSV.\n¿Desea cargar los datos ahora?"
                )
                if response:
                    self.load_data()
    
    def load_data(self):
        """Carga los datos según el modo seleccionado"""
        try:
            if self.ingestion_mode == "automatic":
                self.load_data_automatic()
            else:
                self.load_data_manual()
        except Exception as e:
            messagebox.showerror("Error de carga", f"Error al cargar datos:\n{str(e)}")
    
    def load_data_automatic(self):
        """Carga datos usando ingesta inteligente"""
        try:
            ingestion = SmartCSVIngestion()
            results = ingestion.process_directory(self.data_dir)
            
            if results['success']:
                self.unified_df = results['unified_df']
                self.current_data = results
                
                messagebox.showinfo(
                    "Carga exitosa",
                    f"Datos cargados correctamente:\n"
                    f"- Archivos procesados: {results['files_processed']}\n"
                    f"- Filas totales: {len(self.unified_df)}\n"
                    f"- Columnas totales: {len(self.unified_df.columns)}"
                )
                
                # Actualizar interfaz
                self.main_window.on_data_loaded(self.unified_df, self.current_data)
            else:
                messagebox.showerror("Error", results['message'])
        
        except Exception as e:
            messagebox.showerror("Error", f"Error en la ingesta automática:\n{str(e)}")
    
    def load_data_manual(self):
        """Carga datos CAF usando el método manual"""
        age_path = os.path.join(self.data_dir, CAF_AGE_FILE)
        fam_path = os.path.join(self.data_dir, CAF_FAM_FILE)
        
        if not os.path.exists(age_path) or not os.path.exists(fam_path):
            messagebox.showerror(
                "Archivos no encontrados",
                f"No se encontraron los archivos en:\n{self.data_dir}\n\n"
                f"Archivos requeridos:\n- {CAF_AGE_FILE}\n- {CAF_FAM_FILE}"
            )
            return
        
        try:
            loader = CAFDataLoader()
            combined_df = loader.load_and_merge_caf_data(self.data_dir)
            long_df = loader.caf_to_long_format(combined_df)
            indicator_panel = loader.create_indicator_panel(long_df)
            
            self.current_data = (combined_df, long_df, indicator_panel)
            self.unified_df = indicator_panel
            
            messagebox.showinfo(
                "Carga exitosa",
                f"Datos CAF cargados correctamente:\n"
                f"- Filas: {len(indicator_panel)}\n"
                f"- Columnas: {len(indicator_panel.columns)}"
            )
            
            # Actualizar interfaz
            self.main_window.on_data_loaded(self.unified_df, self.current_data)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar datos CAF:\n{str(e)}")
    
    def select_data_directory(self):
        """Permite seleccionar un directorio de datos"""
        directory = filedialog.askdirectory(
            title="Seleccionar directorio de datos",
            initialdir=self.data_dir
        )
        
        if directory:
            self.data_dir = directory
            self.main_window.update_data_dir_label(directory)
            
            # Buscar archivos de datos soportados
            from utils.multi_format_loader import MultiFormatLoader
            loader = MultiFormatLoader()
            supported_formats = loader.get_supported_formats()
            
            data_files = []
            for ext in supported_formats.keys():
                data_files.extend([f for f in os.listdir(directory) if f.endswith(ext)])
            
            if data_files:
                response = messagebox.askyesno(
                    "Cargar datos",
                    f"Se encontraron {len(data_files)} archivo(s) de datos soportados.\n¿Desea cargarlos ahora?"
                )
                if response:
                    self.load_data()
    
    def select_data_files(self):
        """Permite seleccionar archivos de datos individuales"""
        from utils.multi_format_loader import MultiFormatLoader
        loader = MultiFormatLoader()
        supported_formats = loader.get_supported_formats()
        
        # Crear lista de tipos de archivo para el diálogo
        filetypes = []
        # Poner "Archivos soportados" primero para que sea el filtro por defecto
        filetypes.append(("Archivos soportados", "*.csv;*.xlsx"))
        filetypes.append(("Todos los archivos", "*.*"))
        # Agregar formatos individuales después
        for ext, description in supported_formats.items():
            filetypes.append((description, f"*{ext}"))
        
        # Usar el directorio de datos como directorio inicial
        initial_dir = self.data_dir if self.data_dir and os.path.exists(self.data_dir) else os.getcwd()
        
        files = filedialog.askopenfilenames(
            title="Seleccionar archivos de datos",
            filetypes=filetypes,
            initialdir=initial_dir
        )
        
        if files:
            # Cargar archivos seleccionados
            self.load_selected_files(files)
        else:
            messagebox.showinfo("Sin selección", "No se seleccionaron archivos")
    
    def load_selected_files(self, file_paths):
        """Carga archivos seleccionados individualmente"""
        try:
            from utils.multi_format_loader import MultiFormatLoader
            from utils.smart_ingestion import SmartCSVIngestion
            
            # Crear cargador multi-formato
            loader = MultiFormatLoader()
            
            print(f"DEBUG: Cargando archivos seleccionados: {len(file_paths)}")
            for file_path in file_paths:
                print(f"DEBUG:   - {Path(file_path).name}")
            
            # Procesar cada archivo
            all_tables = []
            for file_path in file_paths:
                try:
                    # Cargar archivo
                    df, metadata = loader.load_file(file_path)
                    
                    if df is not None and not df.empty:
                        # Usar SmartCSVIngestion para procesar
                        ingestion = SmartCSVIngestion()
                        tables = ingestion.process_data_file(file_path)
                        
                        print(f"DEBUG: Tablas encontradas en {Path(file_path).name}: {len(tables)}")
                        all_tables.extend(tables)
                        
                        print(f"✓ Archivo cargado: {Path(file_path).name} ({len(df)} filas, {len(df.columns)} columnas)")
                    else:
                        print(f"✗ Error: Archivo vacío o inválido: {Path(file_path).name}")
                        
                except Exception as e:
                    print(f"✗ Error cargando {Path(file_path).name}: {str(e)}")
                    messagebox.showerror("Error", f"Error al cargar {Path(file_path).name}:\n{str(e)}")
            
            print(f"DEBUG: Total de tablas procesadas: {len(all_tables)}")
            
            if all_tables:
                # Unificar tablas
                ingestion = SmartCSVIngestion()
                print(f"DEBUG: Unificando {len(all_tables)} tablas...")
                self.unified_df = ingestion.unify_tables_by_time(all_tables)
                
                # Crear metadata de resultados
                self.current_data = {
                    'success': True,
                    'message': f"Archivos cargados: {len(file_paths)}",
                    'files': file_paths,
                    'files_processed': len(file_paths),
                    'tables': all_tables,
                    'unified_df': self.unified_df
                }
                
                # Notificar a la ventana principal
                self.main_window.on_data_loaded(self.unified_df, self.current_data)
                
                messagebox.showinfo("Carga exitosa", 
                                  f"Se cargaron {len(file_paths)} archivo(s) exitosamente.\n"
                                  f"Total de filas: {len(self.unified_df)}")
            else:
                messagebox.showwarning("Sin datos", "No se pudieron cargar datos válidos de los archivos seleccionados.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar archivos:\n{str(e)}")
    
    def open_step_by_step_ingestion(self):
        """Abre la ventana de ingesta paso a paso"""
        from tkinter_ui.step_by_step_ingestion_ui import StepByStepIngestionUI
        
        # Crear ventana modal
        step_window = tk.Toplevel(self.root)
        step_window.title("Ingesta de Datos Paso a Paso")
        step_window.geometry("1000x700")
        step_window.transient(self.root)
        step_window.grab_set()
        
        # Centrar ventana
        step_window.update_idletasks()
        x = (step_window.winfo_screenwidth() // 2) - (1000 // 2)
        y = (step_window.winfo_screenheight() // 2) - (700 // 2)
        step_window.geometry(f"1000x700+{x}+{y}")
        
        # Crear frame de ingesta paso a paso
        ingestion_frame = StepByStepIngestionUI(step_window, self.main_window)
        ingestion_frame.pack(fill=tk.BOTH, expand=True)
        
        # Botón de cerrar
        close_btn = ttk.Button(step_window, text="Cerrar", command=step_window.destroy)
        close_btn.pack(pady=10)
    
    def toggle_ingestion_mode(self, mode):
        """Cambia el modo de ingesta"""
        self.ingestion_mode = mode
    
    def export_data(self, df, default_name="export"):
        """Exporta un DataFrame a CSV"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{default_name}.csv"
        )
        
        if file_path:
            try:
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Exportación exitosa", f"Datos exportados a:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error de exportación", f"Error al exportar:\n{str(e)}")
    
    def show_about(self):
        """Muestra información sobre la aplicación"""
        about_text = f"""
CAF Dashboard - EDA Avanzado
Versión 2.0

Aplicación de análisis estadístico completo para datos CAF.

Capacidades:
✓ Ingesta inteligente de datos
✓ Análisis univariado (completo)
✓ Análisis bivariado (todas las combinaciones)
✓ Análisis multivariado (PCA, Clustering, Outliers)
✓ Series temporales
✓ Análisis espacial
✓ Machine Learning
✓ Generación de reportes

Dependencias disponibles:
"""
        
        for dep, available in DEPENDENCIES.items():
            status = "✓" if available else "✗"
            about_text += f"\n{status} {dep}"
        
        messagebox.showinfo("Acerca de", about_text)
    
    def run(self):
        """Inicia la aplicación"""
        # Centrar ventana en pantalla
        self.center_window()
        
        # Iniciar loop principal
        self.root.mainloop()
    
    def center_window(self):
        """Centra la ventana en la pantalla"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def quit_app(self):
        """Cierra la aplicación"""
        response = messagebox.askyesno(
            "Salir", 
            "¿Está seguro que desea salir de la aplicación?"
        )
        if response:
            try:
                # Limpiar memoria antes de cerrar
                self.cleanup_all_memory()
                
                # Cerrar ventana
                self.root.quit()
                
            except Exception as e:
                print(f"[ERROR] Error al cerrar aplicación: {str(e)}")
                # Forzar cierre
                self.root.quit()
    
    def cleanup_all_memory(self):
        """Limpia toda la memoria de la aplicación"""
        try:
            # Limpiar ventana principal
            if hasattr(self, 'main_window') and hasattr(self.main_window, 'cleanup_all_memory'):
                self.main_window.cleanup_all_memory()
            
            # Limpiar variables de la aplicación
            if hasattr(self, 'current_data'):
                del self.current_data
                self.current_data = None
            
            if hasattr(self, 'unified_df'):
                del self.unified_df
                self.unified_df = None
            
            if hasattr(self, 'data_dir'):
                self.data_dir = None
            
            # Limpiar referencias
            self.main_window = None
            
            # Forzar garbage collection
            import gc
            gc.collect()
            
            print("[CLEANUP] Memoria de aplicación limpiada")
            
        except Exception as e:
            print(f"[ERROR] Error en cleanup_all_memory: {str(e)}")


def main():
    """Función principal"""
    try:
        # Limpiar memoria al inicio
        import gc
        gc.collect()
        
        # Crear y ejecutar aplicación
        app = CAFDashboardApp()
        app.run()
        
    except Exception as e:
        print(f"[ERROR] Error en main: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Limpieza final
        try:
            import gc
            gc.collect()
        except:
            pass


if __name__ == "__main__":
    main()

