"""
Frame de generación de reportes
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from datetime import datetime


class ReportsFrame(ttk.Frame):
    """Frame para generación de reportes"""
    
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.current_df = None
        self.create_widgets()
    
    def create_widgets(self):
        # Panel izquierdo
        left_panel = ttk.Frame(self, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        ttk.Label(left_panel, text="Generación de Reportes",
                 font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        # Opciones
        options_frame = ttk.LabelFrame(left_panel, text="Secciones", padding=10)
        options_frame.pack(fill=tk.X, pady=5)
        
        self.section_vars = {}
        sections = ["Resumen ejecutivo", "Estadísticas descriptivas", 
                   "Correlaciones", "Tendencias"]
        
        for section in sections:
            var = tk.BooleanVar(value=True)
            self.section_vars[section] = var
            ttk.Checkbutton(options_frame, text=section, 
                          variable=var).pack(anchor=tk.W, pady=2)
        
        # Formato
        format_frame = ttk.LabelFrame(left_panel, text="Formato", padding=10)
        format_frame.pack(fill=tk.X, pady=5)
        
        self.format_var = tk.StringVar(value="Markdown")
        for fmt in ["Markdown", "HTML", "Texto"]:
            ttk.Radiobutton(format_frame, text=fmt, 
                          variable=self.format_var, 
                          value=fmt).pack(anchor=tk.W)
        
        ttk.Button(left_panel, text="📄 Generar Reporte",
                  command=self.generate_report,
                  style='Accent.TButton').pack(fill=tk.X, pady=20)
        
        # Panel derecho
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(right_panel, text="Vista Previa del Reporte",
                 font=('Segoe UI', 11, 'bold')).pack(pady=5)
        
        # Texto con scrollbar
        text_frame = ttk.Frame(right_panel)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.preview_text = tk.Text(text_frame, wrap=tk.WORD,
                                   yscrollcommand=scrollbar.set)
        self.preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.preview_text.yview)
    
    def update_data(self, df, data_results):
        self.current_df = df
    
    def generate_report(self):
        if self.current_df is None:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        
        try:
            selected_sections = [s for s, v in self.section_vars.items() if v.get()]
            report_format = self.format_var.get()
            
            report = self.create_report_content(selected_sections)
            
            # Mostrar vista previa
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, report)
            
            # Preguntar si desea guardar
            if messagebox.askyesno("Guardar reporte", 
                                  "¿Desea guardar el reporte generado?"):
                self.save_report(report, report_format)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar reporte:\n{str(e)}")
    
    def create_report_content(self, sections):
        """Crea el contenido del reporte"""
        data_columns = [col for col in self.current_df.columns if not col.startswith('_')]
        numeric_cols = [col for col in data_columns 
                       if pd.api.types.is_numeric_dtype(self.current_df[col])]
        
        report = f"""
# REPORTE DE ANÁLISIS DE DATOS CAF
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

"""
        
        if "Resumen ejecutivo" in sections:
            report += f"""
## RESUMEN EJECUTIVO

- Total de registros: {len(self.current_df)}
- Columnas totales: {len(data_columns)}
- Variables numéricas: {len(numeric_cols)}
- Variables categóricas: {len(data_columns) - len(numeric_cols)}
- Memoria utilizada: {self.current_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

"""
        
        if "Estadísticas descriptivas" in sections and numeric_cols:
            report += "\n## ESTADÍSTICAS DESCRIPTIVAS\n\n"
            
            for col in numeric_cols[:5]:  # Primeras 5 variables
                series = pd.to_numeric(self.current_df[col], errors='coerce').dropna()
                if len(series) > 0:
                    report += f"""
**{col}**
- Media: {series.mean():.2f}
- Mediana: {series.median():.2f}
- Desv.Std: {series.std():.2f}
- Mín: {series.min():.2f}
- Máx: {series.max():.2f}

"""
        
        if "Correlaciones" in sections and len(numeric_cols) >= 2:
            report += "\n## CORRELACIONES PRINCIPALES\n\n"
            
            corr_matrix = self.current_df[numeric_cols].corr()
            
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            "Var1": corr_matrix.columns[i],
                            "Var2": corr_matrix.columns[j],
                            "Corr": corr_matrix.iloc[i, j]
                        })
            
            if high_corr:
                for pair in sorted(high_corr, key=lambda x: abs(x["Corr"]), reverse=True)[:10]:
                    report += f"- {pair['Var1']} ↔ {pair['Var2']}: r = {pair['Corr']:.3f}\n"
            else:
                report += "*No se encontraron correlaciones fuertes (|r| > 0.7)*\n"
        
        if "Tendencias" in sections:
            report += "\n## TENDENCIAS Y OBSERVACIONES\n\n"
            report += "- Análisis de tendencias temporales\n"
            report += "- Patrones identificados en los datos\n"
            report += "- Recomendaciones para análisis futuros\n"
        
        report += f"\n\n{'='*60}\n"
        report += "Fin del reporte\n"
        
        return report
    
    def save_report(self, content, report_format):
        """Guarda el reporte"""
        ext = ".md" if report_format == "Markdown" else ".html" if report_format == "HTML" else ".txt"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[(f"{report_format} files", f"*{ext}"), ("All files", "*.*")],
            initialfile=f"reporte_caf_{datetime.now().strftime('%Y%m%d')}{ext}"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            messagebox.showinfo("Guardado", f"Reporte guardado en:\n{file_path}")

