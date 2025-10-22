"""
Visor de gráficos para Tkinter usando Matplotlib.
Proporciona un visor simple con barra de herramientas (zoom/pan/guardar)
y utilidades para abrir la última figura como imagen externa.
"""

import os
import tempfile
import webbrowser
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class PlotViewer(ttk.Frame):
    """Frame para mostrar gráficos de Matplotlib en Tkinter"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.current_fig = None
        self.temp_file = None
        self._build_ui()

    def _build_ui(self):
        # Controles superiores
        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.open_btn = ttk.Button(
            controls, text="Abrir imagen", command=self.open_in_browser, state=tk.DISABLED
        )
        self.open_btn.pack(side=tk.LEFT, padx=2)

        self.save_btn = ttk.Button(
            controls, text="Guardar gráfico", command=self.save_plot, state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=2)

        self.info_label = ttk.Label(controls, text="")
        self.info_label.pack(side=tk.RIGHT, padx=5)

        # Área de gráfico
        self.plot_frame = ttk.Frame(self, relief=tk.SUNKEN, borderwidth=1)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.message_label = ttk.Label(
            self.plot_frame,
            text="Seleccione opciones de análisis para generar gráficos",
            foreground="#999",
        )
        self.message_label.pack(expand=True)

    # Compatibilidad: no soportamos Plotly en esta versión
    def show_plotly_figure(self, fig, title="Gráfico"):
        self.show_error("Plotly no está soportado. Use Matplotlib.")

    def show_matplotlib_figure(self, fig, title="Gráfico"):
        try:
            # Limpiar placeholder
            if hasattr(self, "message_label"):
                self.message_label.pack_forget()

            for w in self.plot_frame.winfo_children():
                w.destroy()

            self.current_fig = fig

            # Canvas
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Toolbar
            try:
                toolbar_frame = ttk.Frame(self.plot_frame)
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
                toolbar_frame.pack(fill=tk.X)
            except Exception:
                pass

            # Temp image para "Abrir imagen"
            try:
                temp_dir = tempfile.gettempdir()
                self.temp_file = os.path.join(temp_dir, "caf_plot.png")
                fig.savefig(self.temp_file, dpi=150, bbox_inches="tight")
            except Exception:
                self.temp_file = None

            self.info_label.config(text=f"✓ {title}")
            self.save_btn.config(state=tk.NORMAL)
            self.open_btn.config(
                state=tk.NORMAL if (self.temp_file and os.path.exists(self.temp_file)) else tk.DISABLED
            )
        except Exception as e:
            self.show_error(f"Error al mostrar gráfico: {e}")

    def open_in_browser(self):
        if self.temp_file and os.path.exists(self.temp_file):
            webbrowser.open(f"file://{os.path.abspath(self.temp_file)}")

    def save_plot(self):
        if self.current_fig is None:
            return
        from tkinter import filedialog, messagebox

        file_types = [("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
        default_ext = ".png"

        file_path = filedialog.asksaveasfilename(
            defaultextension=default_ext, filetypes=file_types, initialfile=f"grafico{default_ext}"
        )
        if not file_path:
            return
        try:
            self.current_fig.savefig(file_path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Guardado", f"Gráfico guardado en:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar gráfico:\n{e}")

    def clear(self):
        self.current_fig = None
        self.temp_file = None
        for w in self.plot_frame.winfo_children():
            w.destroy()
        self.message_label = ttk.Label(
            self.plot_frame,
            text="Seleccione opciones de análisis para generar gráficos",
            foreground="#999",
        )
        self.message_label.pack(expand=True)
        self.open_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.info_label.config(text="")

    def show_error(self, message: str):
        for w in self.plot_frame.winfo_children():
            w.destroy()
        ttk.Label(self.plot_frame, text=f"⚠️ {message}", foreground="red").pack(expand=True)


class MultiPlotViewer(ttk.Frame):
    """Frame para mostrar múltiples gráficos en pestañas"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.viewers = {}

    def add_plot(self, name, fig, title: str = "", plot_type: str = "matplotlib"):
        # Crear viewer si no existe
        if name not in self.viewers:
            viewer = PlotViewer(self.notebook)
            self.notebook.add(viewer, text=name)
            self.viewers[name] = viewer

        # Mostrar gráfico (solo Matplotlib)
        viewer = self.viewers[name]
        viewer.show_matplotlib_figure(fig, title or name)

        # Seleccionar la pestaña
        try:
            idx = list(self.viewers.keys()).index(name)
            self.notebook.select(idx)
        except Exception:
            self.notebook.select(len(self.viewers) - 1)

        self.update_idletasks()

        # Intentar cambiar a la pestaña "Gráficos" del contenedor, si existe
        try:
            current = self
            while current and not hasattr(current, "notebook"):
                current = current.master
            if current and hasattr(current, "notebook"):
                for i in range(current.notebook.index("end")):
                    if current.notebook.tab(i, "text") == "Gráficos":
                        current.notebook.select(i)
                        break
        except Exception:
            pass

