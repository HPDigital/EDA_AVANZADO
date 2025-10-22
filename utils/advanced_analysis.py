"""
Funciones de análisis avanzado (numérico y categórico) usadas por la app y reportes.
Implementación ligera sin dependencias externas, compatible con Streamlit.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    import streamlit as st
except Exception:  # modo no-UI
    class st:  # type: ignore
        @staticmethod
        def subheader(*args, **kwargs):
            pass

        @staticmethod
        def dataframe(*args, **kwargs):
            pass

        @staticmethod
        def plotly_chart(*args, **kwargs):
            pass

        @staticmethod
        def write(*args, **kwargs):
            pass


def display_advanced_numeric_analysis(series: pd.Series, name: str):
    """Muestra estadísticas y gráficos básicos para una serie numérica."""
    st.subheader(f"Análisis numérico: {name}")
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        st.write("Sin datos numéricos válidos.")
        return

    desc = s.describe(percentiles=[0.01, 0.05, 0.1, 0.9, 0.95, 0.99]).to_frame("Valor")
    desc.loc["IQR"] = desc.loc["75%", "Valor"] - desc.loc["25%", "Valor"]
    st.dataframe(desc, use_container_width=True)

    fig = make_hist_box(s, name)
    st.plotly_chart(fig, use_container_width=True)


def display_advanced_categorical_analysis(series: pd.Series, name: str):
    """Muestra distribución categórica y tabla de frecuencias."""
    st.subheader(f"Análisis categórico: {name}")
    s = series.astype(str).fillna("(NA)")
    vc = s.value_counts(dropna=False)
    df = vc.rename_axis(name).reset_index(name="Frecuencia")
    st.dataframe(df, use_container_width=True)

    fig = px.bar(df.head(30), x=name, y="Frecuencia", title=f"Top categorías de {name}")
    st.plotly_chart(fig, use_container_width=True)


def get_advanced_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen compacto por columna numérica."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return pd.DataFrame()
    desc = df[num_cols].describe().T
    desc["IQR"] = desc["75%"] - desc["25%"]
    return desc


def get_advanced_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen compacto por columna categórica: n únicos y top categoría.""" 
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for c in cat_cols:
        s = df[c].astype(str)
        vc = s.value_counts()
        top = vc.index[0] if len(vc) else ""
        rows.append({"col": c, "unique": s.nunique(dropna=False), "top": top})
    return pd.DataFrame(rows)


def make_hist_box(s: pd.Series, name: str):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s, nbinsx=30, name="Histograma"))
    fig.add_trace(go.Box(y=s, name="Boxplot"))
    fig.update_layout(title=f"Distribución {name}")
    return fig

