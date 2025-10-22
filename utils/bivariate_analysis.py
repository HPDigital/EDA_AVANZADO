"""
Funciones de análisis bivariado para la app (UI simple) apoyadas en SciPy/Plotly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
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


def display_numeric_numeric_analysis(df: pd.DataFrame, numeric_cols: list):
    st.subheader("Numérico vs Numérico")
    cols = [c for c in numeric_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        st.write("Se requieren 2 o más columnas numéricas.")
        return
    corr = df[cols].corr()
    st.dataframe(corr, use_container_width=True)
    fig = px.imshow(corr, text_auto=False, title="Matriz de correlaciones")
    st.plotly_chart(fig, use_container_width=True)


def display_numeric_categorical_analysis(df: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    st.subheader("Numérico vs Categórico")
    num = [c for c in numeric_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in categorical_cols if c in df.columns]
    if not num or not cat:
        st.write("Necesitas al menos una numérica y una categórica.")
        return

    x = st.selectbox("Variable numérica", num)
    g = st.selectbox("Variable categórica", cat)
    if not x or not g:
        return

    fig = px.box(df, x=g, y=x, points="outliers", title=f"{x} por {g}")
    st.plotly_chart(fig, use_container_width=True)

    # ANOVA de una vía (si hay >=2 grupos)
    groups = [pd.to_numeric(v.dropna(), errors="coerce") for _, v in df.groupby(g)[x]]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) >= 2:
        try:
            f, p = stats.f_oneway(*groups)
            st.write(f"ANOVA: F={f:.3f}, p={p:.3g}")
        except Exception as e:
            st.write(f"ANOVA no disponible: {e}")


def display_categorical_categorical_analysis(df: pd.DataFrame, categorical_cols: list):
    st.subheader("Categórico vs Categórico")
    cat = [c for c in categorical_cols if c in df.columns]
    if len(cat) < 2:
        st.write("Se requieren 2 o más categóricas.")
        return

    a = st.selectbox("Variable A", cat)
    b = st.selectbox("Variable B", [c for c in cat if c != a])
    if not a or not b:
        return

    ct = pd.crosstab(df[a], df[b])
    st.dataframe(ct, use_container_width=True)
    try:
        from scipy.stats import chi2_contingency

        chi2, p, dof, _ = chi2_contingency(ct)
        st.write(f"Chi-cuadrado: χ²={chi2:.3f}, p={p:.3g}, dof={dof}")
    except Exception as e:
        st.write(f"Chi-cuadrado no disponible: {e}")

