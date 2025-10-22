"""
Funciones de análisis multivariado (p. ej., PCA) para la app.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px

try:
    import streamlit as st
except Exception:  # modo no-UI
    class st:  # type: ignore
        @staticmethod
        def subheader(*args, **kwargs):
            pass

        @staticmethod
        def plotly_chart(*args, **kwargs):
            pass


def display_advanced_multivariate_analysis(df: pd.DataFrame, numeric_cols: list):
    """PCA 2D simple sobre variables numéricas seleccionadas."""
    st.subheader("Análisis multivariado (PCA)")
    cols = [c for c in numeric_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        st.write("Se requieren 2 o más columnas numéricas para PCA.")
        return

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(X) < 5:
        st.write("Datos insuficientes para PCA.")
        return

    Xs = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=2)
    comps = pca.fit_transform(Xs)
    out = pd.DataFrame({"PC1": comps[:, 0], "PC2": comps[:, 1]})

    fig = px.scatter(out, x="PC1", y="PC2", title="PCA (2 componentes)")
    st.plotly_chart(fig, use_container_width=True)

