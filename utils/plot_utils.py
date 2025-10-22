"""
Utilidades de visualización usando Matplotlib y Seaborn
Compatible con Tkinter - sin dependencias de Plotly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de seaborn
sns.set_style("whitegrid")
sns.set_palette("husl")

# Configurar matplotlib para mejor visualización
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class PlotBuilder:
    """Constructor de gráficos con Matplotlib y Seaborn"""

    @staticmethod
    def create_histogram(data: pd.Series, title: str = "", xlabel: str = "", bins: int = 30):
        """Crea un histograma"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(data.dropna(), bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
        ax.set_title(title or f'Histograma de {data.name}')
        ax.set_xlabel(xlabel or data.name)
        ax.set_ylabel('Frecuencia')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def create_boxplot(data: pd.Series, title: str = ""):
        """Crea un boxplot"""
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.boxplot(data.dropna(), vert=True, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                  medianprops=dict(color='red', linewidth=2))
        ax.set_title(title or f'Boxplot de {data.name}')
        ax.set_ylabel(data.name)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_violin_plot(data: pd.Series, title: str = ""):
        """Crea un violin plot"""
        fig, ax = plt.subplots(figsize=(8, 6))

        parts = ax.violinplot([data.dropna()], vert=True, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)

        ax.set_title(title or f'Violin Plot de {data.name}')
        ax.set_ylabel(data.name)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_qq_plot(data: pd.Series, title: str = ""):
        """Crea un Q-Q plot"""
        from scipy import stats

        fig, ax = plt.subplots(figsize=(8, 6))

        stats.probplot(data.dropna(), dist="norm", plot=ax)
        ax.set_title(title or f'Q-Q Plot de {data.name}')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def create_bar_plot(data: pd.Series, title: str = "", top_n: int = None):
        """Crea un gráfico de barras"""
        fig, ax = plt.subplots(figsize=(10, 6))

        counts = data.value_counts()
        if top_n:
            counts = counts.head(top_n)

        counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(title or f'Frecuencias de {data.name}')
        ax.set_xlabel(data.name)
        ax.set_ylabel('Frecuencia')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_pie_chart(data: pd.Series, title: str = "", top_n: int = 10):
        """Crea un gráfico de pastel"""
        fig, ax = plt.subplots(figsize=(10, 8))

        counts = data.value_counts().head(top_n)
        colors = sns.color_palette('pastel', len(counts))

        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title(title or f'Distribución de {data.name}')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_scatter_plot(x: pd.Series, y: pd.Series, title: str = ""):
        """Crea un scatter plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(x, y, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.set_title(title or f'{y.name} vs {x.name}')
        ax.set_xlabel(x.name)
        ax.set_ylabel(y.name)
        ax.grid(True, alpha=0.3)

        # Añadir línea de tendencia
        z = np.polyfit(x.dropna(), y.dropna(), 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8, label=f'Tendencia: y={z[0]:.3f}x+{z[1]:.3f}')
        ax.legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, title: str = ""):
        """Crea un heatmap de correlación"""
        fig, ax = plt.subplots(figsize=(12, 10))

        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title(title or 'Matriz de Correlación')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_pairplot(df: pd.DataFrame, hue: str = None):
        """Crea un pair plot"""
        g = sns.pairplot(df, hue=hue, diag_kind='hist', plot_kws={'alpha': 0.6})
        g.fig.suptitle('Pair Plot', y=1.02)
        plt.tight_layout()
        return g.fig

    @staticmethod
    def create_time_series_plot(data: pd.Series, title: str = ""):
        """Crea un gráfico de serie temporal con mejor proporcionalidad"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Limpiar datos
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            ax.text(0.5, 0.5, 'No hay datos válidos para mostrar', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title or f'Serie Temporal de {data.name}')
            return fig

        # Determinar el mejor estilo de visualización según la densidad de datos
        n_points = len(clean_data)
        
        if n_points > 1000:
            # Para muchos datos, usar líneas más finas y posiblemente submuestreo
            linewidth = 0.8
            alpha = 0.7
            # Si hay demasiados puntos, submuestrear para mejorar la visualización
            if n_points > 5000:
                step = max(1, n_points // 2000)  # Mostrar máximo 2000 puntos
                clean_data = clean_data.iloc[::step]
        else:
            linewidth = 1.5
            alpha = 1.0

        # Crear el gráfico principal
        ax.plot(clean_data.index, clean_data.values, 
                linewidth=linewidth, color='blue', alpha=alpha, 
                marker='', linestyle='-')

        # Configurar título y etiquetas
        ax.set_title(title or f'Serie Temporal de {data.name}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Tiempo', fontsize=12)
        ax.set_ylabel(data.name, fontsize=12)

        # Configurar el eje Y con rango más amplio y proporcionalidad correcta
        y_min, y_max = clean_data.min(), clean_data.max()
        y_range = y_max - y_min
        
        if y_range > 0:
            # Calcular un rango más amplio basado en los datos
            # Si el rango actual es muy pequeño, expandir significativamente
            if y_range < 10:  # Rango pequeño como 5-10
                # Expandir a un rango más amplio y centrado
                center = (y_min + y_max) / 2
                # Crear un rango más amplio: -10 a +80 o similar
                new_min = max(-10, center - 50)  # Mínimo -10, centrado en los datos
                new_max = min(80, center + 50)   # Máximo 80, centrado en los datos
                
                # Si los datos están muy concentrados, usar rango fijo más amplio
                if y_range < 5:
                    new_min = -10
                    new_max = 80
                    
                ax.set_ylim(new_min, new_max)
            else:
                # Para rangos normales, usar margen del 10%
                y_margin = y_range * 0.10
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Configurar grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Mejorar formato de fechas en el eje X si es datetime
        if pd.api.types.is_datetime64_any_dtype(clean_data.index):
            # Formatear fechas de manera más legible
            import matplotlib.dates as mdates
            
            # Determinar el intervalo apropiado según el rango temporal
            date_range = clean_data.index.max() - clean_data.index.min()
            
            if date_range.days <= 30:
                # Menos de un mes: mostrar días
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, date_range.days // 10)))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            elif date_range.days <= 365:
                # Menos de un año: mostrar meses
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, date_range.days // 300)))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
            else:
                # Más de un año: mostrar años
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
            # Rotar etiquetas para mejor legibilidad
            plt.xticks(rotation=45, ha='right')
        else:
            # Para índices no temporales, usar formato numérico
            plt.xticks(rotation=45, ha='right')

        # Ajustar layout
        plt.tight_layout()
        
        return fig

    @staticmethod
    def create_regression_plot(x: pd.Series, y: pd.Series, title: str = ""):
        """Crea un gráfico de regresión con intervalo de confianza"""
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
        ax.set_title(title or f'Regresión: {y.name} vs {x.name}')
        ax.set_xlabel(x.name)
        ax.set_ylabel(y.name)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def create_residuals_plot(y_true, y_pred, title: str = ""):
        """Crea un gráfico de residuos"""
        residuals = y_true - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Residuals vs Fitted
        ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Valores Predichos')
        ax1.set_ylabel('Residuos')
        ax1.set_title('Residuos vs Valores Ajustados')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Q-Q Plot de residuos
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot de Residuos')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title or 'Diagnóstico de Residuos', fontsize=14, y=1.02)
        plt.tight_layout()
        return fig

    @staticmethod
    def create_grouped_bar_plot(df: pd.DataFrame, x_col: str, y_col: str,
                               group_col: str, title: str = ""):
        """Crea un gráfico de barras agrupadas"""
        fig, ax = plt.subplots(figsize=(12, 6))

        df_pivot = df.pivot_table(values=y_col, index=x_col, columns=group_col, aggfunc='mean')
        df_pivot.plot(kind='bar', ax=ax, edgecolor='black')

        ax.set_title(title or f'{y_col} por {x_col} y {group_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(title=group_col)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_box_plot_by_category(data: pd.DataFrame, numeric_col: str,
                                    category_col: str, title: str = ""):
        """Crea boxplots por categoría"""
        fig, ax = plt.subplots(figsize=(12, 6))

        data.boxplot(column=numeric_col, by=category_col, ax=ax, patch_artist=True)
        ax.set_title(title or f'{numeric_col} por {category_col}')
        ax.set_xlabel(category_col)
        ax.set_ylabel(numeric_col)
        plt.xticks(rotation=45, ha='right')
        plt.suptitle('')  # Remover título automático

        plt.tight_layout()
        return fig

    @staticmethod
    def create_violin_plot_by_category(data: pd.DataFrame, numeric_col: str,
                                      category_col: str, title: str = ""):
        """Crea violin plots por categoría"""
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.violinplot(data=data, x=category_col, y=numeric_col, ax=ax)
        ax.set_title(title or f'{numeric_col} por {category_col}')
        ax.set_xlabel(category_col)
        ax.set_ylabel(numeric_col)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_density_plot(data: pd.Series, title: str = ""):
        """Crea un gráfico de densidad (KDE)"""
        fig, ax = plt.subplots(figsize=(10, 6))

        data.dropna().plot(kind='density', ax=ax, linewidth=2)
        ax.set_title(title or f'Densidad de {data.name}')
        ax.set_xlabel(data.name)
        ax.set_ylabel('Densidad')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def create_cluster_plot(X: np.ndarray, labels: np.ndarray, title: str = ""):
        """Crea un gráfico de clusters (2D o 3D si hay más dimensiones usa PCA)"""
        from sklearn.decomposition import PCA

        # Si hay más de 2 dimensiones, reducir con PCA
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_
            xlabel = f'PC1 ({explained_var[0]:.1%})'
            ylabel = f'PC2 ({explained_var[1]:.1%})'
        else:
            X_plot = X
            xlabel = 'Dimensión 1'
            ylabel = 'Dimensión 2'

        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis',
                           alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.set_title(title or 'Análisis de Clusters')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        # Añadir colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_confusion_matrix(cm: np.ndarray, labels: List[str] = None, title: str = ""):
        """Crea un heatmap de matriz de confusión"""
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title or 'Matriz de Confusión')
        ax.set_ylabel('Etiqueta Real')
        ax.set_xlabel('Etiqueta Predicha')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_feature_importance_plot(importances: np.ndarray, feature_names: List[str],
                                      title: str = "", top_n: int = 20):
        """Crea un gráfico de importancia de características"""
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))

        # Ordenar por importancia
        indices = np.argsort(importances)[::-1][:top_n]

        ax.barh(range(len(indices)), importances[indices], color='skyblue', edgecolor='black')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importancia')
        ax.set_title(title or f'Top {top_n} Características Más Importantes')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_roc_curve(fpr, tpr, roc_auc, title: str = ""):
        """Crea curva ROC"""
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title or 'Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def close_all_figures():
    """Cierra todas las figuras de matplotlib"""
    plt.close('all')
