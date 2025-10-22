"""
Módulo de análisis multivariado para CAF Dashboard
Contiene lógica de análisis de múltiples variables sin dependencias de UI
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings("ignore")


class MultivariateAnalyzer:
    """Clase para realizar análisis multivariado completo"""

    def __init__(self, df: pd.DataFrame, variables: List[str] = None):
        self.df = df
        self.variables = variables if variables else self._get_numeric_columns()
        self.df_clean = self._clean_data()
        self.df_scaled = None

    def _get_numeric_columns(self) -> List[str]:
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def _clean_data(self) -> pd.DataFrame:
        return self.df[self.variables].dropna()

    def prepare_data(self, impute_method: str = 'mean', scale_method: str = 'standard') -> pd.DataFrame:
        df_prep = self.df[self.variables].copy()

        if impute_method == 'median':
            df_prep = df_prep.fillna(df_prep.median())
        elif impute_method == 'mode':
            df_prep = df_prep.fillna(df_prep.mode().iloc[0] if len(df_prep.mode()) > 0 else 0)
        else:
            df_prep = df_prep.fillna(df_prep.mean())

        if scale_method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df_prep = pd.DataFrame(scaler.fit_transform(df_prep), columns=df_prep.columns, index=df_prep.index)
        elif scale_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            df_prep = pd.DataFrame(scaler.fit_transform(df_prep), columns=df_prep.columns, index=df_prep.index)
        elif scale_method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            df_prep = pd.DataFrame(scaler.fit_transform(df_prep), columns=df_prep.columns, index=df_prep.index)

        self.df_scaled = df_prep
        return df_prep

    def correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        df = self.df_clean if self.df_scaled is None else self.df_scaled
        return df.corr(method=method)

    def find_high_correlations(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        corr_matrix = self.correlation_matrix()
        high_corr = []
        for i in range(len(self.variables)):
            for j in range(i + 1, len(self.variables)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr.append({'var1': self.variables[i], 'var2': self.variables[j], 'correlation': float(corr_val)})
        return sorted(high_corr, key=lambda x: abs(x['correlation']), reverse=True)

    def perform_pca(self, n_components: int = None) -> Dict[str, Any]:
        try:
            from sklearn.decomposition import PCA
            df = self.df_scaled if self.df_scaled is not None else self.prepare_data()
            if n_components is None:
                n_components = min(len(self.variables), len(df))
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(df)
            return {
                'components': pca_result,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumsum_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'loadings': pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=self.variables).to_dict()
            }
        except Exception as e:
            return {'error': str(e)}

    def perform_kmeans(self, n_clusters: int = 3) -> Dict[str, Any]:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            df = self.df_scaled if self.df_scaled is not None else self.prepare_data()
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(df)
            return {
                'labels': labels.tolist(),
                'centers': kmeans.cluster_centers_.tolist(),
                'inertia': float(kmeans.inertia_),
                'silhouette_score': float(silhouette_score(df, labels))
            }
        except Exception as e:
            return {'error': str(e)}

    def detect_outliers_isolation_forest(self, contamination: float = 0.1) -> Dict[str, Any]:
        try:
            from sklearn.ensemble import IsolationForest
            df = self.df_scaled if self.df_scaled is not None else self.prepare_data()
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(df)
            return {
                'labels': outlier_labels.tolist(),
                'n_outliers': int((outlier_labels == -1).sum()),
                'outlier_indices': np.where(outlier_labels == -1)[0].tolist()
            }
        except Exception as e:
            return {'error': str(e)}

    def generate_summary(self) -> Dict[str, Any]:
        n_variables = len(self.variables)
        n_observations = len(self.df_clean)
        summary = {'n_variables': n_variables, 'n_observations': n_observations, 'findings': [], 'alerts': []}

        high_corr = self.find_high_correlations(threshold=0.7)
        pca_result = self.perform_pca()

        if 'error' not in pca_result:
            explained_var = pca_result['explained_variance_ratio']
            effective_dims = sum(1 for var in explained_var if var > 0.01)
            summary['findings'].append(f"Dimensionalidad efectiva: {effective_dims} dimensiones")

        if high_corr:
            summary['findings'].append(f"Se encontraron {len(high_corr)} pares altamente correlacionados")

        return summary
