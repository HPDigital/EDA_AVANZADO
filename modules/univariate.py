"""
Módulo de análisis univariado para CAF Dashboard
Contiene toda la lógica de análisis estadístico sin dependencias de UI
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings("ignore")


class UnivariateAnalyzer:
    """Clase para realizar análisis univariado completo"""

    def __init__(self, series: pd.Series, col_name: str = None):
        """
        Inicializa el analizador univariado

        Args:
            series: Serie de pandas a analizar
            col_name: Nombre de la columna (opcional)
        """
        self.series = series
        self.col_name = col_name or "Variable"
        self.clean_series = self._clean_data()

    def _clean_data(self) -> pd.Series:
        """Limpia y prepara los datos"""
        try:
            # Intentar convertir a numérico
            return pd.to_numeric(self.series, errors='coerce').dropna()
        except:
            # Si falla, retornar sin NaN
            return self.series.dropna()

    def is_numeric(self) -> bool:
        """Verifica si la serie es numérica"""
        return pd.api.types.is_numeric_dtype(self.clean_series)

    def get_basic_info(self) -> Dict[str, Any]:
        """Obtiene información básica de la variable"""
        return {
            'name': self.col_name,
            'n_valid': len(self.clean_series),
            'n_missing': self.series.isna().sum(),
            'n_total': len(self.series),
            'missing_pct': (self.series.isna().sum() / len(self.series) * 100) if len(self.series) > 0 else 0,
            'dtype': str(self.clean_series.dtype),
            'is_numeric': self.is_numeric()
        }

    # ==================== ANÁLISIS NUMÉRICO ====================

    def get_descriptive_stats(self) -> Dict[str, float]:
        """Calcula estadísticos descriptivos completos"""
        if not self.is_numeric() or len(self.clean_series) == 0:
            return {}

        stats_dict = {
            'count': len(self.clean_series),
            'mean': float(self.clean_series.mean()),
            'median': float(self.clean_series.median()),
            'std': float(self.clean_series.std()),
            'var': float(self.clean_series.var()),
            'min': float(self.clean_series.min()),
            'max': float(self.clean_series.max()),
            'range': float(self.clean_series.max() - self.clean_series.min()),
            'skewness': float(stats.skew(self.clean_series)),
            'kurtosis': float(stats.kurtosis(self.clean_series)),
            'cv': float((self.clean_series.std() / self.clean_series.mean() * 100)) if self.clean_series.mean() != 0 else 0
        }

        # Estadísticos robustos
        try:
            from scipy.stats import median_abs_deviation
            stats_dict['mad'] = float(median_abs_deviation(self.clean_series))
            stats_dict['trimmed_mean'] = float(stats.trim_mean(self.clean_series, 0.1))
            stats_dict['iqr'] = float(self.clean_series.quantile(0.75) - self.clean_series.quantile(0.25))
        except:
            pass

        return stats_dict

    def get_percentiles(self, percentiles: List[int] = None) -> Dict[str, float]:
        """Calcula percentiles especificados"""
        if not self.is_numeric() or len(self.clean_series) == 0:
            return {}

        if percentiles is None:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

        return {
            f'P{p}': float(self.clean_series.quantile(p / 100))
            for p in percentiles
        }

    def test_normality(self) -> Dict[str, Dict[str, float]]:
        """Realiza tests de normalidad"""
        if not self.is_numeric() or len(self.clean_series) < 3:
            return {}

        results = {}

        # Shapiro-Wilk
        if 3 <= len(self.clean_series) <= 5000:
            try:
                w_stat, p_val = stats.shapiro(self.clean_series)
                results['shapiro'] = {'statistic': float(w_stat), 'p_value': float(p_val)}
            except:
                pass

        # Kolmogorov-Smirnov
        try:
            ks_stat, ks_p = stats.kstest(
                self.clean_series,
                'norm',
                args=(self.clean_series.mean(), self.clean_series.std())
            )
            results['ks'] = {'statistic': float(ks_stat), 'p_value': float(ks_p)}
        except:
            pass

        # Anderson-Darling
        try:
            ad_result = stats.anderson(self.clean_series, dist='norm')
            results['anderson'] = {
                'statistic': float(ad_result.statistic),
                'critical_values': [float(v) for v in ad_result.critical_values],
                'significance_levels': [float(s) for s in ad_result.significance_level]
            }
        except:
            pass

        # D'Agostino
        if len(self.clean_series) >= 20:
            try:
                k2_stat, k2_p = stats.normaltest(self.clean_series)
                results['dagostino'] = {'statistic': float(k2_stat), 'p_value': float(k2_p)}
            except:
                pass

        # Jarque-Bera
        try:
            jb_stat, jb_p = stats.jarque_bera(self.clean_series)
            results['jarque_bera'] = {'statistic': float(jb_stat), 'p_value': float(jb_p)}
        except:
            pass

        return results

    def detect_outliers(self) -> Dict[str, Any]:
        """Detecta outliers usando múltiples métodos"""
        if not self.is_numeric() or len(self.clean_series) == 0:
            return {}

        results = {}

        # Método IQR
        q1 = self.clean_series.quantile(0.25)
        q3 = self.clean_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_iqr = self.clean_series[(self.clean_series < lower_bound) | (self.clean_series > upper_bound)]

        results['iqr'] = {
            'n_outliers': len(outliers_iqr),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_indices': outliers_iqr.index.tolist(),
            'outlier_values': outliers_iqr.values.tolist()
        }

        # Método Z-score
        z_scores = np.abs(stats.zscore(self.clean_series))
        outliers_z = self.clean_series[z_scores > 3]
        results['zscore'] = {
            'n_outliers': len(outliers_z),
            'threshold': 3,
            'outlier_indices': outliers_z.index.tolist(),
            'outlier_values': outliers_z.values.tolist()
        }

        # Método de Tukey (modificado)
        outliers_tukey = self.clean_series[(self.clean_series < q1 - 3*iqr) | (self.clean_series > q3 + 3*iqr)]
        results['tukey'] = {
            'n_outliers': len(outliers_tukey),
            'outlier_indices': outliers_tukey.index.tolist(),
            'outlier_values': outliers_tukey.values.tolist()
        }

        # Isolation Forest
        try:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(self.clean_series.values.reshape(-1, 1))
            outliers_iso = self.clean_series[outlier_labels == -1]
            results['isolation_forest'] = {
                'n_outliers': len(outliers_iso),
                'outlier_indices': outliers_iso.index.tolist(),
                'outlier_values': outliers_iso.values.tolist()
            }
        except:
            pass

        return results

    def apply_transformation(self, transform_type: str = 'boxcox') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aplica transformaciones para normalización

        Args:
            transform_type: Tipo de transformación ('boxcox', 'yeo-johnson', 'log', 'sqrt', 'reciprocal')

        Returns:
            Tupla de (datos transformados, información de la transformación)
        """
        if not self.is_numeric() or len(self.clean_series) == 0:
            return self.clean_series.values, {'error': 'No numeric data'}

        info = {'type': transform_type}

        try:
            if transform_type == 'boxcox':
                if np.all(self.clean_series > 0):
                    transformed, lambda_val = stats.boxcox(self.clean_series)
                    info['lambda'] = float(lambda_val)
                else:
                    return self.clean_series.values, {'error': 'BoxCox requires positive values'}

            elif transform_type == 'yeo-johnson':
                from sklearn.preprocessing import PowerTransformer
                pt = PowerTransformer(method='yeo-johnson')
                transformed = pt.fit_transform(self.clean_series.values.reshape(-1, 1)).flatten()
                info['lambda'] = float(pt.lambdas_[0])

            elif transform_type == 'log':
                if np.all(self.clean_series > 0):
                    transformed = np.log(self.clean_series)
                else:
                    transformed = np.log1p(self.clean_series - self.clean_series.min() + 1)
                    info['adjusted'] = True

            elif transform_type == 'sqrt':
                if np.all(self.clean_series >= 0):
                    transformed = np.sqrt(self.clean_series)
                else:
                    transformed = np.sqrt(self.clean_series - self.clean_series.min())
                    info['adjusted'] = True

            elif transform_type == 'reciprocal':
                if np.all(self.clean_series != 0):
                    transformed = 1 / self.clean_series
                else:
                    return self.clean_series.values, {'error': 'Reciprocal requires non-zero values'}

            else:
                return self.clean_series.values, {'error': f'Unknown transformation: {transform_type}'}

            info['success'] = True
            return transformed, info

        except Exception as e:
            return self.clean_series.values, {'error': str(e)}

    def calculate_autocorrelation(self, max_lags: int = 20) -> Dict[str, np.ndarray]:
        """Calcula ACF y PACF"""
        if not self.is_numeric() or len(self.clean_series) < 5:
            return {}

        max_lags = min(max_lags, len(self.clean_series) - 1)

        # ACF manual
        def autocorr(x, max_lags):
            x = np.array(x) - np.mean(x)
            n = len(x)
            acf = np.correlate(x, x, mode='full')[n-1:]
            acf = acf / acf[0]
            return acf[:max_lags+1]

        acf_vals = autocorr(self.clean_series, max_lags)

        result = {'acf': acf_vals}

        # PACF si statsmodels está disponible
        try:
            from statsmodels.tsa.stattools import pacf as pacf_func
            pacf_vals = pacf_func(self.clean_series, nlags=max_lags, method='ywm')
            result['pacf'] = pacf_vals
        except:
            pass

        return result

    def test_randomness(self) -> Dict[str, Any]:
        """Tests de aleatoriedad y tendencia"""
        if not self.is_numeric() or len(self.clean_series) < 10:
            return {}

        results = {}

        # Runs test
        median = np.median(self.clean_series)
        runs = []
        current_run = None
        for val in self.clean_series:
            is_above = val > median
            if is_above != current_run:
                runs.append(is_above)
                current_run = is_above

        n_runs = len(runs)
        n_pos = sum(runs)
        n_neg = len(runs) - n_pos
        n = len(self.clean_series)

        if n_pos > 0 and n_neg > 0:
            expected_runs = ((2 * n_pos * n_neg) / n) + 1
            var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / ((n ** 2) * (n - 1))

            if var_runs > 0:
                z = (n_runs - expected_runs) / np.sqrt(var_runs)
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                results['runs_test'] = {
                    'z': float(z),
                    'p_value': float(p_value),
                    'n_runs': n_runs,
                    'expected_runs': float(expected_runs)
                }

        # Mann-Kendall (test de tendencia)
        try:
            x = np.arange(n)
            tau, p_trend = kendalltau(x, self.clean_series)
            results['mann_kendall'] = {
                'tau': float(tau),
                'p_value': float(p_trend)
            }
        except:
            pass

        return results

    def apply_smoothing(self, method: str = 'ma', **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aplica técnicas de suavizado

        Args:
            method: Método de suavizado ('ma', 'ewm', 'savgol', 'loess')
            **kwargs: Parámetros específicos del método

        Returns:
            Tupla de (datos suavizados, información)
        """
        if not self.is_numeric() or len(self.clean_series) < 5:
            return self.clean_series.values, {'error': 'Insufficient data'}

        info = {'method': method}

        try:
            if method == 'ma':
                window = kwargs.get('window', 5)
                smoothed = self.clean_series.rolling(window=window, center=True).mean()
                info['window'] = window

            elif method == 'ewm':
                alpha = kwargs.get('alpha', 0.3)
                smoothed = self.clean_series.ewm(alpha=alpha).mean()
                info['alpha'] = alpha

            elif method == 'savgol':
                from scipy.signal import savgol_filter
                window = kwargs.get('window', 7)
                poly_order = kwargs.get('poly_order', 2)
                if window % 2 == 0:
                    window += 1
                smoothed = pd.Series(savgol_filter(self.clean_series, window, poly_order))
                info['window'] = window
                info['poly_order'] = poly_order

            elif method == 'loess':
                from statsmodels.nonparametric.smoothers_lowess import lowess
                frac = kwargs.get('frac', 0.3)
                smoothed_vals = lowess(self.clean_series, np.arange(len(self.clean_series)), frac=frac, return_sorted=False)
                smoothed = pd.Series(smoothed_vals)
                info['frac'] = frac

            else:
                return self.clean_series.values, {'error': f'Unknown method: {method}'}

            return smoothed.values, info

        except Exception as e:
            return self.clean_series.values, {'error': str(e)}

    # ==================== ANÁLISIS CATEGÓRICO ====================

    def get_frequency_table(self) -> pd.DataFrame:
        """Genera tabla de frecuencias para variables categóricas"""
        if len(self.clean_series) == 0:
            return pd.DataFrame()

        freq_table = self.clean_series.value_counts().reset_index()
        freq_table.columns = ['Category', 'Frequency']
        freq_table['Percentage'] = (freq_table['Frequency'] / len(self.clean_series) * 100).round(2)
        freq_table['Cumulative_Pct'] = freq_table['Percentage'].cumsum().round(2)

        return freq_table

    def get_diversity_indices(self) -> Dict[str, float]:
        """Calcula índices de diversidad para variables categóricas"""
        if len(self.clean_series) == 0:
            return {}

        freq_table = self.clean_series.value_counts()
        p = freq_table.values / freq_table.sum()

        indices = {}

        # Shannon
        indices['shannon'] = float(-sum(p * np.log(p + 1e-12)))

        # Simpson
        indices['simpson'] = float(sum(p**2))

        # Gini-Simpson
        indices['gini_simpson'] = float(1 - indices['simpson'])

        # Pielou (equitatividad)
        if len(p) > 1:
            indices['pielou'] = float(indices['shannon'] / np.log(len(p)))

        # Gini index
        indices['gini'] = float(1 - sum(p**2))

        # Berger-Parker
        indices['berger_parker'] = float(max(p))

        # Margalef
        if len(self.clean_series) > 1:
            indices['margalef'] = float((len(p) - 1) / np.log(len(self.clean_series)))

        return indices

    def test_uniformity(self) -> Dict[str, Dict[str, float]]:
        """Tests de uniformidad para variables categóricas"""
        if len(self.clean_series) < 3:
            return {}

        freq_table = self.clean_series.value_counts()
        results = {}

        # Chi-cuadrado de bondad de ajuste (uniforme)
        try:
            chi2_stat, chi2_p = stats.chisquare(freq_table.values)
            results['chi_square'] = {'statistic': float(chi2_stat), 'p_value': float(chi2_p)}
        except:
            pass

        return results

    def generate_summary(self) -> Dict[str, Any]:
        """Genera un resumen ejecutivo del análisis"""
        summary = {
            'basic_info': self.get_basic_info(),
            'findings': [],
            'alerts': [],
            'quality': 'Unknown',
            'recommendation': 'Unknown'
        }

        if self.is_numeric():
            stats_dict = self.get_descriptive_stats()
            normality = self.test_normality()
            outliers = self.detect_outliers()

            # Análisis de distribución
            skewness = stats_dict.get('skewness', 0)
            if abs(skewness) < 0.5:
                dist_type = "Simétrica"
            elif skewness > 0.5:
                dist_type = "Sesgada a la derecha"
            else:
                dist_type = "Sesgada a la izquierda"

            summary['findings'].append(f"Distribución {dist_type.lower()} (asimetría: {skewness:.3f})")

            # Normalidad
            if 'shapiro' in normality:
                is_normal = normality['shapiro']['p_value'] > 0.05
                summary['findings'].append(f"Los datos {'siguen' if is_normal else 'NO siguen'} una distribución normal")

            # Outliers
            n_outliers_iqr = outliers.get('iqr', {}).get('n_outliers', 0)
            if n_outliers_iqr > 0:
                summary['findings'].append(f"Se detectaron {n_outliers_iqr} valores atípicos (IQR)")
                summary['alerts'].append("Presencia de valores atípicos que requieren revisión")

            # Calidad de datos
            missing_pct = summary['basic_info']['missing_pct']
            if missing_pct < 5:
                summary['quality'] = "Excelente"
            elif missing_pct < 15:
                summary['quality'] = "Buena"
                summary['alerts'].append(f"Porcentaje moderado de valores faltantes ({missing_pct:.1f}%)")
            elif missing_pct < 30:
                summary['quality'] = "Regular"
                summary['alerts'].append(f"Alto porcentaje de valores faltantes ({missing_pct:.1f}%)")
            else:
                summary['quality'] = "Pobre"
                summary['alerts'].append(f"Muy alto porcentaje de valores faltantes ({missing_pct:.1f}%)")

            # Recomendación
            if missing_pct < 5 and n_outliers_iqr == 0:
                summary['recommendation'] = "Excelente"
            elif missing_pct < 15:
                summary['recommendation'] = "Buena"
            else:
                summary['recommendation'] = "Requiere atención"

        else:
            # Análisis categórico
            freq_table = self.get_frequency_table()
            diversity = self.get_diversity_indices()

            summary['findings'].append(f"Variable categórica con {len(freq_table)} categorías únicas")
            summary['findings'].append(f"Índice de Shannon: {diversity.get('shannon', 0):.3f}")

            # Calidad basada en missing
            missing_pct = summary['basic_info']['missing_pct']
            if missing_pct < 5:
                summary['quality'] = "Excelente"
            elif missing_pct < 15:
                summary['quality'] = "Buena"
            else:
                summary['quality'] = "Regular"
                summary['alerts'].append(f"Alto porcentaje de valores faltantes ({missing_pct:.1f}%)")

            summary['recommendation'] = summary['quality']

        return summary


def analyze_numeric(series: pd.Series, col_name: str = None) -> Dict[str, Any]:
    """
    Función de conveniencia para análisis numérico completo

    Args:
        series: Serie de pandas numérica
        col_name: Nombre de la columna

    Returns:
        Diccionario con todos los resultados del análisis
    """
    analyzer = UnivariateAnalyzer(series, col_name)

    if not analyzer.is_numeric():
        return {'error': 'Series is not numeric'}

    return {
        'basic_info': analyzer.get_basic_info(),
        'descriptive_stats': analyzer.get_descriptive_stats(),
        'percentiles': analyzer.get_percentiles(),
        'normality_tests': analyzer.test_normality(),
        'outliers': analyzer.detect_outliers(),
        'autocorrelation': analyzer.calculate_autocorrelation(),
        'randomness_tests': analyzer.test_randomness(),
        'summary': analyzer.generate_summary()
    }


def analyze_categorical(series: pd.Series, col_name: str = None) -> Dict[str, Any]:
    """
    Función de conveniencia para análisis categórico completo

    Args:
        series: Serie de pandas categórica
        col_name: Nombre de la columna

    Returns:
        Diccionario con todos los resultados del análisis
    """
    analyzer = UnivariateAnalyzer(series, col_name)

    return {
        'basic_info': analyzer.get_basic_info(),
        'frequency_table': analyzer.get_frequency_table(),
        'diversity_indices': analyzer.get_diversity_indices(),
        'uniformity_tests': analyzer.test_uniformity(),
        'summary': analyzer.generate_summary()
    }
