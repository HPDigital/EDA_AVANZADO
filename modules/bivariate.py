"""
Módulo de análisis bivariado para CAF Dashboard
Contiene toda la lógica de análisis de relaciones entre dos variables sin dependencias de UI
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, kendalltau
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings("ignore")


class BivariateAnalyzer:
    """Clase para realizar análisis bivariado completo"""

    def __init__(self, df: pd.DataFrame, var1: str, var2: str):
        """
        Inicializa el analizador bivariado

        Args:
            df: DataFrame con los datos
            var1: Nombre de la primera variable
            var2: Nombre de la segunda variable
        """
        self.df = df
        self.var1 = var1
        self.var2 = var2
        self.df_clean = self._clean_data()
        self.var1_type = self._get_var_type(var1)
        self.var2_type = self._get_var_type(var2)

    def _clean_data(self) -> pd.DataFrame:
        """Limpia y prepara los datos"""
        return self.df[[self.var1, self.var2]].dropna()

    def _get_var_type(self, var: str) -> str:
        """Determina el tipo de variable"""
        if pd.api.types.is_numeric_dtype(self.df_clean[var]):
            return 'numeric'
        return 'categorical'

    def get_analysis_type(self) -> str:
        """Retorna el tipo de análisis bivariado"""
        if self.var1_type == 'numeric' and self.var2_type == 'numeric':
            return 'numeric-numeric'
        elif self.var1_type == 'numeric' or self.var2_type == 'numeric':
            return 'numeric-categorical'
        return 'categorical-categorical'

    # ==================== ANÁLISIS NUMÉRICO-NUMÉRICO ====================

    def correlation_analysis(self) -> Dict[str, Dict[str, float]]:
        """Calcula correlaciones múltiples"""
        if self.get_analysis_type() != 'numeric-numeric':
            return {}

        x = self.df_clean[self.var1].values
        y = self.df_clean[self.var2].values

        results = {}

        # Pearson
        try:
            r, p = stats.pearsonr(x, y)
            results['pearson'] = {'r': float(r), 'p_value': float(p)}

            # Intervalo de confianza para Pearson
            n = len(x)
            if n > 3:
                z = np.arctanh(r)
                se = 1.0 / np.sqrt(n - 3)
                z_crit = stats.norm.ppf(0.975)
                ci_low = np.tanh(z - z_crit * se)
                ci_high = np.tanh(z + z_crit * se)
                results['pearson']['ci_95'] = [float(ci_low), float(ci_high)]
        except:
            pass

        # Spearman
        try:
            rho, p = stats.spearmanr(x, y)
            results['spearman'] = {'rho': float(rho), 'p_value': float(p)}
        except:
            pass

        # Kendall
        try:
            tau, p = stats.kendalltau(x, y)
            results['kendall'] = {'tau': float(tau), 'p_value': float(p)}
        except:
            pass

        # Distance correlation
        try:
            import dcor
            dc = dcor.distance_correlation(x, y)
            results['distance'] = {'dcor': float(dc)}
        except:
            pass

        return results

    def regression_analysis(self) -> Dict[str, Any]:
        """Realiza análisis de regresión lineal"""
        if self.get_analysis_type() != 'numeric-numeric':
            return {}

        x = self.df_clean[self.var1].values
        y = self.df_clean[self.var2].values

        # Regresión lineal
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Predicciones y residuos
        y_pred = intercept + slope * x
        residuals = y - y_pred

        results = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'predictions': y_pred.tolist(),
            'residuals': residuals.tolist()
        }

        # Tests de residuos
        try:
            # Test de normalidad de residuos
            if len(residuals) <= 5000:
                w_stat, p_shapiro = stats.shapiro(residuals)
                results['residuals_normality'] = {
                    'statistic': float(w_stat),
                    'p_value': float(p_shapiro)
                }

            # Durbin-Watson
            try:
                from statsmodels.stats.diagnostic import durbin_watson
                dw = durbin_watson(residuals)
                results['durbin_watson'] = float(dw)
            except:
                pass
        except:
            pass

        return results

    def homogeneity_tests(self) -> Dict[str, Dict[str, float]]:
        """Tests de homogeneidad de varianza"""
        if self.get_analysis_type() != 'numeric-numeric':
            return {}

        x = self.df_clean[self.var1].values
        y = self.df_clean[self.var2].values

        results = {}

        # Dividir en grupos por cuartiles
        try:
            x_quartiles = pd.qcut(x, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            groups = [y[x_quartiles == q] for q in ['Q1', 'Q2', 'Q3', 'Q4'] if len(y[x_quartiles == q]) > 0]

            if len(groups) >= 2:
                # Test de Levene
                levene_stat, levene_p = stats.levene(*groups)
                results['levene'] = {'statistic': float(levene_stat), 'p_value': float(levene_p)}

                # Test de Bartlett
                bartlett_stat, bartlett_p = stats.bartlett(*groups)
                results['bartlett'] = {'statistic': float(bartlett_stat), 'p_value': float(bartlett_p)}
        except:
            pass

        return results

    # ==================== ANÁLISIS NUMÉRICO-CATEGÓRICO ====================

    def group_statistics(self) -> pd.DataFrame:
        """Estadísticos descriptivos por grupo"""
        if self.get_analysis_type() != 'numeric-categorical':
            return pd.DataFrame()

        # Determinar cuál es numérica y cuál categórica
        if self.var1_type == 'numeric':
            num_var, cat_var = self.var1, self.var2
        else:
            num_var, cat_var = self.var2, self.var1

        stats_by_group = self.df_clean.groupby(cat_var)[num_var].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)

        return stats_by_group

    def anova_test(self) -> Dict[str, Any]:
        """Realiza ANOVA o t-test según corresponda"""
        if self.get_analysis_type() != 'numeric-categorical':
            return {}

        # Determinar cuál es numérica y cuál categórica
        if self.var1_type == 'numeric':
            num_var, cat_var = self.var1, self.var2
        else:
            num_var, cat_var = self.var2, self.var1

        groups = [group_data[num_var].values for name, group_data in self.df_clean.groupby(cat_var)]
        n_groups = len(groups)

        results = {'n_groups': n_groups}

        if n_groups == 2:
            # Test t
            group1, group2 = groups[0], groups[1]

            # Test de Levene
            levene_stat, levene_p = stats.levene(group1, group2)
            results['levene'] = {'statistic': float(levene_stat), 'p_value': float(levene_p)}

            # Test t (con o sin varianzas iguales)
            equal_var = levene_p > 0.05
            t_stat, t_p = stats.ttest_ind(group1, group2, equal_var=equal_var)
            results['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'equal_var': equal_var
            }

            # Cohen's d
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) +
                                (len(group2) - 1) * np.var(group2, ddof=1)) /
                               (len(group1) + len(group2) - 2))
            cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
            results['cohens_d'] = float(cohens_d)

        elif n_groups > 2:
            # ANOVA
            f_stat, f_p = stats.f_oneway(*groups)
            results['anova'] = {'f_statistic': float(f_stat), 'p_value': float(f_p)}

            # Eta cuadrado
            ss_between = sum([len(group) * (np.mean(group) - np.mean(np.concatenate(groups)))**2
                            for group in groups])
            ss_total = sum([np.sum((group - np.mean(group))**2) for group in groups])
            eta_squared = ss_between / (ss_between + ss_total) if (ss_between + ss_total) > 0 else 0
            results['eta_squared'] = float(eta_squared)

        return results

    def nonparametric_tests(self) -> Dict[str, Any]:
        """Tests no paramétricos para comparación de grupos"""
        if self.get_analysis_type() != 'numeric-categorical':
            return {}

        # Determinar cuál es numérica y cuál categórica
        if self.var1_type == 'numeric':
            num_var, cat_var = self.var1, self.var2
        else:
            num_var, cat_var = self.var2, self.var1

        groups = [group_data[num_var].values for name, group_data in self.df_clean.groupby(cat_var)]
        n_groups = len(groups)

        results = {}

        if n_groups == 2:
            # Mann-Whitney U
            u_stat, u_p = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            results['mann_whitney'] = {
                'u_statistic': float(u_stat),
                'p_value': float(u_p)
            }

            # Tamaño del efecto (r)
            r_effect = u_stat / (len(groups[0]) * len(groups[1]))
            results['effect_size_r'] = float(r_effect)

        elif n_groups > 2:
            # Kruskal-Wallis
            h_stat, h_p = stats.kruskal(*groups)
            results['kruskal_wallis'] = {
                'h_statistic': float(h_stat),
                'p_value': float(h_p)
            }

            # Epsilon cuadrado
            n_total = sum([len(group) for group in groups])
            epsilon_sq = (h_stat - n_groups + 1) / (n_total - n_groups)
            results['epsilon_squared'] = float(epsilon_sq)

        return results

    # ==================== ANÁLISIS CATEGÓRICO-CATEGÓRICO ====================

    def contingency_table(self) -> pd.DataFrame:
        """Crea tabla de contingencia"""
        if self.get_analysis_type() != 'categorical-categorical':
            return pd.DataFrame()

        return pd.crosstab(self.df_clean[self.var1], self.df_clean[self.var2], margins=True)

    def chi_square_test(self) -> Dict[str, Any]:
        """Test Chi-cuadrado de independencia"""
        if self.get_analysis_type() != 'categorical-categorical':
            return {}

        contingency = pd.crosstab(self.df_clean[self.var1], self.df_clean[self.var2])

        chi2, p_value, dof, expected = chi2_contingency(contingency)

        results = {
            'chi_square': float(chi2),
            'p_value': float(p_value),
            'dof': int(dof),
            'expected_freq': expected.tolist()
        }

        # Test exacto de Fisher (para tablas 2x2)
        if contingency.shape == (2, 2):
            try:
                odds_ratio, fisher_p = stats.fisher_exact(contingency)
                results['fisher_exact'] = {
                    'odds_ratio': float(odds_ratio),
                    'p_value': float(fisher_p)
                }
            except:
                pass

        return results

    def association_measures(self) -> Dict[str, float]:
        """Medidas de asociación para variables categóricas"""
        if self.get_analysis_type() != 'categorical-categorical':
            return {}

        contingency = pd.crosstab(self.df_clean[self.var1], self.df_clean[self.var2])
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()

        measures = {}

        # Cramér's V
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        measures['cramers_v'] = float(cramers_v)

        # Coeficiente de contingencia
        contingency_coef = np.sqrt(chi2 / (chi2 + n))
        measures['contingency_coefficient'] = float(contingency_coef)

        # Tschuprow's T
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            r, c = contingency.shape
            tschuprow_t = np.sqrt(chi2 / (n * np.sqrt((r - 1) * (c - 1))))
            measures['tschuprow_t'] = float(tschuprow_t)

        return measures

    # ==================== ANÁLISIS GENERAL ====================

    def generate_summary(self) -> Dict[str, Any]:
        """Genera resumen ejecutivo del análisis bivariado"""
        analysis_type = self.get_analysis_type()

        summary = {
            'analysis_type': analysis_type,
            'var1': self.var1,
            'var2': self.var2,
            'n_observations': len(self.df_clean),
            'findings': [],
            'alerts': [],
            'recommendation': 'Unknown'
        }

        if analysis_type == 'numeric-numeric':
            corr = self.correlation_analysis()
            regression = self.regression_analysis()

            if 'pearson' in corr:
                r = corr['pearson']['r']
                p = corr['pearson']['p_value']

                # Interpretación de correlación
                if abs(r) < 0.1:
                    strength = "muy débil"
                elif abs(r) < 0.3:
                    strength = "débil"
                elif abs(r) < 0.5:
                    strength = "moderada"
                elif abs(r) < 0.7:
                    strength = "fuerte"
                else:
                    strength = "muy fuerte"

                direction = "positiva" if r > 0 else "negativa"
                summary['findings'].append(f"Correlación {strength} {direction} (r = {r:.3f})")

                if p < 0.05:
                    summary['findings'].append(f"Relación estadísticamente significativa (p = {p:.3f})")
                else:
                    summary['alerts'].append(f"Relación NO significativa (p = {p:.3f})")

            if 'r_squared' in regression:
                r2 = regression['r_squared']
                summary['findings'].append(f"R² = {r2:.3f} ({r2*100:.1f}% de varianza explicada)")

                if r2 < 0.1:
                    summary['alerts'].append("Bajo poder explicativo del modelo lineal")

            # Recomendación
            if 'pearson' in corr:
                r = abs(corr['pearson']['r'])
                p = corr['pearson']['p_value']
                if r > 0.5 and p < 0.05:
                    summary['recommendation'] = "Excelente"
                elif r > 0.3 and p < 0.05:
                    summary['recommendation'] = "Buena"
                elif p < 0.05:
                    summary['recommendation'] = "Revisar"
                else:
                    summary['recommendation'] = "No significativa"

        elif analysis_type == 'numeric-categorical':
            anova = self.anova_test()

            if 't_test' in anova:
                p = anova['t_test']['p_value']
                if p < 0.05:
                    summary['findings'].append(f"Diferencia significativa entre grupos (p = {p:.3f})")
                    summary['recommendation'] = "Significativa"
                else:
                    summary['alerts'].append(f"No hay diferencia significativa (p = {p:.3f})")
                    summary['recommendation'] = "No significativa"

            elif 'anova' in anova:
                p = anova['anova']['p_value']
                if p < 0.05:
                    summary['findings'].append(f"Diferencias significativas entre grupos (p = {p:.3f})")
                    summary['recommendation'] = "Significativa"
                else:
                    summary['alerts'].append(f"No hay diferencias significativas (p = {p:.3f})")
                    summary['recommendation'] = "No significativa"

        elif analysis_type == 'categorical-categorical':
            chi2_result = self.chi_square_test()
            assoc = self.association_measures()

            if 'p_value' in chi2_result:
                p = chi2_result['p_value']
                if p < 0.05:
                    summary['findings'].append(f"Asociación significativa (p = {p:.3f})")
                else:
                    summary['alerts'].append(f"No hay asociación significativa (p = {p:.3f})")

            if 'cramers_v' in assoc:
                v = assoc['cramers_v']
                if v < 0.1:
                    strength = "muy débil"
                elif v < 0.3:
                    strength = "débil"
                elif v < 0.5:
                    strength = "moderada"
                else:
                    strength = "fuerte"

                summary['findings'].append(f"Fuerza de asociación {strength} (V = {v:.3f})")

            # Recomendación
            if 'p_value' in chi2_result and chi2_result['p_value'] < 0.05:
                if 'cramers_v' in assoc and assoc['cramers_v'] > 0.3:
                    summary['recommendation'] = "Fuerte"
                else:
                    summary['recommendation'] = "Moderada"
            else:
                summary['recommendation'] = "No significativa"

        return summary


# ==================== FUNCIONES DE CONVENIENCIA ====================

def analyze_numeric_numeric(df: pd.DataFrame, var1: str, var2: str) -> Dict[str, Any]:
    """Análisis completo de dos variables numéricas"""
    analyzer = BivariateAnalyzer(df, var1, var2)

    if analyzer.get_analysis_type() != 'numeric-numeric':
        return {'error': 'Both variables must be numeric'}

    return {
        'analysis_type': 'numeric-numeric',
        'correlation': analyzer.correlation_analysis(),
        'regression': analyzer.regression_analysis(),
        'homogeneity': analyzer.homogeneity_tests(),
        'summary': analyzer.generate_summary()
    }


def analyze_numeric_categorical(df: pd.DataFrame, num_var: str, cat_var: str) -> Dict[str, Any]:
    """Análisis completo de variable numérica vs categórica"""
    analyzer = BivariateAnalyzer(df, num_var, cat_var)

    if analyzer.get_analysis_type() != 'numeric-categorical':
        return {'error': 'Variables must be numeric and categorical'}

    return {
        'analysis_type': 'numeric-categorical',
        'group_statistics': analyzer.group_statistics().to_dict(),
        'anova': analyzer.anova_test(),
        'nonparametric': analyzer.nonparametric_tests(),
        'summary': analyzer.generate_summary()
    }


def analyze_categorical_categorical(df: pd.DataFrame, var1: str, var2: str) -> Dict[str, Any]:
    """Análisis completo de dos variables categóricas"""
    analyzer = BivariateAnalyzer(df, var1, var2)

    if analyzer.get_analysis_type() != 'categorical-categorical':
        return {'error': 'Both variables must be categorical'}

    return {
        'analysis_type': 'categorical-categorical',
        'contingency_table': analyzer.contingency_table().to_dict(),
        'chi_square': analyzer.chi_square_test(),
        'association': analyzer.association_measures(),
        'summary': analyzer.generate_summary()
    }
