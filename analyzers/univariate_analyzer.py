"""
Univariate analysis module
Refactored to use the new architecture
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
import warnings

from core.base_analyzer import BaseAnalyzer, AnalysisResult

warnings.filterwarnings('ignore')


class UnivariateAnalyzer(BaseAnalyzer):
    """Univariate analysis analyzer"""
    
    def __init__(self):
        super().__init__("UnivariateAnalyzer")
    
    def analyze(self, variable: str, analysis_type: str = "descriptive", **kwargs) -> AnalysisResult:
        """Perform univariate analysis"""
        if not self.validate_data():
            return AnalysisResult(success=False, error_message="No data available")
        
        if variable not in self.data.columns:
            return AnalysisResult(success=False, error_message=f"Variable '{variable}' not found")
        
        try:
            if analysis_type == "descriptive":
                return self._descriptive_analysis(variable)
            elif analysis_type == "distribution":
                return self._distribution_analysis(variable)
            elif analysis_type == "outliers":
                return self._outlier_analysis(variable)
            elif analysis_type == "transformations":
                return self._transformation_analysis(variable)
            else:
                return AnalysisResult(success=False, error_message=f"Unknown analysis type: {analysis_type}")
                
        except Exception as e:
            return AnalysisResult(success=False, error_message=str(e))
    
    def _descriptive_analysis(self, variable: str) -> AnalysisResult:
        """Perform descriptive statistics analysis"""
        series = self.data[variable]
        
        if self.column_types.get(variable) == 'quantitative':
            # Numeric analysis
            stats_dict = {
                'count': series.count(),
                'mean': series.mean(),
                'median': series.median(),
                'mode': series.mode().iloc[0] if not series.mode().empty else None,
                'std': series.std(),
                'var': series.var(),
                'min': series.min(),
                'max': series.max(),
                'range': series.max() - series.min(),
                'q25': series.quantile(0.25),
                'q75': series.quantile(0.75),
                'iqr': series.quantile(0.75) - series.quantile(0.25),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'missing': series.isna().sum(),
                'missing_pct': (series.isna().sum() / len(series)) * 100
            }
        else:
            # Categorical analysis
            value_counts = series.value_counts()
            stats_dict = {
                'count': series.count(),
                'unique': series.nunique(),
                'most_frequent': value_counts.index[0] if not value_counts.empty else None,
                'most_frequent_count': value_counts.iloc[0] if not value_counts.empty else 0,
                'most_frequent_pct': (value_counts.iloc[0] / len(series)) * 100 if not value_counts.empty else 0,
                'missing': series.isna().sum(),
                'missing_pct': (series.isna().sum() / len(series)) * 100,
                'value_counts': value_counts.head(10).to_dict()
            }
        
        return AnalysisResult(
            success=True,
            data=stats_dict,
            metadata={'variable': variable, 'type': self.column_types.get(variable)}
        )
    
    def _distribution_analysis(self, variable: str) -> AnalysisResult:
        """Analyze distribution of variable"""
        series = self.data[variable].dropna()
        
        if self.column_types.get(variable) == 'quantitative':
            # Numeric distribution
            distribution_info = {
                'is_normal': self._test_normality(series),
                'shapiro_stat': None,
                'shapiro_pvalue': None,
                'ks_stat': None,
                'ks_pvalue': None
            }
            
            # Shapiro-Wilk test (for small samples)
            if len(series) <= 5000:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(series)
                    distribution_info['shapiro_stat'] = shapiro_stat
                    distribution_info['shapiro_pvalue'] = shapiro_p
                except:
                    pass
            
            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_p = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
                distribution_info['ks_stat'] = ks_stat
                distribution_info['ks_pvalue'] = ks_p
            except:
                pass
            
        else:
            # Categorical distribution
            value_counts = series.value_counts()
            total = len(series)
            
            distribution_info = {
                'entropy': self._calculate_entropy(series),
                'gini_coefficient': self._calculate_gini_coefficient(series),
                'concentration_ratio': (value_counts.iloc[0] / total) if not value_counts.empty else 0,
                'effective_categories': self._calculate_effective_categories(series)
            }
        
        return AnalysisResult(
            success=True,
            data=distribution_info,
            metadata={'variable': variable, 'type': self.column_types.get(variable)}
        )
    
    def _outlier_analysis(self, variable: str) -> AnalysisResult:
        """Analyze outliers in variable"""
        if self.column_types.get(variable) != 'quantitative':
            return AnalysisResult(success=False, error_message="Outlier analysis only for quantitative variables")
        
        series = self.data[variable].dropna()
        
        # IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        # Z-score method
        z_scores = np.abs(stats.zscore(series))
        z_outliers = series[z_scores > 3]
        
        # Modified Z-score method
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        modified_z_outliers = series[np.abs(modified_z_scores) > 3.5]
        
        outlier_info = {
            'iqr_outliers': {
                'count': len(iqr_outliers),
                'percentage': (len(iqr_outliers) / len(series)) * 100,
                'values': iqr_outliers.tolist()
            },
            'z_outliers': {
                'count': len(z_outliers),
                'percentage': (len(z_outliers) / len(series)) * 100,
                'values': z_outliers.tolist()
            },
            'modified_z_outliers': {
                'count': len(modified_z_outliers),
                'percentage': (len(modified_z_outliers) / len(series)) * 100,
                'values': modified_z_outliers.tolist()
            },
            'bounds': {
                'iqr_lower': lower_bound,
                'iqr_upper': upper_bound,
                'z_threshold': 3,
                'modified_z_threshold': 3.5
            }
        }
        
        return AnalysisResult(
            success=True,
            data=outlier_info,
            metadata={'variable': variable, 'type': 'quantitative'}
        )
    
    def _transformation_analysis(self, variable: str) -> AnalysisResult:
        """Analyze transformations for variable"""
        if self.column_types.get(variable) != 'quantitative':
            return AnalysisResult(success=False, error_message="Transformation analysis only for quantitative variables")
        
        series = self.data[variable].dropna()
        
        # Only analyze positive values for log transformations
        positive_series = series[series > 0]
        
        transformations = {}
        
        if len(positive_series) > 0:
            # Log transformation
            log_series = np.log(positive_series)
            transformations['log'] = {
                'skewness': log_series.skew(),
                'kurtosis': log_series.kurtosis(),
                'is_normal': self._test_normality(log_series)
            }
            
            # Square root transformation
            sqrt_series = np.sqrt(positive_series)
            transformations['sqrt'] = {
                'skewness': sqrt_series.skew(),
                'kurtosis': sqrt_series.kurtosis(),
                'is_normal': self._test_normality(sqrt_series)
            }
        
        # Box-Cox transformation
        try:
            boxcox_series, lambda_param = stats.boxcox(positive_series)
            transformations['boxcox'] = {
                'lambda': lambda_param,
                'skewness': boxcox_series.skew(),
                'kurtosis': boxcox_series.kurtosis(),
                'is_normal': self._test_normality(boxcox_series)
            }
        except:
            transformations['boxcox'] = {'error': 'Could not compute Box-Cox transformation'}
        
        # Original series stats
        transformations['original'] = {
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'is_normal': self._test_normality(series)
        }
        
        return AnalysisResult(
            success=True,
            data=transformations,
            metadata={'variable': variable, 'type': 'quantitative'}
        )
    
    def _test_normality(self, series: pd.Series) -> bool:
        """Test if series is normally distributed"""
        try:
            if len(series) < 3:
                return False
            
            # Use Shapiro-Wilk for small samples, KS test for large samples
            if len(series) <= 5000:
                stat, p_value = stats.shapiro(series)
            else:
                stat, p_value = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
            
            return p_value > 0.05
        except:
            return False
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of categorical variable"""
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _calculate_gini_coefficient(self, series: pd.Series) -> float:
        """Calculate Gini coefficient for categorical variable"""
        value_counts = series.value_counts()
        n = len(series)
        gini = 1 - np.sum((value_counts / n) ** 2)
        return gini
    
    def _calculate_effective_categories(self, series: pd.Series) -> float:
        """Calculate effective number of categories"""
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        return 1 / np.sum(probabilities ** 2)
    
    def create_visualization(self, variable: str, plot_type: str = "histogram", **kwargs) -> AnalysisResult:
        """Create visualization for variable"""
        if not self.validate_data():
            return AnalysisResult(success=False, error_message="No data available")
        
        if variable not in self.data.columns:
            return AnalysisResult(success=False, error_message=f"Variable '{variable}' not found")
        
        try:
            series = self.data[variable].dropna()
            
            if plot_type == "histogram":
                return self._create_histogram(series, variable, **kwargs)
            elif plot_type == "boxplot":
                return self._create_boxplot(series, variable, **kwargs)
            elif plot_type == "barplot":
                return self._create_barplot(series, variable, **kwargs)
            elif plot_type == "qqplot":
                return self._create_qqplot(series, variable, **kwargs)
            else:
                return AnalysisResult(success=False, error_message=f"Unknown plot type: {plot_type}")
                
        except Exception as e:
            return AnalysisResult(success=False, error_message=str(e))
    
    def _create_histogram(self, series: pd.Series, variable: str, **kwargs) -> AnalysisResult:
        """Create histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.column_types.get(variable) == 'quantitative':
            ax.hist(series, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel(variable)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of {variable}')
        else:
            value_counts = series.value_counts().head(20)
            ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_xlabel(variable)
            ax.set_ylabel('Count')
            ax.set_title(f'Bar Chart of {variable}')
        
        plt.tight_layout()
        
        return AnalysisResult(
            success=True,
            data=fig,
            metadata={'variable': variable, 'plot_type': 'histogram'}
        )
    
    def _create_boxplot(self, series: pd.Series, variable: str, **kwargs) -> AnalysisResult:
        """Create boxplot"""
        if self.column_types.get(variable) != 'quantitative':
            return AnalysisResult(success=False, error_message="Boxplot only for quantitative variables")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(series, vert=True)
        ax.set_ylabel(variable)
        ax.set_title(f'Boxplot of {variable}')
        
        plt.tight_layout()
        
        return AnalysisResult(
            success=True,
            data=fig,
            metadata={'variable': variable, 'plot_type': 'boxplot'}
        )
    
    def _create_barplot(self, series: pd.Series, variable: str, **kwargs) -> AnalysisResult:
        """Create barplot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        value_counts = series.value_counts().head(20)
        ax.bar(range(len(value_counts)), value_counts.values)
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.set_xlabel(variable)
        ax.set_ylabel('Count')
        ax.set_title(f'Bar Chart of {variable}')
        
        plt.tight_layout()
        
        return AnalysisResult(
            success=True,
            data=fig,
            metadata={'variable': variable, 'plot_type': 'barplot'}
        )
    
    def _create_qqplot(self, series: pd.Series, variable: str, **kwargs) -> AnalysisResult:
        """Create Q-Q plot"""
        if self.column_types.get(variable) != 'quantitative':
            return AnalysisResult(success=False, error_message="Q-Q plot only for quantitative variables")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(series, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot of {variable}')
        
        plt.tight_layout()
        
        return AnalysisResult(
            success=True,
            data=fig,
            metadata={'variable': variable, 'plot_type': 'qqplot'}
        )
