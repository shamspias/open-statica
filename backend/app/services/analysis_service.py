"""
Analysis Service for OpenStatica
Coordinates statistical analyses and manages results
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import pingouin as pg
from datetime import datetime
import asyncio
import logging
from app.core.base import Result

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for coordinating statistical analyses"""

    def __init__(self):
        self.analysis_cache = {}
        self.supported_tests = {
            'descriptive': self.run_descriptive_analysis,
            'frequency': self.run_frequency_analysis,
            'correlation': self.run_correlation_analysis,
            'ttest': self.run_ttest,
            'anova': self.run_anova,
            'chi_square': self.run_chi_square,
            'regression': self.run_regression,
            'factor_analysis': self.run_factor_analysis,
            'pca': self.run_pca,
            'time_series': self.run_time_series_analysis
        }

    async def analyze(self,
                      data: pd.DataFrame,
                      analysis_type: str,
                      variables: List[str],
                      options: Dict[str, Any] = None) -> Result:
        """Main entry point for analyses"""
        options = options or {}

        if analysis_type not in self.supported_tests:
            return Result.fail(f"Unsupported analysis type: {analysis_type}")

        try:
            analysis_func = self.supported_tests[analysis_type]
            result = await analysis_func(data, variables, options)

            # Cache result
            cache_key = self._generate_cache_key(analysis_type, variables, options)
            self.analysis_cache[cache_key] = result

            return Result.ok(result)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return Result.fail(str(e))

    async def run_descriptive_analysis(self,
                                       data: pd.DataFrame,
                                       variables: List[str],
                                       options: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive descriptive statistics"""
        include_advanced = options.get('include_advanced', True)
        confidence_level = options.get('confidence_level', 0.95)

        results = {}

        for var in variables:
            if var not in data.columns:
                continue

            col_data = data[var].dropna()

            if pd.api.types.is_numeric_dtype(col_data):
                # Basic statistics
                stats_dict = {
                    'n': len(col_data),
                    'missing': data[var].isnull().sum(),
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'mode': float(col_data.mode()[0]) if len(col_data.mode()) > 0 else None,
                    'std': float(col_data.std()),
                    'variance': float(col_data.var()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'range': float(col_data.max() - col_data.min()),
                    'q1': float(col_data.quantile(0.25)),
                    'q3': float(col_data.quantile(0.75)),
                    'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25))
                }

                if include_advanced:
                    # Advanced statistics
                    stats_dict.update({
                        'skewness': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis()),
                        'sem': float(col_data.sem()),
                        'mad': float(col_data.mad()),
                        'cv': float(col_data.std() / col_data.mean() * 100) if col_data.mean() != 0 else None,
                        'geometric_mean': float(stats.gmean(col_data[col_data > 0])) if (col_data > 0).any() else None,
                        'harmonic_mean': float(stats.hmean(col_data[col_data > 0])) if (col_data > 0).any() else None,
                        'trimmed_mean': float(stats.trim_mean(col_data, 0.1))
                    })

                    # Confidence interval
                    ci = stats.t.interval(
                        confidence_level,
                        len(col_data) - 1,
                        loc=col_data.mean(),
                        scale=col_data.sem()
                    )
                    stats_dict['confidence_interval'] = [float(ci[0]), float(ci[1])]

                    # Normality tests
                    if len(col_data) >= 3:
                        shapiro_stat, shapiro_p = stats.shapiro(col_data[:5000] if len(col_data) > 5000 else col_data)
                        stats_dict['normality'] = {
                            'shapiro_wilk': {
                                'statistic': float(shapiro_stat),
                                'p_value': float(shapiro_p),
                                'is_normal': shapiro_p > 0.05
                            }
                        }

                    # Outliers detection
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

                    stats_dict['outliers'] = {
                        'count': len(outliers),
                        'percentage': len(outliers) / len(col_data) * 100,
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }

                results[var] = stats_dict

            else:
                # Categorical variable
                value_counts = data[var].value_counts()
                results[var] = {
                    'type': 'categorical',
                    'n': len(col_data),
                    'missing': data[var].isnull().sum(),
                    'unique': data[var].nunique(),
                    'mode': value_counts.index[0] if len(value_counts) > 0 else None,
                    'mode_freq': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'value_counts': value_counts.head(20).to_dict(),
                    'entropy': float(stats.entropy(value_counts))
                }

        return results

    async def run_frequency_analysis(self,
                                     data: pd.DataFrame,
                                     variables: List[str],
                                     options: Dict[str, Any]) -> Dict[str, Any]:
        """Run frequency analysis"""
        results = {}

        for var in variables:
            if var not in data.columns:
                continue

            if pd.api.types.is_numeric_dtype(data[var]):
                # Create frequency bins for numeric data
                n_bins = options.get('n_bins', 10)
                col_data = data[var].dropna()

                counts, bin_edges = np.histogram(col_data, bins=n_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                results[var] = {
                    'type': 'numeric',
                    'bins': bin_centers.tolist(),
                    'frequencies': counts.tolist(),
                    'relative_frequencies': (counts / len(col_data)).tolist(),
                    'cumulative_frequencies': np.cumsum(counts).tolist(),
                    'bin_edges': bin_edges.tolist()
                }
            else:
                # Categorical frequency
                value_counts = data[var].value_counts()
                value_pcts = data[var].value_counts(normalize=True) * 100

                results[var] = {
                    'type': 'categorical',
                    'categories': value_counts.index.tolist(),
                    'frequencies': value_counts.values.tolist(),
                    'percentages': value_pcts.values.tolist(),
                    'cumulative_percentages': value_pcts.cumsum().values.tolist()
                }

        return results

    async def run_correlation_analysis(self,
                                       data: pd.DataFrame,
                                       variables: List[str],
                                       options: Dict[str, Any]) -> Dict[str, Any]:
        """Run correlation analysis"""
        method = options.get('method', 'pearson')
        include_p_values = options.get('include_p_values', True)

        # Filter numeric variables
        numeric_vars = [v for v in variables if v in data.columns and pd.api.types.is_numeric_dtype(data[v])]

        if len(numeric_vars) < 2:
            raise ValueError("Need at least 2 numeric variables for correlation analysis")

        # Calculate correlation matrix
        corr_matrix = data[numeric_vars].corr(method=method)

        results = {
            'correlation_matrix': corr_matrix.to_dict(),
            'method': method,
            'n_observations': len(data[numeric_vars].dropna())
        }

        # Calculate p-values if requested
        if include_p_values:
            p_values = pd.DataFrame(
                np.zeros((len(numeric_vars), len(numeric_vars))),
                columns=numeric_vars,
                index=numeric_vars
            )

            for i, var1 in enumerate(numeric_vars):
                for j, var2 in enumerate(numeric_vars):
                    if i != j:
                        if method == 'pearson':
                            _, p = stats.pearsonr(
                                data[var1].dropna(),
                                data[var2].dropna()
                            )
                        elif method == 'spearman':
                            _, p = stats.spearmanr(
                                data[var1].dropna(),
                                data[var2].dropna()
                            )
                        elif method == 'kendall':
                            _, p = stats.kendalltau(
                                data[var1].dropna(),
                                data[var2].dropna()
                            )
                        p_values.iloc[i, j] = p

            results['p_values'] = p_values.to_dict()

        # Find significant correlations
        significant_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5 and abs(corr_val) < 1:
                    significant_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })

        results['significant_correlations'] = significant_corr

        return results

    async def run_ttest(self,
                        data: pd.DataFrame,
                        variables: List[str],
                        options: Dict[str, Any]) -> Dict[str, Any]:
        """Run t-test analysis"""
        test_type = options.get('test_type', 'independent')
        alpha = options.get('alpha', 0.05)

        results = {
            'test_type': test_type,
            'alpha': alpha
        }

        if test_type == 'one_sample':
            # One-sample t-test
            test_value = options.get('test_value', 0)
            var = variables[0]

            sample_data = data[var].dropna()
            t_stat, p_value = stats.ttest_1samp(sample_data, test_value)

            # Effect size (Cohen's d)
            cohens_d = (sample_data.mean() - test_value) / sample_data.std()

            results.update({
                'variable': var,
                'test_value': test_value,
                'sample_mean': float(sample_data.mean()),
                'sample_std': float(sample_data.std()),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'significant': p_value < alpha,
                'conclusion': f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis"
            })

        elif test_type == 'independent':
            # Independent samples t-test
            if len(variables) == 2:
                # Two numeric variables
                var1_data = data[variables[0]].dropna()
                var2_data = data[variables[1]].dropna()
            else:
                # One numeric, one grouping variable
                numeric_var = options.get('numeric_var')
                group_var = options.get('group_var')

                groups = data[group_var].unique()
                if len(groups) != 2:
                    raise ValueError(f"Expected 2 groups, found {len(groups)}")

                var1_data = data[data[group_var] == groups[0]][numeric_var].dropna()
                var2_data = data[data[group_var] == groups[1]][numeric_var].dropna()

            # Test for equal variances
            levene_stat, levene_p = stats.levene(var1_data, var2_data)
            equal_var = levene_p > 0.05

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(var1_data, var2_data, equal_var=equal_var)

            # Effect size
            pooled_std = np.sqrt(
                ((len(var1_data) - 1) * var1_data.var() +
                 (len(var2_data) - 1) * var2_data.var()) /
                (len(var1_data) + len(var2_data) - 2)
            )
            cohens_d = (var1_data.mean() - var2_data.mean()) / pooled_std

            results.update({
                'group1_mean': float(var1_data.mean()),
                'group1_std': float(var1_data.std()),
                'group1_n': len(var1_data),
                'group2_mean': float(var2_data.mean()),
                'group2_std': float(var2_data.std()),
                'group2_n': len(var2_data),
                'mean_difference': float(var1_data.mean() - var2_data.mean()),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'levene_test': {
                    'statistic': float(levene_stat),
                    'p_value': float(levene_p),
                    'equal_variances': equal_var
                },
                'cohens_d': float(cohens_d),
                'significant': p_value < alpha
            })

        elif test_type == 'paired':
            # Paired samples t-test
            if len(variables) != 2:
                raise ValueError("Paired t-test requires exactly 2 variables")

            var1_data = data[variables[0]].dropna()
            var2_data = data[variables[1]].dropna()

            # Ensure same length
            paired_data = pd.DataFrame({
                'var1': var1_data,
                'var2': var2_data
            }).dropna()

            differences = paired_data['var2'] - paired_data['var1']

            t_stat, p_value = stats.ttest_rel(paired_data['var1'], paired_data['var2'])

            # Effect size
            cohens_d = differences.mean() / differences.std()

            results.update({
                'var1_mean': float(paired_data['var1'].mean()),
                'var2_mean': float(paired_data['var2'].mean()),
                'mean_difference': float(differences.mean()),
                'std_difference': float(differences.std()),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'significant': p_value < alpha
            })

        return results

    async def run_anova(self,
                        data: pd.DataFrame,
                        variables: List[str],
                        options: Dict[str, Any]) -> Dict[str, Any]:
        """Run ANOVA analysis"""
        anova_type = options.get('type', 'one_way')
        alpha = options.get('alpha', 0.05)
        post_hoc = options.get('post_hoc', True)

        results = {
            'type': anova_type,
            'alpha': alpha
        }

        if anova_type == 'one_way':
            dependent_var = options.get('dependent')
            factor_var = options.get('factor')

            # Prepare data for ANOVA
            groups = []
            group_labels = []

            for group in data[factor_var].unique():
                group_data = data[data[factor_var] == group][dependent_var].dropna()
                if len(group_data) > 0:
                    groups.append(group_data.values)
                    group_labels.append(str(group))

            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)

            # Calculate effect size (eta-squared)
            grand_mean = data[dependent_var].mean()
            ss_total = ((data[dependent_var] - grand_mean) ** 2).sum()
            ss_between = sum(
                len(g) * (np.mean(g) - grand_mean) ** 2
                for g in groups
            )
            eta_squared = ss_between / ss_total

            results.update({
                'dependent': dependent_var,
                'factor': factor_var,
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'eta_squared': float(eta_squared),
                'significant': p_value < alpha,
                'n_groups': len(groups),
                'group_means': {
                    label: float(np.mean(g))
                    for label, g in zip(group_labels, groups)
                }
            })

            # Post-hoc tests if significant
            if post_hoc and p_value < alpha and len(groups) > 2:
                # Tukey HSD
                df_long = data[[dependent_var, factor_var]].dropna()
                tukey_result = pairwise_tukeyhsd(
                    df_long[dependent_var],
                    df_long[factor_var],
                    alpha=alpha
                )

                results['post_hoc'] = {
                    'method': 'Tukey HSD',
                    'results': str(tukey_result)
                }

        elif anova_type == 'two_way':
            dependent_var = options.get('dependent')
            factor1 = options.get('factor1')
            factor2 = options.get('factor2')

            # Use pingouin for two-way ANOVA
            aov = pg.anova(
                data=data,
                dv=dependent_var,
                between=[factor1, factor2]
            )

            results.update({
                'dependent': dependent_var,
                'factors': [factor1, factor2],
                'anova_table': aov.to_dict(orient='records')
            })

        return results

    async def run_chi_square(self,
                             data: pd.DataFrame,
                             variables: List[str],
                             options: Dict[str, Any]) -> Dict[str, Any]:
        """Run chi-square test"""
        if len(variables) != 2:
            raise ValueError("Chi-square test requires exactly 2 categorical variables")

        # Create contingency table
        contingency_table = pd.crosstab(data[variables[0]], data[variables[1]])

        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # Calculate Cramér's V for effect size
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim))

        results = {
            'variables': variables,
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'cramers_v': float(cramers_v),
            'significant': p_value < options.get('alpha', 0.05),
            'contingency_table': contingency_table.to_dict(),
            'expected_frequencies': expected.tolist()
        }

        return results

    async def run_regression(self,
                             data: pd.DataFrame,
                             variables: List[str],
                             options: Dict[str, Any]) -> Dict[str, Any]:
        """Run regression analysis"""
        regression_type = options.get('type', 'linear')
        dependent_var = options.get('dependent')
        independent_vars = options.get('independent', [v for v in variables if v != dependent_var])

        # Prepare data
        df = data[[dependent_var] + independent_vars].dropna()
        X = df[independent_vars]
        y = df[dependent_var]

        # Add constant
        X = sm.add_constant(X)

        results = {
            'type': regression_type,
            'dependent': dependent_var,
            'independent': independent_vars,
            'n_observations': len(df)
        }

        if regression_type == 'linear':
            # OLS regression
            model = sm.OLS(y, X).fit()

            results.update({
                'coefficients': model.params.to_dict(),
                'std_errors': model.bse.to_dict(),
                't_values': model.tvalues.to_dict(),
                'p_values': model.pvalues.to_dict(),
                'r_squared': float(model.rsquared),
                'adjusted_r_squared': float(model.rsquared_adj),
                'f_statistic': float(model.fvalue),
                'f_pvalue': float(model.f_pvalue),
                'aic': float(model.aic),
                'bic': float(model.bic)
            })

            # Diagnostic tests
            residuals = model.resid
            fitted = model.fittedvalues

            # Durbin-Watson for autocorrelation
            dw = durbin_watson(residuals)

            # Breusch-Pagan for heteroscedasticity
            bp_test = het_breuschpagan(residuals, X)

            # Jarque-Bera for normality
            jb_test = stats.jarque_bera(residuals)

            results['diagnostics'] = {
                'durbin_watson': float(dw),
                'breusch_pagan': {
                    'statistic': float(bp_test[0]),
                    'p_value': float(bp_test[1])
                },
                'jarque_bera': {
                    'statistic': float(jb_test[0]),
                    'p_value': float(jb_test[1])
                }
            }

        elif regression_type == 'logistic':
            # Logistic regression
            model = sm.Logit(y, X).fit()

            results.update({
                'coefficients': model.params.to_dict(),
                'std_errors': model.bse.to_dict(),
                'z_values': model.tvalues.to_dict(),
                'p_values': model.pvalues.to_dict(),
                'pseudo_r_squared': float(model.prsquared),
                'log_likelihood': float(model.llf),
                'aic': float(model.aic),
                'bic': float(model.bic)
            })

        return results

    async def run_factor_analysis(self,
                                  data: pd.DataFrame,
                                  variables: List[str],
                                  options: Dict[str, Any]) -> Dict[str, Any]:
        """Run factor analysis"""
        from factor_analyzer import FactorAnalyzer
        from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

        n_factors = options.get('n_factors', 3)
        rotation = options.get('rotation', 'varimax')

        # Prepare data
        df = data[variables].dropna()

        # Test assumptions
        chi_square, p_value = calculate_bartlett_sphericity(df)
        kmo_all, kmo_model = calculate_kmo(df)

        # Perform factor analysis
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
        fa.fit(df)

        # Get results
        loadings = fa.loadings_
        communalities = fa.get_communalities()
        eigenvalues = fa.get_eigenvalues()[0]
        variance = fa.get_factor_variance()

        results = {
            'n_factors': n_factors,
            'rotation': rotation,
            'bartlett_test': {
                'chi_square': float(chi_square),
                'p_value': float(p_value)
            },
            'kmo': float(kmo_model),
            'loadings': pd.DataFrame(
                loadings,
                columns=[f'Factor{i + 1}' for i in range(n_factors)],
                index=variables
            ).to_dict(),
            'communalities': dict(zip(variables, communalities)),
            'eigenvalues': eigenvalues.tolist(),
            'variance_explained': {
                'SS_loadings': variance[0].tolist(),
                'proportion_var': variance[1].tolist(),
                'cumulative_var': variance[2].tolist()
            }
        }

        return results

    async def run_pca(self,
                      data: pd.DataFrame,
                      variables: List[str],
                      options: Dict[str, Any]) -> Dict[str, Any]:
        """Run Principal Component Analysis"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        n_components = options.get('n_components', min(len(variables), 5))
        standardize = options.get('standardize', True)

        # Prepare data
        df = data[variables].dropna()

        if standardize:
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
        else:
            df_scaled = df.values

        # Perform PCA
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(df_scaled)

        # Calculate loadings
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        results = {
            'n_components': n_components,
            'explained_variance': pca.explained_variance_.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'loadings': pd.DataFrame(
                loadings,
                columns=[f'PC{i + 1}' for i in range(n_components)],
                index=variables
            ).to_dict(),
            'scores': pd.DataFrame(
                scores,
                columns=[f'PC{i + 1}' for i in range(n_components)]
            ).head(100).to_dict()  # First 100 scores
        }

        return results

    async def run_time_series_analysis(self,
                                       data: pd.DataFrame,
                                       variables: List[str],
                                       options: Dict[str, Any]) -> Dict[str, Any]:
        """Run time series analysis"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import adfuller, acf, pacf

        time_var = options.get('time_var')
        value_var = variables[0]
        period = options.get('period', 12)

        # Prepare time series
        if time_var:
            df = data.set_index(time_var)
        else:
            df = data

        ts = df[value_var].dropna()

        results = {
            'variable': value_var,
            'n_observations': len(ts)
        }

        # Stationarity test (ADF)
        adf_result = adfuller(ts, autolag='AIC')
        results['adf_test'] = {
            'statistic': float(adf_result[0]),
            'p_value': float(adf_result[1]),
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }

        # Decomposition
        if len(ts) >= 2 * period:
            decomposition = seasonal_decompose(ts, model='additive', period=period)

            results['decomposition'] = {
                'trend': decomposition.trend.dropna().head(50).tolist(),
                'seasonal': decomposition.seasonal.dropna().head(50).tolist(),
                'residual': decomposition.resid.dropna().head(50).tolist()
            }

        # ACF and PACF
        nlags = min(40, len(ts) // 4)
        acf_values = acf(ts, nlags=nlags)
        pacf_values = pacf(ts, nlags=nlags)

        results['autocorrelation'] = {
            'acf': acf_values.tolist(),
            'pacf': pacf_values.tolist(),
            'nlags': nlags
        }

        return results

    def _generate_cache_key(self, analysis_type: str, variables: List[str], options: Dict) -> str:
        """Generate cache key for analysis results"""
        import hashlib
        import json

        key_data = {
            'type': analysis_type,
            'variables': sorted(variables),
            'options': options
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def export_results(self, results: Dict[str, Any], format: str = 'json') -> bytes:
        """Export analysis results"""
        import json
        import io

        if format == 'json':
            return json.dumps(results, indent=2).encode('utf-8')

        elif format == 'html':
            html = self._results_to_html(results)
            return html.encode('utf-8')

        elif format == 'pdf':
            # Would need reportlab or similar
            pass

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _results_to_html(self, results: Dict[str, Any]) -> str:
        """Convert results to HTML format"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>OpenStatica Analysis Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #6366f1; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Analysis Results</h1>
        """

        html += self._dict_to_html_table(results)
        html += "</body></html>"

        return html

    def _dict_to_html_table(self, d: Dict, level: int = 0) -> str:
        """Convert dictionary to HTML table"""
        html = "<table>"

        for key, value in d.items():
            html += f"<tr><td><strong>{key}</strong></td><td>"

            if isinstance(value, dict):
                html += self._dict_to_html_table(value, level + 1)
            elif isinstance(value, list):
                html += ", ".join(str(v) for v in value[:10])
                if len(value) > 10:
                    html += "..."
            else:
                html += str(value)

            html += "</td></tr>"

        html += "</table>"
        return html
