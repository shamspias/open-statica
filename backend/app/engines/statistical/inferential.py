from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
from app.core.base import BaseAnalyzer, Result


class TTestAnalyzer(BaseAnalyzer):
    """Analyzer for T-tests"""

    def __init__(self):
        super().__init__("ttest")

    async def execute(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Perform T-test"""
        try:
            test_type = params.get("test_type", "independent")

            if test_type == "one_sample":
                return await self._one_sample_ttest(data, params)
            elif test_type == "independent":
                return await self._independent_ttest(data, params)
            elif test_type == "paired":
                return await self._paired_ttest(data, params)
            else:
                return Result.fail(f"Unknown test type: {test_type}")

        except Exception as e:
            return Result.fail(str(e))

    async def _one_sample_ttest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """One-sample T-test"""
        column = params.get("column")
        test_value = params.get("test_value", 0)
        alternative = params.get("alternative", "two-sided")

        sample_data = data[column].dropna()

        # Perform test
        t_stat, p_value = stats.ttest_1samp(sample_data, test_value)

        # Calculate effect size (Cohen's d)
        cohens_d = (sample_data.mean() - test_value) / sample_data.std()

        # Confidence interval
        ci = stats.t.interval(0.95, len(sample_data) - 1,
                              loc=sample_data.mean(),
                              scale=sample_data.sem())

        results = {
            "test": "One-Sample T-Test",
            "variable": column,
            "test_value": test_value,
            "sample_size": len(sample_data),
            "sample_mean": float(sample_data.mean()),
            "sample_std": float(sample_data.std()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": len(sample_data) - 1,
            "confidence_interval_95": [float(ci[0]), float(ci[1])],
            "effect_size": {
                "cohens_d": float(cohens_d),
                "interpretation": self._interpret_cohens_d(cohens_d)
            },
            "conclusion": self._interpret_p_value(p_value)
        }

        return Result.ok(results)

    async def _independent_ttest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Independent samples T-test"""
        group_col = params.get("group_column")
        value_col = params.get("value_column")

        # Get unique groups
        groups = data[group_col].unique()
        if len(groups) != 2:
            return Result.fail(f"Expected 2 groups, found {len(groups)}")

        # Split data into groups
        group1_data = data[data[group_col] == groups[0]][value_col].dropna()
        group2_data = data[data[group_col] == groups[1]][value_col].dropna()

        # Test for equal variances (Levene's test)
        levene_stat, levene_p = stats.levene(group1_data, group2_data)

        # Perform T-test
        equal_var = levene_p > 0.05
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)

        # Calculate effect size
        pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() +
                              (len(group2_data) - 1) * group2_data.var()) /
                             (len(group1_data) + len(group2_data) - 2))
        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std

        results = {
            "test": "Independent Samples T-Test",
            "groups": {
                str(groups[0]): {
                    "n": len(group1_data),
                    "mean": float(group1_data.mean()),
                    "std": float(group1_data.std()),
                    "sem": float(group1_data.sem())
                },
                str(groups[1]): {
                    "n": len(group2_data),
                    "mean": float(group2_data.mean()),
                    "std": float(group2_data.std()),
                    "sem": float(group2_data.sem())
                }
            },
            "mean_difference": float(group1_data.mean() - group2_data.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": len(group1_data) + len(group2_data) - 2,
            "equal_variances_assumed": equal_var,
            "levene_test": {
                "statistic": float(levene_stat),
                "p_value": float(levene_p)
            },
            "effect_size": {
                "cohens_d": float(cohens_d),
                "interpretation": self._interpret_cohens_d(cohens_d)
            },
            "conclusion": self._interpret_p_value(p_value)
        }

        return Result.ok(results)

    async def _paired_ttest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Paired samples T-test"""
        before_col = params.get("before_column")
        after_col = params.get("after_column")

        # Remove rows with missing values
        paired_data = data[[before_col, after_col]].dropna()
        before_data = paired_data[before_col]
        after_data = paired_data[after_col]

        # Calculate differences
        differences = after_data - before_data

        # Perform test
        t_stat, p_value = stats.ttest_rel(before_data, after_data)

        # Effect size
        cohens_d = differences.mean() / differences.std()

        # Confidence interval for mean difference
        ci = stats.t.interval(0.95, len(differences) - 1,
                              loc=differences.mean(),
                              scale=differences.sem())

        results = {
            "test": "Paired Samples T-Test",
            "n_pairs": len(paired_data),
            "before": {
                "mean": float(before_data.mean()),
                "std": float(before_data.std())
            },
            "after": {
                "mean": float(after_data.mean()),
                "std": float(after_data.std())
            },
            "difference": {
                "mean": float(differences.mean()),
                "std": float(differences.std()),
                "sem": float(differences.sem()),
                "min": float(differences.min()),
                "max": float(differences.max())
            },
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": len(differences) - 1,
            "confidence_interval_95": [float(ci[0]), float(ci[1])],
            "effect_size": {
                "cohens_d": float(cohens_d),
                "interpretation": self._interpret_cohens_d(cohens_d)
            },
            "conclusion": self._interpret_p_value(p_value)
        }

        return Result.ok(results)

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_p_value(self, p: float, alpha: float = 0.05) -> str:
        """Interpret p-value"""
        if p < alpha:
            return f"Statistically significant (p={p:.4f} < {alpha})"
        else:
            return f"Not statistically significant (p={p:.4f} >= {alpha})"


class ANOVAAnalyzer(BaseAnalyzer):
    """Analyzer for ANOVA tests"""

    def __init__(self):
        super().__init__("anova")

    async def execute(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Perform ANOVA"""
        try:
            anova_type = params.get("type", "one_way")

            if anova_type == "one_way":
                return await self._one_way_anova(data, params)
            elif anova_type == "two_way":
                return await self._two_way_anova(data, params)
            elif anova_type == "repeated_measures":
                return await self._repeated_measures_anova(data, params)
            else:
                return Result.fail(f"Unknown ANOVA type: {anova_type}")

        except Exception as e:
            return Result.fail(str(e))

    async def _one_way_anova(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """One-way ANOVA"""
        dependent = params.get("dependent")
        factor = params.get("factor")

        # Perform ANOVA
        aov = pg.anova(data=data, dv=dependent, between=factor)

        # Post-hoc tests if significant
        p_value = aov['p-unc'].values[0]
        post_hoc = None
        if p_value < 0.05:
            post_hoc = pg.pairwise_tukey(data=data, dv=dependent, between=factor)

        # Homogeneity of variance test
        homogeneity = pg.homoscedasticity(data=data, dv=dependent, group=factor)

        # Effect size (eta-squared)
        ss_between = aov['SS'].values[0]
        ss_total = aov['SS'].sum()
        eta_squared = ss_between / ss_total

        results = {
            "test": "One-Way ANOVA",
            "dependent": dependent,
            "factor": factor,
            "anova_table": aov.to_dict(orient='records')[0],
            "effect_size": {
                "eta_squared": float(eta_squared),
                "interpretation": self._interpret_eta_squared(eta_squared)
            },
            "homogeneity_test": homogeneity.to_dict(orient='records')[0],
            "post_hoc": post_hoc.to_dict(orient='records') if post_hoc is not None else None,
            "conclusion": self._interpret_p_value(p_value)
        }

        return Result.ok(results)

    async def _two_way_anova(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Two-way ANOVA"""
        dependent = params.get("dependent")
        factor1 = params.get("factor1")
        factor2 = params.get("factor2")

        # Perform two-way ANOVA
        aov = pg.anova(data=data, dv=dependent, between=[factor1, factor2])

        results = {
            "test": "Two-Way ANOVA",
            "dependent": dependent,
            "factors": [factor1, factor2],
            "anova_table": aov.to_dict(orient='records'),
            "main_effects": {
                factor1: {
                    "F": float(aov.loc[aov['Source'] == factor1, 'F'].values[0]),
                    "p_value": float(aov.loc[aov['Source'] == factor1, 'p-unc'].values[0])
                },
                factor2: {
                    "F": float(aov.loc[aov['Source'] == factor2, 'F'].values[0]),
                    "p_value": float(aov.loc[aov['Source'] == factor2, 'p-unc'].values[0])
                }
            },
            "interaction": {
                "F": float(aov.loc[aov['Source'].str.contains('*'), 'F'].values[0]) if any(
                    aov['Source'].str.contains('*')) else None,
                "p_value": float(aov.loc[aov['Source'].str.contains('*'), 'p-unc'].values[0]) if any(
                    aov['Source'].str.contains('*')) else None
            }
        }

        return Result.ok(results)

    async def _repeated_measures_anova(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Repeated measures ANOVA"""
        subject = params.get("subject")
        within = params.get("within")
        dependent = params.get("dependent")

        # Perform repeated measures ANOVA
        aov = pg.rm_anova(data=data, dv=dependent, within=within, subject=subject)

        # Sphericity test
        sphericity = pg.sphericity(data=data, dv=dependent, within=within, subject=subject)

        results = {
            "test": "Repeated Measures ANOVA",
            "dependent": dependent,
            "within_factor": within,
            "anova_table": aov.to_dict(orient='records')[0],
            "sphericity_test": sphericity.to_dict(orient='records')[0] if sphericity is not None else None
        }

        return Result.ok(results)

    def _interpret_eta_squared(self, eta: float) -> str:
        """Interpret eta-squared effect size"""
        if eta < 0.01:
            return "negligible"
        elif eta < 0.06:
            return "small"
        elif eta < 0.14:
            return "medium"
        else:
            return "large"

    def _interpret_p_value(self, p: float, alpha: float = 0.05) -> str:
        """Interpret p-value"""
        if p < alpha:
            return f"Statistically significant (p={p:.4f} < {alpha})"
        else:
            return f"Not statistically significant (p={p:.4f} >= {alpha})"
