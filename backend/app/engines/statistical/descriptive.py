from typing import List, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats
from app.core.base import BaseAnalyzer, Result


class DescriptiveAnalyzer(BaseAnalyzer):
    """Analyzer for descriptive statistics"""

    def __init__(self):
        super().__init__("descriptive_statistics")

    async def execute(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Calculate descriptive statistics"""
        try:
            columns = params.get("columns", data.columns.tolist())
            include_advanced = params.get("include_advanced", True)

            results = {}

            for col in columns:
                if col not in data.columns:
                    continue

                if pd.api.types.is_numeric_dtype(data[col]):
                    results[col] = self._analyze_numeric(data[col], include_advanced)
                else:
                    results[col] = self._analyze_categorical(data[col])

            return Result.ok(results, columns=columns)

        except Exception as e:
            return Result.fail(str(e))

    def _analyze_numeric(self, series: pd.Series, include_advanced: bool) -> Dict[str, Any]:
        """Analyze numeric column"""
        clean_data = series.dropna()

        stats_dict = {
            # Basic statistics
            "count": len(clean_data),
            "missing": series.isna().sum(),
            "missing_pct": (series.isna().sum() / len(series)) * 100,
            "mean": float(clean_data.mean()),
            "median": float(clean_data.median()),
            "mode": float(clean_data.mode()[0]) if len(clean_data.mode()) > 0 else None,
            "std": float(clean_data.std()),
            "variance": float(clean_data.var()),
            "min": float(clean_data.min()),
            "max": float(clean_data.max()),
            "range": float(clean_data.max() - clean_data.min()),

            # Quartiles
            "q1": float(clean_data.quantile(0.25)),
            "q2": float(clean_data.quantile(0.50)),
            "q3": float(clean_data.quantile(0.75)),
            "iqr": float(clean_data.quantile(0.75) - clean_data.quantile(0.25)),

            # Percentiles
            "percentiles": {
                f"p{p}": float(clean_data.quantile(p / 100))
                for p in [5, 10, 25, 50, 75, 90, 95, 99]
            }
        }

        if include_advanced:
            # Manual MAD to avoid pandas version differences
            mad_val = float(np.mean(np.abs(clean_data - clean_data.mean()))) if len(clean_data) else None
            cv_val = float((clean_data.std() / clean_data.mean()) * 100) if clean_data.mean() != 0 else None

            stats_dict.update({
                "skewness": float(clean_data.skew()),
                "kurtosis": float(clean_data.kurtosis()),
                "sem": float(clean_data.sem()),
                "mad": mad_val,
                "cv": cv_val,
                "geometric_mean": float(stats.gmean(clean_data[clean_data > 0])) if (clean_data > 0).any() else None,
                "harmonic_mean": float(stats.hmean(clean_data[clean_data > 0])) if (clean_data > 0).any() else None,
                "trimmed_mean": float(stats.trim_mean(clean_data, 0.1)),
                "ci_95": self._confidence_interval(clean_data, 0.95),
                "ci_99": self._confidence_interval(clean_data, 0.99),
                "outliers": self._detect_outliers(clean_data),
                "normality": self._test_normality(clean_data)
            })

        return stats_dict

    def _analyze_categorical(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical column"""
        value_counts = series.value_counts()
        value_pcts = series.value_counts(normalize=True) * 100

        return {
            "count": len(series.dropna()),
            "missing": series.isna().sum(),
            "missing_pct": (series.isna().sum() / len(series)) * 100,
            "unique": series.nunique(),
            "unique_pct": (series.nunique() / len(series)) * 100,
            "mode": series.mode()[0] if len(series.mode()) > 0 else None,
            "mode_freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "mode_pct": float(value_pcts.iloc[0]) if len(value_pcts) > 0 else 0,

            # Top values
            "top_values": value_counts.head(10).to_dict(),
            "top_percentages": value_pcts.head(10).to_dict(),

            # Entropy (measure of randomness)
            "entropy": float(stats.entropy(value_counts))
        }

    def _confidence_interval(self, data: pd.Series, confidence: float) -> List[float]:
        """Calculate confidence interval"""
        mean = data.mean()
        sem = data.sem()
        interval = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
        return [float(interval[0]), float(interval[1])]

    def _detect_outliers(self, data: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = data[(data < lower_bound) | (data > upper_bound)]

        return {
            "count": len(outliers),
            "percentage": (len(outliers) / len(data)) * 100,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "values": outliers.tolist()[:100]  # Limit to 100 outliers
        }

    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """Test for normality"""
        if len(data) < 3:
            return {"error": "Not enough data for normality test"}

        # Shapiro-Wilk test (good for small samples)
        if len(data) <= 5000:
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
        else:
            # Kolmogorov-Smirnov test for larger samples
            stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            test_name = "Kolmogorov-Smirnov"

        return {
            "test": test_name,
            "statistic": float(stat),
            "p_value": float(p_value),
            "is_normal": p_value > 0.05
        }


class FrequencyAnalyzer(BaseAnalyzer):
    """Analyzer for frequency distributions"""

    def __init__(self):
        super().__init__("frequency_distribution")

    async def execute(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Calculate frequency distributions"""
        try:
            columns = params.get("columns", data.columns.tolist())
            bins = params.get("bins", "auto")

            results = {}

            for col in columns:
                if col not in data.columns:
                    continue

                if pd.api.types.is_numeric_dtype(data[col]):
                    results[col] = self._numeric_frequency(data[col], bins)
                else:
                    results[col] = self._categorical_frequency(data[col])

            return Result.ok(results)

        except Exception as e:
            return Result.fail(str(e))

    def _numeric_frequency(self, series: pd.Series, bins) -> Dict[str, Any]:
        """Calculate frequency for numeric data"""
        clean_data = series.dropna()

        # Create histogram
        if bins == "auto":
            bins = min(50, int(np.sqrt(len(clean_data))))

        counts, edges = np.histogram(clean_data, bins=bins)

        # Create bins labels
        bin_labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]

        return {
            "type": "numeric",
            "bins": bin_labels,
            "frequencies": counts.tolist(),
            "relative_frequencies": (counts / len(clean_data)).tolist(),
            "cumulative_frequencies": np.cumsum(counts).tolist(),
            "bin_edges": edges.tolist(),
            "statistics": {
                "total": len(clean_data),
                "bin_width": float(edges[1] - edges[0]),
                "num_bins": len(bin_labels)
            }
        }

    def _categorical_frequency(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate frequency for categorical data"""
        value_counts = series.value_counts()
        value_pcts = series.value_counts(normalize=True) * 100
        cumulative_pcts = value_pcts.cumsum()

        return {
            "type": "categorical",
            "categories": value_counts.index.tolist(),
            "frequencies": value_counts.values.tolist(),
            "relative_frequencies": value_pcts.values.tolist(),
            "cumulative_frequencies": cumulative_pcts.values.tolist(),
            "statistics": {
                "total": len(series.dropna()),
                "unique_values": series.nunique(),
                "mode": value_counts.index[0] if len(value_counts) > 0 else None
            }
        }
