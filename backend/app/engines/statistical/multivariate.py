from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from app.core.base import BaseAnalyzer, Result


class MultivariateAnalyzer(BaseAnalyzer):
    """Analyzer for multivariate statistical methods"""

    def __init__(self):
        super().__init__("multivariate_analysis")

    async def execute(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Execute multivariate analysis"""
        try:
            analysis_type = params.get("type")

            if analysis_type == "pca":
                return await self._pca_analysis(data, params)
            elif analysis_type == "factor":
                return await self._factor_analysis(data, params)
            elif analysis_type == "lda":
                return await self._lda_analysis(data, params)
            elif analysis_type == "canonical":
                return await self._canonical_correlation(data, params)
            elif analysis_type == "manova":
                return await self._manova(data, params)
            else:
                return Result.fail(f"Unknown analysis type: {analysis_type}")

        except Exception as e:
            return Result.fail(str(e))

    async def _pca_analysis(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Principal Component Analysis"""
        variables = params.get("variables")
        n_components = params.get("n_components", None)

        # Prepare data
        X = data[variables].dropna()

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Calculate loadings
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        # Create loadings DataFrame
        loadings_df = pd.DataFrame(
            loadings,
            columns=[f'PC{i + 1}' for i in range(len(pca.components_))],
            index=variables
        )

        # Kaiser criterion (eigenvalues > 1)
        eigenvalues = pca.explained_variance_
        n_kaiser = np.sum(eigenvalues > 1)

        results = {
            "method": "Principal Component Analysis",
            "n_observations": len(X),
            "n_variables": len(variables),
            "n_components": len(pca.components_),
            "n_components_kaiser": int(n_kaiser),
            "explained_variance": pca.explained_variance_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "loadings": loadings_df.to_dict(),
            "component_scores": {
                f"PC{i + 1}": X_pca[:, i].tolist()[:100]  # First 100 scores
                for i in range(min(3, len(pca.components_)))
            },
            "interpretation": self._interpret_pca(pca.explained_variance_ratio_)
        }

        return Result.ok(results)

    async def _factor_analysis(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Exploratory Factor Analysis"""
        variables = params.get("variables")
        n_factors = params.get("n_factors", None)
        rotation = params.get("rotation", "varimax")

        # Prepare data
        X = data[variables].dropna()

        # Test assumptions
        chi_square, p_value = calculate_bartlett_sphericity(X)
        kmo_all, kmo_model = calculate_kmo(X)

        # Determine number of factors if not specified
        if n_factors is None:
            # Use eigenvalue > 1 criterion
            fa_test = FactorAnalyzer(n_factors=len(variables), rotation=None)
            fa_test.fit(X)
            eigenvalues = fa_test.get_eigenvalues()[0]
            n_factors = np.sum(eigenvalues > 1)

        # Perform factor analysis
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
        fa.fit(X)

        # Get results
        loadings = fa.loadings_
        communalities = fa.get_communalities()
        uniqueness = fa.get_uniquenesses()
        factor_variance = fa.get_factor_variance()

        # Create loadings DataFrame
        loadings_df = pd.DataFrame(
            loadings,
            columns=[f'Factor{i + 1}' for i in range(n_factors)],
            index=variables
        )

        results = {
            "method": "Exploratory Factor Analysis",
            "n_observations": len(X),
            "n_variables": len(variables),
            "n_factors": n_factors,
            "rotation": rotation,
            "bartlett_test": {
                "chi_square": float(chi_square),
                "p_value": float(p_value),
                "suitable": p_value < 0.05
            },
            "kmo": {
                "overall": float(kmo_model),
                "variables": {var: float(kmo_all[i]) for i, var in enumerate(variables)},
                "interpretation": self._interpret_kmo(kmo_model)
            },
            "loadings": loadings_df.to_dict(),
            "communalities": {var: float(communalities[i]) for i, var in enumerate(variables)},
            "uniqueness": {var: float(uniqueness[i]) for i, var in enumerate(variables)},
            "variance_explained": {
                "SS_loadings": factor_variance[0].tolist(),
                "proportion_var": factor_variance[1].tolist(),
                "cumulative_var": factor_variance[2].tolist()
            }
        }

        return Result.ok(results)

    async def _lda_analysis(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Linear Discriminant Analysis"""
        features = params.get("features")
        target = params.get("target")

        # Prepare data
        X = data[features].dropna()
        y = data.loc[X.index, target]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform LDA
        lda = LinearDiscriminantAnalysis()
        X_lda = lda.fit_transform(X_scaled, y)

        # Get results
        n_components = min(len(np.unique(y)) - 1, len(features))

        results = {
            "method": "Linear Discriminant Analysis",
            "n_observations": len(X),
            "n_features": len(features),
            "n_classes": len(np.unique(y)),
            "n_components": n_components,
            "explained_variance_ratio": lda.explained_variance_ratio_.tolist() if hasattr(lda,
                                                                                          'explained_variance_ratio_') else None,
            "coefficients": lda.coef_.tolist(),
            "intercept": lda.intercept_.tolist(),
            "means": {
                str(cls): lda.means_[i].tolist()
                for i, cls in enumerate(lda.classes_)
            },
            "classification_accuracy": float(lda.score(X_scaled, y))
        }

        return Result.ok(results)

    async def _canonical_correlation(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Canonical Correlation Analysis"""
        x_vars = params.get("x_variables")
        y_vars = params.get("y_variables")

        # Prepare data
        X = data[x_vars].dropna()
        Y = data.loc[X.index, y_vars]

        # Standardize
        X_std = (X - X.mean()) / X.std()
        Y_std = (Y - Y.mean()) / Y.std()

        # Calculate correlation matrices
        n = len(X)
        Rxx = X_std.T @ X_std / n
        Ryy = Y_std.T @ Y_std / n
        Rxy = X_std.T @ Y_std / n
        Ryx = Rxy.T

        # Solve eigenvalue problem
        Rxx_inv = np.linalg.inv(Rxx)
        Ryy_inv = np.linalg.inv(Ryy)

        M = Rxx_inv @ Rxy @ Ryy_inv @ Ryx
        eigenvalues, eigenvectors = np.linalg.eig(M)

        # Sort by eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Canonical correlations
        canonical_corr = np.sqrt(np.real(eigenvalues))

        # Canonical loadings
        x_loadings = eigenvectors[:, :min(len(x_vars), len(y_vars))]

        results = {
            "method": "Canonical Correlation Analysis",
            "n_observations": n,
            "x_variables": x_vars,
            "y_variables": y_vars,
            "canonical_correlations": canonical_corr.tolist(),
            "eigenvalues": eigenvalues.real.tolist(),
            "wilks_lambda": float(np.prod(1 - eigenvalues[:min(len(x_vars), len(y_vars))])),
            "x_loadings": x_loadings.real.tolist()
        }

        return Result.ok(results)

    async def _manova(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Multivariate Analysis of Variance"""
        dependent_vars = params.get("dependent_variables")
        factor = params.get("factor")

        # This is a simplified MANOVA implementation
        # For production, use statsmodels.multivariate.manova

        # Group data
        groups = data.groupby(factor)
        k = len(groups)  # number of groups
        p = len(dependent_vars)  # number of dependent variables
        n = len(data)

        # Calculate SSCP matrices
        # Total SSCP
        Y = data[dependent_vars].values
        Y_mean = Y.mean(axis=0)
        T = (Y - Y_mean).T @ (Y - Y_mean)

        # Within-groups SSCP
        W = np.zeros((p, p))
        for name, group in groups:
            Y_g = group[dependent_vars].values
            Y_g_mean = Y_g.mean(axis=0)
            W += (Y_g - Y_g_mean).T @ (Y_g - Y_g_mean)

        # Between-groups SSCP
        B = T - W

        # Wilks' Lambda
        wilks_lambda = np.linalg.det(W) / np.linalg.det(T)

        # Approximate F-statistic
        df1 = p * (k - 1)
        df2 = p * (n - k)
        F = ((1 - wilks_lambda) / wilks_lambda) * (df2 / df1)

        from scipy.stats import f
        p_value = 1 - f.cdf(F, df1, df2)

        results = {
            "method": "MANOVA",
            "dependent_variables": dependent_vars,
            "factor": factor,
            "n_groups": k,
            "n_observations": n,
            "wilks_lambda": float(wilks_lambda),
            "f_statistic": float(F),
            "df1": df1,
            "df2": df2,
            "p_value": float(p_value),
            "conclusion": self._interpret_p_value(p_value)
        }

        return Result.ok(results)

    def _interpret_pca(self, variance_ratio: np.ndarray) -> str:
        """Interpret PCA results"""
        cumsum = np.cumsum(variance_ratio)

        # Find number of components for 80% variance
        n_80 = np.argmax(cumsum >= 0.8) + 1
        n_90 = np.argmax(cumsum >= 0.9) + 1

        return (f"First {n_80} components explain {cumsum[n_80 - 1] * 100:.1f}% of variance. "
                f"First {n_90} components explain {cumsum[n_90 - 1] * 100:.1f}% of variance.")

    def _interpret_kmo(self, kmo: float) -> str:
        """Interpret KMO measure"""
        if kmo >= 0.9:
            return "Excellent"
        elif kmo >= 0.8:
            return "Good"
        elif kmo >= 0.7:
            return "Acceptable"
        elif kmo >= 0.6:
            return "Mediocre"
        elif kmo >= 0.5:
            return "Poor"
        else:
            return "Unacceptable"

    def _interpret_p_value(self, p: float, alpha: float = 0.05) -> str:
        """Interpret p-value"""
        if p < alpha:
            return f"Statistically significant (p={p:.4f} < {alpha})"
        else:
            return f"Not statistically significant (p={p:.4f} >= {alpha})"
