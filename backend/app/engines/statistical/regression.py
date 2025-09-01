from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures
from app.core.base import BaseAnalyzer, Result


class RegressionAnalyzer(BaseAnalyzer):
    """Analyzer for regression analysis"""

    def __init__(self):
        super().__init__("regression_analysis")

    async def execute(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Perform regression analysis"""
        try:
            regression_type = params.get("type", "linear")

            if regression_type == "linear":
                return await self._linear_regression(data, params)
            elif regression_type == "multiple":
                return await self._multiple_regression(data, params)
            elif regression_type == "polynomial":
                return await self._polynomial_regression(data, params)
            elif regression_type == "logistic":
                return await self._logistic_regression(data, params)
            elif regression_type == "stepwise":
                return await self._stepwise_regression(data, params)
            elif regression_type == "quantile":
                return await self._quantile_regression(data, params)
            else:
                return Result.fail(f"Unknown regression type: {regression_type}")

        except Exception as e:
            return Result.fail(str(e))

    async def _linear_regression(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Simple linear regression"""
        x_col = params.get("independent")
        y_col = params.get("dependent")

        # Prepare data
        df = data[[x_col, y_col]].dropna()
        X = df[x_col]
        y = df[y_col]

        # Add constant
        X_with_const = sm.add_constant(X)

        # Fit model
        model = sm.OLS(y, X_with_const).fit()

        # Get predictions
        predictions = model.predict(X_with_const)
        residuals = y - predictions

        # Diagnostic tests
        diagnostics = self._regression_diagnostics(model, residuals, X_with_const)

        results = {
            "model": "Simple Linear Regression",
            "formula": f"{y_col} ~ {x_col}",
            "n_observations": len(df),
            "coefficients": {
                "intercept": float(model.params[0]),
                x_col: float(model.params[1])
            },
            "std_errors": {
                "intercept": float(model.bse[0]),
                x_col: float(model.bse[1])
            },
            "t_values": {
                "intercept": float(model.tvalues[0]),
                x_col: float(model.tvalues[1])
            },
            "p_values": {
                "intercept": float(model.pvalues[0]),
                x_col: float(model.pvalues[1])
            },
            "confidence_intervals": {
                "intercept": model.conf_int()[0].tolist(),
                x_col: model.conf_int()[1].tolist()
            },
            "r_squared": float(model.rsquared),
            "adjusted_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
            "aic": float(model.aic),
            "bic": float(model.bic),
            "diagnostics": diagnostics,
            "residuals_summary": {
                "mean": float(residuals.mean()),
                "std": float(residuals.std()),
                "min": float(residuals.min()),
                "max": float(residuals.max())
            }
        }

        return Result.ok(results)

    async def _multiple_regression(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Multiple linear regression"""
        independent_vars = params.get("independent", [])
        dependent_var = params.get("dependent")

        # Prepare data
        all_vars = independent_vars + [dependent_var]
        df = data[all_vars].dropna()
        X = df[independent_vars]
        y = df[dependent_var]

        # Add constant
        X_with_const = sm.add_constant(X)

        # Fit model
        model = sm.OLS(y, X_with_const).fit()

        # Get predictions
        predictions = model.predict(X_with_const)
        residuals = y - predictions

        # Calculate VIF for multicollinearity
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        # Diagnostic tests
        diagnostics = self._regression_diagnostics(model, residuals, X_with_const)

        results = {
            "model": "Multiple Linear Regression",
            "formula": f"{dependent_var} ~ {' + '.join(independent_vars)}",
            "n_observations": len(df),
            "n_predictors": len(independent_vars),
            "coefficients": {
                "intercept": float(model.params[0]),
                **{var: float(model.params[i + 1]) for i, var in enumerate(independent_vars)}
            },
            "std_errors": {
                "intercept": float(model.bse[0]),
                **{var: float(model.bse[i + 1]) for i, var in enumerate(independent_vars)}
            },
            "t_values": {
                "intercept": float(model.tvalues[0]),
                **{var: float(model.tvalues[i + 1]) for i, var in enumerate(independent_vars)}
            },
            "p_values": {
                "intercept": float(model.pvalues[0]),
                **{var: float(model.pvalues[i + 1]) for i, var in enumerate(independent_vars)}
            },
            "confidence_intervals": {
                "intercept": model.conf_int()[0].tolist(),
                **{var: model.conf_int()[i + 1].tolist() for i, var in enumerate(independent_vars)}
            },
            "r_squared": float(model.rsquared),
            "adjusted_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
            "aic": float(model.aic),
            "bic": float(model.bic),
            "vif": vif_data.to_dict(orient='records'),
            "diagnostics": diagnostics
        }

        return Result.ok(results)

    async def _polynomial_regression(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Polynomial regression"""
        x_col = params.get("independent")
        y_col = params.get("dependent")
        degree = params.get("degree", 2)

        # Prepare data
        df = data[[x_col, y_col]].dropna()
        X = df[[x_col]]
        y = df[y_col]

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Create DataFrame with polynomial features
        feature_names = [f"{x_col}^{i}" for i in range(1, degree + 1)]
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)

        # Add constant
        X_with_const = sm.add_constant(X_poly_df)

        # Fit model
        model = sm.OLS(y, X_with_const).fit()

        results = {
            "model": f"Polynomial Regression (degree={degree})",
            "formula": f"{y_col} ~ {' + '.join(feature_names)}",
            "n_observations": len(df),
            "degree": degree,
            "coefficients": {
                "intercept": float(model.params[0]),
                **{feat: float(model.params[i + 1]) for i, feat in enumerate(feature_names)}
            },
            "r_squared": float(model.rsquared),
            "adjusted_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
            "aic": float(model.aic),
            "bic": float(model.bic)
        }

        return Result.ok(results)

    async def _logistic_regression(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Logistic regression"""
        independent_vars = params.get("independent", [])
        dependent_var = params.get("dependent")

        # Prepare data
        all_vars = independent_vars + [dependent_var]
        df = data[all_vars].dropna()
        X = df[independent_vars]
        y = df[dependent_var]

        # Convert y to binary if needed
        if y.nunique() == 2:
            y = pd.get_dummies(y, drop_first=True).iloc[:, 0]

        # Add constant
        X_with_const = sm.add_constant(X)

        # Fit model
        model = sm.Logit(y, X_with_const).fit()

        # Calculate odds ratios
        odds_ratios = np.exp(model.params)

        # Get predictions
        predictions = model.predict(X_with_const)
        predicted_classes = (predictions > 0.5).astype(int)

        # Calculate accuracy metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        accuracy = accuracy_score(y, predicted_classes)
        precision = precision_score(y, predicted_classes)
        recall = recall_score(y, predicted_classes)
        f1 = f1_score(y, predicted_classes)
        auc = roc_auc_score(y, predictions)

        results = {
            "model": "Logistic Regression",
            "n_observations": len(df),
            "coefficients": {
                "intercept": float(model.params[0]),
                **{var: float(model.params[i + 1]) for i, var in enumerate(independent_vars)}
            },
            "odds_ratios": {
                "intercept": float(odds_ratios[0]),
                **{var: float(odds_ratios[i + 1]) for i, var in enumerate(independent_vars)}
            },
            "std_errors": {
                "intercept": float(model.bse[0]),
                **{var: float(model.bse[i + 1]) for i, var in enumerate(independent_vars)}
            },
            "z_values": {
                "intercept": float(model.tvalues[0]),
                **{var: float(model.tvalues[i + 1]) for i, var in enumerate(independent_vars)}
            },
            "p_values": {
                "intercept": float(model.pvalues[0]),
                **{var: float(model.pvalues[i + 1]) for i, var in enumerate(independent_vars)}
            },
            "pseudo_r_squared": float(model.prsquared),
            "log_likelihood": float(model.llf),
            "aic": float(model.aic),
            "bic": float(model.bic),
            "classification_metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "auc_roc": float(auc)
            }
        }

        return Result.ok(results)

    async def _stepwise_regression(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Stepwise regression using forward/backward selection"""
        independent_vars = params.get("independent", [])
        dependent_var = params.get("dependent")
        method = params.get("method", "forward")  # forward, backward, or both
        threshold = params.get("threshold", 0.05)

        # Prepare data
        all_vars = independent_vars + [dependent_var]
        df = data[all_vars].dropna()
        X = df[independent_vars]
        y = df[dependent_var]

        if method == "forward":
            selected_vars = await self._forward_selection(X, y, threshold)
        elif method == "backward":
            selected_vars = await self._backward_elimination(X, y, threshold)
        else:  # both
            selected_vars = await self._bidirectional_selection(X, y, threshold)

        # Fit final model with selected variables
        if selected_vars:
            X_selected = X[selected_vars]
            X_with_const = sm.add_constant(X_selected)
            model = sm.OLS(y, X_with_const).fit()

            results = {
                "model": f"Stepwise Regression ({method})",
                "selected_variables": selected_vars,
                "n_selected": len(selected_vars),
                "n_original": len(independent_vars),
                "threshold": threshold,
                "coefficients": {
                    "intercept": float(model.params[0]),
                    **{var: float(model.params[i + 1]) for i, var in enumerate(selected_vars)}
                },
                "p_values": {
                    "intercept": float(model.pvalues[0]),
                    **{var: float(model.pvalues[i + 1]) for i, var in enumerate(selected_vars)}
                },
                "r_squared": float(model.rsquared),
                "adjusted_r_squared": float(model.rsquared_adj),
                "aic": float(model.aic),
                "bic": float(model.bic)
            }
        else:
            results = {
                "model": f"Stepwise Regression ({method})",
                "error": "No variables selected"
            }

        return Result.ok(results)

    async def _forward_selection(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> List[str]:
        """Forward selection algorithm"""
        selected = []
        remaining = list(X.columns)

        while remaining:
            best_pvalue = 1.0
            best_var = None

            for var in remaining:
                vars_to_test = selected + [var]
                X_test = sm.add_constant(X[vars_to_test])
                model = sm.OLS(y, X_test).fit()

                # Get p-value for the new variable
                pvalue = model.pvalues[var]

                if pvalue < best_pvalue:
                    best_pvalue = pvalue
                    best_var = var

            if best_pvalue < threshold:
                selected.append(best_var)
                remaining.remove(best_var)
            else:
                break

        return selected

    async def _backward_elimination(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> List[str]:
        """Backward elimination algorithm"""
        selected = list(X.columns)

        while len(selected) > 0:
            X_test = sm.add_constant(X[selected])
            model = sm.OLS(y, X_test).fit()

            # Get variable with highest p-value
            pvalues = model.pvalues[1:]  # Exclude intercept
            max_pvalue = pvalues.max()

            if max_pvalue > threshold:
                worst_var = pvalues.idxmax()
                selected.remove(worst_var)
            else:
                break

        return selected

    async def _bidirectional_selection(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> List[str]:
        """Bidirectional (stepwise) selection"""
        # Start with forward selection
        selected = await self._forward_selection(X, y, threshold)

        # Then apply backward elimination on selected variables
        if selected:
            X_selected = X[selected]
            selected = await self._backward_elimination(X_selected, y, threshold)

        return selected

    async def _quantile_regression(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Quantile regression"""
        independent_vars = params.get("independent", [])
        dependent_var = params.get("dependent")
        quantiles = params.get("quantiles", [0.25, 0.5, 0.75])

        # Prepare data
        all_vars = independent_vars + [dependent_var]
        df = data[all_vars].dropna()
        X = df[independent_vars]
        y = df[dependent_var]

        # Add constant
        X_with_const = sm.add_constant(X)

        results = {
            "model": "Quantile Regression",
            "quantiles": {}
        }

        # Fit model for each quantile
        for q in quantiles:
            model = sm.QuantReg(y, X_with_const).fit(q=q)

            results["quantiles"][str(q)] = {
                "coefficients": {
                    "intercept": float(model.params[0]),
                    **{var: float(model.params[i + 1]) for i, var in enumerate(independent_vars)}
                },
                "pseudo_r_squared": float(model.prsquared),
                "scale": float(model.scale)
            }

        return Result.ok(results)

    def _regression_diagnostics(self, model, residuals, X) -> Dict[str, Any]:
        """Perform regression diagnostic tests"""
        diagnostics = {}

        # Durbin-Watson test for autocorrelation
        from statsmodels.stats.stattools import durbin_watson
        dw = durbin_watson(residuals)
        diagnostics["durbin_watson"] = float(dw)

        # Breusch-Pagan test for heteroscedasticity
        bp_test = het_breuschpagan(residuals, X)
        diagnostics["breusch_pagan"] = {
            "statistic": float(bp_test[0]),
            "p_value": float(bp_test[1])
        }

        # Jarque-Bera test for normality of residuals
        jb_test = stats.jarque_bera(residuals)
        diagnostics["jarque_bera"] = {
            "statistic": float(jb_test[0]),
            "p_value": float(jb_test[1])
        }

        # Ljung-Box test for autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        diagnostics["ljung_box"] = {
            "statistics": lb_test['lb_stat'].tolist(),
            "p_values": lb_test['lb_pvalue'].tolist()
        }

        return diagnostics
