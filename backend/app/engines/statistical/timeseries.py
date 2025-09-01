from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
from app.core.base import BaseAnalyzer, Result


class TimeSeriesAnalyzer(BaseAnalyzer):
    """Analyzer for time series analysis"""

    def __init__(self):
        super().__init__("timeseries_analysis")

    async def execute(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Execute time series analysis"""
        try:
            analysis_type = params.get("type")

            if analysis_type == "decomposition":
                return await self._decomposition(data, params)
            elif analysis_type == "stationarity":
                return await self._stationarity_tests(data, params)
            elif analysis_type == "arima":
                return await self._arima_analysis(data, params)
            elif analysis_type == "sarima":
                return await self._sarima_analysis(data, params)
            elif analysis_type == "exponential_smoothing":
                return await self._exponential_smoothing(data, params)
            elif analysis_type == "autocorrelation":
                return await self._autocorrelation_analysis(data, params)
            elif analysis_type == "forecast":
                return await self._forecast(data, params)
            else:
                return Result.fail(f"Unknown analysis type: {analysis_type}")

        except Exception as e:
            return Result.fail(str(e))

    async def _decomposition(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Seasonal decomposition"""
        column = params.get("column")
        date_column = params.get("date_column")
        period = params.get("period", None)
        model = params.get("model", "additive")  # additive or multiplicative

        # Prepare time series
        if date_column:
            data = data.set_index(date_column)
        ts = data[column].dropna()

        # Perform decomposition
        decomposition = seasonal_decompose(ts, model=model, period=period)

        results = {
            "method": "Seasonal Decomposition",
            "model": model,
            "period": period,
            "components": {
                "trend": decomposition.trend.dropna().tolist()[:100],
                "seasonal": decomposition.seasonal.dropna().tolist()[:100],
                "residual": decomposition.resid.dropna().tolist()[:100]
            },
            "statistics": {
                "trend_strength": float(1 - np.var(decomposition.resid.dropna()) / np.var(ts - decomposition.seasonal)),
                "seasonal_strength": float(
                    1 - np.var(decomposition.resid.dropna()) / np.var(ts - decomposition.trend.dropna()))
            }
        }

        return Result.ok(results)

    async def _stationarity_tests(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Test for stationarity"""
        column = params.get("column")
        ts = data[column].dropna()

        # ADF test
        adf_result = adfuller(ts, autolag='AIC')

        # KPSS test
        kpss_result = kpss(ts, regression='c', nlags='auto')

        # Calculate rolling statistics
        window = min(12, len(ts) // 4)
        rolling_mean = ts.rolling(window=window).mean()
        rolling_std = ts.rolling(window=window).std()

        results = {
            "method": "Stationarity Tests",
            "n_observations": len(ts),
            "adf_test": {
                "statistic": float(adf_result[0]),
                "p_value": float(adf_result[1]),
                "critical_values": {
                    "1%": float(adf_result[4]['1%']),
                    "5%": float(adf_result[4]['5%']),
                    "10%": float(adf_result[4]['10%'])
                },
                "conclusion": "Stationary" if adf_result[1] < 0.05 else "Non-stationary"
            },
            "kpss_test": {
                "statistic": float(kpss_result[0]),
                "p_value": float(kpss_result[1]),
                "critical_values": {
                    "1%": float(kpss_result[3]['1%']),
                    "2.5%": float(kpss_result[3]['2.5%']),
                    "5%": float(kpss_result[3]['5%']),
                    "10%": float(kpss_result[3]['10%'])
                },
                "conclusion": "Stationary" if kpss_result[1] > 0.05 else "Non-stationary"
            },
            "rolling_statistics": {
                "mean_variance": float(rolling_mean.var()),
                "std_variance": float(rolling_std.var())
            }
        }

        return Result.ok(results)

    async def _arima_analysis(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """ARIMA model analysis"""
        column = params.get("column")
        order = params.get("order", None)  # (p, d, q)
        auto = params.get("auto", True)

        ts = data[column].dropna()

        if auto:
            # Auto ARIMA
            model = pm.auto_arima(
                ts,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True
            )
            order = model.order
        else:
            if order is None:
                order = (1, 1, 1)
            model = ARIMA(ts, order=order).fit()

        # Get model diagnostics
        if hasattr(model, 'summary'):
            aic = model.aic
            bic = model.bic
        else:
            aic = model.aic()
            bic = model.bic()

        # Residual diagnostics
        residuals = model.resid if hasattr(model, 'resid') else model.resid()

        results = {
            "method": "ARIMA Model",
            "order": order,
            "auto_selected": auto,
            "aic": float(aic),
            "bic": float(bic),
            "coefficients": {
                f"ar{i + 1}": float(c) for i, c in enumerate(model.arparams if hasattr(model, 'arparams') else [])
            },
            "residual_diagnostics": {
                "mean": float(residuals.mean()),
                "std": float(residuals.std()),
                "skewness": float(residuals.skew()),
                "kurtosis": float(residuals.kurtosis())
            }
        }

        return Result.ok(results)

    async def _sarima_analysis(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """SARIMA model analysis"""
        column = params.get("column")
        order = params.get("order", (1, 1, 1))
        seasonal_order = params.get("seasonal_order", (1, 1, 1, 12))

        ts = data[column].dropna()

        # Fit SARIMA model
        model = SARIMAX(ts, order=order, seasonal_order=seasonal_order).fit()

        results = {
            "method": "SARIMA Model",
            "order": order,
            "seasonal_order": seasonal_order,
            "aic": float(model.aic),
            "bic": float(model.bic),
            "log_likelihood": float(model.llf),
            "coefficients": model.params.to_dict()
        }

        return Result.ok(results)

    async def _exponential_smoothing(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Exponential smoothing analysis"""
        column = params.get("column")
        trend = params.get("trend", "add")
        seasonal = params.get("seasonal", "add")
        seasonal_periods = params.get("seasonal_periods", 12)

        ts = data[column].dropna()

        # Fit model
        model = ExponentialSmoothing(
            ts,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        ).fit()

        results = {
            "method": "Exponential Smoothing",
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
            "aic": float(model.aic),
            "bic": float(model.bic),
            "smoothing_parameters": {
                "alpha": float(model.params['smoothing_level']),
                "beta": float(model.params.get('smoothing_trend', 0)),
                "gamma": float(model.params.get('smoothing_seasonal', 0))
            },
            "fitted_values": model.fittedvalues.tolist()[:100]
        }

        return Result.ok(results)

    async def _autocorrelation_analysis(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Autocorrelation and partial autocorrelation analysis"""
        column = params.get("column")
        nlags = params.get("nlags", 40)

        ts = data[column].dropna()

        # Calculate ACF and PACF
        acf_values = acf(ts, nlags=nlags)
        pacf_values = pacf(ts, nlags=nlags)

        # Find significant lags
        n = len(ts)
        confidence_interval = 1.96 / np.sqrt(n)

        significant_acf_lags = [i for i, v in enumerate(acf_values[1:], 1)
                                if abs(v) > confidence_interval]
        significant_pacf_lags = [i for i, v in enumerate(pacf_values[1:], 1)
                                 if abs(v) > confidence_interval]

        results = {
            "method": "Autocorrelation Analysis",
            "n_observations": n,
            "nlags": nlags,
            "acf": acf_values.tolist(),
            "pacf": pacf_values.tolist(),
            "confidence_interval": float(confidence_interval),
            "significant_acf_lags": significant_acf_lags,
            "significant_pacf_lags": significant_pacf_lags,
            "suggested_ar_order": len(significant_pacf_lags),
            "suggested_ma_order": len(significant_acf_lags)
        }

        return Result.ok(results)

    async def _forecast(self, data: pd.DataFrame, params: Dict[str, Any]) -> Result:
        """Generate forecasts"""
        column = params.get("column")
        model_type = params.get("model_type", "arima")
        horizon = params.get("horizon", 12)

        ts = data[column].dropna()

        if model_type == "arima":
            # Auto ARIMA for forecasting
            model = pm.auto_arima(ts, suppress_warnings=True)
            forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True)

            results = {
                "method": "ARIMA Forecast",
                "model_order": model.order,
                "horizon": horizon,
                "forecast": forecast.tolist(),
                "lower_bound": conf_int[:, 0].tolist(),
                "upper_bound": conf_int[:, 1].tolist(),
                "in_sample_metrics": {
                    "mape": float(np.mean(np.abs((ts - model.predict_in_sample()) / ts)) * 100),
                    "rmse": float(np.sqrt(np.mean((ts - model.predict_in_sample()) ** 2)))
                }
            }
        else:
            # Exponential smoothing forecast
            model = ExponentialSmoothing(ts, seasonal_periods=12).fit()
            forecast = model.forecast(horizon)

            results = {
                "method": "Exponential Smoothing Forecast",
                "horizon": horizon,
                "forecast": forecast.tolist(),
                "in_sample_metrics": {
                    "mape": float(np.mean(np.abs((ts - model.fittedvalues) / ts)) * 100),
                    "rmse": float(np.sqrt(np.mean((ts - model.fittedvalues) ** 2)))
                }
            }

        return Result.ok(results)
