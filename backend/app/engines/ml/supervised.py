from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score
)
from app.core.base import BaseMLModel, Result
import time


class ClassificationEngine(BaseMLModel):
    """Engine for classification algorithms"""

    def __init__(self, algorithm: str):
        super().__init__(f"classification_{algorithm}", "classification")
        self.algorithm = algorithm

    async def _setup(self):
        """Initialize the model based on algorithm"""
        if self.algorithm == "logistic":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(max_iter=1000)

        elif self.algorithm == "svm":
            from sklearn.svm import SVC
            self.model = SVC(probability=True)

        elif self.algorithm == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100)

        elif self.algorithm == "xgboost":
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier()
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier()

        elif self.algorithm == "nn":
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50))

        elif self.algorithm == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier()

        elif self.algorithm == "nb":
            from sklearn.naive_bayes import GaussianNB
            self.model = GaussianNB()

        elif self.algorithm == "dt":
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier()

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    async def train(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """Train the classification model"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Perform cross-validation if requested
        if kwargs.get('cross_validate', True):
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            cv_mean = None
            cv_std = None

        # Hyperparameter tuning if requested
        if kwargs.get('auto_ml', False):
            self.model = await self._auto_tune(X, y)

        # Train the model
        self.model.fit(X, y)
        self.is_trained = True

        # Calculate feature importance if available
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                X.columns if hasattr(X, 'columns') else range(X.shape[1]),
                self.model.feature_importances_
            ))
        elif hasattr(self.model, 'coef_'):
            feature_importance = dict(zip(
                X.columns if hasattr(X, 'columns') else range(X.shape[1]),
                np.abs(self.model.coef_[0]) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
            ))

        training_time = time.time() - start_time

        return {
            "algorithm": self.algorithm,
            "training_time": training_time,
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "feature_importance": feature_importance,
            "model_params": self.model.get_params()
        }

    async def predict(self, X: Any, **kwargs) -> Any:
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        predictions = self.model.predict(X)

        if kwargs.get('probability', False):
            probabilities = self.model.predict_proba(X)
            return {'predictions': predictions, 'probabilities': probabilities}

        return predictions

    async def evaluate(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """Evaluate model performance"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        predictions = self.model.predict(X)

        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y, predictions)),
            "precision": float(precision_score(y, predictions, average='weighted')),
            "recall": float(recall_score(y, predictions, average='weighted')),
            "f1_score": float(f1_score(y, predictions, average='weighted'))
        }

        # Add AUC-ROC for binary classification
        if len(np.unique(y)) == 2:
            probabilities = self.model.predict_proba(X)[:, 1]
            metrics["auc_roc"] = float(roc_auc_score(y, probabilities))

        # Confusion matrix
        cm = confusion_matrix(y, predictions)

        return {
            "metrics": metrics,
            "confusion_matrix": cm.tolist()
        }

    async def _auto_tune(self, X: Any, y: Any) -> Any:
        """Automatic hyperparameter tuning"""
        param_grid = self._get_param_grid()

        grid_search = GridSearchCV(
            self.model, param_grid, cv=3,
            scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X, y)

        return grid_search.best_estimator_

    def _get_param_grid(self) -> Dict[str, list]:
        """Get parameter grid for hyperparameter tuning"""
        if self.algorithm == "logistic":
            return {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif self.algorithm == "svm":
            return {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        elif self.algorithm == "rf":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        else:
            return {}

    async def execute(self, data: Any, params: Dict[str, Any]) -> Any:
        """Execute classification task"""
        # This method is for compatibility with base engine
        if params.get('mode') == 'train':
            return await self.train(data, params.get('target'), **params)
        elif params.get('mode') == 'predict':
            return await self.predict(data, **params)
        else:
            raise ValueError("Mode must be 'train' or 'predict'")


class RegressionEngine(BaseMLModel):
    """Engine for regression algorithms"""

    def __init__(self, algorithm: str):
        super().__init__(f"regression_{algorithm}", "regression")
        self.algorithm = algorithm

    async def _setup(self):
        """Initialize the regression model"""
        if self.algorithm == "linear":
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()

        elif self.algorithm == "ridge":
            from sklearn.linear_model import Ridge
            self.model = Ridge()

        elif self.algorithm == "lasso":
            from sklearn.linear_model import Lasso
            self.model = Lasso()

        elif self.algorithm == "elastic":
            from sklearn.linear_model import ElasticNet
            self.model = ElasticNet()

        elif self.algorithm == "svr":
            from sklearn.svm import SVR
            self.model = SVR()

        elif self.algorithm == "rf_reg":
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100)

        elif self.algorithm == "xgb_reg":
            try:
                import xgboost as xgb
                self.model = xgb.XGBRegressor()
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor()

        elif self.algorithm == "nn_reg":
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(hidden_layer_sizes=(100, 50))

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    async def train(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """Train the regression model"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Cross-validation
        if kwargs.get('cross_validate', True):
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            cv_mean = None
            cv_std = None

        # Train model
        self.model.fit(X, y)
        self.is_trained = True

        # Feature importance
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                X.columns if hasattr(X, 'columns') else range(X.shape[1]),
                self.model.feature_importances_
            ))
        elif hasattr(self.model, 'coef_'):
            feature_importance = dict(zip(
                X.columns if hasattr(X, 'columns') else range(X.shape[1]),
                np.abs(self.model.coef_)
            ))

        training_time = time.time() - start_time

        return {
            "algorithm": self.algorithm,
            "training_time": training_time,
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std,
            "feature_importance": feature_importance,
            "model_params": self.model.get_params()
        }

    async def predict(self, X: Any, **kwargs) -> Any:
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        predictions = self.model.predict(X)

        if kwargs.get('return_uncertainty', False) and hasattr(self.model, 'predict_std'):
            uncertainty = self.model.predict_std(X)
            return {'predictions': predictions, 'uncertainty': uncertainty}

        return predictions

    async def evaluate(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """Evaluate model performance"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        predictions = self.model.predict(X)

        metrics = {
            "mse": float(mean_squared_error(y, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
            "mae": float(mean_absolute_error(y, predictions)),
            "r2": float(r2_score(y, predictions))
        }

        # Calculate residuals
        residuals = y - predictions

        return {
            "metrics": metrics,
            "residuals_stats": {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals))
            }
        }

    async def execute(self, data: Any, params: Dict[str, Any]) -> Any:
        """Execute regression task"""
        if params.get('mode') == 'train':
            return await self.train(data, params.get('target'), **params)
        elif params.get('mode') == 'predict':
            return await self.predict(data, **params)
        else:
            raise ValueError("Mode must be 'train' or 'predict'")
