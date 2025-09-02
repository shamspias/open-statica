"""
Machine Learning Service for OpenStatica
Manages ML model training, evaluation, and deployment
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    RandomizedSearchCV, StratifiedKFold, KFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    RFE, RFECV, SelectFromModel
)
import joblib
import pickle
import asyncio
import logging
from datetime import datetime
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MLService:
    """Service for machine learning operations"""

    def __init__(self, model_cache_path: str = "./models"):
        self.model_cache_path = Path(model_cache_path)
        self.model_cache_path.mkdir(exist_ok=True)
        self.trained_models = {}
        self.model_metadata = {}

    async def train_model(self,
                          data: pd.DataFrame,
                          task: str,
                          algorithm: str,
                          features: List[str],
                          target: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """Train a machine learning model"""

        # Prepare data
        X, y = await self._prepare_data(data, features, target, task)

        # Split data if supervised
        if task in ['classification', 'regression']:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=1 - kwargs.get('train_test_split', 0.8),
                random_state=kwargs.get('random_state', 42),
                stratify=y if task == 'classification' else None
            )
        else:
            X_train, X_test = X, None
            y_train, y_test = None, None

        # Get model
        model = await self._get_model(task, algorithm)

        # Hyperparameter tuning if requested
        if kwargs.get('auto_ml', False):
            model = await self._tune_hyperparameters(
                model, X_train, y_train, task,
                cv_folds=kwargs.get('cv_folds', 5)
            )

        # Train model
        if task in ['classification', 'regression']:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train)

        # Generate model ID
        model_id = self._generate_model_id(task, algorithm)

        # Store model
        self.trained_models[model_id] = model

        # Evaluate model
        evaluation = await self._evaluate_model(
            model, X_test, y_test, task,
            cross_validate=kwargs.get('cross_validate', True)
        )

        # Get feature importance
        feature_importance = await self._get_feature_importance(model, features)

        # Store metadata
        self.model_metadata[model_id] = {
            'task': task,
            'algorithm': algorithm,
            'features': features,
            'target': target,
            'trained_at': datetime.now().isoformat(),
            'training_size': len(X_train) if X_train is not None else len(X),
            'parameters': model.get_params() if hasattr(model, 'get_params') else {}
        }

        # Save model to disk
        await self._save_model(model, model_id)

        return {
            'model_id': model_id,
            'task': task,
            'algorithm': algorithm,
            'metrics': evaluation.get('metrics', {}),
            'feature_importance': feature_importance,
            'confusion_matrix': evaluation.get('confusion_matrix'),
            'training_time': evaluation.get('training_time', 0),
            'cv_scores': evaluation.get('cv_scores')
        }

    async def _prepare_data(self,
                            data: pd.DataFrame,
                            features: List[str],
                            target: Optional[str],
                            task: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for training"""

        # Select features
        X = data[features].copy()

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            # Use label encoding for now (could be improved with one-hot encoding)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Handle missing values
        X = X.fillna(X.mean())

        # Get target if supervised
        y = None
        if target and task in ['classification', 'regression']:
            y = data[target].copy()

            # Encode target if classification
            if task == 'classification' and y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

        return X, y

    async def _get_model(self, task: str, algorithm: str):
        """Get model instance based on task and algorithm"""

        if task == 'classification':
            if algorithm == 'logistic':
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(max_iter=1000)
            elif algorithm == 'svm':
                from sklearn.svm import SVC
                return SVC(probability=True)
            elif algorithm == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif algorithm == 'xgboost':
                try:
                    import xgboost as xgb
                    return xgb.XGBClassifier(random_state=42)
                except ImportError:
                    from sklearn.ensemble import GradientBoostingClassifier
                    return GradientBoostingClassifier(random_state=42)
            elif algorithm == 'nn':
                from sklearn.neural_network import MLPClassifier
                return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
            elif algorithm == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier()
            elif algorithm == 'nb':
                from sklearn.naive_bayes import GaussianNB
                return GaussianNB()
            elif algorithm == 'dt':
                from sklearn.tree import DecisionTreeClassifier
                return DecisionTreeClassifier(random_state=42)

        elif task == 'regression':
            if algorithm == 'linear':
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            elif algorithm == 'ridge':
                from sklearn.linear_model import Ridge
                return Ridge()
            elif algorithm == 'lasso':
                from sklearn.linear_model import Lasso
                return Lasso()
            elif algorithm == 'elastic':
                from sklearn.linear_model import ElasticNet
                return ElasticNet()
            elif algorithm == 'svr':
                from sklearn.svm import SVR
                return SVR()
            elif algorithm == 'rf_reg':
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif algorithm == 'xgb_reg':
                try:
                    import xgboost as xgb
                    return xgb.XGBRegressor(random_state=42)
                except ImportError:
                    from sklearn.ensemble import GradientBoostingRegressor
                    return GradientBoostingRegressor(random_state=42)
            elif algorithm == 'nn_reg':
                from sklearn.neural_network import MLPRegressor
                return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)

        elif task == 'clustering':
            if algorithm == 'kmeans':
                from sklearn.cluster import KMeans
                return KMeans(n_clusters=3, random_state=42)
            elif algorithm == 'dbscan':
                from sklearn.cluster import DBSCAN
                return DBSCAN(eps=0.5, min_samples=5)
            elif algorithm == 'hierarchical':
                from sklearn.cluster import AgglomerativeClustering
                return AgglomerativeClustering(n_clusters=3)
            elif algorithm == 'gaussian':
                from sklearn.mixture import GaussianMixture
                return GaussianMixture(n_components=3, random_state=42)
            elif algorithm == 'meanshift':
                from sklearn.cluster import MeanShift
                return MeanShift()
            elif algorithm == 'spectral':
                from sklearn.cluster import SpectralClustering
                return SpectralClustering(n_clusters=3, random_state=42)

        elif task == 'dimensionality':
            if algorithm == 'pca':
                from sklearn.decomposition import PCA
                return PCA(n_components=2)
            elif algorithm == 'tsne':
                from sklearn.manifold import TSNE
                return TSNE(n_components=2, random_state=42)
            elif algorithm == 'umap':
                try:
                    import umap
                    return umap.UMAP(n_components=2, random_state=42)
                except ImportError:
                    from sklearn.decomposition import PCA
                    return PCA(n_components=2)
            elif algorithm == 'lda':
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                return LinearDiscriminantAnalysis()
            elif algorithm == 'ica':
                from sklearn.decomposition import FastICA
                return FastICA(n_components=2, random_state=42)

        elif task == 'anomaly':
            if algorithm == 'isolation':
                from sklearn.ensemble import IsolationForest
                return IsolationForest(random_state=42)
            elif algorithm == 'lof':
                from sklearn.neighbors import LocalOutlierFactor
                return LocalOutlierFactor(novelty=True)
            elif algorithm == 'ocsvm':
                from sklearn.svm import OneClassSVM
                return OneClassSVM()
            elif algorithm == 'elliptic':
                from sklearn.covariance import EllipticEnvelope
                return EllipticEnvelope(random_state=42)

        raise ValueError(f"Unknown task/algorithm combination: {task}/{algorithm}")

    async def _tune_hyperparameters(self,
                                    model,
                                    X_train: pd.DataFrame,
                                    y_train: Optional[pd.Series],
                                    task: str,
                                    cv_folds: int = 5) -> Any:
        """Tune model hyperparameters"""

        # Define parameter grids for different models
        param_grids = {
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'RandomForestRegressor': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'SVC': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            },
            'KMeans': {
                'n_clusters': [2, 3, 4, 5, 6],
                'init': ['k-means++', 'random']
            }
        }

        model_name = type(model).__name__

        if model_name in param_grids:
            if task in ['classification', 'regression']:
                # Use GridSearchCV for supervised learning
                grid_search = GridSearchCV(
                    model,
                    param_grids[model_name],
                    cv=cv_folds,
                    scoring='accuracy' if task == 'classification' else 'neg_mean_squared_error',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                return grid_search.best_estimator_
            else:
                # For unsupervised, we might need different approach
                # For now, return the original model
                return model

        return model

    async def _evaluate_model(self,
                              model,
                              X_test: Optional[pd.DataFrame],
                              y_test: Optional[pd.Series],
                              task: str,
                              cross_validate: bool = True) -> Dict[str, Any]:
        """Evaluate model performance"""
        import time

        start_time = time.time()
        evaluation = {'task': task}

        if task == 'classification' and X_test is not None:
            y_pred = model.predict(X_test)

            evaluation['metrics'] = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
            }

            # Add AUC-ROC for binary classification
            if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                evaluation['metrics']['auc_roc'] = float(roc_auc_score(y_test, y_proba))

            evaluation['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()

        elif task == 'regression' and X_test is not None:
            y_pred = model.predict(X_test)

            evaluation['metrics'] = {
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred))
            }

        elif task == 'clustering':
            if hasattr(model, 'labels_'):
                labels = model.labels_

                # Calculate clustering metrics
                if len(np.unique(labels)) > 1:
                    evaluation['metrics'] = {
                        'silhouette_score': float(silhouette_score(X_test or model.cluster_centers_, labels)),
                        'calinski_harabasz': float(calinski_harabasz_score(X_test or model.cluster_centers_, labels)),
                        'davies_bouldin': float(davies_bouldin_score(X_test or model.cluster_centers_, labels)),
                        'n_clusters': len(np.unique(labels))
                    }

        evaluation['training_time'] = time.time() - start_time

        return evaluation

    async def _get_feature_importance(self,
                                      model,
                                      features: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from model"""

        importance = None

        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = dict(zip(features, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # Linear models
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef[0])
            else:
                coef = np.abs(coef)
            importance = dict(zip(features, coef))

        # Normalize importance values
        if importance:
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}

        return importance

    async def predict(self,
                      model_id: str,
                      data: Union[pd.DataFrame, Dict, List]) -> np.ndarray:
        """Make predictions with trained model"""

        if model_id not in self.trained_models:
            # Try to load from disk
            model = await self._load_model(model_id)
            if model is None:
                raise ValueError(f"Model {model_id} not found")
            self.trained_models[model_id] = model

        model = self.trained_models[model_id]
        metadata = self.model_metadata.get(model_id, {})

        # Prepare input data
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)

        # Select features used during training
        if 'features' in metadata:
            data = data[metadata['features']]

        # Handle categorical variables (same as training)
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

        # Handle missing values
        data = data.fillna(data.mean())

        # Make predictions
        predictions = model.predict(data)

        return predictions

    async def evaluate(self,
                       model_id: str,
                       X_test: pd.DataFrame,
                       y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate existing model on new data"""

        if model_id not in self.trained_models:
            model = await self._load_model(model_id)
            if model is None:
                raise ValueError(f"Model {model_id} not found")
            self.trained_models[model_id] = model

        model = self.trained_models[model_id]
        metadata = self.model_metadata.get(model_id, {})
        task = metadata.get('task', 'classification')

        return await self._evaluate_model(model, X_test, y_test, task)

    async def feature_selection(self,
                                X: pd.DataFrame,
                                y: pd.Series,
                                method: str = 'univariate',
                                n_features: int = 10) -> List[str]:
        """Select best features"""

        if method == 'univariate':
            # Univariate feature selection
            selector = SelectKBest(f_classif if y.dtype == 'object' else f_regression, k=n_features)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

        elif method == 'rfe':
            # Recursive Feature Elimination
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            if y.dtype == 'object':
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)

            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(X, y)
            selected_features = X.columns[selector.support_].tolist()

        elif method == 'l1':
            # L1-based feature selection
            from sklearn.linear_model import LogisticRegression, Lasso

            if y.dtype == 'object':
                estimator = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            else:
                estimator = Lasso(alpha=0.01, random_state=42)

            selector = SelectFromModel(estimator, max_features=n_features)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        return selected_features

    async def cross_validate_model(self,
                                   model,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   cv: int = 5,
                                   scoring: str = None) -> Dict[str, Any]:
        """Perform cross-validation"""

        if scoring is None:
            # Auto-detect scoring based on target type
            if y.dtype == 'object' or len(np.unique(y)) < 10:
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        return {
            'scores': scores.tolist(),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'cv_folds': cv,
            'scoring': scoring
        }

    async def automl(self,
                     data: pd.DataFrame,
                     target: str,
                     task: str = 'auto',
                     time_budget: int = 300,
                     **kwargs) -> Dict[str, Any]:
        """Automated machine learning"""

        # Auto-detect task if not specified
        if task == 'auto':
            if data[target].dtype == 'object' or data[target].nunique() < 10:
                task = 'classification'
            else:
                task = 'regression'

        # Try different algorithms
        algorithms = {
            'classification': ['rf', 'xgboost', 'logistic', 'svm'],
            'regression': ['rf_reg', 'xgb_reg', 'linear', 'ridge']
        }

        best_model = None
        best_score = -np.inf if task == 'regression' else 0
        best_algorithm = None
        results = []

        features = [col for col in data.columns if col != target]

        for algorithm in algorithms.get(task, []):
            try:
                # Train model
                result = await self.train_model(
                    data=data,
                    task=task,
                    algorithm=algorithm,
                    features=features,
                    target=target,
                    auto_ml=False,  # Don't tune within automl
                    **kwargs
                )

                # Get primary metric
                if task == 'classification':
                    score = result['metrics'].get('accuracy', 0)
                else:
                    score = -result['metrics'].get('mse', np.inf)

                results.append({
                    'algorithm': algorithm,
                    'score': score,
                    'metrics': result['metrics']
                })

                if score > best_score:
                    best_score = score
                    best_algorithm = algorithm
                    best_model = result

            except Exception as e:
                logger.warning(f"Failed to train {algorithm}: {e}")
                continue

        return {
            'best_model': best_model,
            'best_algorithm': best_algorithm,
            'best_score': best_score,
            'all_results': results,
            'task': task
        }

    def _generate_model_id(self, task: str, algorithm: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hash_input = f"{task}_{algorithm}_{timestamp}"
        model_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{task}_{algorithm}_{model_hash}"

    async def _save_model(self, model, model_id: str):
        """Save model to disk"""
        try:
            model_path = self.model_cache_path / f"{model_id}.pkl"
            joblib.dump(model, model_path)

            # Save metadata
            metadata_path = self.model_cache_path / f"{model_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.model_metadata.get(model_id, {}), f)

        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")

    async def _load_model(self, model_id: str):
        """Load model from disk"""
        try:
            model_path = self.model_cache_path / f"{model_id}.pkl"
            if model_path.exists():
                model = joblib.load(model_path)

                # Load metadata
                metadata_path = self.model_cache_path / f"{model_id}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.model_metadata[model_id] = json.load(f)

                return model

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")

        return None

    async def export_model(self, model_id: str, format: str = 'pickle') -> bytes:
        """Export model in various formats"""

        if model_id not in self.trained_models:
            model = await self._load_model(model_id)
            if model is None:
                raise ValueError(f"Model {model_id} not found")
            self.trained_models[model_id] = model

        model = self.trained_models[model_id]

        if format == 'pickle':
            import io
            buffer = io.BytesIO()
            pickle.dump(model, buffer)
            return buffer.getvalue()

        elif format == 'joblib':
            import io
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            return buffer.getvalue()

        elif format == 'onnx':
            # Would need skl2onnx
            pass

        elif format == 'pmml':
            # Would need sklearn2pmml
            pass

        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information"""

        if model_id in self.model_metadata:
            return self.model_metadata[model_id]

        # Try to load metadata from disk
        metadata_path = self.model_cache_path / f"{model_id}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)

        return {}

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        models = []

        # Add loaded models
        for model_id in self.trained_models.keys():
            info = await self.get_model_info(model_id)
            models.append({
                'model_id': model_id,
                'loaded': True,
                **info
            })

        # Add saved models
        for model_file in self.model_cache_path.glob("*.pkl"):
            model_id = model_file.stem
            if model_id not in self.trained_models:
                info = await self.get_model_info(model_id)
                models.append({
                    'model_id': model_id,
                    'loaded': False,
                    **info
                })

        return models

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model"""

        # Remove from memory
        if model_id in self.trained_models:
            del self.trained_models[model_id]

        if model_id in self.model_metadata:
            del self.model_metadata[model_id]

        # Remove from disk
        model_path = self.model_cache_path / f"{model_id}.pkl"
        metadata_path = self.model_cache_path / f"{model_id}_metadata.json"

        deleted = False
        if model_path.exists():
            model_path.unlink()
            deleted = True

        if metadata_path.exists():
            metadata_path.unlink()

        return deleted
