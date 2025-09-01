from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from app.core.base import BaseMLModel, Result
import time


class ClusteringEngine(BaseMLModel):
    """Engine for clustering algorithms"""

    def __init__(self, algorithm: str):
        super().__init__(f"clustering_{algorithm}", "clustering")
        self.algorithm = algorithm
        self.labels_ = None

    async def _setup(self):
        """Initialize clustering model"""
        if self.algorithm == "kmeans":
            from sklearn.cluster import KMeans
            self.model = KMeans(n_clusters=3, random_state=42)

        elif self.algorithm == "dbscan":
            from sklearn.cluster import DBSCAN
            self.model = DBSCAN(eps=0.5, min_samples=5)

        elif self.algorithm == "hierarchical":
            from sklearn.cluster import AgglomerativeClustering
            self.model = AgglomerativeClustering(n_clusters=3)

        elif self.algorithm == "gaussian":
            from sklearn.mixture import GaussianMixture
            self.model = GaussianMixture(n_components=3, random_state=42)

        elif self.algorithm == "meanshift":
            from sklearn.cluster import MeanShift
            self.model = MeanShift()

        elif self.algorithm == "spectral":
            from sklearn.cluster import SpectralClustering
            self.model = SpectralClustering(n_clusters=3, random_state=42)

        elif self.algorithm == "optics":
            from sklearn.cluster import OPTICS
            self.model = OPTICS(min_samples=5)

        elif self.algorithm == "birch":
            from sklearn.cluster import Birch
            self.model = Birch(n_clusters=3)

        else:
            raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")

    async def train(self, X: Any, y: Any = None, **kwargs) -> Dict[str, Any]:
        """Fit clustering model"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Update parameters if provided
        if 'n_clusters' in kwargs and hasattr(self.model, 'n_clusters'):
            self.model.n_clusters = kwargs['n_clusters']

        # Standardize features if requested
        if kwargs.get('standardize', True):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        # Fit model
        if self.algorithm == "gaussian":
            self.model.fit(X_scaled)
            self.labels_ = self.model.predict(X_scaled)
        else:
            self.labels_ = self.model.fit_predict(X_scaled)

        self.is_trained = True

        # Calculate metrics
        metrics = await self._calculate_metrics(X_scaled, self.labels_)

        # Get cluster centers if available
        centers = None
        if hasattr(self.model, 'cluster_centers_'):
            centers = self.model.cluster_centers_.tolist()
        elif hasattr(self.model, 'means_'):
            centers = self.model.means_.tolist()

        training_time = time.time() - start_time

        return {
            "algorithm": self.algorithm,
            "n_clusters": len(np.unique(self.labels_)),
            "labels": self.labels_.tolist(),
            "metrics": metrics,
            "centers": centers,
            "training_time": training_time
        }

    async def predict(self, X: Any, **kwargs) -> Any:
        """Predict cluster labels for new data"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        # Standardize if needed
        if kwargs.get('standardize', True):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        if hasattr(self.model, 'predict'):
            return self.model.predict(X_scaled)
        else:
            # For algorithms without predict method
            return self.model.fit_predict(X_scaled)

    async def evaluate(self, X: Any, y: Any = None, **kwargs) -> Dict[str, Any]:
        """Evaluate clustering quality"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        labels = self.labels_ if self.labels_ is not None else self.model.labels_
        metrics = await self._calculate_metrics(X, labels)

        return {"metrics": metrics}

    async def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering metrics"""
        metrics = {}

        if len(np.unique(labels)) > 1:
            metrics["silhouette_score"] = float(silhouette_score(X, labels))
            metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X, labels))
            metrics["davies_bouldin_score"] = float(davies_bouldin_score(X, labels))

        # Calculate inertia for KMeans
        if hasattr(self.model, 'inertia_'):
            metrics["inertia"] = float(self.model.inertia_)

        return metrics

    async def execute(self, data: Any, params: Dict[str, Any]) -> Any:
        """Execute clustering task"""
        return await self.train(data, **params)


class DimensionalityReductionEngine(BaseMLModel):
    """Engine for dimensionality reduction"""

    def __init__(self, algorithm: str):
        super().__init__(f"dim_reduction_{algorithm}", "dimensionality_reduction")
        self.algorithm = algorithm

    async def _setup(self):
        """Initialize dimensionality reduction model"""
        if self.algorithm == "pca":
            from sklearn.decomposition import PCA
            self.model = PCA(n_components=2)

        elif self.algorithm == "tsne":
            from sklearn.manifold import TSNE
            self.model = TSNE(n_components=2, random_state=42)

        elif self.algorithm == "umap":
            try:
                import umap
                self.model = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                # Fallback to PCA if UMAP not installed
                from sklearn.decomposition import PCA
                self.model = PCA(n_components=2)

        elif self.algorithm == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            self.model = LinearDiscriminantAnalysis()

        elif self.algorithm == "ica":
            from sklearn.decomposition import FastICA
            self.model = FastICA(n_components=2, random_state=42)

        elif self.algorithm == "nmf":
            from sklearn.decomposition import NMF
            self.model = NMF(n_components=2, random_state=42)

        elif self.algorithm == "factor":
            from sklearn.decomposition import FactorAnalysis
            self.model = FactorAnalysis(n_components=2, random_state=42)

        elif self.algorithm == "truncated_svd":
            from sklearn.decomposition import TruncatedSVD
            self.model = TruncatedSVD(n_components=2, random_state=42)

        else:
            raise ValueError(f"Unknown dimensionality reduction algorithm: {self.algorithm}")

    async def train(self, X: Any, y: Any = None, **kwargs) -> Dict[str, Any]:
        """Fit dimensionality reduction model"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Update components if provided
        if 'n_components' in kwargs and hasattr(self.model, 'n_components'):
            self.model.n_components = kwargs['n_components']

        # Fit model
        if self.algorithm in ["lda"] and y is not None:
            self.model.fit(X, y)
        else:
            self.model.fit(X)

        self.is_trained = True

        # Transform data
        X_transformed = self.model.transform(X)

        # Get explained variance if available
        explained_variance = None
        if hasattr(self.model, 'explained_variance_ratio_'):
            explained_variance = self.model.explained_variance_ratio_.tolist()

        # Get components if available
        components = None
        if hasattr(self.model, 'components_'):
            components = self.model.components_.tolist()

        training_time = time.time() - start_time

        return {
            "algorithm": self.algorithm,
            "n_components": self.model.n_components if hasattr(self.model, 'n_components') else None,
            "transformed_shape": X_transformed.shape,
            "explained_variance": explained_variance,
            "components": components,
            "training_time": training_time
        }

    async def predict(self, X: Any, **kwargs) -> Any:
        """Transform new data"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        return self.model.transform(X)

    async def evaluate(self, X: Any, y: Any = None, **kwargs) -> Dict[str, Any]:
        """Evaluate dimensionality reduction"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        X_transformed = self.model.transform(X)

        metrics = {
            "original_shape": X.shape,
            "reduced_shape": X_transformed.shape,
            "reduction_ratio": X_transformed.shape[1] / X.shape[1]
        }

        # Reconstruction error for applicable methods
        if hasattr(self.model, 'inverse_transform'):
            X_reconstructed = self.model.inverse_transform(X_transformed)
            reconstruction_error = np.mean((X - X_reconstructed) ** 2)
            metrics["reconstruction_error"] = float(reconstruction_error)

        return {"metrics": metrics}

    async def execute(self, data: Any, params: Dict[str, Any]) -> Any:
        """Execute dimensionality reduction task"""
        return await self.train(data, params.get('target'), **params)


class AnomalyDetectionEngine(BaseMLModel):
    """Engine for anomaly detection"""

    def __init__(self, algorithm: str):
        super().__init__(f"anomaly_{algorithm}", "anomaly_detection")
        self.algorithm = algorithm

    async def _setup(self):
        """Initialize anomaly detection model"""
        if self.algorithm == "isolation":
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(random_state=42)

        elif self.algorithm == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            self.model = LocalOutlierFactor(novelty=True)

        elif self.algorithm == "ocsvm":
            from sklearn.svm import OneClassSVM
            self.model = OneClassSVM()

        elif self.algorithm == "elliptic":
            from sklearn.covariance import EllipticEnvelope
            self.model = EllipticEnvelope(random_state=42)

        else:
            raise ValueError(f"Unknown anomaly detection algorithm: {self.algorithm}")

    async def train(self, X: Any, y: Any = None, **kwargs) -> Dict[str, Any]:
        """Train anomaly detection model"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Fit model
        self.model.fit(X)
        self.is_trained = True

        # Predict anomalies on training data
        predictions = self.model.predict(X)
        anomaly_scores = None

        if hasattr(self.model, 'score_samples'):
            anomaly_scores = self.model.score_samples(X)
        elif hasattr(self.model, 'decision_function'):
            anomaly_scores = self.model.decision_function(X)

        # Calculate statistics
        n_anomalies = np.sum(predictions == -1)
        anomaly_ratio = n_anomalies / len(predictions)

        training_time = time.time() - start_time

        return {
            "algorithm": self.algorithm,
            "n_samples": len(X),
            "n_anomalies": int(n_anomalies),
            "anomaly_ratio": float(anomaly_ratio),
            "predictions": predictions.tolist(),
            "anomaly_scores": anomaly_scores.tolist() if anomaly_scores is not None else None,
            "training_time": training_time
        }

    async def predict(self, X: Any, **kwargs) -> Any:
        """Predict anomalies"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        predictions = self.model.predict(X)

        if kwargs.get('return_scores', False):
            if hasattr(self.model, 'score_samples'):
                scores = self.model.score_samples(X)
            elif hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X)
            else:
                scores = None

            return {'predictions': predictions, 'scores': scores}

        return predictions

    async def evaluate(self, X: Any, y: Any = None, **kwargs) -> Dict[str, Any]:
        """Evaluate anomaly detection"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        predictions = self.model.predict(X)

        metrics = {
            "n_anomalies": int(np.sum(predictions == -1)),
            "anomaly_ratio": float(np.sum(predictions == -1) / len(predictions))
        }

        # If true labels provided, calculate accuracy metrics
        if y is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            # Convert -1/1 to 0/1 for metrics
            pred_binary = (predictions == -1).astype(int)
            y_binary = (y == -1).astype(int) if -1 in y else y

            metrics.update({
                "accuracy": float(accuracy_score(y_binary, pred_binary)),
                "precision": float(precision_score(y_binary, pred_binary, zero_division=0)),
                "recall": float(recall_score(y_binary, pred_binary, zero_division=0)),
                "f1_score": float(f1_score(y_binary, pred_binary, zero_division=0))
            })

        return {"metrics": metrics}

    async def execute(self, data: Any, params: Dict[str, Any]) -> Any:
        """Execute anomaly detection task"""
        return await self.train(data, **params)
