"""
Machine Learning Models for OpenStatica
Pydantic models for ML operations
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime


class MLTask(str, Enum):
    """Machine learning task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality"
    ANOMALY_DETECTION = "anomaly"
    TIME_SERIES_FORECASTING = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    REINFORCEMENT_LEARNING = "reinforcement"


class Algorithm(str, Enum):
    """ML algorithms"""
    # Classification
    LOGISTIC_REGRESSION = "logistic"
    SVM = "svm"
    RANDOM_FOREST = "rf"
    GRADIENT_BOOSTING = "gb"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    NAIVE_BAYES = "nb"
    KNN = "knn"
    DECISION_TREE = "dt"

    # Deep Learning
    MLP = "mlp"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    AUTOENCODER = "autoencoder"

    # Clustering
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    MEAN_SHIFT = "mean_shift"
    SPECTRAL = "spectral"

    # Dimensionality Reduction
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"
    LDA = "lda"
    ICA = "ica"
    NMF = "nmf"


class OptimizationMethod(str, Enum):
    """Hyperparameter optimization methods"""
    GRID_SEARCH = "grid"
    RANDOM_SEARCH = "random"
    BAYESIAN = "bayesian"
    HYPERBAND = "hyperband"
    OPTUNA = "optuna"
    GENETIC = "genetic"


class MetricType(str, Enum):
    """Evaluation metrics"""
    # Classification
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    LOG_LOSS = "log_loss"
    MATTHEWS_CORR = "matthews"

    # Regression
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"
    MAPE = "mape"
    EXPLAINED_VARIANCE = "explained_variance"

    # Clustering
    SILHOUETTE = "silhouette"
    CALINSKI_HARABASZ = "calinski_harabasz"
    DAVIES_BOULDIN = "davies_bouldin"
    INERTIA = "inertia"


class HyperparametersConfig(BaseModel):
    """Hyperparameter configuration"""
    algorithm: Algorithm
    parameters: Dict[str, Any]
    search_space: Optional[Dict[str, List[Any]]] = None
    optimization_method: Optional[OptimizationMethod] = None
    n_trials: Optional[int] = 100
    cv_folds: Optional[int] = 5
    scoring_metric: Optional[MetricType] = None


class FeatureImportance(BaseModel):
    """Feature importance scores"""
    feature: str
    importance: float
    std: Optional[float] = None
    rank: Optional[int] = None


class ModelMetrics(BaseModel):
    """Model evaluation metrics"""
    metric_name: MetricType
    value: float
    std: Optional[float] = None
    confidence_interval: Optional[List[float]] = None


class MLRequest(BaseModel):
    """Base ML request"""
    session_id: str
    task: MLTask
    algorithm: Algorithm
    features: List[str]
    target: Optional[str] = None  # For supervised learning

    # Data split
    train_test_split: float = 0.8
    validation_split: Optional[float] = None
    stratify: bool = True
    random_state: Optional[int] = 42

    # Training options
    auto_ml: bool = False
    cross_validate: bool = True
    cv_folds: int = 5

    # Hyperparameters
    hyperparameters: Optional[Dict[str, Any]] = {}
    optimize_hyperparameters: bool = False
    optimization_config: Optional[HyperparametersConfig] = None

    # Options
    options: Dict[str, Any] = {}

    @validator('train_test_split')
    def validate_split(cls, v):
        if not 0 < v < 1:
            raise ValueError('Train test split must be between 0 and 1')
        return v


class MLResult(BaseModel):
    """ML training result"""
    model_id: str
    task: MLTask
    algorithm: Algorithm

    # Training info
    training_time: float
    n_samples_train: int
    n_samples_test: Optional[int] = None
    n_features: int

    # Metrics
    metrics: Dict[str, float]
    cv_scores: Optional[List[float]] = None

    # Feature importance
    feature_importance: Optional[List[FeatureImportance]] = None

    # Model specific
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Dict[str, float]]] = None
    roc_curve: Optional[Dict[str, List[float]]] = None
    pr_curve: Optional[Dict[str, List[float]]] = None

    # Clustering specific
    cluster_centers: Optional[List[List[float]]] = None
    cluster_labels: Optional[List[int]] = None
    n_clusters: Optional[int] = None

    # Hyperparameters
    best_hyperparameters: Optional[Dict[str, Any]] = None
    hyperparameter_search_results: Optional[List[Dict[str, Any]]] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    execution_time: float


class ModelTrainingRequest(MLRequest):
    """Detailed model training request"""

    # Advanced options
    class_weight: Optional[Union[str, Dict[int, float]]] = None
    sample_weight: Optional[List[float]] = None

    # Early stopping
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 0.001

    # Ensemble
    ensemble: bool = False
    ensemble_method: Optional[str] = None  # voting, stacking, blending
    base_models: Optional[List[Algorithm]] = None

    # Feature engineering
    feature_engineering: Optional[Dict[str, Any]] = None
    polynomial_features: bool = False
    interaction_terms: bool = False

    # Preprocessing
    scale_features: bool = True
    encode_categorical: bool = True
    handle_missing: str = "drop"  # drop, impute_mean, impute_median, impute_mode
    handle_outliers: Optional[str] = None  # clip, remove, transform

    # Interpretability
    compute_shap: bool = False
    compute_lime: bool = False
    compute_permutation_importance: bool = False


class ModelPredictionRequest(BaseModel):
    """Model prediction request"""
    session_id: str
    model_id: str
    data: Union[List[Dict[str, Any]], str]  # Data or session_id with data

    # Prediction options
    return_probabilities: bool = False
    return_confidence: bool = False
    return_explanations: bool = False

    # Batch processing
    batch_size: Optional[int] = None

    # Post-processing
    threshold: Optional[float] = 0.5  # For binary classification
    top_k: Optional[int] = None  # For multi-class


class ModelPredictionResult(BaseModel):
    """Model prediction result"""
    model_id: str
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    confidence_scores: Optional[List[float]] = None
    explanations: Optional[List[Dict[str, Any]]] = None

    # Metadata
    n_predictions: int
    prediction_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelEvaluationRequest(BaseModel):
    """Model evaluation request"""
    session_id: str
    model_id: str
    test_data: Union[str, Dict[str, Any]]  # Session ID or direct data

    # Evaluation options
    metrics: List[MetricType]
    compute_confusion_matrix: bool = True
    compute_roc_curve: bool = True
    compute_pr_curve: bool = True
    compute_calibration: bool = False

    # Cross-validation
    cross_validate: bool = False
    cv_folds: int = 5

    # Bootstrap
    bootstrap: bool = False
    n_bootstrap: int = 1000


class ModelEvaluationResult(BaseModel):
    """Model evaluation result"""
    model_id: str

    # Metrics
    metrics: Dict[str, ModelMetrics]

    # Classification specific
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Dict[str, float]]] = None
    roc_curve: Optional[Dict[str, Any]] = None
    pr_curve: Optional[Dict[str, Any]] = None
    calibration_curve: Optional[Dict[str, Any]] = None

    # Regression specific
    residuals: Optional[List[float]] = None
    qq_plot: Optional[Dict[str, Any]] = None

    # Cross-validation
    cv_scores: Optional[Dict[str, List[float]]] = None

    # Bootstrap
    bootstrap_scores: Optional[Dict[str, Dict[str, float]]] = None

    # Evaluation metadata
    n_samples: int
    evaluation_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelComparisonRequest(BaseModel):
    """Request to compare multiple models"""
    session_id: str
    model_ids: List[str]
    test_data: Union[str, Dict[str, Any]]
    metrics: List[MetricType]

    # Statistical tests
    statistical_test: Optional[str] = None  # t-test, wilcoxon, friedman
    significance_level: float = 0.05


class ModelComparisonResult(BaseModel):
    """Result of model comparison"""
    comparison_table: Dict[str, Dict[str, float]]
    best_model: str
    ranking: List[str]

    # Statistical tests
    statistical_test_results: Optional[Dict[str, Any]] = None
    significant_differences: Optional[Dict[str, bool]] = None

    # Visualizations
    performance_plot: Optional[str] = None  # Base64 encoded plot

    timestamp: datetime = Field(default_factory=datetime.now)


class AutoMLRequest(MLRequest):
    """AutoML request"""
    time_budget: Optional[int] = 3600  # seconds
    max_models: Optional[int] = 20

    # Search space
    algorithms_to_try: Optional[List[Algorithm]] = None
    include_ensembles: bool = True
    include_deep_learning: bool = False

    # Optimization
    optimization_metric: MetricType
    optimization_direction: str = "maximize"  # maximize, minimize

    # Feature selection
    feature_selection: bool = True
    max_features: Optional[int] = None

    # Model selection
    refit_best_model: bool = True
    keep_top_k: int = 5


class AutoMLResult(BaseModel):
    """AutoML result"""
    best_model_id: str
    best_score: float

    # Leaderboard
    leaderboard: List[Dict[str, Any]]

    # Feature importance across models
    aggregated_feature_importance: List[FeatureImportance]

    # Selected features
    selected_features: Optional[List[str]] = None

    # Training history
    training_history: List[Dict[str, Any]]

    # Best hyperparameters
    best_hyperparameters: Dict[str, Any]

    # Ensemble details
    ensemble_weights: Optional[Dict[str, float]] = None

    # Metadata
    total_time: float
    n_models_trained: int
    timestamp: datetime = Field(default_factory=datetime.now)


class DeepLearningConfig(BaseModel):
    """Deep learning specific configuration"""
    architecture: str  # mlp, cnn, rnn, transformer, etc.
    layers: List[Dict[str, Any]]

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str

    # Regularization
    dropout_rate: Optional[float] = None
    l1_regularization: Optional[float] = None
    l2_regularization: Optional[float] = None

    # Callbacks
    early_stopping: bool = True
    reduce_lr: bool = True
    checkpoint: bool = True

    # GPU
    use_gpu: bool = False
    distributed_training: bool = False


class ModelExportRequest(BaseModel):
    """Request to export a model"""
    model_id: str
    format: str = "pickle"  # pickle, joblib, onnx, pmml, tensorflow, pytorch
    include_preprocessor: bool = True
    include_metadata: bool = True
    compression: Optional[str] = None  # gzip, zip


class ModelDeploymentRequest(BaseModel):
    """Request to deploy a model"""
    model_id: str
    deployment_type: str  # rest_api, batch, streaming, edge

    # API deployment
    endpoint_name: Optional[str] = None
    auto_scaling: bool = True
    min_instances: int = 1
    max_instances: int = 10

    # Monitoring
    enable_monitoring: bool = True
    log_predictions: bool = True
    alert_thresholds: Optional[Dict[str, float]] = None

    # A/B testing
    enable_ab_testing: bool = False
    traffic_percentage: Optional[float] = None
