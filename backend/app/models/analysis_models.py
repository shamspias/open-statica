"""
Analysis Models for OpenStatica
Pydantic models for statistical analysis operations
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime


class TestType(str, Enum):
    """Types of statistical tests"""
    # Parametric tests
    TTEST_ONE = "ttest_one"
    TTEST_INDEPENDENT = "ttest_independent"
    TTEST_PAIRED = "ttest_paired"
    ANOVA_ONE = "anova_one"
    ANOVA_TWO = "anova_two"
    ANOVA_REPEATED = "anova_repeated"
    ANCOVA = "ancova"
    MANOVA = "manova"

    # Non-parametric tests
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"

    # Correlation tests
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    PARTIAL = "partial_correlation"

    # Association tests
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    MCNEMAR = "mcnemar"
    COCHRAN_Q = "cochran_q"

    # Normality tests
    SHAPIRO_WILK = "shapiro_wilk"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    ANDERSON_DARLING = "anderson_darling"
    JARQUE_BERA = "jarque_bera"


class RegressionType(str, Enum):
    """Types of regression models"""
    LINEAR = "linear"
    MULTIPLE = "multiple"
    POLYNOMIAL = "polynomial"
    LOGISTIC = "logistic"
    ORDINAL = "ordinal"
    MULTINOMIAL = "multinomial"
    POISSON = "poisson"
    NEGATIVE_BINOMIAL = "negative_binomial"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    QUANTILE = "quantile"
    ROBUST = "robust"
    COX = "cox"
    MIXED = "mixed"


class PostHocTest(str, Enum):
    """Post-hoc test types"""
    TUKEY = "tukey"
    BONFERRONI = "bonferroni"
    SCHEFFE = "scheffe"
    SIDAK = "sidak"
    HOLM = "holm"
    DUNNETT = "dunnett"
    GAMES_HOWELL = "games_howell"


class AnalysisRequest(BaseModel):
    """Base request for analysis"""
    session_id: str
    columns: List[str]
    options: Dict[str, Any] = {}
    confidence_level: float = 0.95
    missing_values: str = "exclude"  # exclude, include, impute

    @validator('confidence_level')
    def validate_confidence(cls, v):
        if not 0 < v < 1:
            raise ValueError('Confidence level must be between 0 and 1')
        return v


class AnalysisResponse(BaseModel):
    """Base response for analysis"""
    test_name: str
    results: Dict[str, Any]
    interpretation: Optional[str] = None
    assumptions_met: Optional[Dict[str, bool]] = None
    warnings: Optional[List[str]] = None
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class DescriptiveStatisticsRequest(AnalysisRequest):
    """Request for descriptive statistics"""
    include_advanced: bool = True
    percentiles: List[float] = [25, 50, 75]
    include_outliers: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, mad


class DescriptiveStatisticsResult(BaseModel):
    """Result of descriptive statistics"""
    column: str
    count: int
    missing: int
    mean: Optional[float]
    median: Optional[float]
    mode: Optional[Union[float, str]]
    std: Optional[float]
    variance: Optional[float]
    min: Optional[float]
    max: Optional[float]
    range: Optional[float]
    q1: Optional[float]
    q3: Optional[float]
    iqr: Optional[float]
    skewness: Optional[float]
    kurtosis: Optional[float]
    sem: Optional[float]  # Standard error of mean
    cv: Optional[float]  # Coefficient of variation
    percentiles: Dict[str, float]
    confidence_interval: Optional[List[float]]
    outliers: Optional[Dict[str, Any]]


class StatisticalTestRequest(AnalysisRequest):
    """Request for statistical test"""
    test_type: TestType
    alpha: float = 0.05
    alternative: str = "two-sided"  # two-sided, less, greater
    paired: bool = False
    equal_variance: bool = True

    # Test-specific parameters
    group_column: Optional[str] = None
    value_column: Optional[str] = None
    test_value: Optional[float] = None

    # Post-hoc analysis
    post_hoc: Optional[PostHocTest] = None
    correction: Optional[str] = None  # bonferroni, holm, etc.

    @validator('alpha')
    def validate_alpha(cls, v):
        if not 0 < v < 1:
            raise ValueError('Alpha must be between 0 and 1')
        return v


class StatisticalTestResult(AnalysisResponse):
    """Result of statistical test"""
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[Union[float, List[float]]] = None
    effect_size: Optional[Dict[str, float]] = None
    confidence_interval: Optional[List[float]] = None
    power: Optional[float] = None
    sample_sizes: Optional[Dict[str, int]] = None
    group_statistics: Optional[Dict[str, Dict[str, float]]] = None
    post_hoc_results: Optional[Dict[str, Any]] = None
    assumptions: Optional[Dict[str, Dict[str, Any]]] = None
    decision: str  # reject or fail to reject


class CorrelationRequest(AnalysisRequest):
    """Request for correlation analysis"""
    method: str = "pearson"  # pearson, spearman, kendall
    pairwise: bool = True
    min_periods: Optional[int] = None

    # Partial correlation
    control_variables: Optional[List[str]] = None

    # Options
    compute_p_values: bool = True
    compute_confidence_intervals: bool = True
    handle_multicollinearity: bool = False


class CorrelationResult(AnalysisResponse):
    """Result of correlation analysis"""
    correlation_matrix: Dict[str, Dict[str, float]]
    p_value_matrix: Optional[Dict[str, Dict[str, float]]] = None
    confidence_intervals: Optional[Dict[str, Dict[str, List[float]]]] = None
    n_observations: Dict[str, Dict[str, int]]
    significant_correlations: List[Dict[str, Any]]
    vif_scores: Optional[Dict[str, float]] = None  # Variance Inflation Factor


class RegressionRequest(AnalysisRequest):
    """Request for regression analysis"""
    regression_type: RegressionType
    dependent_variable: str
    independent_variables: List[str]

    # Model options
    include_intercept: bool = True
    standardize: bool = False

    # Feature selection
    feature_selection: Optional[str] = None  # forward, backward, stepwise, lasso
    selection_threshold: Optional[float] = 0.05

    # Regularization
    alpha: Optional[float] = None  # For regularized regression
    l1_ratio: Optional[float] = None  # For elastic net

    # Validation
    cross_validate: bool = False
    cv_folds: int = 5

    # Diagnostics
    compute_diagnostics: bool = True
    compute_influence: bool = True


class RegressionResult(AnalysisResponse):
    """Result of regression analysis"""
    model_type: str
    formula: str

    # Coefficients
    coefficients: Dict[str, float]
    standard_errors: Dict[str, float]
    t_values: Dict[str, float]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, List[float]]

    # Model fit
    r_squared: float
    adjusted_r_squared: float
    f_statistic: Optional[float]
    f_p_value: Optional[float]
    aic: float
    bic: float
    log_likelihood: Optional[float]

    # Predictions
    residuals: Optional[List[float]] = None
    fitted_values: Optional[List[float]] = None

    # Diagnostics
    durbin_watson: Optional[float] = None
    condition_number: Optional[float] = None
    vif: Optional[Dict[str, float]] = None

    # Assumption tests
    normality_test: Optional[Dict[str, float]] = None
    heteroscedasticity_test: Optional[Dict[str, float]] = None
    autocorrelation_test: Optional[Dict[str, float]] = None

    # Influence measures
    cooks_distance: Optional[List[float]] = None
    leverage: Optional[List[float]] = None
    studentized_residuals: Optional[List[float]] = None

    # Cross-validation results
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None


class MultivariateRequest(AnalysisRequest):
    """Request for multivariate analysis"""
    analysis_type: str  # pca, factor, lda, cca, manova
    n_components: Optional[int] = None
    rotation: Optional[str] = None  # varimax, promax, oblimin
    method: Optional[str] = None

    # PCA/Factor specific
    standardize: bool = True

    # LDA specific
    target_variable: Optional[str] = None

    # CCA specific
    x_variables: Optional[List[str]] = None
    y_variables: Optional[List[str]] = None


class MultivariateResult(AnalysisResponse):
    """Result of multivariate analysis"""
    method: str
    n_components: int

    # PCA/Factor Analysis
    eigenvalues: Optional[List[float]] = None
    explained_variance: Optional[List[float]] = None
    explained_variance_ratio: Optional[List[float]] = None
    cumulative_variance_ratio: Optional[List[float]] = None
    loadings: Optional[Dict[str, List[float]]] = None
    scores: Optional[List[List[float]]] = None
    communalities: Optional[Dict[str, float]] = None
    uniqueness: Optional[Dict[str, float]] = None

    # Model adequacy
    kmo: Optional[float] = None  # Kaiser-Meyer-Olkin
    bartlett_sphericity: Optional[Dict[str, float]] = None

    # Rotation
    rotation_matrix: Optional[List[List[float]]] = None

    # LDA
    class_means: Optional[Dict[str, List[float]]] = None
    coefficients: Optional[List[List[float]]] = None
    classification_accuracy: Optional[float] = None

    # CCA
    canonical_correlations: Optional[List[float]] = None
    canonical_loadings: Optional[Dict[str, List[List[float]]]] = None
    wilks_lambda: Optional[float] = None


class TimeSeriesRequest(AnalysisRequest):
    """Request for time series analysis"""
    date_column: str
    value_column: str
    frequency: Optional[str] = None  # D, W, M, Q, Y

    # Analysis type
    analysis_type: str  # decomposition, arima, forecast, etc.

    # Decomposition
    decomposition_type: Optional[str] = "additive"  # additive, multiplicative

    # ARIMA
    order: Optional[List[int]] = None  # (p, d, q)
    seasonal_order: Optional[List[int]] = None  # (P, D, Q, s)
    auto_arima: bool = False

    # Forecasting
    forecast_periods: Optional[int] = None
    confidence_level: float = 0.95


class TimeSeriesResult(AnalysisResponse):
    """Result of time series analysis"""
    analysis_type: str

    # Decomposition
    trend: Optional[List[float]] = None
    seasonal: Optional[List[float]] = None
    residual: Optional[List[float]] = None

    # Stationarity tests
    adf_statistic: Optional[float] = None
    adf_p_value: Optional[float] = None
    kpss_statistic: Optional[float] = None
    kpss_p_value: Optional[float] = None

    # ARIMA
    model_order: Optional[List[int]] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    coefficients: Optional[Dict[str, float]] = None

    # Forecast
    forecast_values: Optional[List[float]] = None
    forecast_lower: Optional[List[float]] = None
    forecast_upper: Optional[List[float]] = None
    forecast_dates: Optional[List[str]] = None

    # Accuracy metrics
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None


class SurvivalAnalysisRequest(AnalysisRequest):
    """Request for survival analysis"""
    time_column: str
    event_column: str
    covariates: Optional[List[str]] = None

    analysis_type: str = "kaplan_meier"  # kaplan_meier, cox, aft
    groups: Optional[str] = None

    # Cox specific
    ties_method: str = "efron"  # efron, breslow

    # Time points for survival probability
    time_points: Optional[List[float]] = None
