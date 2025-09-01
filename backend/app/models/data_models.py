"""
Data Models for OpenStatica
Pydantic models for data operations
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class FileFormat(str, Enum):
    """Supported file formats"""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"
    STATA = "stata"
    SPSS = "spss"
    SAS = "sas"


class DataType(str, Enum):
    """Data types for columns"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"


class TransformationType(str, Enum):
    """Types of data transformations"""
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"
    ENCODE = "encode"
    IMPUTE = "impute"
    BINNING = "binning"
    LOG_TRANSFORM = "log"
    SQRT_TRANSFORM = "sqrt"
    POLYNOMIAL = "polynomial"
    DIFFERENCE = "difference"


class ImputationStrategy(str, Enum):
    """Imputation strategies for missing values"""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "forward"
    BACKWARD_FILL = "backward"
    INTERPOLATE = "interpolate"
    DROP = "drop"
    CONSTANT = "constant"


class ColumnInfo(BaseModel):
    """Information about a data column"""
    name: str
    dtype: DataType
    missing_count: int = 0
    missing_percentage: float = 0.0
    unique_count: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    mode: Optional[Union[str, float]] = None
    categories: Optional[List[str]] = None


class DataStatistics(BaseModel):
    """Overall data statistics"""
    row_count: int
    column_count: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    text_columns: List[str]
    missing_cells: int
    missing_percentage: float
    memory_usage: float  # in MB
    duplicated_rows: int


class DataPreview(BaseModel):
    """Data preview for display"""
    columns: List[str]
    data: List[Dict[str, Any]]
    total_rows: int
    preview_rows: int = 10


class DataUploadRequest(BaseModel):
    """Request for data upload"""
    filename: str
    format: FileFormat
    encoding: str = "utf-8"
    delimiter: Optional[str] = ","
    has_header: bool = True
    parse_dates: bool = True
    na_values: Optional[List[str]] = None


class DataUploadResponse(BaseModel):
    """Response after data upload"""
    session_id: str
    filename: str
    format: FileFormat
    rows: int
    columns: int
    column_names: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: Optional[List[str]] = []
    preview: List[Dict[str, Any]]
    statistics: Optional[DataStatistics] = None
    upload_time: datetime = Field(default_factory=datetime.now)
    warnings: Optional[List[str]] = None


class DataTransformRequest(BaseModel):
    """Request for data transformation"""
    session_id: str
    transformation_type: TransformationType
    columns: List[str]
    options: Dict[str, Any] = {}

    # Transformation-specific options
    imputation_strategy: Optional[ImputationStrategy] = None
    constant_value: Optional[Union[str, float]] = None
    encoding_type: Optional[str] = None  # onehot, label, ordinal
    bins: Optional[int] = None
    polynomial_degree: Optional[int] = None
    lag: Optional[int] = None


class DataFilterRequest(BaseModel):
    """Request for data filtering"""
    session_id: str
    filters: List[Dict[str, Any]]
    logic: str = "and"  # and/or logic for multiple filters

    @validator('logic')
    def validate_logic(cls, v):
        if v not in ['and', 'or']:
            raise ValueError('Logic must be "and" or "or"')
        return v


class DataAggregateRequest(BaseModel):
    """Request for data aggregation"""
    session_id: str
    group_by: List[str]
    aggregations: Dict[str, List[str]]  # column: [operations]

    class Config:
        schema_extra = {
            "example": {
                "session_id": "abc123",
                "group_by": ["category", "region"],
                "aggregations": {
                    "sales": ["sum", "mean", "count"],
                    "profit": ["sum", "max"]
                }
            }
        }


class DataMergeRequest(BaseModel):
    """Request for merging datasets"""
    left_session_id: str
    right_session_id: str
    how: str = "inner"  # inner, outer, left, right
    left_on: List[str]
    right_on: List[str]
    suffixes: List[str] = ["_left", "_right"]

    @validator('how')
    def validate_merge_type(cls, v):
        valid_types = ['inner', 'outer', 'left', 'right']
        if v not in valid_types:
            raise ValueError(f'Merge type must be one of {valid_types}')
        return v


class DataExportRequest(BaseModel):
    """Request for data export"""
    session_id: str
    format: FileFormat
    columns: Optional[List[str]] = None
    include_index: bool = False
    encoding: str = "utf-8"
    compression: Optional[str] = None  # gzip, zip, etc.


class DataValidationRule(BaseModel):
    """Data validation rule"""
    column: str
    rule_type: str  # range, pattern, unique, not_null, etc.
    parameters: Dict[str, Any]
    error_message: Optional[str] = None


class DataQualityReport(BaseModel):
    """Data quality assessment report"""
    session_id: str
    overall_score: float  # 0-100
    completeness_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    column_quality: Dict[str, Dict[str, float]]


class DataSplitRequest(BaseModel):
    """Request for splitting data"""
    session_id: str
    split_type: str = "random"  # random, stratified, time-based
    test_size: float = 0.2
    validation_size: Optional[float] = None
    stratify_column: Optional[str] = None
    random_state: Optional[int] = 42
    shuffle: bool = True


class DataSplitResponse(BaseModel):
    """Response after data split"""
    train_session_id: str
    test_session_id: str
    validation_session_id: Optional[str] = None
    train_size: int
    test_size: int
    validation_size: Optional[int] = None
    split_proportions: Dict[str, float]


class VariableMetadata(BaseModel):
    """Metadata for a variable"""
    name: str
    label: Optional[str] = None
    description: Optional[str] = None
    unit: Optional[str] = None
    scale: Optional[str] = None  # nominal, ordinal, interval, ratio
    role: Optional[str] = None  # input, target, id, weight
    value_labels: Optional[Dict[Union[int, str], str]] = None
    missing_values: Optional[List[Any]] = None


class DatasetMetadata(BaseModel):
    """Metadata for entire dataset"""
    name: str
    description: Optional[str] = None
    source: Optional[str] = None
    created_date: datetime = Field(default_factory=datetime.now)
    modified_date: datetime = Field(default_factory=datetime.now)
    author: Optional[str] = None
    license: Optional[str] = None
    tags: Optional[List[str]] = []
    variables: List[VariableMetadata] = []
    notes: Optional[str] = None
