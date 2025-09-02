"""
Validators for OpenStatica
Input validation and data quality checks
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import re
from pydantic import BaseModel, validator
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for data inputs"""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe and return validation results"""
        issues = []
        warnings = []

        # Check if empty
        if df.empty:
            issues.append("Dataframe is empty")
            return {'valid': False, 'issues': issues, 'warnings': warnings}

        # Check dimensions
        if len(df.columns) == 0:
            issues.append("No columns in dataframe")

        if len(df) > 1000000:
            warnings.append(f"Large dataset: {len(df)} rows may impact performance")

        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            issues.append(f"Duplicate column names: {duplicate_cols}")

        # Check for completely empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            warnings.append(f"Empty columns: {empty_cols}")

        # Check for high missing data
        missing_pct = df.isna().sum() / len(df)
        high_missing = missing_pct[missing_pct > 0.5].to_dict()
        if high_missing:
            warnings.append(f"Columns with >50% missing: {list(high_missing.keys())}")

        # Check data types
        mixed_types = []
        for col in df.select_dtypes(include='object').columns:
            if df[col].apply(type).nunique() > 1:
                mixed_types.append(col)

        if mixed_types:
            warnings.append(f"Columns with mixed types: {mixed_types}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict()
        }

    @staticmethod
    def validate_column_names(columns: List[str]) -> Dict[str, Any]:
        """Validate column names"""
        issues = []
        cleaned = []

        for col in columns:
            # Check for invalid characters
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col):
                issues.append(f"Invalid column name: {col}")
                # Clean the name
                clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                if not clean_col[0].isalpha():
                    clean_col = 'col_' + clean_col
                cleaned.append(clean_col)
            else:
                cleaned.append(col)

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'cleaned': cleaned
        }

    @staticmethod
    def validate_numeric_data(data: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Validate numeric data"""
        issues = []
        warnings = []

        if isinstance(data, pd.Series):
            data = data.values

        # Check for infinite values
        if np.any(np.isinf(data)):
            issues.append("Data contains infinite values")

        # Check for NaN
        nan_count = np.sum(np.isnan(data))
        if nan_count > 0:
            warnings.append(f"Data contains {nan_count} NaN values")

        # Check range
        if len(data) > 0:
            data_clean = data[~np.isnan(data)]
            if len(data_clean) > 0:
                data_range = np.ptp(data_clean)
                if data_range == 0:
                    warnings.append("Data has zero variance")
                elif data_range > 1e10:
                    warnings.append("Very large data range may cause numerical issues")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    @staticmethod
    def validate_categorical_data(data: pd.Series) -> Dict[str, Any]:
        """Validate categorical data"""
        issues = []
        warnings = []

        # Check cardinality
        unique_ratio = data.nunique() / len(data)

        if unique_ratio > 0.9:
            warnings.append("High cardinality - might be better as numeric or ID")

        if unique_ratio == 1:
            warnings.append("Each value is unique - not suitable for categorical analysis")

        # Check for empty strings
        if data.dtype == 'object':
            empty_count = (data == '').sum()
            if empty_count > 0:
                warnings.append(f"Contains {empty_count} empty strings")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'n_unique': data.nunique(),
            'unique_ratio': unique_ratio
        }


class FileValidator:
    """Validator for file uploads"""

    ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json', '.parquet', '.sav', '.dta', '.sas7bdat'}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

    @classmethod
    def validate_file(cls, filename: str, file_size: int) -> Dict[str, Any]:
        """Validate uploaded file"""
        issues = []

        # Check extension
        import os
        _, ext = os.path.splitext(filename.lower())

        if ext not in cls.ALLOWED_EXTENSIONS:
            issues.append(f"File type {ext} not supported")

        # Check size
        if file_size > cls.MAX_FILE_SIZE:
            issues.append(f"File too large: {file_size / 1024 / 1024:.2f}MB (max: 100MB)")

        if file_size == 0:
            issues.append("File is empty")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'extension': ext,
            'size_mb': file_size / 1024 / 1024
        }


class AnalysisValidator:
    """Validator for analysis parameters"""

    @staticmethod
    def validate_test_assumptions(test_type: str, data: pd.DataFrame, **params) -> Dict[str, Any]:
        """Validate statistical test assumptions"""
        assumptions = {}

        if test_type == 'ttest':
            # Check normality
            from scipy import stats

            var = params.get('variable')
            if var and var in data.columns:
                _, p_value = stats.shapiro(data[var].dropna()[:5000])
                assumptions['normality'] = {
                    'test': 'Shapiro-Wilk',
                    'p_value': p_value,
                    'passed': p_value > 0.05
                }

            # Check sample size
            assumptions['sample_size'] = {
                'n': len(data),
                'adequate': len(data) >= 30
            }

        elif test_type == 'anova':
            # Check homogeneity of variance
            groups = params.get('groups', [])
            if len(groups) >= 2:
                from scipy.stats import levene
                group_data = [data[g].dropna() for g in groups if g in data.columns]
                if len(group_data) >= 2:
                    stat, p_value = levene(*group_data)
                    assumptions['homogeneity'] = {
                        'test': 'Levene',
                        'p_value': p_value,
                        'passed': p_value > 0.05
                    }

        elif test_type == 'regression':
            # Check multicollinearity
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            predictors = params.get('predictors', [])
            if len(predictors) > 1:
                X = data[predictors].dropna()
                vif_data = pd.DataFrame()
                vif_data["Variable"] = predictors
                vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                                   for i in range(X.shape[1])]

                assumptions['multicollinearity'] = {
                    'vif': vif_data.to_dict(),
                    'passed': all(vif_data["VIF"] < 10)
                }

        return assumptions

    @staticmethod
    def validate_ml_parameters(task: str, algorithm: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ML parameters"""
        issues = []
        warnings = []

        # Check train/test split
        train_size = params.get('train_test_split', 0.8)
        if not 0.5 <= train_size <= 0.95:
            warnings.append(f"Unusual train/test split: {train_size}")

        # Check cross-validation folds
        cv_folds = params.get('cv_folds', 5)
        if cv_folds < 2:
            issues.append("Cross-validation requires at least 2 folds")
        elif cv_folds > 20:
            warnings.append("High number of CV folds may be slow")

        # Algorithm-specific checks
        if algorithm == 'kmeans':
            n_clusters = params.get('n_clusters', 3)
            if n_clusters < 2:
                issues.append("K-means requires at least 2 clusters")
            elif n_clusters > 20:
                warnings.append("High number of clusters may lead to overfitting")

        elif algorithm in ['rf', 'random_forest']:
            n_estimators = params.get('n_estimators', 100)
            if n_estimators < 10:
                warnings.append("Low number of trees may underfit")
            elif n_estimators > 1000:
                warnings.append("High number of trees may be slow")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }


class RequestValidator:
    """Validator for API requests"""

    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """Validate session ID format"""
        # UUID format
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        return bool(uuid_pattern.match(session_id))

    @staticmethod
    def validate_variable_selection(variables: List[str],
                                    available: List[str]) -> Dict[str, Any]:
        """Validate variable selection"""
        issues = []

        if not variables:
            issues.append("No variables selected")

        missing = set(variables) - set(available)
        if missing:
            issues.append(f"Variables not found: {list(missing)}")

        if len(variables) > 100:
            issues.append("Too many variables selected (max: 100)")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'selected': len(variables),
            'missing': list(missing)
        }


class OutputValidator:
    """Validator for outputs and results"""

    @staticmethod
    def validate_results(results: Dict[str, Any], expected_keys: List[str]) -> Dict[str, Any]:
        """Validate analysis results structure"""
        issues = []

        missing_keys = set(expected_keys) - set(results.keys())
        if missing_keys:
            issues.append(f"Missing expected keys: {list(missing_keys)}")

        # Check for None values in critical fields
        for key in expected_keys:
            if key in results and results[key] is None:
                issues.append(f"Null value for required field: {key}")

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    @staticmethod
    def sanitize_output(data: Any) -> Any:
        """Sanitize output data"""
        if isinstance(data, dict):
            return {k: OutputValidator.sanitize_output(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [OutputValidator.sanitize_output(v) for v in data]
        elif isinstance(data, float):
            if np.isnan(data):
                return None
            elif np.isinf(data):
                return str(data)
            return data
        elif isinstance(data, np.ndarray):
            return OutputValidator.sanitize_output(data.tolist())
        elif pd.isna(data):
            return None
        return data


# Pydantic models for request validation
class DataUploadValidator(BaseModel):
    """Validator for data upload requests"""
    filename: str
    format: str

    @validator('format')
    def validate_format(cls, v):
        allowed = ['csv', 'excel', 'json', 'parquet', 'spss', 'stata', 'sas']
        if v not in allowed:
            raise ValueError(f'Format must be one of {allowed}')
        return v


class AnalysisRequestValidator(BaseModel):
    """Validator for analysis requests"""
    session_id: str
    analysis_type: str
    variables: List[str]
    options: Dict[str, Any] = {}

    @validator('session_id')
    def validate_session(cls, v):
        if not RequestValidator.validate_session_id(v):
            raise ValueError('Invalid session ID format')
        return v

    @validator('variables')
    def validate_variables(cls, v):
        if not v:
            raise ValueError('At least one variable must be selected')
        if len(v) > 100:
            raise ValueError('Too many variables (max: 100)')
        return v


class MLRequestValidator(BaseModel):
    """Validator for ML requests"""
    session_id: str
    task: str
    algorithm: str
    features: List[str]
    target: Optional[str] = None

    @validator('task')
    def validate_task(cls, v):
        allowed = ['classification', 'regression', 'clustering',
                   'dimensionality', 'anomaly']
        if v not in allowed:
            raise ValueError(f'Task must be one of {allowed}')
        return v

    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError('At least one feature must be selected')
        return v

    @validator('target')
    def validate_target(cls, v, values):
        task = values.get('task')
        if task in ['classification', 'regression'] and not v:
            raise ValueError(f'Target variable required for {task}')
        return v


def validate_input(validator_class: BaseModel, data: Dict[str, Any]) -> Dict[str, Any]:
    """Generic input validation using Pydantic models"""
    try:
        validated = validator_class(**data)
        return {'valid': True, 'data': validated.dict()}
    except Exception as e:
        return {'valid': False, 'errors': str(e)}
