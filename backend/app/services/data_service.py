"""
Data Service for OpenStatica
Handles all data operations including loading, transformation, and management
"""

import pandas as pd
import numpy as np
import polars as pl
from typing import Dict, Any, List, Optional, Union, Tuple
import io
import json
from pathlib import Path
import asyncio
from datetime import datetime
import hashlib
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DataService:
    """Service for data operations"""

    def __init__(self):
        self.supported_formats = {
            'csv': self.read_csv,
            'excel': self.read_excel,
            'json': self.read_json,
            'parquet': self.read_parquet,
            'stata': self.read_stata,
            'spss': self.read_spss,
            'sas': self.read_sas
        }

    async def load_data(self, file_content: bytes, file_type: str, **kwargs) -> pd.DataFrame:
        """Load data from various file formats"""
        if file_type not in self.supported_formats:
            raise ValueError(f"Unsupported file type: {file_type}")

        reader = self.supported_formats[file_type]
        return await reader(file_content, **kwargs)

    async def read_csv(self, content: bytes, **kwargs) -> pd.DataFrame:
        """Read CSV file"""
        try:
            # Try to detect encoding
            encoding = kwargs.get('encoding', 'utf-8')
            delimiter = kwargs.get('delimiter', ',')

            # Use StringIO for pandas
            text_content = content.decode(encoding)
            df = pd.read_csv(
                io.StringIO(text_content),
                sep=delimiter,
                **{k: v for k, v in kwargs.items() if k not in ['encoding', 'delimiter']}
            )

            return self.clean_dataframe(df)

        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise

    async def read_excel(self, content: bytes, **kwargs) -> pd.DataFrame:
        """Read Excel file"""
        try:
            sheet_name = kwargs.get('sheet_name', 0)
            df = pd.read_excel(
                io.BytesIO(content),
                sheet_name=sheet_name,
                **{k: v for k, v in kwargs.items() if k != 'sheet_name'}
            )

            return self.clean_dataframe(df)

        except Exception as e:
            logger.error(f"Error reading Excel: {e}")
            raise

    async def read_json(self, content: bytes, **kwargs) -> pd.DataFrame:
        """Read JSON file"""
        try:
            text_content = content.decode('utf-8')
            df = pd.read_json(io.StringIO(text_content), **kwargs)
            return self.clean_dataframe(df)

        except Exception as e:
            logger.error(f"Error reading JSON: {e}")
            raise

    async def read_parquet(self, content: bytes, **kwargs) -> pd.DataFrame:
        """Read Parquet file"""
        try:
            df = pd.read_parquet(io.BytesIO(content), **kwargs)
            return self.clean_dataframe(df)

        except Exception as e:
            logger.error(f"Error reading Parquet: {e}")
            raise

    async def read_stata(self, content: bytes, **kwargs) -> pd.DataFrame:
        """Read Stata file"""
        try:
            df = pd.read_stata(io.BytesIO(content), **kwargs)
            return self.clean_dataframe(df)
        except Exception as e:
            logger.error(f"Error reading Stata: {e}")
            raise

    async def read_spss(self, content: bytes, **kwargs) -> pd.DataFrame:
        """Read SPSS file"""
        try:
            # Use pyreadstat for SPSS files
            import pyreadstat
            df, meta = pyreadstat.read_sav(io.BytesIO(content))

            # Store metadata
            df.attrs['spss_metadata'] = meta

            return self.clean_dataframe(df)
        except ImportError:
            logger.error("pyreadstat not installed. Install with: pip install pyreadstat")
            raise
        except Exception as e:
            logger.error(f"Error reading SPSS: {e}")
            raise

    async def read_sas(self, content: bytes, **kwargs) -> pd.DataFrame:
        """Read SAS file"""
        try:
            df = pd.read_sas(io.BytesIO(content), **kwargs)
            return self.clean_dataframe(df)
        except Exception as e:
            logger.error(f"Error reading SAS: {e}")
            raise

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe"""
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Strip whitespace from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        # Convert obvious numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass

        return df

    async def analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze and categorize data types"""
        analysis = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': [],
            'binary': [],
            'ordinal': []
        }

        for col in df.columns:
            dtype = df[col].dtype
            unique_ratio = df[col].nunique() / len(df)

            if pd.api.types.is_numeric_dtype(dtype):
                if df[col].nunique() == 2:
                    analysis['binary'].append(col)
                elif unique_ratio < 0.05 and df[col].nunique() < 20:
                    analysis['ordinal'].append(col)
                else:
                    analysis['numeric'].append(col)

            elif pd.api.types.is_datetime64_any_dtype(dtype):
                analysis['datetime'].append(col)

            else:
                # String/object type
                if df[col].nunique() == 2:
                    analysis['binary'].append(col)
                elif unique_ratio < 0.5:
                    analysis['categorical'].append(col)
                else:
                    analysis['text'].append(col)

        return analysis

    async def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        report = {
            'completeness': {},
            'validity': {},
            'consistency': {},
            'uniqueness': {},
            'overall_score': 0
        }

        # Completeness
        missing_counts = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        missing_cells = missing_counts.sum()

        report['completeness'] = {
            'missing_by_column': missing_counts.to_dict(),
            'total_missing': int(missing_cells),
            'completeness_rate': float(1 - missing_cells / total_cells),
            'columns_with_missing': list(missing_counts[missing_counts > 0].index)
        }

        # Validity checks
        validity_issues = []
        for col in df.select_dtypes(include=[np.number]).columns:
            # Check for outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

            if len(outliers) > 0:
                validity_issues.append({
                    'column': col,
                    'issue': 'outliers',
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100
                })

        report['validity'] = {
            'issues': validity_issues,
            'validity_score': 1 - (len(validity_issues) / len(df.columns))
        }

        # Uniqueness
        duplicate_rows = df.duplicated().sum()
        report['uniqueness'] = {
            'duplicate_rows': int(duplicate_rows),
            'duplicate_rate': float(duplicate_rows / len(df)),
            'unique_identifier_candidates': []
        }

        # Find potential unique identifiers
        for col in df.columns:
            if df[col].nunique() == len(df):
                report['uniqueness']['unique_identifier_candidates'].append(col)

        # Consistency checks
        consistency_issues = []

        # Check for mixed data types in string columns
        for col in df.select_dtypes(include=['object']).columns:
            mixed_types = False
            has_numbers = df[col].astype(str).str.match(r'^\d+$').any()
            has_text = df[col].astype(str).str.match(r'^[a-zA-Z]+$').any()

            if has_numbers and has_text:
                consistency_issues.append({
                    'column': col,
                    'issue': 'mixed_data_types'
                })

        report['consistency'] = {
            'issues': consistency_issues,
            'consistency_score': 1 - (len(consistency_issues) / len(df.columns))
        }

        # Calculate overall score
        report['overall_score'] = np.mean([
            report['completeness']['completeness_rate'],
            report['validity']['validity_score'],
            1 - report['uniqueness']['duplicate_rate'],
            report['consistency']['consistency_score']
        ]) * 100

        return report

    async def transform_data(self, df: pd.DataFrame, transformations: List[Dict]) -> pd.DataFrame:
        """Apply transformations to dataframe"""
        df_transformed = df.copy()

        for transform in transformations:
            transform_type = transform.get('type')
            columns = transform.get('columns', [])
            options = transform.get('options', {})

            if transform_type == 'normalize':
                df_transformed = await self.normalize_columns(df_transformed, columns, **options)
            elif transform_type == 'encode':
                df_transformed = await self.encode_categorical(df_transformed, columns, **options)
            elif transform_type == 'impute':
                df_transformed = await self.impute_missing(df_transformed, columns, **options)
            elif transform_type == 'bin':
                df_transformed = await self.bin_numeric(df_transformed, columns, **options)
            elif transform_type == 'log':
                df_transformed = await self.log_transform(df_transformed, columns, **options)
            elif transform_type == 'difference':
                df_transformed = await self.difference_transform(df_transformed, columns, **options)
            elif transform_type == 'interaction':
                df_transformed = await self.create_interactions(df_transformed, columns, **options)

        return df_transformed

    async def normalize_columns(self, df: pd.DataFrame, columns: List[str], method: str = 'zscore') -> pd.DataFrame:
        """Normalize numeric columns"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        df_norm = df.copy()

        if method == 'zscore':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df_norm[col] = scaler.fit_transform(df[[col]])

        return df_norm

    async def encode_categorical(self, df: pd.DataFrame, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            elif method == 'label':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
            elif method == 'ordinal':
                # User should provide mapping
                pass

        return df_encoded

    async def impute_missing(self, df: pd.DataFrame, columns: List[str], strategy: str = 'mean') -> pd.DataFrame:
        """Impute missing values"""
        df_imputed = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df_imputed[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df_imputed[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df_imputed[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else None, inplace=True)
            elif strategy == 'forward':
                df_imputed[col].fillna(method='ffill', inplace=True)
            elif strategy == 'backward':
                df_imputed[col].fillna(method='bfill', inplace=True)
            elif strategy == 'interpolate':
                df_imputed[col].interpolate(inplace=True)
            elif strategy == 'drop':
                df_imputed = df_imputed.dropna(subset=[col])

        return df_imputed

    async def bin_numeric(self, df: pd.DataFrame, columns: List[str], n_bins: int = 5,
                          strategy: str = 'quantile') -> pd.DataFrame:
        """Bin numeric variables"""
        df_binned = df.copy()

        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            if strategy == 'quantile':
                df_binned[f'{col}_binned'] = pd.qcut(df[col], n_bins, labels=False, duplicates='drop')
            elif strategy == 'uniform':
                df_binned[f'{col}_binned'] = pd.cut(df[col], n_bins, labels=False)

        return df_binned

    async def log_transform(self, df: pd.DataFrame, columns: List[str], offset: float = 1.0) -> pd.DataFrame:
        """Apply log transformation"""
        df_log = df.copy()

        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Add offset to handle zero values
                df_log[f'{col}_log'] = np.log(df[col] + offset)

        return df_log

    async def difference_transform(self, df: pd.DataFrame, columns: List[str], periods: int = 1) -> pd.DataFrame:
        """Apply differencing transformation"""
        df_diff = df.copy()

        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df_diff[f'{col}_diff'] = df[col].diff(periods)

        return df_diff

    async def create_interactions(self, df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
        """Create interaction terms"""
        from itertools import combinations

        df_interact = df.copy()
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

        for r in range(2, min(degree + 1, len(numeric_cols) + 1)):
            for combo in combinations(numeric_cols, r):
                interaction_name = '_x_'.join(combo)
                df_interact[interaction_name] = df[list(combo)].prod(axis=1)

        return df_interact

    async def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers in data"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = {}

        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_indices = df.index[z_scores > 3].tolist()

            elif method == 'isolation':
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1)
                outlier_labels = iso_forest.fit_predict(df[[col]].dropna())
                outlier_indices = df.index[outlier_labels == -1].tolist()

            outliers[col] = {
                'indices': outlier_indices,
                'count': len(outlier_indices),
                'percentage': len(outlier_indices) / len(df) * 100
            }

        return outliers

    async def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()

            # Add additional statistics
            for col in numeric_cols:
                summary['numeric_summary'][col].update({
                    'skewness': float(df[col].skew()),
                    'kurtosis': float(df[col].kurtosis()),
                    'variance': float(df[col].var()),
                    'sem': float(df[col].sem()),
                    'mad': float(df[col].mad()),
                    'range': float(df[col].max() - df[col].min()),
                    'iqr': float(df[col].quantile(0.75) - df[col].quantile(0.25)),
                    'cv': float(df[col].std() / df[col].mean() * 100) if df[col].mean() != 0 else None
                })

        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            summary['categorical_summary'][col] = {
                'unique': df[col].nunique(),
                'top': value_counts.index[0] if len(value_counts) > 0 else None,
                'freq': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'missing': int(df[col].isnull().sum()),
                'value_counts': value_counts.head(10).to_dict()
            }

        return summary

    async def export_data(self, df: pd.DataFrame, format: str, **kwargs) -> bytes:
        """Export dataframe to various formats"""
        output = io.BytesIO()

        if format == 'csv':
            csv_string = df.to_csv(**kwargs)
            output.write(csv_string.encode('utf-8'))

        elif format == 'excel':
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, **kwargs)

        elif format == 'json':
            json_string = df.to_json(**kwargs)
            output.write(json_string.encode('utf-8'))

        elif format == 'parquet':
            df.to_parquet(output, **kwargs)

        elif format == 'html':
            html_string = df.to_html(**kwargs)
            output.write(html_string.encode('utf-8'))

        elif format == 'latex':
            latex_string = df.to_latex(**kwargs)
            output.write(latex_string.encode('utf-8'))

        else:
            raise ValueError(f"Unsupported export format: {format}")

        output.seek(0)
        return output.read()

    async def merge_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame, **merge_kwargs) -> pd.DataFrame:
        """Merge two datasets"""
        return pd.merge(df1, df2, **merge_kwargs)

    async def pivot_data(self, df: pd.DataFrame, **pivot_kwargs) -> pd.DataFrame:
        """Pivot dataframe"""
        return df.pivot_table(**pivot_kwargs)

    async def melt_data(self, df: pd.DataFrame, **melt_kwargs) -> pd.DataFrame:
        """Melt dataframe from wide to long format"""
        return df.melt(**melt_kwargs)
