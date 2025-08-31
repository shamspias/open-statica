from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from pydantic import BaseModel
import asyncio
import numpy as np
import pandas as pd
from enum import Enum

T = TypeVar('T')


class EngineType(str, Enum):
    """Types of computation engines"""
    STATISTICAL = "statistical"
    ML = "ml"
    DEEP_LEARNING = "deep_learning"
    VISUALIZATION = "visualization"
    PREPROCESSING = "preprocessing"


class ComputeBackend(str, Enum):
    """Computation backends"""
    LOCAL = "local"
    DISTRIBUTED = "distributed"
    GPU = "gpu"
    CLOUD = "cloud"


class BaseEngine(ABC):
    """Base class for all computation engines"""

    def __init__(self, name: str, engine_type: EngineType):
        self.name = name
        self.engine_type = engine_type
        self.backend = ComputeBackend.LOCAL
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the engine"""
        if not self._initialized:
            await self._setup()
            self._initialized = True

    @abstractmethod
    async def _setup(self) -> None:
        """Setup engine resources"""
        pass

    @abstractmethod
    async def execute(self, data: Any, params: Dict[str, Any]) -> Any:
        """Execute computation"""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass

    def validate_input(self, data: Any) -> bool:
        """Validate input data"""
        return True


class BaseAnalyzer(BaseEngine):
    """Base class for statistical analyzers"""

    def __init__(self, name: str):
        super().__init__(name, EngineType.STATISTICAL)

    async def _setup(self) -> None:
        """Setup analyzer"""
        pass

    def prepare_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Prepare data for analysis"""
        return data[columns].dropna()

    def calculate_effect_size(self, *args, **kwargs) -> float:
        """Calculate effect size for the analysis"""
        raise NotImplementedError


class BaseMLModel(BaseEngine):
    """Base class for ML models"""

    def __init__(self, name: str, model_type: str):
        super().__init__(name, EngineType.ML)
        self.model_type = model_type
        self.model = None
        self.is_trained = False

    @abstractmethod
    async def train(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        pass

    @abstractmethod
    async def predict(self, X: Any, **kwargs) -> Any:
        """Make predictions"""
        pass

    @abstractmethod
    async def evaluate(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """Evaluate model performance"""
        pass

    async def save(self, path: str) -> None:
        """Save model to disk"""
        pass

    async def load(self, path: str) -> None:
        """Load model from disk"""
        pass


class BasePlugin(ABC):
    """Base class for plugins"""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.enabled = True

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize plugin"""
        pass

    @abstractmethod
    def get_engines(self) -> List[BaseEngine]:
        """Get engines provided by this plugin"""
        pass

    @abstractmethod
    def get_routes(self) -> List[Any]:
        """Get API routes provided by this plugin"""
        pass


class Result(BaseModel, Generic[T]):
    """Generic result wrapper"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

    @classmethod
    def ok(cls, data: T, **metadata) -> "Result[T]":
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata) -> "Result[T]":
        return cls(success=False, error=error, metadata=metadata)


class DataInfo(BaseModel):
    """Information about uploaded data"""
    rows: int
    columns: int
    column_names: List[str]
    column_types: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    missing_values: Dict[str, int]
    memory_usage: float  # in MB


class AnalysisRequest(BaseModel):
    """Base request for analysis"""
    session_id: str
    columns: List[str]
    options: Dict[str, Any] = {}
    backend: ComputeBackend = ComputeBackend.LOCAL


class AnalysisResult(BaseModel):
    """Base result for analysis"""
    test_name: str
    results: Dict[str, Any]
    visualizations: Optional[List[Dict[str, Any]]] = None
    interpretation: Optional[str] = None
    execution_time: float
