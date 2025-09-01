from typing import List
from app.core.registry import EngineRegistry
from .supervised import ClassificationEngine, RegressionEngine
from .unsupervised import ClusteringEngine, DimensionalityReductionEngine


async def initialize_ml_engines(registry: EngineRegistry) -> List[str]:
    """
    Pre-register a small set of commonly used ML engines so they're
    available immediately. API endpoints will still create/register
    ad-hoc engines as needed.
    """
    names = []
    defaults = [
        ClassificationEngine("rf"),
        RegressionEngine("linear"),
        ClusteringEngine("kmeans"),
        DimensionalityReductionEngine("pca"),
    ]
    for engine in defaults:
        if not registry.get(engine.name):
            await registry.register(engine)
            names.append(engine.name)
    return names
