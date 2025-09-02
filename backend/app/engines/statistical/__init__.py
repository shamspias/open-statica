"""
Statistical Engines for OpenStatica
"""

from typing import List
from app.core.registry import EngineRegistry
from .descriptive import DescriptiveAnalyzer, FrequencyAnalyzer
from .inferential import TTestAnalyzer, ANOVAAnalyzer


async def initialize_statistical_engines(registry: EngineRegistry) -> List[str]:
    """
    Pre-register commonly used statistical engines
    """
    names = []
    engines = [
        DescriptiveAnalyzer(),
        FrequencyAnalyzer(),
        TTestAnalyzer(),
        ANOVAAnalyzer()
    ]

    for engine in engines:
        if not registry.get(engine.name):
            await registry.register(engine)
            names.append(engine.name)

    return names
