from typing import Dict, List, Optional, Type
from app.core.base import BaseEngine, EngineType
import logging

logger = logging.getLogger(__name__)


class EngineRegistry:
    """Registry for all computation engines"""

    def __init__(self):
        self._engines: Dict[str, BaseEngine] = {}
        self._engine_types: Dict[EngineType, List[str]] = {
            engine_type: [] for engine_type in EngineType
        }

    async def register(self, engine: BaseEngine) -> None:
        """Register a new engine"""
        if engine.name in self._engines:
            logger.warning(f"Engine {engine.name} already registered, overwriting")

        await engine.initialize()
        self._engines[engine.name] = engine
        self._engine_types[engine.engine_type].append(engine.name)
        logger.info(f"Registered engine: {engine.name} ({engine.engine_type})")

    def get(self, name: str) -> Optional[BaseEngine]:
        """Get engine by name"""
        return self._engines.get(name)

    def get_by_type(self, engine_type: EngineType) -> List[BaseEngine]:
        """Get all engines of a specific type"""
        engine_names = self._engine_types.get(engine_type, [])
        return [self._engines[name] for name in engine_names]

    def list_engines(self) -> Dict[str, List[str]]:
        """List all registered engines by type"""
        return {
            engine_type.value: names
            for engine_type, names in self._engine_types.items()
            if names
        }

    async def cleanup(self) -> None:
        """Cleanup all engines"""
        for engine in self._engines.values():
            await engine.cleanup()
