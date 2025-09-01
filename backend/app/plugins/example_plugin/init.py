from app.core.base import BasePlugin, BaseEngine
from typing import List, Any


class ExampleEngine(BaseEngine):
    """Example custom engine"""

    def __init__(self):
        super().__init__("example_engine", "custom")

    async def _setup(self):
        """Setup engine"""
        pass

    async def execute(self, data: Any, params: dict) -> Any:
        """Execute custom logic"""
        return {"message": "Example engine executed", "data": data}


class Plugin(BasePlugin):
    """Example plugin for OpenStatica"""

    def __init__(self, name: str, version: str):
        super().__init__(name, version)
        self.engines = []

    async def initialize(self):
        """Initialize plugin"""
        # Create and setup engines
        example_engine = ExampleEngine()
        await example_engine.initialize()
        self.engines.append(example_engine)

    def get_engines(self) -> List[BaseEngine]:
        """Get engines provided by this plugin"""
        return self.engines

    def get_routes(self) -> List[Any]:
        """Get API routes provided by this plugin"""
        # Could return FastAPI routers here
        return []
