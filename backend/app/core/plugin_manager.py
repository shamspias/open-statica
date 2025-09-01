import os
import importlib.util
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from app.core.base import BasePlugin, BaseEngine

logger = logging.getLogger(__name__)


class PluginManager:
    """Manages loading and lifecycle of plugins"""

    def __init__(self, plugin_path: str):
        self.plugin_path = Path(plugin_path)
        self.plugins: Dict[str, BasePlugin] = {}
        self.enabled_plugins: List[str] = []

    async def load_plugins(self) -> None:
        """Load all plugins from plugin directory"""
        if not self.plugin_path.exists():
            logger.warning(f"Plugin directory {self.plugin_path} does not exist")
            return

        for plugin_dir in self.plugin_path.iterdir():
            if plugin_dir.is_dir() and not plugin_dir.name.startswith('_'):
                try:
                    await self.load_plugin(plugin_dir.name)
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_dir.name}: {e}")

    async def load_plugin(self, plugin_name: str) -> None:
        """Load a specific plugin"""
        plugin_path = self.plugin_path / plugin_name

        # Check for plugin.json metadata
        metadata_path = plugin_path / "plugin.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {"name": plugin_name, "version": "0.0.1"}

        # Load main module
        main_module_path = plugin_path / "__init__.py"
        if not main_module_path.exists():
            main_module_path = plugin_path / "main.py"

        if not main_module_path.exists():
            raise FileNotFoundError(f"No main module found for plugin {plugin_name}")

        # Import plugin module
        spec = importlib.util.spec_from_file_location(
            f"plugins.{plugin_name}",
            main_module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get plugin class
        if hasattr(module, 'Plugin'):
            PluginClass = module.Plugin
        else:
            # Try to find a class that inherits from BasePlugin
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, BasePlugin) and obj != BasePlugin:
                    PluginClass = obj
                    break
            else:
                raise ValueError(f"No Plugin class found in {plugin_name}")

        # Instantiate plugin
        plugin = PluginClass(
            name=metadata.get("name", plugin_name),
            version=metadata.get("version", "0.0.1")
        )

        # Initialize plugin
        await plugin.initialize()

        # Store plugin
        self.plugins[plugin_name] = plugin
        self.enabled_plugins.append(plugin_name)

        logger.info(f"Loaded plugin: {plugin_name} v{plugin.version}")

    async def unload_plugin(self, plugin_name: str) -> None:
        """Unload a plugin"""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            # Cleanup if method exists
            if hasattr(plugin, 'cleanup'):
                await plugin.cleanup()

            del self.plugins[plugin_name]
            self.enabled_plugins.remove(plugin_name)

            logger.info(f"Unloaded plugin: {plugin_name}")

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a specific plugin"""
        return self.plugins.get(plugin_name)

    def get_all_engines(self) -> List[BaseEngine]:
        """Get all engines from all plugins"""
        engines = []
        for plugin in self.plugins.values():
            if plugin.enabled:
                engines.extend(plugin.get_engines())
        return engines

    def get_all_routes(self) -> List[Any]:
        """Get all API routes from all plugins"""
        routes = []
        for plugin in self.plugins.values():
            if plugin.enabled:
                routes.extend(plugin.get_routes())
        return routes

    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded plugins with metadata"""
        return {
            name: {
                "version": plugin.version,
                "enabled": plugin.enabled,
                "engines": [e.name for e in plugin.get_engines()],
                "routes": len(plugin.get_routes())
            }
            for name, plugin in self.plugins.items()
        }

    async def enable_plugin(self, plugin_name: str) -> None:
        """Enable a plugin"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = True
            logger.info(f"Enabled plugin: {plugin_name}")

    async def disable_plugin(self, plugin_name: str) -> None:
        """Disable a plugin"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = False
            logger.info(f"Disabled plugin: {plugin_name}")
