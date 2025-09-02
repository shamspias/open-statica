"""
Distributed Compute Backend for OpenStatica
Handles distributed computation across multiple nodes
"""

from typing import Any, Dict, List, Optional, Callable
import asyncio
import logging
from app.core.base import ComputeBackend

logger = logging.getLogger(__name__)


class DistributedComputeBackend:
    """Distributed computation backend (Dask/Ray integration)"""

    def __init__(self, cluster_address: Optional[str] = None):
        self.cluster_address = cluster_address
        self.backend_type = ComputeBackend.DISTRIBUTED
        self.client = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize distributed backend"""
        try:
            # Try Dask first
            from dask.distributed import Client
            if self.cluster_address:
                self.client = Client(self.cluster_address)
            else:
                self.client = Client(n_workers=2, threads_per_worker=2)
            self.backend_name = 'dask'
            logger.info("Initialized Dask distributed backend")
        except ImportError:
            try:
                # Fallback to Ray
                import ray
                if not ray.is_initialized():
                    ray.init(address=self.cluster_address)
                self.client = ray
                self.backend_name = 'ray'
                logger.info("Initialized Ray distributed backend")
            except ImportError:
                logger.warning("No distributed backend available, falling back to local")
                from .local import LocalComputeBackend
                self.fallback = LocalComputeBackend()
                self.backend_name = 'local_fallback'

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function on distributed backend"""
        if self.backend_name == 'dask':
            future = self.client.submit(func, *args, **kwargs)
            return await asyncio.wrap_future(future)
        elif self.backend_name == 'ray':
            remote_func = self.client.remote(func)
            return await asyncio.wrap_future(remote_func.remote(*args, **kwargs))
        else:
            return await self.fallback.execute(func, *args, **kwargs)

    async def map(self, func: Callable, iterable: List) -> List:
        """Distributed map operation"""
        if self.backend_name == 'dask':
            futures = self.client.map(func, iterable)
            return await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])
        elif self.backend_name == 'ray':
            remote_func = self.client.remote(func)
            futures = [remote_func.remote(item) for item in iterable]
            return await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])
        else:
            return await self.fallback.map(func, iterable)

    async def scatter(self, data: Any) -> Any:
        """Scatter data across cluster"""
        if self.backend_name == 'dask':
            return await asyncio.wrap_future(self.client.scatter(data))
        elif self.backend_name == 'ray':
            return self.client.put(data)
        else:
            return data

    def cleanup(self):
        """Cleanup distributed resources"""
        if self.backend_name == 'dask' and self.client:
            self.client.close()
        elif self.backend_name == 'ray':
            import ray
            if ray.is_initialized():
                ray.shutdown()
        elif hasattr(self, 'fallback'):
            self.fallback.cleanup()

    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        info = {
            'type': 'distributed',
            'backend': self.backend_name
        }

        if self.backend_name == 'dask' and self.client:
            info.update({
                'workers': len(self.client.nthreads()),
                'dashboard_link': self.client.dashboard_link
            })
        elif self.backend_name == 'ray':
            import ray
            if ray.is_initialized():
                info.update({
                    'nodes': len(ray.nodes()),
                    'resources': ray.cluster_resources()
                })

        return info
