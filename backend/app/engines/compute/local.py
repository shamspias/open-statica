"""
Local Compute Backend for OpenStatica
Handles computation on local machine resources
"""

import asyncio
import concurrent.futures
import multiprocessing as mp
from typing import Any, Dict, List, Callable, Optional
import numpy as np
import pandas as pd
from app.core.base import ComputeBackend
import logging

logger = logging.getLogger(__name__)


class LocalComputeBackend:
    """Local computation backend using multiprocessing"""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        self.backend_type = ComputeBackend.LOCAL

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)

    async def map(self, func: Callable, iterable: List) -> List:
        """Map function over iterable in parallel"""
        loop = asyncio.get_event_loop()
        futures = [loop.run_in_executor(self.executor, func, item) for item in iterable]
        return await asyncio.gather(*futures)

    async def reduce(self, func: Callable, iterable: List, initializer=None) -> Any:
        """Reduce operation"""
        from functools import reduce
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: reduce(func, iterable, initializer) if initializer else reduce(func, iterable)
        )

    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)

    async def compute_statistics(self, data: np.ndarray, operations: List[str]) -> Dict[str, Any]:
        """Compute multiple statistics in parallel"""
        stats_funcs = {
            'mean': np.mean,
            'std': np.std,
            'median': np.median,
            'min': np.min,
            'max': np.max,
            'var': np.var,
            'skew': lambda x: float(pd.Series(x).skew()),
            'kurtosis': lambda x: float(pd.Series(x).kurtosis())
        }

        results = {}
        tasks = []

        for op in operations:
            if op in stats_funcs:
                tasks.append(self.execute(stats_funcs[op], data))

        computed = await asyncio.gather(*tasks)

        for op, result in zip(operations, computed):
            results[op] = float(result) if not isinstance(result, (list, np.ndarray)) else result

        return results

    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            'type': 'local',
            'max_workers': self.max_workers,
            'cpu_count': mp.cpu_count(),
            'backend': 'ProcessPoolExecutor'
        }
