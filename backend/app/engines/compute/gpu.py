"""
GPU Compute Backend for OpenStatica
Handles GPU-accelerated computation
"""

from typing import Any, Dict, List, Optional, Callable
import logging
import numpy as np
from app.core.base import ComputeBackend

logger = logging.getLogger(__name__)


class GPUComputeBackend:
    """GPU computation backend using CuPy/PyTorch/TensorFlow"""

    def __init__(self, device_id: Optional[int] = 0):
        self.device_id = device_id
        self.backend_type = ComputeBackend.GPU
        self.backend_lib = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize GPU backend"""
        try:
            # Try CuPy first (best for numerical computing)
            import cupy as cp
            self.backend_lib = cp
            self.backend_name = 'cupy'
            self.device = cp.cuda.Device(self.device_id)
            logger.info(f"Initialized CuPy GPU backend on device {self.device_id}")
        except ImportError:
            try:
                # Try PyTorch
                import torch
                if torch.cuda.is_available():
                    self.backend_lib = torch
                    self.backend_name = 'pytorch'
                    self.device = torch.device(f'cuda:{self.device_id}')
                    logger.info(f"Initialized PyTorch GPU backend on device {self.device_id}")
                else:
                    raise RuntimeError("CUDA not available")
            except (ImportError, RuntimeError):
                try:
                    # Try TensorFlow
                    import tensorflow as tf
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        self.backend_lib = tf
                        self.backend_name = 'tensorflow'
                        tf.config.set_visible_devices(gpus[self.device_id], 'GPU')
                        logger.info(f"Initialized TensorFlow GPU backend")
                    else:
                        raise RuntimeError("No GPU available")
                except (ImportError, RuntimeError) as e:
                    logger.warning(f"No GPU backend available: {e}")
                    # Fallback to CPU with NumPy
                    self.backend_lib = np
                    self.backend_name = 'numpy_cpu'
                    logger.info("Falling back to NumPy CPU backend")

    def to_gpu(self, data: np.ndarray) -> Any:
        """Transfer data to GPU"""
        if self.backend_name == 'cupy':
            return self.backend_lib.asarray(data)
        elif self.backend_name == 'pytorch':
            return self.backend_lib.tensor(data, device=self.device)
        elif self.backend_name == 'tensorflow':
            return self.backend_lib.constant(data)
        else:
            return data

    def to_cpu(self, data: Any) -> np.ndarray:
        """Transfer data from GPU to CPU"""
        if self.backend_name == 'cupy':
            return data.get()
        elif self.backend_name == 'pytorch':
            return data.cpu().numpy()
        elif self.backend_name == 'tensorflow':
            return data.numpy()
        else:
            return np.asarray(data)

    async def execute(self, func: Callable, data: Any, **kwargs) -> Any:
        """Execute function on GPU"""
        # Transfer to GPU
        gpu_data = self.to_gpu(data)

        # Execute operation
        if self.backend_name in ['cupy', 'numpy_cpu']:
            result = func(gpu_data, **kwargs)
        elif self.backend_name == 'pytorch':
            with self.backend_lib.cuda.device(self.device_id):
                result = func(gpu_data, **kwargs)
        elif self.backend_name == 'tensorflow':
            with self.backend_lib.device(f'/GPU:{self.device_id}'):
                result = func(gpu_data, **kwargs)
        else:
            result = func(gpu_data, **kwargs)

        # Transfer back to CPU
        return self.to_cpu(result)

    async def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU-accelerated matrix multiplication"""
        a_gpu = self.to_gpu(a)
        b_gpu = self.to_gpu(b)

        if self.backend_name in ['cupy', 'numpy_cpu']:
            result = self.backend_lib.matmul(a_gpu, b_gpu)
        elif self.backend_name == 'pytorch':
            result = self.backend_lib.matmul(a_gpu, b_gpu)
        elif self.backend_name == 'tensorflow':
            result = self.backend_lib.matmul(a_gpu, b_gpu)
        else:
            result = np.matmul(a, b)

        return self.to_cpu(result)

    async def compute_statistics(self, data: np.ndarray, operations: List[str]) -> Dict[str, float]:
        """GPU-accelerated statistics computation"""
        gpu_data = self.to_gpu(data)
        results = {}

        for op in operations:
            if op == 'mean':
                result = self.backend_lib.mean(gpu_data)
            elif op == 'std':
                result = self.backend_lib.std(gpu_data)
            elif op == 'var':
                result = self.backend_lib.var(gpu_data)
            elif op == 'min':
                result = self.backend_lib.min(gpu_data)
            elif op == 'max':
                result = self.backend_lib.max(gpu_data)
            else:
                continue

            results[op] = float(self.to_cpu(result))

        return results

    def cleanup(self):
        """Cleanup GPU resources"""
        if self.backend_name == 'cupy':
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
        elif self.backend_name == 'pytorch':
            import torch
            torch.cuda.empty_cache()
        elif self.backend_name == 'tensorflow':
            import tensorflow as tf
            tf.keras.backend.clear_session()

    def get_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        info = {
            'type': 'gpu',
            'backend': self.backend_name,
            'device_id': self.device_id
        }

        if self.backend_name == 'cupy':
            import cupy
            mempool = cupy.get_default_memory_pool()
            info.update({
                'device_name': cupy.cuda.Device(self.device_id).name,
                'memory_used': mempool.used_bytes(),
                'memory_total': mempool.total_bytes()
            })
        elif self.backend_name == 'pytorch':
            import torch
            if torch.cuda.is_available():
                info.update({
                    'device_name': torch.cuda.get_device_name(self.device_id),
                    'memory_allocated': torch.cuda.memory_allocated(self.device_id),
                    'memory_reserved': torch.cuda.memory_reserved(self.device_id)
                })

        return info
