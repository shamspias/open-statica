"""
Decorators for OpenStatica
Utility decorators for validation, caching, and performance
"""

import functools
import time
import asyncio
from typing import Callable, Any, Optional
import logging
from fastapi import HTTPException, status
import hashlib
import json

logger = logging.getLogger(__name__)


def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time"""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def cache(ttl: int = 300):
    """Simple in-memory cache decorator with TTL"""

    def decorator(func: Callable) -> Callable:
        cache_data = {}
        cache_time = {}

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key
            key = hashlib.md5(
                f"{args}{kwargs}".encode()
            ).hexdigest()

            # Check cache
            if key in cache_data:
                if time.time() - cache_time[key] < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache_data[key]

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            cache_data[key] = result
            cache_time[key] = time.time()

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create cache key
            key = hashlib.md5(
                f"{args}{kwargs}".encode()
            ).hexdigest()

            # Check cache
            if key in cache_data:
                if time.time() - cache_time[key] < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache_data[key]

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache_data[key] = result
            cache_time[key] = time.time()

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_session(func: Callable) -> Callable:
    """Decorator to validate session exists"""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract session_id from kwargs or first positional arg
        session_id = kwargs.get('session_id')
        if not session_id and len(args) > 1:
            # Assuming first arg is self/request, second is session_id
            session_id = args[1] if isinstance(args[1], str) else None

        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session ID required"
            )

        return await func(*args, **kwargs)

    return wrapper


def require_data(func: Callable) -> Callable:
    """Decorator to ensure data is loaded in session"""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # This would need access to session_manager
        # For now, just pass through
        return await func(*args, **kwargs)

    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry failed operations"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
                        logger.warning(f"Retry {attempt + 1}/{max_attempts} for {func.__name__}")
                    else:
                        logger.error(f"All retries failed for {func.__name__}")

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))
                        logger.warning(f"Retry {attempt + 1}/{max_attempts} for {func.__name__}")
                    else:
                        logger.error(f"All retries failed for {func.__name__}")

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def rate_limit(calls: int = 10, period: int = 60):
    """Simple rate limiting decorator"""

    def decorator(func: Callable) -> Callable:
        call_times = []

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()

            # Remove old calls outside the period
            nonlocal call_times
            call_times = [t for t in call_times if now - t < period]

            # Check rate limit
            if len(call_times) >= calls:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {calls} calls per {period} seconds"
                )

            # Record this call
            call_times.append(now)

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def log_activity(level: str = "INFO"):
    """Decorator to log function calls"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            log_func = getattr(logger, level.lower(), logger.info)
            log_func(f"Calling {func.__name__} with args={args[1:]} kwargs={kwargs}")

            try:
                result = await func(*args, **kwargs)
                log_func(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            log_func = getattr(logger, level.lower(), logger.info)
            log_func(f"Calling {func.__name__} with args={args[1:]} kwargs={kwargs}")

            try:
                result = func(*args, **kwargs)
                log_func(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {e}")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_input(**validators):
    """Decorator to validate input parameters"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    if not validator(value):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid value for parameter: {param_name}"
                        )

            return await func(*args, **kwargs)

        return wrapper

    return decorator
