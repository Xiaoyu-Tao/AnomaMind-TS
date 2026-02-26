from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
import os
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM config - supports OpenAI API compatible services or local VLLM"""
    # Basic config
    provider: str = "local"  # "local" or "custom"
    model_name: str = "gpt-4"  # model name
    api_key: Optional[str] = None  # API key (optional)
    base_url: str = "http://localhost:8000/v1"  # API endpoint URL (required)

    # Model params
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    
    # Other config
    extra_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-init processing"""
        # If API key not set, try env vars
        if self.api_key is None:
            self.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # 如果base_url未设置，根据provider设置默认值
        if self.base_url is None or self.base_url == "":
            if self.provider == "local":
                # 本地VLLM默认地址
                self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
            elif self.provider == "custom":
                # custom必须提供base_url
                self.base_url = os.getenv("LLM_BASE_URL")
                if not self.base_url:
                    raise ValueError("custom provider must provide base_url")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "extra_params": self.extra_params
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """Create config from dict"""
        return cls(**config_dict)


class LLMConfigManager:
    """LLM config manager"""
    
    def __init__(self):
        self.configs: Dict[str, LLMConfig] = {}
        self.default_config: Optional[LLMConfig] = None
    
    def register_config(self, name: str, config: LLMConfig):
        """Register a config"""
        self.configs[name] = config
    
    def get_config(self, name: Optional[str] = None) -> LLMConfig:
        """Get config"""
        if name and name in self.configs:
            return self.configs[name]
        elif self.default_config:
            return self.default_config
        else:
            # Return default config
            return LLMConfig()
    
    def set_default_config(self, config: LLMConfig):
        """设置默认配置"""
        self.default_config = config


def retry_llm_call(
    max_retries: int = 3,
    retry_delay: float = 2.0,
    backoff_factor: float = 1.5,
    retryable_exceptions: tuple = None
):
    """
    Retry decorator for LLM calls.

    Args:
        max_retries: Max retries (default 3)
        retry_delay: Initial retry delay in seconds (default 2.0)
        backoff_factor: Backoff factor (default 1.5)
        retryable_exceptions: Tuple of retryable exception types

    Returns:
        Decorator function
    """
    if retryable_exceptions is None:
        # Default retryable exceptions
        try:
            from openai import APIConnectionError, APITimeoutError, InternalServerError
            from httpx import ConnectError, TimeoutException
            retryable_exceptions = (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                ConnectError,
                TimeoutException,
                ConnectionError,
                OSError,
            )
        except ImportError:
            # Fallback if import fails
            retryable_exceptions = (ConnectionError, OSError, TimeoutError)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = retry_delay
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"LLM call failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}"
                        )
                        logger.info(f"Retrying in {current_delay:.2f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"LLM call failed after {max_retries} retries: {type(e).__name__}: {str(e)}"
                        )
                        raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = retry_delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"LLM call failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}"
                        )
                        logger.info(f"Retrying in {current_delay:.2f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"LLM call failed after {max_retries} retries: {type(e).__name__}: {str(e)}"
                        )
                        raise
        
        # Return appropriate wrapper for sync/async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def create_llm(config: LLMConfig) -> BaseChatModel:
    """
    Create LLM from config (OpenAI API compatible or local VLLM).

    Args:
        config: LLM config

    Returns:
        LLM instance (ChatOpenAI, OpenAI API compatible)
    """
    if config.provider not in ["local", "custom"]:
        raise ValueError(f"Unsupported provider: {config.provider}, use 'local' or 'custom'")

    # Both local and custom use ChatOpenAI (OpenAI API compatible)
    llm_kwargs = {
        "model": config.model_name,
        "temperature": config.temperature,
        "base_url": config.base_url,  # Required: API endpoint URL
    }
    
    # API key (optional for some services)
    if config.api_key:
        llm_kwargs["api_key"] = config.api_key
    
    # Optional params
    if config.max_tokens:
        llm_kwargs["max_tokens"] = config.max_tokens
    
    if config.timeout:
        llm_kwargs["timeout"] = config.timeout
    
    # Extra params: vLLM params need special handling
    # LangChain ChatOpenAI may not support all vLLM params
    model_kwargs = {}
    default_params = {}
    
    if config.extra_params:
        # vLLM generation params (sent in request body)
        generation_params = [
            "repetition_penalty",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "logprobs",
            "logit_bias",
        ]
        
        # Split generation params and others
        for key, value in config.extra_params.items():
            if key in generation_params:
                # vLLM params: passed via default_params (auto-added to each call)
                default_params[key] = value
            else:
                # Other params go to model_kwargs
                model_kwargs[key] = value
        
        # Set model_kwargs if any
        if model_kwargs:
            llm_kwargs["model_kwargs"] = model_kwargs
    
    llm = ChatOpenAI(**llm_kwargs)
    
    # vLLM params (e.g. repetition_penalty) passed via extra_body
    if default_params:
        # Save original create method
        original_create = llm.client.create
        
        def create_with_vllm_params(*args, **kwargs):
            """Wrap create to add vLLM params via extra_body"""
            if 'extra_body' in kwargs and kwargs['extra_body']:
                # Merge with existing extra_body
                if isinstance(kwargs['extra_body'], dict):
                    kwargs['extra_body'].update(default_params)
                else:
                    kwargs['extra_body'] = {**default_params, **kwargs['extra_body']}
            else:
                # Create new extra_body if needed
                kwargs['extra_body'] = default_params.copy()
            
            return original_create(*args, **kwargs)
        
        # Replace method
        llm.client.create = create_with_vllm_params
        
        # Handle async client too
        if hasattr(llm, 'async_client') and llm.async_client:
            original_async_create = llm.async_client.create
            
            async def async_create_with_vllm_params(*args, **kwargs):
                """Wrap async create to add vLLM params via extra_body"""
                if 'extra_body' in kwargs and kwargs['extra_body']:
                    if isinstance(kwargs['extra_body'], dict):
                        kwargs['extra_body'].update(default_params)
                    else:
                        kwargs['extra_body'] = {**default_params, **kwargs['extra_body']}
                else:
                    kwargs['extra_body'] = default_params.copy()
                
                return await original_async_create(*args, **kwargs)
            
            llm.async_client.create = async_create_with_vllm_params
    
    return llm


# Global config manager instance
_global_config_manager = LLMConfigManager()


def get_global_config_manager() -> LLMConfigManager:
    """Get global config manager"""
    return _global_config_manager


def create_llm_from_config(
    provider: str = "local",
    model_name: str = "gpt-4",
    api_key: Optional[str] = None,
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    **kwargs
) -> BaseChatModel:
    """
    Convenience function to create LLM (OpenAI API compatible or local VLLM).

    Args:
        provider: "local" or "custom"
        model_name: model name
        api_key: API key (optional)
        base_url: API URL (default: http://localhost:8000/v1)
        temperature: temperature
        max_tokens: max tokens
        timeout: timeout
        **kwargs: other params

    Returns:
        LLM instance
    """
    config = LLMConfig(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_params=kwargs if kwargs else None
    )
    return create_llm(config)

