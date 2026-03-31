from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


@dataclass
class LLMConfig:

    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: str = "http://localhost:8000/v1"

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None

    # provider-specific / vLLM sampling params etc.
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.base_url:
            self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")


def create_llm(config: LLMConfig) -> BaseChatModel:
    """
    Create LLM from config (OpenAI API compatible or local VLLM).

    Args:
        config: LLM config

    Returns:
        LLM instance (ChatOpenAI, OpenAI API compatible)
    """
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
    
    model_kwargs: Dict[str, Any] = {}
    default_extra_body: Dict[str, Any] = {}

    if config.extra_params:
        generation_params = {
            "repetition_penalty",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "logprobs",
            "logit_bias",
        }

        for key, value in config.extra_params.items():
            if key in generation_params:
                default_extra_body[key] = value
            else:
                model_kwargs[key] = value

        if model_kwargs:
            llm_kwargs["model_kwargs"] = model_kwargs
    
    llm = ChatOpenAI(**llm_kwargs)
    
    if default_extra_body:
        original_create = llm.client.create

        def create_with_extra_body(*args, **kwargs):
            extra_body = kwargs.get("extra_body")
            if isinstance(extra_body, dict) and extra_body:
                kwargs["extra_body"] = {**default_extra_body, **extra_body}
            elif extra_body:
                kwargs["extra_body"] = {**default_extra_body, **extra_body}
            else:
                kwargs["extra_body"] = default_extra_body.copy()
            return original_create(*args, **kwargs)

        llm.client.create = create_with_extra_body

        if getattr(llm, "async_client", None):
            original_async_create = llm.async_client.create

            async def async_create_with_extra_body(*args, **kwargs):
                extra_body = kwargs.get("extra_body")
                if isinstance(extra_body, dict) and extra_body:
                    kwargs["extra_body"] = {**default_extra_body, **extra_body}
                elif extra_body:
                    kwargs["extra_body"] = {**default_extra_body, **extra_body}
                else:
                    kwargs["extra_body"] = default_extra_body.copy()
                return await original_async_create(*args, **kwargs)

            llm.async_client.create = async_create_with_extra_body
    
    return llm

