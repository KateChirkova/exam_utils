"""
Подключение к LLM — только конфиг и клиент.
Вызовы API делаете сами через get_llm_client().
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 1024


def load_llm_config(provider: str = "openai") -> LLMConfig:
    """Загрузить настройки из .env."""
    provider = provider.lower()
    if provider == "openai":
        return LLMConfig(
            provider="openai",
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    if provider in ("ollama", "local"):
        return LLMConfig(
            provider="ollama",
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'ollama'.")


def get_llm_client(config: LLMConfig | None = None, provider: str = "openai"):
    """
    Создать OpenAI-совместимый клиент.

    Пример вызова (в вашем коде):
        client = get_llm_client()
        cfg = load_llm_config()
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": "..."}],
        )
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("pip install openai") from exc

    config = config or load_llm_config(provider)
    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


if __name__ == "__main__":
    cfg = load_llm_config()
    print(f"provider={cfg.provider}, model={cfg.model}, key_set={bool(cfg.api_key)}")
