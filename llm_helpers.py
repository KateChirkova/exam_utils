"""
Подключение к LLM — только инфраструктура: конфиг, клиент, отправка сообщения.
Без готовых промптов и без автоматических вызовов из других модулей.
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
    """Создать OpenAI-совместимый клиент."""
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


def chat(
    messages: list[dict[str, str]],
    config: LLMConfig | None = None,
    provider: str = "openai",
) -> str:
    """
    Отправить список сообщений в LLM, вернуть текст ответа.
    Промпт формируете сами — здесь только транспорт.

    Пример:
        chat([
            {"role": "user", "content": "Объясни overfitting в двух предложениях"},
        ])
    """
    config = config or load_llm_config(provider)
    client = get_llm_client(config, provider)
    response = client.chat.completions.create(
        model=config.model or ("gpt-4o-mini" if config.provider == "openai" else "llama3.2"),
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return response.choices[0].message.content or ""


def chat_stream(
    messages: list[dict[str, str]],
    config: LLMConfig | None = None,
    provider: str = "openai",
):
    """
    Потоковый ответ (генератор фрагментов текста).

    Пример:
        for chunk in chat_stream([{"role": "user", "content": "Привет"}]):
            print(chunk, end="", flush=True)
    """
    config = config or load_llm_config(provider)
    client = get_llm_client(config, provider)
    stream = client.chat.completions.create(
        model=config.model or ("gpt-4o-mini" if config.provider == "openai" else "llama3.2"),
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta


def check_connection(provider: str = "openai") -> bool:
    """Проверить, доступен ли LLM (короткий ping)."""
    try:
        reply = chat([{"role": "user", "content": "OK"}], provider=provider)
        print(f"LLM OK: {reply[:40]}")
        return True
    except Exception as exc:
        print(f"LLM failed: {exc}")
        return False


if __name__ == "__main__":
    cfg = load_llm_config()
    print(f"provider={cfg.provider}, model={cfg.model}, key_set={bool(cfg.api_key)}")
