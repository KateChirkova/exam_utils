"""
Подключение к LLM: конфиг из .env, клиент, вызов API.
Ключи берутся из файла .env (см. .env.example) — не из репозитория.
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
    """
    Загрузить настройки из .env.

    OpenAI / GigaChat / др.: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
    Ollama (локально, без ключа): OLLAMA_BASE_URL, OLLAMA_MODEL
    """
    provider = provider.lower()
    if provider == "openai":
        return LLMConfig(
            provider="openai",
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    if provider in ("ollama", "local"):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        return LLMConfig(
            provider="ollama",
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=base_url,
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


def check_llm_ready(provider: str = "openai") -> bool:
    """
    Проверить, что настройки LLM заданы (ключ в .env или Ollama).
    Не делает запрос к API.
    """
    cfg = load_llm_config(provider)
    if cfg.provider == "openai" and not cfg.api_key:
        print(
            "Ключ не найден. Скопируйте .env.example → .env и укажите OPENAI_API_KEY.\n"
            "Ключ получают на сайте провайдера (OpenAI, GigaChat и т.д.)."
        )
        return False
    print(f"LLM готов: provider={cfg.provider}, model={cfg.model}")
    return True


def chat(
    messages: list[dict[str, str]],
    provider: str = "openai",
    config: LLMConfig | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Отправить сообщения в LLM, вернуть текст ответа.

    Пример:
        chat([{"role": "user", "content": "Привет"}])
        chat([{"role": "user", "content": query}], provider="ollama")
    """
    config = config or load_llm_config(provider)
    client = get_llm_client(config, provider)
    model = config.model or ("gpt-4o-mini" if config.provider == "openai" else "llama3.2")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature is not None else config.temperature,
        max_tokens=max_tokens or config.max_tokens,
    )
    return response.choices[0].message.content or ""


def ask(
    prompt: str,
    system: str | None = None,
    provider: str = "openai",
) -> str:
    """
    Короткий вызов: один вопрос пользователя (+ опционально system).

    Пример:
        answer = ask("Объясни RAG в двух предложениях")
        answer = ask(query, system="Отвечай только по контексту Wiki.")
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, provider=provider)


if __name__ == "__main__":
    check_llm_ready()
    check_llm_ready("ollama")
