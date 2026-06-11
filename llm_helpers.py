"""
Подключение к LLM — конфиг и клиент.
Вызов API (ollama, gigachat, llama_cpp, transformers) пишете в своём коде.

Ключ с сайта нужен только для openai / gigachat.
Локально: ollama, llama-cpp-python, transformers — без ключей.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    model_path: str | None = None
    temperature: float = 0.3
    max_tokens: int = 1024


def load_llm_config(provider: str = "ollama") -> LLMConfig:
    """
    Настройки из .env.

    ollama       — OLLAMA_MODEL
    llama_cpp    — LLAMA_CPP_MODEL_PATH
    transformers — TRANSFORMERS_MODEL
    openai       — OPENAI_API_KEY, OPENAI_MODEL
    gigachat     — GIGACHAT_CREDENTIALS, GIGACHAT_MODEL
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
        return LLMConfig(
            provider="ollama",
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        )
    if provider == "gigachat":
        return LLMConfig(
            provider="gigachat",
            model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
            api_key=os.getenv("GIGACHAT_CREDENTIALS"),
        )
    if provider in ("llama_cpp", "llama-cpp"):
        return LLMConfig(
            provider="llama_cpp",
            model_path=os.getenv("LLAMA_CPP_MODEL_PATH", "./models/model.gguf"),
        )
    if provider == "transformers":
        return LLMConfig(
            provider="transformers",
            model=os.getenv("TRANSFORMERS_MODEL", "ai-forever/rugpt3small_based_on_gpt2"),
        )
    raise ValueError(
        f"Unknown provider: {provider}. "
        "Use: ollama, llama_cpp, transformers, openai, gigachat"
    )


def check_llm_ready(provider: str = "ollama") -> bool:
    """Проверка настроек (без запроса к API)."""
    cfg = load_llm_config(provider)

    if cfg.provider == "openai" and not cfg.api_key:
        print("openai: нужен OPENAI_API_KEY в .env")
        return False
    if cfg.provider == "gigachat" and not cfg.api_key:
        print("gigachat: нужен GIGACHAT_CREDENTIALS в .env")
        return False
    if cfg.provider == "llama_cpp" and not os.path.isfile(cfg.model_path or ""):
        print(f"llama_cpp: файл модели не найден: {cfg.model_path}")
        return False

    print(f"LLM готов: provider={cfg.provider}, model={cfg.model or cfg.model_path}")
    return True


def get_llm_client(config: LLMConfig | None = None, provider: str = "openai"):
    """
    OpenAI-совместимый клиент (openai / Ollama через /v1).

    Пример:
        cfg = load_llm_config("openai")
        client = get_llm_client(cfg)
        r = client.chat.completions.create(model=cfg.model, messages=[...])
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
    elif config.provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        kwargs["base_url"] = base if base.endswith("/v1") else base + "/v1"
    return OpenAI(**kwargs)


if __name__ == "__main__":
    for p in ("ollama", "llama_cpp", "transformers", "openai", "gigachat"):
        print(f"--- {p} ---")
        check_llm_ready(p)
