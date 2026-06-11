"""
Подключение к LLM — несколько бэкендов из requirements экзамена:
openai, ollama, gigachat, llama-cpp-python, transformers.

Ключ с сайта нужен только для облака (openai / gigachat).
Локально: ollama, llama-cpp-python, transformers — без ключей (модель скачана заранее).
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

    ollama          — без ключа (OLLAMA_MODEL)
    llama_cpp       — без ключа (LLAMA_CPP_MODEL_PATH к .gguf файлу)
    transformers    — без ключа (TRANSFORMERS_MODEL, веса скачаны заранее)
    openai          — OPENAI_API_KEY (+ опционально OPENAI_BASE_URL)
    gigachat        — GIGACHAT_CREDENTIALS (ключ заранее в .env)
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
        print("openai: нужен OPENAI_API_KEY в .env (или используйте provider='ollama')")
        return False
    if cfg.provider == "gigachat" and not cfg.api_key:
        print("gigachat: нужен GIGACHAT_CREDENTIALS в .env (или provider='ollama')")
        return False
    if cfg.provider == "llama_cpp" and not os.path.isfile(cfg.model_path or ""):
        print(f"llama_cpp: файл модели не найден: {cfg.model_path}")
        return False

    print(f"LLM готов: provider={cfg.provider}, model={cfg.model or cfg.model_path}")
    return True


def get_llm_client(config: LLMConfig | None = None, provider: str = "openai"):
    """OpenAI-совместимый клиент (openai / прокси / Ollama через /v1)."""
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


def _chat_openai(messages, config: LLMConfig, temperature, max_tokens) -> str:
    client = get_llm_client(config)
    model = config.model or "gpt-4o-mini"
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature is not None else config.temperature,
        max_tokens=max_tokens or config.max_tokens,
    )
    return r.choices[0].message.content or ""


def _chat_ollama(messages, config: LLMConfig) -> str:
    try:
        import ollama
    except ImportError:
        return _chat_openai(messages, config, None, config.max_tokens)

    model = config.model or "llama3.2"
    r = ollama.chat(model=model, messages=messages)
    return r["message"]["content"]


def _chat_gigachat(messages, config: LLMConfig) -> str:
    from gigachat import GigaChat

    with GigaChat(
        credentials=config.api_key,
        model=config.model,
        verify_ssl_certs=False,
    ) as giga:
        r = giga.chat(messages)
    return r.choices[0].message.content


def _chat_llama_cpp(messages, config: LLMConfig, max_tokens) -> str:
    from llama_cpp import Llama

    llm = Llama(model_path=config.model_path, verbose=False)
    r = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens or config.max_tokens,
        temperature=config.temperature,
    )
    return r["choices"][0]["message"]["content"]


def _chat_transformers(messages, config: LLMConfig, max_tokens) -> str:
    from transformers import pipeline

    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    gen = pipeline(
        "text-generation",
        model=config.model,
        max_new_tokens=max_tokens or config.max_tokens,
    )
    out = gen(text, do_sample=True, temperature=config.temperature)[0]["generated_text"]
    return out[len(text):].strip() if out.startswith(text) else out


def chat(
    messages: list[dict[str, str]],
    provider: str = "ollama",
    config: LLMConfig | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Запрос к LLM. По умолчанию ollama — без ключа.

    Примеры:
        chat([{"role": "user", "content": "Привет"}])
        chat([...], provider="llama_cpp")
        chat([...], provider="transformers")
    """
    config = config or load_llm_config(provider)
    p = config.provider

    if p == "openai":
        return _chat_openai(messages, config, temperature, max_tokens)
    if p == "ollama":
        return _chat_ollama(messages, config)
    if p == "gigachat":
        return _chat_gigachat(messages, config)
    if p == "llama_cpp":
        return _chat_llama_cpp(messages, config, max_tokens)
    if p == "transformers":
        return _chat_transformers(messages, config, max_tokens)
    raise ValueError(f"Unsupported provider: {p}")


def ask(
    prompt: str,
    system: str | None = None,
    provider: str = "ollama",
) -> str:
    """Короткий вопрос одной строкой."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, provider=provider)


if __name__ == "__main__":
    for p in ("ollama", "llama_cpp", "transformers", "openai", "gigachat"):
        print(f"--- {p} ---")
        check_llm_ready(p)
