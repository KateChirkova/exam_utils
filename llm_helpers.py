"""
Подключение к LLM для подготовки к экзамену.
Не генерирует готовые ответы на билеты — только шаблоны промптов и обёртки API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "Ты помощник при подготовке к экзамену. "
    "Объясняй концепции и указывай на ошибки в коде, но не пиши полное готовое решение задачи. "
    "Отвечай на русском языке, кратко и по существу."
)


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 1024


def load_llm_config(provider: str = "openai") -> LLMConfig:
    """Загрузить конфиг из переменных окружения (.env)."""
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
    Создать клиент OpenAI-compatible API.
    pip install openai; задайте OPENAI_API_KEY в .env
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install openai: pip install openai") from exc

    config = config or load_llm_config(provider)
    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def ask_llm(
    prompt: str,
    system: str = DEFAULT_SYSTEM_PROMPT,
    config: LLMConfig | None = None,
    provider: str = "openai",
) -> str:
    """Один запрос к LLM. Возвращает текст ответа."""
    config = config or load_llm_config(provider)
    client = get_llm_client(config, provider)

    if config.provider == "ollama":
        response = client.chat.completions.create(
            model=config.model or "llama3.2",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    else:
        response = client.chat.completions.create(
            model=config.model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    return response.choices[0].message.content or ""


def build_concept_prompt(topic: str) -> str:
    """Промпт для объяснения теории (не ответ на тестовый билет)."""
    return (
        f"Объясни экзаменационную тему «{topic}»: определение, 2–3 ключевых факта, "
        f"типичная ошибка студента. Без готовых ответов на конкретные тестовые вопросы."
    )


def build_code_review_prompt(code: str, language: str = "java") -> str:
    """Промпт для ревью черновика кода (LRU, REST, log parser и т.д.)."""
    return (
        f"Проверь черновик на {language}. Укажи архитектурные замечания, "
        f"потенциальные баги и что проверить в unit-тестах. Не переписывай решение целиком.\n\n"
        f"```{language}\n{code}\n```"
    )


def build_regex_help_prompt(log_sample: str) -> str:
    """Помощь с regex для server.log — только подсказки по группам Pattern."""
    return (
        "Помоги составить Java Pattern для строк лога формата:\n"
        "[2024-03-15 10:23:45] LEVEL ServiceName - message\n\n"
        "Примеры строк:\n"
        f"{log_sample}\n\n"
        "Дай только структуру групп (timestamp, level, service, message) и типичные ошибки Matcher. "
        "Не пиши полный Java-класс парсера."
    )


def explain_concept(topic: str, provider: str = "openai") -> str:
    """Объяснение темы через LLM (требует API key)."""
    return ask_llm(build_concept_prompt(topic), provider=provider)


def review_code_snippet(code: str, language: str = "java", provider: str = "openai") -> str:
    """Ревью фрагмента кода через LLM."""
    return ask_llm(build_code_review_prompt(code, language), provider=provider)


def check_llm_connection(provider: str = "openai") -> bool:
    """Проверка доступности LLM (короткий ping-запрос)."""
    try:
        answer = ask_llm("Ответь одним словом: OK", provider=provider)
        print(f"LLM connection OK. Response: {answer[:50]}")
        return True
    except Exception as exc:
        print(f"LLM connection failed: {exc}")
        print("Set OPENAI_API_KEY in .env or run Ollama locally.")
        return False


if __name__ == "__main__":
    print("LLM helpers loaded.")
    print("Example prompts:")
    print(build_concept_prompt("volatile в Java"))
    print("---")
    cfg = load_llm_config()
    print(f"Config: provider={cfg.provider}, model={cfg.model}, key_set={bool(cfg.api_key)}")
