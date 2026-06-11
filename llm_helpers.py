
from __future__ import annotations

import os
from dataclasses import dataclass

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
    """Настройки из .env для ollama / lm_studio / llama_cpp / transformers."""
    provider = provider.lower()
    if provider in ("ollama", "local"):
        return LLMConfig(
            provider="ollama",
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        )
    if provider in ("lm_studio", "lmstudio", "lm-studio"):
        base = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        return LLMConfig(
            provider="lm_studio",
            model=os.getenv("LM_STUDIO_MODEL", "local-model"),
            api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
            base_url=base,
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
        f"Unknown provider: {provider}. Use: ollama, lm_studio, llama_cpp, transformers"
    )


def check_llm_ready(provider: str = "ollama") -> bool:
    """Проверка настроек (без запроса к API)."""
    cfg = load_llm_config(provider)

    if cfg.provider == "llama_cpp" and not os.path.isfile(cfg.model_path or ""):
        print(f"llama_cpp: файл модели не найден: {cfg.model_path}")
        return False

    print(f"LLM готов: provider={cfg.provider}, model={cfg.model or cfg.model_path}")
    return True
