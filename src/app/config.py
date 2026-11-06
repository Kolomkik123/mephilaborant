"""Конфигурация из переменных окружения и .env.

Один источник правды для всех модулей. Значения по умолчанию подобраны так,
чтобы система поднималась без ключей и без интернета: LLM_PROVIDER=mock.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Минимальный разбор .env без внешней зависимости."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "mock"))
    openrouter_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY", ""))
    # Имена внешних моделей задаются в .env. Доступность выбранной модели
    # зависит от провайдера; это не модули других участников команды.
    model: str = field(
        default_factory=lambda: _env("LLM_MODEL", "qwen/qwen3-235b-a22b-2507")
    )
    vision_model: str = field(
        default_factory=lambda: _env("LLM_VISION_MODEL", "qwen/qwen3-vl-235b-a22b-instruct")
    )
    local_base_url: str = field(
        default_factory=lambda: _env("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1")
    )
    local_model: str = field(default_factory=lambda: _env("LOCAL_MODEL", "qwen3"))

    max_agent_steps: int = field(default_factory=lambda: _env_int("MAX_AGENT_STEPS", 8))
    request_timeout_s: int = field(default_factory=lambda: _env_int("REQUEST_TIMEOUT_S", 60))
    price_limit_rub: float = field(default_factory=lambda: _env_float("PRICE_LIMIT_RUB", 3.0))
    usd_rub: float = field(default_factory=lambda: _env_float("USD_RUB", 95.0))

    index_backend: str = field(default_factory=lambda: _env("INDEX_BACKEND", "bm25"))
    index_dir: Path = field(default_factory=lambda: ROOT / _env("INDEX_DIR", "data/index"))
    methodics_dir: Path = field(
        default_factory=lambda: ROOT / _env("METHODICS_DIR", "data/methodics")
    )
    artifacts_dir: Path = field(default_factory=lambda: ROOT / "artifacts")

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"


settings = Settings()
