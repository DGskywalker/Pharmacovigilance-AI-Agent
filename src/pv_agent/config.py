"""Central configuration, driven by environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    db_path: Path
    model: str
    max_steps: int
    max_tokens: int
    sql_row_limit: int = 200
    sql_timeout_s: float = 5.0
    tool_result_char_limit: int = 12_000


def get_settings() -> Settings:
    return Settings(
        db_path=Path(os.getenv("PV_DB_PATH", ROOT / "data" / "faers.db")),
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        max_steps=int(os.getenv("PV_MAX_STEPS", "8")),
        max_tokens=int(os.getenv("PV_MAX_TOKENS", "2048")),
    )
