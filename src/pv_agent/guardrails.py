"""Validation for the free-form SQL tool. Defence in depth:

1. Text checks: single statement, must start with SELECT/WITH, no write/DDL keywords.
2. Row cap enforced by wrapping the query in a LIMIT.
3. Runtime: read-only connection + SQLite authorizer + wall-clock timeout (see db.py).
"""
from __future__ import annotations

import re

_BLOCKED = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex|truncate)\b",
    re.IGNORECASE,
)


class SQLRejected(ValueError):
    pass


def validate_sql(sql: str, row_limit: int = 200) -> str:
    """Return a safe, row-limited version of `sql` or raise SQLRejected."""
    if not sql or not sql.strip():
        raise SQLRejected("Empty query.")
    cleaned = sql.strip().rstrip(";").strip()
    if ";" in cleaned:
        raise SQLRejected("Only a single statement is allowed.")
    if "--" in cleaned or "/*" in cleaned:
        raise SQLRejected("SQL comments are not allowed.")
    if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
        raise SQLRejected("Only SELECT queries are allowed.")
    hit = _BLOCKED.search(cleaned)
    if hit:
        raise SQLRejected(f"Keyword '{hit.group(0).upper()}' is not allowed (read-only access).")
    return f"SELECT * FROM ({cleaned}) LIMIT {int(row_limit)}"
