"""SQLite schema, builders and a hardened read-only connection."""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
DROP TABLE IF EXISTS demo;  DROP TABLE IF EXISTS drug;  DROP TABLE IF EXISTS reac;
DROP TABLE IF EXISTS outc;  DROP TABLE IF EXISTS meta;  DROP TABLE IF EXISTS reac_counts;
CREATE TABLE demo (
    primaryid TEXT PRIMARY KEY, caseid TEXT, age_years REAL, age_group TEXT, sex TEXT,
    weight_kg REAL, country TEXT, report_date TEXT, report_year INTEGER, report_quarter TEXT
);
CREATE TABLE drug (primaryid TEXT, drug_seq INTEGER, role_cod TEXT, drugname TEXT, prod_ai TEXT);
CREATE TABLE reac (primaryid TEXT, pt TEXT);
CREATE TABLE outc (primaryid TEXT, outc_cod TEXT);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""

INDEXES = """
CREATE INDEX idx_drug_name ON drug(drugname);
CREATE INDEX idx_drug_ai   ON drug(prod_ai);
CREATE INDEX idx_drug_pid  ON drug(primaryid);
CREATE INDEX idx_reac_pid  ON reac(primaryid);
CREATE INDEX idx_reac_pt   ON reac(pt);
CREATE INDEX idx_outc_pid  ON outc(primaryid);
"""

OUTCOME_LABELS = {
    "DE": "Death",
    "LT": "Life-threatening",
    "HO": "Hospitalization",
    "DS": "Disability",
    "CA": "Congenital anomaly",
    "RI": "Required intervention",
    "OT": "Other serious",
}


def age_group(age_years: pd.Series) -> pd.Series:
    bins = [-0.001, 17.999, 44.999, 64.999, 74.999, 200]
    labels = ["0-17", "18-44", "45-64", "65-74", "75+"]
    return pd.cut(age_years, bins=bins, labels=labels).astype("object").fillna("Unknown")


def write_database(
    db_path: str | Path,
    demo: pd.DataFrame,
    drug: pd.DataFrame,
    reac: pd.DataFrame,
    outc: pd.DataFrame,
    source: str,
) -> Path:
    """Create the database from four normalized frames and add indexes + metadata."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        demo.to_sql("demo", conn, if_exists="append", index=False)
        drug.to_sql("drug", conn, if_exists="append", index=False)
        reac.drop_duplicates().to_sql("reac", conn, if_exists="append", index=False)
        outc.drop_duplicates().to_sql("outc", conn, if_exists="append", index=False)
        conn.executescript(INDEXES)
        conn.execute(
            "CREATE TABLE reac_counts AS "
            "SELECT pt, COUNT(DISTINCT primaryid) AS n_cases FROM reac GROUP BY pt"
        )
        conn.execute("CREATE UNIQUE INDEX idx_rc_pt ON reac_counts(pt)")
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [
                ("source", source),
                ("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


# --- read-only access -----------------------------------------------------

# sqlite3 authorizer action codes: SELECT=21, READ=20, FUNCTION=31, RECURSIVE=33
_ALLOWED_ACTIONS = {21, 20, 31, 33}


def _authorizer(action, *_args):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def connect_ro(db_path: str | Path) -> sqlite3.Connection:
    """Open the database read-only (OS-level protection against any write)."""
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Build it first: `pv-agent build-db --synthetic`"
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def run_with_limits(
    conn: sqlite3.Connection, sql: str, params=(), timeout_s: float = 5.0, authorize: bool = False
) -> list[dict]:
    """Execute a query with a wall-clock timeout and (optionally) a SELECT-only authorizer."""
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    if authorize:
        conn.set_authorizer(_authorizer)
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.set_progress_handler(None, 0)
        if authorize:
            conn.set_authorizer(None)
