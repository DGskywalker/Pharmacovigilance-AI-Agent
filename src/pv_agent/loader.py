"""Load real FDA FAERS quarterly ASCII extracts into the project database.

Download quarterly "ASCII" zips from the FDA FAERS page, extract them into one
folder, then run:  pv-agent build-db --faers-dir path/to/folder

Expected files per quarter (case-insensitive): DEMOyyQq.txt, DRUGyyQq.txt,
REACyyQq.txt, OUTCyyQq.txt, '$'-delimited. Cases are de-duplicated by keeping the
most recent version of each CASEID (latest FDA receipt date).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .db import age_group, write_database

_AGE_TO_YEARS = {"YR": 1.0, "DEC": 10.0, "MON": 1 / 12, "WK": 1 / 52, "DY": 1 / 365, "HR": 1 / 8760}


def _read(folder: Path, prefix: str, usecols: list[str]) -> pd.DataFrame:
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name.upper().startswith(prefix)
                   and p.suffix.lower() == ".txt")
    if not files:
        raise FileNotFoundError(f"No {prefix}*.txt files found under {folder}")
    frames = []
    for f in files:
        df = pd.read_csv(f, sep="$", dtype=str, encoding="latin-1", on_bad_lines="skip", low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        frames.append(df[[c for c in usecols if c in df.columns]])
    return pd.concat(frames, ignore_index=True)


def build_from_faers(faers_dir: str | Path, db_path: str | Path) -> Path:
    folder = Path(faers_dir)
    demo = _read(folder, "DEMO", ["primaryid", "caseid", "fda_dt", "rept_dt", "age", "age_cod", "sex",
                                 "wt", "wt_cod", "occr_country", "reporter_country"])
    drug = _read(folder, "DRUG", ["primaryid", "drug_seq", "role_cod", "drugname", "prod_ai"])
    reac = _read(folder, "REAC", ["primaryid", "pt"])
    outc = _read(folder, "OUTC", ["primaryid", "outc_cod"])

    # keep latest version per case
    demo = demo.sort_values(["caseid", "fda_dt", "primaryid"]).drop_duplicates("caseid", keep="last")

    age = pd.to_numeric(demo.get("age"), errors="coerce") * demo.get("age_cod", pd.Series(dtype=str)).map(_AGE_TO_YEARS)
    age = age.where((age >= 0) & (age <= 120))
    wt = pd.to_numeric(demo.get("wt"), errors="coerce")
    wt = wt.where(demo.get("wt_cod", pd.Series(index=demo.index, dtype=str)).fillna("KG").str.upper() != "LBS", wt * 0.453592)
    date = pd.to_datetime(demo["rept_dt"].fillna(demo["fda_dt"]), format="%Y%m%d", errors="coerce")
    country = demo["occr_country"] if "occr_country" in demo else demo.get("reporter_country")

    out_demo = pd.DataFrame({
        "primaryid": demo["primaryid"], "caseid": demo["caseid"], "age_years": age.round(1),
        "age_group": age_group(age), "sex": demo["sex"].fillna("UNK"), "weight_kg": wt.round(1),
        "country": country, "report_date": date.dt.strftime("%Y-%m-%d"),
        "report_year": date.dt.year, "report_quarter": date.dt.year.astype("Int64").astype(str) + "Q" + date.dt.quarter.astype("Int64").astype(str),
    })
    out_demo.loc[date.isna(), ["report_year", "report_quarter"]] = [None, None]

    keep = set(out_demo["primaryid"])
    drug = drug[drug["primaryid"].isin(keep)].copy()
    drug["drugname"] = drug["drugname"].fillna("").str.strip().str.upper()
    drug["prod_ai"] = drug["prod_ai"].fillna("").str.strip().str.upper()
    drug["drug_seq"] = pd.to_numeric(drug["drug_seq"], errors="coerce").fillna(0).astype(int)
    drug = drug[drug["drugname"] != ""]
    reac = reac[reac["primaryid"].isin(keep)].copy()
    reac["pt"] = reac["pt"].str.strip().str.upper()
    outc = outc[outc["primaryid"].isin(keep)].copy()
    outc["outc_cod"] = outc["outc_cod"].str.strip().str.upper()

    return write_database(db_path, out_demo, drug, reac, outc,
                          source=f"FDA FAERS ASCII extracts from {folder.name} ({len(out_demo):,} de-duplicated cases)")
