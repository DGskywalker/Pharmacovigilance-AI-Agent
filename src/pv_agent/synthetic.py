"""Synthetic FAERS-style data generator.

IMPORTANT: this data is entirely artificial. It mimics the *structure* of the FDA
Adverse Event Reporting System (FAERS) and embeds known drug-reaction associations
so the signal-detection pipeline and the agent can be demonstrated and evaluated
without downloading real data. It must never be used for real safety conclusions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .db import OUTCOME_LABELS, age_group, write_database

# (generic, brands, base weight)
DRUGS = [
    ("PREDNISONE", ["DELTASONE"], 9),
    ("DEXAMETHASONE", ["DECADRON"], 6),
    ("METHYLPREDNISOLONE", ["MEDROL"], 5),
    ("HYDROCORTISONE", ["CORTEF"], 3),
    ("METFORMIN", ["GLUCOPHAGE"], 8),
    ("ATORVASTATIN", ["LIPITOR"], 8),
    ("IBUPROFEN", ["ADVIL", "MOTRIN"], 7),
    ("ADALIMUMAB", ["HUMIRA"], 6),
    ("SEMAGLUTIDE", ["OZEMPIC", "WEGOVY"], 6),
    ("WARFARIN", ["COUMADIN"], 4),
    ("LISINOPRIL", ["ZESTRIL"], 5),
    ("OMEPRAZOLE", ["PRILOSEC"], 5),
    ("LEVOTHYROXINE", ["SYNTHROID"], 4),
    ("AMLODIPINE", ["NORVASC"], 4),
    ("SERTRALINE", ["ZOLOFT"], 4),
]

# Embedded ground-truth associations: drug -> {reaction: relative weight}
SIGNALS = {
    "PREDNISONE": {"OSTEOPOROSIS": 4, "HYPERGLYCAEMIA": 4, "CUSHINGOID": 2, "ADRENAL INSUFFICIENCY": 2},
    "DEXAMETHASONE": {"HYPERGLYCAEMIA": 4, "INSOMNIA": 3, "ADRENAL INSUFFICIENCY": 2},
    "METHYLPREDNISOLONE": {"OSTEOPOROSIS": 3, "HYPERGLYCAEMIA": 3, "CUSHINGOID": 2},
    "HYDROCORTISONE": {"ADRENAL INSUFFICIENCY": 4, "CUSHINGOID": 2},
    "METFORMIN": {"LACTIC ACIDOSIS": 5, "DIARRHOEA": 2},
    "ATORVASTATIN": {"MYALGIA": 4, "RHABDOMYOLYSIS": 3},
    "IBUPROFEN": {"GASTROINTESTINAL HAEMORRHAGE": 5, "RENAL FAILURE ACUTE": 3},
    "ADALIMUMAB": {"TUBERCULOSIS": 3, "SEPSIS": 3, "INJECTION SITE PAIN": 4},
    "SEMAGLUTIDE": {"PANCREATITIS": 3, "NAUSEA": 3, "VOMITING": 3},
    "WARFARIN": {"HAEMORRHAGE": 5, "INTERNATIONAL NORMALISED RATIO INCREASED": 4},
}

BACKGROUND = {
    "HEADACHE": 8, "NAUSEA": 7, "FATIGUE": 8, "DIZZINESS": 6, "RASH": 6, "PYREXIA": 5,
    "DYSPNOEA": 5, "PAIN": 7, "DIARRHOEA": 5, "VOMITING": 5, "DRUG INEFFECTIVE": 9,
    "ARTHRALGIA": 4, "ASTHENIA": 4, "PRURITUS": 4, "ANXIETY": 3,
}

SEVERE = {
    "LACTIC ACIDOSIS", "SEPSIS", "GASTROINTESTINAL HAEMORRHAGE", "HAEMORRHAGE", "TUBERCULOSIS",
    "PANCREATITIS", "ADRENAL INSUFFICIENCY", "RHABDOMYOLYSIS", "RENAL FAILURE ACUTE",
}

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
YEAR_WEIGHT = np.array([1.0, 1.0, 1.0, 1.2, 1.4, 1.6])
SEMAGLUTIDE_YEAR_FACTOR = np.array([0.3, 0.5, 1.0, 2.0, 4.0, 6.0])  # emerging-trend example
COUNTRIES = ["US", "GB", "DE", "FR", "JP", "AE", "IN", "CA"]
COUNTRY_P = [0.45, 0.1, 0.08, 0.07, 0.08, 0.04, 0.08, 0.1]
OUTCOME_CODES = list(OUTCOME_LABELS)


def _choice(rng, weights: dict):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def generate(n_cases: int = 25_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    generics = [d[0] for d in DRUGS]
    brands = {d[0]: d[1] for d in DRUGS}
    base_w = np.array([d[2] for d in DRUGS], dtype=float)
    sema_idx = generics.index("SEMAGLUTIDE")

    demo_rows, drug_rows, reac_rows, outc_rows = [], [], [], []
    year_p = YEAR_WEIGHT / YEAR_WEIGHT.sum()

    for i in range(n_cases):
        pid = str(10_000_000 + i)
        year_i = int(rng.choice(len(YEARS), p=year_p))
        year = YEARS[year_i]
        w = base_w.copy()
        w[sema_idx] *= SEMAGLUTIDE_YEAR_FACTOR[year_i]
        primary = generics[int(rng.choice(len(generics), p=w / w.sum()))]

        # drugs on the report: primary suspect, optional secondary suspect, concomitants
        used = [(primary, "PS")]
        if rng.random() < 0.25:
            sec = generics[int(rng.integers(len(generics)))]
            if sec != primary:
                used.append((sec, "SS"))
        for _ in range(int(rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))):
            con = generics[int(rng.integers(len(generics)))]
            if con not in [u[0] for u in used]:
                used.append((con, "C"))
        for seq, (g, role) in enumerate(used, start=1):
            name = rng.choice(brands[g] + [g]) if rng.random() < 0.4 else g
            drug_rows.append((pid, seq, role, name, g))

        # reactions: signal-driven for suspect drugs, background otherwise
        suspects = [u[0] for u in used if u[1] in ("PS", "SS") and u[0] in SIGNALS]
        n_react = int(rng.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1]))
        reactions = set()
        for _ in range(n_react):
            if suspects and rng.random() < 0.4:
                reactions.add(_choice(rng, SIGNALS[suspects[int(rng.integers(len(suspects)))]]))
            else:
                reactions.add(_choice(rng, BACKGROUND))
        for r in reactions:
            reac_rows.append((pid, r))

        # outcomes
        severe = bool(reactions & SEVERE)
        if rng.random() < (0.9 if severe else 0.45):
            if severe:
                w_out = {"HO": 0.5, "DE": 0.15, "LT": 0.15, "OT": 0.12, "DS": 0.04, "RI": 0.04}
            else:
                w_out = {"HO": 0.5, "OT": 0.35, "DE": 0.04, "LT": 0.04, "DS": 0.04, "RI": 0.03}
            for _ in range(int(rng.choice([1, 2], p=[0.8, 0.2]))):
                outc_rows.append((pid, _choice(rng, w_out)))

        # demographics
        age = float(np.clip(rng.normal(55, 18), 1, 95)) if rng.random() > 0.05 else np.nan
        sex = rng.choice(["F", "M", "UNK"], p=[0.58, 0.40, 0.02])
        wt = float(np.clip(rng.normal(75, 16), 35, 160)) if rng.random() > 0.1 else np.nan
        month, day = int(rng.integers(1, 13)), int(rng.integers(1, 29))
        demo_rows.append(
            (pid, pid, age, sex, wt, str(rng.choice(COUNTRIES, p=COUNTRY_P)),
             f"{year}-{month:02d}-{day:02d}", year, f"{year}Q{(month - 1) // 3 + 1}")
        )

    demo = pd.DataFrame(
        demo_rows,
        columns=["primaryid", "caseid", "age_years", "sex", "weight_kg", "country",
                 "report_date", "report_year", "report_quarter"],
    )
    demo["age_group"] = age_group(demo["age_years"])
    demo = demo[["primaryid", "caseid", "age_years", "age_group", "sex", "weight_kg", "country",
                 "report_date", "report_year", "report_quarter"]]
    drug = pd.DataFrame(drug_rows, columns=["primaryid", "drug_seq", "role_cod", "drugname", "prod_ai"])
    reac = pd.DataFrame(reac_rows, columns=["primaryid", "pt"])
    outc = pd.DataFrame(outc_rows, columns=["primaryid", "outc_cod"])
    return demo, drug, reac, outc


def build_synthetic_db(db_path, n_cases: int = 25_000, seed: int = 42):
    demo, drug, reac, outc = generate(n_cases, seed)
    return write_database(
        db_path, demo, drug, reac, outc,
        source=f"SYNTHETIC FAERS-style data (n={n_cases}, seed={seed}) - not real patient data",
    )
