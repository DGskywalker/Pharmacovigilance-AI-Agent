"""Disproportionality statistics used in pharmacovigilance signal detection.

Contingency table for a (drug, reaction) pair:

                     reaction   no reaction
    drug                a            b
    all other drugs     c            d

PRR  = [a/(a+b)] / [c/(c+d)]
ROR  = (a*d) / (b*c)
Evans criteria (a common screening rule): a >= 3, PRR >= 2, chi-square >= 4.
A Haldane-Anscombe correction (+0.5) is applied to the ratio estimates when
any cell is zero.
"""
from __future__ import annotations

import math

Z95 = 1.959964


def disproportionality(a: int, b: int, c: int, d: int) -> dict:
    n = a + b + c + d
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    chi2 = n * (a * d - b * c) ** 2 / denom if denom else 0.0

    corrected = min(a, b, c, d) == 0
    a2, b2, c2, d2 = (x + 0.5 for x in (a, b, c, d)) if corrected else (a, b, c, d)

    prr = (a2 / (a2 + b2)) / (c2 / (c2 + d2))
    se_prr = math.sqrt(1 / a2 - 1 / (a2 + b2) + 1 / c2 - 1 / (c2 + d2))
    ror = (a2 * d2) / (b2 * c2)
    se_ror = math.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2)

    return {
        "cases": a,
        "prr": round(prr, 2),
        "prr_ci_low": round(math.exp(math.log(prr) - Z95 * se_prr), 2),
        "prr_ci_high": round(math.exp(math.log(prr) + Z95 * se_prr), 2),
        "ror": round(ror, 2),
        "ror_ci_low": round(math.exp(math.log(ror) - Z95 * se_ror), 2),
        "ror_ci_high": round(math.exp(math.log(ror) + Z95 * se_ror), 2),
        "chi2": round(chi2, 1),
        "flagged": bool(a >= 3 and prr >= 2 and chi2 >= 4),
    }
