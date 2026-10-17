import math

from pv_agent.stats import disproportionality


def test_known_values():
    r = disproportionality(a=20, b=80, c=100, d=9800)
    # PRR = (20/100) / (100/9900) = 19.8 ; ROR = (20*9800)/(80*100) = 24.5
    assert math.isclose(r["prr"], 19.8, rel_tol=1e-2)
    assert math.isclose(r["ror"], 24.5, rel_tol=1e-2)
    assert r["flagged"] is True
    assert r["ror_ci_low"] < r["ror"] < r["ror_ci_high"]


def test_no_association_not_flagged():
    r = disproportionality(a=10, b=90, c=100, d=900)
    assert r["flagged"] is False


def test_zero_cell_uses_correction():
    r = disproportionality(a=5, b=95, c=0, d=900)
    assert r["prr"] > 1 and math.isfinite(r["ror"])


def test_min_case_rule():
    assert disproportionality(a=2, b=2, c=1, d=1000)["flagged"] is False
