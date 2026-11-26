"""FAO-56 ch. 7 bare-soil evaporation (core/soil_evaporation.py) against the equations and FAO-56 Example 31."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import soil_evaporation as se  # noqa: E402


def test_tew_eq73():
    assert se.tew(0.23, 0.10, 0.10) == pytest.approx(18.0)        # FAO-56 Example 35: TEW = 18 mm


def test_kr_example31():
    assert se.kr(9.45, 20, 9) == pytest.approx((20 - 9.45) / (20 - 9))   # Example 31: 0.96
    assert se.kr(5, 20, 9) == 1.0 and se.kr(20, 20, 9) == 0.0


def test_balance_stage1_stage2_and_rain():
    t = se.bare_soil(range(6), [0, 0, 0, 0, 30, 0], [5] * 6, tew_mm=20, rew_mm=9)
    assert list(t.ke[:2]) == [1.2, 1.2]                            # stage 1: Ke = Kc max (Kcb 0, few 1)
    assert t.de_end_mm[1] == pytest.approx(12.0) and t.kr[2] < 1    # De > REW -> stage 2
    assert t.de_end_mm[4] == 0 and t.dp_mm[4] > 0                  # heavy rain refills the layer, excess drains
    assert (t.de_end_mm <= 20).all() and (t.de_end_mm >= 0).all()   # eq. 78
