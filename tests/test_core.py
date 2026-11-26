"""Unit and synthetic tests of the core modules (no project data needed). Run from the framework folder: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import calibration, config, eto, lue, sampling, season, stats  # noqa: E402
from core import drivers as dv  # noqa: E402
from core import ensemble as en  # noqa: E402


# ---------------------------------------------------------------- equations
def test_casa_ts_published_form():
    """Potter et al. (1993) eq. 6-7, capped at 1."""
    t, topt = np.array([0.0, 10, 19.5, 30, 40]), 19.5
    t1 = 0.8 + 0.02 * topt - 0.0005 * topt ** 2
    t2 = 1.1814 / (1 + np.exp(0.2 * (topt - 10 - t))) / (1 + np.exp(0.3 * (-topt - 10 + t)))
    assert np.allclose(lue.ts(t, topt), np.clip(t1 * t2, 0, 1))
    assert (lue.ts(np.linspace(-20, 50, 701), topt) <= 1).all()


def test_water_scalar_bounds():
    x = np.array([-0.5, 0.0, 0.4, 1.0, 1.7])
    assert np.allclose(lue.ws(x, "ef"), [0, 0, 0.4, 1, 1])
    assert np.allclose(lue.ws(x, "relative_et"), [0.5, 0.5, 0.7, 1, 1])
    with pytest.raises(ValueError):
        lue.ws(x, "unknown")


def test_fapar_relation_and_yield_units():
    assert np.allclose(lue.fapar_from_ndvi([0.0, 0.5, 1.0], 1.257, -0.161), [0.0, 0.4675, 1.0])
    dry, fresh = lue.crop_yield([100.0], 1.0, 0.25)          # 100 g DM m-2 = 1 t DM ha-1
    assert np.isclose(dry[0], 1.0) and np.isclose(fresh[0], 4.0)


def test_eto_fao56_example_18():
    """FAO-56 Example 18 (Brussels, 6 July): Ra 41.09 MJ m-2 d-1, ETo 3.9 mm d-1 (ea 1.409 kPa given as Tdew)."""
    ea = 1.409
    tdew = 237.3 * np.log(ea / 0.6108) / (17.27 - np.log(ea / 0.6108))
    assert abs(eto.extraterrestrial_radiation(50 + 48 / 60, 187) - 41.09) < 0.05
    e = eto.eto_fao56(12.3, 21.5, tdew, 22.07, 2.078, 100.1, 50 + 48 / 60, 100, 187)
    assert abs(e - 3.88) < 0.02
    assert abs(eto.wind_2m(1.0) - 0.748) < 1e-3                # eq. 47


# ---------------------------------------------------------------- curve fitting and integration
def _periods():
    s = pd.date_range("2025-10-01", periods=14, freq="16D")
    return s, s + pd.Timedelta(days=16)


def test_double_logistic_period_mean_fit_conserves_totals():
    s, e = _periods()
    v = 5 * np.exp(-0.5 * ((np.arange(14) - 6) / 2.0) ** 2) + 0.5
    p, info = dv.fit_double_logistic(s, e, v, s[0], scale=True, vmax_hi=2.0)
    day = pd.date_range(s[0], e[-1] - pd.Timedelta(days=1))
    r = dv.double_logistic(np.arange(len(day)), p)
    assert abs(r.sum() / (16 * v.sum()) - 1) < 1e-3              # daily increments sum to the period totals
    assert info["r2"] > 0.95


def test_fit_needs_enough_periods():
    s, e = _periods()
    with pytest.raises(ValueError):
        dv.fit_double_logistic(s, e, np.full(14, np.nan), s[0])


def test_periods_are_half_open():
    s, e = pd.to_datetime(["2025-10-01", "2025-10-16"]), pd.to_datetime(["2025-10-16", "2025-11-01"])
    k = dv.period_of_day(pd.date_range("2025-10-01", "2025-11-01"), s, e)
    assert (np.bincount(k[k >= 0]) == [15, 16]).all() and k[-1] == -1


# ---------------------------------------------------------------- phenology
def _ndvi(shape):
    days = pd.date_range("2025-09-01", "2026-03-31")
    x = np.arange(len(days))
    nd = 0.2 + 0.6 * np.exp(-0.5 * ((x - 150) / 30.0) ** 2) if shape == "cycle" else 0.2 + 0.6 / (1 + np.exp(-(x - 90) / 8))
    return pd.DataFrame({"date": days, "ndvi_smooth": nd})


def test_season_end_found_for_a_crop_cycle():
    sd = season.season_dates(_ndvi("cycle"), ["2025-10-01", "2026-03-31"], ["2025-11-01", "2026-02-28"], 0.2, 0.15)
    assert sd["end_flag"] == "ok" and sd["start_flag"] == "ok" and sd["emergence"] < sd["season_end"]


def test_season_censored_when_still_green():
    sd = season.season_dates(_ndvi("rising"), ["2025-10-01", "2026-03-31"], ["2025-11-01", "2026-02-28"], 0.2, 0.15)
    assert sd["end_flag"] != "ok" and sd["season_end"] == pd.Timestamp("2026-03-31")


# ---------------------------------------------------------------- sampling and estimation (synthetic population)
def _population(n=600, seed=0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"field_id": [f"F{i:05d}" for i in range(n)], "cell": rng.choice(["22.7_76.2", "22.8_76.3", "22.9_76.4"], n),
                      "area_ha": rng.lognormal(-0.9, 0.8, n), "source": "parcel", "mixed": False, "review": False,
                      "prob": 0.9})
    f["y"] = 15 + 4 * np.log1p(f.area_ha) + rng.normal(0, 2, n)
    return f


def test_srs_inclusion_probabilities_exact():
    f = _population()
    rules = sampling.eligibility(f, {"source": ["parcel"], "mixed": False, "review": False, "min_prob": 0.5,
                                     "min_area_ha": 0.0}, {})
    sel, frames = sampling.select_srs(f, rules, {"fields_per_cell": 5, "min_fields_per_cell": 2, "seed": 1})
    ht = sel.groupby(sel.field_id.map(f.set_index("field_id").cell)).inclusion_prob.apply(lambda p: (1 / p).sum())
    assert np.allclose(ht.sort_index().to_numpy(), frames.set_index("cell").frame_n.sort_index().to_numpy())
    # empirical inclusion frequency over repeated draws = n_h / N_h
    hits = pd.Series(0, index=f.field_id)
    for s in range(400):
        hits[sampling.select_srs(f, rules, {"fields_per_cell": 5, "min_fields_per_cell": 2, "seed": s})[0].field_id] += 1
    pi = f.cell.map(5 / f.groupby("cell").size()).to_numpy()
    assert abs((hits.to_numpy() / 400).mean() - pi.mean()) < 0.002


def test_ratio_estimator_targets_yield_per_hectare():
    f = _population(4000, seed=3)
    one = f.assign(cell="c")
    true = (one.area_ha * one.y).sum() / one.area_ha.sum()
    rng = np.random.default_rng(5)
    est = [sampling.stratified_estimate(one.y.iloc[i], one.cell.iloc[i], one.area_ha.iloc[i], pd.Series({"c": 1.0}))[0]
           for i in (rng.choice(len(one), 30, replace=False) for _ in range(3000))]
    assert abs(np.mean(est) - true) < 0.1                       # unweighted field mean would be ~1 t/ha low here


def test_ratio_weights():
    w = sampling.ratio_weights(pd.Series(["a", "a", "b"]), [1.0, 3.0, 2.0], pd.Series({"a": 40.0, "b": 10.0}))
    assert np.allclose(w, [10, 30, 10])


# ---------------------------------------------------------------- ensemble, calibration, statistics
def test_variance_partition_additive_model_is_exact():
    g = pd.DataFrame([(a, b) for a in (1, 2, 3) for b in (10, 20)], columns=["a", "b"]).assign(weight=1 / 6)
    g["y"] = 2 * g.a + g.b
    sh = en.variance_partition(g, "y", factors=["a", "b"])
    assert abs(sh["interaction"]) < 1e-9 and abs(sh["a"] + sh["b"] - 100) < 1e-9


def test_ratio_of_means_and_metrics():
    assert np.isclose(calibration.ratio_of_means([20, 10], [30, 10], [100, 1]), 2010 / 3010)
    m = stats.metrics([10, 12, 14, 16], [11, 12, 15, 15])
    assert np.isclose(m["bias"], 0.25) and np.isclose(m["rmse"], np.sqrt(0.75))
    assert np.isclose(m["r2_determination"], 1 - 3 / 20) and 0 <= m["r2_pearson"] <= 1
    assert stats.wquantile([1, 2, 3, 4], [1, 1, 1, 1], [0.5])[0] == 2


def test_shapefile_fingerprint_includes_attribute_table(tmp_path):
    for ext, txt in ((".shp", b"geom"), (".dbf", b"attr1"), (".shx", b"idx")):
        (tmp_path / f"x{ext}").write_bytes(txt)
    a = config.fingerprint(tmp_path / "x.shp")
    (tmp_path / "x.dbf").write_bytes(b"attr2")
    assert config.fingerprint(tmp_path / "x.shp") != a
