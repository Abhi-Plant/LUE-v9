"""Pixel level (v0.8.0, core/pixels.py): the vectorised model of steps 03-05 equals the field-unit code for every
pixel; interpolation, statistics, pixel index, raster reading, storage and aggregation. Synthetic data only (no
project data needed). Run from the framework folder: pytest -q"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config, stats  # noqa: E402
from core import ensemble as en  # noqa: E402
from core import pixels as px  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return copy.deepcopy(config.load("dewas_2025_26"))


# ---------------------------------------------------------------- interpolation and statistics
def test_interp_rows_equals_numpy_interp():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(40, 60))
    y[rng.random(y.shape) < 0.6] = np.nan
    y[0] = np.nan                                                 # a row without a value
    y[1, :] = np.nan
    y[1, 17] = 0.4                                                # a row with one value
    x = np.arange(60, dtype=float)
    got = px.interp_rows(y, x)
    for r in range(len(y)):
        ok = ~np.isnan(y[r])
        want = np.interp(x, x[ok], y[r, ok]) if ok.any() else np.full(60, np.nan)
        assert np.allclose(got[r], want, equal_nan=True, atol=1e-12)


def test_interp_at_equals_numpy_interp():
    rng = np.random.default_rng(4)
    xo = np.sort(rng.choice(np.arange(0, 200), 25, replace=False)).astype(float)
    yo = rng.random((30, 25))
    yo[rng.random(yo.shape) < 0.5] = np.nan
    yo[0] = np.nan
    x = np.arange(-5, 210, dtype=float)
    got = px.interp_at(xo, yo, x)
    for r in range(len(yo)):
        ok = ~np.isnan(yo[r])
        want = np.interp(x, xo[ok], yo[r, ok]) if ok.any() else np.full(len(x), np.nan)
        assert np.allclose(got[r], want, equal_nan=True, atol=1e-12)
    assert px.interp_at(np.zeros(0), np.zeros((3, 0)), x).shape == (3, len(x))


def test_member_stats_rows_equals_member_stats():
    rng = np.random.default_rng(5)
    y = rng.gamma(4, 5, size=(20, 162))
    w = rng.random(162)
    got = px.member_stats_rows(y, w, [0.1, 0.5, 0.9])
    for r in range(len(y)):
        s = en.member_stats(y[r], w, [0.1, 0.5, 0.9])
        for k in ("mean", "sd", "p10", "p50", "p90"):
            assert got[k][r] == pytest.approx(s[k], rel=1e-12)
    y[3, 7] = np.nan
    assert np.isnan(px.member_stats_rows(y, w, [0.5])["p50"][3])


def test_encode_decode_half_step_and_range():
    v = np.array([-1.0, -0.12345, 0.0, 0.5, 0.99996, np.nan])
    q = px.encode(v, 1e-4, "NDVI")
    back = px.decode(q, 1e-4)
    assert np.isnan(back[-1]) and q[-1] == px.NODATA
    assert np.nanmax(np.abs(back[:-1] - v[:-1])) <= 0.5e-4 + 1e-12
    with pytest.raises(SystemExit):
        px.encode(np.array([4.0]), 1e-4, "EF")


# ---------------------------------------------------------------- pixel index, rasters
def test_pixel_index_centre_rule_and_representative_point():
    import geopandas as gpd
    from shapely.geometry import box
    key = (32643, 10.0, 0.0, 3000.0)                                # x = col * 10, y = 3000 - row * 10
    g = gpd.GeoDataFrame({"field_id": ["A", "B", "C"]},
                         geometry=[box(1000, 2000, 1030, 2030),          # 3 x 3 pixels, centres inside
                                   box(1101, 2001, 1104, 2004),          # 3 m square: no centre inside
                                   box(1000, 1950, 1015, 1960)],         # one centre (1005, 1955) inside
                         crs=32643)
    extent = (100, 96, 20, 12)                                      # cols 100-119 (x 1000-1200), rows 96-107
    pix, first, count, rule = px.pixel_index(g, key, extent)
    assert list(count) == [9, 1, 1] and rule == ["centre inside", "representative point", "centre inside"]
    assert list(first) == [0, 9, 10]
    assert set(zip(pix["row"][:9], pix["col"][:9])) == {(r, c) for r in range(1, 4) for c in range(0, 3)}
    assert (pix["row"][9], pix["col"][9]) == (3, 10)                # the pixel x 1100-1110, y 2000-2010
    assert (pix["row"][10], pix["col"][10]) == (8, 0)               # centre (1015, 1955) on the edge: not inside
    assert pix["x"][10] == 1005.0 and pix["y"][10] == 1955.0


def test_raster_values_in_strips_equal_direct_indexing(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    a = np.arange(70, dtype=np.float32).reshape(10, 7)
    p = tmp_path / "r.tif"
    with rasterio.open(p, "w", driver="GTiff", width=7, height=10, count=1, dtype="float32", crs="EPSG:32643",
                       transform=from_origin(0, 100, 10, 10), nodata=np.nan) as ds:
        ds.write(a, 1)
    rows = np.array([0, 9, 4, 4, 3, -1, 10, 2])
    cols = np.array([0, 6, 2, 5, 1, 0, 0, 7])
    v, fill = px.raster_values(p, rows, cols, strip=3)
    assert np.isnan(fill)
    assert np.array_equal(v[:5], a[rows[:5], cols[:5]]) and np.isnan(v[5:]).all()
    r, c = px.scene_positions(p, np.array([5.0, 10.0, 69.9]), np.array([95.0, 90.0, 0.1]), 32643)
    assert list(r) == [0, 1, 9] and list(c) == [0, 1, 6]           # an edge belongs to the pixel right / below


# ---------------------------------------------------------------- the model: vectorised = field-unit code
def synthetic(cfg, n=28, seed=11):
    """NDVI of crop cycles with clouds, spikes and special cases; Landsat 8 / 9 EF and ETa; daily weather of two
    cells."""
    rng = np.random.default_rng(seed)
    sc = cfg["season"]
    start = pd.Timestamp(sc["window"][0]) - pd.Timedelta(days=sc["lead_days"])
    end = pd.Timestamp(sc["window"][1])
    dates = pd.Series(pd.date_range(start, end, freq="5D"))
    t = (dates - start).dt.days.to_numpy(float)
    V = np.empty((n, len(dates)))
    for i in range(n):
        n1, n2 = rng.uniform(60, 110), rng.uniform(150, 190)
        lo, hi = rng.uniform(0.15, 0.3), rng.uniform(0.6, 0.85)
        V[i] = lo + (hi - lo) * (1 / (1 + np.exp(-0.1 * (t - n1))) + 1 / (1 + np.exp(0.08 * (t - n2))) - 1)
        V[i] += rng.normal(0, 0.015, len(t))
    V[rng.random(V.shape) < 0.3] = np.nan                         # clouds
    for i in range(0, n, 4):                                      # short negative spikes (haze)
        k = rng.integers(5, len(t) - 5)
        V[i, k] = V[i, k] - 0.3 if not np.isnan(V[i, k]) else np.nan
        V[i, k + 1] = V[i, k + 1] - 0.25 if not np.isnan(V[i, k + 1]) else np.nan
    V[1] = np.nan                                                 # no Sentinel-2 value
    V[3] = 0.25 + rng.normal(0, 0.01, len(t))                     # no crop cycle
    V[4, :] = np.nan
    V[4, [10, 15]] = [0.3, 0.6]                                   # two clear dates (< min_used_obs)
    V[5, :] = np.nan
    V[5, 12] = 0.4                                                # one clear date
    ovp = pd.concat([pd.DataFrame({"date": pd.date_range(start + pd.Timedelta(days=3), end, freq="16D"), "sensor": "L8"}),
                     pd.DataFrame({"date": pd.date_range(start + pd.Timedelta(days=11), end, freq="16D"),
                                   "sensor": "L9"})]).sort_values(["sensor", "date"]).reset_index(drop=True)
    EF = rng.uniform(0.2, 0.95, (n, len(ovp)))
    EF[rng.random(EF.shape) < 0.35] = np.nan
    EF[2] = np.nan                                                # no Landsat EF
    EF[6, (ovp.sensor == "L8").to_numpy()] = np.nan               # Landsat 8 missing: Landsat 9 for both
    EF[7, (ovp.sensor == "L9").to_numpy()] = np.nan               # Landsat 9 missing
    ETA = EF * rng.uniform(3, 6, EF.shape)
    ETA[rng.random(ETA.shape) < 0.1] = np.nan
    days = pd.date_range(start, end, freq="D")
    rows = []
    for cell, dt in (("22.9_76.1", 0.0), ("23.0_76.2", 1.5)):
        doy = days.dayofyear.to_numpy()
        tmin = 12 + dt + 6 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 1, len(days))
        rows.append(pd.DataFrame({"cell": cell, "date": days, "rs_mj_m2": rng.uniform(12, 22, len(days)),
                                  "tmin_c": tmin, "tmax_c": tmin + rng.uniform(10, 15, len(days)),
                                  "eto_mm": rng.uniform(2, 5, len(days))}))
    e3 = pd.concat(rows, ignore_index=True)
    cells = np.where(np.arange(n) % 2 == 0, "22.9_76.1", "23.0_76.2")
    return dates, V, ovp, EF, ETA, e3, cells, start


def test_vectorised_model_equals_field_unit_code(cfg):
    dates, V, ovp, EF, ETA, e3, cells, start = synthetic(cfg)
    W = px.Weather(cfg, e3)
    grid, combos, km, kc, c = px.combos_of(cfg)
    sc = cfg["season"]
    fractions = sorted(set(sc["ensemble_fractions"]) | {sc["amplitude_fraction"]})
    obs_off = (dates - start).dt.days.to_numpy()
    ovp_off = (ovp.date - start).dt.days.to_numpy()
    cell_idx = np.array([W.index[x] for x in cells])
    code, n_used, amp, s, cen, qual, n_ef = px.model_vec(V, EF, ETA, obs_off, ovp_off, ovp.sensor.to_numpy(), cell_idx,
                                                         W, cfg, combos, fractions)
    assert {1, 2, 3, 4, 5} <= set(code.tolist()) and (code == 0).sum() >= 15      # every case is exercised
    idx = np.arange(len(V))
    chk, mem = px.unit_check(cfg, idx, V, EF, ETA, dates, ovp, cells, e3)
    cl = cfg["ndvi_cleaning"]
    mine = [px.reason_text(int(code[i]), n_used[i], float(amp[i]), cl) for i in idx]
    assert mine == chk.reason.tolist()
    w0 = pd.Timestamp(sc["window"][0])
    ok = code == 0
    i_w0 = sc["lead_days"]
    for i in np.flatnonzero(ok):
        assert w0 + pd.Timedelta(days=int(cen["emergence"][i] - i_w0)) == chk.emergence[i]
        assert w0 + pd.Timedelta(days=int(cen["season_end"][i] - i_w0)) == chk.season_end[i]
    fresh = px.yields_from_s(s, grid, km)[2]
    um = mem.pivot(index="unit", columns="member", values="fresh_t_ha")
    for i in np.flatnonzero(ok):
        assert np.allclose(fresh[i], um.loc[f"PX{i}"].reindex(grid.member).to_numpy(), rtol=1e-9, atol=1e-9)
    assert np.isnan(s[~ok]).all()
    assert np.allclose(px.central_yield(s[ok], c, kc), fresh[ok][:, int(np.flatnonzero(
        (grid.eps_max == c["eps_max"]) & (grid.f_par == c["f_par"]) & (grid.hi == c["hi"]) & (grid.dm == c["dm"])
        & np.isclose(grid.season_p, c["season_p"]))[0])])


def test_one_sensor_fallback_uses_the_other(cfg):
    """A pixel without Landsat 8 EF uses Landsat 9 for both sensors (field_drivers): its season sums equal those of
    the same pixel without any Landsat 8 overpass."""
    dates, V, ovp, EF, ETA, e3, cells, start = synthetic(cfg)
    W = px.Weather(cfg, e3)
    grid, combos, km, kc, c = px.combos_of(cfg)
    sc = cfg["season"]
    fractions = sorted(set(sc["ensemble_fractions"]) | {sc["amplitude_fraction"]})
    i = 6
    l9 = (ovp.sensor == "L9").to_numpy()
    off, o_off, sen = (dates - start).dt.days.to_numpy(), (ovp.date - start).dt.days.to_numpy(), ovp.sensor.to_numpy()
    ci = np.array([W.index[cells[i]]])
    a = px.model_vec(V[[i]], EF[[i]], ETA[[i]], off, o_off, sen, ci, W, cfg, combos, fractions)
    b = px.model_vec(V[[i]], EF[[i]][:, l9], ETA[[i]][:, l9], off, o_off[l9], sen[l9], ci, W, cfg, combos, fractions)
    assert a[0][0] == 0 and np.allclose(a[3], b[3], rtol=1e-12)


# ---------------------------------------------------------------- aggregation
def test_parcel_fill_and_level_members_are_area_weighted(cfg):
    grid, combos, km, kc, c = px.combos_of(cfg)
    rng = np.random.default_rng(8)
    parcel = np.array([0, 0, 0, 1, 1, 2, 3, 3])
    s = rng.uniform(300, 900, (8, len(combos)))
    ok = np.array([True, True, False, True, True, False, True, False])
    sp, cnt = px.parcel_means(s, ok, parcel, 5)
    assert list(cnt) == [2, 2, 0, 1, 0]
    assert np.allclose(sp[0], s[[0, 1]].mean(0)) and np.isnan(sp[2]).all() and np.isnan(sp[4]).all()
    area = np.array([2.0, 1.0, 0.5, 3.0, 1.0])
    cell = np.array(["a", "a", "a", "b", "c"])
    tehsil = np.array(["T", "T", "T", "T", "U"])
    spf, status = px.fill_parcels(sp, area, cell, tehsil)
    assert list(status) == ["modelled", "modelled", "cell estimate", "modelled", "no estimate"]
    assert np.allclose(spf[2], (2 * sp[0] + 1 * sp[1]) / 3)
    mem, a = px.level_members(spf, area, tehsil, grid, km, "tehsil")
    assert set(mem.tehsil) == {"T"} and a["T"] == pytest.approx(6.5)
    y = px.yields_from_s(spf[:4], grid, km)[2]
    want = (y * area[:4, None]).sum(0) / area[:4].sum()           # the members are linear in S
    assert np.allclose(mem.set_index("member").fresh_t_ha.reindex(grid.member).to_numpy(), want, rtol=1e-12)
    assert stats.wquantile([1, 2, 3], [1, 1, 1], [0.5])[0] == 2


def test_weather_cells_nearest():
    m = px.weather_cells(["22.9_76.1", "23.4_76.6"], ["22.9_76.1", "23.3_76.5"])
    assert m == {"22.9_76.1": "22.9_76.1", "23.4_76.6": "23.3_76.5"}
