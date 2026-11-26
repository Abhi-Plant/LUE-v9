"""Model drivers: 16-day periods [start, end), interpolation in time, weather adjustment to the period means,
double-logistic fit (Beck et al. 2006, eq. 3), smoothing spline (Reinsch 1967), Landsat gap filling; the input
tables and daily drivers of field units (one implementation for the sample, steps 01 and 04, and for the check of the
vectorised pixel model, step 05b)."""
import numpy as np
import pandas as pd

from . import eto, lue


def _day_number(dates):
    """Days since 1970-01-01 as float, independent of the datetime unit (ns / us) of the pandas version."""
    return (pd.DatetimeIndex(dates) - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)


def interpolate_dates(days, dates, values):
    """Values at observation dates (e.g. Landsat overpasses), linear in time, constant beyond the first / last date.
    Missing values are skipped; returns NaN everywhere when no value is available."""
    x = np.asarray(_day_number(dates), dtype=float)
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    if not ok.any():
        return np.full(len(days), np.nan)
    o = np.argsort(x[ok])
    return np.interp(np.asarray(_day_number(days), dtype=float), x[ok][o], v[ok][o])


# ---------------------------------------------------------------- v0.3: 16-day table -> daily curves
# Periods are [period, period_end): the next period starts on the end date of the previous one.

def period_mid(start, end):
    """Mid-time of a period [start, end)."""
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return s + (e - s) / 2


def period_of_day(days, starts, ends):
    """Index of the period containing each day (-1 = outside every period)."""
    x = np.asarray(_day_number(days), dtype=float)
    s, e = np.asarray(_day_number(starts), dtype=float), np.asarray(_day_number(ends), dtype=float)
    k = np.searchsorted(s, x, side="right") - 1
    ok = (k >= 0) & (x < e[np.clip(k, 0, len(e) - 1)])
    return np.where(ok, k, -1)


def adjust_to_periods(dates, values, starts, ends, table_values, mode):
    """Make daily values reproduce the period table: 'ratio' scales, 'delta' shifts (per tehsil-period).
    Only periods with daily values are adjusted. Returns adjusted values and a factor per period (NaN = no days)."""
    v = np.asarray(values, dtype=float)
    k = period_of_day(dates, starts, ends)
    fac = np.full(len(starts), np.nan)
    out = v.copy()
    for i, tv in enumerate(np.asarray(table_values, dtype=float)):
        sel = k == i
        if not sel.any() or np.isnan(tv):
            continue
        dm = v[sel].mean()
        if mode == "ratio":
            fac[i] = tv / dm
            out[sel] = v[sel] * fac[i]
        elif mode == "delta":
            fac[i] = tv - dm
            out[sel] = v[sel] + fac[i]
        elif mode == "none":
            fac[i] = 0.0
        else:
            raise ValueError(mode)
    return out, fac


def double_logistic(t, p):
    """Beck et al. (2006) double logistic: v = vmin + (vmax - vmin) [1/(1+e^(-m1 (t-n1))) + 1/(1+e^(m2 (t-n2))) - 1];
    t in days; n1 / n2 = mid-dates of green-up / senescence, m1 / m2 = their rates (d-1)."""
    vmin, vmax, n1, m1, n2, m2 = p
    t = np.asarray(t, dtype=float)
    return vmin + (vmax - vmin) * (1.0 / (1.0 + np.exp(-m1 * (t - n1))) + 1.0 / (1.0 + np.exp(m2 * (t - n2))) - 1.0)


def fit_double_logistic(starts, ends, values, t0, starts_n1=(40, 60, 80), starts_n2=(120, 140, 160), scale=False,
                        vmax_hi=1.0, x0=None):
    """Fit the double logistic so that its MEAN over every period equals the period value (period-mean fit, not a
    point fit at mid-period). t = days since t0. Multi-start bounded least squares (x0: extra start, e.g. the fit of
    a neighbouring configuration). scale=True: values divided by their maximum before fitting and vmin / vmax scaled
    back (for quantities not bounded by 1, e.g. growth rates); vmax_hi = upper bound of vmax on the fitted scale.
    Returns parameters (original units) and fit statistics."""
    from scipy.optimize import least_squares
    s = np.asarray(_day_number(starts), dtype=float) - float(_day_number([t0])[0])
    e = np.asarray(_day_number(ends), dtype=float) - float(_day_number([t0])[0])
    y = np.asarray(values, dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 6:
        raise ValueError(f"double logistic: {int(ok.sum())} period values, at least 6 needed for 6 parameters")
    s, e, y = s[ok], e[ok], y[ok]
    k = 1.0
    if scale:
        k = float(np.max(np.abs(y))) or 1.0
    ys = y / k
    t = np.concatenate([np.arange(a, b) for a, b in zip(s, e)])
    idx = np.concatenate([np.full(int(b - a), i) for i, (a, b) in enumerate(zip(s, e))])
    cnt = np.bincount(idx).astype(float)

    def resid(p):
        return np.bincount(idx, weights=double_logistic(t, p)) / cnt - ys

    lo, hi = [0.0, 0.0, s.min(), 0.005, s.min() + 30, 0.005], [vmax_hi, vmax_hi, e.max(), 1.0, e.max(), 1.0]
    lo_, hi_ = np.array(lo) + 1e-6, np.array(hi) - 1e-6
    starts = [np.clip([ys.min(), ys.max(), n1, 0.1, n2, 0.1], lo_, hi_) for n1 in starts_n1 for n2 in starts_n2]
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).copy()
        x0[:2] /= k
        starts.insert(0, np.clip(x0, lo_, hi_))
    best = None
    for p0 in starts:
        r = least_squares(resid, p0, bounds=(lo, hi))
        if r.x[4] > r.x[2] and (best is None or r.cost < best.cost):
            best = r
    if best is None:
        raise RuntimeError("double logistic: no fit with senescence after green-up")
    f = best.fun * k
    p = best.x.copy()
    p[:2] *= k
    return p, {"n_obs": int(len(y)), "rmse": float(np.sqrt(np.mean(f ** 2))),
               "r2": float(1 - np.sum(f ** 2) / np.sum((y - y.mean()) ** 2)), "bias": float(f.mean())}


def eval_double_logistic(days, p, t0, lo=0.0, hi=1.0):
    t = np.asarray(_day_number(days), dtype=float) - float(_day_number([t0])[0])
    return np.clip(double_logistic(t, p), lo, hi)


def fit_smoothing_spline(dates, values, sigma, min_obs=5, lam_grid=None):
    """Cubic smoothing spline (Reinsch 1967) through values at their dates. Smoothing parameter by the discrepancy
    principle (Morozov 1966): the least smoothing whose residual RMSE reaches the measurement noise sigma.
    Fewer than min_obs values -> linear interpolation (method 'linear'). Returns (callable on day numbers, info)."""
    from scipy.interpolate import make_smoothing_spline
    x = np.asarray(_day_number(dates), dtype=float)
    y = np.asarray(values, dtype=float)
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    o = np.argsort(x)
    x, y = x[o], y[o]
    if len(y) == 0:
        return None, {"method": "none", "n_obs": 0}
    if len(y) < min_obs:
        return (lambda t: np.interp(t, x, y)), {"method": "linear", "n_obs": int(len(y)), "lam": np.nan,
                                                "rmse": 0.0, "x_first": x[0], "x_last": x[-1]}
    lam_grid = np.logspace(-2, 7, 181) if lam_grid is None else lam_grid
    for lam in lam_grid:
        sp = make_smoothing_spline(x, y, lam=lam)
        rmse = float(np.sqrt(np.mean((sp(x) - y) ** 2)))
        if rmse >= sigma:
            break
    return sp, {"method": "smoothing_spline", "n_obs": int(len(y)), "lam": float(lam), "rmse": rmse,
                "x_first": x[0], "x_last": x[-1]}


def eval_spline(days, fn, info, lo, hi):
    """Spline on the daily grid, held constant beyond the first / last observation, bounded to lo-hi."""
    if fn is None:
        return np.full(len(days), np.nan)
    t = np.clip(np.asarray(_day_number(days), dtype=float), info["x_first"], info["x_last"])
    return np.clip(fn(t), lo, hi)


def sensor_noise(a, b):
    """Noise of one sensor from paired values of two sensors: sd(a - b) / sqrt(2) (differences of the same period;
    includes real date-to-date change, so it is an upper bound of the noise)."""
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    d = d[~np.isnan(d)]
    return float(d.std(ddof=1) / np.sqrt(2)), int(len(d))


def fill_sensors(a, b, times):
    """Two-sensor series per period (e.g. Landsat 8 and 9 EF): a missing value takes the other sensor of the same
    period; where both are missing, linear interpolation in time (constant beyond the first / last value)."""
    a1 = pd.Series(np.asarray(a, dtype=float)).fillna(pd.Series(np.asarray(b, dtype=float)))
    b1 = pd.Series(np.asarray(b, dtype=float)).fillna(pd.Series(np.asarray(a, dtype=float)))
    x = np.asarray(_day_number(times), dtype=float)
    out = []
    for v in (a1, b1):
        v = v.to_numpy()
        ok = ~np.isnan(v)
        out.append(np.interp(x, x[ok], v[ok]) if ok.any() else v)
    return out[0], out[1]


# ---------------------------------------------------------------- field units: input tables and daily drivers
def clip_shares(t):
    """Clear shares within 1e-9 of 0 or 1 set to the bound (sums of pixel weights can round past them; a larger
    excess stays and fails the range check). Returns (rows set, largest share before)."""
    vf, eps = t.valid_fraction, 1e-9
    r = (vf > 1) & (vf <= 1 + eps) | (vf < 0) & (vf >= -eps)
    t["valid_fraction"] = vf.where(~r, vf.clip(0, 1))
    return int(r.sum()), float(vf.max())


def weather_eto(e3):
    """E3 with eto_mm: FAO-56 Penman-Monteith ETo from the ERA5-Land variables when E3 has none. Returns (E3, computed)."""
    if "eto_mm" in e3:
        return e3, False
    need = ["tdew_c", "u10_ms", "sp_kpa", "elev_m", "lat"]
    miss = [c for c in need if c not in e3]
    if miss:
        raise SystemExit(f"E3 needs eto_mm or {need} (missing {miss})")
    e3["eto_mm"] = eto.eto_fao56(e3.tmin_c, e3.tmax_c, e3.tdew_c, e3.rs_mj_m2, eto.wind_2m(e3.u10_ms), e3.sp_kpa,
                                 e3.lat, e3.elev_m, e3.date.dt.dayofyear)
    return e3, True


def field_tables(cfg, units, e1, e2, e3):
    """Input tables of field units from the per-field values E1 (Sentinel-2 per date), E2 (Landsat per overpass) and
    the daily weather E3 per cell (with eto_mm): obs (NDVI per date), thermal (EF and ETa per overpass; an overpass
    with a clear share below ef.min_valid_fraction is set to missing) and the daily weather of every unit (its cell).
    Returns (obs, thermal, weather, (overpasses set to missing, overpasses with EF))."""
    obs = pd.DataFrame({"unit": e1.field_id, "date": e1.date, "ndvi": e1.ndvi, "valid_fraction": e1.valid_fraction,
                        "n_scenes": e1.get("n_scenes", 1)})
    thermal = pd.DataFrame({"unit": e2.field_id, "date": e2.date, "sensor": e2.sensor.str.upper(), "ef": e2.ef,
                            "eta_mm_day": e2.eta_mm_day, "valid_fraction": e2.valid_fraction,
                            "representative": "overpass"})
    low = thermal.valid_fraction.fillna(0) < cfg["ef"]["min_valid_fraction"]   # same rule as the Sentinel-2 dates
    n_low = (int((low & thermal.ef.notna()).sum()), int(thermal.ef.notna().sum()))
    thermal.loc[low, ["ef", "eta_mm_day"]] = np.nan
    weather = units[["unit", "cell"]].merge(e3.rename(columns={"rs_mj_m2": "rs", "tmin_c": "tmin", "tmax_c": "tmax",
                                                               "eto_mm": "eto"}), on="cell").drop(columns="cell")
    weather["lat"] = weather.unit.map(units.set_index("unit").lat)
    weather["lon"] = weather.unit.map(units.set_index("unit").lon)
    return obs, thermal, weather, n_low


def field_drivers(cfg, units, w, th, nd, seas, ob=None, st=None):
    """Daily drivers of field units, approach 1 (method document): daily weather of the unit's cell, fAPAR from the
    daily NDVI (SL2P fAPAR on the used dates when ob has fapar_sl2p; st = step-03 observation status), Kc and ETc,
    Landsat EF and ETa/ETc per sensor linear in time between the overpasses (a missing sensor takes the other),
    season and heat days, CASA Topt of the unit (its NDVI-peak month). w: weather; th: thermal (overpass rows);
    nd: daily NDVI (ndvi_smooth); seas: central season by unit. Returns (daily table, Topt table or None, missing
    sensors)."""
    from . import model
    days = pd.date_range(*cfg["season"]["window"], freq="D")
    cc, hs = cfg["crop_coefficient"], cfg["heat_stress"]
    rel = pd.Series(cfg["fapar"][cfg["fapar"]["relation_field_units"]])
    sl2p = ob is not None and "fapar_sl2p" in ob
    wg = dict(tuple(w[w.date.isin(days)].groupby("unit")))
    ng, tg = dict(tuple(nd.groupby("unit"))), dict(tuple(th.groupby(["unit", "sensor"])))
    og = dict(tuple(ob.groupby("unit"))) if sl2p else {}
    sg = dict(tuple(st.groupby("unit"))) if sl2p else {}
    rows, missing = [], []
    for u in units.unit:
        wu = wg.get(u, w.iloc[:0]).sort_values("date")
        if len(wu) != len(days):
            raise SystemExit(f"{u}: weather has {len(wu)} of {len(days)} days")
        rs, tmean, tmax = wu.rs.to_numpy(), ((wu.tmin + wu.tmax) / 2).to_numpy(), wu.tmax.to_numpy()
        ndvi = ng[u].set_index("date").ndvi_smooth.reindex(days).to_numpy()
        if sl2p:
            o = og[u].merge(sg[u][["date", "status"]], on="date")
            o = o[o.status == "used"]
            fapar = np.clip(interpolate_dates(days, o.date, o.fapar_sl2p), 0, 1)
        else:
            fapar = lue.fapar_from_ndvi(ndvi, rel.slope, rel.intercept)
        kc = lue.crop_coefficient(ndvi, cc)
        etc = kc * wu.eto.to_numpy()
        etc_s = pd.Series(etc, index=days)
        r = {"unit": u, "date": days, "rs": rs, "tmean": tmean, "tmax": tmax, "eto": wu.eto.to_numpy(), "ndvi": ndvi,
             "fapar": fapar, "kc": kc, "etc": etc}
        sensors = [s for s in ("L8", "L9") if (u, s) in tg and tg[(u, s)].ef.notna().any()]
        if not sensors:
            raise SystemExit(f"{u}: no Landsat EF (step 03 should have excluded it)")
        for s in ("L8", "L9"):
            src = s if s in sensors else sensors[0]                      # one sensor missing: use the other
            if src != s:
                missing.append({"unit": u, "missing": s, "used_instead": src})
            t = tg[(u, src)].sort_values("date")
            ret = t.eta_mm_day.to_numpy() / t.date.map(etc_s).to_numpy()   # overpass ETa vs ETc of that day
            r[f"ef_{s.lower()}"] = np.clip(interpolate_dates(days, t.date, t.ef), 0, 1)
            r[f"ret_{s.lower()}"] = np.clip(interpolate_dates(days, t.date, ret), 0, 1.5)
        s0 = seas.loc[u]
        r["in_season"] = (days >= s0.emergence) & (days <= s0.season_end)
        bulk0 = s0.emergence + pd.Timedelta(days=hs["bulking_start_days_after_emergence"])
        r["heat_day"] = (days >= bulk0) & (days <= s0.season_end) & (r["tmax"] > hs["tmax_threshold_c"])
        rows.append(pd.DataFrame(r))
    d = pd.concat(rows, ignore_index=True)
    topt = None
    if cfg["temperature_scalar"]["topt"] == "casa_ndvi_peak_month":   # Topt per unit from its own NDVI peak month
        topt = model.casa_topt(d, seas)
        d["topt_unit"] = d.unit.map(topt.set_index("unit").topt_c)
    d["ef"] = (d.ef_l8 + d.ef_l9) / 2
    d["ret"] = (d.ret_l8 + d.ret_l9) / 2
    return d, topt, missing
