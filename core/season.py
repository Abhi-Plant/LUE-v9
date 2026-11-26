"""NDVI cleaning (quality filter, despiking, daily interpolation, Savitzky-Golay) and season dates.

Season (method document step 1: "sowing and harvest dates from NDVI rise and fall"):
  emergence     = first day after the pre-season minimum where NDVI >= base_start + p * (peak - base_start)
  season end = first day after the peak where NDVI <= base_end + p * (peak - base_end)
with p the amplitude fraction (TIMESAT-type thresholds, Jonsson & Eklundh 2004)."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def clean_series(obs, min_valid, drop, max_span_days):
    """obs: one tehsil, columns date, ndvi, valid_fraction. Returns obs with status: used | low_valid | despiked.
    A spike is a date below both neighbours by more than `drop` (vs their mean) whose neighbours are at most
    `max_span_days` apart - short dips (fog, haze) are removed, a genuine seasonal trough is kept."""
    o = obs.sort_values("date").copy()
    o["status"] = np.where(o.ndvi.isna() | (o.valid_fraction < min_valid), "low_valid", "used")
    while True:                                   # drop the worst negative spike, one at a time
        u = o[o.status == "used"]
        v = u.ndvi.to_numpy()
        if len(v) < 3:
            break
        prev, nxt = v[:-2], v[2:]
        mid = v[1:-1]
        span = (u.date.iloc[2:].to_numpy() - u.date.iloc[:-2].to_numpy()) / np.timedelta64(1, "D")
        gap = (prev + nxt) / 2 - mid
        spike = (mid < np.minimum(prev, nxt)) & (gap > drop) & (span <= max_span_days)
        if not spike.any():
            break
        k = int(np.argmax(np.where(spike, gap, -np.inf)))
        o.loc[u.index[k + 1], "status"] = "despiked"
    return o


def daily_series(obs_clean, start, end, window, order):
    """Daily NDVI from the used observations: linear interpolation, constant beyond the ends, SG smoothing."""
    u = obs_clean[obs_clean.status == "used"].set_index("date").ndvi
    days = pd.date_range(start, end, freq="D")
    lin = u.reindex(u.index.union(days)).interpolate(method="time").reindex(days).ffill().bfill()
    smooth = savgol_filter(lin.to_numpy(), window, order, mode="interp")
    return pd.DataFrame({"date": days, "ndvi_linear": lin.to_numpy(), "ndvi_smooth": np.clip(smooth, -0.1, 1.0)})


def season_dates(daily, window, peak_search, p, min_drop):
    """Emergence and season-end dates from a daily NDVI series (one unit). The end of the season counts as found
    only if NDVI fell after the peak by at least min_drop (the min_amplitude processing setting): otherwise the canopy
    is still green at the window end, season end is set to the window end and flagged."""
    d = daily.set_index("date").ndvi_smooth
    w0, w1 = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    pk = d[pd.Timestamp(peak_search[0]):pd.Timestamp(peak_search[1])]
    peak_date, peak = pk.idxmax(), pk.max()
    pre = d[w0:peak_date]
    base_start_date, base_start = pre.idxmin(), pre.min()
    post = d[peak_date:w1]
    base_end_date, base_end = post.idxmin(), post.min()
    thr_start = base_start + p * (peak - base_start)
    thr_end = base_end + p * (peak - base_end)
    rise = d[base_start_date:peak_date]
    emergence = rise[rise >= thr_start].index.min()
    fall = d[peak_date:w1]
    below = fall[fall <= thr_end]
    end_found = bool(peak - base_end >= min_drop) and not below.empty   # post minimum always meets the threshold
    end = below.index.min() if end_found else w1
    first = d.index.min()                         # censoring: the cycle must be seen inside the series / search window
    start_ok = base_start_date > first and emergence > first
    peak_ok = pk.index.min() < peak_date < pk.index.max()
    return {"emergence": emergence, "season_end": end, "season_days": int((end - emergence).days) + 1,
            "start_flag": "ok" if start_ok else "start censored: pre-season minimum not observed (NDVI rising from "
                                                "the first day) - emergence may be earlier",
            "peak_flag": "ok" if peak_ok else "peak at the edge of the peak search window",
            "peak_date": peak_date, "peak_ndvi": peak, "base_start_date": base_start_date,
            "base_start_ndvi": base_start, "base_end_date": base_end_date, "base_end_ndvi": base_end,
            "amplitude": float(peak - max(base_start, base_end)),
            "threshold_start": thr_start, "threshold_end": thr_end, "amplitude_fraction": p,
            "end_flag": "ok" if end_found else "end not reached by window end - set to window end"}


def unit_seasons(nd, units, has_thermal, sc, cl, fractions, start):
    """Season of every unit (steps 03 and 11): quality filter and despiking, daily smoothed series, season dates at
    every amplitude fraction, quality counts. nd: unit, date, ndvi, valid_fraction; has_thermal: unit -> number of
    Landsat EF values; sc / cl: config season / ndvi_cleaning. A unit without Sentinel-2 values, Landsat EF, two clear
    dates or a usable crop cycle is excluded. Returns (obs, daily, seasons, quality, excluded) as lists."""
    obs, daily, seasons, quality, excluded = [], [], [], [], []
    by_unit = dict(tuple(nd.groupby("unit")))
    for u in units:
        g = by_unit.get(u)
        if g is None or g["ndvi"].notna().sum() == 0:
            excluded.append({"unit": u, "reason": "no Sentinel-2 observations"})
            continue
        if has_thermal.get(u, 0) == 0:
            excluded.append({"unit": u, "reason": "no Landsat EF observations"})
            continue
        o = clean_series(g[["unit", "date", "ndvi", "valid_fraction"]], cl["min_valid_fraction"], cl["despike_drop"],
                         cl["despike_max_span_days"])
        if (o.status == "used").sum() < 2:
            excluded.append({"unit": u, "reason": f"{int((o.status == 'used').sum())} clear observations"})
            continue
        dd = daily_series(o, start, sc["window"][1], cl["savgol_window_days"], cl["savgol_order"])
        dd.insert(0, "unit", u)
        ok, why = usable(o, dd, cl["min_used_obs"], cl["min_amplitude"], sc["peak_search"])
        if not ok:
            excluded.append({"unit": u, "reason": why})
            continue
        for p in fractions:
            seasons.append({"unit": u, **season_dates(dd, sc["window"], sc["peak_search"], p, cl["min_amplitude"])})
        dj = o[(o.date >= sc["quality_window"][0]) & (o.date <= sc["quality_window"][1])]
        quality.append({"unit": u, "clear_obs_quality_window": int((dj.status == "used").sum()),
                        "despiked_dates": int((o.status == "despiked").sum()),
                        "low_valid_dates": int((o.status == "low_valid").sum())})
        obs.append(o)
        daily.append(dd)
    return obs, daily, seasons, quality, excluded


def usable(obs_clean, daily, min_used, min_amplitude, peak_search):
    """(ok, reason): enough clear observations and a crop cycle with a clear NDVI amplitude."""
    n = int((obs_clean.status == "used").sum())
    if n < min_used:
        return False, f"{n} clear observations (< {min_used})"
    d = daily.set_index("date").ndvi_smooth
    pk = d[pd.Timestamp(peak_search[0]):pd.Timestamp(peak_search[1])]
    if pk.isna().all():
        return False, "no NDVI in the peak search window"
    amp = float(pk.max() - d.min())
    if amp < min_amplitude:
        return False, f"NDVI amplitude {amp:.2f} (< {min_amplitude}): no clear crop cycle"
    return True, "ok"
