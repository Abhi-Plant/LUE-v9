"""Season sums and crop yield per run unit (tehsil or field) - one implementation for steps 05, 07 and 08.

Daily driver table d (step 04): unit, date, rs, tmean, fapar, ef_l8, ef_l9, ret_l8, ret_l9 (ETa/ETc), ...
Season table (step 03): unit, amplitude_fraction, emergence, season_end.
S = sum over the season of Ts(Topt) * Ws * fAPAR * Rs (MJ m-2); biomass B = eps_max * f * S (g m-2).

Two ways to get the daily values that are summed (config biomass_method):
  fit_drivers  (approach 1) daily Ts * Ws * fAPAR * Rs from the daily driver curves of step 04
  fit_biomass  (approach 2, v0.3 central) the growth rate r_p = Ts(Tmean_p) * Ws_p * fAPAR_p * Rs_p of every 16-day
               period, fitted by a double logistic (period-mean fit) and read daily: r(t). Daily growth
               = eps_max * f * r(t) (g m-2 d-1); eps_max and f only scale the curve, so one fit per (Topt, Ws form,
               Landsat mix) and unit is exact for all members. Fits: step 04 growth_fits.csv.
"""
import numpy as np
import pandas as pd

from . import lue
from .drivers import _day_number, double_logistic
from .ensemble import member_grid


def topt_of(d, topt):
    """Topt of every row of d. CASA definition (config temperature_scalar.topt: casa_ndvi_peak_month): the unit's
    Topt (column topt_unit, step 04) plus topt, which is then an offset fixed at 0; else topt itself (a crop value)."""
    return d["topt_unit"].to_numpy(float) + topt if "topt_unit" in d else topt


def casa_topt(d, seasons):
    """CASA Topt per unit: mean daily air temperature of the calendar month in which NDVI reaches its maximum -
    "the air temperature in the month when the FAS NDVI reaches its maximum for the year" (Potter et al. 1993 p.816);
    "the mean air temperature during the month of maximum leaf area index or NDVI development" (Bastiaanssen & Ali
    2003 p.324). d: unit, date, tmean (daily); seasons: unit -> peak_date (step 03)."""
    rows = []
    for u, g in d.groupby("unit"):
        peak = pd.Timestamp(seasons.loc[u, "peak_date"])
        m = g[g.date.dt.to_period("M") == peak.to_period("M")]
        rows.append({"unit": u, "peak_date": peak.date(), "peak_month": str(peak.to_period("M")),
                     "days": len(m), "topt_c": float(m.tmean.mean())})
    return pd.DataFrame(rows)


def season_mask(d, seasons, p):
    """Boolean: day inside emergence..season end of the unit at amplitude fraction p."""
    s = seasons[np.isclose(seasons.amplitude_fraction, p)].set_index("unit")
    if s.empty:
        raise SystemExit(f"no season dates for amplitude fraction {p}")
    return (d.date >= d.unit.map(s.emergence)) & (d.date <= d.unit.map(s.season_end))


def scalar_input(d, form, alpha=0.5):
    """Input of the water scalar: EF or ETa/ETc mixed from Landsat 8 (weight alpha) and 9 (1 - alpha)."""
    if form in ("ef", "casa_ef"):
        return alpha * d.ef_l8.to_numpy() + (1 - alpha) * d.ef_l9.to_numpy()
    if form == "relative_et":
        return alpha * d.ret_l8.to_numpy() + (1 - alpha) * d.ret_l9.to_numpy()
    if form == "none":
        return np.ones(len(d))
    raise ValueError(form)


def period_rate(pd_, topt, form, alpha=0.5):
    """Approach 2: growth rate per 16-day period without eps_max and f, r_p = Ts(Tmean_p) Ws_p fAPAR_p Rs_p
    (MJ m-2 d-1). pd_ = step-04 period_drivers (ef_l8, ef_l9, ret_l8, ret_l9 already gap-filled)."""
    return lue.ts(pd_.tmean.to_numpy(), topt_of(pd_, topt)) * lue.ws(scalar_input(pd_, form, alpha), form) * pd_.fapar.to_numpy() \
        * pd_.rs.to_numpy()


def fit_key(form, topt, alpha):
    """Fits are stored per (Ws form, Topt, Landsat mix); the mix does not matter for Ws = 1."""
    return form, round(float(topt), 4), 0.5 if form == "none" else round(float(alpha), 4)


def rate_daily(d, fits, topt, form, alpha=0.5):
    """Approach 2: fitted growth-rate curve r(t) of each unit on the rows of d (unit, date)."""
    f, t, a = fit_key(form, topt, alpha)
    sel = fits[(fits.ws_form == f) & np.isclose(fits.topt, t) & np.isclose(fits.alpha, a)].set_index("unit")
    if sel.empty or not set(d.unit) <= set(sel.index):
        raise SystemExit(f"no growth-rate fit for Ws {f}, Topt {t}, mix {a} - run step 04 (grid in config)")
    x = np.asarray(_day_number(d.date), dtype=float)
    out = np.empty(len(d))
    for u, idx in d.groupby("unit").indices.items():
        r = sel.loc[u]
        p = [r.vmin, r.vmax, r.n1, r.m1, r.n2, r.m2]
        out[idx] = double_logistic(x[idx] - float(_day_number([pd.Timestamp(r.t0)])[0]), p)
    return np.clip(out, 0.0, None)


def season_sum(d, topt, form, mask, alpha=0.5, fapar_col="fapar", fits=None):
    """S per unit (Series). fits given: approach 2 (fitted growth-rate curve); else approach 1 (daily drivers)."""
    if fits is not None:
        if fapar_col != "fapar":
            raise ValueError("an alternative fAPAR column is an approach-1 calculation (fits=None)")
        v = rate_daily(d, fits, topt, form, alpha)
    else:
        v = lue.ts(d.tmean.to_numpy(), topt_of(d, topt)) * lue.ws(scalar_input(d, form, alpha), form) * d[fapar_col].to_numpy() \
            * d.rs.to_numpy()
    return pd.Series(np.where(mask, v, 0.0), index=d.index).groupby(d.unit).sum()


def unit_yield(S, eps_max, f, hi, dm):
    b = eps_max * f * S
    dry, fresh = lue.crop_yield(b.to_numpy(), hi, dm)
    return pd.DataFrame({"unit": b.index, "biomass_g_m2": b.to_numpy(), "dry_t_ha": dry, "fresh_t_ha": fresh})


def central_config(cfg):
    P = cfg["parameters"]
    return {"eps_max": P["eps_max"]["central"], "f_par": P["f_par"]["central"], "topt": P["topt"]["central"],
            "hi": P["hi"]["central"], "dm": P["dm"]["central"], "ws_form": "ef",
            "season_p": cfg["season"]["amplitude_fraction"]}


def run(d, seasons, c, alpha=0.5, fapar_col="fapar", fits=None):
    """Yield per unit for one configuration c (dict with the member keys); fits: approach 2."""
    S = season_sum(d, c["topt"], c["ws_form"], season_mask(d, seasons, c["season_p"]), alpha, fapar_col, fits)
    return unit_yield(S, c["eps_max"], c["f_par"], c["hi"], c["dm"])


def members(d, seasons, cfg, fits=None):
    """All LUE-ENS members per unit. S is computed once per (Topt, Ws form, season fraction); fits: approach 2."""
    grid = member_grid(cfg)
    masks = {p: season_mask(d, seasons, p) for p in grid.season_p.unique()}
    S = {(t, w, p): season_sum(d, t, w, masks[p], fits=fits) for t in grid.topt.unique() for w in grid.ws_form.unique()
         for p in grid.season_p.unique()}
    out = []
    for r in grid.itertuples():
        y = unit_yield(S[(r.topt, r.ws_form, r.season_p)], r.eps_max, r.f_par, r.hi, r.dm)
        out.append(y.assign(member=r.member))
    m = pd.concat(out, ignore_index=True)
    return m.merge(grid, on="member")[["unit", "member", "eps_max", "f_par", "topt", "hi", "dm", "ws_form",
                                       "season_p", "weight", "biomass_g_m2", "dry_t_ha", "fresh_t_ha"]], grid


def period_method(m, seasons, c):
    """16-day variant (tehsil units, CSV only, v0.3): B = sum_periods n_season_days * eps * Ts(Tmean_p) * Ws(EF_p) *
    fAPAR_p * f * Rs_p (EF form). A period without EF takes the EF of the neighbouring periods (linear in time)."""
    rows = []
    s = seasons[np.isclose(seasons.amplitude_fraction, c["season_p"])]
    for _, r in s.iterrows():
        mt = m[m.tehsil == r.unit].sort_values("period").reset_index(drop=True)
        ef = mt.set_index("mid").ef.interpolate(method="time", limit_direction="both").to_numpy()
        day = pd.date_range(r.emergence, r.season_end, freq="D")
        k = pd.Series(np.searchsorted(mt.period.to_numpy(), day.to_numpy(), side="right") - 1).value_counts()
        to = float(mt.topt_unit.iloc[0]) + c["topt"] if "topt_unit" in mt else c["topt"]
        v = sum(n * float(lue.ts(mt.tmean[i], to)) * float(lue.ws(ef[i], "ef")) * mt.fapar[i] * mt.rs[i]
                for i, n in k.items())
        rows.append((r.unit, float(v)))
    return unit_yield(pd.Series(dict(rows)), c["eps_max"], c["f_par"], c["hi"], c["dm"])
