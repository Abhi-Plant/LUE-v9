"""Step 10 - framework validation: independent checks of equations, step outputs, ensemble, calibration, Sobol
evaluation and reproducibility. Without observed yields of the season this is verification of the implementation
(does the code compute what the method states?) - not validation of yield accuracy.

  A  equations      core functions vs independent re-implementations of the published forms
                    (Potter et al. 1993 eq. 6-8; method document; Beck et al. 2006 eq. 3; Allen & Pereira 2009)
  B  input QC       step 01 checks: no FAIL
  C  step outputs   16-day growth rates recomputed from period_drivers; curve fit vs the 16-day values; daily sum =
                    season biomass; dry and fresh yield from biomass; season length
  D  ensemble       member count from config, weights sum to 1, district statistics recomputed, central = member
  E  calibration    calibrated district yield = benchmark; implied eps_max = eps_max x factor
  F  sensitivity    vectorised Sobol model (step 07) = model.run at grid points
  G  plausibility   ranges of fAPAR, Ts, Ws, season vs AquaCrop sowing windows, agreement with AquaCrop (information)
  H  reproducibility  optional --compare <archived outputs folder>: key results identical
  I  audit diagnostics (information; definitions in the detail column): background fAPAR share of season growth,
                    Landsat EF and ETa/ETc at full canopy, ETa vs EF Rn / lambda, 16-day table area vs crop-map area
  J  production layers: input registration and provenance, growth-curve hold-out, probability sample (field units)
  K  pixel yield model (step 05b, field units): its checks (vectorised model = field-unit code on a sample of
                    pixels), district members recomputed from the parcel values, final district value recomputed from
                    the calibrated pixel-based members
Writes: validation_checks.csv, manifest. Stops with an error if any check FAILs.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import calibration as cal
from core import config, lue, model, stats
from core import drivers as dv
from core import ensemble as en

TOL = 1e-6


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, group, name, ok, detail, info=False):
        self.rows.append({"group": group, "check": name, "status": "INFO" if info else ("PASS" if ok else "FAIL"),
                          "detail": detail})


def equations(C, cfg):
    t = np.array([0.0, 5, 10, 15, 18, 19.5, 21, 25, 30, 35])
    for topt in (18.0, 19.5, 21.0):
        t1 = 0.8 + 0.02 * topt - 0.0005 * topt ** 2                                   # Potter 1993 eq. 6
        t2 = 1.1814 / (1 + np.exp(0.2 * (topt - 10 - t))) / (1 + np.exp(0.3 * (-topt - 10 + t)))   # eq. 7
        ok = abs(lue.casa_t1(topt) - t1) < TOL and np.allclose(lue.casa_t2(t, topt), t2, atol=TOL) and \
            np.allclose(lue.ts(t, topt), np.clip(t1 * t2, 0, 1), atol=TOL)
        C.add("A equations", f"Ts = T1 x T2 (Potter et al. 1993 eq. 6-7), Topt {topt}", ok,
              f"Ts at 10/19.5/30 degC: {np.round(lue.ts(np.array([10, 19.5, 30.0]), topt), 4).tolist()}")
    x = np.array([-0.2, 0, 0.3, 0.7, 1.0, 1.4])
    ok = np.allclose(lue.ws(x, "ef"), np.clip(x, 0, 1)) and \
        np.allclose(lue.ws(x, "relative_et"), 0.5 + 0.5 * np.clip(x, 0, 1))             # Potter 1993 eq. 8
    C.add("A equations", "Ws = EF; Ws = 0.5 + 0.5 min(1, ETa/ETc) (method doc; Potter eq. 8)", ok, "bounded 0-1")
    ok = np.allclose(lue.par([10.0, 20.0], 0.48), [4.8, 9.6])
    C.add("A equations", "PAR = f x Rs (method document)", ok, "f 0.48")
    dry, fresh = lue.crop_yield([800.0], 0.8, 0.225)
    C.add("A equations", "Y_dry = HI x B; Y_fresh = Y_dry / DM (method document)",
          abs(dry[0] - 6.4) < TOL and abs(fresh[0] - 6.4 / 0.225) < TOL, f"B 800 g/m2 -> {dry[0]:.2f} / {fresh[0]:.2f} t/ha")
    cc = cfg["crop_coefficient"]
    nd = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    fc = np.clip((nd - cc["ndvi_soil"]) / (cc["ndvi_full"] - cc["ndvi_soil"]), 0, 1) ** 2               # Carlson & Ripley
    kd = np.minimum.reduce([np.ones_like(fc), cc["ml"] * fc, fc ** (1 / (1 + cc["crop_height_m"]))])   # Allen & Pereira
    C.add("A equations", "Kc = Kc_ini + (Kc_mid - Kc_ini) Kd (Allen & Pereira 2009; Carlson & Ripley 1997)",
          np.allclose(lue.crop_coefficient(nd, cc), cc["kc_ini"] + (cc["kc_mid"] - cc["kc_ini"]) * kd), "")
    p = [1.0, 12.0, 55.0, 0.1, 100.0, 0.05]
    tt = np.arange(0, 180.0)
    beck = p[0] + (p[1] - p[0]) * (1 / (1 + np.exp(-p[3] * (tt - p[2]))) + 1 / (1 + np.exp(p[5] * (tt - p[4]))) - 1)
    C.add("A equations", "double logistic (Beck et al. 2006 eq. 3)", np.allclose(dv.double_logistic(tt, p), beck), "")
    s, e = pd.to_datetime(["2025-10-01", "2025-10-16"]), pd.to_datetime(["2025-10-16", "2025-11-01"])
    k = dv.period_of_day(pd.date_range("2025-10-01", "2025-11-01"), s, e)
    C.add("A equations", "16-day periods are [start, end): every day in exactly one period",
          (np.bincount(k[k >= 0]) == [15, 16]).all() and k[-1] == -1, f"days per period {np.bincount(k[k >= 0]).tolist()}")


def audit_diagnostics(C, cfg, S, c, pdv, sea, units):
    """I - information diagnostics of the tehsil 16-day inputs (definitions in the detail text)."""
    pin = pd.read_csv(S("01_inputs", "period_inputs.csv"), parse_dates=["period"])
    pdv["ins"] = [sea.loc[u, "emergence"] <= m_ <= sea.loc[u, "season_end"] for u, m_ in zip(pdv.unit, pdv.mid)]
    bg = pin.groupby("tehsil").fapar.min()
    shares = {}
    for u, g in pdv.groupby("unit"):
        days = pd.date_range(sea.loc[u, "emergence"], sea.loc[u, "season_end"])
        k_ = pd.Series(np.searchsorted(g.period.to_numpy(), days.to_numpy(), side="right") - 1).value_counts()
        base_ = (lue.ts(g.tmean.to_numpy(), model.topt_of(g, c["topt"])) * np.clip((g.ef_l8 + g.ef_l9).to_numpy() / 2, 0, 1)
                 * g.rs.to_numpy())
        tot = sum(n_ * base_[i] * g.fapar.to_numpy()[i] for i, n_ in k_.items())
        net = sum(n_ * base_[i] * max(g.fapar.to_numpy()[i] - bg[u], 0) for i, n_ in k_.items())
        shares[u] = 100 * (1 - net / tot)
    sh = pd.Series(shares)
    C.add("I audit diagnostics", "background fAPAR share of season growth (central)", True,
          f"{sh.min():.0f}-{sh.max():.0f} % by tehsil (highest {sh.idxmax()}); background = the tehsil's minimum 16-day "
          "fAPAR (soil, residue, mixed pixels); SL2P fAPAR (method document) would separate canopy from soil", info=True)
    x_ = pdv[pdv.ins].merge(pin[["tehsil", "period", "eta_l8", "eta_l9", "rn"]].rename(columns={"tehsil": "unit"}),
                            on=["unit", "period"])
    full = x_[x_.fapar >= 0.6]
    eta = full[["eta_l8", "eta_l9"]].mean(axis=1)
    C.add("I audit diagnostics", "Landsat water signal at full canopy (in-season periods with fAPAR >= 0.6)", True,
          f"n {len(full)}: EF {((full.ef_l8 + full.ef_l9) / 2).mean():.2f}; ETa {eta.mean():.2f} vs ETc "
          f"{full.etc_mm_day.mean():.2f} mm/d (ratio {(eta / full.etc_mm_day).mean():.2f}); "
          f"corr(EF, fAPAR) in season {((x_.ef_l8 + x_.ef_l9) / 2).corr(x_.fapar):.2f}", info=True)
    ok_ = pin.ef.notna() & pin[["eta_l8", "eta_l9"]].mean(axis=1).notna() & (pin.ef > 0)
    rr = pin[["eta_l8", "eta_l9"]].mean(axis=1)[ok_] / (pin.ef[ok_] * pin.rn[ok_] / 2.45)
    C.add("I audit diagnostics", "table ETa / (EF x Rn / 2.45)", True,
          f"median {rr.median():.2f} ({rr.min():.2f}-{rr.max():.2f}): ETa is derived from EF and ERA5 Rn, "
          "not an independent water signal", info=True)
    ta = pin.groupby("tehsil").area_ha.first()
    ma = units.set_index("tehsil").weight_ha
    C.add("I audit diagnostics", "16-day table area vs crop-map polygon area (weights, production)", True,
          f"table {ta.sum():,.0f} ha vs map {ma.sum():,.0f} ha (ratio {(ta / ma.reindex(ta.index)).min():.2f}-"
          f"{(ta / ma.reindex(ta.index)).max():.2f}); weights and production use the map", info=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", "--district", dest="run", required=True, help="run config (district_season)")
    ap.add_argument("--compare", help="outputs folder of an earlier run of the same run config (reproducibility)")
    args = ap.parse_args()
    cfg = config.load(args.run)
    out = config.output_dir(cfg, "10_validate")
    S = lambda step, name: config.step_output(cfg, step, name)  # noqa: E731
    C = Checks()
    equations(C, cfg)

    # ---- B input QC
    qc = pd.read_csv(S("01_inputs", "qc_checks.csv"))
    C.add("B input QC", "step 01 checks without FAIL", not (qc.status == "FAIL").any(),
          f"{(qc.status == 'PASS').sum()} PASS, {(qc.status == 'WARN').sum()} WARN: "
          f"{'; '.join(qc[qc.status == 'WARN'].check)}")

    # ---- C step outputs (central configuration)
    c = model.central_config(cfg)
    method = config.biomass_method(cfg)
    daily = pd.read_csv(S("05_lue", "lue_daily_central.csv"), parse_dates=["date"])
    cen = pd.read_csv(S("05_lue", "central_unit.csv")).set_index("unit")
    fits = None
    if method == "fit_biomass":                      # 16-day growth rates -> double logistic (tehsil units)
        pdv = pd.read_csv(S("04_drivers", "period_drivers.csv"), parse_dates=["period", "period_end", "mid"])
        pts = pd.read_csv(S("05_lue", "growth_points_central.csv"), parse_dates=["period"])
        to = model.topt_of(pdv, c["topt"])
        g_ind = c["eps_max"] * c["f_par"] * pdv.rs * pdv.fapar * np.clip((pdv.ef_l8 + pdv.ef_l9) / 2, 0, 1) * \
            np.clip((0.8 + 0.02 * to - 0.0005 * to ** 2) * 1.1814 /
                    (1 + np.exp(0.2 * (to - 10 - pdv.tmean))) / (1 + np.exp(0.3 * (-to - 10 + pdv.tmean))), 0, 1)
        j = pdv.assign(g_ind=g_ind).merge(pts[["unit", "period", "growth_point_g_m2_d"]], on=["unit", "period"])
        dmax = float((j.g_ind - j.growth_point_g_m2_d).abs().max())
        C.add("C step outputs", "16-day growth rates = independent recomputation from period_drivers", dmax < 1e-6,
              f"{len(j)} tehsil-periods, max |diff| {dmax:.2e} g/m2/d")
        ok_pts = pts.dropna(subset=["curve_mean_g_m2_d"])
        bias = ok_pts.groupby("unit").difference.mean()
        C.add("C step outputs", "fitted curve averaged over each period: unbiased per tehsil (|mean diff| < 0.1 g/m2/d)",
              bool((bias.abs() < 0.1).all()), f"mean diff {bias.min():+.3f} to {bias.max():+.3f}; r "
              f"{ok_pts.growth_point_g_m2_d.corr(ok_pts.curve_mean_g_m2_d):.3f}")
        fits = pd.read_csv(S("04_drivers", "growth_fits.csv"))
        C.add("C step outputs", "all growth-rate fits R2 >= 0.90", bool((fits.r2 >= 0.90).all()),
              f"{len(fits)} fits, R2 {fits.r2.min():.3f}-{fits.r2.max():.3f}; bias max {fits.bias.abs().max():.1e}")
        f0 = fits[(fits.ws_form == c["ws_form"]) & np.isclose(fits.topt, c["topt"]) & np.isclose(fits.alpha, 0.5)].set_index("unit")
        t_d = (daily.date - pd.Timestamp(cfg["season"]["window"][0])).dt.days.to_numpy(float)
        beck = np.array([f0.loc[u].vmin + (f0.loc[u].vmax - f0.loc[u].vmin) * (
            1 / (1 + np.exp(-f0.loc[u].m1 * (t - f0.loc[u].n1))) + 1 / (1 + np.exp(f0.loc[u].m2 * (t - f0.loc[u].n2))) - 1)
            for u, t in zip(daily.unit, t_d)])
        dmax = float(np.abs(c["eps_max"] * c["f_par"] * np.clip(beck, 0, None) - daily.growth_curve_all_days_g_m2_d).max())
        C.add("C step outputs", "daily growth = eps_max x f x fitted curve (Beck eq. 3, recomputed)", dmax < 1e-6,
              f"max |diff| {dmax:.2e}")
    else:                                            # per-date data: daily drivers interpolated (field units)
        dr = pd.read_csv(S("04_drivers", "drivers_daily.csv"), parse_dates=["date"])
        cols_ = ["unit", "date", "rs", "tmean", "fapar", "ef_l8", "ef_l9"] + (["topt_unit"] if "topt_unit" in dr else [])
        x_ = daily.merge(dr[cols_], on=["unit", "date"], suffixes=("", "_d"))
        tc_ = "topt_unit_d" if "topt_unit_d" in x_ else "topt_unit"
        to = x_[tc_].to_numpy(float) + c["topt"] if tc_ in x_ else c["topt"]
        ts_ = np.clip((0.8 + 0.02 * to - 0.0005 * to ** 2) * 1.1814 /
                      (1 + np.exp(0.2 * (to - 10 - x_.tmean_d))) / (1 + np.exp(0.3 * (-to - 10 + x_.tmean_d))), 0, 1)
        g_ind = np.where(x_.in_season, c["eps_max"] * ts_ * np.clip((x_.ef_l8 + x_.ef_l9) / 2, 0, 1) * x_.fapar_d
                         * c["f_par"] * x_.rs_d, 0.0)
        dmax = float(np.abs(g_ind - x_.growth_g_m2_d).max())
        C.add("C step outputs", "daily growth = eps_max x Ts x Ws x fAPAR x f x Rs (recomputed from drivers_daily)",
              dmax < 1e-6, f"{len(x_):,} unit-days, max |diff| {dmax:.2e} g/m2/d")
    bsum = daily[daily.in_season].groupby("unit").growth_g_m2_d.sum()
    dmax = float((bsum - cen.biomass_g_m2).abs().max())
    C.add("C step outputs", "season biomass = sum of daily growth, emergence to season end", dmax < 1e-6,
          f"max |diff| {dmax:.2e} g/m2")
    dmax = max(float((cen.dry_t_ha - c["hi"] * cen.biomass_g_m2 * 0.01).abs().max()),
               float((cen.fresh_t_ha - cen.dry_t_ha / c["dm"]).abs().max()))
    C.add("C step outputs", "dry = HI x B x 0.01; fresh = dry / DM (central)", dmax < 1e-6, f"max |diff| {dmax:.2e} t/ha")
    sea = pd.read_csv(S("03_season", "season.csv"), parse_dates=["emergence", "season_end"]).set_index("unit")
    nd_ = daily[daily.in_season].groupby("unit").size()
    C.add("C step outputs", "days summed = season length (step 03)", bool((nd_ == sea.season_days.reindex(nd_.index)).all()),
          f"{nd_.min()}-{nd_.max()} days")

    # ---- D ensemble
    mem = pd.read_csv(S("05_lue", "members.csv"))
    grid = pd.read_csv(S("05_lue", "member_grid.csv"))
    units = pd.read_csv(S("03_season", "units.csv"))
    P_ = cfg["parameters"]
    n_exp = int(np.prod([len(P_[k]["values"]) for k in ("eps_max", "f_par", "topt", "hi", "dm")])) * \
        len(cfg["water_stress"]["forms"]) * len(cfg["season"]["ensemble_fractions"])
    C.add("D ensemble", f"{n_exp} members per unit (config grid), weights sum to 1",
          len(grid) == n_exp and abs(grid.weight.sum() - 1) < 1e-9 and (mem.groupby("unit").size() == n_exp).all(),
          f"{len(grid)} members, sum of weights {grid.weight.sum():.12f}")
    a = en.aggregate_members(mem, units.set_index("unit").weight_ha)
    a["w"] = a.member.map(grid.set_index("member").weight)
    m_ind = float(np.average(a.fresh_t_ha, weights=a.w))
    ed = pd.read_csv(S("06_ensemble", "ensemble_district.csv"))
    m_out = float(ed.fresh_t_ha_mean.iloc[0])
    C.add("D ensemble", "district LUE-ENS mean of the field sample recomputed from members", abs(m_ind - m_out) < 1e-9,
          f"{m_ind:.4f} vs {m_out:.4f} t/ha")
    key = (grid.ws_form == c["ws_form"]) & np.isclose(grid.season_p, c["season_p"]) & \
        np.logical_and.reduce([np.isclose(grid[k], c[k]) for k in ("eps_max", "f_par", "topt", "hi", "dm")])
    cm = mem[mem.member == int(grid[key].member.iloc[0])].set_index("unit").fresh_t_ha
    C.add("D ensemble", "central configuration = its ensemble member",
          float((cm - cen.fresh_t_ha.reindex(cm.index)).abs().max()) < 1e-9, "")

    # ---- E calibration
    latest = json.load(open(S("08_calibrate", "latest.json"), encoding="utf-8"))
    cal_dir = config.step_output(cfg, "08_calibrate", latest["calibration_id"])
    fac = pd.read_csv(cal_dir / "calibration_factor.csv").iloc[0]
    dist = pd.read_csv(S("09_aggregate", "district_final.csv")).iloc[0]
    pixel = config.units(cfg) == "field"                 # v0.8.0: calibration on the pixel-based district value
    m_cal = float(pd.read_csv(S("05b_pixel_model", "ensemble_district.csv")).fresh_t_ha_mean.iloc[0]) if pixel else m_out
    if fac.status == "placeholder":                  # one district benchmark: district value = benchmark by design
        C.add("E calibration", "calibrated district yield = benchmark (placeholder calibration)",
              abs(dist.final_t_ha - fac.observed_t_ha) < 0.01, f"{dist.final_t_ha:.3f} vs {fac.observed_t_ha} t/ha "
              f"({latest['calibration_id']})")
        C.add("E calibration", "factor = benchmark / LUE-ENS district mean" + (" (pixel yield model)" if pixel else ""),
              abs(fac.factor - fac.observed_t_ha / m_cal) < 1e-9, f"factor {fac.factor:.4f}")
    else:                                            # observed tehsil yields: ratio of means, leave-one-out
        o = pd.read_csv(cal_dir / "tehsil_obs_vs_sim.csv")
        area = o.area_ha.to_numpy(float) if o.area_ha.notna().all() else None
        f_ind = cal.ratio_of_means(o.observed_yield_t_ha, o.sim, area)
        C.add("E calibration", "factor = ratio of means of observed and simulated tehsil yields (Cochran 1977)",
              abs(fac.factor - f_ind) < 1e-9, f"factor {fac.factor:.4f} vs recomputed {f_ind:.4f}, {len(o)} tehsils "
              f"({latest['calibration_id']})")
        C.add("E calibration", "leave-one-out prediction for every observed tehsil",
              bool(o.calibrated_loo.notna().all()), f"{int(o.calibrated_loo.notna().sum())} of {len(o)}")
    C.add("E calibration", "implied eps_max = eps_max x factor",
          abs(fac.implied_eps_max_central - c["eps_max"] * fac.factor) < 1e-9,
          f"implied eps_max {fac.implied_eps_max_central:.3f} g/MJ")
    ev = cfg["parameters"]["eps_max"]["values"]
    inside = min(ev) <= fac.implied_eps_max_central <= max(ev)
    C.add("E calibration", f"implied eps_max inside the documented range {min(ev)}-{max(ev)} g/MJ", inside,
          f"{fac.implied_eps_max_central:.2f} g/MJ" + ("" if inside else " - outside the range: the uncalibrated level "
          "or the calibration yields are questionable"), info=True)

    # ---- F sensitivity: vectorised Sobol model = model.run at grid points
    spec = importlib.util.spec_from_file_location("s07", Path(__file__).with_name("07_sensitivity.py"))
    s07 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s07)
    d = pd.read_csv(S("04_drivers", "drivers_daily.csv"), parse_dates=["date"]).sort_values(["unit", "date"]).reset_index(drop=True)
    seasons = pd.read_csv(S("03_season", "season_grid.csv"), parse_dates=["emergence", "season_end"])
    uo = sorted(d.unit.unique())
    lo, hi = cfg["season"]["sobol_range"]
    pg = np.round(np.arange(lo, hi + 1e-9, 0.01), 2)
    masks = np.stack([model.season_mask(d, seasons, p).to_numpy().reshape(len(uo), d.date.nunique()) for p in pg])
    w = units.set_index("unit").weight_ha.reindex(uo).to_numpy(float)
    teh_ = units.set_index("unit").tehsil.reindex(uo).tolist()
    if fits is not None:
        Sg, tg, ag = s07.rate_grid(d, fits, masks, uo)
    else:
        arr = {k: d[k].to_numpy(float).reshape(len(uo), d.date.nunique())
               for k in ("rs", "tmean", "fapar", "ef_l8", "ef_l9", "ret_l8", "ret_l9")}
        arr["topt_unit"] = (d.topt_unit.to_numpy(float).reshape(len(uo), d.date.nunique()) if "topt_unit" in d
                            else np.zeros((len(uo), d.date.nunique())))
    dmax = 0.0
    t_lo, t_hi = cfg["parameters"]["topt"]["range"]          # CASA Topt: offsets, all 0
    for topt, sw, mix, sp in ((c["topt"], 0, 0.5, 0.2), (t_lo, 1, 0.3, 0.1), (t_hi, 0, 1.0, 0.3)):
        X = np.array([[c["eps_max"], c["f_par"], topt, c["hi"], c["dm"], sw, sp, mix]])   # central values of the crop file
        yd = (s07.evaluate_fit(X, Sg, tg, ag, pg, w, teh_) if fits is not None else
              s07.evaluate(X, arr, masks, pg, w, teh_))[0][0]
        y = model.run(d, seasons, {**c, "topt": topt, "ws_form": "ef" if sw == 0 else "relative_et", "season_p": sp},
                      alpha=mix, fits=fits)
        dmax = max(dmax, abs(yd - stats.area_mean(y.set_index("unit").fresh_t_ha.reindex(uo), w)))
    C.add("F sensitivity", "Sobol model (step 07) = model.run at grid points", dmax < 1e-6, f"max |diff| {dmax:.2e} t/ha")

    # ---- G plausibility (information)
    C.add("G plausibility", "daily fAPAR, Ts, Ws within 0-1", bool(daily[["fapar", "ts", "ws"]].apply(
        lambda s_: s_.between(0, 1).all()).all()), "")
    if "days_after_sowing_first" in sea:                     # optional comparison model registered (step 03)
        ac = sea[["days_after_sowing_first", "days_after_sowing_last"]]
        C.add("G plausibility", "emergence after the first AquaCrop sowing date (independent check)",
              bool((ac.days_after_sowing_first > 0).all()), f"emergence {ac.days_after_sowing_first.min()}-"
              f"{ac.days_after_sowing_first.max()} d after the first sowing, {ac.days_after_sowing_last.min()}-"
              f"{ac.days_after_sowing_last.max()} d after the last", info=not (ac.days_after_sowing_first > 0).all())
    p_cmp = config.step_output(cfg, "09_aggregate", "comparison_aquacrop.csv", required=False)
    cmp_ = pd.read_csv(p_cmp) if p_cmp.exists() else pd.DataFrame()
    if "calibrated_fresh_t_ha" in cmp_:                       # AquaCrop calibrated tehsil yield (step 09 comparison)
        r = np.corrcoef(cmp_.final_t_ha, cmp_.calibrated_fresh_t_ha)[0, 1]
        C.add("G plausibility", "tehsil pattern vs AquaCrop (calibrated to the same benchmark; not independent truth)",
              True, f"r {r:.3f}, mean |diff| {(cmp_.final_t_ha - cmp_.calibrated_fresh_t_ha).abs().mean():.2f} t/ha",
              info=True)

    # ---- H reproducibility
    if args.compare:
        old = Path(args.compare)
        for step, name, col in (("05_lue", "central_unit.csv", "fresh_t_ha"),
                                ("06_ensemble", "ensemble_tehsil.csv", "fresh_t_ha_mean"),
                                ("09_aggregate", "tehsil_final.csv", "final_t_ha")):
            a_ = pd.read_csv(old / step / name)
            b_ = pd.read_csv(S(step, name))
            k_ = "unit" if "unit" in a_ else "tehsil"
            dmax = float((a_.set_index(k_)[col] - b_.set_index(k_)[col].reindex(a_[k_])).abs().max())
            C.add("H reproducibility", f"{step}/{name} {col} reproduced (|diff| < 1e-4 t/ha; iterative curve fits "
                  f"converge to ~1e-8)", dmax < 1e-4, f"{old.name}: max |diff| {dmax:.2e} t/ha")

    # ---- J production layers (information): provenance, growth-fit hold-out, sample design
    pv = pd.read_csv(S("01_inputs", "provenance.csv"))
    req = cfg["provenance"]["required_fields"]
    und = int(sum(pv[f].astype(str).str.startswith(cfg["provenance"]["undocumented_value"]).sum() for f in req))
    C.add("J production layers", "inputs registered (sha256) and provenance documented", True,
          f"{int(pv.matches_registered.eq(True).sum())} of {len(pv)} inputs match their registered sha256; "
          f"{und} of {len(pv) * len(req)} provenance items not documented (step 01 provenance.csv)", info=True)
    if method == "fit_biomass":
        ho = pd.read_csv(S("04_drivers", "growth_fit_holdout.csv"))
        hs = ho[ho.in_season]
        C.add("J production layers", "growth-curve hold-out (leave one period out, central configuration)", True,
              f"{len(hs)} in-season periods: RMSE {np.sqrt(np.mean(hs.error ** 2)):.3f} MJ m-2 d-1 "
              f"({100 * np.sqrt(np.mean(hs.error ** 2)) / hs.rate.mean():.0f} % of the mean rate); season sum change "
              f"{hs.season_sum_change_pct.min():+.1f} to {hs.season_sum_change_pct.max():+.1f} %", info=True)
    if config.units(cfg) == "field":
        fl = pd.read_csv(S("02_fields", "fields.csv"))
        stc = pd.read_csv(S("02_fields", "cell_strata.csv")).set_index("cell")
        if "inclusion_prob" in fl and "frame_n" in stc:
            ht = fl.groupby("cell").inclusion_prob.apply(lambda p_: (1 / p_).sum())
            C.add("J production layers", "probability sample: sum of 1/pi per cell = frame size",
                  bool(np.allclose(ht, stc.frame_n.reindex(ht.index))), f"{len(fl)} fields, {len(ht)} cells")
        else:                                      # sample of step 02 v0.3.2 (kept: a season's sample is drawn once)
            C.add("J production layers", "probability sample: sum of 1/pi per cell = frame size", False,
                  f"NOT VERIFIABLE: the sample ({len(fl)} fields) was drawn by step 02 of v0.3.2, which records neither "
                  "inclusion probabilities nor frame sizes; the district sampling error uses N_h = all parcels of the "
                  "cell (upper bound). A sample drawn by step 02 of v0.5 records both", info=True)

    # ---- I audit diagnostics (information; tehsil units - they use the 16-day table)
    if config.units(cfg) == "tehsil":
        audit_diagnostics(C, cfg, S, c, pd.read_csv(S("04_drivers", "period_drivers.csv"),
                                                    parse_dates=["period", "period_end", "mid"]), sea, units)

    # ---- K pixel yield model (field units)
    if pixel:
        from core import pixels as px
        ck5 = pd.read_csv(S("05b_pixel_model", "checks.csv"))
        C.add("K pixel yield model", "step 05b checks without FAIL (incl. vectorised model = field-unit code)",
              not (ck5.status == "FAIL").any(), "; ".join(f"{r.check}: {r.status}" for r in ck5.itertuples()
                                                         if r.status in ("PASS", "FAIL", "WARN")))
        pv = pd.read_csv(S("05b_pixel_model", "parcel_values.csv"))
        cb = pd.read_csv(S("05b_pixel_model", "combos.csv"))
        g5 = pd.read_csv(S("05b_pixel_model", "member_grid.csv"))
        md = pd.read_csv(S("05b_pixel_model", "members_district.csv")).set_index("member")
        sk = [f"s_{k}" for k in cb.combo]
        v = pv[pv.status != "no estimate"]
        s_bar = (v[sk].to_numpy() * v.area_ha.to_numpy()[:, None]).sum(axis=0) / v.area_ha.sum()
        km_ = np.array([int(cb[(np.isclose(cb.topt, r.topt)) & (cb.ws_form == r.ws_form)
                               & np.isclose(cb.season_p, r.season_p)].combo.iloc[0]) for r in g5.itertuples()])
        fresh = px.yields_from_s(s_bar[None, :], g5, km_)[2][0]
        dmax = float(np.abs(fresh - md.fresh_t_ha.reindex(g5.member).to_numpy()).max())
        C.add("K pixel yield model", "district members = crop-area-weighted parcel values -> member equations "
              "(recomputed)", dmax < 1e-9, f"{len(v):,} parcels, {len(g5)} members, max |diff| {dmax:.2e} t/ha")
        f_m = pd.read_csv(cal_dir / "member_factors.csv", index_col=0).member_factor
        w_ = g5.set_index("member").weight
        fin_ind = float(np.average(md.fresh_t_ha.reindex(w_.index) * f_m.reindex(w_.index), weights=w_))
        C.add("K pixel yield model", "final district value = weighted mean of the calibrated pixel-based members "
              "(recomputed)", abs(fin_ind - dist.final_t_ha) < 1e-9, f"{fin_ind:.4f} vs {dist.final_t_ha:.4f} t/ha")
        if "stratified_estimate_t_ha" in dist and pd.notna(dist.get("ci95_low")):
            ins = bool(dist.ci95_low <= dist.final_t_ha <= dist.ci95_high)
            C.add("K pixel yield model", "pixel-based district value inside the 95 % interval of the design-based "
                  "sample estimate", ins, f"pixel-based {dist.final_t_ha:.2f} t/ha (every crop-map pixel); sample "
                  f"{dist.stratified_estimate_t_ha:.2f} t/ha, 95 % interval {dist.ci95_low:.2f}-{dist.ci95_high:.2f}",
                  info=not ins)

    # run-time and memory benchmark of the steps (from their manifests)
    bench = []
    run_dir = config.ROOT / "outputs" / cfg["run_id"]
    mfs = [m_ for m_ in sorted(run_dir.glob("[0-9][0-9]*_*/manifest.json")) if m_.parent.name != "10_validate"]
    mfs.append(cal_dir / "manifest.json")                       # step 08: the current calibration folder
    for mf in sorted(mfs, key=lambda m_: str(m_.relative_to(run_dir))):
        mj = json.load(open(mf, encoding="utf-8"))
        bench.append({"step": mj.get("step"), "run_seconds": mj.get("run_seconds"), "peak_memory_mb": mj.get("peak_memory_mb"),
                      "framework_version": mj.get("framework_version"), "run_at": mj.get("run_at")})
    pd.DataFrame(bench).to_csv(out / "benchmarks.csv", index=False)

    t = pd.DataFrame(C.rows)
    t.to_csv(out / "validation_checks.csv", index=False)
    config.write_manifest(out, "10_validate", {"outputs": config.ROOT / "outputs" / cfg["run_id"]},
                          {"validation_checks": out / "validation_checks.csv"},
                          {"pass": int((t.status == "PASS").sum()), "fail": int((t.status == "FAIL").sum())})
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(t.to_string(index=False))
    print(f"\n{(t.status == 'PASS').sum()} PASS, {(t.status == 'FAIL').sum()} FAIL, {(t.status == 'INFO').sum()} INFO "
          f"- written to {out}")
    if (t.status == "FAIL").any():
        raise SystemExit("validation FAILED")


if __name__ == "__main__":
    main()
