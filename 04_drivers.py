"""Step 04 - drivers per run unit (season window 1 Oct - 31 Mar) and the 16-day growth-rate fits.

Tehsil units, central method (biomass_method: fit_biomass):
  period_drivers.csv  the 16-day values per tehsil-period: fAPAR, Rs, Tmean (table), Landsat 8/9 EF and ETa/ETc
                      (ETa of the period / mean daily ETc of the period); a missing sensor value takes the other
                      sensor of the period, both missing -> linear in time
  growth_fits.csv     double logistic (Beck et al. 2006, eq. 3) fitted to the 14 growth rates
                      r_p = Ts(Tmean_p) Ws_p fAPAR_p Rs_p so that its mean over each period equals r_p, for every
                      Ws form x Topt x Landsat mix needed by steps 05-07 (eps_max and f scale the curve exactly)
Daily drivers (drivers_daily.csv): daily ERA5 weather scaled (Rs) / shifted (T) to the 16-day means; Kc and ETc
  (Allen & Pereira 2009; Carlson & Ripley 1997; FAO-56); heat days (Tmax > threshold in bulking, flag only); and the
  approach-1 driver curves - fAPAR double logistic, EF and ETa/ETc smoothing splines (Reinsch 1967) with smoothing set
  to the Landsat 8-9 noise - used only for the step-07 alternatives and quality flags, never for LUE-ENS.
  growth_fit_holdout.csv  leave-one-period-out hold-out of the central fit: prediction of each left-out period and
                      the change of the season sum without it
Field units: per-overpass values, linear daily interpolation (approach 1; no growth fits) - core.drivers.field_drivers,
the function step 05b runs for the pixels it checks its vectorised model with.
A unit with only one Landsat sensor uses that sensor for both (sensor_fallback.csv).
Caveat: Landsat ETa is a clear-sky (overpass-day) value while ETc covers all days; ETa/ETc is an index of relative
water use, not an exact stress ratio (one reason the two water-scalar forms are kept as ensemble members).
Writes: drivers_daily.csv, drivers_summary.csv, sensor_fallback.csv, [period_drivers.csv, growth_fits.csv,
curve_fits.csv, period_check.csv, adjustment_factors.csv, curve_fits.png], manifest.
"""
import numpy as np
import pandas as pd

from core import config, lue, model, plots
from core import drivers as dv


def growth_rate_fits(cfg, m, d, t0):
    """Approach 2 (tehsil units): 16-day drivers per tehsil-period with Landsat gaps filled, and the double-logistic
    fit of the growth rate r_p = Ts Ws fAPAR Rs for every (Ws form, Topt, Landsat mix) needed by steps 05-07."""
    rows = []
    for u, mt in m.groupby("tehsil"):
        mt = mt.sort_values("period").reset_index(drop=True)
        du = d[d.unit == u]
        k = dv.period_of_day(du.date, mt.period, mt.period_end)
        etc_p = pd.Series(du.etc.to_numpy()).groupby(k).mean()
        etc_p = pd.Series(np.arange(len(mt))).map(etc_p).to_numpy()     # NaN: period after the window (no ETo)
        ef8, ef9 = dv.fill_sensors(mt.ef_l8, mt.ef_l9, mt.mid)
        r8, r9 = dv.fill_sensors(mt.eta_l8 / etc_p, mt.eta_l9 / etc_p, mt.mid)
        rows.append(pd.DataFrame({"unit": u, "period": mt.period, "period_end": mt.period_end, "mid": mt.mid,
                                  "fapar": mt.fapar, "rs": mt.rs, "tmean": mt.tmean,
                                  "ef_l8_table": mt.ef_l8, "ef_l9_table": mt.ef_l9, "ef_l8": ef8, "ef_l9": ef9,
                                  "etc_mm_day": etc_p, "ret_l8": np.clip(r8, 0, 1.5), "ret_l9": np.clip(r9, 0, 1.5)}))
    pdv = pd.concat(rows, ignore_index=True)
    if "topt_unit" in d:                                     # CASA Topt of the unit (the Topt grid is then an offset)
        pdv["topt_unit"] = pdv.unit.map(d.groupby("unit").topt_unit.first())
    P, cb = cfg["parameters"], cfg["curve_fit"]["biomass"]
    lo, hi = P["topt"]["range"]
    topts = sorted(set(np.round(np.arange(lo, hi + 1e-9, cb["topt_step"]), 4)) | {round(v, 4) for v in P["topt"]["values"]})
    mixes = np.round(np.arange(0, 1 + 1e-9, cb["mix_step"]), 4)
    combos = [(f, t, a) for f in ("ef", "relative_et") for t in topts for a in mixes] + \
             [(f, t, 0.5) for f in ("casa_ef", "none") for t in topts]
    out = []
    for u, g in pdv.groupby("unit"):
        prev = {}
        for f, t, a in combos:
            r = model.period_rate(g, t, f, a)
            p, info = dv.fit_double_logistic(g.period, g.period_end, r, t0, scale=True, vmax_hi=cb["vmax_hi"],
                                             x0=prev.get(f))
            prev[f] = p
            out.append({"unit": u, "ws_form": f, "topt": t, "alpha": a, "t0": t0, "vmin": p[0], "vmax": p[1],
                        "n1": p[2], "m1": p[3], "n2": p[4], "m2": p[5], **info})
    return pdv, pd.DataFrame(out)


def growth_fit_holdout(cfg, pdv, seas, t0):
    """Leave-one-period-out hold-out of the central growth-rate fit (Ws = EF, central Topt, equal Landsat mix): each
    period is left out, the double logistic refitted on the others and the left-out period mean predicted. Also the
    change of the season sum (emergence -> season end, central threshold) when that period is left out."""
    cb, topt = cfg["curve_fit"]["biomass"], cfg["parameters"]["topt"]["central"]
    rows = []
    for u, g in pdv.groupby("unit"):
        g = g.sort_values("period").reset_index(drop=True)
        r = model.period_rate(g, topt, "ef", 0.5)
        s0, s1 = seas.loc[u, "emergence"], seas.loc[u, "season_end"]
        days = pd.date_range(s0, s1, freq="D")
        x = np.asarray(dv._day_number(days), float) - float(dv._day_number([t0])[0])

        def season_sum(p):
            return float(np.clip(dv.double_logistic(x, p), 0, None).sum())

        p_all, _ = dv.fit_double_logistic(g.period, g.period_end, r, t0, scale=True, vmax_hi=cb["vmax_hi"])
        s_all = season_sum(p_all)
        for i in range(len(g)):
            if np.isnan(r[i]):
                continue
            ri = r.copy()
            ri[i] = np.nan
            p_i, _ = dv.fit_double_logistic(g.period, g.period_end, ri, t0, scale=True, vmax_hi=cb["vmax_hi"], x0=p_all)
            dd = pd.date_range(g.period[i], g.period_end[i] - pd.Timedelta(days=1), freq="D")
            xi = np.asarray(dv._day_number(dd), float) - float(dv._day_number([t0])[0])
            pred = float(dv.double_logistic(xi, p_i).mean())
            in_season = bool((g.period_end[i] > s0) and (g.period[i] <= s1))
            rows.append({"unit": u, "period": g.period[i], "in_season": in_season, "rate": float(r[i]),
                         "predicted_left_out": pred, "error": pred - float(r[i]),
                         "season_sum_change_pct": 100 * (season_sum(p_i) / s_all - 1)})
    return pd.DataFrame(rows)


def tehsil_drivers(cfg, units, w, th, nd, seas, m, rel, sig_ef, days, t0):
    """Daily drivers of tehsil units: daily weather and fAPAR adjusted / fitted to the 16-day table, Kc and ETc,
    Landsat EF and ETa/ETc smoothing splines per sensor (noise = Landsat 8-9 difference). Field units:
    core.drivers.field_drivers. Returns (daily table, Topt table or None, missing sensors, weather factors, curve
    fits, ETa/ETc noise, its pairs)."""
    cc, hs, wcfg, cf = cfg["crop_coefficient"], cfg["heat_stress"], cfg["weather"], cfg.get("curve_fit", {})
    ce = cf["ef_eta"]
    # ---- pass 1: weather, fAPAR, Kc, ETc; Landsat values per sensor (ETa/ETc at the observation dates)
    base, lsat, factors, fits, missing_sensor = {}, {}, [], [], []
    for u in units.unit:
        wu = w[(w.unit == u) & w.date.isin(days)].sort_values("date")
        if len(wu) != len(days):
            raise SystemExit(f"{u}: weather has {len(wu)} of {len(days)} days")
        rs, tmean, tmax = wu.rs.to_numpy(), ((wu.tmin + wu.tmax) / 2).to_numpy(), wu.tmax.to_numpy()
        ndvi = nd[nd.unit == u].set_index("date").ndvi_smooth.reindex(days).to_numpy()
        r = {"unit": u, "date": days}
        # daily shape, 16-day level of the primary table
        mt = m[m.tehsil == u].sort_values("period").reset_index(drop=True)
        ps, pe = mt.period, mt.period_end
        rs, frs = dv.adjust_to_periods(days, rs, ps, pe, mt.rs, wcfg["rs_adjust"])
        tmean, ftm = dv.adjust_to_periods(days, tmean, ps, pe, mt.tmean, wcfg["tmean_adjust"])
        tmax, ftx = dv.adjust_to_periods(days, tmax, ps, pe, mt.tmax, wcfg["tmax_adjust"])
        factors.append(pd.DataFrame({"unit": u, "period": ps, "period_end": pe, "rs_ratio": frs,
                                     "tmean_delta_c": ftm, "tmax_delta_c": ftx}).dropna(subset=["rs_ratio"]))
        p, info = dv.fit_double_logistic(ps, pe, mt.fapar, t0)
        fapar = dv.eval_double_logistic(days, p, t0)
        fits.append({"unit": u, "variable": "fapar", "method": "double_logistic", **info,
                     "vmin": p[0], "vmax": p[1], "greenup_mid": t0 + pd.Timedelta(days=float(p[2])),
                     "greenup_rate_d": p[3], "senescence_mid": t0 + pd.Timedelta(days=float(p[4])),
                     "senescence_rate_d": p[5], "curve_peak": float(fapar.max()),
                     "curve_peak_date": days[int(np.argmax(fapar))]})
        r["fapar_perdate_ndvi"] = lue.fapar_from_ndvi(ndvi, rel.slope, rel.intercept)   # v0.2.1 path
        r["fapar_period_linear"] = np.clip(dv.interpolate_dates(days, mt.mid, mt.fapar), 0, 1)
        kc = lue.crop_coefficient(ndvi, cc)
        etc = kc * wu.eto.to_numpy()
        etc_s = pd.Series(etc, index=days)
        r.update({"rs": rs, "tmean": tmean, "tmax": tmax, "eto": wu.eto.to_numpy(), "ndvi": ndvi, "fapar": fapar,
                  "kc": kc, "etc": etc})
        base[u] = r
        sensors = [s for s in ("L8", "L9") if th[(th.unit == u) & (th.sensor == s)].ef.notna().any()]
        if not sensors:
            raise SystemExit(f"{u}: no Landsat EF (step 03 should have excluded it)")
        for s in ("L8", "L9"):
            src = s if s in sensors else sensors[0]                      # one sensor missing: use the other
            if src != s:
                missing_sensor.append({"unit": u, "missing": s, "used_instead": src})
            t = th[(th.unit == u) & (th.sensor == src)].sort_values("date").copy()
            if (t.representative == "period").all():                    # 16-day ETa vs mean ETc of the period
                k = dv.period_of_day(days, t.period, t.period_end)
                etc_p = pd.Series(etc).groupby(k).mean()
                etc_ref = pd.Series(np.arange(len(t))).map(etc_p).to_numpy()   # NaN: period outside the window
            else:                                                         # overpass ETa vs ETc of that day
                etc_ref = t.date.map(etc_s).to_numpy()
            t["ret"] = t.eta_mm_day.to_numpy() / etc_ref
            lsat[(u, s)] = t
    sig_ret = n_ret = None
    pr = pd.concat([lsat[(u, "L8")][["date", "ret"]].assign(unit=u) for u in units.unit]).merge(
        pd.concat([lsat[(u, "L9")][["date", "ret"]].assign(unit=u) for u in units.unit]), on=["unit", "date"])
    fb = {x["unit"] for x in missing_sensor}                          # copies of the other sensor: not a pair
    pr = pr[~pr.unit.isin(fb)]
    sig_ret, n_ret = dv.sensor_noise(pr.ret_x, pr.ret_y)

    # ---- pass 2: daily EF and ETa/ETc curves per sensor
    rows = []
    for u in units.unit:
        r = base[u]
        for s in ("L8", "L9"):
            t, sl = lsat[(u, s)], s.lower()
            lin_ef = np.clip(dv.interpolate_dates(days, t.date, t.ef), 0, 1)
            lin_ret = np.clip(dv.interpolate_dates(days, t.date, t.ret), 0, 1.5)
            for var, sig, bounds, lin in (("ef", sig_ef, cf["ef_bounds"], lin_ef),
                                          ("ret", sig_ret, cf["ret_bounds"], lin_ret)):
                fn, info = dv.fit_smoothing_spline(t.date, t[var], sig, ce["min_obs"])
                r[f"{var}_{sl}"] = dv.eval_spline(days, fn, info, *bounds)
                r[f"{var}_lin_{sl}"] = lin
                fits.append({"unit": u, "variable": f"{var}_{sl}", "method": info["method"],
                             "n_obs": info["n_obs"], "rmse": info.get("rmse"), "lam": info.get("lam"),
                             "noise_sigma": sig})
        s0 = seas.loc[u]
        r["in_season"] = (days >= s0.emergence) & (days <= s0.season_end)
        bulk0 = s0.emergence + pd.Timedelta(days=hs["bulking_start_days_after_emergence"])
        r["heat_day"] = (days >= bulk0) & (days <= s0.season_end) & (r["tmax"] > hs["tmax_threshold_c"])
        rows.append(pd.DataFrame(r))
    d = pd.concat(rows, ignore_index=True)
    topt_tab = None
    if cfg["temperature_scalar"]["topt"] == "casa_ndvi_peak_month":   # Topt per unit from its own NDVI peak month
        topt_tab = model.casa_topt(d, seas)
        d["topt_unit"] = d.unit.map(topt_tab.set_index("unit").topt_c)
    d["ef"] = (d.ef_l8 + d.ef_l9) / 2
    d["ret"] = (d.ret_l8 + d.ret_l9) / 2
    return d, topt_tab, missing_sensor, factors, fits, sig_ret, n_ret


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    out = config.output_dir(cfg, "04_drivers")
    config.check_writable(out)
    mode = config.units(cfg)
    paths = {"units": config.step_output(cfg, "03_season", "units.csv"),
             "weather_daily": config.step_output(cfg, "01_inputs", "weather_daily.csv"),
             "obs_thermal": config.step_output(cfg, "01_inputs", "obs_thermal.csv"),
             "obs_ndvi": config.step_output(cfg, "01_inputs", "obs_ndvi.csv"),
             "ndvi_daily": config.step_output(cfg, "03_season", "ndvi_daily.csv"),
             "ndvi_observations": config.step_output(cfg, "03_season", "ndvi_observations.csv"),
             "season": config.step_output(cfg, "03_season", "season.csv")}
    if mode == "tehsil":
        paths["period_inputs"] = config.step_output(cfg, "01_inputs", "period_inputs.csv")
        paths["fapar_relation"] = config.step_output(cfg, "01_inputs", "fapar_relation.csv")
    units = pd.read_csv(paths["units"])
    w = pd.read_csv(paths["weather_daily"], parse_dates=["date"])
    th = pd.read_csv(paths["obs_thermal"], parse_dates=["date"] + (["period", "period_end"] if mode == "tehsil" else []))
    nd = pd.read_csv(paths["ndvi_daily"], parse_dates=["date"])
    seas = pd.read_csv(paths["season"], parse_dates=["emergence", "season_end", "peak_date"]).set_index("unit")
    days = pd.date_range(*cfg["season"]["window"], freq="D")
    t0 = days[0]
    cf = cfg.get("curve_fit", {})
    if mode == "tehsil":
        m = pd.read_csv(paths["period_inputs"], parse_dates=["period", "period_end", "mid"])
        rel = pd.read_csv(paths["fapar_relation"]).set_index("relation").loc[cfg["fapar"]["relation"]]
        sig_ef, n_ef = dv.sensor_noise(m.ef_l8, m.ef_l9)
        ce = cf["ef_eta"]
        d, topt_tab, missing_sensor, factors, fits, sig_ret, n_ret = tehsil_drivers(cfg, units, w, th, nd, seas, m, rel,
                                                                                    sig_ef, days, t0)
    else:                                                    # field units: the same function as step 11 (every parcel)
        ob = pd.read_csv(paths["obs_ndvi"], parse_dates=["date"])
        st = pd.read_csv(paths["ndvi_observations"], parse_dates=["date"])
        sl2p = "fapar_sl2p" in ob
        d, topt_tab, missing_sensor = dv.field_drivers(cfg, units, w, th, nd, seas, ob if sl2p else None,
                                                       st if sl2p else None)
        sig_ret = None
    if topt_tab is not None:
        topt_tab.to_csv(out / "topt_casa.csv", index=False)
        print(f"Topt (CASA, mean air temperature of the NDVI-peak month): {topt_tab.topt_c.min():.1f}-"
              f"{topt_tab.topt_c.max():.1f} degC over {len(topt_tab)} units, peak months "
              f"{topt_tab.peak_month.value_counts().sort_index().to_dict()}")

    ins = d[d.in_season]
    g = ins.groupby("unit")
    summ = pd.DataFrame({"season_days": g.size(), "rs": g.rs.mean(), "tmean": g.tmean.mean(), "fapar": g.fapar.mean(),
                         "ef": g.ef.mean(), "eta_over_etc": g.ret.mean(), "kc": g.kc.mean(), "eto_mm": g.eto.sum(),
                         "etc_mm": g.etc.sum(), "ws_ef": g.ef.mean(),
                         "ws_relative_et": g.ret.apply(lambda x: float(lue.ws(x, "relative_et").mean())),
                         "heat_days": d.groupby("unit").heat_day.sum()}).reset_index()
    d.to_csv(out / "drivers_daily.csv", index=False, date_format="%Y-%m-%d")
    summ.to_csv(out / "drivers_summary.csv", index=False)
    pd.DataFrame(missing_sensor, columns=["unit", "missing", "used_instead"]).to_csv(out / "sensor_fallback.csv",
                                                                                   index=False)
    if missing_sensor:
        print(f"{len(missing_sensor)} unit-sensor combinations without data: the other Landsat sensor used")
    if mode == "tehsil":
        pd.concat(factors).to_csv(out / "adjustment_factors.csv", index=False, date_format="%Y-%m-%d")
        fit = pd.DataFrame(fits)
        fit.to_csv(out / "curve_fits.csv", index=False, date_format="%Y-%m-%d")
        pdv, gfit = growth_rate_fits(cfg, m, d, t0)
        pdv.to_csv(out / "period_drivers.csv", index=False, date_format="%Y-%m-%d %H:%M")
        gfit.to_csv(out / "growth_fits.csv", index=False, date_format="%Y-%m-%d")
        ho = growth_fit_holdout(cfg, pdv, seas, t0)
        ho.to_csv(out / "growth_fit_holdout.csv", index=False, date_format="%Y-%m-%d")
        hs = ho[ho.in_season]
        print(f"growth-fit hold-out (leave one period out, central, in-season periods): RMSE "
              f"{np.sqrt(np.mean(hs.error ** 2)):.3f} of rates {hs.rate.mean():.3f} MJ m-2 d-1 (mean); season sum "
              f"change {hs.season_sum_change_pct.min():+.1f} to {hs.season_sum_change_pct.max():+.1f} %")
        # the daily curves averaged over each 16-day period (inside the window) vs the table
        chk = []
        for u, mt in m.groupby("tehsil"):
            du = d[d.unit == u]
            k = dv.period_of_day(du.date, mt.period, mt.period_end)
            for col, tcol in (("fapar", "fapar"), ("ef_l8", "ef_l8"), ("ef_l9", "ef_l9")):
                cm = pd.Series(du[col].to_numpy()).groupby(k).mean().drop(-1, errors="ignore")
                tv = mt.reset_index(drop=True)[tcol]
                chk.append(pd.DataFrame({"unit": u, "variable": col, "period": mt.period.to_numpy()[cm.index],
                                         "curve_mean": cm.to_numpy(), "table": tv.to_numpy()[cm.index]}))
        chk = pd.concat(chk)
        chk["difference"] = chk.curve_mean - chk.table
        chk.to_csv(out / "period_check.csv", index=False, date_format="%Y-%m-%d")
        for v, c in chk.dropna(subset=["table"]).groupby("variable"):
            print(f"{v}: daily curve vs 16-day table, r {c.curve_mean.corr(c.table):.3f}, RMSE "
                  f"{np.sqrt(np.mean(c.difference ** 2)):.3f}, mean diff {c.difference.mean():+.3f} ({len(c)} periods)")
        print(f"Landsat noise (sd(L8-L9)/sqrt2): EF {sig_ef:.3f} ({n_ef} pairs), ETa/ETc {sig_ret:.3f} ({n_ret} pairs)")
        lr = fit[fit.method == "linear"]
        if len(lr):
            print(f"{len(lr)} unit-variable curves with < {ce['min_obs']} values: linear interpolation\n{lr.to_string()}")
        sw = seas.reset_index()[["unit", "emergence", "season_end"]].rename(columns={"unit": "tehsil"})
        plots.curve_panels(out / "curve_fits.png", d.rename(columns={"unit": "tehsil"}),
                           m.rename(columns={"mid": "date"}), sw,
                           f"Daily curves fitted to the 16-day tehsil means, {cfg['aoi']['name']} {cfg['aoi']['season']}")
    config.write_manifest(out, "04_drivers", paths, {"folder": out},
                          {"units": mode, "noise_sigma_ef": sig_ef if mode == "tehsil" else None,
                           "noise_sigma_ret": sig_ret})
    pd.set_option("display.width", 250)
    print(f"daily rows {len(d)} ({d.unit.nunique()} units x {len(days)} days); in season {len(ins)}")
    if mode == "tehsil":
        fp = fit[fit.variable == "fapar"][["unit", "rmse", "r2", "bias", "greenup_mid", "senescence_mid",
                                           "curve_peak", "curve_peak_date"]].copy()
        for c in ("greenup_mid", "senescence_mid", "curve_peak_date"):
            fp[c] = pd.to_datetime(fp[c]).dt.strftime("%d %b %Y")
        print(fp.round(3).to_string(index=False))
    print(summ.round(3).head(20).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
