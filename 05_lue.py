"""Step 05 - LUE biomass and crop yield for every LUE-ENS member and the central configuration.

Members: full factorial eps_max x f x Topt x HI x DM x water-scalar form (config: Ws = EF) x season fraction,
product weights (core.ensemble.member_grid). Central configuration: central parameter values, Ws = EF (method
document), season fraction 0.20 - kept for traceability to the method document and v0.1.

v0.3 biomass_method (config):
  fit_biomass (central, approach 2): growth rate of every 16-day period g_p = eps_max * Ts(Tmean_p) * Ws_p *
      fAPAR_p * f * Rs_p (g m-2 d-1) -> double-logistic curve fitted to the 16-day values (step 04 growth_fits.csv)
      -> read every day -> summed from emergence to season end.
  fit_drivers (approach 1): daily growth from the daily driver curves of step 04; written next to approach 2.
Daily detail and monthly / 16-day biomass are written for the central configuration.
Writes: members.csv, member_grid.csv, central_unit.csv, lue_daily_central.csv, biomass_monthly_central.csv,
[growth_points_central.csv, biomass_period_central.csv], figures/fig1-fig6 (curve fitting), manifest.
"""
import numpy as np
import pandas as pd

from core import config, lue, model, plots
from core import drivers as dv


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    out = config.output_dir(cfg, "05_lue")
    config.check_writable(out)
    paths = {"drivers_daily": config.step_output(cfg, "04_drivers", "drivers_daily.csv"),
             "season_grid": config.step_output(cfg, "03_season", "season_grid.csv")}
    method = config.biomass_method(cfg)
    fits = pdv = None
    if method == "fit_biomass":
        paths["growth_fits"] = config.step_output(cfg, "04_drivers", "growth_fits.csv")
        paths["period_drivers"] = config.step_output(cfg, "04_drivers", "period_drivers.csv")
        fits = pd.read_csv(paths["growth_fits"])
        pdv = pd.read_csv(paths["period_drivers"], parse_dates=["period", "period_end", "mid"])
    d = pd.read_csv(paths["drivers_daily"], parse_dates=["date"])
    seasons = pd.read_csv(paths["season_grid"], parse_dates=["emergence", "season_end"])

    mem, grid = model.members(d, seasons, cfg, fits)
    c = model.central_config(cfg)
    cen = model.run(d, seasons, c, fits=fits)
    key = np.logical_and.reduce([np.isclose(grid[k], c[k]) if k != "ws_form" else grid[k] == c[k] for k in c])
    if key.sum() != 1:
        raise SystemExit("central configuration is not exactly one member of the grid")
    cm = mem[mem.member == int(grid[key].member.iloc[0])].set_index("unit").fresh_t_ha
    diff = float((cm - cen.set_index("unit").fresh_t_ha).abs().max())
    if diff > 1e-9:
        raise SystemExit(f"central configuration differs from its member by {diff}")
    ef_ = c["eps_max"] * c["f_par"]                                   # g DM per MJ of Rs-equivalent rate

    # daily detail of the central configuration
    x = d.copy()
    x["in_season"] = model.season_mask(d, seasons, c["season_p"])
    x["par"] = lue.par(x.rs, c["f_par"])
    x["apar"] = x.fapar * x.par
    x["ts"] = lue.ts(x.tmean, model.topt_of(x, c["topt"]))
    x["ws"] = lue.ws(model.scalar_input(x, c["ws_form"]), c["ws_form"])
    x["eps"] = c["eps_max"] * x.ts * x.ws
    x["growth_drivers_g_m2_d"] = np.where(x.in_season, x.eps * x.apar, 0.0)          # approach 1
    if fits is not None:                                                              # approach 2 (central)
        x["growth_curve_all_days_g_m2_d"] = ef_ * model.rate_daily(x, fits, c["topt"], c["ws_form"])
        x["growth_g_m2_d"] = np.where(x.in_season, x.growth_curve_all_days_g_m2_d, 0.0)
    else:
        x["growth_g_m2_d"] = x.growth_drivers_g_m2_d
    x["biomass_cum_g_m2"] = x.groupby("unit").growth_g_m2_d.cumsum()
    x["biomass_cum_drivers_g_m2"] = x.groupby("unit").growth_drivers_g_m2_d.cumsum()
    x["month"] = x.date.dt.strftime("%Y-%m")
    mon = x[x.in_season].groupby(["unit", "month"]).agg(days=("growth_g_m2_d", "size"), apar_mj_m2=("apar", "sum"),
                                                        biomass_g_m2=("growth_g_m2_d", "sum")).reset_index()
    mon["share_pct"] = 100 * mon.biomass_g_m2 / mon.groupby("unit").biomass_g_m2.transform("sum")
    s = x[x.in_season].groupby("unit")
    cen = cen.merge(pd.DataFrame({"biomass_method": method,
                                  "biomass_approach1_g_m2": s.growth_drivers_g_m2_d.sum(),
                                  "sum_apar_mj_m2": s.apar.sum(),
                                  "ts_eff": s.apply(lambda g: (g.ts * g.apar).sum() / g.apar.sum(), include_groups=False),
                                  "ws_eff": s.apply(lambda g: (g.ws * g.apar).sum() / g.apar.sum(), include_groups=False)}),
                    left_on="unit", right_index=True)

    mem.to_csv(out / "members.csv", index=False)
    grid.to_csv(out / "member_grid.csv", index=False)
    cen.to_csv(out / "central_unit.csv", index=False)
    cols = ["unit", "date", "in_season", "rs", "par", "fapar", "apar", "tmean", "ts", "ws", "eps"] + \
           (["growth_curve_all_days_g_m2_d"] if fits is not None else []) + \
           ["growth_g_m2_d", "biomass_cum_g_m2", "growth_drivers_g_m2_d", "biomass_cum_drivers_g_m2"]
    x[cols].to_csv(out / "lue_daily_central.csv", index=False, date_format="%Y-%m-%d")
    mon.to_csv(out / "biomass_monthly_central.csv", index=False)

    pts = None
    if pdv is not None:              # the 16-day growth points the curve was fitted to, and the curve over each period
        pts = pdv.copy()
        pts["par"] = lue.par(pts.rs, c["f_par"])
        pts["apar"] = pts.fapar * pts.par
        pts["ts"] = lue.ts(pts.tmean, model.topt_of(pts, c["topt"]))
        pts["ef"] = model.scalar_input(pts, "ef")
        pts["ws"] = lue.ws(model.scalar_input(pts, c["ws_form"]), c["ws_form"])
        pts["growth_point_g_m2_d"] = c["eps_max"] * pts.ts * pts.ws * pts.apar
        cm_ = []
        for u, g in pts.groupby("unit"):
            xu = x[x.unit == u]
            k = dv.period_of_day(xu.date, g.period, g.period_end)
            cm_.append(pd.Series(xu.growth_curve_all_days_g_m2_d.to_numpy()).groupby(k).mean()
                       .reindex(range(len(g))).set_axis(g.index))
        pts["curve_mean_g_m2_d"] = pd.concat(cm_)     # NaN: period after the window (curve not read there)
        pts["difference"] = pts.curve_mean_g_m2_d - pts.growth_point_g_m2_d
        pts[["unit", "period", "period_end", "mid", "rs", "par", "fapar", "apar", "tmean", "ts", "ef_l8_table",
             "ef_l9_table", "ef", "ws", "growth_point_g_m2_d", "curve_mean_g_m2_d", "difference"]] \
            .to_csv(out / "growth_points_central.csv", index=False, date_format="%Y-%m-%d %H:%M")
        f0 = fits[(fits.ws_form == c["ws_form"]) & np.isclose(fits.topt, c["topt"]) & np.isclose(fits.alpha, 0.5)]
        ok = pts.difference.notna()
        print(f"growth curve vs 16-day growth points (central): R2 {f0.r2.min():.3f}-{f0.r2.max():.3f}, RMSE "
              f"{np.sqrt(np.mean(pts.difference[ok] ** 2)):.2f} g m-2 d-1, mean diff {pts.difference[ok].mean():+.3f}")

    pp = config.step_output(cfg, "01_inputs", "period_inputs.csv", required=False)
    if config.units(cfg) == "tehsil" and pp.exists():           # biomass per 16-day period of the input table
        paths["period_inputs"] = pp
        pt = pd.read_csv(pp, parse_dates=["period", "period_end"])
        per = []
        for u, mt in pt.groupby("tehsil"):
            xu = x[x.unit == u]
            k = dv.period_of_day(xu.date, mt.period, mt.period_end)
            a = xu.assign(k=k)[xu.in_season.to_numpy() & (k >= 0)].groupby("k").agg(
                days=("growth_g_m2_d", "size"), apar_mj_m2=("apar", "sum"), biomass_g_m2=("growth_g_m2_d", "sum"))
            per.append(a.assign(unit=u, period=mt.period.to_numpy()[a.index], period_end=mt.period_end.to_numpy()[a.index]))
        per = pd.concat(per)[["unit", "period", "period_end", "days", "apar_mj_m2", "biomass_g_m2"]]
        per["share_pct"] = 100 * per.biomass_g_m2 / per.groupby("unit").biomass_g_m2.transform("sum")
        per.to_csv(out / "biomass_period_central.csv", index=False, date_format="%Y-%m-%d")
    # figures of the biomass curve fitting (central configuration)
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    tag = f"{cfg['aoi']['name']} {cfg['aoi']['season']}, central configuration"
    xt = x.rename(columns={"unit": "tehsil"})
    if pts is not None:
        units = pd.read_csv(config.step_output(cfg, "03_season", "units.csv")).set_index("unit")
        sea = seasons[np.isclose(seasons.amplitude_fraction, c["season_p"])].rename(columns={"unit": "tehsil"})
        plots.period_inputs_panels(fd / "fig1_16day_inputs.png", pdv, units.weight_ha,
                                   f"1. Inputs of each 16-day period, {cfg['aoi']['name']} {cfg['aoi']['season']}")
        plots.growth_fit_panels(fd / "fig2_growth_rate_curve_fit.png", xt, pts.rename(columns={"unit": "tehsil"}), sea,
                                f"2. Biomass growth rate of each 16-day period and the fitted daily curve "
                                f"(double logistic), {tag}")
        plots.fit_check_panels(fd / "fig3_curve_fit_check.png", pts,
                               f"3. Check of the curve fit: curve averaged over each period vs the 16-day growth rate, {tag}")
    plots.cumulative_panels(fd / "fig4_cumulative_biomass.png", xt,
                            f"4. Seasonal biomass: daily curve summed from emergence to season end, {tag}")
    if config.units(cfg) == "tehsil" and pp.exists():
        plots.period_share_bars(fd / "fig5_biomass_by_period.png", per,
                                f"5. Share of season biomass formed in each 16-day period (bar: tehsil mean; line: "
                                f"tehsil range), {cfg['aoi']['name']} {cfg['aoi']['season']}")
    plots.tehsil_biomass_yield(fd / "fig6_tehsil_biomass_yield.png", cen,
                               f"6. Season biomass and fresh yield (HI {c['hi']}, DM {c['dm']}) by tehsil, {tag}")
    old = out / "biomass_curve_central.png"                       # replaced by figures/ (old figure had two y-axes)
    if old.exists():
        old.unlink()
    config.write_manifest(out, "05_lue", paths, {"folder": out},
                          {"central": c, "members": len(grid), "biomass_method": method})
    pd.set_option("display.width", 250)
    print(f"biomass method: {method}; {len(grid)} members x {mem.unit.nunique()} units; central = member check OK")
    w = mem.groupby("unit").apply(lambda g: pd.Series(
        {"ens_mean": np.average(g.fresh_t_ha, weights=g.weight),
         **{f"mean_{f}": np.average(g[g.ws_form == f].fresh_t_ha, weights=g[g.ws_form == f].weight)
            for f in sorted(g.ws_form.unique())}}), include_groups=False)
    print(cen.set_index("unit")[["biomass_g_m2", "biomass_approach1_g_m2", "dry_t_ha", "fresh_t_ha", "ts_eff",
                                 "ws_eff"]].join(w).round(2).to_string())
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
