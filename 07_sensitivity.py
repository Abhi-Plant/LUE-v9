"""Step 07 - global sensitivity (Sobol) and reported method alternatives.

Sobol (SALib; Saltelli et al. 2010), continuous ranges from config: eps_max, f, Topt, HI, DM, [water-scalar form
switch, only when LUE-ENS has two forms], season amplitude fraction (0.10-0.30, step-03 grid), Landsat 8/9 mix.
Two outputs:
  district  - uncalibrated district fresh yield (what calibration must fix)
  pattern   - tehsil yield / district yield (what limits tehsil accuracy after a one-factor calibration);
              indices per tehsil and their mean.
v0.3 biomass_method fit_biomass: the model is the fitted 16-day growth-rate curve (step 04 growth_fits.csv); Sobol
uses the fit grid (bilinear in Topt and Landsat mix). Approach 1 (driver curves) is reported as an alternative.
Reported alternatives (central configuration, one change each; never in LUE-ENS): Ws = 0.5 + 0.5 EF, Ws = 1,
EF of Landsat 8 or 9 only, fAPAR = 1.257 NDVI - 0.161 (Bastiaanssen & Ali 2003 eq. 4), [tehsil units (v0.3): fAPAR from the per-date NDVI (v0.2.1
path), no curve fit (fAPAR and EF / ETa linear between the 16-day means), 16-day (CSV-only) calculation, 16-day
calculation with median inputs]; one-at-a-time parameter table.
Landsat 8 / 9 choice per tehsil (sensor_choice_tehsil.csv): tehsil / district yield with L8 only, L9 only and the
equal mix - the range of the largest pattern factor, reported next to the LUE-ENS P10-P90 in step 09.
Writes: sobol_indices.csv, alternatives.csv, sensor_choice_tehsil.csv, parameters_one_at_a_time.csv, sobol.png,
[method_comparison.csv, fig7_method_comparison.png], manifest.
"""
import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from core import config, lue, model, plots, stats

LABELS = {"eps_max": "eps_max (light-use efficiency)", "f_par": "PAR fraction f", "topt": "Topt (temperature)",
          "hi": "harvest index", "dm": "dry matter of the harvested part", "ws_switch": "water-scalar form (EF | relative ET)",
          "season_p": "season threshold (emergence / season end)", "sensor_mix": "Landsat 8 / 9 mix (EF, ETa)"}


def evaluate(X, arr, masks, p_grid, w_unit, unit_tehsil, chunk=256):
    """Vectorised model: X rows = [eps, f, topt, hi, dm, ws_switch, season_p, sensor_mix] ->
    district fresh yield and tehsil / district ratios."""
    tehs = sorted(set(unit_tehsil))
    T = np.array([[1.0 if t == u else 0.0 for u in unit_tehsil] for t in tehs])    # tehsil x unit
    Wt = T * w_unit[None, :]
    dist, rel = [], []
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        ts = lue.ts(arr["tmean"][None], arr["topt_unit"][None] + x[:, 2, None, None])   # CASA: Topt per unit + offset
        a = x[:, 7, None, None]
        ef = a * arr["ef_l8"][None] + (1 - a) * arr["ef_l9"][None]
        ret = a * arr["ret_l8"][None] + (1 - a) * arr["ret_l9"][None]
        use_ef = (x[:, 5] < 0.5)[:, None, None]
        ws = np.where(use_ef, np.clip(ef, 0, 1), 0.5 + 0.5 * np.clip(ret, 0, 1))
        pidx = np.abs(p_grid[None, :] - x[:, 6, None]).argmin(axis=1)
        S = (masks[pidx] * ts * ws * arr["fapar"][None] * arr["rs"][None]).sum(axis=2)          # eval x unit
        fresh = x[:, 0, None] * x[:, 1, None] * S * x[:, 3, None] * lue.G_M2_TO_T_HA / x[:, 4, None]
        d = (fresh * w_unit[None]).sum(axis=1) / w_unit.sum()
        t = (fresh @ Wt.T) / Wt.sum(axis=1)[None]
        dist.append(d)
        rel.append(t / d[:, None])
    return np.concatenate(dist), np.vstack(rel), tehs


def rate_grid(d, fits, masks, uorder):
    """Approach 2: season sums S[form, topt, mix, p, unit] of the fitted growth-rate curves on the step-04 fit grid."""
    g = fits[fits.ws_form.isin(["ef", "relative_et"])]
    tg, ag = np.sort(g.topt.unique()), np.sort(g.alpha.unique())
    S = np.zeros((2, len(tg), len(ag), masks.shape[0], len(uorder)))
    for i, f in enumerate(("ef", "relative_et")):
        for j, t in enumerate(tg):
            for k, a in enumerate(ag):
                r = model.rate_daily(d, fits, t, f, a).reshape(len(uorder), -1)
                S[i, j, k] = (masks * r[None]).sum(axis=2)
    return S, tg, ag


def evaluate_fit(X, S, tg, ag, p_grid, w_unit, unit_tehsil):
    """Approach 2 version of evaluate(): S bilinear in Topt and Landsat mix between fitted grid points, nearest
    season fraction; eps_max, f, HI, DM scale exactly."""
    tehs = sorted(set(unit_tehsil))
    T = np.array([[1.0 if t == u else 0.0 for u in unit_tehsil] for t in tehs])
    Wt = T * w_unit[None, :]
    form = (X[:, 5] >= 0.5).astype(int)
    pidx = np.abs(p_grid[None, :] - X[:, 6, None]).argmin(axis=1)
    if len(tg) == 1:                                          # one Topt grid point (CASA: offset 0)
        tg = np.array([tg[0], tg[0] + 1.0])
        S = np.concatenate([S, S], axis=1)
    jt = np.clip(np.searchsorted(tg, X[:, 2]) - 1, 0, len(tg) - 2)
    ka = np.clip(np.searchsorted(ag, X[:, 7]) - 1, 0, len(ag) - 2)
    wt = ((X[:, 2] - tg[jt]) / (tg[jt + 1] - tg[jt]))[:, None]
    wa = ((X[:, 7] - ag[ka]) / (ag[ka + 1] - ag[ka]))[:, None]
    Sv = ((1 - wt) * (1 - wa) * S[form, jt, ka, pidx] + wt * (1 - wa) * S[form, jt + 1, ka, pidx]
          + (1 - wt) * wa * S[form, jt, ka + 1, pidx] + wt * wa * S[form, jt + 1, ka + 1, pidx])
    fresh = X[:, 0, None] * X[:, 1, None] * Sv * X[:, 3, None] * lue.G_M2_TO_T_HA / X[:, 4, None]
    dist = (fresh * w_unit[None]).sum(axis=1) / w_unit.sum()
    rel = ((fresh @ Wt.T) / Wt.sum(axis=1)[None]) / dist[:, None]
    return dist, rel, tehs


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    out = config.output_dir(cfg, "07_sensitivity")
    config.check_writable(out)
    mode = config.units(cfg)
    paths = {"drivers_daily": config.step_output(cfg, "04_drivers", "drivers_daily.csv"),
             "season_grid": config.step_output(cfg, "03_season", "season_grid.csv"),
             "units": config.step_output(cfg, "03_season", "units.csv")}
    if mode == "tehsil":
        paths["period_inputs"] = config.step_output(cfg, "01_inputs", "period_inputs.csv")
        paths["fapar_relation"] = config.step_output(cfg, "01_inputs", "fapar_relation.csv")
    method = config.biomass_method(cfg)
    fits = None
    if method == "fit_biomass":
        paths["growth_fits"] = config.step_output(cfg, "04_drivers", "growth_fits.csv")
        fits = pd.read_csv(paths["growth_fits"])
    d = pd.read_csv(paths["drivers_daily"], parse_dates=["date"]).sort_values(["unit", "date"])
    seasons = pd.read_csv(paths["season_grid"], parse_dates=["emergence", "season_end"])
    units = pd.read_csv(paths["units"]).set_index("unit")
    P, so = cfg["parameters"], cfg["sobol"]
    uorder = sorted(d.unit.unique())
    nday = d.date.nunique()
    arr = {c: d[c].to_numpy(float).reshape(len(uorder), nday) for c in
           ("rs", "tmean", "fapar", "ef_l8", "ef_l9", "ret_l8", "ret_l9")}
    arr["topt_unit"] = (d.topt_unit.to_numpy(float).reshape(len(uorder), nday) if "topt_unit" in d
                        else np.zeros((len(uorder), nday)))          # crop-value Topt: the parameter itself
    lo, hi = cfg["season"]["sobol_range"]
    p_grid = np.round(np.arange(lo, hi + 1e-9, 0.01), 2)
    masks = np.stack([model.season_mask(d, seasons, p).to_numpy().reshape(len(uorder), nday) for p in p_grid])
    w_unit = units.weight_ha.reindex(uorder).to_numpy(float)
    unit_tehsil = units.tehsil.reindex(uorder).tolist()

    two_forms = len(cfg["water_stress"]["forms"]) > 1          # v0.3.2: LUE-ENS uses Ws = EF only
    model_cols = ["eps_max", "f_par", "topt", "hi", "dm", "ws_switch", "season_p", "sensor_mix"]
    fixed = {k: P[k]["central"] for k in ("eps_max", "f_par", "topt", "hi", "dm") if P[k]["range"][0] == P[k]["range"][1]}
    if not two_forms:
        fixed["ws_switch"] = 0.0                                  # v0.3.2: LUE-ENS uses Ws = EF only
    names = [k for k in model_cols if k not in fixed]           # CASA Topt is data, not a sampled parameter
    rng_ = {**{k: P[k]["range"] for k in ("eps_max", "f_par", "topt", "hi", "dm")}, "ws_switch": [0, 1],
            "season_p": [lo, hi], "sensor_mix": [0, 1]}
    problem = {"num_vars": len(names), "names": names, "bounds": [rng_[k] for k in names]}
    X = sobol_sample.sample(problem, so["base_samples"], calc_second_order=False, seed=so["seed"])
    Xm = np.column_stack([X[:, names.index(k)] if k in names else np.full(len(X), fixed[k]) for k in model_cols])
    chunk = int(np.clip(2_000_000 // (len(uorder) * nday), 4, 256))   # bounded memory (field units: many units)
    if fits is None:
        Yd, Yr, tehs = evaluate(Xm, arr, masks, p_grid, w_unit, unit_tehsil, chunk=chunk)
    else:
        d = d.reset_index(drop=True)
        Sg, tg, ag = rate_grid(d, fits, masks, uorder)
        Yd, Yr, tehs = evaluate_fit(Xm, Sg, tg, ag, p_grid, w_unit, unit_tehsil)
    rows = []

    def analyse(y, output, group):
        r = sobol_analyze.analyze(problem, y, calc_second_order=False, num_resamples=so["bootstrap"], seed=so["seed"])
        for k, n in enumerate(names):
            rows.append({"output": output, "group": group, "factor": n, "label": LABELS[n], "S1": r["S1"][k],
                         "S1_conf": r["S1_conf"][k], "ST": r["ST"][k], "ST_conf": r["ST_conf"][k]})

    analyse(Yd, "district fresh yield (uncalibrated)", "district")
    for j, t in enumerate(tehs):
        if np.ptp(Yr[:, j]) > 1e-12:
            analyse(Yr[:, j], "tehsil / district (pattern)", t)
    sob = pd.DataFrame(rows)
    pat = sob[sob.output.str.startswith("tehsil")].groupby(["factor", "label"])[["S1", "S1_conf", "ST", "ST_conf"]] \
        .mean().reset_index().assign(output="tehsil / district (pattern)", group="mean over tehsils")
    sob = pd.concat([sob, pat], ignore_index=True)

    # ---- reported alternatives (central configuration, one change each)
    c = model.central_config(cfg)
    cen = model.run(d, seasons, c, fits=fits)
    W = units.weight_ha
    a1 = "approach 1 (drivers fitted daily), " if fits is not None else ""

    def dmean(y):
        return stats.area_mean(y.set_index("unit").fresh_t_ha, W.reindex(y.unit))

    base = dmean(cen)
    alts = [("Ws = 0.5 + 0.5 EF (CASA form with EF)", model.run(d, seasons, {**c, "ws_form": "casa_ef"}, fits=fits)),
            ("Ws = 1 (no water stress, upper bound)", model.run(d, seasons, {**c, "ws_form": "none"}, fits=fits)),
            ("Ws = 0.5 + 0.5 ETa/ETc (Potter eq. 8; not in LUE-ENS)", model.run(d, seasons, {**c, "ws_form": "relative_et"}, fits=fits)),
            ("EF Landsat 8 only", model.run(d, seasons, c, alpha=1.0, fits=fits)),
            ("EF Landsat 9 only", model.run(d, seasons, c, alpha=0.0, fits=fits))]
    if fits is not None:
        alts.append(("approach 1: fAPAR / EF curves fitted daily, then daily biomass", model.run(d, seasons, c)))
    if mode == "tehsil":
        rel = pd.read_csv(paths["fapar_relation"]).set_index("relation").loc["bastiaanssen_ali_2003"]
        d["fapar_ba"] = lue.fapar_from_ndvi(d.ndvi, rel.slope, rel.intercept)
        m = pd.read_csv(paths["period_inputs"], parse_dates=["period", "period_end", "mid"])
        lin = d.assign(ef_l8=d.ef_lin_l8, ef_l9=d.ef_lin_l9, ret_l8=d.ret_lin_l8, ret_l9=d.ret_lin_l9)
        alts += [(a1 + "fAPAR = 1.257 NDVI - 0.161 (Bastiaanssen & Ali 2003 eq. 4) from per-date NDVI", model.run(d, seasons, c, fapar_col="fapar_ba")),
                 (a1 + "fAPAR from per-date S2 NDVI, relation fitted on the table (v0.2.1 path)",
                  model.run(d, seasons, c, fapar_col="fapar_perdate_ndvi")),
                 (a1 + "no curve fit: fAPAR linear between 16-day means", model.run(d, seasons, c, fapar_col="fapar_period_linear")),
                 (a1 + "no curve fit: EF linear between 16-day means", model.run(lin, seasons, c)),
                 (a1 + "no curve fit: fAPAR and EF linear", model.run(lin, seasons, c, fapar_col="fapar_period_linear")),
                 ("16-day calculation, no curve (each period's rate x its season days)", model.period_method(m, seasons, c)),
                 ("16-day calculation, median fAPAR and EF", model.period_method(
                     m.assign(fapar=m.fapar_median, ef=m.ef_median), seasons, c))]
    alt = pd.DataFrame([{"alternative": a, "district_fresh_t_ha": dmean(y)} for a, y in alts])
    alt["change_pct_vs_central"] = 100 * (alt.district_fresh_t_ha / base - 1)

    # Landsat 8 / 9 choice per tehsil (pattern: tehsil / district yield), central configuration. LUE-ENS keeps the
    # equal mix; the sensor choice is the largest pattern factor in Sobol, so its range is reported per tehsil.
    def tehsil_rel(y):
        y = y.set_index("unit").fresh_t_ha
        w, teh = W.reindex(y.index), units.tehsil.reindex(y.index)
        return ((y * w).groupby(teh).sum() / w.groupby(teh).sum()) / stats.area_mean(y, w)

    sens = pd.DataFrame({"rel_mix_0_5": tehsil_rel(cen),
                         "rel_l8_only": tehsil_rel(model.run(d, seasons, c, alpha=1.0, fits=fits)),
                         "rel_l9_only": tehsil_rel(model.run(d, seasons, c, alpha=0.0, fits=fits))})
    sens["l8_only_vs_mix_pct"] = 100 * (sens.rel_l8_only / sens.rel_mix_0_5 - 1)
    sens["l9_only_vs_mix_pct"] = 100 * (sens.rel_l9_only / sens.rel_mix_0_5 - 1)
    sens = sens.rename_axis("tehsil").reset_index()

    oat = []
    for k in model.central_config(cfg):
        if k in ("ws_form", "season_p"):
            continue
        lo_, hi_ = min(P[k]["values"]), max(P[k]["values"])
        rl = dmean(model.run(d, seasons, {**c, k: lo_}, fits=fits))
        rh = dmean(model.run(d, seasons, {**c, k: hi_}, fits=fits))
        oat.append({"parameter": k, "low": lo_, "high": hi_, "result_at_low": rl, "result_at_high": rh,
                    "swing_pct_of_central": 100 * abs(rh - rl) / base})
    oat = pd.DataFrame(oat)

    if fits is not None and mode == "tehsil":          # fig 7: does the biomass curve fitting change the result?
        pick = [("approach 1: fAPAR / EF curves fitted daily, then daily biomass",
                 "Approach 1: fAPAR and EF curves fitted daily, then daily biomass"),
                (a1 + "no curve fit: fAPAR and EF linear", "Approach 1 without curves: fAPAR and EF linear"),
                (a1 + "fAPAR from per-date S2 NDVI, relation fitted on the table (v0.2.1 path)",
                 "Approach 1, fAPAR from per-date NDVI (v0.2.1 route)"),
                ("16-day calculation, no curve (each period's rate x its season days)",
                 "No curve: each 16-day growth rate x its season days")]
        a = alt.set_index("alternative")
        tab = pd.DataFrame([{"method": "Biomass curve fitting (approach 2, v3 central)", "district_fresh_t_ha": base,
                             "change_pct": 0.0}] +
                           [{"method": lab, "district_fresh_t_ha": a.loc[k, "district_fresh_t_ha"],
                             "change_pct": a.loc[k, "change_pct_vs_central"]} for k, lab in pick])
        tab.to_csv(out / "method_comparison.csv", index=False)
        plots.method_comparison(out / "fig7_method_comparison.png", tab, tab.method.iloc[0],
                                f"7. District fresh yield by calculation method (central configuration, "
                                f"uncalibrated), {cfg['aoi']['name']} {cfg['aoi']['season']}")
    sob.to_csv(out / "sobol_indices.csv", index=False)
    alt.to_csv(out / "alternatives.csv", index=False)
    sens.to_csv(out / "sensor_choice_tehsil.csv", index=False)
    oat.to_csv(out / "parameters_one_at_a_time.csv", index=False)
    order = sob[sob.group == "district"].sort_values("ST").factor.tolist()
    panels = [(f"District fresh yield, uncalibrated (N runs {len(X):,})",
               sob[sob.group == "district"].set_index("factor").loc[order].reset_index()),
              ("Tehsil / district yield ratio (pattern), mean of tehsils",
               sob[sob.group == "mean over tehsils"].set_index("factor").loc[order].reset_index())]
    plots.sobol_bars(out / "sobol.png", panels,
                     f"What drives the LUE yield estimate - Sobol sensitivity, {cfg['aoi']['name']} {cfg['aoi']['season']}")
    config.write_manifest(out, "07_sensitivity", paths, {"folder": out},
                          {"sobol_runs": len(X), "problem": {k: v for k, v in problem.items()}})
    pd.set_option("display.width", 250)
    print(f"Sobol: {len(X):,} model runs, {len(names)} factors")
    print(sob[sob.group.isin(["district", "mean over tehsils"])].pivot_table(index="factor", columns="group",
                                                                              values=["S1", "ST"]).round(3).to_string())
    print(f"\ncentral district {base:.2f} t/ha")
    print(alt.round(2).to_string(index=False))
    print(sens.round(3).to_string(index=False))
    print(oat.round(2).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
