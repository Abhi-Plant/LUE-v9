"""Step 03 - crop season per run unit from the Sentinel-2 NDVI time series.

Per unit: quality filter (clear share of crop pixels), removal of short negative spikes (fog, haze, cloud edges),
daily interpolation and Savitzky-Golay smoothing; emergence and season end at amplitude fractions 0.10-0.30
(step 0.01: Sobol grid; the LUE-ENS members use season.ensemble_fractions). Quality flags per unit: clear
observations in the quality window (run config), despiked dates, end of season reached. Independent check: AquaCrop sowing windows.
Usable unit: >= min_used_obs clear dates, NDVI amplitude >= min_amplitude, Landsat EF available. Tehsil units:
the run stops otherwise; field units: the field is excluded and its cell's area goes to the other fields of the cell
in proportion to their parcel area (ratio estimator; cells without any usable field are reported as not represented).
Season end: found only if NDVI fell after the peak by >= min_amplitude, else set to the window end and flagged.
Censoring flags: start (pre-season minimum not observed), peak (at the edge of the search window), end.
Observed crop calendar (run config inputs.observed_calendar: unit, emergence, season_end from field logs): errors
in days (bias, MAE, RMSE) in season_vs_observed[_stats].csv.
Writes: units.csv (run units of steps 04-09), excluded_units.csv, ndvi_observations.csv, ndvi_daily.csv,
season.csv (central), season_grid.csv, season_quality.csv, season_ndvi.png, manifest.
"""
import numpy as np
import pandas as pd

from core import config, plots, sampling
from core import season as ss


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    out = config.output_dir(cfg, "03_season")
    config.check_writable(out)
    sc, cl = cfg["season"], cfg["ndvi_cleaning"]
    p_obs = config.step_output(cfg, "01_inputs", "obs_ndvi.csv")
    p_u = config.step_output(cfg, "01_inputs", "units.csv")
    nd = pd.read_csv(p_obs, parse_dates=["date"])
    units = pd.read_csv(p_u)
    start = pd.Timestamp(sc["window"][0]) - pd.Timedelta(days=sc["lead_days"])   # smoothing support at window start
    nd = nd[(nd.date >= start) & (nd.date <= pd.Timestamp(sc["window"][1]))]
    lo, hi = sc["sobol_range"]
    grid = sorted(set(np.round(np.arange(lo, hi + 1e-9, 0.01), 2)) | set(sc["ensemble_fractions"]) |
                  {sc["amplitude_fraction"]})

    p_th = config.step_output(cfg, "01_inputs", "obs_thermal.csv")
    th = pd.read_csv(p_th)
    has_thermal = th[th.ef.notna()].groupby("unit").size()
    obs, daily, seasons, quality, excluded = ss.unit_seasons(nd, units.unit, has_thermal, sc, cl, grid, start)
    excl = pd.DataFrame(excluded, columns=["unit", "reason"])
    if config.units(cfg) == "tehsil" and len(excl):
        raise SystemExit(f"tehsil units cannot be excluded silently - check the inputs:\n{excl.to_string(index=False)}")
    if not obs:
        raise SystemExit("no usable unit")
    # valid units; field units: design weight re-distributed to the remaining fields of the cell
    valid = units[~units.unit.isin(excl.unit)].copy()
    if "cell" in units and len(excl):
        cell_area = units.groupby("cell").weight_ha.sum()
        valid["weight_ha"] = sampling.ratio_weights(valid.cell, valid.area_ha, cell_area)
        lost = sorted(set(units.cell) - set(valid.cell))
        excl = pd.concat([excl, pd.DataFrame([{"unit": f"cell {c}", "reason": f"all fields excluded: "
                                                f"{cell_area[c]:.1f} ha not represented"} for c in lost])])
    obs, daily = pd.concat(obs), pd.concat(daily)
    seas = pd.DataFrame(seasons)
    seas["amplitude_fraction"] = seas.amplitude_fraction.round(2)
    qual = pd.DataFrame(quality)
    central = seas[np.isclose(seas.amplitude_fraction, sc["amplitude_fraction"])].reset_index(drop=True)
    qual = qual.merge(central[["unit", "start_flag", "peak_flag", "end_flag", "season_days"]], on="unit")

    # optional independent check: sowing windows of a crop model by tehsil (run config inputs.aquacrop_tehsil;
    # emergence should follow sowing by ~1-4 weeks)
    p_ac = config.optional_input(cfg, "aquacrop_tehsil")
    central = central.merge(units[["unit", "tehsil"]], on="unit")
    if p_ac is not None:
        ac = pd.read_csv(p_ac, parse_dates=["sowing_first", "sowing_last"])[["tehsil", "sowing_first", "sowing_last"]]
        central = central.merge(ac, on="tehsil", how="left")
        central["days_after_sowing_first"] = (central.emergence - central.sowing_first).dt.days
        central["days_after_sowing_last"] = (central.emergence - central.sowing_last).dt.days

    # validation against an observed crop calendar (field logs), when the run config gives one
    inputs_used = {"obs_ndvi": p_obs, "obs_thermal": p_th, "units": p_u}
    if p_ac is not None:
        inputs_used["aquacrop_tehsil"] = p_ac
    if cfg["aoi"]["inputs"].get("observed_calendar"):
        p_cal = config.input_path(cfg, "observed_calendar")
        inputs_used["observed_calendar"] = p_cal
        oc = pd.read_csv(p_cal, parse_dates=[c for c in ("sowing", "emergence", "season_end")
                                             if c in pd.read_csv(p_cal, nrows=0).columns])
        cmp_cal = central[["unit", "emergence", "season_end"]].merge(oc, on="unit", suffixes=("", "_observed"))
        rows = []
        for v in ("emergence", "season_end"):
            if f"{v}_observed" in cmp_cal:
                e = (cmp_cal[v] - cmp_cal[f"{v}_observed"]).dt.days.dropna()
                cmp_cal[f"{v}_error_days"] = (cmp_cal[v] - cmp_cal[f"{v}_observed"]).dt.days
                rows.append({"date": v, "n": len(e), "bias_days": e.mean(), "mae_days": e.abs().mean(),
                             "rmse_days": float(np.sqrt((e ** 2).mean())) if len(e) else np.nan})
        cmp_cal.to_csv(out / "season_vs_observed.csv", index=False, date_format="%Y-%m-%d")
        pd.DataFrame(rows).to_csv(out / "season_vs_observed_stats.csv", index=False)
        print("season dates vs observed calendar:\n", pd.DataFrame(rows).round(1).to_string(index=False))
    obs.to_csv(out / "ndvi_observations.csv", index=False, date_format="%Y-%m-%d")
    daily.to_csv(out / "ndvi_daily.csv", index=False, date_format="%Y-%m-%d")
    central.to_csv(out / "season.csv", index=False, date_format="%Y-%m-%d")
    seas.to_csv(out / "season_grid.csv", index=False, date_format="%Y-%m-%d")
    qual.to_csv(out / "season_quality.csv", index=False)
    valid.to_csv(out / "units.csv", index=False)               # run units of steps 04-09 (re-weighted if excluded)
    excl.to_csv(out / "excluded_units.csv", index=False)
    show = sorted(daily.unit.unique())
    if len(show) > 20:
        show = sorted(np.random.default_rng(1).choice(show, 20, replace=False))
    w0 = pd.Timestamp(sc["window"][0])
    plots.season_panels(out / "season_ndvi.png", daily[daily.unit.isin(show) & (daily.date >= w0)].rename(columns={"unit": "tehsil"}),
                        obs[obs.unit.isin(show) & (obs.date >= w0)].rename(columns={"unit": "tehsil"}),
                        central.loc[central.unit.isin(show), ["unit", "emergence", "season_end", "season_days"]]
                        .rename(columns={"unit": "tehsil"}),
                        f"Sentinel-2 NDVI of crop fields and crop season, {cfg['aoi']['name']} {cfg['aoi']['season']} "
                        f"(threshold {int(100 * sc['amplitude_fraction'])} % of amplitude)")
    config.write_manifest(out, "03_season", inputs_used, {"folder": out})
    print(f"units: {len(valid)} valid, {len(excl)} excluded" + (f"\n{excl.to_string(index=False)}" if len(excl) else ""))
    pd.set_option("display.width", 250)
    cols = ["unit", "emergence", "season_end", "season_days", "peak_date", "peak_ndvi", "base_start_ndvi",
            "end_flag", "days_after_sowing_first", "days_after_sowing_last"]
    cols = [c for c in cols if c in central.columns]          # sowing columns exist only with inputs.aquacrop_tehsil
    print(central[cols].head(20).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    alt = seas[seas.amplitude_fraction.isin(sc["ensemble_fractions"])].pivot(index="unit", columns="amplitude_fraction",
                                                                            values="season_days")
    print("\nseason length (days) by ensemble fraction:\n", alt.head(20).to_string())
    print("\nquality:\n", qual.head(20).to_string(index=False))
    print(f"\n{len(grid)} amplitude fractions x {daily.unit.nunique()} units; written to {out}")


if __name__ == "__main__":
    main()
