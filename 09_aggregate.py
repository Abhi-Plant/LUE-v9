"""Step 09 - final products: tehsil, district and ERA5-Land cell tables, quality flags, maps and the results workbook.

Final product: the current calibration of step 08 (LUE-Y2-provisional with a placeholder benchmark, LUE-Y2 with
observed yields); next to it the uncalibrated LUE-ENS and the central configuration of the method document.
Levels (v0.8.0, field units): tehsil, district and ERA5-Land cell from the pixel yield model of step 05b - yield of
every 10 m crop-map pixel, mean per parcel, crop-area-weighted to cell, tehsil and district (members aggregated
first, then calibrated member-wise). The field sample (steps 01-07) gives next to it the design-based stratified
district estimate with its sampling error, the Sobol and variance analysis and the Landsat 8 / 9 choice. Tehsil
units: cell values are the parcel-area-weighted mix of the tehsil values (inputs are tehsil means - stated on the map).
Quality flags per tehsil: clear Sentinel-2 dates in the quality window, despiked dates, Landsat 8/9 EF difference, heat days,
season end reached, mask vs official area.
Writes: tehsil_final.csv, district_final.csv, cell_final.csv, quality_flags.csv, comparison_aquacrop.csv,
yield_map.png, cell_yield_map.png, 09_LUE_results.xlsx, manifest.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from core import calibration as calib
from core import config, plots, sampling, stats
from core import ensemble as en


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    out = config.output_dir(cfg, "09_aggregate")
    config.check_writable(out)
    aoi, mode, q = cfg["aoi"], config.units(cfg), cfg["ensemble"]["quantiles"]
    S = lambda step, name, req=True: config.step_output(cfg, step, name, req)   # noqa: E731
    paths = {"units": S("03_season", "units.csv"), "qc_checks": S("01_inputs", "qc_checks.csv"),
             "season": S("03_season", "season.csv"), "season_quality": S("03_season", "season_quality.csv"),
             "drivers_daily": S("04_drivers", "drivers_daily.csv"), "drivers_summary": S("04_drivers", "drivers_summary.csv"),
             "central_unit": S("05_lue", "central_unit.csv"),
             "ensemble_tehsil": S("05b_pixel_model", "ensemble_tehsil.csv"),
             "ensemble_district": S("05b_pixel_model", "ensemble_district.csv"),
             "members_tehsil": S("05b_pixel_model", "members_tehsil.csv"),
             "members_district": S("05b_pixel_model", "members_district.csv"),
             "members_cell": S("05b_pixel_model", "members_cell.csv"),
             "ensemble_cell": S("05b_pixel_model", "ensemble_cell.csv"),
             "pixel_checks": S("05b_pixel_model", "checks.csv"),
             "members_unit": S("06_ensemble", "members_unit.csv"),
             "variance_shares": S("06_ensemble", "variance_shares.csv"),
             "sobol": S("07_sensitivity", "sobol_indices.csv"), "alternatives": S("07_sensitivity", "alternatives.csv"),
             "oat": S("07_sensitivity", "parameters_one_at_a_time.csv"),
             "sensor_choice": S("07_sensitivity", "sensor_choice_tehsil.csv"),
             "boundaries": config.input_path(cfg, "boundaries"), "crop_mask": config.input_path(cfg, "crop_mask")}
    if config.optional_input(cfg, "aquacrop_tehsil") is not None:   # optional comparison model (never merged)
        paths["aquacrop_tehsil"] = config.optional_input(cfg, "aquacrop_tehsil")
    units = pd.read_csv(paths["units"])
    cal = calib.latest(cfg)
    product = cal["factor"]["product"] if cal is not None else "LUE-ENS (uncalibrated)"
    ens_t = pd.read_csv(paths["ensemble_tehsil"])
    ens_d = pd.read_csv(paths["ensemble_district"]).iloc[0]
    mem_t = pd.read_csv(paths["members_tehsil"])
    f_m = cal["member_factors"] if cal is not None else pd.Series(1.0, index=mem_t.member.unique())

    def final_stats(members, level):
        rows = []
        x = members.assign(v=members.fresh_t_ha * members.member.map(f_m))
        for g, y in x.groupby(level):
            s = en.member_stats(y.v, y.weight, q)
            rows.append({level: g, "final_t_ha": s["mean"], "final_p10": s["p10"], "final_p90": s["p90"]})
        return pd.DataFrame(rows)

    # ---- tehsil table (pixel yield model, step 05b)
    tehsil_area = ens_t.set_index("tehsil").weight_ha                     # crop-map area of the parcels with a value
    fin = final_stats(mem_t, "tehsil").merge(
        ens_t[["tehsil", "fresh_t_ha_mean", "fresh_t_ha_p10", "fresh_t_ha_p90"]].rename(columns={
            "fresh_t_ha_mean": "lue_ens_t_ha", "fresh_t_ha_p10": "lue_ens_p10", "fresh_t_ha_p90": "lue_ens_p90"}), on="tehsil")
    fin["central_config_t_ha"] = fin.tehsil.map(ens_t.set_index("tehsil").central_t_ha)
    fin["modelled_area_share"] = fin.tehsil.map(ens_t.set_index("tehsil").modelled_area_ha / tehsil_area)
    fin["pixels"] = fin.tehsil.map(ens_t.set_index("tehsil").pixels)
    fin["sampled_fields"] = fin.tehsil.map(units.groupby("tehsil").size()).fillna(0).astype(int)
    sens = pd.read_csv(paths["sensor_choice"]).set_index("tehsil")        # Landsat 8 / 9 choice (step 07)
    fin["final_l8_only_t_ha"] = fin.final_t_ha * (1 + fin.tehsil.map(sens.l8_only_vs_mix_pct) / 100)
    fin["final_l9_only_t_ha"] = fin.final_t_ha * (1 + fin.tehsil.map(sens.l9_only_vs_mix_pct) / 100)
    fin["area_ha"] = fin.tehsil.map(tehsil_area)
    fin["production_t"] = fin.final_t_ha * fin.area_ha
    fin.insert(1, "product", product)

    # ---- quality flags per tehsil
    sq = pd.read_csv(paths["season_quality"]).merge(units[["unit", "tehsil", "weight_ha"]], on="unit")
    dr = pd.read_csv(paths["drivers_daily"], parse_dates=["date"]).merge(units[["unit", "tehsil"]], on="unit")
    ins = dr[dr.in_season]
    efd = ins.assign(d=(ins.ef_l8 - ins.ef_l9).abs()).groupby("tehsil").d.mean()
    summ = pd.read_csv(paths["drivers_summary"]).merge(units[["unit", "tehsil", "weight_ha"]], on="unit")
    wavg = lambda df, c: df.groupby("tehsil").apply(lambda g: stats.area_mean(g[c], g.weight_ha), include_groups=False)  # noqa: E731
    qf = pd.DataFrame({"tehsil": fin.tehsil})
    qf["clear_s2_dates_quality_window"] = qf.tehsil.map(wavg(sq, "clear_obs_quality_window"))
    qf["despiked_dates"] = qf.tehsil.map(wavg(sq, "despiked_dates"))
    qf["season_end_reached_share"] = qf.tehsil.map(sq.assign(ok=(sq.end_flag == "ok").astype(float))
                                                   .groupby("tehsil").ok.mean())
    for fl in ("start_flag", "peak_flag"):                     # censoring of the NDVI cycle (step 03)
        qf[fl.replace("_flag", "_observed_share")] = qf.tehsil.map(sq.assign(ok=(sq[fl] == "ok").astype(float))
                                                                   .groupby("tehsil").ok.mean())
    qf["ef_l8_l9_abs_diff"] = qf.tehsil.map(efd)
    qf["heat_days"] = qf.tehsil.map(wavg(summ, "heat_days"))
    qf["ws_ef"] = qf.tehsil.map(wavg(summ, "ws_ef"))
    qf["ws_relative_et"] = qf.tehsil.map(wavg(summ, "ws_relative_et"))
    qs = cfg["quality_flags"]                                   # processing settings (config), no literature values
    th = {"clear_s2_dates_quality_window": ("<", qs["min_clear_s2_dates"]),
          "ef_l8_l9_abs_diff": (">", qs["max_ef_l8_l9_abs_diff"]), "heat_days": (">", qs["max_heat_days"]),
          "season_end_reached_share": ("<", 1.0), "start_observed_share": ("<", 1.0),
          "peak_observed_share": ("<", 1.0)}
    qf["flags"] = qf.apply(lambda r: "; ".join(f"{k} {r[k]:.2f}" for k, (op, v) in th.items()
                                               if (r[k] < v if op == "<" else r[k] > v)) or "none", axis=1)
    no_field = qf.tehsil.isin(fin.tehsil[fin.sampled_fields == 0])        # flags come from the sampled fields
    qf.loc[no_field, "flags"] = "no sampled field in the tehsil (quality flags are computed from the field sample)"
    fin = fin.merge(qf[["tehsil", "flags"]], on="tehsil")

    # ---- district
    dist = {"district": aoi["name"], "season": aoi["season"], "product": product,
            "p10_p90_meaning": cfg["ensemble"]["spread_label"],
            "lue_ens_t_ha": ens_d.fresh_t_ha_mean, "lue_ens_p10": ens_d.fresh_t_ha_p10, "lue_ens_p90": ens_d.fresh_t_ha_p90,
            "central_config_t_ha": ens_d.central_t_ha, "area_ha": ens_d.weight_ha,
            "modelled_area_share": ens_d.modelled_area_ha / ens_d.weight_ha, "pixels": int(ens_d.pixels),
            "modelled_pixels": int(ens_d.modelled_pixels), "level_from": "pixel yield model (step 05b): pixel -> "
                                                                         "parcel -> district, crop-area weighted"}
    fd = final_stats(pd.read_csv(paths["members_district"]), "district").iloc[0]
    dist.update({"final_t_ha": fd.final_t_ha, "final_p10": fd.final_p10, "final_p90": fd.final_p90})
    dist["production_t"] = dist["final_t_ha"] * dist["area_ha"]
    if cal is not None:
        f = cal["factor"]
        dist.update({"calibration_id": cal["calibration_id"], "benchmark_t_ha": f.observed_t_ha,
                     "common_factor": f.factor, "implied_eps_max_central": f.implied_eps_max_central,
                     "implied_eps_max_range": f.implied_eps_max_range, "warning": f.warning})
    if mode == "field":                                        # stratified district estimate and sampling error
        mu = pd.read_csv(paths["members_unit"])
        fu = final_stats(mu, "unit").merge(units, on="unit")
        strata = pd.read_csv(S("02_fields", "cell_strata.csv")).set_index("cell")
        # A sample drawn by step 02 of v0.3.2 is kept (a season's sample is drawn once): it names the crop area
        # cell_<crop>_ha and records no frame size. Its frame is at most every parcel of the cell (frame_parcels,
        # as v0.3.2 aggregated): the smallest finite-population correction, so the sampling SE is an upper bound.
        old_area = [c for c in strata if c.startswith("cell_") and c.endswith("_ha") and c != "cell_crop_ha"]
        if "cell_crop_ha" not in strata and len(old_area) == 1:
            strata = strata.rename(columns={old_area[0]: "cell_crop_ha"})
        frame_note = "frame_n of step 02"
        if "frame_n" not in strata:
            strata["frame_n"] = strata.frame_parcels
            frame_note = "sample of step 02 v0.3.2 without frame size: N_h = all parcels of the cell (SE is an upper bound)"
        est, se, n1 = sampling.stratified_estimate(fu.final_t_ha, fu.cell, fu.area_ha,     # separate ratio estimator
                                                   strata.cell_crop_ha, strata.frame_n)   # N_h = frame size
        dist.update({"stratified_estimate_t_ha": est, "sampling_se_t_ha": se, "ci95_low": est - 1.96 * se,   # 1.96: two-sided 95 % normal quantile
                     "ci95_high": est + 1.96 * se, "cells_with_one_field": n1, "sampling_frame": frame_note,
                     "stratified_estimate_note": "design-based estimate from the field sample (steps 01-07), "
                                                 "independent of the pixel-based final value"})
    else:
        dist["sampling_se_t_ha"] = "not applicable - tehsil units use tehsil means of the whole mask (no sample)"
    ref_area = aoi["benchmark"].get("area_ha")
    dist["official_area_ha"] = ref_area
    dist["official_area_source"] = aoi["benchmark"].get("area_source")
    if ref_area:
        dist["production_on_official_area_t"] = dist["final_t_ha"] * ref_area
    dist["area_note"] = "production uses the satellite crop mask; the official area is older and smaller - " \
                        "an area accuracy assessment is needed before production is quoted"
    district = pd.DataFrame([dist])

    # ---- cells and maps
    step = aoi["inputs"]["era5_cell_deg"]
    bd = gpd.read_file(paths["boundaries"])
    tb = bd[bd.DISTRICT == aoi["inputs"]["boundary_district"]].to_crs(4326).copy()
    cm = gpd.read_file(paths["crop_mask"], layer=aoi["inputs"]["crop_mask_layer"])
    pt = cm.to_crs(cm.estimate_utm_crs()).geometry.representative_point().to_crs(4326)
    frame = pd.DataFrame({"lon": pt.x, "lat": pt.y, "area_ha": cm.area_ha, "tehsil": cm[aoi["inputs"]["crop_mask_tehsil_field"]]})
    names = frame.groupby(frame.tehsil.str.upper().str.replace(" ", "_")).tehsil.first()
    tb["tehsil"] = tb.SUB_DIST.map(names).fillna(tb.SUB_DIST.str.title())
    frame["cell_lat"] = (frame.lat / step).round() * step
    frame["cell_lon"] = (frame.lon / step).round() * step
    frame["cell"] = [f"{a:.1f}_{o:.1f}" for a, o in zip(frame.cell_lat, frame.cell_lon)]
    if mode == "field":
        cf = final_stats(pd.read_csv(paths["members_cell"]), "cell")
        cell_note = ("Cell value = every crop-map pixel of the cell: yield per 10 m pixel, mean per parcel, "
                     "crop-area-weighted to the cell (members aggregated first)")
    else:
        frame["final"] = frame.tehsil.map(fin.set_index("tehsil").final_t_ha)
        cf = frame.groupby("cell").apply(lambda x: pd.Series({"final_t_ha": stats.area_mean(x.final, x.area_ha)}),
                                         include_groups=False).reset_index()
        cell_note = ("Cell value = crop-area-weighted mix of the tehsil values of the mapped parcels in the cell\n"
                     "(LUE inputs are tehsil means: cell differences show the tehsil mix, not within-tehsil variation)")
    cstat = frame.groupby("cell").agg(cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"),
                                      parcels=("area_ha", "size"), cell_crop_ha=("area_ha", "sum"),
                                      tehsils=("tehsil", lambda x: ", ".join(sorted(set(x))))).reset_index()
    cell_final = cstat.merge(cf, on="cell", how="left")
    cells = gpd.GeoDataFrame(cell_final, crs=4326, geometry=[box(o - step / 2, a - step / 2, o + step / 2, a + step / 2)
                                                             for a, o in zip(cell_final.cell_lat, cell_final.cell_lon)])
    pts = None
    if mode == "field":
        pts = pd.read_csv(S("02_fields", "fields.csv"))[["lon", "lat"]]
    is_prov = cal is not None and cal["factor"].status == "placeholder"
    head = (f"{aoi['name']} district, {product}: {dist['final_t_ha']:.1f} t/ha fresh"
            + (f" (= benchmark {cal['factor'].observed_t_ha:g} t/ha, member-wise calibration)" if is_prov else "")
            + f"; uncalibrated LUE-ENS {dist['lue_ens_t_ha']:.1f} (P10-P90 {dist['lue_ens_p10']:.1f}-{dist['lue_ens_p90']:.1f})")
    ev = cfg["parameters"]["eps_max"]["values"]
    warn = ((cal["factor"].warning + "\n") if is_prov else "") + \
        (f"Implied eps_max {dist.get('implied_eps_max_central', np.nan):.2f} g/MJ (documented {min(ev)}-{max(ev)}). "
         if cal is not None else "") + "Tehsil P10-P90: pattern uncertainty after calibration (season threshold, Topt); " \
        "Landsat 8 / 9 choice in tehsil_final (final_l8_only / final_l9_only)"
    ref_area_txt = f"{ref_area:,} ha" if ref_area else "not given"
    tv = {r.tehsil: (f"{r.final_t_ha:.1f} t/ha", f"P10-P90 {r.final_p10:.1f}-{r.final_p90:.1f}") for r in fin.itertuples()}
    plots.grid_map(out / "yield_map.png", tb, cells, frame,
                   f"{aoi['name']}: tehsil crop yield estimate ({product}, fresh, t/ha), {aoi['season']}\n"
                   f"Light-use-efficiency ensemble (Sentinel-2 fAPAR, Landsat EF / ETa, ERA5)"
                   + (" - workflow test, not validated" if is_prov else ""),
                   f"{head}\nProduction {dist['production_t']:,.0f} t on {dist['area_ha']:,.0f} ha crop mask "
                   f"(official area {ref_area_txt})\nCells: mapped crop area of the crop-mask "
                   f"parcels ({frame.area_ha.sum():,.0f} ha in {len(frame):,} parcels)\n{warn}",
                   tehsil_values=tv, points=pts, point_label="Sampled field")
    plots.grid_map(out / "cell_yield_map.png", tb, cells, frame,
                   f"{aoi['name']}: crop yield by ERA5-Land grid cell ({product}, fresh, t/ha), {aoi['season']}\n"
                   f"Light-use-efficiency ensemble, {cells.final_t_ha.notna().sum()} cells"
                   + (" - workflow test, not validated" if is_prov else ""),
                   f"{cell_note}\n{head}\n{warn}", cell_value="final_t_ha",
                   cell_bins=plots.value_bins(cells.final_t_ha), cell_unit="t/ha",
                   cell_legend=f"ERA5-Land cell (0.1 deg): {product}", cell_labels=True, points=pts,
                   point_label="Sampled field")

    # ---- optional comparison with AquaCrop (independent model, calibrated to the same district benchmark): not validation
    cmp_ = cmp_stats = None
    if "aquacrop_tehsil" in paths:
        ac = pd.read_csv(paths["aquacrop_tehsil"])[["tehsil", "calibrated_fresh_t_ha", "potential_fresh_t_ha"]]
        cmp_ = fin[["tehsil", "final_t_ha", "lue_ens_t_ha"]].merge(ac, on="tehsil")
        cmp_["lue_minus_aquacrop"] = cmp_.final_t_ha - cmp_.calibrated_fresh_t_ha
        cmp_stats = {"pearson_r": cmp_.final_t_ha.corr(cmp_.calibrated_fresh_t_ha),
                     "spearman_rho": cmp_.final_t_ha.corr(cmp_.calibrated_fresh_t_ha, method="spearman"),
                     "mean_abs_diff_t_ha": cmp_.lue_minus_aquacrop.abs().mean(), "n": len(cmp_)}

    # ---- outputs
    fin = fin.sort_values("final_t_ha", ascending=False)
    fin.to_csv(out / "tehsil_final.csv", index=False)
    district.to_csv(out / "district_final.csv", index=False)
    cell_final.to_csv(out / "cell_final.csv", index=False)
    qf.to_csv(out / "quality_flags.csv", index=False)
    if cmp_ is not None:
        cmp_.to_csv(out / "comparison_aquacrop.csv", index=False)
    P = cfg["parameters"]
    params = pd.DataFrame([{"parameter": k, "central": v["central"], "ensemble_values": str(v["values"]),
                            "sobol_range": str(v["range"]), "source": v["source"]} for k, v in P.items()] +
                          [{"parameter": f"water scalar: {k}", "central": "member", "ensemble_values": f"weight {v['weight']}",
                            "sobol_range": "switch", "source": f"{v['equation']} - {v['source']}"}
                           for k, v in cfg["water_stress"]["forms"].items()] +
                          [{"parameter": "season amplitude fraction", "central": cfg["season"]["amplitude_fraction"],
                            "ensemble_values": str(cfg["season"]["ensemble_fractions"]),
                            "sobol_range": str(cfg["season"]["sobol_range"]), "source": "TIMESAT-type threshold"}])
    fapar_txt = ("fAPAR is the NDVI-based value of the 16-day table (not SL2P): background fAPAR from soil and mixed "
                 "pixels is counted as growth (share per tehsil: step 10, I audit diagnostics); the fix is SL2P fAPAR "
                 "(method document). " if mode == "tehsil" else
                 ("fAPAR: SL2P from E1. " if "fapar_sl2p" in pd.read_csv(S("01_inputs", "obs_ndvi.csv"), nrows=1)
                  else "fAPAR = 1.257 NDVI - 0.161 (Bastiaanssen & Ali 2003 eq. 4; E1 without SL2P; the method "
                       "document prescribes SL2P): background fAPAR from soil is counted as growth. "))
    agg_txt = ("Sentinel-2, Landsat and ERA5 are averaged per tehsil and period in the input table before the model; the "
               "method document (step 8) prescribes yield per pixel/field then area-weighted aggregation (aggregation "
               "bias, no between-field variation). " if mode == "tehsil" else
               "Field units (v0.8.0): yield per 10 m crop-map pixel (step 05b), mean per parcel, crop-area-weighted "
               "to cell, tehsil and district (method document step 8: per pixel, then area-weighted aggregation); "
               "Landsat EF and ETa of the 30 m pixel holding the pixel centre (thermal pixels can mix the crop with "
               "other land in small parcels); the field sample gives the design-based district estimate and its "
               "sampling error (district sheet). ")
    limits = (f"NOT VALIDATED unless observed yields of {aoi['season']} are in the calibration file; calibration input: "
              f"{cfg['calibration']['input_file']} (benchmark: {aoi['benchmark'].get('source', 'not given')}). "
              + fapar_txt +
              "Landsat EF (Ws = EF: Bastiaanssen & Ali 2003 eq. 7; Yuan et al. 2007 eq. 5) is a scene-calibrated "
              "energy-balance ratio, not a soil-water stress coefficient Ks; thermal pixels can mix the crop with other "
              "land in small parcels (EF and ETa/ETc at full canopy, corr(EF, fAPAR): step 10; crop-map parcel sources: "
              "qc_checks). Crop parameters, their sources and status (verified / verified_range / adapted / "
              f"not_verified / setting): {Path(cfg['crop']['file']).name}, listed in step 00a parameters.csv; eps_max is "
              "an uncalibrated literature prior until calibrated. " + agg_txt +
              f"Heat-day flag threshold ({cfg['heat_stress']['tmax_threshold_c']} degC) is not from a source - flag only. "
              "Nutrients, pests, diseases and marketable losses are not simulated")
    method_txt = ("dry yield = HI x sum of daily growth, emergence to season end; daily growth = double-logistic "
                  "curve fitted to the period growth rates eps_max x Ts x Ws x fAPAR x f x Rs (approach 2); fresh = dry / DM"
                  if config.biomass_method(cfg) == "fit_biomass" else
                  "dry yield = HI x sum of daily growth eps_max x Ts x Ws x fAPAR x f x Rs, emergence to season "
                  "end; fAPAR and EF interpolated daily between clear observation dates; fresh = dry / DM")
    yb = cfg["yield_basis"]
    readme = pd.DataFrame({"item": ["Final product", "Method", "Yield basis", "Ensemble", "Calibration", "Uncertainty",
                                    "Status", "Inputs", "Area", "Quality flags", "Limitations"], "value": [
        f"{product}, fresh yield, {aoi['season']}",
        method_txt,
        f"B = {yb['biomass']}; HI = {yb['harvest_index']}; DM = {yb['fresh_dm']}",
        f"LUE-ENS: {len(f_m)} members = parameters (eps_max, f, Topt, HI, DM) x water scalar "
        f"({' | '.join(cfg['water_stress']['forms'])}) x season "
        f"threshold ({' / '.join(f'{p:.2f}' for p in cfg['season']['ensemble_fractions'])}); water scalar: "
        f"{cfg['water_stress']['status']}; Landsat 8 / 9 equal mix (L8-only / L9-only per tehsil reported)",
        "member-wise ratio of means to the benchmark (each member calibrated); common factor and implied eps_max reported"
        if cal is not None else "none",
        "P10-P90 are a " + cfg["ensemble"]["spread_label"] + ". Uncalibrated: across documented values and "
        "method choices; calibrated P10-P90: pattern "
        "uncertainty (season threshold, Topt). Landsat 8 / 9 choice, the largest pattern factor (Sobol): "
        "final_l8_only / final_l9_only. Every source and its handling: sheet uncertainty_sources",
        (cal["factor"].warning if is_prov else "see validation sheet") + " " + cfg["calibration"]["model_scope_note"],
        "16-day tehsil table (Sentinel-2 every clear ~5-day date over all crop-map fields, Landsat EF, ERA5; "
        "averaged per tehsil and 16-day period) -> growth rate per period -> double-logistic daily curve -> summed; "
        "per-date S2 NDVI (season dates, Kc), daily weather" if mode == "tehsil" else
        "pixel stacks PX of step 00c (Sentinel-2 NDVI per 10 m pixel and date, Landsat EF / ETa per overpass at the "
        "pixel centre, from the stored rasters) and the ERA5-Land weather E3; field exports E1-E3 for the sample",
        f"crop-map crop area {dist['area_ha']:,.0f} ha (polygon area of the classified parcels; the 16-day "
        f"table area statistic is not used) vs official {ref_area_txt} ({aoi['benchmark'].get('area_source', '')}) - "
        "production uses the crop-map area; map accuracy not assessed",
        f"clear S2 dates {cfg['season']['quality_window'][0]} to {cfg['season']['quality_window'][1]} < "
        f"{qs['min_clear_s2_dates']}; |EF L8 - L9| > {qs['max_ef_l8_l9_abs_diff']}; heat days (Tmax > "
        f"{cfg['heat_stress']['tmax_threshold_c']} degC in bulking) > {qs['max_heat_days']}; season start, peak or end censored "
        "(thresholds: processing settings)",
        limits]})
    xl = out / "09_LUE_results.xlsx"
    with pd.ExcelWriter(xl) as x:
        readme.to_excel(x, sheet_name="read_me", index=False)
        fin.to_excel(x, sheet_name="tehsil_final", index=False)
        district.T.reset_index().rename(columns={"index": "item", 0: "value"}).to_excel(x, sheet_name="district", index=False)
        cell_final.to_excel(x, sheet_name="cell_final", index=False)
        qf.to_excel(x, sheet_name="quality_flags", index=False)
        pd.read_csv(paths["season"]).to_excel(x, sheet_name="season", index=False)
        pd.read_csv(paths["drivers_summary"]).to_excel(x, sheet_name="drivers_summary", index=False)
        ens_t.to_excel(x, sheet_name="lue_ens_uncalibrated", index=False)
        pd.read_csv(paths["variance_shares"]).to_excel(x, sheet_name="variance_shares", index=False)
        pd.read_csv(paths["sobol"]).to_excel(x, sheet_name="sobol", index=False)
        pd.read_csv(paths["alternatives"]).to_excel(x, sheet_name="alternatives", index=False)
        pd.read_csv(paths["oat"]).to_excel(x, sheet_name="one_at_a_time", index=False)
        if cal is not None:
            cal["factor"].to_frame("value").reset_index().to_excel(x, sheet_name="calibration", index=False)
            pd.read_csv(cal["folder"] / "validation.csv").to_excel(x, sheet_name="validation", index=False)
            pd.read_csv(cal["folder"] / "acceptance.csv").to_excel(x, sheet_name="acceptance_targets", index=False)
        if cmp_ is not None:
            pd.concat([cmp_, pd.DataFrame([cmp_stats])]).to_excel(x, sheet_name="comparison_aquacrop", index=False)
        params.to_excel(x, sheet_name="parameters", index=False)
        pd.DataFrame(cfg["uncertainty_sources"]).to_excel(x, sheet_name="uncertainty_sources", index=False)
        pd.read_csv(paths["qc_checks"]).to_excel(x, sheet_name="qc_checks", index=False)
    config.write_manifest(out, "09_aggregate", paths, {"folder": out},
                          {"product": product, "calibration_id": cal["calibration_id"] if cal is not None else None})
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(district.T.to_string(header=False))
    print()
    print(fin[["tehsil", "final_t_ha", "final_p10", "final_p90", "lue_ens_t_ha", "central_config_t_ha", "area_ha",
               "flags"]].round(2).to_string(index=False))
    if cmp_stats is not None:
        print("\nLUE vs AquaCrop (calibrated, same benchmark):", {k: round(v, 3) for k, v in cmp_stats.items()})
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
