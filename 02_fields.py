"""Step 02 - field sample for the field-level LUE run: stratified simple random sample without replacement,
strata = ERA5-Land 0.1 deg cells (Cochran 1977).

Frame: the crop parcels of the crop mask (representative point -> nearest ERA5-Land grid point = cell). Eligible =
pure parcels (config sample.eligibility); rules relaxed cell by cell only where a cell has fewer than
min_fields_per_cell eligible parcels (every relaxation flagged). n_h = min(fields_per_cell, N_h) parcels drawn at
random: inclusion probability n_h / N_h (fields.csv inclusion_prob). Yield per hectare: separate ratio estimator with
its sampling error (core.sampling); frame coverage = share of the cell's mapped crop area inside the frame.
Run once per season, before step 00c: fields.gpkg (layer fields) is read by 00c_download.py for exports E1-E3.
Writes: fields.csv, fields.gpkg (fields, cells), cell_strata.csv, sample_map.png, checks.csv, manifest.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from core import config, plots, sampling


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    out = config.output_dir(cfg, "02_fields")
    config.check_writable(out)
    aoi, sc = cfg["aoi"], cfg["aoi"]["sample"]
    inp, step = aoi["inputs"], aoi["inputs"]["era5_cell_deg"]
    p_cm = config.input_path(cfg, "crop_mask")
    p_bd = config.input_path(cfg, "boundaries")

    cm, frame, utm = sampling.parcel_frame(cfg)           # all parcels of the crop map (field_id, cell)
    rules = sampling.eligibility(frame, sc["eligibility"], sc.get("relaxed_values", {}))
    frame["eligible"] = np.logical_and.reduce([frame[f"ok_{k}"].to_numpy(bool) for k in rules])
    if sc.get("design", "stratified_srs") != "stratified_srs":
        raise SystemExit(f"sample.design {sc.get('design')} not supported (stratified_srs)")
    sel, frames = sampling.select_srs(frame, rules, sc)

    strata = frame.groupby("cell").agg(cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"),
                                       frame_parcels=("field_id", "size"), cell_crop_ha=("area_ha", "sum"),
                                       eligible_parcels=("eligible", "sum"),
                                       tehsils=("tehsil", lambda x: ";".join(sorted(set(x)))))
    f = sel.merge(frame, on="field_id")
    f["nearest_sampled_km"] = sampling.nearest_km(f[["x_m", "y_m"]])
    strata["sampled_fields"] = f.groupby("cell").size().reindex(strata.index).fillna(0).astype(int)
    fr = frames.set_index("cell")
    strata["frame_n"] = fr.frame_n.reindex(strata.index).fillna(0).astype(int)
    strata["frame_area_ha"] = fr.frame_area_ha.reindex(strata.index).fillna(0.0)
    strata["frame_coverage"] = strata.frame_area_ha / strata.cell_crop_ha
    strata["rules_relaxed"] = fr.rules_relaxed.reindex(strata.index)
    f["design_weight_ha"] = f.cell.map(strata.cell_crop_ha / strata.sampled_fields)
    strata = strata.reset_index()

    checks = []
    unsampled = strata[strata.sampled_fields == 0]
    checks.append({"check": "every cell with the crop has a sampled field", "status": "PASS" if unsampled.empty else "WARN",
                   "detail": f"{len(unsampled)} cells without a field ({unsampled.cell_crop_ha.sum():.1f} ha not represented)"})
    few = strata[(strata.sampled_fields > 0) & (strata.sampled_fields < sc["min_fields_per_cell"])]
    checks.append({"check": f"cells with >= {sc['min_fields_per_cell']} fields (sampling-error estimate)",
                   "status": "PASS" if few.empty else "WARN",
                   "detail": f"{len(few)} cells with fewer ({few.cell_crop_ha.sum():.1f} ha; pooled variance used)"})
    checks.append({"check": "design weights sum to the sampled cells' crop area", "status":
                   "PASS" if abs(f.design_weight_ha.sum() - strata[strata.sampled_fields > 0].cell_crop_ha.sum()) < 1e-6
                   else "FAIL", "detail": f"{f.design_weight_ha.sum():,.1f} ha"})
    ht = f.groupby("cell").inclusion_prob.apply(lambda p: (1 / p).sum())
    fn = strata.set_index("cell").frame_n.reindex(ht.index)
    checks.append({"check": "inclusion probabilities: sum of 1/pi per cell = frame size (Horvitz-Thompson count)",
                   "status": "PASS" if np.allclose(ht, fn) else "FAIL",
                   "detail": f"pi {f.inclusion_prob.min():.4f}-{f.inclusion_prob.max():.4f}; {int(fn.sum()):,} frame parcels"})
    cov = strata.frame_area_ha.sum() / strata.cell_crop_ha.sum()
    checks.append({"check": "frame coverage (mapped crop area inside the sampling frame)", "status": "INFO",
                   "detail": f"{100 * cov:.0f} % of {strata.cell_crop_ha.sum():,.0f} ha; parcels outside the frame "
                             f"(mixed segments, small parcels) take their cell's ratio estimate (assumption)"})
    gd = cfg["era5_land_grid"]
    ll = np.r_[strata.cell_lat, strata.cell_lon]
    off = np.abs(ll / gd["step_deg"] - np.round(ll / gd["step_deg"]))
    checks.append({"check": f"cells are ERA5-Land grid points ({gd['step_deg']} deg)",
                   "status": "PASS" if (off < 1e-6).all() else "FAIL",
                   "detail": f"{len(strata)} cells; spatial spread: nearest sampled neighbour median "
                             f"{f.nearest_sampled_km.median():.1f} km"})
    relaxed = (f.rules_relaxed != "none").mean()
    checks.append({"check": "eligibility rules relaxed", "status": "INFO",
                   "detail": f"{100 * relaxed:.0f} % of fields from cells with relaxed rules; "
                             f"{f.rules_relaxed.value_counts().to_dict()}"})
    checks.append({"check": "pure parcels among sampled fields", "status": "INFO",
                   "detail": f"{(f.source == 'parcel').mean() * 100:.0f} % source=parcel; frame: {frame.eligible.sum():,} "
                             f"eligible of {len(frame):,} parcels"})
    ck = pd.DataFrame(checks)

    cols = ["field_id", "parcel_id", "tehsil", "cell", "lon", "lat", "x_m", "y_m", "area_ha", "source", "prob",
            "inclusion_prob", "frame_n", "design_weight_ha", "rules_relaxed", "nearest_sampled_km"]
    f = f.sort_values("field_id")[cols]
    f.to_csv(out / "fields.csv", index=False)
    strata.to_csv(out / "cell_strata.csv", index=False)
    ck.to_csv(out / "checks.csv", index=False)
    geo = cm.set_index("field_id").loc[f.field_id, ["geometry"]].reset_index()
    gf = gpd.GeoDataFrame(f.merge(geo, on="field_id"), geometry="geometry", crs=cm.crs)
    cells = gpd.GeoDataFrame(strata, crs=4326, geometry=[box(o - step / 2, a - step / 2, o + step / 2, a + step / 2)
                                                         for a, o in zip(strata.cell_lat, strata.cell_lon)])
    gf.to_file(out / "fields.gpkg", layer="fields", driver="GPKG")
    cells.to_file(out / "fields.gpkg", layer="cells", driver="GPKG")

    bd = gpd.read_file(p_bd)
    tb = bd[bd.DISTRICT == inp["boundary_district"]].to_crs(4326).copy()
    names = frame.groupby(frame.tehsil.str.upper().str.replace(" ", "_")).tehsil.first()
    tb["tehsil"] = tb.SUB_DIST.map(names).fillna(tb.SUB_DIST.str.title())
    plots.grid_map(out / "sample_map.png", tb, cells, frame, f"{aoi['name']}: field sample for the field-level LUE run, "
                   f"{aoi['season']}\n{len(f)} crop parcels ({100 * (f.source == 'parcel').mean():.0f} % pure), up to "
                   f"{sc['fields_per_cell']} per ERA5-Land cell (stratified simple random sample)",
                   f"Stratified simple random sample (strata = ERA5-Land 0.1 deg cells); inclusion probability = "
                   f"n_h / N_h; yield per ha by the separate ratio estimator\nEligible:{sc['eligibility']}\nCells: mapped crop area of the crop-mask parcels "
                   f"({frame.area_ha.sum():,.0f} ha in {len(frame):,} parcels)\nfields.gpkg is read by step 00c "
                   f"for exports E1 (Sentinel-2), E2 (Landsat) and E3 (ERA5-Land), see README",
                   points=f, point_label="Sampled field")
    # a step with a FAIL check leaves no manifest: run.py then never treats it as complete
    if not (ck.status == "FAIL").any():
        config.write_manifest(out, "02_fields", {"crop_mask": p_cm, "boundaries": p_bd}, {"folder": out},
                              {"seed": sc["seed"], "utm_crs": str(utm), "design": "stratified_srs"})
    pd.set_option("display.width", 250, "display.max_colwidth", 160)
    print(ck.to_string(index=False))
    print(f"\n{len(f)} fields in {int((strata.sampled_fields > 0).sum())} of {len(strata)} cells; fields per cell: "
          f"{strata.sampled_fields.value_counts().sort_index().to_dict()}")
    print(f.groupby("tehsil").agg(fields=("field_id", "size"), weight_ha=("design_weight_ha", "sum")).round(1).to_string())
    print(f"\nwritten to {out}")
    if (ck.status == "FAIL").any():
        raise SystemExit("step 02 stopped: FAIL checks above")


if __name__ == "__main__":
    main()
