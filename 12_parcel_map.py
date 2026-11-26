"""Step 12 - parcel-mean yield map (v0.9.0): the LUE model for every parcel of the crop map on the parcel's
area-weighted mean inputs, from the stored rasters of step 00c, next to the pixel-based parcel yield of step 11.

Why: the final products (steps 09, 11) model every 10 m pixel and then average (pixel -> parcel). Observed yields
(crop cutting, farmer yields) are plot or parcel values, and the model is not linear in NDVI and EF; a single
pixel's NDVI series is also noisier than a parcel mean (edge pixels, residual clouds), which moves its season dates.
This step models each parcel as one unit - exactly as the field sample of steps 01-05 - so that every parcel has
both values: the parcel-mean yield (this step) and the pixel-based yield (step 11). Their difference shows where
the pixel aggregation moves the yield. The final tehsil and district values stay those of step 09.

Chain: GeoTIFFs of step 00c -> values per parcel -> field-unit model of steps 01-05 -> calibration of step 08.
  Sentinel-2  NDVI and clear mask per date (<run>_rasters/ndvi, 10 m; stac.sentinel2.coverage all_parcels): the
              area-weighted mean NDVI and the clear share of every parcel (pixel share inside the polygon), as E1
  Landsat     clear mask, EF and ETa per scene (<run>_rasters/landsat): clear share and the area-weighted EF and ETa
              of the clear pixels, one row per parcel and overpass, as E2
  Weather     ERA5-Land daily per 0.1 degree cell (E3); a parcel in a cell without E3 takes the nearest cell of E3
Model: the code of steps 01, 03, 04, 05 (core.drivers.field_tables, core.season.unit_seasons, core.drivers
field_drivers, core.model) with every LUE-ENS member; calibration with the member factors of step 08 (the factors
fitted on the pixel-based values, applied unchanged: the parcel-mean district value therefore differs from the
benchmark by the aggregation effect - reported, not corrected).
A parcel the season rules of step 03 exclude (no clear Sentinel-2 date, no Landsat EF, no clear NDVI cycle) is not
modelled: it takes the area-weighted yield of the modelled parcels of its cell, else of its tehsil (status column).
Tehsil and district: production = sum of parcel yield x parcel crop area; P10 / P90 by member-level aggregation over
the modelled parcels (as step 06).
Checks: the sampled fields reproduce the exports E1, E2 and the central yield of step 05; parcel production sums to
the tehsil and district totals; parcel-mean vs pixel-based yield per parcel and district (step 11, step 09); the
parcel-mean district yield vs the 95 % interval of the stratified sample estimate of step 09.
Writes: parcel_yield.gpkg (layer parcels), parcel_yield.csv, parcel_mean_yield_<run>.tif (bands
parcel_map.yield_raster_bands on the 10 m grid of the Sentinel-2 rasters) with its JPG map, tehsil_parcel_map.csv,
district_parcel_map.csv, checks.csv, manifest.
"""
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from core import calibration as calib
from core import config, model, quicklook
from core import drivers as dv
from core import ensemble as en
from core import pixels as px
from core import season as ss
from core import stac_ingest as si

VALUES = ["final_t_ha", "final_p10", "final_p50", "final_p90", "lue_ens_t_ha", "central_t_ha"]
FOLDER = "12_parcel_map"


def s2_values(cfg, P, parcels, grid):
    """E1 rows (field_id, date, ndvi, valid_fraction, n_scenes) of every parcel from the NDVI and clear rasters of
    the Sentinel-2 dates of the run period. Returns (table, (grid key, extent) of the rasters)."""
    run, folder = cfg["run_id"], P["rasters"] / "ndvi"
    recs = [r for r in (si.read_json(p) for p in sorted(folder.glob(f"{run}_ndvi_*.json"))) if r
            and P["start"] <= pd.Timestamp(r["date"]) <= P["end"]]
    if not recs:
        raise SystemExit(f"no Sentinel-2 rasters of the run period in {folder} - run step 00c")
    rows, t0, where = [], time.time(), None
    for k, r in enumerate(recs, 1):
        mf = r["made_from"]
        key, extent = (int(mf["grid"][0]), *mf["grid"][1:]), mf["extent"]
        where = where or (key, extent)
        rows += si.s2_rows(run, r["date"], len(mf["granules"]), parcels, grid, key, extent, folder)
        if k % 10 == 0 or k == len(recs):
            print(f"  Sentinel-2 rasters: {k}/{len(recs)} dates, {(time.time() - t0) / 60:.1f} min", flush=True)
    e1 = pd.DataFrame(rows, columns=["field_id", "date", "ndvi", "valid_fraction", "n_scenes"])
    e1["date"] = pd.to_datetime(e1.date)
    return e1, where


def landsat_values(P, parcels, grid, p_items):
    """E2 rows (one per parcel and overpass) from the clear, EF and ETa rasters of every Landsat scene."""
    ras = P["rasters"] / "landsat"
    items = pd.read_csv(p_items)
    items = items[items.status.astype(str).eq("read")
                  & items.item_id.map(lambda i: (ras / f"{i}_clear.tif").exists())]
    rows, t0 = [], time.time()
    for k, it in enumerate(items.itertuples(), 1):
        rows += si.landsat_rows(ras, it.item_id, it.date, it.sensor, parcels, grid)
        if k % 10 == 0 or k == len(items):
            print(f"  Landsat rasters: {k}/{len(items)} scenes, {(time.time() - t0) / 60:.1f} min", flush=True)
    e2 = pd.DataFrame(rows, columns=["field_id", "date", "sensor", "ef", "eta_mm_day", "valid_fraction", "item_id"])
    e2 = si.overpass_rows(e2)
    e2["date"] = pd.to_datetime(e2.date)
    return e2, len(items)


def yield_raster(path, parcels, table, bands, where, cfg):
    """Parcel values burnt into the 10 m grid of the Sentinel-2 rasters (pixel centre inside the parcel), one band per
    column of `bands`; no data = no parcel."""
    import rasterio
    from rasterio import features
    from rasterio.transform import from_origin
    (epsg, res, px0, py0), (gc, gr, width, height) = where
    transform = from_origin(px0 + gc * res, py0 - gr * res, res, res)
    g = parcels.to_crs(epsg)
    idx = features.rasterize(((geom, i + 1) for i, geom in enumerate(g.geometry)), out_shape=(height, width),
                             transform=transform, fill=0, dtype="int32")
    vals = table.set_index("field_id").reindex(g.field_id)
    prof = si.tif_profile(width, height, len(bands), "float32", f"EPSG:{epsg}", transform, np.nan)

    def write(tmp):
        with rasterio.open(tmp, "w", **prof) as ds:
            for b, name in enumerate(bands, 1):
                v = np.concatenate([[np.nan], vals[name].to_numpy(np.float32)]).astype(np.float32)
                ds.write(v[idx], b)                                   # index 0 = no parcel -> no data
                ds.set_band_description(b, name)
            ds.update_tags(run=cfg["run_id"], crop=cfg["aoi"]["crop"], season=cfg["aoi"]["season"],
                           content="fresh yield (t/ha) of the parcel of each pixel, step 12 parcel-mean yield map "
                                   "(model on the parcel's mean inputs); bands: " + ", ".join(bands))
    config.write_atomic(path, write)
    return int((idx > 0).sum())


def member_level(acc, grid, q):
    """P10 / P50 / P90 of member-level aggregates: acc = (group, member) -> area x value sum (av) and area (a)."""
    w = grid.set_index("member").weight
    rows = []
    for g, x in acc.groupby(level=0):
        v = (x.av / x.a).droplevel(0)
        s = en.member_stats(v.to_numpy(), w.reindex(v.index).to_numpy(), q)
        rows.append({"group": g, **{f"final_p{int(round(100 * k))}": s[f"p{int(round(100 * k))}"] for k in q}})
    return pd.DataFrame(rows).set_index("group")


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    if config.units(cfg) != "field" or not cfg["aoi"]["inputs"].get("crop_mask"):
        raise SystemExit("step 12 needs field units (config units: field) and a crop map (inputs.crop_mask)")
    out = config.output_dir(cfg, FOLDER)
    config.check_writable(out)
    run, pm, sc, cl = cfg["run_id"], cfg["parcel_map"], cfg["season"], cfg["ndvi_cleaning"]
    q, cov = cfg["ensemble"]["quantiles"], cfg["stac"]["sentinel2"].get("coverage", "sample")
    P = si.run_paths(cfg)
    S = lambda step, name: config.step_output(cfg, step, name, required=False)   # noqa: E731
    paths = {"crop_mask": config.input_path(cfg, "crop_mask"),
             "e3_weather_daily": P["exports"]["e3_weather_daily"],
             "landsat_items": P["folder"]["landsat"] / "landsat_items.csv",
             "pixel_parcel_yield": S("11_pixel_map", "parcel_yield.csv"),
             "pixel_district": S("11_pixel_map", "district_pixel_map.csv"),
             "pixel_tehsil": S("11_pixel_map", "tehsil_pixel_map.csv"),
             "district_final": S("09_aggregate", "district_final.csv")}
    miss = [str(p) for p in paths.values() if not Path(p).exists()]
    if miss:
        raise SystemExit("step 12 needs step 00c (rasters, E3) and steps 01-11 first - missing:\n  " + "\n  ".join(miss))
    optional = {"e1_s2_perdate": P["exports"]["e1_s2_perdate"], "e2_landsat_perdate": P["exports"]["e2_landsat_perdate"],
                "sample_units": S("03_season", "units.csv"), "central_unit": S("05_lue", "central_unit.csv")}
    paths.update({k: p for k, p in optional.items() if Path(p).exists()})
    cal = calib.latest(cfg)
    if cal is not None:
        paths["member_factors"] = cal["folder"] / "member_factors.csv"
    t_start = time.time()

    # ---- values of every parcel from the stored rasters
    parcels = si.load_parcels(cfg)
    print(f"run {run}: {len(parcels):,} parcels of the crop map, {parcels.area_ha.sum():,.1f} ha; Sentinel-2 coverage "
          f"of step 00c: {cov}", flush=True)
    grid = si.FieldGrid(parcels)
    e1, where = s2_values(cfg, P, parcels, grid)
    e2, n_scenes = landsat_values(P, parcels, grid, paths["landsat_items"])
    dv.clip_shares(e1)
    dv.clip_shares(e2)
    e3, _ = dv.weather_eto(pd.read_csv(paths["e3_weather_daily"], parse_dates=["date"]))
    wcell = px.weather_cells(parcels.cell, e3.cell)
    units = pd.DataFrame({"unit": parcels.field_id, "tehsil": parcels.tehsil, "cell": parcels.cell.map(wcell),
                          "area_ha": parcels.area_ha, "lon": parcels.lon, "lat": parcels.lat})

    # ---- the field-unit model of steps 01-05 for every parcel, in chunks
    start = pd.Timestamp(sc["window"][0]) - pd.Timedelta(days=sc["lead_days"])
    fractions = sorted(set(sc["ensemble_fractions"]) | {sc["amplitude_fraction"]})
    c = model.central_config(cfg)
    rows, excluded, acc_t, acc_d, mgrid = [], [], [], [], None
    e1g, e2g = e1.groupby("field_id").indices, e2.groupby("field_id").indices
    n = int(pm["chunk_parcels"])
    for i in range(0, len(units), n):
        u = units.iloc[i:i + n].reset_index(drop=True)
        k1 = np.concatenate([e1g.get(f, []) for f in u.unit] + [[]]).astype(int)
        k2 = np.concatenate([e2g.get(f, []) for f in u.unit] + [[]]).astype(int)
        obs, thermal, weather, _ = dv.field_tables(cfg, u, e1.iloc[k1], e2.iloc[k2], e3)
        nd = obs[(obs.date >= start) & (obs.date <= pd.Timestamp(sc["window"][1]))]
        has_th = thermal[thermal.ef.notna()].groupby("unit").size()
        _, daily, seasons, quality, excl = ss.unit_seasons(nd, u.unit, has_th, sc, cl, fractions, start)
        excluded += excl
        if seasons:
            seas = pd.DataFrame(seasons)
            seas["amplitude_fraction"] = seas.amplitude_fraction.round(2)
            central = seas[np.isclose(seas.amplitude_fraction, sc["amplitude_fraction"])].set_index("unit")
            valid = u[u.unit.isin(central.index)]
            d, _, _ = dv.field_drivers(cfg, valid, weather, thermal, pd.concat(daily), central)
            mem, mgrid = model.members(d, seas, cfg)
            cen = model.run(d, seas, c).set_index("unit").fresh_t_ha
            f_m = cal["member_factors"] if cal is not None else pd.Series(1.0, index=mgrid.member)
            mem["v"] = mem.fresh_t_ha * mem.member.map(f_m)
            qual = pd.DataFrame(quality).set_index("unit")
            n_ef = has_th.reindex(valid.unit).fillna(0)
            for uid, g in mem.groupby("unit", sort=False):
                s = en.member_stats(g.v, g.weight, q)
                cs = central.loc[uid]
                rows.append({"field_id": uid, "final_t_ha": s["mean"], "final_p10": s["p10"], "final_p50": s["p50"],
                             "final_p90": s["p90"], "lue_ens_t_ha": float(np.average(g.fresh_t_ha, weights=g.weight)),
                             "central_t_ha": float(cen[uid]), "emergence": cs.emergence, "season_end": cs.season_end,
                             "season_days": cs.season_days, "peak_date": cs.peak_date, "peak_ndvi": cs.peak_ndvi,
                             "start_flag": cs.start_flag, "peak_flag": cs.peak_flag, "end_flag": cs.end_flag,
                             "clear_s2_dates_quality_window": int(qual.loc[uid, "clear_obs_quality_window"]),
                             "landsat_ef_overpasses": int(n_ef[uid])})
            a, te = valid.set_index("unit").area_ha, valid.set_index("unit").tehsil
            mm = pd.DataFrame({"tehsil": mem.unit.map(te).to_numpy(), "member": mem.member.to_numpy(),
                               "a": mem.unit.map(a).to_numpy()})
            mm["av"] = mm.a * mem.v.to_numpy()
            acc_t.append(mm.groupby(["tehsil", "member"])[["av", "a"]].sum())
            acc_d.append(mm.assign(tehsil="district").groupby(["tehsil", "member"])[["av", "a"]].sum())
        print(f"  parcels {min(i + n, len(units)):,}/{len(units):,} modelled or excluded, "
              f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    if mgrid is None:
        raise SystemExit("no parcel could be modelled - check the rasters of step 00c (stac.sentinel2.coverage)")

    # ---- parcel table: modelled parcels, the others take the cell (else tehsil) mean
    res = pd.DataFrame(rows)
    ex = pd.DataFrame(excluded, columns=["unit", "reason"]).rename(columns={"unit": "field_id"})
    sample = set(pd.read_csv(paths["sample_units"]).unit) if "sample_units" in paths else set()
    t = parcels.drop(columns="geometry")[["field_id", "parcel_id", "tehsil", "cell", "area_ha", "source", "prob",
                                          "mixed", "lon", "lat"]].copy()
    t["weather_cell"] = units.cell.to_numpy()
    t["sampled"] = t.field_id.isin(sample)
    t = t.merge(res, on="field_id", how="left").merge(ex, on="field_id", how="left")
    t["status"] = np.where(t.final_t_ha.notna(), "modelled", "")
    mod = t[t.status == "modelled"]
    for level in ("cell", "tehsil"):
        means = mod.groupby(level)[VALUES].apply(lambda x: x.mul(mod.loc[x.index, "area_ha"], axis=0).sum()
                                                 / mod.loc[x.index, "area_ha"].sum())
        fill = (t.status == "") & t[level].isin(means.index)
        for col in VALUES:
            t.loc[fill, col] = t.loc[fill, level].map(means[col])
        t.loc[fill, "status"] = f"{level} estimate (not modelled: " + t.loc[fill, "reason"].fillna("") + ")"
    t.loc[t.status == "", "status"] = "no estimate (not modelled: " + t.reason.fillna("") + ")"
    t["production_t"] = t.final_t_ha * t.area_ha
    t["weather_from_nearest_cell"] = t.cell != t.weather_cell
    # next to it: the pixel-based parcel yield of step 11 (pixel model, then mean of the pixels)
    p11 = pd.read_csv(paths["pixel_parcel_yield"]).set_index("field_id")
    t["pixel_based_t_ha"] = t.field_id.map(p11.final_t_ha)
    t["pixel_based_status"] = t.field_id.map(p11.status) if "status" in p11 else ""
    t["pixel_minus_parcel_t_ha"] = t.pixel_based_t_ha - t.final_t_ha
    for col in ("emergence", "season_end", "peak_date"):
        t[col] = pd.to_datetime(t[col]).dt.strftime("%Y-%m-%d")

    # ---- tehsil and district
    bands = list(pm["yield_raster_bands"])
    acc_t = pd.concat(acc_t).groupby(level=[0, 1]).sum()
    acc_d = pd.concat(acc_d).groupby(level=[0, 1]).sum()

    def level_table(g, ml):
        r = pd.DataFrame({"parcels": g.size(), "area_ha": g.area_ha.sum(),
                          "modelled_area_ha": g.apply(lambda x: x.area_ha[x.status == "modelled"].sum(),
                                                      include_groups=False),
                          "production_t": g.production_t.sum()})
        r["modelled_area_share"] = r.modelled_area_ha / r.area_ha
        r["final_t_ha"] = r.production_t / r.area_ha
        for col in ("lue_ens_t_ha", "central_t_ha"):
            r[col] = g.apply(lambda x: np.average(x[col], weights=x.area_ha) if x[col].notna().any() else np.nan,
                             include_groups=False)
        return r.join(ml)

    tv = t.dropna(subset=["final_t_ha"])
    teh = level_table(tv.groupby("tehsil"), member_level(acc_t, mgrid, q)).rename_axis("tehsil").reset_index()
    t11 = pd.read_csv(paths["pixel_tehsil"]).set_index("tehsil")
    teh["pixel_based_t_ha"] = teh.tehsil.map(t11.final_t_ha)
    teh["pixel_minus_parcel_t_ha"] = teh.pixel_based_t_ha - teh.final_t_ha
    dist = level_table(tv.assign(g="district").groupby("g"), member_level(acc_d, mgrid, q)).reset_index(drop=True)
    dist.insert(0, "district", cfg["aoi"]["name"])
    dist.insert(1, "product", (cal["factor"]["product"] if cal is not None else "LUE-ENS (uncalibrated)")
                + ", parcel-mean model (comparison product; final values: step 09)")
    d9 = pd.read_csv(paths["district_final"]).iloc[0]
    for k in ("final_t_ha", "stratified_estimate_t_ha", "sampling_se_t_ha", "ci95_low", "ci95_high"):
        dist[f"step09_{k}"] = d9.get(k, np.nan)
    dist["pixel_minus_parcel_t_ha"] = dist.step09_final_t_ha - dist.final_t_ha
    dist["p10_p90_meaning"] = cfg["ensemble"]["spread_label"] + " (member-level aggregation over the modelled parcels)"

    # ---- checks
    ck = []
    add = lambda name, status, detail: ck.append({"check": name, "status": status, "detail": detail})   # noqa: E731
    have_nd = e1[e1.ndvi.notna()].field_id.nunique()
    add("Sentinel-2 rasters cover the parcels (stac.sentinel2.coverage all_parcels)",
        "PASS" if cov == "all_parcels" and have_nd == len(parcels) else "WARN",
        f"coverage {cov}; {have_nd:,} of {len(parcels):,} parcels with an NDVI value on {e1.date.nunique()} dates")
    have_ef = e2[e2.ef.notna()].field_id.nunique()
    add("Landsat rasters cover the parcels", "PASS" if have_ef == len(parcels) else "WARN",
        f"{have_ef:,} of {len(parcels):,} parcels with EF; {n_scenes} scenes")
    if "e1_s2_perdate" in paths:
        x1 = pd.read_csv(paths["e1_s2_perdate"], parse_dates=["date"]).merge(e1, on=["field_id", "date"],
                                                                           suffixes=("_export", "_map"))
        d1 = max(float((x1.ndvi_export - x1.ndvi_map).abs().max()),
                 float((x1.valid_fraction_export - x1.valid_fraction_map).abs().max())) if len(x1) else 0.0
        add("sampled fields: values from the Sentinel-2 rasters = export E1", "PASS" if d1 <= 1e-9 else "FAIL",
            f"{len(x1):,} field-dates; largest |difference| of NDVI and clear share {d1:.2e}")
    if "e2_landsat_perdate" in paths:
        x2 = pd.read_csv(paths["e2_landsat_perdate"], parse_dates=["date"]).merge(e2, on=["field_id", "date", "sensor"],
                                                                                suffixes=("_export", "_map"))
        d2 = float((x2.ef_export - x2.ef_map).abs().max()) if len(x2) else 0.0
        add("sampled fields: EF from the Landsat rasters = export E2", "PASS" if d2 <= 1e-6 else "FAIL",
            f"{len(x2):,} field-overpasses; largest |EF difference| {d2:.2e} (rasters hold EF as float32)")
    if "central_unit" in paths:
        c5 = pd.read_csv(paths["central_unit"]).merge(res[["field_id", "central_t_ha"]], left_on="unit",
                                                      right_on="field_id")
        d5 = float((c5.fresh_t_ha - c5.central_t_ha).abs().max()) if len(c5) else np.nan
        add("sampled fields: central yield of step 12 = step 05", "PASS" if len(c5) and d5 <= pm["sample_tolerance_t_ha"]
            else "FAIL", f"{len(c5)} fields; largest |difference| {d5:.2e} t/ha (tolerance {pm['sample_tolerance_t_ha']})")
    stc = t.status.str.split(" ").str[0].value_counts().to_dict()
    add("parcels modelled", "INFO", f"{stc}; modelled area {mod.area_ha.sum():,.1f} of {t.area_ha.sum():,.1f} ha "
        f"({100 * mod.area_ha.sum() / t.area_ha.sum():.0f} %); reasons not modelled: "
        f"{ex.reason.str.replace(r'[0-9.]+', '#', regex=True).value_counts().to_dict()}")
    add("weather of the parcel's own ERA5-Land cell", "INFO",
        f"{int(t.weather_from_nearest_cell.sum()):,} parcels in a cell without E3 take the nearest cell of E3")
    tot = [t.production_t.sum(), teh.production_t.sum(), float(dist.production_t.iloc[0])]
    add("production: parcels = sum of tehsils = district", "PASS" if np.allclose(tot, tot[0]) else "FAIL",
        f"{tot[0]:,.1f} t; {teh.production_t.sum():,.1f} t; {tot[2]:,.1f} t")
    both = t[(t.status == "modelled") & t.pixel_based_t_ha.notna()
             & t.pixel_based_status.astype(str).str.startswith("modelled")]
    if len(both):
        dlt = both.pixel_minus_parcel_t_ha
        r = float(np.corrcoef(both.pixel_based_t_ha, both.final_t_ha)[0, 1]) if len(both) > 2 else np.nan
        add("parcels: pixel-based yield (step 11) - parcel-mean yield (this step)", "INFO",
            f"{len(both):,} parcels modelled by both: difference mean {dlt.mean():+.2f}, median {dlt.median():+.2f}, "
            f"P10 {dlt.quantile(0.1):+.2f}, P90 {dlt.quantile(0.9):+.2f}, mean |difference| {dlt.abs().mean():.2f} "
            f"t/ha, r {r:.3f} (pixel yields then mean vs mean inputs then yield; pixel season dates are their own)")
    fm, f9 = float(dist.final_t_ha.iloc[0]), float(d9.get("final_t_ha", np.nan))
    add("district: final value of step 09 (pixel-based) vs parcel-mean model", "INFO",
        f"pixel-based {f9:.2f} t/ha, parcel-mean {fm:.2f} t/ha, difference {f9 - fm:+.2f} t/ha "
        f"({100 * (f9 - fm) / fm:+.1f} %); both calibrated with the member factors of step 08")
    lo, hi = d9.get("ci95_low", np.nan), d9.get("ci95_high", np.nan)
    inside = bool(lo <= fm <= hi) if np.isfinite([lo, hi]).all() else None
    add("parcel-mean district yield inside the 95 % interval of the stratified sample estimate (step 09)",
        "PASS" if inside else ("WARN" if inside is False else "INFO"),
        f"parcel-mean {fm:.2f} t/ha (all parcels, model-based); sample {d9.get('stratified_estimate_t_ha', np.nan):.2f} "
        f"t/ha, 95 % interval {lo:.2f}-{hi:.2f} (design-based)")
    ck = pd.DataFrame(ck)

    # ---- outputs
    t.to_csv(out / "parcel_yield.csv", index=False)
    gpd.GeoDataFrame(t, geometry=parcels.set_index("field_id").geometry.reindex(t.field_id).values, crs=4326) \
        .to_file(out / "parcel_yield.gpkg", layer="parcels", driver="GPKG")
    tif = out / f"parcel_mean_yield_{run}.tif"
    px_n = yield_raster(tif, parcels, t, bands, where, cfg)
    quicklook.make_map(tif, out / f"parcel_mean_yield_{run}.jpg", cfg)
    teh.to_csv(out / "tehsil_parcel_map.csv", index=False)
    dist.to_csv(out / "district_parcel_map.csv", index=False)
    ck.to_csv(out / "checks.csv", index=False)
    if not (ck.status == "FAIL").any():           # a step with a FAIL check leaves no manifest (never complete)
        config.write_manifest(out, FOLDER, paths, {"folder": out},
                              {"parcels": len(t), "modelled": int((t.status == "modelled").sum()),
                               "calibration_id": cal["calibration_id"] if cal is not None else None,
                               "raster_bands": bands, "raster_pixels_with_a_parcel": px_n})
    pd.set_option("display.width", 250, "display.max_colwidth", 200)
    print(ck.to_string(index=False))
    print("\n", teh[["tehsil", "parcels", "area_ha", "modelled_area_share", "final_t_ha", "final_p10", "final_p90",
                     "pixel_based_t_ha", "production_t"]].round(2).to_string(index=False))
    print("\n", dist[["district", "area_ha", "final_t_ha", "final_p10", "final_p90", "production_t",
                      "step09_final_t_ha", "step09_ci95_low", "step09_ci95_high"]].round(2).to_string(index=False))
    print(f"\n{len(t):,} parcels, {(time.time() - t_start) / 60:.1f} min; written to {out}")
    if (ck.status == "FAIL").any():
        raise SystemExit("step 12 stopped: FAIL checks above")


if __name__ == "__main__":
    main()
