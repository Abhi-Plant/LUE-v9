"""Step 11 - pixel yield map: the calibrated yield of every crop-map pixel and parcel, from the pixel yield model of
step 05b and the calibration of step 08.

Every LUE-ENS member of a pixel (and of a parcel) is calibrated with its member factor of step 08, as in step 09;
then the statistics across the members: final_t_ha (weighted mean), final_p10 / p50 / p90. central_t_ha = central
configuration of the method document, uncalibrated; lue_ens_t_ha = uncalibrated LUE-ENS mean. A pixel that step 05b
could not model holds its parcel's value (source column of step 05b); a parcel without a modelled pixel holds the
value of its cell, else of its tehsil (status).
Tehsil and district: production = sum of parcel yield x parcel crop area (crop map), yield = production / area;
P10 / P90 by member-level aggregation (members aggregated first, as step 06) - equal to step 09 by construction
(checked). The pixel-based district value is set next to the design-based estimate of the field sample (step 09).
Writes: yield_<run>.tif (bands pixel_map.yield_raster_bands, float32, on the 10 m grid of the Sentinel-2 rasters;
no data = no crop pixel), yield_<run>.jpg, parcel_yield.gpkg (layer parcels), parcel_yield.csv,
tehsil_pixel_map.csv, district_pixel_map.csv, checks.csv, manifest.
"""
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from core import calibration as calib
from core import config, model, quicklook
from core import ensemble as en
from core import pixels as px
from core import stac_ingest as si


def member_level(mem, level, f_m, q):
    """P10 / P50 / P90 of the calibrated member-level aggregates of a level (step 05b members_<level>)."""
    rows = []
    x = mem.assign(v=mem.fresh_t_ha * mem.member.map(f_m))
    for g, y in x.groupby(level):
        s = en.member_stats(y.v, y.weight, q)
        rows.append({level: g, "final_mean_members": s["mean"],
                     **{f"final_p{int(round(100 * k))}": s[f"p{int(round(100 * k))}"] for k in q}})
    return pd.DataFrame(rows)


def write_raster(path, rows, cols, values, bands, key, extent, cfg):
    """Pixel values on the 10 m grid of the Sentinel-2 rasters, one band per name of `bands`, written in strips of
    rows (memory independent of the raster size). Returns the number of grid pixels with a value."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    (epsg, res, x0, y0), (gc, gr, width, height) = key, extent
    transform = from_origin(x0 + gc * res, y0 - gr * res, res, res)
    prof = si.tif_profile(width, height, len(bands), "float32", f"EPSG:{epsg}", transform, np.nan)
    o = np.argsort(rows, kind="stable")
    rs = rows[o]

    def write(tmp):
        with rasterio.open(tmp, "w", **prof) as ds:
            for s0 in range(0, height, si.BLOCK):
                a, b = np.searchsorted(rs, s0), np.searchsorted(rs, s0 + si.BLOCK)
                if a == b:
                    continue                                    # no crop pixel: the blocks stay empty (no data)
                sel, h = o[a:b], min(si.BLOCK, height - s0)
                for bi, name in enumerate(bands, 1):
                    strip = np.full((h, width), np.nan, np.float32)
                    strip[rows[sel] - s0, cols[sel]] = values[name][sel]
                    ds.write(strip, bi, window=Window(0, s0, width, h))
            for bi, name in enumerate(bands, 1):
                ds.set_band_description(bi, name)
            ds.update_tags(run=cfg["run_id"], crop=cfg["aoi"]["crop"], season=cfg["aoi"]["season"],
                           content="fresh yield (t/ha) of every 10 m crop-map pixel, step 11 pixel yield map; bands: "
                                   + ", ".join(bands))
    config.write_atomic(path, write)
    return int(len(np.unique(rows.astype(np.int64) * width + cols)))


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    if config.units(cfg) != "field" or not cfg["aoi"]["inputs"].get("crop_mask"):
        raise SystemExit("step 11 needs field units (config units: field) and a crop map (inputs.crop_mask)")
    out = config.output_dir(cfg, "11_pixel_map")
    config.check_writable(out)
    run, pm, q = cfg["run_id"], cfg["pixel_map"], cfg["ensemble"]["quantiles"]
    P = si.run_paths(cfg)
    S = lambda name, step="05b_pixel_model": config.step_output(cfg, step, name)   # noqa: E731
    paths = {"pixel_s": S("pixel_s.npy"), "pixel_code": S("pixel_code.npy"), "pixel_source": S("pixel_source.npy"),
             "parcel_values": S("parcel_values.csv"), "combos": S("combos.csv"), "member_grid": S("member_grid.csv"),
             "members_tehsil": S("members_tehsil.csv"), "members_district": S("members_district.csv"),
             "tehsil_final": S("tehsil_final.csv", "09_aggregate"),
             "district_final": S("district_final.csv", "09_aggregate"),
             "pixel_stacks_record": px.stack_folder(P) / "record.json", "pixels": px.stack_folder(P) / "pixels.npz"}
    t_start = time.time()
    cal = calib.latest(cfg)
    grid, combos = pd.read_csv(paths["member_grid"]), pd.read_csv(paths["combos"], index_col="combo")
    km = px.combo_index(grid, combos)
    f_m = cal["member_factors"] if cal is not None else pd.Series(1.0, index=grid.member)
    fv = f_m.reindex(grid.member).to_numpy(float)
    if np.isnan(fv).any():
        raise SystemExit("member factors of step 08 do not cover the members of step 05b - run step 08 again")
    w = grid.weight.to_numpy(float)
    c = model.central_config(cfg)
    kc = int(combos.index[np.isclose(combos.topt, c["topt"]) & (combos.ws_form == c["ws_form"])
                          & np.isclose(combos.season_p, c["season_p"])][0])
    product = cal["factor"]["product"] if cal is not None else "LUE-ENS (uncalibrated)"

    # ---- parcels: calibrated members -> statistics
    pv = pd.read_csv(paths["parcel_values"])
    sk = [f"s_{k}" for k in combos.index]
    fresh = px.yields_from_s(pv[sk].to_numpy(float), grid, km)[2] * fv[None, :]
    st_ = px.member_stats_rows(fresh, w, q)
    t = pv.drop(columns=sk).copy()
    t["final_t_ha"] = st_["mean"]
    for k in q:
        t[f"final_p{int(round(100 * k))}"] = st_[f"p{int(round(100 * k))}"]
    t["production_t"] = t.final_t_ha * t.area_ha
    t.insert(1, "product", product)

    # ---- tehsil and district
    tv = t.dropna(subset=["final_t_ha"])
    levels = {}
    for level, g in (("tehsil", tv.tehsil), ("district", pd.Series("district", index=tv.index))):
        x = tv.assign(_g=g.to_numpy()).groupby("_g")
        r = pd.DataFrame({"parcels": x.size(), "area_ha": x.area_ha.sum(), "production_t": x.production_t.sum(),
                          "modelled_area_ha": x.apply(lambda y: y.area_ha[y.status == "modelled"].sum(),
                                                      include_groups=False),
                          "pixels": x.n_pixels.sum(), "modelled_pixels": x.n_modelled_pixels.sum()})
        r["final_t_ha"] = r.production_t / r.area_ha
        r["modelled_area_share"] = r.modelled_area_ha / r.area_ha
        for col in ("lue_ens_t_ha", "central_t_ha"):
            r[col] = x.apply(lambda y, c_=col: np.average(y[c_], weights=y.area_ha), include_groups=False)
        ml = member_level(pd.read_csv(paths[f"members_{level}"]), level, f_m, q).set_index(level)
        r = r.join(ml)
        levels[level] = r.rename_axis(level).reset_index()
    teh, dist = levels["tehsil"], levels["district"]
    dist["district"] = cfg["aoi"]["name"]
    dist.insert(1, "product", product)
    d9, t9 = pd.read_csv(paths["district_final"]).iloc[0], pd.read_csv(paths["tehsil_final"])
    for k in ("final_t_ha", "stratified_estimate_t_ha", "sampling_se_t_ha", "ci95_low", "ci95_high"):
        dist[f"step09_{k}"] = d9.get(k, np.nan)
    dist["p10_p90_meaning"] = cfg["ensemble"]["spread_label"] + " (member-level aggregation over every crop-map pixel)"

    # ---- pixels: calibrated members -> statistics, in chunks
    s_px = np.load(paths["pixel_s"], mmap_mode="r")
    source = np.load(paths["pixel_source"])
    z = np.load(paths["pixels"])
    rows, cols = z["row"], z["col"]
    bands = list(pm["yield_raster_bands"])
    vals = {b: np.full(len(rows), np.nan, np.float32) for b in bands}
    step = int(pm["chunk_pixels"])
    for a in range(0, len(rows), step):
        b = min(a + step, len(rows))
        sa = np.asarray(s_px[a:b], float)
        fr0 = px.yields_from_s(sa, grid, km)[2]                     # uncalibrated members
        st_p = px.member_stats_rows(fr0 * fv[None, :], w, q)
        have = {"final_t_ha": st_p["mean"], "central_t_ha": px.central_yield(sa, c, kc),
                "lue_ens_t_ha": fr0 @ (w / w.sum()),
                **{f"final_p{int(round(100 * k))}": st_p[f"p{int(round(100 * k))}"] for k in q}}
        for name in bands:
            if name not in have:
                raise SystemExit(f"pixel_map.yield_raster_bands: unknown band {name} (one of {sorted(have)})")
            vals[name][a:b] = np.where(source[a:b] == 4, np.nan, have[name])
    rec = si.read_json(paths["pixel_stacks_record"])
    key = (int(rec["grid"][0]), float(rec["grid"][1]), float(rec["grid"][2]), float(rec["grid"][3]))
    extent = tuple(int(v) for v in rec["extent"])
    tif = out / f"yield_{run}.tif"
    n_grid = write_raster(tif, rows.astype(np.int64), cols.astype(np.int64), vals, bands, key, extent, cfg)
    quicklook.make_map(tif, out / f"yield_{run}.jpg", cfg)

    # ---- checks
    ck = []
    add = lambda name, status, detail: ck.append({"check": name, "status": status, "detail": detail})   # noqa: E731
    tot = [float(t.production_t.sum()), float(teh.production_t.sum()), float(dist.production_t.iloc[0])]
    add("production: parcels = sum of tehsils = district", "PASS" if np.allclose(tot, tot[0]) else "FAIL",
        f"{tot[0]:,.1f} t; {tot[1]:,.1f} t; {tot[2]:,.1f} t")
    fd = float(dist.final_t_ha.iloc[0])
    dm_ = float(dist.final_mean_members.iloc[0])
    add("district: production / area = mean of the calibrated district members", "PASS" if abs(fd - dm_) < 1e-6
        else "FAIL", f"{fd:.6f} vs {dm_:.6f} t/ha (linear aggregation)")
    add("district = step 09 final product", "PASS" if abs(fd - float(d9.final_t_ha)) < 1e-6 else "FAIL",
        f"{fd:.4f} vs {float(d9.final_t_ha):.4f} t/ha")
    m9 = teh.merge(t9[["tehsil", "final_t_ha"]], on="tehsil", suffixes=("", "_09"))
    d_t = float((m9.final_t_ha - m9.final_t_ha_09).abs().max()) if len(m9) else np.nan
    add("tehsils = step 09 final product", "PASS" if len(m9) == len(teh) and d_t < 1e-6 else "FAIL",
        f"{len(m9)} of {len(teh)} tehsils, max |difference| {d_t:.2e} t/ha")
    st_c = t.status.value_counts().to_dict()
    add("parcels", "INFO", f"{st_c}; area with own pixels {t.area_ha[t.status == 'modelled'].sum():,.1f} of "
        f"{t.area_ha.sum():,.1f} ha")
    src = pd.Series(source).map({0: "own value", 1: "parcel value", 2: "cell estimate", 3: "tehsil estimate",
                                 4: "no estimate"}).value_counts().to_dict()
    add("yield GeoTIFF", "INFO", f"{len(rows):,} crop pixels ({src}) on {n_grid:,} grid pixels (a pixel of two "
        f"overlapping parcels shows the later one); bands {bands}; 10 m, EPSG:{key[0]}")
    lo, hi = d9.get("ci95_low", np.nan), d9.get("ci95_high", np.nan)
    inside = bool(lo <= fd <= hi) if np.isfinite([lo, hi]).all() else None
    add("pixel-based district yield inside the 95 % interval of the stratified sample estimate (step 09)",
        "PASS" if inside else ("WARN" if inside is False else "INFO"),
        f"pixel map {fd:.2f} t/ha (every crop-map pixel, model-based); sample "
        f"{d9.get('stratified_estimate_t_ha', np.nan):.2f} t/ha, 95 % interval {lo:.2f}-{hi:.2f} (design-based)")
    ck = pd.DataFrame(ck)

    # ---- outputs
    parcels = si.load_parcels(cfg).set_index("field_id")
    t.to_csv(out / "parcel_yield.csv", index=False)
    gpd.GeoDataFrame(t, geometry=parcels.geometry.reindex(t.field_id).values, crs=4326) \
        .to_file(out / "parcel_yield.gpkg", layer="parcels", driver="GPKG")
    teh.to_csv(out / "tehsil_pixel_map.csv", index=False)
    dist.to_csv(out / "district_pixel_map.csv", index=False)
    ck.to_csv(out / "checks.csv", index=False)
    if not (ck.status == "FAIL").any():           # a step with a FAIL check leaves no manifest (never complete)
        config.write_manifest(out, "11_pixel_map", paths, {"folder": out},
                              {"pixels": int(len(rows)), "parcels": len(t), "raster_bands": bands,
                               "grid_pixels_with_a_value": n_grid,
                               "calibration_id": cal["calibration_id"] if cal is not None else None})
    pd.set_option("display.width", 250, "display.max_colwidth", 200)
    print(ck.to_string(index=False))
    print("\n", teh[["tehsil", "parcels", "pixels", "area_ha", "modelled_area_share", "final_t_ha", "final_p10",
                     "final_p90", "production_t"]].round(2).to_string(index=False))
    print("\n", dist[["district", "product", "area_ha", "final_t_ha", "final_p10", "final_p90", "production_t",
                      "step09_stratified_estimate_t_ha", "step09_ci95_low", "step09_ci95_high"]].round(2)
          .to_string(index=False))
    print(f"\n{len(rows):,} pixels, {len(t):,} parcels, {(time.time() - t_start) / 60:.1f} min; written to {out}")
    if (ck.status == "FAIL").any():
        raise SystemExit("step 11 stopped: FAIL checks above")


if __name__ == "__main__":
    main()
