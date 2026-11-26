"""Step 00d - verification of the downloads of step 00c: is every dataset complete and correct?

For each dataset (Sentinel-2 NDVI, Landsat scenes and clear shares, Landsat EF and ETa, ERA5-Land weather,
elevation, pixel stacks):
  completeness  the STAC catalogue as it is now against what was downloaded; every unit, date, cell, day and hour
  integrity     the stored GeoTIFFs (Sentinel-2 bands, Landsat bands, hourly weather, elevation, preprocessing and
                raster results) are present, open and unchanged since step 00c wrote them (sha256); a sample of the
                stored Sentinel-2 and Landsat pixels equals the source pixel by pixel
  sample values a random sample of export values recomputed from the sources with other methods
                (core/stac_verify.py) and compared within the tolerances below
Nothing is corrected here. FAIL = a dataset is incomplete or wrong (run step 00c again, it continues where it
stopped). NOT PRODUCED = step 00c could not make the dataset (Landsat EF and ETa without an EF function).
Pixel stacks (export px): made from the current rasters and crop map by the current Landsat rule, every file
unchanged (sha256), and a sample of dates and overpasses read again from the rasters: NDVI at the pixel centres; EF,
ETa and clear share of the pixel squares by a second method - the code of the field export E2
(stac_ingest.landsat_rows, overpass_rows) with each sampled square as a field polygon: equal within half a storage
step.
Exit code: 0 everything verified; 2 downloads verified, a dataset not produced; 1 a FAIL.
Reads: the files of step 00c for the run (or the test period --start / --end).
Writes: outputs/<run>/00d_verify/ verification.csv, summary.csv, sample_ndvi.csv, sample_landsat_clear.csv,
sample_weather.csv, sample_stored_s2_pixels.csv, sample_stored_landsat_pixels.csv, sample_elevation.csv,
sample_pixel_stacks.csv, manifest.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from core import config
from core import stac_ingest as si
from core import stac_verify as sv

# Largest accepted difference between an export value and its recomputation by the second method. Settings of this
# check only (no model value): the two methods differ in how the pixels on the edge of a unit are weighted.
TOLERANCE = config.read_yaml(config.ROOT / "config" / "project.yaml")["verification_tolerance"]   # settings
S2, LS, EF, WX, EL, REC = ("Sentinel-2 NDVI (E1)", "Landsat scenes and clear shares (E2)", "Landsat EF and ETa (E2)",
                           "ERA5-Land weather (E3)", "Elevation (E3)", "Download record")
RAS = "Preprocessing and raster results"
PX = "Pixel stacks (PX)"


def files_check(R, name, what, index):
    """Every GeoTIFF of a file table of step 00c is present, unchanged (sha256) and opens."""
    if not index.exists():
        R.add(name, f"{what}: present, unchanged and readable", "FAIL", f"no {index} - run step 00c")
        return
    listed, missing, changed, unreadable, size = sv.stored_files(index)
    R.ok(name, f"{what}: present, unchanged (sha256) and readable", listed > 0 and not (missing or changed or unreadable),
         f"{listed - len(missing) - len(changed) - len(unreadable)} of {listed} GeoTIFFs, {size / 1e6:,.0f} MB ({index.name})"
         + (f"; missing: {sv.short(missing)}" if missing else "") + (f"; changed: {sv.short(changed)}" if changed else "")
         + (f"; do not open: {sv.short(unreadable)}" if unreadable else ""))


def catalogue_check(R, name, now, table):
    have = set(table.item_id)
    new, gone = set(now) - have, have - set(now)
    R.ok(name, "every item the STAC catalogue lists now was downloaded", not new,
         f"catalogue {len(now)}, downloaded {len(have)}" + (f"; not downloaded: {sv.short(new)}" if new else ""))
    if gone:
        R.add(name, "downloaded items no longer in the catalogue", "WARN", sv.short(gone))
    bad = table[table.status != "read"]
    R.ok(name, "every item was read", bad.empty, f"{len(table) - len(bad)} of {len(table)} read"
         + (f"; failed: {sv.short(bad.item_id)}" if len(bad) else ""))


def sentinel2(R, cfg, P, fields, out, n, seed):
    folder, p_e1 = P["folder"]["sentinel2"], P["exports"]["e1_s2_perdate"]
    if not (folder / "s2_items.csv").exists() or not p_e1.exists():
        R.add(S2, "downloaded", "FAIL", f"missing: {folder / 's2_items.csv'} or {p_e1} - run step 00c")
        return {}
    items, e1 = pd.read_csv(folder / "s2_items.csv"), pd.read_csv(p_e1)
    units = fields                          # the polygons step 00c searched and stored (stac.sentinel2.coverage)
    if cfg["stac"]["sentinel2"].get("coverage", "sample") == "all_parcels" and cfg["aoi"]["inputs"].get("crop_mask"):
        units = si.load_parcels(cfg)
    catalogue_check(R, S2, sv.catalogue_now(cfg, units, P["start"], P["end"], "sentinel2"), items)
    recs = {i: si.read_json(folder / "granules" / f"{i}.json") for i in items.item_id}
    no_rec = [i for i, r in recs.items() if r is None]
    part = [i for i, r in recs.items() if r and r["files"] and set(r["files"]) != {"scl", "red", "nir"}]
    R.ok(S2, "every granule is stored as GeoTIFF with its record (red, NIR, scene classification)",
         not no_rec and not part, f"{len(recs) - len(no_rec)} of {len(recs)} granules with a record, "
         f"{sum(1 for r in recs.values() if r and r['files'])} with pixels of a unit"
         + (f"; no record: {sv.short(no_rec)}" if no_rec else "") + (f"; bands missing: {sv.short(part)}" if part else ""))
    files_check(R, S2, "stored Sentinel-2 bands", folder / "s2_files.csv")
    dates = sorted(items.date.unique())
    R.ok(S2, "dates inside the download period", dates[0] >= f"{P['start']:%Y-%m-%d}" and dates[-1] <= f"{P['end']:%Y-%m-%d}",
         f"{len(dates)} dates, {dates[0]} to {dates[-1]}; period {P['start']:%Y-%m-%d} to {P['end']:%Y-%m-%d}")
    R.ok(S2, "export has one row per unit and date", len(e1) == len(fields) * len(dates)
         and not e1.duplicated(["field_id", "date"]).any() and set(e1.field_id) == set(fields.field_id)
         and set(e1.date) == set(dates), f"{len(e1)} rows; {len(fields)} units x {len(dates)} dates = "
         f"{len(fields) * len(dates)}")
    has = e1[e1.ndvi.notna()].groupby("field_id").size().reindex(fields.field_id).fillna(0)
    R.ok(S2, "every unit has NDVI values", bool((has > 0).all()),
         f"{int((has == 0).sum())} units without; dates with NDVI per unit: min {int(has.min())}, median "
         f"{has.median():.0f}, max {int(has.max())}")
    R.ok(S2, "value ranges", bool(e1.ndvi.dropna().between(-1, 1).all() and e1.valid_fraction.between(0, 1 + 1e-9).all()
                                   and not e1.valid_fraction.isna().any()),
         f"NDVI {e1.ndvi.min():.3f} to {e1.ndvi.max():.3f}; valid_fraction {e1.valid_fraction.min():.3f} to "
         f"{e1.valid_fraction.max():.3f}")
    written = {}
    if n:
        t = sv.stored_pixels_s2(cfg, folder / "granules", items, n, seed)
        t.to_csv(out / "sample_stored_s2_pixels.csv", index=False)
        R.ok(S2, f"stored pixels of {t.item_id.nunique()} granules equal the source", len(t) > 0
             and int(t.pixels_different.sum()) == 0, f"{int(t.pixels_different.sum())} of {int(t.pixels.sum())} pixels "
             f"differ ({int(t.pixels_with_data.sum())} with data; red, NIR and scene classification in the window of "
             f"one unit per granule); seed {seed}; sample_stored_s2_pixels.csv")
        written["sample_stored_s2_pixels"] = out / "sample_stored_s2_pixels.csv"
    if n:
        t = sv.ndvi_sample(cfg, fields, e1, items, n, seed)
        t.to_csv(out / "sample_ndvi.csv", index=False)
        dn, dv = t.ndvi_diff.abs().max(), t.valid_fraction_diff.abs().max()
        R.ok(S2, f"sample of {len(t)} unit-dates recomputed from the source",
             bool(len(t)) and not t.check_ndvi.isna().any() and dn <= TOLERANCE["ndvi"]
             and dv <= TOLERANCE["valid_fraction"],
             f"largest |difference|: NDVI {dn:.4f} (tolerance {TOLERANCE['ndvi']}), valid_fraction {dv:.4f} "
             f"(tolerance {TOLERANCE['valid_fraction']}); seed {seed}; sample_ndvi.csv")
        written["sample_ndvi"] = out / "sample_ndvi.csv"
    return written


def landsat(R, cfg, P, fields, out, n, seed):
    folder = P["folder"]["landsat"]
    p_items, p_clear, p_e2 = folder / "landsat_items.csv", folder / "landsat_field_overpass_clear.csv", P["exports"]["e2_landsat_perdate"]
    written = {}
    if not p_items.exists() or not p_clear.exists():
        R.add(LS, "downloaded", "FAIL", f"missing: {p_items} or {p_clear} - run step 00c")
    else:
        items, clear = pd.read_csv(p_items), pd.read_csv(p_clear)
        catalogue_check(R, LS, sv.catalogue_now(cfg, fields, P["start"], P["end"], "landsat"), items)
        cov = clear.groupby("field_id").date.nunique().reindex(fields.field_id).fillna(0)
        R.ok(LS, "every unit is inside a scene", bool((cov > 0).all()) and set(clear.item_id) <= set(items.item_id),
             f"{len(clear)} unit-scene rows from {clear.item_id.nunique()} scenes; overpass dates per unit: min "
             f"{int(cov.min())}, median {cov.median():.0f}, max {int(cov.max())}")
        R.ok(LS, "value range of the clear share", bool(clear.valid_fraction.between(0, 1 + 1e-9).all()),
             f"{clear.valid_fraction.min():.3f} to {clear.valid_fraction.max():.3f}")
        cl = cfg["stac"]["landsat"]
        qa, keep = cl["qa_asset"], cl.get("keep_source_bands", True)
        recs = {i: si.read_json(folder / "scenes" / f"{i}.json") for i in items.item_id}
        if keep:
            bad = [i for i, r in recs.items() if r is None or (r.get("window") and qa not in (r.get("files") or {}))]
            R.ok(LS, f"every scene is stored as GeoTIFF with its record ({qa}; with an EF function the bands it reads)",
                 not bad, f"{len(recs) - len(bad)} of {len(recs)} scenes; bands stored: "
                 f"{pd.Series([b for r in recs.values() if r for b in (r.get('files') or {})]).value_counts().sort_index().to_dict()}"
                 + (f"; not stored: {sv.short(bad)}" if bad else ""))
            files_check(R, LS, "stored Landsat bands", folder / "landsat_files.csv")
        else:                                       # source bands deleted once the scene results were stored
            ras = P["rasters"] / "landsat"
            bad = [i for i, r in recs.items() if r is None or "rows" not in r
                   or (r.get("window") and not (ras / f"{i}_clear.tif").exists())]
            R.ok(LS, "every scene has its record and result rasters (source bands not kept: "
                 "stac.landsat.keep_source_bands false)", not bad, f"{len(recs) - len(bad)} of {len(recs)} scenes; "
                 f"clear mask, EF and ETa rasters in {ras}" + (f"; missing: {sv.short(bad)}" if bad else ""))
        if n and keep:
            t = sv.stored_pixels_landsat(cfg, folder / "scenes", items, n, seed)
            t.to_csv(out / "sample_stored_landsat_pixels.csv", index=False)
            R.ok(LS, f"stored pixels of {len(t)} bands equal the source", len(t) > 0
                 and int(t.pixels_different.sum()) == 0, f"{int(t.pixels_different.sum())} of {int(t.pixels.sum())} "
                 f"pixels differ (bands {sorted(t.band.unique())}, a 512 x 512 part of the scene window each); seed "
                 f"{seed}; sample_stored_landsat_pixels.csv")
            written["sample_stored_landsat_pixels"] = out / "sample_stored_landsat_pixels.csv"
        if n:
            t = sv.clear_sample(cfg, fields, clear, n, seed)
            t.to_csv(out / "sample_landsat_clear.csv", index=False)
            d = t.valid_fraction_diff.abs().max()
            R.ok(LS, f"sample of {len(t)} unit-scenes recomputed from the source",
                 bool(len(t)) and not t.check_valid_fraction.isna().any() and d <= TOLERANCE["valid_fraction"],
                 f"largest |difference| of the clear share {d:.4f} (tolerance {TOLERANCE['valid_fraction']}); seed "
                 f"{seed}; sample_landsat_clear.csv")
            written["sample_landsat_clear"] = out / "sample_landsat_clear.csv"
    if p_e2.exists():
        e2 = pd.read_csv(p_e2)
        R.ok(EF, "export has one row per unit and overpass, EF between 0 and 1",
             not e2.duplicated(["field_id", "date", "sensor"]).any() and bool(e2.ef.dropna().between(0, 1).all())
             and e2.ef.notna().any(), f"{len(e2)} rows, {int(e2.ef.notna().sum())} with EF, "
             f"{e2.ef.min():.3f} to {e2.ef.max():.3f}; function {cfg['stac']['landsat'].get('ef_function')}")
    else:
        R.add(EF, "export file", "NOT PRODUCED",
              f"{p_e2.name} does not exist: step 00c has not written EF and ETa (EF function: "
              f"{cfg['stac']['landsat'].get('ef_function') or 'none configured'}; with SEBAL run E3 first, then "
              "--only e2). Not a download failure - steps 01-10 cannot run without it")
    return written


def weather(R, cfg, P, fields, out, n, seed):
    folder, p_e3, p_el = P["folder"]["era5land"], P["exports"]["e3_weather_daily"], P["folder"]["dem"] / "cell_elevation.csv"
    cells = si.cells_of(fields)
    written = {}
    if not p_e3.exists():
        R.add(WX, "downloaded", "FAIL", f"missing: {p_e3} - run step 00c")
        return written
    e3 = pd.read_csv(p_e3, parse_dates=["date"])
    hours, tname, problems, facts = sv.weather_hours(cfg, cells, folder / "raw", P["start"], P["end"])
    R.ok(WX, "hourly GeoTIFFs: every hour, variable and cell; unchanged since they were written (sha256)", not problems,
         "; ".join(problems) if problems else
         f"{facts['files']} GeoTIFFs of {facts['months']} months, {facts['hours_present']} of {facts['hours_expected']} "
         f"hours ({facts['first_hour']} to {facts['last_hour']} UTC), {facts['cells']} cells, no missing value")
    for sub, var in (("raw", cfg["stac"]["era5land"]["variables"]), ("raw_precip", cfg["stac"]["era5land"].get("precipitation"))):
        if var and any((folder / sub).glob("*.nc")):
            k, diff = sv.netcdf_check(folder / sub, var)
            R.ok(WX, f"hourly GeoTIFFs equal the netCDF files still in {sub}/", not diff,
                 f"{k} GeoTIFFs compared value by value with the netCDF files of the Climate Data Store"
                 + (f"; {sv.short(diff)}" if diff else ""))
    n_days = (P["end"] - P["start"]).days + 1
    per = e3.groupby("cell").date.nunique()
    R.ok(WX, "export has every cell and day, no missing value", len(e3) == len(cells) * n_days
         and set(e3.cell) == set(cells.cell) and bool((per == n_days).all()) and not e3.isna().any().any()
         and e3.date.min() == P["start"] and e3.date.max() == P["end"],
         f"{len(e3)} rows; {len(cells)} cells x {n_days} days = {len(cells) * n_days}; "
         f"{int(e3.isna().sum().sum())} missing values; {e3.date.min():%Y-%m-%d} to {e3.date.max():%Y-%m-%d}")
    rng = {k: tuple(v) for k, v in cfg["data_qc"]["e3_ranges"].items() if k in e3}   # settings
    bad = {c: int((~e3[c].between(lo, hi)).sum()) for c, (lo, hi) in rng.items()}
    R.ok(WX, "physical ranges", sum(bad.values()) == 0 and bool((e3.tmin_c < e3.tmax_c).all()),
         "; ".join(f"{c} {e3[c].min():.2f} to {e3[c].max():.2f}" for c in rng) + f"; out of range {bad}")
    precip, pr = None, cfg["stac"]["era5land"].get("precipitation")
    if pr and "precip_mm" in e3:                     # precipitation: own hourly files in raw_precip/ (step 00c)
        ph, pt, pp, pf = sv.weather_hours(cfg, cells, folder / "raw_precip", P["start"], P["end"], variables=pr)
        R.ok(WX, "hourly precipitation GeoTIFFs: every hour and cell; unchanged since they were written", not pp,
             "; ".join(pp) if pp else f"{pf['files']} GeoTIFFs, {pf['hours_present']} of {pf['hours_expected']} hours "
             f"({pf['first_hour']} to {pf['last_hour']} UTC), {pf['cells']} cells, no missing value")
        if not pp:
            precip = (ph, pt, list(pr.values())[0])
    if n and hours is not None and not problems:
        t = sv.weather_sample(hours, tname, e3, n, seed, cfg["aoi"]["local_day_utc_offset_hours"], precip)
        t.to_csv(out / "sample_weather.csv", index=False)
        d = {c: float(t[f"{c}_diff"].abs().max()) for c in rng if f"{c}_diff" in t}
        unchecked = [c for c in rng if f"{c}_diff" not in t]
        R.ok(WX, f"sample of {len(t)} cell-days recomputed from the hourly files",
             max(d.values()) <= TOLERANCE["weather"] and not unchecked,
             "largest |difference|: " + ", ".join(f"{c} {v:.1e}" for c, v in d.items())
             + f" (tolerance {TOLERANCE['weather']}); seed {seed}; sample_weather.csv"
             + (f"; NOT recomputed: {unchecked}" if unchecked else ""))
        written["sample_weather"] = out / "sample_weather.csv"
    elif n:
        R.add(WX, "sample recomputed from the hourly files", "FAIL", "not possible: the hourly files are incomplete")
    if not p_el.exists():
        R.add(EL, "downloaded", "FAIL", f"missing: {p_el} - run step 00c")
    else:
        el = pd.read_csv(p_el)
        m = e3.drop_duplicates("cell")[["cell", "elev_m"]].merge(el[["cell", "elev_m", "n_pixels"]], on="cell", how="left",
                                                                  suffixes=("", "_dem"))
        R.ok(EL, "every cell has an elevation, the same as in the weather export", set(el.cell) >= set(cells.cell)
             and bool((m.n_pixels > 0).all()) and float((m.elev_m - m.elev_m_dem).abs().max()) < 1e-6,
             f"{len(el)} cells, {el.elev_m.min():.0f} to {el.elev_m.max():.0f} m, {int(el.n_pixels.min())} to "
             f"{int(el.n_pixels.max())} pixels per cell")
        cd = cfg["stac"]["dem"]
        p_tif, rec = P["folder"]["dem"] / f"{cd['collection']}_cells.tif", si.read_json(P["folder"]["dem"] / f"{cd['collection']}_cells.json")
        R.ok(EL, "elevation GeoTIFF present and unchanged since it was written (sha256)", bool(rec) and p_tif.exists()
             and config.sha256(p_tif) == rec["files"]["elevation"]["sha256"],
             f"{p_tif.name}, {len(rec['tiles'])} source tiles" if rec else f"no record of {p_tif.name} - run step 00c")
        if n and rec:
            t = sv.elevation_sample(cfg, cells, n, seed).merge(el[["cell", "elev_m", "n_pixels"]], on="cell")
            t["elev_diff"] = t.elev_m - t.check_elev_m
            t.to_csv(out / "sample_elevation.csv", index=False)
            R.ok(EL, f"sample of {len(t)} cells recomputed from the source tiles", len(t) > 0
                 and float(t.elev_diff.abs().max()) <= TOLERANCE["weather"] and bool((t.n_pixels == t.check_n_pixels).all()),
                 f"largest |difference| {t.elev_diff.abs().max():.1e} m (tolerance {TOLERANCE['weather']}), pixel "
                 f"counts {'equal' if (t.n_pixels == t.check_n_pixels).all() else 'DIFFERENT'}; seed {seed}; "
                 "sample_elevation.csv")
            written["sample_elevation"] = out / "sample_elevation.csv"
    return written


def pixel_stacks(R, cfg, P, out, n, seed, n_px=2000, n_sq=300):
    """Export px: made from the current rasters and crop map, files unchanged, values equal to the rasters."""
    from core import pixels as px
    folder = px.stack_folder(P)
    rec = si.read_json(folder / "record.json")
    if not rec:
        R.add(PX, "made", "FAIL", f"no {folder / 'record.json'} - run step 00c with export px")
        return {}
    bad = [k for k, f in rec["files"].items() if not (folder / k).exists() or config.sha256(folder / k) != f["sha256"]]
    R.ok(PX, "every file present and unchanged since step 00c wrote it (sha256)", not bad,
         f"{len(rec['files'])} files, {sum(f['bytes'] for f in rec['files'].values()) / 1e6:,.0f} MB"
         + (f"; missing or changed: {bad}" if bad else ""))
    if bad:
        return {}
    key, extent, recs = px.s2_grid(cfg, P)
    ovp = px.landsat_overpasses(cfg, P)
    parcels = si.load_parcels(cfg)
    now = {"grid": list(key), "extent": list(extent), "parcels": px.parcels_fingerprint(parcels),
           "ndvi": {r["date"]: r["files"]["ndvi"]["sha256"] for r in recs}}
    old = {k: rec["made_from"].get(k) for k in now}
    diff = [k for k in now if now[k] != old[k]]
    R.ok(PX, "made from the current NDVI rasters, grid and crop map", not diff,
         f"{len(recs)} dates, {len(parcels):,} parcels" + (f"; changed since: {diff} - run step 00c --only px" if diff else ""))
    have = {i for ids in ovp.item_ids for i in ids}
    R.ok(PX, "made from the current Landsat scenes", have == set(rec["made_from"]["landsat"]),
         f"{len(ovp)} overpasses, {len(have)} scenes")
    rule = (rec.get("scales") or {}).get("landsat_rule")
    R.ok(PX, "Landsat values of the pixel square (area-weighted, as E2; v0.9.0)", rule == px.LANDSAT_RULE,
         f"stacks made with: {rule or 'the Landsat pixel at the pixel centre (v0.8.0)'}"
         + ("" if rule == px.LANDSAT_RULE else " - run step 00c --only px"))
    if rule != px.LANDSAT_RULE:
        return {}
    st = px.Stacks(P)
    R.ok(PX, "one stack row per date and overpass, one column per pixel",
         list(st.nd.shape) == [len(recs), st.n] and list(st.ef.shape) == [len(ovp), st.n]
         and list(st.eta.shape) == [len(ovp), st.n] and list(st.clear.shape) == [len(ovp), st.n]
         and len(st.parcels) == len(parcels) and int(st.parcels.n_pixels.sum()) == st.n,
         f"NDVI {list(st.nd.shape)}, EF / ETa {list(st.ef.shape)}; {st.n:,} pixels, {len(st.parcels):,} parcels")
    if not n or diff:
        return {}
    rng = np.random.default_rng(seed)
    cols = np.sort(rng.choice(st.n, min(n_px, st.n), replace=False))
    rows, sc = [], st.sc
    for k in np.sort(rng.choice(len(recs), min(n, len(recs)), replace=False)):
        r = recs[k]
        v, _ = px.raster_values(P["rasters"] / "ndvi" / r["files"]["ndvi"]["file"], st.pix["row"][cols],
                                st.pix["col"][cols])
        s_ = px.decode(st.nd[k, cols], sc["ndvi_scale"])
        same_nan = bool((np.isnan(v) == np.isnan(s_)).all())
        d = float(np.nanmax(np.abs(v - s_))) if (~np.isnan(v)).any() else 0.0
        rows.append({"layer": "ndvi", "when": r["date"], "pixels": len(cols), "with_value": int((~np.isnan(v)).sum()),
                     "same_missing": same_nan, "max_abs_diff": d, "half_step": sc["ndvi_scale"] / 2})
    ras = P["rasters"] / "landsat"
    import geopandas as gpd
    from shapely import box
    lc = cols[np.sort(rng.choice(len(cols), min(n_sq, len(cols)), replace=False))]   # squares: polygon code (slower)
    hx = key[1] / 2
    squares = gpd.GeoDataFrame({"field_id": [f"PX{i}" for i in lc]},
                               geometry=box(st.pix["x"][lc] - hx, st.pix["y"][lc] - hx, st.pix["x"][lc] + hx,
                                            st.pix["y"][lc] + hx), crs=int(key[0]))
    fgrid = si.FieldGrid(squares)
    items = pd.read_csv(P["folder"]["landsat"] / "landsat_items.csv").set_index("item_id")
    for k in np.sort(rng.choice(len(ovp), min(n, len(ovp)), replace=False)):
        o = ovp.iloc[k]
        e2 = []
        for i in o.item_ids:
            e2 += si.landsat_rows(ras, i, items.date[i], items.sensor[i], squares, fgrid)
        e2 = pd.DataFrame(e2, columns=["field_id", "date", "sensor", "ef", "eta_mm_day", "valid_fraction", "item_id"])
        e2 = si.overpass_rows(e2).set_index("field_id").reindex(squares.field_id) if len(e2) else             pd.DataFrame(np.nan, index=squares.field_id, columns=["ef", "eta_mm_day", "valid_fraction"])
        for name, arr, col in (("ef", st.ef, "ef"), ("eta", st.eta, "eta_mm_day"), ("clear", st.clear, "valid_fraction")):
            v = e2[col].to_numpy(float)
            s_ = px.decode(arr[k, lc], sc[f"{name}_scale"])
            same_nan = bool((np.isnan(v) == np.isnan(s_)).all())
            d = float(np.nanmax(np.abs(v - s_))) if (~np.isnan(v)).any() else 0.0
            rows.append({"layer": name, "when": f"{o.date} {o.sensor}", "pixels": len(lc),
                         "with_value": int((~np.isnan(v)).sum()), "same_missing": same_nan,
                         "max_abs_diff": d, "half_step": sc[f"{name}_scale"] / 2})
    t = pd.DataFrame(rows)
    t.to_csv(out / "sample_pixel_stacks.csv", index=False)
    ok = bool(t.same_missing.all() and (t.max_abs_diff <= t.half_step * (1 + 1e-6)).all())
    R.ok(PX, f"sample of {int((t.layer == 'ndvi').sum())} dates and {int((t.layer == 'ef').sum())} overpasses read "
         "again from the rasters (NDVI at the pixel centres; Landsat by the E2 field code on the pixel squares)", ok,
         f"{len(cols)} pixels per date, {len(lc)} squares per overpass; largest |difference| NDVI "
         f"{t[t.layer == 'ndvi'].max_abs_diff.max():.1e}, EF {t[t.layer == 'ef'].max_abs_diff.max():.1e}, ETa "
         f"{t[t.layer == 'eta'].max_abs_diff.max():.1e}, clear share {t[t.layer == 'clear'].max_abs_diff.max():.1e} "
         f"(storage rounding: half a step); missing values {'equal' if t.same_missing.all() else 'DIFFERENT'}; seed "
         f"{seed}; sample_pixel_stacks.csv")
    return {"sample_pixel_stacks": out / "sample_pixel_stacks.csv"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run config in config/aoi/ (e.g. guna_2025_26)")
    ap.add_argument("--only", default="e1,e2,e3,px", help="datasets to verify, e.g. e1,e3")
    ap.add_argument("--start", help="test period start YYYY-MM-DD (verifies the files of that test period)")
    ap.add_argument("--end", help="test period end YYYY-MM-DD")
    ap.add_argument("--samples", type=int, default=8, help="values recomputed from the source per dataset (0 = none)")
    ap.add_argument("--seed", type=int, default=1, help="random seed of the samples")
    a = ap.parse_args()
    only = {x.strip().lower() for x in a.only.split(",")}
    if not only <= {"e1", "e2", "e3", "px"} or not only:
        raise SystemExit("--only takes e1, e2, e3, px")
    cfg = config.load(a.run)
    P = si.run_paths(cfg, a.start, a.end)
    out = config.output_dir(cfg, f"00d_verify{P['tag']}")
    config.check_writable(out)
    for old in out.glob("sample_*.csv"):
        old.unlink()                                               # samples of an earlier verification
    p_fields = config.step_output(cfg, "02_fields", "fields.gpkg")
    fields = si.load_fields(p_fields)
    print(f"run {cfg['run_id']}: {len(fields)} units, {fields.cell.nunique()} cells, {P['start']:%Y-%m-%d} to "
          f"{P['end']:%Y-%m-%d}" + (" (TEST period)" if P["test"] else ""), flush=True)
    R, written = sv.Report(), {}

    p_man = config.ROOT / "outputs" / cfg["run_id"] / f"00c_download{P['tag']}" / "manifest.json"
    if not p_man.exists():
        R.add(REC, "step 00c finished and recorded its files", "FAIL", f"no {p_man} - run step 00c")
    else:
        listed, changed, missing, m = sv.manifest_files(p_man)
        R.ok(REC, "files recorded by step 00c are present and unchanged (sha256)", not changed and not missing,
             f"{listed} files recorded on {m.get('run_at')}" + (f"; changed: {sv.short(changed)}" if changed else "")
             + (f"; missing: {sv.short(missing)}" if missing else ""))
        same = m.get("period") == [f"{P['start']:%Y-%m-%d}", f"{P['end']:%Y-%m-%d}"]
        R.ok(REC, "recorded period is the period of the run", same, f"recorded {m.get('period')}, exports made "
             f"{m.get('exports_made')}")
        not_made = sorted(only - set(m.get("exports_made") or []))
        if not_made:
            R.add(REC, "last run of step 00c made every dataset", "WARN",
                  f"not made in the last run: {not_made} (their files come from an earlier run of step 00c)")
    if "e1" in only:
        written.update(sentinel2(R, cfg, P, fields, out, a.samples, a.seed))
    if "e2" in only:
        written.update(landsat(R, cfg, P, fields, out, a.samples, a.seed))
    if "e3" in only:
        written.update(weather(R, cfg, P, fields, out, a.samples, a.seed))
    if "px" in only:
        written.update(pixel_stacks(R, cfg, P, out, a.samples, a.seed))
    files_check(R, RAS, "GeoTIFFs of the preprocessing and raster results", P["rasters"] / "raster_files.csv")

    t = R.table()
    t.to_csv(out / "verification.csv", index=False)
    rank = {"FAIL": 0, "NOT PRODUCED": 1, "WARN": 2, "PASS": 3}
    verdict = {"FAIL": "INCOMPLETE OR WRONG", "NOT PRODUCED": "NOT PRODUCED", "WARN": "VERIFIED, with a warning",
               "PASS": "VERIFIED"}
    summary = pd.DataFrame([{"dataset": d, "result": verdict[min(g.status, key=rank.get)], "checks": len(g),
                             "passed": int((g.status == "PASS").sum()),
                             "to_look_at": " | ".join(f"{r.check}: {r.detail}" for r in g.itertuples() if r.status != "PASS")}
                            for d, g in t.groupby("dataset", sort=False)])
    summary.to_csv(out / "summary.csv", index=False)
    inputs = {"fields": p_fields, **{k: v for k, v in P["exports"].items() if v.exists()}}
    config.write_manifest(out, "00d_verify", inputs,
                          {"verification": out / "verification.csv", "summary": out / "summary.csv", **written},
                          {"period": [f"{P['start']:%Y-%m-%d}", f"{P['end']:%Y-%m-%d}"], "test_period": P["test"],
                           "datasets": sorted(only), "samples": a.samples, "seed": a.seed, "tolerance": TOLERANCE})
    pd.set_option("display.width", 250, "display.max_colwidth", 150)
    print("\n" + summary[["dataset", "result", "checks", "passed"]].to_string(index=False))
    for r in summary.itertuples():
        if r.to_look_at:
            print(f"\n{r.dataset}: {r.to_look_at}")
    print(f"\nverification, samples and manifest: {out}")
    if (t.status == "FAIL").any():
        print("\nstep 00d: FAIL - a dataset is incomplete or wrong (see above)")
        sys.exit(1)
    if (t.status == "NOT PRODUCED").any():
        print("\nstep 00d: every downloaded dataset is verified; a dataset was not produced (see above)")
        sys.exit(2)
    print("\nstep 00d: every dataset is complete and verified")


if __name__ == "__main__":
    main()
