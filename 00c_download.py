"""Step 00c - data download through STAC APIs: the exports E1, E2, E3 of the units of the run.

Units: the field sample of step 02 (run with a crop map) or the units of step 00b (run from an area polygon).
No Google Earth Engine. Sources (config/project.yaml, section stac):
  E1  Sentinel-2 L2A (STAC collection sentinel-2-l2a): NDVI and clear share per field and acquisition date
  E3  ERA5-Land hourly (Climate Data Store, STAC catalogue; retrieval with cdsapi) -> daily values per 0.1 degree
      cell and local calendar day; terrain elevation from the STAC DEM collection. Made before E2: the SEBAL EF
      function reads its hourly and daily weather.
  E2  Landsat 8/9 Collection 2 Level-2 (landsat-c2-l2): clear share per field and overpass; EF and ETa per field
      by the energy-balance function of stac.landsat.ef_function (core.ef_sebal: SEBAL; hot / cold pixel of every
      scene in data/stac/landsat/<run>/sebal_anchors.csv) - without a function E2 is NOT written (the clear shares
      are kept in data/stac/landsat/<run>/)
  PX  pixel stacks (core/pixels.py): every 10 m Sentinel-2 pixel whose centre is inside a parcel of the crop map,
      with its NDVI per date and per overpass the Landsat EF, ETa and clear share of the 10 m pixel square,
      area-weighted over the clear 30 m pixels it overlaps as E2 computes them for a field (v0.9.0), read from the
      rasters of E1 and E2 (no download; needs stac.sentinel2.coverage all_parcels) -> <folder of the export
      files>/<run>_pixels/ (ndvi.npy, landsat_ef.npy, landsat_eta.npy, landsat_clear.npy, pixels.npz, parcels.csv,
      record.json).
      Read by the pixel yield model (step 05b). Made again only when the rasters, the crop map or the settings changed
Period: season window start - lead_days to the window end. Fields: outputs/<run>/02_fields/fields.gpkg.
Every satellite and weather dataset is stored as GeoTIFF and the export values are computed from the stored files
(core/stac_ingest.py): Sentinel-2 bands per granule, Landsat bands per scene, hourly weather per month and variable,
elevation. The preprocessing results are GeoTIFFs too: clear mask, reflectance and NDVI per Sentinel-2 date, clear
mask, EF and ETa per Landsat scene, daily weather with Tmean and ETo.
Storage settings (v0.7.0): stac.sentinel2.coverage all_parcels stores and preprocesses the Sentinel-2 pixels of every
parcel of the crop map (read by export px for the pixel stacks; E1 stays the table of the sample);
stac.landsat.keep_source_bands false deletes the Landsat bands of a scene once its results are stored (clear mask,
EF and ETa rasters, field rows) - a scene computed again (changed SEBAL settings) downloads its bands again.
Writes: the export files of the run config (field_exports), the item tables, file tables and GeoTIFFs in
data/stac/<source>/<run>/, the preprocessing and raster results in <folder of the export files>/<run>_rasters/,
qc_checks.csv and manifest in outputs/<run>/00c_download/.
Test on a short period first: --start 2026-01-01 --end 2026-01-07 writes to data/_test/ and never to the export files.
Granules, scenes and weather months already stored are kept: an interrupted run continues where it stopped.
Exit code: 0 complete; 2 everything downloaded but E2 not written (no EF function); 3 a granule, scene or weather
request could not be retrieved (run the step again); 1 any other stop.
Next: step 00d (00d_verify.py) verifies the downloads.
"""
import argparse
import math
import sys

import pandas as pd

from core import config
from core import stac_ingest as si


class QC:
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail):
        self.rows.append({"check": name, "status": status, "detail": detail})

    def ok(self, name, passed, detail, warn=False):
        self.add(name, "PASS" if passed else ("WARN" if warn else "FAIL"), detail)

    def table(self):
        return pd.DataFrame(self.rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run config in config/aoi/ (e.g. dewas_2025_26)")
    ap.add_argument("--only", default="e1,e2,e3,px", help="exports to make, e.g. e1,e3 (px: pixel stacks from the "
                    "stored rasters)")
    ap.add_argument("--start", help="test period start YYYY-MM-DD (writes to data/_test/)")
    ap.add_argument("--end", help="test period end YYYY-MM-DD (writes to data/_test/)")
    ap.add_argument("--workers", type=int, help="parallel downloads of this run (default: config stac.max_workers); "
                    "use fewer for large units - every granule in work holds the pixels of all its units in memory")
    args = ap.parse_args()
    cfg = config.load(args.run)
    if args.workers:
        if args.workers < 1:
            raise SystemExit("--workers must be 1 or more")
        cfg["stac"]["max_workers"] = args.workers
    only = {x.strip().lower() for x in args.only.split(",")}
    if not only <= {"e1", "e2", "e3", "px"} or not only:
        raise SystemExit("--only takes e1, e2, e3, px")
    P = si.run_paths(cfg, args.start, args.end)
    start, end, test, exports, folder, rasters = (P["start"], P["end"], P["test"], P["exports"], P["folder"],
                                                  P["rasters"])
    if end < start:
        raise SystemExit("end before start")
    run = cfg["run_id"]
    out = config.output_dir(cfg, f"00c_download{P['tag']}")
    for p in exports.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    config.check_writable(out)
    p_fields = config.step_output(cfg, "02_fields", "fields.gpkg")
    fields = si.load_fields(p_fields)
    c2, cl = cfg["stac"]["sentinel2"], cfg["stac"]["landsat"]
    units = None                                                  # pixels stored for these polygons (E1)
    if c2.get("coverage", "sample") == "all_parcels" and cfg["aoi"]["inputs"].get("crop_mask"):
        units = si.load_parcels(cfg)
    n_days = (end - start).days + 1
    qc, written, blocked = QC(), {}, None
    print(f"run {run}: {len(fields)} fields, {fields.cell.nunique()} cells, {start:%Y-%m-%d} to {end:%Y-%m-%d}"
          + (" (TEST period)" if test else "")
          + (f"; Sentinel-2 pixels of all {len(units)} parcels of the crop map" if units is not None else ""), flush=True)

    if "e1" in only:
        e1, meta = si.s2_export(cfg, fields, start, end, folder["sentinel2"], rasters, units)
        config.to_csv_atomic(e1, exports["e1_s2_perdate"], index=False)
        written.update({"e1_s2_perdate": exports["e1_s2_perdate"], "s2_items": folder["sentinel2"] / "s2_items.csv",
                        "s2_files": folder["sentinel2"] / "s2_files.csv"})
        tifs = pd.read_csv(folder["sentinel2"] / "s2_files.csv")
        per = tifs.file.str.rsplit("_", n=1).str[0].str.split("/").str[-1].value_counts().reindex(meta.item_id).fillna(0)
        qc.ok("E1 stored as GeoTIFF: red, NIR and scene classification of every granule", bool(per.isin([0, 3]).all()),
              f"{len(tifs)} GeoTIFFs of {int((per == 3).sum())} granules, {tifs.bytes.sum() / 1e6:,.0f} MB in "
              f"{folder['sentinel2'] / 'granules'}; {int((per == 0).sum())} granules without a pixel of a unit; "
              f"pixel windows of {'all ' + str(len(units)) + ' parcels of the crop map' if units is not None else 'the units'}")
        kinds = ["clear", *(["red", "nir"] if c2.get("reflectance_rasters", True) else []), "ndvi"]
        qc.ok("E1 preprocessing rasters of every date: " + ", ".join(kinds),
              all((rasters / "ndvi" / f"{run}_{k}_{d}.tif").exists() for d in meta.date.unique() for k in kinds),
              f"{meta.date.nunique()} dates x {len(kinds)} GeoTIFFs in {rasters / 'ndvi'} (10 m, grid of the granules); "
              "the table E1 is computed from the NDVI and clear rasters")
        qc.add("E1 source", "INFO", f"{c2['api']} collection {c2['collection']}: {len(meta)} granules, "
               f"{meta.date.nunique()} dates {meta.date.min()} to {meta.date.max()}, tiles "
               f"{meta.mgrs_tile.value_counts().sort_index().to_dict()}, processing baselines "
               f"{sorted(meta.processing_baseline.astype(str).unique())}; reflectance = (DN + BOA_ADD_OFFSET) / "
               f"BOA_QUANTIFICATION_VALUE from each product's metadata ({c2['metadata_asset']}), offsets found: "
               f"{sorted(meta.boa_add_offset.dropna().astype(str).unique()) if 'boa_add_offset' in meta else 'n/a'}")
        qc.ok("E1 one row per field and date", len(e1) == len(fields) * meta.date.nunique()
              and not e1.duplicated(["field_id", "date"]).any(),
              f"{len(e1)} rows = {len(fields)} fields x {meta.date.nunique()} dates")
        has = e1[e1.ndvi.notna()].groupby("field_id").size().reindex(fields.field_id).fillna(0)
        qc.ok("E1 every field has an NDVI value", bool((has > 0).all()),
              f"{int((has == 0).sum())} fields without; NDVI dates per field: min {int(has.min())}, median "
              f"{has.median():.0f}, max {int(has.max())}", warn=True)
        qc.ok("E1 ranges", bool(e1.ndvi.dropna().between(-1, 1).all() and e1.valid_fraction.between(0, 1 + 1e-9).all()),
              f"NDVI {e1.ndvi.min():.3f} to {e1.ndvi.max():.3f}; valid_fraction {e1.valid_fraction.min():.3f} to "
              f"{e1.valid_fraction.max():.3f}")
        mv = cfg["ndvi_cleaning"]["min_valid_fraction"]
        clear = e1[e1.valid_fraction >= mv].groupby("field_id").size().reindex(fields.field_id).fillna(0)
        qc.add(f"E1 dates with valid_fraction >= {mv} (ndvi_cleaning.min_valid_fraction)", "INFO",
               f"{int((e1.valid_fraction >= mv).sum())} of {len(e1)} rows; per field: min {int(clear.min())}, median "
               f"{clear.median():.0f}, max {int(clear.max())}")

    if "e3" in only:                     # before E2: the SEBAL EF function reads the weather of E3
        off = cfg["aoi"].get("local_day_utc_offset_hours")
        if off is None:
            raise SystemExit("run config: local_day_utc_offset_hours missing (local calendar day of the daily weather)")
        e3, elev, files, ssrd_min = si.weather_export(cfg, fields, start, end, folder["era5land"], folder["dem"], off)
        config.to_csv_atomic(e3, exports["e3_weather_daily"], index=False, date_format="%Y-%m-%d")
        ce, cd = cfg["stac"]["era5land"], cfg["stac"]["dem"]
        written.update({"e3_weather_daily": exports["e3_weather_daily"], "era5land_raw": folder["era5land"] / "raw",
                        **({"era5land_raw_precip": folder["era5land"] / "raw_precip"}
                           if (folder["era5land"] / "raw_precip").exists() else {}),
                        "era5land_collection": folder["era5land"] / "era5land_collection.json",
                        "cell_elevation": folder["dem"] / "cell_elevation.csv",
                        "elevation_tif": folder["dem"] / f"{cd['collection']}_cells.tif",
                        "elevation_record": folder["dem"] / f"{cd['collection']}_cells.json"})
        written.update(si.weather_rasters(cfg, e3, rasters / "weather"))
        qc.add("E3 source", "INFO", f"{ce['catalogue']}/collections/{ce['dataset']} (STAC collection; {len(files)} "
               f"months of hourly values retrieved with cdsapi as netCDF and stored as GeoTIFF per variable, one band "
               f"per hour); local day = UTC{off:+g} h; elevation: {cd['api']} collection "
               f"{cd['collection']}, mean of a {cd['window_m']} m window at the cell centre")
        var = {**ce["variables"], **(ce.get("precipitation") or {})}
        tifs = [p for s in files for p in si.era5_tifs(s, (si.read_json(s.with_name(s.name + ".tif.json")) or {})
                                                       .get("variables") or var).values()]
        qc.ok("E3 stored as GeoTIFF: hourly values of every month and variable, elevation of the cells",
              all(p.exists() for p in tifs) and written["elevation_tif"].exists(),
              f"{len(tifs)} hourly GeoTIFFs of {len(files)} months in {folder['era5land']}; "
              f"{written['elevation_tif'].name}; daily rasters (with Tmean and FAO-56 ETo) in {rasters / 'weather'}")
        qc.ok("E3 complete: every cell and day", len(e3) == e3.cell.nunique() * n_days
              and e3.cell.nunique() == fields.cell.nunique() and not e3.isna().any().any(),
              f"{len(e3)} rows = {e3.cell.nunique()} cells x {n_days} days, {int(e3.isna().sum().sum())} missing values")
        rng = {k: tuple(v) for k, v in cfg["data_qc"]["e3_ranges"].items() if k in e3}   # settings
        bad = {c: int((~e3[c].between(lo, hi)).sum()) for c, (lo, hi) in rng.items()}
        qc.ok("E3 physical ranges", sum(bad.values()) == 0 and bool((e3.tmin_c < e3.tmax_c).all()),
              "; ".join(f"{c} {e3[c].min():.2f} to {e3[c].max():.2f}" for c in rng) + f"; out of range {bad}")
        qc.add("E3 smallest hourly radiation (difference of the accumulated values)", "INFO", f"{ssrd_min:.1f} J m-2")
        qc.add("E3 elevation", "INFO", f"{elev.elev_m.min():.0f} to {elev.elev_m.max():.0f} m, "
               f"{int(elev.n_pixels.min())} to {int(elev.n_pixels.max())} pixels per cell")

    if "e2" in only:
        p_anchor = folder["landsat"] / "sebal_anchors.csv"       # rebuilt from the scene results, one row per scene
        if cfg["stac"]["landsat"].get("ef_function"):                # the EF function reads the weather of E3
            need = [exports["e3_weather_daily"], folder["era5land"] / "raw", folder["dem"] / "cell_elevation.csv"]
            miss = [str(p) for p in need if not p.exists()]
            if miss:
                raise SystemExit("E2 with the EF function needs E3 first - missing: " + "; ".join(miss)
                                 + f"\nrun: python 00c_download.py --run {run} --only e3")
        soil = None
        if cfg["sebal"]["hot_pixel"]["method"] == "soil_water_balance" and cfg["stac"]["landsat"].get("ef_function"):
            from core import soil_evaporation as se
            soil = se.load_soil_units(cfg)                          # tehsil soils for the hot-pixel water balance
            qc.add("E2 SEBAL hot pixel: bare-soil water balance (FAO-56 ch. 7)", "INFO",
                   f"{len(soil)} tehsils, TEW {soil.tew_mm.min():.1f}-{soil.tew_mm.max():.1f} mm (eq. 73, Ze "
                   f"{cfg['sebal']['hot_pixel']['ze_m']} m), REW {cfg['sebal']['hot_pixel']['rew_mm']} mm")
        b = fields.total_bounds                                     # DEM box: fields + scene margin + 2 km
        m_km = cfg["stac"]["landsat"]["context_km"] + 2
        dlat, dlon = m_km / 111.32, m_km / (111.32 * math.cos(math.radians((b[1] + b[3]) / 2)))
        context = {"cfg": cfg, "era5_raw": folder["era5land"] / "raw", "e3_file": exports["e3_weather_daily"],
                   "cell_elevation": folder["dem"] / "cell_elevation.csv",
                   "utc_offset": cfg["aoi"].get("local_day_utc_offset_hours"),
                   "dem_file": folder["dem"] / "nasadem_mosaic.tif", "soil_units": soil,
                   "soil_file": config.input_path(cfg, "soil_units") if soil is not None else None,
                   "dem_bbox": [round(float(b[0] - dlon), 4), round(float(b[1] - dlat), 4),
                                round(float(b[2] + dlon), 4), round(float(b[3] + dlat), 4)]}
        e2, meta, has_ef = si.landsat_export(cfg, fields, start, end, folder["landsat"], context, rasters)
        if p_anchor.exists():
            written["sebal_anchors"] = p_anchor
            an = pd.read_csv(p_anchor)
            qc.add("E2 SEBAL hot / cold pixels per scene", "INFO",
                   f"{int(an.status.str.startswith('ok').sum())} of {len(an)} scenes with EF; not converged: "
                   f"{int(an.status.str.contains('not converged').sum())}; see {p_anchor}")
        p_clear = folder["landsat"] / "landsat_field_overpass_clear.csv"
        config.to_csv_atomic(e2[["field_id", "date", "sensor", "valid_fraction", "item_id"]], p_clear, index=False)
        written.update({"landsat_items": folder["landsat"] / "landsat_items.csv", "landsat_clear": p_clear,
                        "landsat_files": folder["landsat"] / "landsat_files.csv"})
        tifs = pd.read_csv(folder["landsat"] / "landsat_files.csv")
        band = tifs.file.str.extract(r"_02_T\d_(.+)\.tif$")[0] if len(tifs) else pd.Series(dtype=str)
        if cl.get("keep_source_bands", True):
            stored = meta.item_id[[(folder["landsat"] / "scenes" / f"{i}_{cl['qa_asset']}.tif").exists()
                                   for i in meta.item_id]]
            qc.ok("E2 stored as GeoTIFF: the bands of every scene that are read", len(stored) > 0
                  and tifs.file.map(lambda f: (folder["landsat"] / f).exists()).all(),
                  f"{len(tifs)} GeoTIFFs of {len(stored)} scenes, {tifs.bytes.sum() / 1e6:,.0f} MB in "
                  f"{folder['landsat'] / 'scenes'} (scene window: all units + {cl['context_km']} km); bands: "
                  f"{band.value_counts().sort_index().to_dict()}; {len(meta) - len(stored)} scenes outside the window; "
                  f"clear mask per scene (and EF function results {cl.get('result_rasters') or []}) in "
                  f"{rasters / 'landsat'}")
        else:                                                     # results kept, source bands deleted
            want = ["clear", *[n for n in (cl.get("result_rasters") or []) if has_ef]]
            stored = meta.item_id[[(rasters / "landsat" / f"{i}_clear.tif").exists() for i in meta.item_id]]
            ok = all((rasters / "landsat" / f"{i}_{n}.tif").exists() for i in stored for n in want)
            qc.ok("E2 scene results stored as GeoTIFF (source bands not kept: stac.landsat.keep_source_bands false)",
                  len(stored) > 0 and ok, f"{len(stored)} scenes with {want} in {rasters / 'landsat'} (scene window: "
                  f"all units + {cl['context_km']} km); {len(meta) - len(stored)} scenes outside the window; "
                  f"{len(tifs)} source bands left in {folder['landsat'] / 'scenes'}")
        qc.add("E2 source", "INFO", f"{cl['api']} collection {cl['collection']} "
               f"({cl.get('collection_category') or 'all categories'}): {len(meta)} scenes, "
               f"{meta.groupby('sensor').date.nunique().to_dict()} dates, {meta.date.min()} to {meta.date.max()}, "
               f"path/row {(meta.wrs_path.astype(str) + '/' + meta.wrs_row.astype(str)).value_counts().sort_index().to_dict()}")
        cov = e2.groupby("field_id").date.nunique().reindex(fields.field_id).fillna(0)
        qc.ok("E2 every field is inside a Landsat scene", bool((cov > 0).all()),
              f"{len(e2)} field-overpass rows; overpass dates per field: min {int(cov.min())}, median "
              f"{cov.median():.0f}, max {int(cov.max())}", warn=True)
        qc.ok("E2 valid_fraction range", bool(e2.valid_fraction.between(0, 1 + 1e-9).all()),
              f"{e2.valid_fraction.min():.3f} to {e2.valid_fraction.max():.3f}; rows with valid_fraction > 0: "
              f"{int((e2.valid_fraction > 0).sum())}")
        dup = int(e2.duplicated(["field_id", "date", "sensor"]).sum())
        qc.add("E2 fields in the overlap of two scenes of one overpass", "INFO",
               f"{dup} rows; the export has one row per field and overpass (mean EF and ETa of the two scenes, "
               "larger clear share)")
        if has_ef:
            one = si.overpass_rows(e2)                                # one row per field and overpass (+ Rn, G, H, LE)
            config.to_csv_atomic(one, exports["e2_landsat_perdate"], index=False)
            written["e2_landsat_perdate"] = exports["e2_landsat_perdate"]
            ef = one.ef.dropna()
            qc.ok("E2 EF range", bool(ef.between(0, 1).all()) and len(ef) > 0,
                  f"{len(ef)} values, {ef.min():.3f} to {ef.max():.3f}; function {cl['ef_function']}")
        else:
            blocked = ("E2 not written: EF = lambda E / (Rn - G) needs an energy-balance function "
                       "(config stac.landsat.ef_function). The v3 exports used the team's SEBAL / METRIC function, "
                       "which is not in this framework. Clear shares per field and overpass: " + str(p_clear))
            qc.add("E2 EF and ETa", "BLOCKED", blocked)

    if "px" in only:                     # after E1 and E2: the pixel stacks are read from their rasters
        from core import pixels as px
        if c2.get("coverage", "sample") != "all_parcels" or not cfg["aoi"]["inputs"].get("crop_mask"):
            raise SystemExit("export px needs a crop map and stac.sentinel2.coverage all_parcels (the Sentinel-2 "
                             "rasters must hold every parcel); run e1 with that setting first")
        parcels = units if units is not None else si.load_parcels(cfg)
        rec, f = px.build_stacks(cfg, P, parcels)
        folder_px = px.stack_folder(P)
        written.update({f"pixels_{n.replace('.', '_')}": folder_px / n for n in rec["files"]})
        written["pixels_record"] = folder_px / "record.json"
        mb = sum(v["bytes"] for v in rec["files"].values()) / 1e6
        qc.ok("PX pixel stacks of every parcel of the crop map", f["pixels"] > 0 and f["parcels"] == len(parcels),
              f"{f['pixels']:,} pixels of {f['parcels']:,} parcels (rule: {f['pixel_rules']}); {f['dates']} "
              f"Sentinel-2 dates, {f['overpasses']} Landsat overpasses; {mb:,.0f} MB in {folder_px}")
        no_px = {k: v for k, v in f["pixel_rules"].items() if k not in ("centre inside", "representative point")}
        qc.ok("PX every parcel has a pixel", not no_px, f"parcels without a pixel: {no_px or 'none'}", warn=True)
        n_all = max(f["pixels"] * f["dates"], 1)
        qc.add("PX pixel-dates with NDVI (clear Sentinel-2 pixel)", "INFO",
               f"{f['pixel_dates_with_ndvi']:,} of {f['pixels'] * f['dates']:,} "
               f"({100 * f['pixel_dates_with_ndvi'] / n_all:.0f} %)")
        n_all = max(f["pixels"] * f["overpasses"], 1)
        qc.ok("PX pixel-overpasses with Landsat EF (clear Landsat pixel)", f["pixel_overpasses_with_ef"] > 0,
              f"{f['pixel_overpasses_with_ef']:,} of {f['pixels'] * f['overpasses']:,} "
              f"({100 * f['pixel_overpasses_with_ef'] / n_all:.0f} %); ETa {f['pixel_overpasses_with_eta']:,}")
        qc.add("PX storage", "INFO", f"int16 = value / scale ({rec['scales']}), {px.NODATA} = no value; made "
               f"{rec['made_at']} in {f['minutes']} min")

    if rasters.exists():
        ras = si.raster_index(rasters)
        written["raster_files"] = rasters / "raster_files.csv"
        qc.add("preprocessing and raster results", "INFO", f"{len(ras)} GeoTIFFs, {ras.bytes.sum() / 1e6:,.0f} MB in "
               f"{rasters} (raster_files.csv: bytes and sha256 of each)")
    t = qc.table()
    config.to_csv_atomic(t, out / "qc_checks.csv", index=False)
    # a step with a FAIL check leaves no manifest: run.py then never treats it as complete
    if not (t.status == "FAIL").any():
        config.write_manifest(out, "00c_download", {"fields": p_fields}, written,
                              {"period": [f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"], "test_period": test,
                               "exports_made": sorted(only), "stac": cfg["stac"], "blocked": blocked})
    pd.set_option("display.width", 250, "display.max_colwidth", 200)
    print(t.to_string(index=False))
    print(f"\nwritten: " + "; ".join(str(p) for p in written.values()))
    print(f"QC and manifest: {out}")
    if (t.status == "FAIL").any():
        raise SystemExit("step 00c stopped: FAIL checks above")
    if blocked:
        print("step 00c incomplete: " + blocked)
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except si.DownloadError as e:
        print(f"step 00c stopped, download not complete: {e}")
        sys.exit(3)
