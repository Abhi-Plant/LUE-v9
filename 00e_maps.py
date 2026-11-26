"""Step 00e - maps: a JPG map of every GeoTIFF of step 00c, for looking at the data.

One map per GeoTIFF (core/quicklook.py):
  s2        Sentinel-2 bands of every granule (B04, B08 as reflectance; SCL as classes)
  landsat   Landsat bands of every scene (QA_PIXEL as clear / not clear; reflectance; surface temperature)
  weather   hourly ERA5-Land GeoTIFFs (mean of the hours of the month)
  dem       elevation
  rasters   preprocessing and raster results: clear mask, reflectance and NDVI per Sentinel-2 date; clear mask, EF and
            ETa per Landsat scene; daily weather (mean of the days), Tmean, ETo, elevation of the cells
A map shows the file reduced to at most 1100 pixels on its longer side; a file with several bands is shown as the
mean of its bands. Scattered small units are enlarged by 2 map pixels so that they can be seen (written on the map).
The maps are pictures only: no step reads them and no value of the model is computed here.
A map that exists and is newer than its GeoTIFF is kept (--redo makes every map again).
Reads: the GeoTIFFs of step 00c for the run (or the test period --start / --end).
Writes: data/maps/<run>/<group>/<name of the GeoTIFF>.jpg (test period: data/_test/<run>_TEST_<dates>/maps/<run>/),
outputs/<run>/00e_maps/maps.csv (one row per map: what is shown, value range, reduction, status) and manifest.
Exit code: 0 every map made; 1 a map could not be made (maps.csv names it).
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from core import config, quicklook
from core import stac_ingest as si

_cfg = {}


def sources(P):
    """{group: [(folder of GeoTIFFs, map sub-folder)]} of a run."""
    f, r = P["folder"], P["rasters"]
    return {"s2": [(f["sentinel2"] / "granules", "sentinel2_granules")],
            "landsat": [(f["landsat"] / "scenes", "landsat_scenes")],
            "weather": [(f["era5land"] / "raw", "era5land_hourly"), (f["era5land"] / "raw_precip", "era5land_hourly_precip")],
            "dem": [(f["dem"], "elevation")],
            "rasters": [(r / "ndvi", "sentinel2_dates"), (r / "landsat", "landsat_results"), (r / "weather", "weather_daily")]}


def job(tif, jpg, run, gpkg):
    """One map, in a worker process."""
    if run not in _cfg:
        _cfg[run] = config.load(run)
    try:
        return {**quicklook.make_map(tif, jpg, _cfg[run], gpkg), "status": "made"}
    except Exception as e:  # noqa: BLE001 - a map that cannot be made is reported, the others are made
        return {"tif": tif.name, "jpg": jpg.name, "status": f"failed: {type(e).__name__}: {str(e)[:200]}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run config in config/aoi/ (e.g. dewas_2025_26)")
    ap.add_argument("--only", default="s2,landsat,weather,dem,rasters", help="groups to map, e.g. rasters,dem")
    ap.add_argument("--start", help="test period start YYYY-MM-DD (maps of the files of that test period)")
    ap.add_argument("--end", help="test period end YYYY-MM-DD")
    ap.add_argument("--workers", type=int, default=4, help="maps made at the same time (default 4)")
    ap.add_argument("--redo", action="store_true", help="make every map again, also the ones that exist")
    a = ap.parse_args()
    cfg = config.load(a.run)
    P = si.run_paths(cfg, a.start, a.end)
    src = sources(P)
    only = [x.strip().lower() for x in a.only.split(",") if x.strip()]
    if not only or not set(only) <= set(src):
        raise SystemExit(f"--only takes {', '.join(src)}")
    out = config.output_dir(cfg, f"00e_maps{P['tag']}")
    config.check_writable(out)
    maps = P["exports"]["e1_s2_perdate"].parent.parent / "maps" / cfg["run_id"]
    gpkg = str(config.step_output(cfg, "02_fields", "fields.gpkg"))
    todo, rows = [], []
    for g in only:
        for folder, sub in src[g]:
            for tif in sorted(folder.glob("*.tif")) if folder.exists() else []:
                jpg = maps / sub / (tif.stem + ".jpg")
                if not a.redo and jpg.exists() and jpg.stat().st_mtime >= tif.stat().st_mtime:
                    rows.append({"group": g, "folder": sub, "tif": tif.name, "jpg": jpg.name, "status": "kept"})
                else:
                    todo.append((g, sub, tif, jpg))
    print(f"run {cfg['run_id']}: {len(todo) + len(rows)} GeoTIFFs in {', '.join(only)}; {len(rows)} maps kept, "
          f"{len(todo)} to make -> {maps}" + (" (TEST period)" if P["test"] else ""), flush=True)
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(max(1, a.workers)) as ex:
            fut = {ex.submit(job, tif, jpg, cfg["run_id"], gpkg): (g, sub) for g, sub, tif, jpg in todo}
            for n, f in enumerate(as_completed(fut), 1):
                g, sub = fut[f]
                rows.append({"group": g, "folder": sub, **f.result()})
                if n % 100 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"  maps: {n}/{len(todo)}, {el / 60:.1f} min, about {el / n * (len(todo) - n) / 60:.0f} min left",
                          flush=True)
    cols = ["group", "folder", "tif", "jpg", "status", "shown", "kind", "classes", "value_min", "value_max",
            "pixels_with_data_share", "reduction", "bands"]
    t = pd.DataFrame(rows, columns=cols).sort_values(["group", "folder", "tif"]).reset_index(drop=True)
    config.to_csv_atomic(t, out / "maps.csv", index=False)
    failed = t[t.status.str.startswith("failed")]
    summary = t.groupby(["group", "folder"]).status.agg(maps="size", made=lambda s: int((s == "made").sum()),
                                                        kept=lambda s: int((s == "kept").sum()),
                                                        failed=lambda s: int(s.str.startswith("failed").sum())).reset_index()
    print(summary.to_string(index=False))
    if failed.empty:
        config.write_manifest(out, "00e_maps", {"fields": gpkg}, {"maps": out / "maps.csv"},
                              {"maps_folder": str(maps), "groups": only, "test_period": P["test"], "maps": len(t),
                               "longer_side_pixels": quicklook.MAX_PX})
    print(f"\n{len(t)} maps in {maps}\ntable and manifest: {out}")
    if len(failed):
        print(failed[["tif", "status"]].to_string(index=False))
        raise SystemExit(f"step 00e: {len(failed)} maps could not be made (maps.csv)")


if __name__ == "__main__":
    main()
