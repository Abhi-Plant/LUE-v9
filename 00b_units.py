"""Step 00b - units of a run that has no crop map: the parts of the area inside each ERA5-Land cell.

NOT crop fields: a unit is the part of the area polygon (run config inputs.boundary_file, set in step 00a) inside one
ERA5-Land cell (inputs.era5_cell_deg), all land cover. Cell ids as in step 02: cell centre = multiple of the cell
size, id 'lat_lon'.
The units are written to outputs/<run>/02_fields/ in the layout that step 00c reads (fields.gpkg, layers fields and
cells). A run with a crop map does not use this step: its units are the field sample of step 02, which this step
never overwrites.
Units that exist and are the same are left as they are. When the area changed, the Sentinel-2 granule cache of the
run (it holds the pixels of the old units) must be deleted first; the step stops and says so.
Reads: the polygon file of the run config. Writes: fields.gpkg (fields, cells), fields.csv, checks.csv, manifest.
Next: step 00c (00c_download.py).
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

from core import config

SOURCE = "district_boundary_x_era5land_cell"


def polygonal(g):
    """Polygon part of an intersection result (lines and points of touching edges dropped); None when empty."""
    parts = [x for x in shapely.get_parts(g) if x.geom_type in ("Polygon", "MultiPolygon") and x.area > 0]
    return shapely.union_all(parts) if parts else None


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    aoi = cfg["aoi"]
    if "boundary_file" not in (aoi.get("inputs") or {}):
        raise SystemExit(f"run {cfg['run_id']} has no inputs.boundary_file: its units are the field sample of step 02 "
                         "(02_fields.py, from the crop map) - step 00b does not apply")
    step = aoi["inputs"]["era5_cell_deg"]
    p_bd = config.input_path(cfg, "boundary_file")
    out = config.output_dir(cfg, "02_fields")
    config.check_writable(out)

    b = gpd.read_file(p_bd)
    if b.crs is None:
        raise SystemExit(f"{p_bd}: no coordinate reference system")
    if b.empty or not b.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise SystemExit(f"{p_bd}: needs polygons, has {b.geom_type.value_counts().to_dict()}")
    b = b.to_crs(4326)
    poly = shapely.make_valid(b.union_all())
    utm = b.estimate_utm_crs()
    minx, miny, maxx, maxy = poly.bounds
    rows = []
    for i in range(round(miny / step), round(maxy / step) + 1):
        for j in range(round(minx / step), round(maxx / step) + 1):
            a, o = i * step, j * step
            cell_box = box(o - step / 2, a - step / 2, o + step / 2, a + step / 2)
            part = polygonal(poly.intersection(cell_box))
            if part is not None:
                rows.append({"cell": f"{a:.1f}_{o:.1f}", "cell_lat": round(a, 6), "cell_lon": round(o, 6),
                             "geometry": part, "cell_box": cell_box})
    if not rows:
        raise SystemExit(f"{p_bd}: no cell intersects the area")
    u = gpd.GeoDataFrame(rows, crs=4326)
    u["field_id"] = [f"{aoi['name'][:3].upper()}_{c}" for c in u.cell]
    u["tehsil"] = "not_assigned"
    pt = u.geometry.representative_point()
    u["lon"], u["lat"] = pt.x, pt.y
    u["area_ha"] = u.to_crs(utm).area / 1e4
    u["cell_share"] = u["area_ha"] / (gpd.GeoSeries(u.cell_box, crs=4326).to_crs(utm).area / 1e4)
    u["source"] = SOURCE
    u = u.sort_values("field_id").reset_index(drop=True)
    total_ha = float(gpd.GeoSeries([poly], crs=4326).to_crs(utm).area.iloc[0] / 1e4)

    gpkg = out / "fields.gpkg"
    if (out / "fields.csv").exists():
        old = pd.read_csv(out / "fields.csv")
        if "source" not in old or not (old.source == SOURCE).all():
            raise SystemExit(f"{out} holds the field sample of step 02 - it is not overwritten by step 00b")
        same = (len(old) == len(u) and list(old.field_id) == list(u.field_id)
                and np.allclose(old.area_ha, u.area_ha, rtol=1e-9) and gpkg.exists())
        if same:
            print(f"units exist and are the same as the area gives now: {len(u)} units in {out} - left as they are")
            print(pd.read_csv(out / "checks.csv").to_string(index=False))
            return
        cache = config.ROOT / "data" / "stac" / "sentinel2" / cfg["run_id"] / "granules"
        if cache.exists() and any(cache.glob("*.tif")):
            raise SystemExit(f"the units in {out} differ from what the area gives now, and the stored Sentinel-2 "
                             f"granules {cache} hold the pixel windows of the old units. Delete that folder (or choose "
                             "another run id in step 00a), then run step 00b again.")

    checks = [
        {"check": "unit type", "status": "INFO",
         "detail": f"{len(u)} units = parts of the area {p_bd.name} inside {step} degree ERA5-Land cells, all land "
                   "cover - NOT crop fields (no crop map of this area is registered)"},
        {"check": "units add up to the area", "status":
            "PASS" if abs(u.area_ha.sum() - total_ha) <= 1e-6 * total_ha else "FAIL",
         "detail": f"units {u.area_ha.sum():.1f} ha, area {total_ha:.1f} ha ({utm.to_string()})"},
        {"check": "one unit per cell, unique ids", "status":
            "PASS" if u.cell.is_unique and u.field_id.is_unique else "FAIL", "detail": f"{u.cell.nunique()} cells"},
        {"check": "unit area", "status": "INFO",
         "detail": f"min {u.area_ha.min():.2f} ha, median {u.area_ha.median():.0f} ha, max {u.area_ha.max():.0f} ha; "
                   f"{int((u.cell_share > 0.999).sum())} cells fully inside the area, "
                   f"{int((u.cell_share < 0.01).sum())} units smaller than 1 % of their cell"},
    ]
    checks = pd.DataFrame(checks)

    cols = ["field_id", "tehsil", "cell", "lon", "lat", "area_ha", "cell_share", "source"]
    u[cols].to_csv(out / "fields.csv", index=False)
    if gpkg.exists():
        gpkg.unlink()
    u[cols + ["geometry"]].to_file(gpkg, layer="fields", driver="GPKG")
    cells = gpd.GeoDataFrame(u[["cell", "cell_lat", "cell_lon", "area_ha", "cell_share"]], geometry=list(u.cell_box),
                             crs=4326)
    cells.to_file(gpkg, layer="cells", driver="GPKG")
    checks.to_csv(out / "checks.csv", index=False)
    # a step with a FAIL check leaves no manifest: run.py then never treats it as complete
    if not (checks.status == "FAIL").any():
        config.write_manifest(out, "00b_units", {"boundary_file": p_bd},
                              {"fields_csv": out / "fields.csv", "fields_gpkg": gpkg, "checks": out / "checks.csv"},
                              {"unit_source": SOURCE, "era5_cell_deg": step, "area_crs": utm.to_string()})
    pd.set_option("display.width", 250, "display.max_colwidth", 200)
    print(checks.to_string(index=False))
    print(f"\n{len(u)} units -> {out}")
    if (checks.status == "FAIL").any():
        raise SystemExit("step 00b stopped: FAIL checks above")


if __name__ == "__main__":
    main()
