"""Data access for the field-unit exports E1-E3 through STAC APIs (step 00c). No Google Earth Engine.

E1  Sentinel-2 L2A  -> NDVI and clear share per field and acquisition date        (STAC: stac.sentinel2)
E2  Landsat 8/9 C2 L2 -> clear share per field and overpass; EF and ETa per field when an EF function is
    configured (stac.landsat.ef_function) - the energy balance itself is not part of this framework
E3  ERA5-Land hourly -> daily weather per 0.1 degree cell (STAC catalogue of the Copernicus Climate Data Store,
    retrieval with cdsapi); terrain elevation of the cell centre from a STAC DEM collection

Storage (v0.6.0): every satellite and weather dataset is stored as GeoTIFF first, and every export value is computed
from the stored files.
  data/stac/sentinel2/<run>/granules/  B04, B08, SCL of every granule: the digital numbers of the source on its own
                                       grid inside the pixel windows of the units (stac.sentinel2.coverage
                                       all_parcels: of every parcel of the crop map), no data elsewhere
  data/stac/landsat/<run>/scenes/      QA_PIXEL of every scene, the bands of stac.landsat.download_assets and every
                                       band the EF function reads: digital numbers in the scene window (all units +
                                       stac.landsat.context_km); <item id>.json = record and result of the scene.
                                       stac.landsat.keep_source_bands false: the bands are deleted once the scene's
                                       results (field rows, clear mask, EF and ETa rasters) are stored
  data/stac/era5land/<run>/raw/, raw_precip/   hourly values, one GeoTIFF per month and variable (one band per hour).
                                       The Climate Data Store delivers netCDF or GRIB only: the netCDF file is
                                       converted at once and compared value by value
  data/stac/dem/<run>/                 elevation of the weather cells, one GeoTIFF on the grid of the source tiles
Preprocessing results and raster results (<folder of the export files>/<run>_rasters/):
  ndvi/      per Sentinel-2 date on the 10 m grid of the run: clear mask, red and NIR surface reflectance, NDVI
  landsat/   per Landsat scene: clear mask; EF, ETa (and the fluxes of stac.landsat.result_rasters) of the EF function
  weather/   daily weather per variable (0.1 degree, one band per day), Tmean, FAO-56 ETo, elevation of the cells
Every stored granule, scene, month and raster has a record (.json) with the sha256 of its files.

Only data access and the spatial / daily aggregation of the exports are defined here. No model equation is defined
or changed: NDVI = (B8 - B4) / (B8 + B4) and the column definitions are those of the v3 export specification
(README "Field units"). Field means are area-weighted means over the pixels of the native UTM grid (weight = share
of the pixel inside the field polygon).
"""
import contextlib
import importlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from core import config

GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF", "GDAL_HTTP_MAX_RETRY": "5",
            "GDAL_HTTP_RETRY_DELAY": "2", "VSI_CACHE": "TRUE"}
BLOCK = 512                    # pixel blocks of the stored GeoTIFFs
STRIP = 2 * BLOCK              # rows copied from a source at a time (memory does not depend on the window)
WEATHER_UNITS = {"rs_mj_m2": "MJ m-2 d-1", "tmin_c": "degC", "tmax_c": "degC", "tdew_c": "degC", "u10_ms": "m s-1",
                 "sp_kpa": "kPa", "precip_mm": "mm d-1", "tmean_c": "degC", "eto_mm": "mm d-1", "elev_m": "m"}


class DownloadError(SystemExit):
    """A granule, scene or weather request that could not be retrieved after the configured tries. Running the
    step again continues: everything already retrieved is kept."""


# ---------------------------------------------------------------- period, folders, fields, STAC search
def run_paths(cfg, start=None, end=None):
    """Period and folders of the download (written by step 00c, read by step 00d). A test period (start or end
    given) has its own folders under data/_test/ and never touches the export files of the run config.
    rasters: folder of the preprocessing and raster results, next to the export files; pixels: the pixel stacks of
    export px (core/pixels.py)."""
    test = bool(start or end)
    s, e = export_period(cfg, start, end)
    run, fx = cfg["run_id"], cfg["aoi"]["field_exports"]
    if test:
        tag = f"_TEST_{s:%Y%m%d}_{e:%Y%m%d}"
        base = config.ROOT / "data" / "_test" / f"{run}{tag}"
        exports = {k: base / "field_exports" / config.path(v).name for k, v in fx.items()}
    else:
        tag, base = "", config.ROOT / "data"
        exports = {k: config.path(v) for k, v in fx.items()}
    folder = {k: base / "stac" / k / run for k in ("sentinel2", "landsat", "era5land", "dem")}
    rasters = exports["e1_s2_perdate"].parent / f"{run}_rasters"
    pixels = exports["e1_s2_perdate"].parent / f"{run}_pixels"    # pixel stacks (export px, v0.8.0)
    return {"start": s, "end": e, "test": test, "tag": tag, "exports": exports, "folder": folder, "rasters": rasters,
            "pixels": pixels}


def export_period(cfg, start=None, end=None):
    """First and last day of the exports: season window start - lead_days to the window end."""
    sc = cfg["season"]
    s = pd.Timestamp(sc["window"][0]) - pd.Timedelta(days=sc["lead_days"])
    e = pd.Timestamp(sc["window"][1])
    return (pd.Timestamp(start) if start else s), (pd.Timestamp(end) if end else e)


def load_fields(path):
    """Sample fields of step 02 (fields.gpkg, layer fields), EPSG:4326, valid polygons, sorted by field_id."""
    import geopandas as gpd
    import shapely
    g = gpd.read_file(path, layer="fields").to_crs(4326)
    miss = [c for c in ("field_id", "tehsil", "cell") if c not in g]
    if miss:
        raise SystemExit(f"{path}: layer fields lacks {miss}")
    if g.field_id.duplicated().any():
        raise SystemExit(f"{path}: duplicate field_id")
    g["geometry"] = shapely.make_valid(g.geometry.values)
    return g.sort_values("field_id").reset_index(drop=True)


def load_parcels(cfg):
    """Every parcel of the crop map of the run (field_id as in the step-02 sample, cell, tehsil, area), EPSG:4326,
    valid polygons, sorted by field_id - the units of stac.sentinel2.coverage all_parcels and of the pixel stacks
    (export px; steps 05b, 11) and of the parcel-mean map (step 12)."""
    import geopandas as gpd
    import shapely
    from core import sampling
    cm, frame, _ = sampling.parcel_frame(cfg)
    g = gpd.GeoDataFrame(frame.drop(columns=["x_m", "y_m"]), geometry=cm.geometry.values, crs=cm.crs).to_crs(4326)
    g["geometry"] = shapely.make_valid(g.geometry.values)
    return g.sort_values("field_id").reset_index(drop=True)


def retry(fn, tries, what):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - network and GDAL errors of any kind are retried, then raised
            if k == tries - 1:
                raise
            print(f"  retry {k + 1}/{tries - 1} {what}: {type(e).__name__}: {str(e)[:150]}", flush=True)
            time.sleep(5 * (k + 1))


def search(api, collection, bbox, start, end, query=None, tries=4):
    """All items of the collection that intersect the bounding box from start to end (both days included, UTC)."""
    import pystac_client

    def run():
        cat = pystac_client.Client.open(api)
        s = cat.search(collections=[collection], bbox=bbox, query=query,
                       datetime=f"{start:%Y-%m-%d}T00:00:00Z/{end:%Y-%m-%d}T23:59:59Z")
        return list(s.items())
    return sorted(retry(run, tries, f"search {collection}"), key=lambda i: (i.datetime, i.id))


def signer(mode):
    """URL signing of the asset links (Planetary Computer needs a short-lived token per link)."""
    if mode == "planetary_computer":
        import planetary_computer
        return planetary_computer.sign
    if mode in (None, "none"):
        return lambda href: href
    raise SystemExit(f"stac sign: {mode} not known (planetary_computer | none)")


def fields_in_item(item, fields, buffer_deg=0.02):
    """field_id of the fields that intersect the item footprint (buffered ~2 km: footprints are approximate)."""
    from shapely.geometry import shape
    return fields.field_id[fields.intersects(shape(item.geometry).buffer(buffer_deg)).to_numpy()].tolist()


# ---------------------------------------------------------------- stored GeoTIFFs and their records
def file_record(p):
    return {"file": p.name, "bytes": p.stat().st_size, "sha256": config.sha256(p)}


def read_json(p):
    """Record of a stored granule, scene, month or raster; None when there is none or it cannot be read."""
    try:
        return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None
    except (OSError, ValueError):
        return None


def write_json(p, rec):
    config.write_atomic(p, lambda tmp: tmp.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8"))


def tif_profile(width, height, count, dtype, crs, transform, nodata):
    """GeoTIFF, DEFLATE (lossless), 512 pixel blocks when the raster is that large. Blocks that are never written
    take no space and read as the no-data value."""
    p = {"driver": "GTiff", "width": int(width), "height": int(height), "count": int(count), "dtype": dtype,
         "crs": crs, "transform": transform, "nodata": nodata, "compress": "deflate", "interleave": "band",
         "predictor": 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2, "BIGTIFF": "IF_SAFER",
         "sparse_ok": True}
    if width >= BLOCK and height >= BLOCK:
        p.update(tiled=True, blockxsize=BLOCK, blockysize=BLOCK)
    return p


def write_tif(path, a, transform, crs, nodata, descriptions=None, tags=None):
    """Array [rows, cols] or [bands, rows, cols] -> GeoTIFF; descriptions: one text per band. Returns its record."""
    import rasterio
    a = a[None] if a.ndim == 2 else a

    def write(tmp):
        with rasterio.open(tmp, "w", **tif_profile(a.shape[2], a.shape[1], a.shape[0], a.dtype.name, crs, transform,
                                                   nodata)) as ds:
            ds.write(a)
            for i, text in enumerate(descriptions or [], 1):
                ds.set_band_description(i, text)
            ds.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
    config.write_atomic(path, write)
    return file_record(path)


def store_clip(src, path, win, parts=None, tags=None, fill=0, scale=None, offset=None):
    """Band 1 of the open raster src in the pixel window win = (col0, row0, width, height) -> GeoTIFF on the grid of
    the source, with its data type and no-data value (fill when the source names none): the values of the source,
    unchanged. parts: pixel windows of src - only these are read and stored, the other pixels are no data;
    None = the whole window. scale, offset: written as the scale and offset of the band (value x scale + offset =
    physical quantity); the stored values stay the digital numbers. Returns the record of the file."""
    import rasterio
    from rasterio.windows import Window
    c0, r0, w, h = win
    nodata = src.nodata if src.nodata is not None else fill
    dtype = src.dtypes[0]

    def write(tmp):
        with rasterio.open(tmp, "w", **tif_profile(w, h, 1, dtype, src.crs, src.window_transform(Window(c0, r0, w, h)),
                                                   nodata)) as dst:
            dst.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
            if scale is not None:
                dst.scales, dst.offsets = (float(scale),), (float(offset or 0.0),)
            for s0 in range(r0, r0 + h, STRIP):
                s1 = min(s0 + STRIP, r0 + h)
                if parts is None:
                    a = src.read(1, window=Window(c0, s0, w, s1 - s0))
                else:
                    a = None
                    for pc, pr, pw, ph in parts:
                        t0, t1 = max(pr, s0), min(pr + ph, s1)
                        if t1 > t0:
                            if a is None:
                                a = np.full((s1 - s0, w), nodata, dtype)
                            a[t0 - s0:t1 - s0, pc - c0:pc - c0 + pw] = src.read(1, window=Window(pc, t0, pw, t1 - t0))
                    if a is None:
                        continue                                  # no unit in these rows: the blocks stay empty
                dst.write(a, 1, window=Window(0, s0 - r0, w, s1 - s0))
    config.write_atomic(path, write)
    return file_record(path)


def write_index(path, folder, pattern="*.json"):
    """Table of the stored GeoTIFFs of a folder (file relative to the table, bytes, sha256), from their records."""
    rows = []
    for p in sorted(folder.glob(pattern)):
        for f in ((read_json(p) or {}).get("files") or {}).values():
            if (p.parent / f["file"]).exists():
                rows.append({"file": (p.parent / f["file"]).relative_to(path.parent).as_posix(), "bytes": f["bytes"],
                             "sha256": f["sha256"]})
    t = pd.DataFrame(rows, columns=["file", "bytes", "sha256"]).sort_values("file").reset_index(drop=True)
    config.to_csv_atomic(t, path, index=False)
    return t


def raster_index(folder):
    """raster_files.csv of the preprocessing and raster results: every GeoTIFF under the folder with bytes and
    sha256 (taken from the record written with the raster, computed for the rasters without one)."""
    known = {}
    for p in folder.rglob("*.json"):
        for f in ((read_json(p) or {}).get("files") or {}).values():
            known[p.parent / f["file"]] = f
    rows = []
    for p in sorted(folder.rglob("*.tif")):
        f = known.get(p)
        if f is None or f["bytes"] != p.stat().st_size:
            f = file_record(p)
        rows.append({"file": p.relative_to(folder).as_posix(), "bytes": f["bytes"], "sha256": f["sha256"]})
    t = pd.DataFrame(rows, columns=["file", "bytes", "sha256"])
    config.to_csv_atomic(t, folder / "raster_files.csv", index=False)
    return t


# ---------------------------------------------------------------- pixel weights of the fields on a raster grid
def read_window(ds, c0, r0, w, h, fill=0):
    """Band 1 in the pixel window (c0, r0, w, h); pixels outside the raster get the fill value."""
    from rasterio.windows import Window
    out = np.full((h, w), fill, ds.dtypes[0])
    a0, b0, a1, b1 = max(c0, 0), max(r0, 0), min(c0 + w, ds.width), min(r0 + h, ds.height)
    if a1 > a0 and b1 > b0:
        out[b0 - r0:b1 - r0, a0 - c0:a1 - c0] = ds.read(1, window=Window(a0, b0, a1 - a0, b1 - b0))
    return out


class FieldGrid:
    """Share of each pixel covered by each field polygon, on north-up grids x = px + col * res, y = py - row * res.
    A grid is identified by (epsg, res, px, py); rasters of the same grid share the weights."""

    def __init__(self, fields):
        self.fields, self.lock, self.proj, self.cache = fields, threading.RLock(), {}, {}

    @staticmethod
    def key(ds):
        t = ds.transform
        if abs(t.b) > 1e-9 or abs(t.d) > 1e-9 or abs(t.a + t.e) > 1e-9:
            raise SystemExit(f"{ds.name}: raster is not north-up with square pixels")
        if ds.crs is None or ds.crs.to_epsg() is None:
            raise SystemExit(f"{ds.name}: no EPSG code")
        return ds.crs.to_epsg(), round(t.a, 6), round(t.c % t.a, 6), round(t.f % t.a, 6)

    @staticmethod
    def origin(ds, key):
        """Global column and row index of the raster's first pixel."""
        _, res, px, py = key
        return round((ds.transform.c - px) / res), round((py - ds.transform.f) / res)

    def geoms(self, epsg):
        with self.lock:
            if epsg not in self.proj:
                self.proj[epsg] = self.fields.to_crs(epsg).set_index("field_id").geometry
            return self.proj[epsg]

    def window(self, key, field_id):
        """Global pixel window (col0, row0, width, height) that holds one field."""
        epsg, res, px, py = key
        minx, miny, maxx, maxy = self.geoms(epsg)[field_id].bounds
        c0, c1 = math.floor((minx - px) / res), math.ceil((maxx - px) / res)
        r0, r1 = math.floor((py - maxy) / res), math.ceil((py - miny) / res)
        return c0, r0, c1 - c0, r1 - r0

    def extent(self, key):
        """Global pixel window that holds all fields."""
        b = np.array([self.window(key, f) for f in self.fields.field_id])
        c0, r0 = int(b[:, 0].min()), int(b[:, 1].min())
        return c0, r0, int((b[:, 0] + b[:, 2]).max()) - c0, int((b[:, 1] + b[:, 3]).max()) - r0

    def scene_window(self, key, origin, width, height, margin_m=0.0):
        """Pixel window of a raster (first pixel = global index origin) that holds all fields plus a margin,
        inside the raster; None when the fields are outside it."""
        _, res, px, py = key
        b = self.geoms(key[0]).total_bounds
        x0, y0 = px + origin[0] * res, py - origin[1] * res        # upper left corner of the raster
        c0 = max(math.floor((b[0] - margin_m - x0) / res), 0)
        c1 = min(math.ceil((b[2] + margin_m - x0) / res), width)
        r0 = max(math.floor((y0 - b[3] - margin_m) / res), 0)
        r1 = min(math.ceil((y0 - b[1] + margin_m) / res), height)
        return [c0, r0, c1 - c0, r1 - r0] if c1 > c0 and r1 > r0 else None

    def weights(self, key, field_id):
        """(global col0, global row0, weights[rows, cols]) of one field."""
        import shapely
        with self.lock:
            hit = self.cache.get((key, field_id))
            if hit is not None:
                return hit
            epsg, res, px, py = key
            geom = self.geoms(epsg)[field_id]
            c0, r0, wd, h = self.window(key, field_id)
            x0, y1 = np.meshgrid(px + np.arange(c0, c0 + wd) * res, py - np.arange(r0, r0 + h) * res)
            boxes = shapely.box(x0, y1 - res, x0 + res, y1)
            shapely.prepare(geom)
            cut = ~shapely.contains_properly(geom, boxes)         # pixels entirely inside the field: share 1
            w = np.ones(boxes.shape)
            w[cut] = shapely.area(shapely.intersection(boxes[cut], geom)) / (res * res)
            self.cache[(key, field_id)] = (c0, r0, w)
            return c0, r0, w


def unit_windows(grid, key, origin, width, height, cand):
    """{field_id: [col0, row0, width, height]}: pixel window of every candidate field on a raster of the grid `key`
    whose first pixel has the global index `origin`, cut to the raster; fields outside the raster are left out."""
    out = {}
    for fid in cand:
        gc, gr, wd, h = grid.window(key, fid)
        a0, b0 = max(gc - origin[0], 0), max(gr - origin[1], 0)
        a1, b1 = min(gc - origin[0] + wd, width), min(gr - origin[1] + h, height)
        if a1 > a0 and b1 > b0:
            out[fid] = [a0, b0, a1 - a0, b1 - b0]
    return out


def coarse(win, k):
    """Window of a band with k times larger pixels and the same origin that holds the pixel window win."""
    c0, r0, wd, h = win
    return [c0 // k, r0 // k, -(-(c0 + wd) // k) - c0 // k, -(-(r0 + h) // k) - r0 // k]


def union(wins, k=1):
    """Smallest window that holds all windows, with corners on multiples of k."""
    c0, r0 = min(w[0] for w in wins) // k * k, min(w[1] for w in wins) // k * k
    c1, r1 = -(-max(w[0] + w[2] for w in wins) // k) * k, -(-max(w[1] + w[3] for w in wins) // k) * k
    return [c0, r0, c1 - c0, r1 - r0]


def strip_plan(wins, rows=BLOCK):
    """{first row of a strip: [[col0, col1], ...]}: for every strip of `rows` rows that holds a window, the column
    ranges of the windows in it, merged where they touch (so no pixel is in two ranges)."""
    plan = {}
    for c0, r0, wd, h in wins:
        for s0 in range(r0 // rows * rows, r0 + h, rows):
            plan.setdefault(s0, []).append([c0, c0 + wd])
    for s0, iv in plan.items():
        iv.sort()
        merged = [iv[0]]
        for a, b in iv[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        plan[s0] = merged
    return dict(sorted(plan.items()))


def run_items(items, work, workers, tries, label, name=lambda i: i.id):
    """work(item) for every item in a thread pool; returns {item name: error text} of the items that failed."""
    failed, t0, done = {}, time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        fut = {ex.submit(retry, (lambda i=i: work(i)), tries, name(i)): i for i in items}
        for f in as_completed(fut):
            done += 1
            try:
                f.result()
            except BaseException as e:  # noqa: BLE001 - reported per item, the run stops afterwards
                failed[name(fut[f])] = f"{type(e).__name__}: {str(e)[:300]}"
            if done % 20 == 0 or done == len(items):
                el = time.time() - t0
                print(f"  {label}: {done}/{len(items)} items, {el / 60:.1f} min, "
                      f"about {el / done * (len(items) - done) / 60:.0f} min left", flush=True)
    return failed


# ---------------------------------------------------------------- E1 Sentinel-2
def _local(tag):
    return tag.rsplit("}", 1)[-1]


def parse_s2_offsets(xml_bytes, bands):
    """BOA_ADD_OFFSET of each band in `bands` (asset names such as B04, B08) and BOA_QUANTIFICATION_VALUE from the
    Level-2A product metadata (MTD_MSIL2A.xml). Reflectance = (DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE (ESA,
    Sentinel-2 products specification, from processing baseline 04.00, 25 Jan 2022). The band of an offset is found
    through the bandId -> physicalBand list of the same file. Products without an offset list (earlier baselines):
    offset 0. Returns ({asset name: offset}, quantification value)."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_bytes)
    band_id, offs, quant = {}, {}, None
    for el in root.iter():
        tag = _local(el.tag)
        if tag == "Spectral_Information" and el.get("physicalBand") is not None:
            band_id[el.get("physicalBand")] = el.get("bandId")
        elif tag == "BOA_ADD_OFFSET":
            offs[el.get("band_id")] = float(el.text)
        elif tag == "BOA_QUANTIFICATION_VALUE":
            quant = float(el.text)
    if quant is None:
        raise SystemExit("Sentinel-2 product metadata: no BOA_QUANTIFICATION_VALUE")
    out = {}
    for b in bands:                                               # asset B04 -> physical band B4, B8A stays B8A
        phys = "B" + (str(int(b[1:])) if b[1:].isdigit() else b[1:])
        if not offs:
            out[b] = 0.0
            continue
        if phys not in band_id or band_id[phys] not in offs:
            raise SystemExit(f"Sentinel-2 product metadata: no BOA_ADD_OFFSET for band {b} ({phys})")
        out[b] = offs[band_id[phys]]
    return out, quant


def s2_offsets(item, sign, c2, tries=4):
    """Offsets of the red and NIR assets and the quantification value of one Sentinel-2 item, read from its product
    metadata asset (config stac.sentinel2.metadata_asset)."""
    import requests
    key = c2["metadata_asset"]
    if key not in item.assets:
        raise SystemExit(f"{item.id}: no asset '{key}' (stac.sentinel2.metadata_asset); assets: {sorted(item.assets)}")

    def get():
        r = requests.get(sign(item.assets[key].href), timeout=120)
        r.raise_for_status()
        return r.content
    a = c2["assets"]
    return parse_s2_offsets(retry(get, tries, f"metadata {item.id}"), [a["red"], a["nir"]])


def s2_item(item, cand, grid, sign, c2, folder):
    """Red, NIR and scene classification of one granule inside the pixel windows of its candidate fields ->
    <folder>/<item id>_<asset>.tif (digital numbers of the source, its grid) and the record <item id>.json, which
    also holds the reflectance offsets and the quantification value of the product metadata."""
    import rasterio
    offs, quant = s2_offsets(item, sign, c2)
    a = c2["assets"]
    add = {"red": offs[a["red"]], "nir": offs[a["nir"]]}
    rec = {"item_id": item.id, "datetime_utc": item.datetime.isoformat(),
           "processing_baseline": str(item.properties.get("s2:processing_baseline")), "add_offset": add,
           "quantification": quant}
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(sign(item.assets[a["scl"]].href)) as scl, \
                rasterio.open(sign(item.assets[a["red"]].href)) as red, \
                rasterio.open(sign(item.assets[a["nir"]].href)) as nir:
            key = grid.key(red)
            if grid.key(nir) != key or red.transform != nir.transform or red.shape != nir.shape:
                raise SystemExit(f"{item.id}: red and NIR grids differ")
            k = round(scl.transform.a / red.transform.a)
            if (abs(scl.transform.c - red.transform.c) > 1e-6 or abs(scl.transform.f - red.transform.f) > 1e-6
                    or abs(scl.transform.a - k * red.transform.a) > 1e-6 or scl.width * k != red.width
                    or scl.height * k != red.height):
                raise SystemExit(f"{item.id}: SCL grid is not a multiple of the 10 m grid with the same origin")
            origin = grid.origin(red, key)
            wins = unit_windows(grid, key, origin, red.width, red.height, cand)
            rec.update(grid=list(key), origin=list(origin), size=[red.width, red.height], scl_factor=k, units=wins,
                       window=None, files={})
            if wins:
                u = union(list(wins.values()), k)                 # SCL (20 m) pixels are whole inside the window
                tags = {"item_id": item.id, "collection": c2["collection"], "datetime_utc": rec["datetime_utc"],
                        "processing_baseline": rec["processing_baseline"],
                        "content": "digital numbers of the source inside the pixel windows of the units"}
                for name, ds, kk in (("scl", scl, k), ("red", red, 1), ("nir", nir, 1)):
                    refl = name in add                            # reflectance = (DN + offset) / quantification value
                    f = store_clip(ds, folder / f"{item.id}_{a[name]}.tif", [v // kk for v in u],
                                   [coarse(w, kk) for w in wins.values()],
                                   {**tags, "asset": a[name], **({"boa_add_offset": add[name],
                                                                  "boa_quantification_value": quant} if refl else {})},
                                   scale=1.0 / quant if refl else None, offset=add[name] / quant if refl else None)
                    rec["files"][name] = {**f, "href": item.assets[a[name]].href}
                rec["window"] = u
    write_json(folder / f"{item.id}.json", rec)
    return rec


def s2_stored(folder, item_id, cand, grid):
    """Record of a granule whose stored GeoTIFFs hold the pixel windows the units need now; None when the granule
    has to be downloaded."""
    rec = read_json(folder / f"{item_id}.json")
    if rec is None or "units" not in rec or "add_offset" not in rec:
        return None
    key = (int(rec["grid"][0]), *rec["grid"][1:])
    if unit_windows(grid, key, rec["origin"], *rec["size"], cand) != rec["units"]:
        return None
    return rec if all((folder / f["file"]).exists() for f in rec["files"].values()) else None


def s2_pixels(scl, red, nir, win, k, c2, add, quant):
    """The preprocessing of one granule in the 10 m pixel window win = (col0, row0, width, height):
    (NDVI float32 with NaN, valid, seen, (red, NIR, used) or None).
    1 scene classification (k times larger pixels, same origin) to 10 m by nearest neighbour;
    2 valid pixel = scene classification in c2['valid_scl']; seen = classification other than 0 (no data);
    3 surface reflectance = (DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE (add: offsets of red and NIR; ESA L2A);
    4 used = valid pixel with red and NIR present (DN > 0) and positive reflectance; NDVI = (NIR - red) / (NIR + red)."""
    c0, r0, wd, h = win
    sc0, sr0, sw, sh = coarse(win, k)
    s = np.repeat(np.repeat(read_window(scl, sc0, sr0, sw, sh), k, 0), k, 1)
    s = s[r0 - sr0 * k:r0 - sr0 * k + h, c0 - sc0 * k:c0 - sc0 * k + wd]
    valid = np.isin(s, c2["valid_scl"])
    nd, bands = np.full((h, wd), np.nan, np.float32), None
    if valid.any():
        r = read_window(red, c0, r0, wd, h).astype(np.int32)
        n = read_window(nir, c0, r0, wd, h).astype(np.int32)
        ok = valid & (r > 0) & (n > 0)                            # DN 0 = no data
        r, n = (r + add["red"]) / quant, (n + add["nir"]) / quant  # surface reflectance (ESA L2A definition)
        ok &= (r > 0) & (n > 0)                                    # reflectance after the offset must be physical
        nd[ok] = (n[ok] - r[ok]) / (r[ok] + n[ok])
        bands = (r, n, ok)
    return nd, valid, s != 0, bands


S2_RASTERS = {   # name: (data type, no data, what the raster holds)
    "clear": ("uint8", 255, "1 = valid pixel (scene classification in valid_scl) in a granule of the date, "
                            "0 = not valid, 255 = not observed"),
    "red": ("uint16", 65535, "red (B04) surface reflectance x 10000 = (DN + BOA_ADD_OFFSET) / "
                             "BOA_QUANTIFICATION_VALUE x 10000 on the pixels used for NDVI; mean of the overlapping "
                             "granules of the date, rounded to 1"),
    "nir": ("uint16", 65535, "NIR (B08) surface reflectance x 10000 = (DN + BOA_ADD_OFFSET) / "
                             "BOA_QUANTIFICATION_VALUE x 10000 on the pixels used for NDVI; mean of the overlapping "
                             "granules of the date, rounded to 1"),
    "ndvi": ("float32", np.nan, "NDVI = (B8 - B4) / (B8 + B4) per granule on the pixels used; mean of the "
                                "overlapping granules of the date"),
}


def s2_date(run, date, recs, key, extent, plan, c2, store, out):
    """The preprocessing results of one acquisition date on the 10 m grid of the run, from the stored granules,
    as GeoTIFF <out>/<run>_<name>_<date>.tif (S2_RASTERS): clear mask, red and NIR reflectance (when
    c2['reflectance_rasters']) and NDVI, with the record <run>_ndvi_<date>.json. Overlapping granules of the
    date: mean per pixel, clear in any. Made again only when the granules of the date or the settings changed."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    names = ["clear", *(["red", "nir"] if c2.get("reflectance_rasters", True) else []), "ndvi"]
    path = {n: out / f"{run}_{n}_{date}.tif" for n in names}
    p_js = out / f"{run}_ndvi_{date}.json"
    made_from = {"granules": {r["item_id"]: [f["sha256"] for f in r["files"].values()] for r in recs},
                 "grid": list(key), "extent": list(extent), "valid_scl": list(c2["valid_scl"]), "rasters": names}
    old = read_json(p_js)
    if old and old.get("made_from") == made_from and all(p.exists() for p in path.values()):
        return old
    (epsg, res, px, py), (gc, gr, width, height) = key, extent
    transform = from_origin(px + gc * res, py - gr * res, res, res)
    tags = {"date_utc": date, "granules": ";".join(r["item_id"] for r in recs), "valid_scl": c2["valid_scl"],
            "grid": f"EPSG:{epsg}, {res:g} m"}
    tmp = {n: p.with_name(p.name + ".part") for n, p in path.items()}
    with contextlib.ExitStack() as st:
        src = []
        for r in recs:
            if r["files"]:
                ds = {n: st.enter_context(rasterio.open(store / r["files"][n]["file"])) for n in ("scl", "red", "nir")}
                src.append((r, ds, r["origin"][0] + r["window"][0] - gc, r["origin"][1] + r["window"][1] - gr))
        dst = {}
        for n in names:
            dtype, nodata, content = S2_RASTERS[n]
            dst[n] = st.enter_context(rasterio.open(tmp[n], "w", **tif_profile(width, height, 1, dtype, f"EPSG:{epsg}",
                                                                              transform, nodata)))
            dst[n].update_tags(**{k: str(v) for k, v in tags.items()}, content=content)
            if n in ("red", "nir"):
                dst[n].scales = (1e-4,)                           # value x 0.0001 = reflectance
        for s0, ranges in plan.items():
            sh = min(BLOCK, height - s0)
            strip = {n: np.full((sh, width), S2_RASTERS[n][1], S2_RASTERS[n][0]) for n in names}
            any_seen = False
            for x0, x1 in ranges:
                shape = (sh, x1 - x0)
                tot, cnt = np.zeros(shape, np.float32), np.zeros(shape, np.intp)
                val, seen = np.zeros(shape, bool), np.zeros(shape, bool)
                refl = {n: np.zeros(shape, np.float64) for n in ("red", "nir") if n in names}
                for r, ds, gx, gy in src:
                    a0, a1 = max(x0, gx), min(x1, gx + r["window"][2])
                    b0, b1 = max(s0, gy), min(s0 + sh, gy + r["window"][3])
                    if a1 <= a0 or b1 <= b0:
                        continue
                    nd, valid, obs, bands = s2_pixels(ds["scl"], ds["red"], ds["nir"],
                                                      (a0 - gx, b0 - gy, a1 - a0, b1 - b0), r["scl_factor"], c2,
                                                      r["add_offset"], r["quantification"])
                    sl = (slice(b0 - s0, b1 - s0), slice(a0 - x0, a1 - x0))
                    ok = ~np.isnan(nd)
                    tot[sl][ok] += nd[ok]                         # granules in the order of the item table
                    cnt[sl] += ok
                    val[sl] |= valid
                    seen[sl] |= obs
                    if bands is not None:
                        for n, x in zip(("red", "nir"), bands[:2]):
                            if n in refl:
                                refl[n][sl][ok] += x[ok]
                if not seen.any():
                    continue
                any_seen = True
                np.divide(tot, cnt, out=tot, where=cnt > 0, casting="unsafe")     # as numpy.nanmean
                tot[cnt == 0] = np.nan
                strip["ndvi"][:, x0:x1] = tot
                strip["clear"][:, x0:x1] = np.where(seen, val.astype(np.uint8), 255)
                for n, x in refl.items():
                    mean = np.clip(np.rint(x / np.maximum(cnt, 1) * 1e4), 0, 65534)
                    strip[n][:, x0:x1] = np.where(cnt > 0, mean, 65535).astype(np.uint16)
            if any_seen:
                for n in names:
                    dst[n].write(strip[n], 1, window=Window(0, s0, width, sh))
    for n in names:
        tmp[n].replace(path[n])
    rec = {"date": date, "made_from": made_from, "files": {n: file_record(p) for n, p in path.items()}}
    write_json(p_js, rec)
    return rec


def array_window(a, c0, r0, w, h, fill):
    """Pixel window (c0, r0, w, h) of an array; pixels outside it get the fill value (as read_window)."""
    out = np.full((h, w), fill, a.dtype)
    a0, b0, a1, b1 = max(c0, 0), max(r0, 0), min(c0 + w, a.shape[1]), min(r0 + h, a.shape[0])
    if a1 > a0 and b1 > b0:
        out[b0 - r0:b1 - r0, a0 - c0:a1 - c0] = a[b0:b1, a0:a1]
    return out


def s2_rows(run, date, n_scenes, fields, grid, key, extent, out):
    """E1 rows of one date from its NDVI and clear rasters: area-weighted mean NDVI over the pixels with a value,
    valid_fraction = clear share of the field's pixels (steps 00c and 11)."""
    import rasterio
    with rasterio.open(out / f"{run}_ndvi_{date}.tif") as d_nd, rasterio.open(out / f"{run}_clear_{date}.tif") as d_cl:
        nd_a, cl_a = d_nd.read(1), d_cl.read(1)
    rows = []
    for fid in fields.field_id:
        c0, r0, w = grid.weights(key, fid)
        nd = array_window(nd_a, c0 - extent[0], r0 - extent[1], w.shape[1], w.shape[0], np.nan)
        valid = array_window(cl_a, c0 - extent[0], r0 - extent[1], w.shape[1], w.shape[0], 255) == 1
        ok = ~np.isnan(nd)
        rows.append((fid, date, (w[ok] * nd[ok]).sum() / w[ok].sum() if w[ok].sum() > 0 else np.nan,
                     (w * valid).sum() / w.sum(), n_scenes))
    return rows


def s2_export(cfg, fields, start, end, folder, rasters, units=None):
    """E1: one row per field and Sentinel-2 acquisition date (UTC). The bands of every granule are stored as
    GeoTIFF (<folder>/granules), the preprocessing results of every date (clear mask, reflectance, NDVI) are made
    from them (<rasters>/ndvi) and the table is computed from those rasters. units: the polygons whose pixels are
    stored and preprocessed (all parcels of the crop map; default the fields) - the table holds the fields only.
    Returns (table, item table)."""
    c2, st, run = cfg["stac"]["sentinel2"], cfg["stac"], cfg["run_id"]
    units = fields if units is None else units
    if set(fields.field_id) - set(units.field_id):
        raise SystemExit("E1: fields of the sample are not parcels of the crop map (field_id) - run step 02 again")
    store, out = folder / "granules", rasters / "ndvi"
    store.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    bbox = [float(x) for x in units.total_bounds]
    items = search(c2["api"], c2["collection"], bbox, start, end, tries=st["retries"])
    cand = {i.id: fields_in_item(i, units) for i in items}
    items = [i for i in items if cand[i.id]]                      # granules that touch a unit
    if not items:
        raise SystemExit(f"no {c2['collection']} items for the fields from {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    grid, sign = FieldGrid(units), signer(c2.get("sign"))
    recs = {i.id: s2_stored(store, i.id, cand[i.id], grid) for i in items}
    todo = [i for i in items if recs[i.id] is None]
    print(f"E1 Sentinel-2: {len(items)} granules on {len({i.datetime.date() for i in items})} dates; "
          f"{len(items) - len(todo)} already stored as GeoTIFF, {len(todo)} to download", flush=True)

    def work(i):
        recs[i.id] = s2_item(i, cand[i.id], grid, sign, c2, store)
    failed = run_items(todo, work, st["max_workers"], st["retries"], "E1")
    rec_of = lambda i, f, default=np.nan: f(recs[i.id]) if recs[i.id] else default  # noqa: E731
    meta = pd.DataFrame([{"item_id": i.id, "date": i.datetime.date().isoformat(), "datetime_utc": i.datetime.isoformat(),
                          "platform": i.properties.get("platform"), "mgrs_tile": i.properties.get("s2:mgrs_tile"),
                          "processing_baseline": i.properties.get("s2:processing_baseline"),
                          "cloud_cover_pct": i.properties.get("eo:cloud_cover"), "candidate_fields": len(cand[i.id]),
                          "status": "failed: " + failed[i.id] if i.id in failed else "read",
                          # red;NIR offsets of the product metadata used for the granule
                          "boa_add_offset": rec_of(i, lambda r: ";".join(f"{r['add_offset'][n]:g}" for n in ("red", "nir"))),
                          "stored_mb": rec_of(i, lambda r: round(sum(f["bytes"] for f in r["files"].values()) / 1e6, 3))}
                         for i in items])
    config.to_csv_atomic(meta, folder / "s2_items.csv", index=False)
    write_index(folder / "s2_files.csv", store)
    if failed:
        raise DownloadError(f"E1: {len(failed)} granules could not be read (see {folder / 's2_items.csv'}); "
                            "run the step again - granules already stored are kept")
    keys = {(int(r["grid"][0]), *[round(float(v), 6) for v in r["grid"][1:]]) for r in recs.values()}
    if len(keys) > 1:
        raise SystemExit(f"E1: granules on more than one pixel grid {sorted(keys)} (several UTM zones) - "
                         "not supported: the per-pixel mean of overlapping granules needs one grid")
    key = keys.pop()
    extent = grid.extent(key)
    plan = strip_plan([[c - extent[0], r - extent[1], w, h] for c, r, w, h in
                       (grid.window(key, f) for f in units.field_id)])
    by_date = meta.groupby("date").item_id.apply(list)
    rows, lock = [], threading.Lock()
    print(f"E1 Sentinel-2: preprocessing rasters of {len(by_date)} dates ({extent[2]} x {extent[3]} pixels, "
          f"EPSG:{key[0]}), then the table", flush=True)

    def per_date(d):
        s2_date(run, d, [recs[i] for i in by_date[d]], key, extent, plan, c2, store, out)
        r = s2_rows(run, d, len(by_date[d]), fields, grid, key, extent, out)
        with lock:
            rows.extend(r)
    bad = run_items(list(by_date.index), per_date, min(3, st["max_workers"]), 1, "E1 rasters", name=str)
    if bad:
        raise SystemExit(f"E1: the rasters of {len(bad)} dates could not be made: {bad}")
    e1 = pd.DataFrame(rows, columns=["field_id", "date", "ndvi", "valid_fraction", "n_scenes"])
    return e1.sort_values(["field_id", "date"]).reset_index(drop=True), meta


# ---------------------------------------------------------------- E2 Landsat 8 / 9
class LandsatScene:
    """One Landsat Collection 2 Level-2 scene for the EF function: the STAC item and its bands in the window that
    holds all fields of the run (plus stac.landsat.context_km on every side), on the 30 m grid of the scene.
    In step 00c a band that is read is stored as GeoTIFF first (data/stac/landsat/<run>/scenes) and read from there
    (local: function asset name -> path of the stored band); without it the band is read from the source."""

    def __init__(self, item, sensor, sign, window, transform, crs, context=None, local=None):
        self.item, self.sensor, self.sign, self.local = item, sensor, sign, local
        self.window, self.transform, self.crs = window, transform, crs
        self.shape = (int(window.height), int(window.width))
        self.datetime = item.datetime
        self.context = context          # run context of step 00c (config, weather files) for the EF function

    def read(self, asset, scaled=True):
        """Band of the scene window. scaled: DN * scale + offset of the item's raster:bands entry, no-data = NaN."""
        import rasterio
        a = self.item.assets[asset]
        if self.local is not None:
            with rasterio.open(self.local(asset)) as ds:
                x = ds.read(1)
        else:
            with rasterio.Env(**GDAL_ENV), rasterio.open(self.sign(a.href)) as ds:
                x = ds.read(1, window=self.window)
        rb = (a.extra_fields.get("raster:bands") or [{}])[0]
        if not scaled or "scale" not in rb:
            return x
        return np.where(x == rb.get("nodata", 0), np.nan, x * rb["scale"] + rb.get("offset", 0.0))


# Optional outputs of the EF function, averaged per field like EF (instantaneous fluxes at the overpass, W m-2)
EB_COLUMNS = ("rn_w_m2", "g_w_m2", "h_w_m2", "le_w_m2")
RESULT_RASTERS = {"ef": "evaporative fraction EF = LE / (Rn - G) at the overpass (0-1)",
                  "eta": "actual evapotranspiration (mm per day)",
                  "rn_w_m2": "net radiation at the overpass (W m-2)", "g_w_m2": "soil heat flux at the overpass (W m-2)",
                  "h_w_m2": "sensible heat flux at the overpass (W m-2)",
                  "le_w_m2": "latent heat flux at the overpass (W m-2)"}


def load_ef_function(spec):
    """'package.module:function' -> the function; None when no EF function is configured."""
    if not spec:
        return None
    mod, _, fn = str(spec).partition(":")
    if not fn:
        raise SystemExit(f"stac.landsat.ef_function must be 'module:function', not {spec}")
    return getattr(importlib.import_module(mod), fn)


def landsat_item(item, cand, grid, sign, cl, ef_fn, folder, rasters=None, context=None):
    """Clear share (and EF / ETa when ef_fn is given) of every candidate field in one scene. QA_PIXEL, the bands of
    cl['download_assets'] and every band the EF function reads are stored as GeoTIFF in <folder> (digital numbers of
    the source in the scene window: all fields + cl['context_km']) and listed in the record <item id>.json; all
    values are computed from the stored files. <rasters>: the clear mask of the scene and the results of the EF
    function (cl['result_rasters']) as GeoTIFF. Returns the field rows."""
    import rasterio
    from rasterio.windows import Window
    sensor = cl["platforms"][item.properties["platform"]]
    fill, bad = 1 << cl["fill_bit"], sum(1 << b for b in cl["masked_bits"])
    p_rec = folder / f"{item.id}.json"
    old, rows = read_json(p_rec) or {}, []
    rec = {k: old[k] for k in ("item_id", "datetime_utc", "source", "window", "files") if k in old}
    with rasterio.Env(**GDAL_ENV), contextlib.ExitStack() as st:
        opened = {}

        def remote(asset):
            if asset not in opened:
                opened[asset] = st.enter_context(rasterio.open(sign(item.assets[asset].href)))
            return opened[asset]
        if "source" not in rec:
            qa = remote(cl["qa_asset"])
            key = grid.key(qa)
            rec = {"item_id": item.id, "datetime_utc": item.datetime.isoformat(),
                   "source": {"grid": list(key), "origin": list(grid.origin(qa, key)), "size": [qa.width, qa.height]},
                   "window": None, "files": {}}
        src = rec["source"]
        key = (int(src["grid"][0]), *src["grid"][1:])
        win = grid.scene_window(key, src["origin"], *src["size"], cl["context_km"] * 1000)
        if rec.get("window") != win:                              # other units or margin than when it was stored
            rec.update(window=win, files={})
        write_json(p_rec, rec)
        if win is None:
            return rows

        def local(asset):
            """Path of the stored band of the scene window; downloaded when it is not there."""
            p = folder / f"{item.id}_{asset}.tif"
            if asset not in rec["files"] or not p.exists():
                ds = remote(asset)
                if (grid.key(ds) != key or list(grid.origin(ds, key)) != src["origin"]
                        or [ds.width, ds.height] != src["size"]):
                    raise SystemExit(f"{item.id}: asset {asset} is on another grid than {cl['qa_asset']}")
                rb = (item.assets[asset].extra_fields.get("raster:bands") or [{}])[0]
                f = store_clip(ds, p, win, fill=fill if asset == cl["qa_asset"] else 0, scale=rb.get("scale"),
                               offset=rb.get("offset"),
                               tags={"item_id": item.id, "asset": asset, "datetime_utc": rec["datetime_utc"],
                                     "sensor": sensor, "content": "digital numbers of the source in the scene window"})
                rec["files"][asset] = {**f, "href": item.assets[asset].href}
                write_json(p_rec, rec)
            return p
        for asset in dict.fromkeys([cl["qa_asset"], *(cl.get("download_assets") or [])]):
            local(asset)
        with rasterio.open(local(cl["qa_asset"])) as qa:
            oc, orow = grid.origin(qa, key)
            ef = eta = None
            extra = {}
            if rasters is not None:                               # preprocessing result: clear mask of the scene
                made_from = {"qa_sha256": rec["files"][cl["qa_asset"]]["sha256"], "fill_bit": cl["fill_bit"],
                             "masked_bits": list(cl["masked_bits"])}
                p_cl, p_js = rasters / f"{item.id}_clear.tif", rasters / f"{item.id}_clear.json"
                done = read_json(p_js)
                if not (done and done.get("made_from") == made_from and p_cl.exists()):
                    q = qa.read(1)
                    mask = np.where((q & fill) != 0, 255, ((q & bad) == 0)).astype(np.uint8)
                    f = write_tif(p_cl, mask, qa.transform, qa.crs, 255, tags={
                        "item_id": item.id, "datetime_utc": rec["datetime_utc"], "sensor": sensor,
                        "content": f"1 = clear ({cl['qa_asset']} bits {cl['masked_bits']} all 0), 0 = not clear, "
                                   f"255 = outside the scene (bit {cl['fill_bit']})"})
                    write_json(p_js, {"item_id": item.id, "made_from": made_from, "files": {"clear": f}})
            if ef_fn is not None:
                scene = LandsatScene(item, sensor, sign, Window(*win), qa.transform, qa.crs, context, local)
                res = ef_fn(scene)
                ef, eta = np.asarray(res["ef"], float), np.asarray(res["eta"], float)
                extra = {k: np.asarray(res[k], float) for k in EB_COLUMNS if k in res}   # optional energy-balance fluxes
                if any(x.shape != scene.shape for x in [ef, eta, *extra.values()]):
                    raise SystemExit(f"{item.id}: EF function returned arrays of another shape than {scene.shape}")
                if rasters is not None:
                    tags = {"item_id": item.id, "datetime_utc": rec["datetime_utc"], "sensor": sensor,
                            "function": cl.get("ef_function")}
                    have = {"ef": ef, "eta": eta, **extra}
                    write_json(rasters / f"{item.id}_ef.json", {"item_id": item.id, "files": {
                        n: write_tif(rasters / f"{item.id}_{n}.tif", have[n].astype(np.float32), qa.transform, qa.crs,
                                     np.nan, tags={**tags, "content": RESULT_RASTERS[n]})
                        for n in (cl.get("result_rasters") or []) if n in have}})
            for fid in cand:
                gc, gr, w = grid.weights(key, fid)
                fc, fr, (h, wd) = gc - oc, gr - orow, w.shape
                q = read_window(qa, fc, fr, wd, h, fill=fill)
                data = (q & fill) == 0                            # pixels of the scene (fill = outside)
                if not w[data].sum() > 0:
                    continue                                      # field outside the scene
                clear = data & ((q & bad) == 0)
                row = {"field_id": fid, "date": item.datetime.date().isoformat(), "sensor": sensor, "ef": np.nan,
                       "eta_mm_day": np.nan, **{k: np.nan for k in extra},
                       "valid_fraction": (w * clear).sum() / w[data].sum(), "item_id": item.id}
                if ef is not None:
                    def crop(x):
                        o = np.full((h, wd), np.nan)
                        a0, b0, a1, b1 = max(fc, 0), max(fr, 0), min(fc + wd, x.shape[1]), min(fr + h, x.shape[0])
                        if a1 > a0 and b1 > b0:
                            o[b0 - fr:b1 - fr, a0 - fc:a1 - fc] = x[b0:b1, a0:a1]
                        return o
                    for name, x in (("ef", crop(ef)), ("eta_mm_day", crop(eta)), *((k, crop(v)) for k, v in extra.items())):
                        ok = clear & ~np.isnan(x)
                        if w[ok].sum() > 0:
                            row[name] = (w[ok] * x[ok]).sum() / w[ok].sum()
                rows.append(row)
    return rows


def scene_fingerprint(cfg, fields, ef_fn, context):
    """Everything a scene result depends on: EF function (and its code), SEBAL and Landsat settings, the fields, the
    daily weather. A cached scene with another fingerprint is computed again."""
    import hashlib
    import inspect
    cl = cfg["stac"]["landsat"]
    h = hashlib.sha256()
    h.update(json.dumps({"ef_function": cl.get("ef_function"), "collection": cl["collection"],
                         "qa": [cl["qa_asset"], cl["fill_bit"], cl["masked_bits"]], "context_km": cl["context_km"],
                         "sebal": cfg.get("sebal") if ef_fn else None}, sort_keys=True, default=str).encode())
    f = fields.sort_values("field_id")
    h.update(("|".join(f.field_id) + "|" + "|".join(g.wkb_hex for g in f.geometry)).encode())
    if ef_fn is not None:
        h.update(Path(inspect.getsourcefile(ef_fn)).read_bytes())
        if context and Path(context["e3_file"]).exists():
            h.update(Path(context["e3_file"]).read_bytes())
        if context and context.get("soil_file"):                          # soils of the hot-pixel water balance
            h.update(Path(context["soil_file"]).read_bytes())
        h.update(Path(inspect.getsourcefile(importlib.import_module("core.soil_evaporation"))).read_bytes())
    return h.hexdigest()


def landsat_export(cfg, fields, start, end, folder, context=None, rasters=None):
    """E2 rows per field and Landsat 8/9 overpass. Returns (table, item table, True when EF was computed).
    context: run context handed to the EF function with every scene (step 00c: config, weather files).
    The bands of every scene are stored as GeoTIFF in <folder>/scenes and kept (a band is downloaded once). Every
    scene's result (field rows and, with an EF function, its anchor-log row) is kept in <folder>/scenes/<item id>.json
    with a fingerprint of everything it depends on: a run that stopped (window closed, laptop off) continues with
    the scenes not yet done, and a scene is computed again - from the stored bands - only when its fingerprint
    changed. <folder>/sebal_anchors.csv is rebuilt from the scenes of the period. rasters: folder of the raster
    results (clear mask per scene, EF and ETa)."""
    cl, st = cfg["stac"]["landsat"], cfg["stac"]
    folder.mkdir(parents=True, exist_ok=True)
    bbox = [float(x) for x in fields.total_bounds]
    query = {"platform": {"in": list(cl["platforms"])}}
    if cl.get("collection_category"):
        query["landsat:collection_category"] = {"eq": cl["collection_category"]}
    items = search(cl["api"], cl["collection"], bbox, start, end, query=query, tries=st["retries"])
    cand = {i.id: fields_in_item(i, fields) for i in items}
    items = [i for i in items if cand[i.id]]
    if not items:
        raise SystemExit(f"no {cl['collection']} items for the fields from {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    grid, sign, ef_fn = FieldGrid(fields), signer(cl.get("sign")), load_ef_function(cl.get("ef_function"))
    scenes = folder / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    ras = None
    if rasters is not None:
        ras = rasters / "landsat"                                 # clear mask per scene; results of the EF function
        ras.mkdir(parents=True, exist_ok=True)
    want = [n for n in (cl.get("result_rasters") or []) if n in RESULT_RASTERS] if ef_fn is not None else []
    fp = scene_fingerprint(cfg, fields, ef_fn, context)

    def cached(i):
        d = read_json(scenes / f"{i.id}.json")
        if not d or d.get("fingerprint") != fp or "rows" not in d:
            return None
        if not all((scenes / f["file"]).exists() for f in (d.get("files") or {}).values()):
            return None                                           # a stored band is missing: the scene is done again
        if ras is not None and d.get("window") and not ((ras / f"{i.id}_clear.tif").exists()
                                                        and all((ras / f"{i.id}_{n}.tif").exists() for n in want)):
            return None
        return d
    done = {i.id: cached(i) for i in items}
    todo = [i for i in items if done[i.id] is None]
    print(f"E2 Landsat: {len(items)} scenes on {len({i.datetime.date() for i in items})} dates; "
          f"{len(items) - len(todo)} already done, {len(todo)} to compute; "
          f"EF function: {cl.get('ef_function') or 'none configured'}", flush=True)
    lock = threading.Lock()

    def work(i):
        ctx = {**context, "anchor_sink": []} if context else None
        r = landsat_item(i, cand[i.id], grid, sign, cl, ef_fn, scenes, ras, ctx)
        d = {**(read_json(scenes / f"{i.id}.json") or {}), "fingerprint": fp, "item_id": i.id, "rows": r,
             "anchor": ctx["anchor_sink"] if ctx else []}
        if not cl.get("keep_source_bands", True) and ras is not None:    # results stored: the bands are not kept
            for f in (d.get("files") or {}).values():
                (scenes / f["file"]).unlink(missing_ok=True)
            d["files"] = {}
        write_json(scenes / f"{i.id}.json", d)
        with lock:
            done[i.id] = d
    failed = run_items(todo, work, st["max_workers"] if ef_fn is None else min(4, st["max_workers"]),
                       st["retries"], "E2")
    out = [r for i in items if done.get(i.id) for r in done[i.id]["rows"]]
    anchors = [a for i in items if done.get(i.id) for a in done[i.id]["anchor"]]
    if ef_fn is not None and anchors:
        import sys as _sys
        cols = getattr(_sys.modules.get(ef_fn.__module__), "LOG_FIELDS", None)
        config.to_csv_atomic(pd.DataFrame(anchors, columns=cols), folder / "sebal_anchors.csv", index=False)
    else:
        (folder / "sebal_anchors.csv").unlink(missing_ok=True)              # no anchors of this period
    meta = pd.DataFrame([{"item_id": i.id, "date": i.datetime.date().isoformat(), "datetime_utc": i.datetime.isoformat(),
                          "sensor": cl["platforms"][i.properties["platform"]],
                          "wrs_path": i.properties.get("landsat:wrs_path"), "wrs_row": i.properties.get("landsat:wrs_row"),
                          "collection_category": i.properties.get("landsat:collection_category"),
                          "cloud_cover_pct": i.properties.get("eo:cloud_cover"), "candidate_fields": len(cand[i.id]),
                          "status": "failed: " + failed[i.id] if i.id in failed else "read"} for i in items])
    config.to_csv_atomic(meta, folder / "landsat_items.csv", index=False)
    write_index(folder / "landsat_files.csv", scenes)
    if failed:
        raise DownloadError(f"E2: {len(failed)} scenes could not be read (see {folder / 'landsat_items.csv'}); "
                            "run the step again - the scenes already done and the bands already stored are kept")
    cols = ["field_id", "date", "sensor", "ef", "eta_mm_day", *[k for k in EB_COLUMNS if any(k in r for r in out)],
            "valid_fraction", "item_id"]
    e2 = pd.DataFrame(out, columns=cols).sort_values(["field_id", "date", "sensor", "item_id"]).reset_index(drop=True)
    return e2, meta, ef_fn is not None


def landsat_rows(ras, item_id, date, sensor, fields, grid):
    """Field rows of one scene from its stored rasters (<ras>/<item id>_clear.tif, _ef.tif, _eta.tif): clear
    share of the field's pixels in the scene and the area-weighted EF and ETa of its clear pixels with a value - the
    values landsat_item computes for the fields of step 00c."""
    import rasterio
    with rasterio.open(ras / f"{item_id}_clear.tif") as dc:
        key = FieldGrid.key(dc)
        (oc, orow), cl, tr = FieldGrid.origin(dc, key), dc.read(1), dc.transform
    val = {}
    for n, col in (("ef", "ef"), ("eta", "eta_mm_day")):
        p = ras / f"{item_id}_{n}.tif"
        if p.exists():
            with rasterio.open(p) as ds:
                if ds.transform != tr or ds.shape != cl.shape:
                    raise SystemExit(f"{p.name}: not on the grid of the clear mask of the scene")
                val[col] = ds.read(1).astype(float)
    rows = []
    for fid in fields.field_id:
        gc, gr, w = grid.weights(key, fid)
        fc, fr, (h, wd) = gc - oc, gr - orow, w.shape
        q = array_window(cl, fc, fr, wd, h, 255)
        data = q != 255                                           # pixels of the scene
        if not w[data].sum() > 0:
            continue                                              # field outside the scene
        clear = q == 1
        row = {"field_id": fid, "date": date, "sensor": sensor, "ef": np.nan, "eta_mm_day": np.nan,
               "valid_fraction": (w * clear).sum() / w[data].sum(), "item_id": item_id}
        for col, x in val.items():
            v = array_window(x, fc, fr, wd, h, np.nan)
            ok = clear & ~np.isnan(v)
            if w[ok].sum() > 0:
                row[col] = (w[ok] * v[ok]).sum() / w[ok].sum()
        rows.append(row)
    return rows


def overpass_rows(e2):
    """One row per field and overpass: a field in the overlap of two scenes of one overpass takes the mean EF, ETa
    (and fluxes) of the two scenes and the larger clear share (export E2)."""
    means = {c: (c, "mean") for c in ["ef", "eta_mm_day", *EB_COLUMNS] if c in e2}
    return e2.groupby(["field_id", "date", "sensor"], as_index=False).agg(**means,
                                                                         valid_fraction=("valid_fraction", "max"))


# ---------------------------------------------------------------- E3 ERA5-Land and elevation
def cells_of(fields):
    """0.1 degree cells of the sample: id 'lat_lon' -> centre latitude and longitude."""
    c = pd.DataFrame({"cell": sorted(fields.cell.unique())})
    c["lat"] = c.cell.str.split("_").str[0].astype(float)
    c["lon"] = c.cell.str.split("_").str[1].astype(float)
    return c


def dem_cells(cfg, cells, folder):
    """Elevation tiles of the STAC DEM collection that hold the cells, as one GeoTIFF on the grid of the tiles
    (<collection>_cells.tif: values of the source; tiles share their edge row / column). Kept when it holds the
    cells. Returns (path, record)."""
    import pystac_client
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    cd, st = cfg["stac"]["dem"], cfg["stac"]
    folder.mkdir(parents=True, exist_ok=True)
    path, p_rec = folder / f"{cd['collection']}_cells.tif", folder / f"{cd['collection']}_cells.json"
    m = cfg["aoi"]["inputs"]["era5_cell_deg"] / 2 + 0.01           # whole cells and a margin
    bounds = [round(float(v), 6) for v in (cells.lon.min() - m, cells.lat.min() - m, cells.lon.max() + m,
                                           cells.lat.max() + m)]
    rec = read_json(p_rec)
    if (rec and rec.get("bounds") == bounds and rec.get("collection") == cd["collection"]
            and rec.get("asset") == cd["asset"] and path.exists()):
        return path, rec
    items = retry(lambda: list(pystac_client.Client.open(cd["api"]).search(collections=[cd["collection"]],
                                                                         bbox=bounds).items()),
                  st["retries"], f"search {cd['collection']}")
    if not items:
        raise SystemExit(f"no {cd['collection']} items for the cells")
    sign, tiles, ref = signer(cd.get("sign")), [], None
    with rasterio.Env(**GDAL_ENV):
        for i in items:
            ds = retry(lambda: rasterio.open(sign(i.assets[cd["asset"]].href)), st["retries"], i.id)
            with ds:
                t = ds.transform
                if abs(t.b) > 1e-12 or abs(t.d) > 1e-12 or abs(t.a + t.e) > 1e-12:
                    raise SystemExit(f"{i.id}: elevation tile is not north-up with square pixels")
                if ref is None:
                    res, nodata = t.a, (ds.nodata if ds.nodata is not None else -32768)
                    ref = (t.c, t.f, ds.crs, ds.dtypes[0])
                    c_lo, c_hi = math.floor((bounds[0] - t.c) / res), math.ceil((bounds[2] - t.c) / res)
                    r_lo, r_hi = math.floor((t.f - bounds[3]) / res), math.ceil((t.f - bounds[1]) / res)
                    mosaic = np.full((r_hi - r_lo, c_hi - c_lo), nodata, ds.dtypes[0])
                elif abs(t.a - res) > 1e-12 or ds.crs != ref[2] or ds.dtypes[0] != ref[3]:
                    raise SystemExit(f"{i.id}: elevation tile on another grid than {items[0].id}")
                dc, dr = (t.c - ref[0]) / res - c_lo, (ref[1] - t.f) / res - r_lo
                if abs(dc - round(dc)) > 1e-3 or abs(dr - round(dr)) > 1e-3:
                    raise SystemExit(f"{i.id}: elevation tile is shifted against {items[0].id}")
                dc, dr = round(dc), round(dr)
                a0, b0 = max(dc, 0), max(dr, 0)
                a1, b1 = min(dc + ds.width, mosaic.shape[1]), min(dr + ds.height, mosaic.shape[0])
                if a1 > a0 and b1 > b0:
                    a = retry(lambda: ds.read(1, window=Window(a0 - dc, b0 - dr, a1 - a0, b1 - b0)), st["retries"], i.id)
                    ok = a != nodata
                    mosaic[b0:b1, a0:a1][ok] = a[ok]
                tiles.append({"id": i.id, "bounds": [float(v) for v in ds.bounds],
                              "href": i.assets[cd["asset"]].href})
    transform = from_origin(ref[0] + c_lo * res, ref[1] - r_lo * res, res, res)
    f = write_tif(path, mosaic, transform, ref[2], nodata,
                  tags={"collection": cd["collection"], "asset": cd["asset"], "tiles": ";".join(t["id"] for t in tiles),
                        "content": "elevation of the source tiles (m) on their grid"})
    rec = {"collection": cd["collection"], "asset": cd["asset"], "bounds": bounds, "tiles": tiles,
           "files": {"elevation": f}}
    write_json(p_rec, rec)
    return path, rec


def cell_elevation(cfg, cells, folder):
    """Mean terrain elevation in a window of stac.dem.window_m around every cell centre, from the stored elevation
    GeoTIFF of the STAC DEM collection (dem_cells). Written to <folder>/cell_elevation.csv with its settings
    (cell_elevation.json)."""
    import rasterio
    from rasterio.windows import from_bounds
    cd = cfg["stac"]["dem"]
    path, rec = dem_cells(cfg, cells, folder)
    half = cd["window_m"] / 2
    dlat = half / 111320.0
    rows = []
    with rasterio.open(path) as ds:
        for c in cells.itertuples():
            dlon = half / (111320.0 * math.cos(math.radians(c.lat)))
            box = (c.lon - dlon, c.lat - dlat, c.lon + dlon, c.lat + dlat)
            win = from_bounds(*box, transform=ds.transform).round_offsets().round_lengths()
            a = read_window(ds, int(win.col_off), int(win.row_off), int(win.width), int(win.height),
                            fill=-32768).astype(float)
            if ds.nodata is not None:
                a[a == ds.nodata] = np.nan
            a[a == -32768] = np.nan
            v = a[~np.isnan(a)]
            if not len(v):
                raise SystemExit(f"cell {c.cell}: no elevation pixels")
            used = [t["id"] for t in rec["tiles"] if not (box[0] >= t["bounds"][2] or box[2] <= t["bounds"][0]
                                                          or box[1] >= t["bounds"][3] or box[3] <= t["bounds"][1])]
            rows.append({"cell": c.cell, "lat": c.lat, "lon": c.lon, "elev_m": float(v.mean()), "n_pixels": len(v),
                         "items": ";".join(used)})
    t = pd.DataFrame(rows)
    config.to_csv_atomic(t, folder / "cell_elevation.csv", index=False)
    config.write_atomic(folder / "cell_elevation.json", lambda tmp: tmp.write_text(
        json.dumps({k: cd[k] for k in ("api", "collection", "asset", "window_m")}), encoding="utf-8"))
    return t


def era5_tifs(stem, v):
    """{netCDF name: GeoTIFF path} of one month of hourly values; stem = folder / file name without extension."""
    return {n: stem.with_name(f"{stem.name}_{n}.tif") for n in v.values()}


def era5_stems(raw):
    """The months of hourly GeoTIFFs in a folder (their records <stem>.tif.json), as [folder / stem]."""
    return [p.with_name(p.name[:-len(".tif.json")]) for p in sorted(Path(raw).glob("*.tif.json"))]


def nc_to_tifs(nc, stem, v, cell_deg, source, dataset=""):
    """Hourly netCDF file of the Climate Data Store -> one GeoTIFF per variable (<stem>_<netCDF name>.tif):
    band = hour (band description = UTC time), pixel = grid node, values of the file unchanged. v: {CDS name:
    netCDF name}. Every GeoTIFF is read again and compared with the netCDF values. Writes the record
    <stem>.tif.json."""
    import rasterio
    import xarray as xr
    from rasterio.transform import from_origin
    with xr.open_dataset(nc, engine="netcdf4") as d:
        tname = "valid_time" if "valid_time" in d.dims else "time"
        miss = [n for n in v.values() if n not in d]
        if miss:
            raise SystemExit(f"{nc.name}: variables {miss} missing (has {list(d.data_vars)})")
        d = d.sortby(tname).sortby("latitude", ascending=False).sortby("longitude")
        lat, lon = d.latitude.values.astype(float), d.longitude.values.astype(float)
        res = float(cell_deg)
        step = {round(float(x), 4) for x in [*np.abs(np.diff(lat)), *np.diff(lon)]}
        if step - {round(res, 4)}:
            raise SystemExit(f"{nc.name}: grid steps {sorted(step)} - not the {res} degree grid")
        times = pd.DatetimeIndex(d[tname].values)
        text = [f"{t:%Y-%m-%dT%H:%M:%SZ}" for t in times]
        transform = from_origin(round(lon[0] - res / 2, 6), round(lat[0] + res / 2, 6), res, res)
        files = {}
        for cds_name, n in v.items():
            a = d[n].transpose(tname, "latitude", "longitude").values
            p = era5_tifs(stem, v)[n]
            files[n] = write_tif(p, a, transform, "EPSG:4326", np.nan, descriptions=text,
                                 tags={"dataset": dataset, "variable": cds_name, "netcdf_name": n,
                                       "units": d[n].attrs.get("units", ""), "long_name": d[n].attrs.get("long_name", ""),
                                       "band": "one hour, band description = valid time (UTC)",
                                       "pixel": "grid node at the pixel centre", "source": source})
            with rasterio.open(p) as ds:
                if not np.array_equal(ds.read(), a, equal_nan=True) or list(ds.descriptions) != text:
                    raise SystemExit(f"{p.name}: the GeoTIFF does not hold the values of {nc.name}")
    rec = {"source": source, "netcdf_sha256": config.sha256(nc), "variables": dict(v), "hours": len(times),
           "first_hour_utc": text[0], "last_hour_utc": text[-1],
           "grid": {"west": transform.c, "north": transform.f, "step_deg": res, "width": len(lon), "height": len(lat)},
           "files": files}
    write_json(stem.with_name(stem.name + ".tif.json"), rec)
    return rec


def era5land_hourly(cfg, cells, start, end, folder, variables=None, sub="raw"):
    """Hourly ERA5-Land of the cell centres from (start - 1 day) to end (UTC), as GeoTIFF per month and variable in
    <folder>/<sub> (variables: {CDS name: netCDF name}, default config stac.era5land.variables). The dataset is
    looked up in the STAC catalogue of the Climate Data Store first. The Climate Data Store delivers netCDF: the
    file is converted to GeoTIFF at once, compared value by value and removed. A netCDF file already in the folder
    (download of v0.5.0 or earlier) is converted without a new request and left where it is. A month already stored
    is kept and not requested again. Returns the months as [folder / file name without extension]."""
    import requests
    ce = cfg["stac"]["era5land"]
    v = dict(variables if variables is not None else ce["variables"])
    cell_deg = cfg["aoi"]["inputs"]["era5_cell_deg"]
    raw = folder / sub
    raw.mkdir(parents=True, exist_ok=True)
    url = f"{ce['catalogue']}/collections/{ce['dataset']}"
    col = retry(lambda: requests.get(url, timeout=120), cfg["stac"]["retries"], "CDS catalogue")
    col.raise_for_status()
    col = col.json()
    iv = col["extent"]["temporal"]["interval"][0]
    rec = {"url": url, "type": col.get("type"), "stac_version": col.get("stac_version"), "id": col.get("id"),
           "temporal_interval": iv, "checked_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    config.write_atomic(folder / "era5land_collection.json",
                        lambda tmp: tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8"))
    if col.get("type") != "Collection" or not col.get("stac_version"):
        raise SystemExit(f"{url} is not a STAC collection")
    if pd.Timestamp(iv[1]).tz_localize(None) < end:
        raise SystemExit(f"{ce['dataset']} ends {iv[1]} - before the export end {end:%Y-%m-%d}")
    area = [round(float(cells.lat.max()), 2), round(float(cells.lon.min()), 2), round(float(cells.lat.min()), 2),
            round(float(cells.lon.max()), 2)]                       # north, west, south, east
    days = pd.date_range(start - pd.Timedelta(days=1), end, freq="D")
    stems = []
    for (y, m), g in pd.Series(days, index=days).groupby([days.year, days.month]):
        stem = raw / f"era5land_hourly_{y}_{m:02d}_d{g.dt.day.min():02d}-{g.dt.day.max():02d}"
        stems.append(stem)
        old = read_json(stem.with_name(stem.name + ".tif.json"))
        if old and old.get("variables") == v and all(p.exists() for p in era5_tifs(stem, v).values()):
            continue
        nc = stem.with_name(stem.name + ".nc")
        if nc.exists():
            print(f"E3 ERA5-Land: {sub}/{nc.name} is on disk - converted to GeoTIFF, no new request", flush=True)
            nc_to_tifs(nc, stem, v, cell_deg, "netCDF file of the Climate Data Store (cdsapi), kept next to the "
                       "GeoTIFFs", ce["dataset"])
            continue
        req = {"variable": list(v), "year": str(y), "month": f"{m:02d}",
               "day": [f"{d:02d}" for d in g.dt.day], "time": [f"{h:02d}:00" for h in range(24)],
               "data_format": "netcdf", "download_format": "unarchived", "area": area}
        import cdsapi
        print(f"E3 ERA5-Land: requesting {y}-{m:02d} ({len(g)} days) from the Climate Data Store", flush=True)
        tmp = stem.with_name(stem.name + ".nc.part")
        try:
            retry(lambda: cdsapi.Client(quiet=True, progress=False).retrieve(ce["dataset"], req, str(tmp)),
                  cfg["stac"]["retries"], f"CDS {y}-{m:02d}")
        except Exception as e:  # noqa: BLE001 - any failure of the request ends the step as a download error
            raise DownloadError(f"E3: the Climate Data Store request for {y}-{m:02d} failed after "
                                f"{cfg['stac']['retries']} tries ({type(e).__name__}: {str(e)[:200]}); run the step "
                                "again - the months already retrieved are kept") from None
        config.write_atomic(stem.with_name(stem.name + ".request.json"), lambda t: t.write_text(
            json.dumps({"dataset": ce["dataset"], **req}, indent=1), encoding="utf-8"))
        nc_to_tifs(tmp, stem, v, cell_deg, "netCDF file of the Climate Data Store (cdsapi), removed after the "
                   "conversion", ce["dataset"])
        tmp.unlink()
    return stems


def era5_tif(path):
    """(hours [UTC, from the band descriptions], values[hour, row, col], transform) of one hourly GeoTIFF."""
    import rasterio
    with rasterio.open(path) as ds:
        return pd.to_datetime(list(ds.descriptions), utc=True).tz_localize(None), ds.read(), ds.transform


def grid_nodes(transform, shape, cells, name=""):
    """(row, col, largest distance in degrees) of the pixels of an hourly GeoTIFF whose centre (grid node) is the
    centre of each cell."""
    lon, lat = cells.lon.to_numpy(), cells.lat.to_numpy()
    col = np.round((lon - transform.c) / transform.a - 0.5).astype(int)
    row = np.round((transform.f - lat) / -transform.e - 0.5).astype(int)
    if col.min() < 0 or row.min() < 0 or col.max() >= shape[1] or row.max() >= shape[0]:
        raise SystemExit(f"{name}: the hourly GeoTIFF does not hold every cell")
    far = max(float(np.abs(transform.c + (col + 0.5) * transform.a - lon).max()),
              float(np.abs(transform.f + (row + 0.5) * transform.e - lat).max()))
    return row, col, far


def era5_hours(stems, cells, v):
    """Hourly values at the cell centres from the hourly GeoTIFFs: (times [UTC], {netCDF name: array[hour, cell]},
    largest distance in degrees between a cell centre and its grid node)."""
    times, data, far = [], {n: [] for n in v.values()}, 0.0
    for stem in stems:
        month = None
        for n, p in era5_tifs(stem, v).items():
            if not p.exists():
                raise SystemExit(f"{p} missing - run step 00c (--only e3)")
            hours, a, t = era5_tif(p)
            row, col, d = grid_nodes(t, a.shape[1:], cells, p.name)
            far = max(far, d)
            if month is None:
                month = hours
            elif not hours.equals(month):
                raise SystemExit(f"{p.name}: other hours than the other variables of the month")
            data[n].append(a[:, row, col])
        times.append(month)
    times = pd.DatetimeIndex(np.concatenate([t.to_numpy() for t in times])).astype("datetime64[ns]")
    return times, {n: np.concatenate(a) for n, a in data.items()}, far


def hourly_frame(stems, cells, v):
    """The hourly GeoTIFFs as a table: cell, time, one column per netCDF name; sorted by cell and time."""
    times, data, far = era5_hours(stems, cells, v)
    if far > 1e-3:
        raise SystemExit(f"{stems[0].parent}: grid nodes up to {far:.4f} degree away from the cell centres")
    h = pd.DataFrame({"cell": np.repeat(cells.cell.to_numpy(), len(times)), "time": np.tile(times, len(cells)),
                      **{n: a.T.ravel() for n, a in data.items()}})
    return h.drop_duplicates(["cell", "time"]).sort_values(["cell", "time"]).reset_index(drop=True)


def era5land_daily(cfg, cells, stems, start, end, utc_offset_hours):
    """Daily values per cell and local calendar day (UTC + utc_offset_hours) from the hourly GeoTIFFs.
    rs = sum of the hourly surface solar radiation downwards (J m-2 -> MJ m-2); tmin / tmax = minimum / maximum of
    the hourly 2 m temperature; tdew = mean 2 m dewpoint; u10 = mean of the hourly 10 m wind speed sqrt(u2 + v2);
    sp = mean surface pressure (Pa -> kPa). Every day needs its 24 hours."""
    v = cfg["stac"]["era5land"]["variables"]                       # CDS name -> netCDF name
    h = hourly_frame(stems, cells, v)
    names = {n: k for k, n in v.items()}
    h = h.rename(columns={n: {"2m_temperature": "t2m", "2m_dewpoint_temperature": "d2m",
                              "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
                              "surface_pressure": "sp", "surface_solar_radiation_downwards": "ssrd"}[names[n]]
                          for n in v.values()})
    step = h.groupby("cell").time.diff().dropna().unique()
    if len(step) != 1 or step[0] != pd.Timedelta(hours=1):
        raise SystemExit(f"ERA5-Land hourly series has gaps (time steps {step})")
    # ssrd is accumulated from 00 UTC: the value at 01 UTC is the first hour, at 00 UTC the total of the day before
    prev = h.groupby("cell").ssrd.shift()
    h["ssrd_h"] = np.where(h.time.dt.hour == 1, h.ssrd, h.ssrd - prev)
    off = pd.Timedelta(hours=utc_offset_hours)
    h["day"] = (h.time + off).dt.floor("D")                         # instantaneous values: local day of the hour
    h["day_flux"] = (h.time - pd.Timedelta(minutes=30) + off).dt.floor("D")   # hour ending at 'time': its middle
    h["wind"] = np.hypot(h.u10, h.v10)
    a = h.groupby(["cell", "day"]).agg(tmin_c=("t2m", "min"), tmax_c=("t2m", "max"), tdew_c=("d2m", "mean"),
                                       u10_ms=("wind", "mean"), sp_kpa=("sp", "mean"), n=("t2m", "size"),
                                       n_missing=("t2m", lambda x: int(x.isna().sum())))
    b = h.dropna(subset=["ssrd_h"]).groupby(["cell", "day_flux"]).agg(rs_mj_m2=("ssrd_h", "sum"),
                                                                       n_flux=("ssrd_h", "size"))
    b.index = b.index.set_names(["cell", "day"])
    t = a.join(b, how="left").reset_index().rename(columns={"day": "date"})
    t = t[(t.date >= start) & (t.date <= end)].copy()
    n_days = (end - start).days + 1
    bad = t[(t.n != 24) | (t.n_flux != 24) | (t.n_missing > 0)]
    if len(bad) or (t.groupby("cell").size() != n_days).any() or t.cell.nunique() != len(cells):
        raise SystemExit(f"E3: incomplete days ({len(bad)} cell-days without 24 hours; "
                         f"{t.groupby('cell').size().min()} of {n_days} days in the shortest cell)")
    t["tmin_c"], t["tmax_c"], t["tdew_c"] = t.tmin_c - 273.15, t.tmax_c - 273.15, t.tdew_c - 273.15
    t["sp_kpa"], t["rs_mj_m2"] = t.sp_kpa / 1000.0, t.rs_mj_m2 / 1e6
    return t.drop(columns=["n", "n_flux", "n_missing"]), float(h.ssrd_h.min())


def era5land_daily_precip(cells, stems, name, start, end, utc_offset_hours):
    """Daily precipitation (mm) per cell and local day from the hourly ERA5-Land total precipitation (m), accumulated
    from 00 UTC like ssrd (ECMWF ERA5-Land documentation): hourly = difference of the accumulated values, the value
    at 01 UTC being the first hour; summed over the 24 hours ending in the local day (by the middle of each hour).
    stems: the months of hourly GeoTIFFs; name: netCDF name of the variable."""
    h = hourly_frame(stems, cells, {"total_precipitation": name}).rename(columns={name: "tp"})
    prev = h.groupby("cell").tp.shift()
    h["tp_h"] = np.where(h.time.dt.hour == 1, h.tp, h.tp - prev)
    h["tp_h"] = h.tp_h.clip(lower=0)                               # rounding of the accumulated values
    h["day"] = (h.time - pd.Timedelta(minutes=30) + pd.Timedelta(hours=utc_offset_hours)).dt.floor("D")
    t = h.dropna(subset=["tp_h"]).groupby(["cell", "day"]).agg(precip_mm=("tp_h", "sum"), n=("tp_h", "size"))
    t = t.reset_index().rename(columns={"day": "date"})
    t = t[(t.date >= start) & (t.date <= end)]
    if (t.n != 24).any():
        raise SystemExit(f"E3 precipitation: {int((t.n != 24).sum())} cell-days without 24 hours")
    t["precip_mm"] = t.precip_mm * 1000.0
    return t[["cell", "date", "precip_mm"]]


def weather_export(cfg, fields, start, end, folder, dem_folder, utc_offset_hours):
    """E3: one row per cell and day. Returns (table, elevation table, months of hourly GeoTIFFs, smallest hourly
    radiation). Precipitation (config stac.era5land.precipitation) is retrieved into its own folder raw_precip/, so
    the hourly files of the other variables stay valid."""
    cells = cells_of(fields)
    elev = cell_elevation(cfg, cells, dem_folder)
    stems = era5land_hourly(cfg, cells, start, end, folder)
    t, ssrd_min = era5land_daily(cfg, cells, stems, start, end, utc_offset_hours)
    cols = ["cell", "date", "rs_mj_m2", "tmin_c", "tmax_c", "tdew_c", "u10_ms", "sp_kpa", "elev_m", "lat"]
    pr = cfg["stac"]["era5land"].get("precipitation")
    if pr:
        sp = era5land_hourly(cfg, cells, start, end, folder, variables=pr, sub="raw_precip")
        t = t.merge(era5land_daily_precip(cells, sp, list(pr.values())[0], start, end, utc_offset_hours),
                    on=["cell", "date"], how="left", validate="one_to_one")
        stems = stems + sp
        cols.append("precip_mm")
    t = t.merge(elev[["cell", "elev_m", "lat"]], on="cell", validate="many_to_one")
    return t[cols].sort_values(["cell", "date"]).reset_index(drop=True), elev, stems, ssrd_min


def weather_rasters(cfg, e3, out):
    """The daily weather table E3 as rasters on the grid of the cells (EPSG:4326, pixel = cell): one GeoTIFF per
    variable with one band per day (band description = date), and the elevation of the cells. Also the two
    values that step 01 derives from E3, computed with the same functions (core/eto.py): tmean_c = (Tmin + Tmax) / 2
    and eto_mm = FAO-56 Penman-Monteith ETo. The table E3 itself is not changed. Cells that are not in the table
    are no data. Returns {name: path}."""
    from rasterio.transform import from_origin

    from core import eto
    out.mkdir(parents=True, exist_ok=True)
    run, res = cfg["run_id"], float(cfg["aoi"]["inputs"]["era5_cell_deg"])
    e3 = e3.assign(date=pd.to_datetime(e3.date))
    e3 = e3.assign(tmean_c=(e3.tmin_c + e3.tmax_c) / 2,
                   eto_mm=eto.eto_fao56(e3.tmin_c, e3.tmax_c, e3.tdew_c, e3.rs_mj_m2, eto.wind_2m(e3.u10_ms), e3.sp_kpa,
                                        e3.lat, e3.elev_m, e3.date.dt.dayofyear))
    lat = e3.cell.str.split("_").str[0].astype(float).to_numpy()
    lon = e3.cell.str.split("_").str[1].astype(float).to_numpy()
    row, col = np.round((lat.max() - lat) / res).astype(int), np.round((lon - lon.min()) / res).astype(int)
    days = pd.DatetimeIndex(sorted(e3.date.unique()))
    day = days.get_indexer(pd.DatetimeIndex(e3.date))
    transform = from_origin(round(lon.min() - res / 2, 6), round(lat.max() + res / 2, 6), res, res)
    shape, written = (int(row.max()) + 1, int(col.max()) + 1), {}
    tags = {"dataset": cfg["stac"]["era5land"]["dataset"], "pixel": f"{res} degree cell of the run",
            "day": f"local calendar day, UTC{cfg['aoi']['local_day_utc_offset_hours']:+g} h"}
    for name in ("rs_mj_m2", "tmin_c", "tmax_c", "tdew_c", "u10_ms", "sp_kpa", "precip_mm", "tmean_c", "eto_mm"):
        if name not in e3:
            continue
        a = np.full((len(days), *shape), np.nan, np.float32)
        a[day, row, col] = e3[name].to_numpy()
        written[f"raster_{name}"] = out / f"{run}_{name}.tif"
        write_tif(written[f"raster_{name}"], a, transform, "EPSG:4326", np.nan,
                  descriptions=[f"{d:%Y-%m-%d}" for d in days],
                  tags={**tags, "variable": name, "units": WEATHER_UNITS[name], "band": "one day, band description = date"})
    a = np.full(shape, np.nan, np.float32)
    a[row, col] = e3.elev_m.to_numpy()
    written["raster_elev_m"] = out / f"{run}_elev_m.tif"
    write_tif(written["raster_elev_m"], a, transform, "EPSG:4326", np.nan,
              tags={"variable": "elev_m", "units": "m", "pixel": tags["pixel"],
                    "content": f"mean elevation in a {cfg['stac']['dem']['window_m']} m window at the cell centre"})
    return written
