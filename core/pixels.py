"""Pixel level (v0.9.0): the pixels of the crop map, their Sentinel-2 and Landsat values read from the stored rasters
of step 00c (pixel stacks, export PX), the LUE model of steps 03-05 for every pixel, and the aggregation pixel ->
parcel -> ERA5-Land cell, tehsil, district.

Pixel       a pixel of the 10 m grid of the Sentinel-2 NDVI rasters whose centre lies inside a parcel of the crop map
            (a pixel inside two overlapping parcels is a pixel of each). A parcel without such a pixel takes the one
            pixel that holds its representative point (pixel_rule 'representative point'), so every parcel has a
            value.
Stacks      written by step 00c (export px) to <folder of the export files>/<run>_pixels/:
              ndvi.npy        dates x pixels, int16: NDVI / pixels.ndvi_scale; pixels.nodata = no NDVI (not clear,
                              not observed). A pixel has NDVI only where its scene classification is valid
                              (core/stac_ingest.s2_pixels), so the pixel's clear share of a date is 1 with NDVI and 0
                              without: the date is used or not, exactly as a field unit with a clear share of 1 or 0.
              landsat_ef.npy  overpasses x pixels, int16 (v0.9.0): the Landsat values of the 10 m pixel SQUARE by the
              landsat_eta.npy rules of the field export E2 (stac_ingest.landsat_rows, overpass_rows) for a unit that
              landsat_clear.npy is the square: per scene the clear share of the square and the area-weighted EF and
                              ETa of its clear 30 m pixels (the square lies in one, two or four of them); over the
                              scenes of one overpass the mean EF / ETa and the largest clear share. EF / ef_scale,
                              ETa / eta_scale, clear share / clear_scale. The model drops an overpass whose clear
                              share is below ef.min_valid_fraction, as for field units (v0.8.0: the value of the one
                              30 m pixel holding the pixel centre, no clear share)
              pixels.npz      parcel (row of parcels.csv), row and col on the NDVI raster grid, x / y of the centre
              parcels.csv     the parcels (field_id as in the step-02 sample) with their first pixel and pixel count
              dates.csv, overpasses.csv, record.json (what the stacks were made from, sha256 of every file)
Model       the equations, settings and decisions of steps 03-05 for field units (core.season.unit_seasons,
            core.drivers.field_drivers, core.model.members), written for arrays of pixels: NDVI cleaning and
            despiking, daily linear interpolation and Savitzky-Golay smoothing, season dates at every amplitude
            fraction, daily fAPAR and Kc from the smoothed NDVI, Landsat 8 / 9 EF and ETa/ETc linear in time between
            the overpasses (a missing sensor takes the other), CASA Topt of the pixel's NDVI-peak month, season sums
            S = sum Ts Ws fAPAR Rs of every (Topt, Ws form, season fraction) of the LUE-ENS member grid. Yield is
            linear in S (fresh = HI eps_max f S x 0.01 / DM), so S of a pixel gives every member's yield; step 05b
            checks the vectorised model against the field-unit code on a sample of pixels.
Aggregation parcel = mean of its modelled pixels (all pixels have the same area); a parcel without a modelled pixel
            takes the crop-area-weighted mean of the modelled parcels of its cell, else of its tehsil; cell, tehsil
            and district = crop-area-weighted mean of the parcels (area_ha of the crop map), members first.
"""
import hashlib
import math
import time

import numpy as np
import pandas as pd

from . import config, lue
from . import stac_ingest as si
from .ensemble import member_grid

NODATA = -32768
REASONS = {1: "no Sentinel-2 observations", 2: "no Landsat EF observations"}   # codes 3-5: reason_text
LANDSAT_RULE = "pixel square, area-weighted, as E2 (v0.9.0)"     # recorded with the stacks; older stacks are refused


# ---------------------------------------------------------------- grid, pixel index, raster values at pixels
def s2_grid(cfg, P):
    """(grid key, extent, records) of the Sentinel-2 NDVI rasters of the run period (step 00c). Every date must be
    on the same 10 m grid and extent."""
    run, folder = cfg["run_id"], P["rasters"] / "ndvi"
    recs = [r for r in (si.read_json(p) for p in sorted(folder.glob(f"{run}_ndvi_*.json"))) if r
            and P["start"] <= pd.Timestamp(r["date"]) <= P["end"]]
    if not recs:
        raise SystemExit(f"no Sentinel-2 NDVI rasters of the run period in {folder} - run step 00c (export e1)")
    grids = {(tuple(r["made_from"]["grid"]), tuple(r["made_from"]["extent"])) for r in recs}
    if len(grids) != 1:
        raise SystemExit(f"the NDVI rasters of {run} are on {len(grids)} grids / extents - run step 00c (e1) again")
    (g, e), = grids
    key = (int(g[0]), float(g[1]), float(g[2]), float(g[3]))
    return key, tuple(int(v) for v in e), sorted(recs, key=lambda r: r["date"])


def parcels_fingerprint(parcels):
    """sha256 of the parcel ids and geometries (the stacks belong to one crop map)."""
    h = hashlib.sha256()
    f = parcels.sort_values("field_id")
    h.update(("|".join(f.field_id) + "|" + "|".join(g.wkb_hex for g in f.geometry)).encode())
    return h.hexdigest()


def pixel_index(parcels, key, extent):
    """Pixels of every parcel on the grid `key` inside `extent` (col0, row0, width, height of the NDVI rasters):
    centre inside the polygon (shapely.contains_xy); a parcel without one takes the pixel of its representative
    point. Returns (pixels: dict of arrays parcel, row, col, x, y; per parcel: first pixel, pixel count, rule)."""
    import shapely
    epsg, res, px, py = key
    gc, gr, width, height = extent
    geoms = parcels.to_crs(epsg).geometry.values
    par, rows, cols = [], [], []
    first, count, rule = np.zeros(len(geoms), np.int64), np.zeros(len(geoms), np.int64), []
    k = 0
    for i, g in enumerate(geoms):
        first[i] = k
        if g is None or g.is_empty:
            rule.append("no geometry")
            continue
        minx, miny, maxx, maxy = g.bounds
        c0, c1 = math.floor((minx - px) / res) - gc, math.ceil((maxx - px) / res) - gc
        r0, r1 = math.floor((py - maxy) / res) - gr, math.ceil((py - miny) / res) - gr
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        cc, rr = cc.ravel(), rr.ravel()
        shapely.prepare(g)
        inside = shapely.contains_xy(g, px + (gc + cc + 0.5) * res, py - (gr + rr + 0.5) * res)
        inside &= (cc >= 0) & (cc < width) & (rr >= 0) & (rr < height)
        if inside.any():
            rule.append("centre inside")
            cc, rr = cc[inside], rr[inside]
        else:
            rule.append("representative point")
            p = g.representative_point()
            cc = np.array([math.floor((p.x - px) / res) - gc])
            rr = np.array([math.floor((py - p.y) / res) - gr])
            if not (0 <= cc[0] < width and 0 <= rr[0] < height):
                rule[-1] = "outside the NDVI rasters"
                continue
        par.append(np.full(len(cc), i, np.int32))
        cols.append(cc.astype(np.int32))
        rows.append(rr.astype(np.int32))
        count[i] = len(cc)
        k += len(cc)
    cat = (lambda a, t: np.concatenate(a) if a else np.zeros(0, t))   # noqa: E731
    col, row = cat(cols, np.int32), cat(rows, np.int32)
    pix = {"parcel": cat(par, np.int32), "row": row, "col": col,
           "x": px + (gc + col.astype(np.float64) + 0.5) * res, "y": py - (gr + row.astype(np.float64) + 0.5) * res}
    return pix, first, count, rule


def raster_values(path, rows, cols, strip=1024):
    """Band 1 of a raster at pixel positions (rows, cols on the raster's own grid); positions outside the raster
    get its no-data value (NaN for float rasters without one). Read in strips of rows: memory does not depend on the
    raster size. Returns (values, no-data value)."""
    import rasterio
    from rasterio.windows import Window
    rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
    with rasterio.open(path) as ds:
        dtype = np.dtype(ds.dtypes[0])
        fill = ds.nodata if ds.nodata is not None else (np.nan if np.issubdtype(dtype, np.floating) else 0)
        out = np.full(len(rows), fill, dtype)
        ok = (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
        idx = np.flatnonzero(ok)
        if not idx.size:
            return out, fill
        o = idx[np.argsort(rows[idx], kind="stable")]
        rs = rows[o]
        for s0 in range(int(rs[0]) // strip * strip, int(rs[-1]) + 1, strip):
            a, b = np.searchsorted(rs, s0), np.searchsorted(rs, s0 + strip)
            if a == b:
                continue
            sel = o[a:b]
            c0, c1 = int(cols[sel].min()), int(cols[sel].max()) + 1
            h = min(strip, ds.height - s0)
            blk = ds.read(1, window=Window(c0, s0, c1 - c0, h))
            out[sel] = blk[rows[sel] - s0, cols[sel] - c0]
    return out, fill


def scene_positions(path, x, y, epsg):
    """(rows, cols) of the points x / y (grid CRS `epsg`) on the grid of a raster (north-up): the pixel that holds
    each point, pixel edges belonging to the pixel on their right / below."""
    import rasterio
    with rasterio.open(path) as ds:
        t, crs = ds.transform, ds.crs
    if crs is None or crs.to_epsg() is None:
        raise SystemExit(f"{path}: no EPSG code")
    if crs.to_epsg() != epsg:
        from pyproj import Transformer
        x, y = Transformer.from_crs(epsg, crs.to_epsg(), always_xy=True).transform(x, y)
    if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
        raise SystemExit(f"{path}: raster is not north-up")
    return (np.floor((t.f - np.asarray(y)) / -t.e).astype(np.int64),
            np.floor((np.asarray(x) - t.c) / t.a).astype(np.int64))


def encode(v, scale, what):
    """float -> int16 (value / scale, rounded; NaN -> NODATA). A value outside the int16 range stops the step."""
    q = np.rint(np.asarray(v, np.float64) / scale)
    bad = ~np.isnan(q) & ((q <= NODATA) | (q > 32767))
    if bad.any():
        raise SystemExit(f"{what}: {int(bad.sum())} values outside the stored range "
                         f"({(NODATA + 1) * scale:g} to {32767 * scale:g}; pixels.*_scale)")
    return np.where(np.isnan(q), NODATA, q).astype(np.int16)


def decode(q, scale):
    q = np.asarray(q)
    return np.where(q == NODATA, np.nan, q.astype(np.float64) * scale)


def landsat_overpasses(cfg, P):
    """Landsat scenes with result rasters of the run period, grouped by overpass (date, sensor): DataFrame date,
    sensor, item_ids (list), sorted by sensor and date."""
    ras = P["rasters"] / "landsat"
    p_items = P["folder"]["landsat"] / "landsat_items.csv"
    if not p_items.exists():
        raise SystemExit(f"no {p_items} - run step 00c (export e2)")
    items = pd.read_csv(p_items)
    items = items[items.status.astype(str).eq("read") & items.item_id.map(lambda i: (ras / f"{i}_clear.tif").exists())
                  & items.date.between(f"{P['start']:%Y-%m-%d}", f"{P['end']:%Y-%m-%d}")]
    if items.empty:
        raise SystemExit(f"no Landsat result rasters of the run period in {ras} - run step 00c (export e2)")
    g = items.groupby(["sensor", "date"]).item_id.apply(lambda s: sorted(s)).reset_index(name="item_ids")
    return g[["date", "sensor", "item_ids"]].sort_values(["sensor", "date"]).reset_index(drop=True)


def square_weights(key, lkey, x, y, chunk=200000):
    """Share of each Landsat pixel (grid lkey = epsg, res, px, py) covered by the pixel squares of the grid `key`
    centred at x / y (CRS of `key`) - stac_ingest.FieldGrid.weights for a field that is the square. Returns (global
    Landsat column [n, k], row [n, k], share [n, k]): the k pixels are the block that can touch a square, pixels it
    does not touch have share 0. Same CRS: the squares are parallel to the Landsat grid and the overlap is the product
    of the overlaps along x and y (exact). Another CRS (another UTM zone): the square corners are projected and the
    polygons intersected (shapely)."""
    es, res, _, _ = key
    el, rl, pxl, pyl = lkey
    x, y, h = np.asarray(x, np.float64), np.asarray(y, np.float64), res / 2
    if int(el) == int(es):
        nk = int(math.ceil(res / rl)) + 1                       # Landsat pixels a square can touch along x (or y)
        d = np.arange(nk)
        c = np.floor((x - h - pxl) / rl).astype(np.int64)[:, None] + d[None, :]
        r = np.floor((pyl - (y + h)) / rl).astype(np.int64)[:, None] + d[None, :]
        ox = np.clip(np.minimum(x[:, None] + h, pxl + (c + 1) * rl) - np.maximum(x[:, None] - h, pxl + c * rl), 0, None)
        oy = np.clip(np.minimum(y[:, None] + h, pyl - r * rl) - np.maximum(y[:, None] - h, pyl - (r + 1) * rl), 0, None)
        n = len(x)
        C = np.broadcast_to(c[:, None, :], (n, nk, nk)).reshape(n, -1)
        R = np.broadcast_to(r[:, :, None], (n, nk, nk)).reshape(n, -1)
        return C, R, (oy[:, :, None] * ox[:, None, :]).reshape(n, -1) / (rl * rl)
    import shapely
    from pyproj import Transformer
    tr = Transformer.from_crs(int(es), int(el), always_xy=True)
    reach = max(1, math.ceil(res / rl))
    dr, dc = np.meshgrid(np.arange(-reach, reach + 1), np.arange(-reach, reach + 1), indexing="ij")
    parts = []
    for a in range(0, len(x), chunk):
        xa, ya = x[a:a + chunk], y[a:a + chunk]
        cx = np.stack([xa - h, xa + h, xa + h, xa - h], 1)
        cy = np.stack([ya + h, ya + h, ya - h, ya - h], 1)
        cx, cy = tr.transform(cx, cy)
        sq = shapely.polygons(np.stack([cx, cy], 2))
        c0 = np.floor((cx.mean(1) - pxl) / rl).astype(np.int64)
        r0 = np.floor((pyl - cy.mean(1)) / rl).astype(np.int64)
        C, R = c0[:, None] + dc.ravel()[None], r0[:, None] + dr.ravel()[None]
        boxes = shapely.box(pxl + C * rl, pyl - (R + 1) * rl, pxl + (C + 1) * rl, pyl - R * rl)
        parts.append((C, R, shapely.area(shapely.intersection(boxes, sq[:, None])) / (rl * rl)))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def landsat_merged(ras, item_ids, x, y, key, cache=None):
    """EF, ETa and clear share of one overpass for the 10 m pixel squares centred at x / y (grid `key`), by the rules
    of the field export E2 (stac_ingest.landsat_rows, overpass_rows) for a unit that is the pixel square:
      per scene    clear share = clear area of the square / its area inside the scene (no row when the square is
                   outside the scene); EF and ETa = area-weighted mean of the clear Landsat pixels with a value
      per overpass mean EF and ETa of the scenes with a value; the largest clear share of the scenes with a row
    cache: dict kept by the caller for the same x / y (weights per Landsat grid). Returns ({ef, eta, clear}: arrays,
    NaN = no value; {ef, eta}: number of squares with a value)."""
    import rasterio
    n = len(x)
    s = {"ef": np.zeros(n), "eta": np.zeros(n)}
    cnt = {"ef": np.zeros(n), "eta": np.zeros(n)}
    share = np.full(n, np.nan)
    cache = {} if cache is None else cache
    for i in item_ids:
        p_cl = ras / f"{i}_clear.tif"
        with rasterio.open(p_cl) as dc:
            lkey = si.FieldGrid.key(dc)
            oc, orow = si.FieldGrid.origin(dc, lkey)
            grid = (dc.transform, dc.shape)
        if lkey not in cache:
            cache[lkey] = square_weights(key, lkey, x, y)
        C, R, w = cache[lkey]
        r, c = (R - orow).ravel(), (C - oc).ravel()
        q = raster_values(p_cl, r, c)[0].reshape(w.shape)           # 255: outside the scene (also outside the raster)
        wd = (w * (q != 255)).sum(1)
        row = wd > 0                                               # the square has a row in this scene
        clear = q == 1
        vf = np.where(row, (w * clear).sum(1) / np.where(row, wd, 1.0), np.nan)
        share = np.fmax(share, vf)
        for k in ("ef", "eta"):
            p = ras / f"{i}_{k}.tif"
            if not p.exists():
                continue
            with rasterio.open(p) as ds:
                if (ds.transform, ds.shape) != grid:
                    raise SystemExit(f"{p.name}: not on the grid of the clear mask of the scene")
            v = raster_values(p, r, c)[0].astype(np.float64).reshape(w.shape)
            ok = clear & ~np.isnan(v)
            sw = (w * ok).sum(1)
            has = row & (sw > 0)
            val = (w * np.where(ok, v, 0.0)).sum(1) / np.where(sw > 0, sw, 1.0)
            s[k][has] += val[has]
            cnt[k][has] += 1
    out = {k: np.where(cnt[k] > 0, s[k] / np.maximum(cnt[k], 1), np.nan) for k in ("ef", "eta")}
    out["clear"] = share
    return out, {k: int((cnt[k] > 0).sum()) for k in ("ef", "eta")}


# ---------------------------------------------------------------- export PX: pixel stacks (step 00c)
def stack_folder(P):
    return P["pixels"]


def build_stacks(cfg, P, parcels):
    """Pixel stacks of every parcel of the crop map from the stored rasters of step 00c (no download). Made again
    only when the rasters, the crop map or the storage settings changed. Returns (record, facts)."""
    ps = cfg["pixels"]
    out = stack_folder(P)
    out.mkdir(parents=True, exist_ok=True)
    key, extent, recs = s2_grid(cfg, P)
    ovp = landsat_overpasses(cfg, P)
    ras = P["rasters"] / "landsat"

    def sha_of(item, kind):
        j = si.read_json(ras / f"{item}_{'clear' if kind == 'clear' else 'ef'}.json") or {}
        return (j.get("files") or {}).get(kind, {}).get("sha256")
    made_from = {"grid": list(key), "extent": list(extent), "parcels": parcels_fingerprint(parcels),
                 "ndvi": {r["date"]: r["files"]["ndvi"]["sha256"] for r in recs},
                 "landsat": {i: [sha_of(i, k) for k in ("clear", "ef", "eta")] for ids in ovp.item_ids for i in ids},
                 "settings": {**{k: ps[k] for k in ("ndvi_scale", "ef_scale", "eta_scale", "clear_scale")},
                              "landsat_rule": LANDSAT_RULE}}
    names = ["ndvi.npy", "landsat_ef.npy", "landsat_eta.npy", "landsat_clear.npy", "pixels.npz", "parcels.csv",
             "dates.csv", "overpasses.csv"]
    old = si.read_json(out / "record.json")
    if old and old.get("made_from") == made_from and all((out / n).exists() for n in names):
        print(f"PX pixel stacks: unchanged since {old.get('made_at')} - kept ({out})", flush=True)
        return old, old["facts"]
    t0 = time.time()
    pix, first, count, rule = pixel_index(parcels, key, extent)
    n_px = len(pix["row"])
    print(f"PX pixel stacks: {n_px:,} pixels of {len(parcels):,} parcels; {len(recs)} Sentinel-2 dates, "
          f"{len(ovp)} Landsat overpasses", flush=True)

    def stack(name, n_rows, fill_row):
        tmp = out / (name + ".part")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int16, shape=(n_rows, n_px))
        facts = fill_row(mm)
        mm.flush()
        del mm
        tmp.replace(out / name)
        return facts

    def ndvi_rows(mm):
        have = 0
        for k, r in enumerate(recs):
            v, _ = raster_values(P["rasters"] / "ndvi" / r["files"]["ndvi"]["file"], pix["row"], pix["col"])
            mm[k] = encode(v, ps["ndvi_scale"], f"NDVI {r['date']}")
            have += int((~np.isnan(v)).sum())
            if (k + 1) % 10 == 0 or k + 1 == len(recs):
                print(f"  NDVI: {k + 1}/{len(recs)} dates, {(time.time() - t0) / 60:.1f} min", flush=True)
        return have

    weights = {}                                               # pixel-square weights per Landsat grid (all overpasses)

    def landsat_rows(mm):
        have = {"ef": 0, "eta": 0}
        for k, o in enumerate(ovp.itertuples()):
            v, n = landsat_merged(ras, o.item_ids, pix["x"], pix["y"], key, weights)
            mm["ef"][k] = encode(v["ef"], ps["ef_scale"], f"EF {o.date} {o.sensor}")
            mm["eta"][k] = encode(v["eta"], ps["eta_scale"], f"ETa {o.date} {o.sensor}")
            mm["clear"][k] = encode(v["clear"], ps["clear_scale"], f"Landsat clear share {o.date} {o.sensor}")
            have = {q: have[q] + n[q] for q in have}
            if (k + 1) % 10 == 0 or k + 1 == len(ovp):
                print(f"  Landsat: {k + 1}/{len(ovp)} overpasses, {(time.time() - t0) / 60:.1f} min", flush=True)
        return have

    have_nd = stack("ndvi.npy", len(recs), ndvi_rows)
    tmp = {q: out / f"landsat_{q}.npy.part" for q in ("ef", "eta", "clear")}
    mm = {q: np.lib.format.open_memmap(p, mode="w+", dtype=np.int16, shape=(len(ovp), n_px)) for q, p in tmp.items()}
    have_ls = landsat_rows(mm)
    for q in list(mm):
        mm[q].flush()
    del mm
    weights.clear()
    for q, p in tmp.items():
        p.replace(out / f"landsat_{q}.npy")
    def write_npz(tmp):
        with open(tmp, "wb") as f:
            np.savez(f, **pix)
    config.write_atomic(out / "pixels.npz", write_npz)
    pt = pd.DataFrame({"field_id": parcels.field_id, "parcel_id": parcels.parcel_id, "tehsil": parcels.tehsil,
                       "cell": parcels.cell, "area_ha": parcels.area_ha, "lon": parcels.lon, "lat": parcels.lat,
                       "first_pixel": first, "n_pixels": count, "pixel_rule": rule})
    config.to_csv_atomic(pt, out / "parcels.csv", index=False)
    config.to_csv_atomic(pd.DataFrame({"date": [r["date"] for r in recs],
                                       "n_scenes": [len(r["made_from"]["granules"]) for r in recs]}),
                         out / "dates.csv", index=False)
    config.to_csv_atomic(ovp.assign(item_ids=ovp.item_ids.map(";".join)), out / "overpasses.csv", index=False)
    facts = {"pixels": n_px, "parcels": len(parcels), "dates": len(recs), "overpasses": len(ovp),
             "pixel_rules": pd.Series(rule).value_counts().to_dict(), "pixel_dates_with_ndvi": have_nd,
             "pixel_overpasses_with_ef": have_ls["ef"], "pixel_overpasses_with_eta": have_ls["eta"],
             "minutes": round((time.time() - t0) / 60, 1)}
    rec = {"made_at": pd.Timestamp.now().isoformat(timespec="seconds"), "made_from": made_from, "facts": facts,
           "shapes": {"ndvi.npy": [len(recs), n_px], "landsat_ef.npy": [len(ovp), n_px],
                      "landsat_eta.npy": [len(ovp), n_px], "landsat_clear.npy": [len(ovp), n_px]},
           "scales": made_from["settings"], "nodata": NODATA, "grid": list(key), "extent": list(extent),
           "files": {n: si.file_record(out / n) for n in names}}
    si.write_json(out / "record.json", rec)
    return rec, facts


class Stacks:
    """The pixel stacks of a run (read-only memory maps)."""

    def __init__(self, P):
        self.folder = stack_folder(P)
        self.rec = si.read_json(self.folder / "record.json")
        if not self.rec:
            raise SystemExit(f"no pixel stacks in {self.folder} - run step 00c with export px "
                             "(python 00c_download.py --run <run> --only px)")
        f = self.folder
        z = np.load(f / "pixels.npz")
        self.pix = {k: z[k] for k in z.files}
        self.parcels = pd.read_csv(f / "parcels.csv")
        self.dates = pd.to_datetime(pd.read_csv(f / "dates.csv").date)
        ov = pd.read_csv(f / "overpasses.csv")
        ov["date"] = pd.to_datetime(ov.date)
        self.overpasses = ov
        self.nd = np.load(f / "ndvi.npy", mmap_mode="r")
        self.ef = np.load(f / "landsat_ef.npy", mmap_mode="r")
        self.eta = np.load(f / "landsat_eta.npy", mmap_mode="r")
        self.sc = self.rec["scales"]
        if self.sc.get("landsat_rule") != LANDSAT_RULE or not (f / "landsat_clear.npy").exists():
            raise SystemExit(f"the pixel stacks in {f} were made by an earlier version (Landsat value at the pixel "
                             "centre) - run step 00c again with export px (python 00c_download.py --run <run> --only px)")
        self.clear = np.load(f / "landsat_clear.npy", mmap_mode="r")
        self.n = len(self.pix["row"])

    def _get(self, sel):
        """(NDVI pixels x dates; EF, ETa and Landsat clear share pixels x overpasses) of the pixels sel, decoded."""
        return (decode(self.nd[:, sel], self.sc["ndvi_scale"]).T, decode(self.ef[:, sel], self.sc["ef_scale"]).T,
                decode(self.eta[:, sel], self.sc["eta_scale"]).T, decode(self.clear[:, sel], self.sc["clear_scale"]).T)

    def chunk(self, a, b):
        return self._get(slice(a, b))

    def take(self, idx):
        return self._get(np.asarray(idx))


# ---------------------------------------------------------------- vectorised helpers
def neighbours(u):
    """For a boolean array (rows x n): index of the previous True before each position (-1: none) and of the next
    True after it (n: none)."""
    n = u.shape[1]
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(u, idx, -1), axis=1)
    prev = np.full_like(last, -1)
    prev[:, 1:] = last[:, :-1]
    nxt_ = np.minimum.accumulate(np.where(u, idx, n)[:, ::-1], axis=1)[:, ::-1]
    nxt = np.full_like(nxt_, n)
    nxt[:, :-1] = nxt_[:, 1:]
    return prev, nxt


def interp_rows(y, x):
    """np.interp of every row of y (rows x n, NaN = no value) at its own positions x (n): linear between the
    values, constant beyond the first / last value, NaN for a row without a value."""
    n = y.shape[1]
    ok = ~np.isnan(y)
    idx = np.arange(n)
    lo = np.maximum.accumulate(np.where(ok, idx, -1), axis=1)
    hi = np.minimum.accumulate(np.where(ok, idx, n)[:, ::-1], axis=1)[:, ::-1]
    li, hj = np.clip(lo, 0, n - 1), np.clip(hi, 0, n - 1)
    ylo, yhi = np.take_along_axis(y, li, 1), np.take_along_axis(y, hj, 1)
    xlo, xhi = x[li], x[hj]
    with np.errstate(invalid="ignore", divide="ignore"):
        val = (yhi - ylo) / (xhi - xlo) * (x[None, :] - xlo) + ylo
    out = np.where(lo >= 0, np.where(hi < n, val, ylo), yhi)
    return np.where(ok, y, out)


def interp_at(xo, yo, x):
    """np.interp of every row of yo (rows x m values at the sorted positions xo, NaN = no value, skipped) at the
    positions x: linear between the values, constant beyond the first / last value; NaN for a row without a value.
    Returns rows x len(x)."""
    r, m = yo.shape
    if m == 0:
        return np.full((r, len(x)), np.nan)
    ok = ~np.isnan(yo)
    idx = np.arange(m)
    last = np.maximum.accumulate(np.where(ok, idx, -1), axis=1)
    firstv = np.minimum.accumulate(np.where(ok, idx, m)[:, ::-1], axis=1)[:, ::-1]
    jr = np.searchsorted(xo, x, side="right") - 1                 # last position <= x
    jl = np.searchsorted(xo, x, side="left")                      # first position >= x
    lo = np.where(jr[None, :] >= 0, last[:, np.clip(jr, 0, m - 1)], -1)
    hi = np.where(jl[None, :] < m, firstv[:, np.clip(jl, 0, m - 1)], m)
    li, hj = np.clip(lo, 0, m - 1), np.clip(hi, 0, m - 1)
    ylo, yhi = np.take_along_axis(yo, li, 1), np.take_along_axis(yo, hj, 1)
    xlo, xhi = xo[li], xo[hj]
    with np.errstate(invalid="ignore", divide="ignore"):
        val = (yhi - ylo) / (xhi - xlo) * (x[None, :] - xlo) + ylo
    out = np.where(lo >= 0, np.where(hi < m, np.where(lo == hi, ylo, val), ylo), yhi)
    return out


def reason_text(code, n_used, amp, cl):
    """Exclusion reason of a pixel, worded as core.season.unit_seasons words it for a field unit."""
    if code in REASONS:
        return REASONS[code]
    if code == 3:
        return f"{int(n_used)} clear observations"
    if code == 4:
        return f"{int(n_used)} clear observations (< {cl['min_used_obs']})"
    if code == 5:
        return f"NDVI amplitude {amp:.2f} (< {cl['min_amplitude']}): no clear crop cycle"
    return ""


def reason_label(code, cl):
    """Short label of an exclusion code (for counts)."""
    return {1: REASONS[1], 2: REASONS[2], 3: "fewer than 2 clear observations",
            4: f"fewer than {cl['min_used_obs']} clear observations",
            5: f"NDVI amplitude below {cl['min_amplitude']} (no clear crop cycle)"}.get(int(code), "")


def member_stats_rows(y, w, quantiles):
    """core.ensemble.member_stats for every row of y (rows x members): weighted mean, SD and quantiles (inverse of
    the weighted empirical CDF, as core.stats.wquantile). Rows with NaN give NaN."""
    w = np.asarray(w, float)
    w = w / w.sum()
    m = y @ w
    out = {"mean": m, "sd": np.sqrt(((y - m[:, None]) ** 2) @ w)}
    o = np.argsort(y, axis=1)
    ys = np.take_along_axis(y, o, 1)
    cw = np.cumsum(w[o], axis=1) / w.sum()
    for q in quantiles:
        k = np.argmax(cw >= q - 1e-12, axis=1)
        out[f"p{int(round(100 * q))}"] = ys[np.arange(len(y)), k]
    bad = np.isnan(y).any(axis=1)
    return {k: np.where(bad, np.nan, v) for k, v in out.items()}


# ---------------------------------------------------------------- the model for an array of pixels
def combos_of(cfg):
    """Member grid, the (Topt, Ws form, season fraction) combinations it needs and the central one, and the index
    of every member's combination. Season sums S are computed once per combination."""
    from . import model
    grid = member_grid(cfg)
    c = model.central_config(cfg)
    keys = list(dict.fromkeys([(float(r.topt), r.ws_form, round(float(r.season_p), 6)) for r in grid.itertuples()]
                              + [(float(c["topt"]), c["ws_form"], round(float(c["season_p"]), 6))]))
    k_of = {k: i for i, k in enumerate(keys)}
    km = np.array([k_of[(float(r.topt), r.ws_form, round(float(r.season_p), 6))] for r in grid.itertuples()])
    kc = k_of[(float(c["topt"]), c["ws_form"], round(float(c["season_p"]), 6))]
    combos = pd.DataFrame(keys, columns=["topt", "ws_form", "season_p"])
    return grid, combos, km, kc, c


def combo_index(grid, combos):
    """Row of combos (combos.csv: topt, ws_form, season_p) of every member of grid (member_grid.csv)."""
    out = []
    for r in grid.itertuples():
        hit = combos.index[np.isclose(combos.topt, r.topt) & (combos.ws_form == r.ws_form)
                           & np.isclose(combos.season_p, r.season_p)]
        if len(hit) != 1:
            raise SystemExit(f"member {r.member}: no single season-sum combination in combos.csv")
        out.append(int(hit[0]))
    return np.array(out)


def yields_from_s(s, grid, km):
    """(biomass, dry, fresh) of every member from season sums s (rows x combinations), as core.model.unit_yield:
    B = eps_max f S; dry = HI B 0.01; fresh = dry / DM."""
    S = s[:, km]
    b = grid.eps_max.to_numpy(float)[None, :] * grid.f_par.to_numpy(float)[None, :] * S
    dry, fresh = lue.crop_yield(b, grid.hi.to_numpy(float)[None, :], grid.dm.to_numpy(float)[None, :])
    return b, dry, fresh


def central_yield(s, c, kc):
    b = c["eps_max"] * c["f_par"] * s[:, kc]
    return lue.crop_yield(b, c["hi"], c["dm"])[1]


class Weather:
    """Daily weather of the model days per E3 cell (arrays cells x days) and the CASA Topt of every cell and month."""

    def __init__(self, cfg, e3):
        days = pd.date_range(*cfg["season"]["window"], freq="D")
        self.days, self.cells = days, sorted(e3.cell.unique())
        w = e3[e3.date.isin(days)].sort_values(["cell", "date"])
        n = w.groupby("cell").size().reindex(self.cells).fillna(0)
        if (n != len(days)).any():
            bad = n[n != len(days)]
            raise SystemExit(f"E3: {len(bad)} cells without every model day, e.g. {bad.index[0]} "
                             f"({int(bad.iloc[0])} of {len(days)} days)")
        a = {k: w[k].to_numpy(float).reshape(len(self.cells), len(days))
             for k in ("rs_mj_m2", "tmin_c", "tmax_c", "eto_mm")}
        self.rs, self.tmax, self.eto = a["rs_mj_m2"], a["tmax_c"], a["eto_mm"]
        self.tmean = (a["tmin_c"] + a["tmax_c"]) / 2
        per = days.to_period("M")
        self.months = sorted(per.unique())
        self.month_of_day = np.array([self.months.index(m) for m in per])
        self.topt = np.stack([self.tmean[:, self.month_of_day == j].mean(axis=1) for j in range(len(self.months))],
                             axis=1)
        self.index = {c: i for i, c in enumerate(self.cells)}


def seasons_vec(V, obs_off, n_ef, cfg, fractions, n_days):
    """Season of every pixel (core.season.unit_seasons for arrays). V: pixels x dates NDVI (NaN = no value) at the
    day offsets obs_off from the series start (window start - lead_days); n_ef: Landsat EF values per pixel.
    Returns code (0 = usable, 1-5 = excluded: reason_text), n_used, amplitude, smoothed daily NDVI (pixels x days,
    NaN for excluded) and per fraction the season arrays (day indices of the series)."""
    from scipy.signal import savgol_filter
    sc, cl = cfg["season"], cfg["ndvi_cleaning"]
    start = pd.Timestamp(sc["window"][0]) - pd.Timedelta(days=sc["lead_days"])
    day0 = lambda d: int((pd.Timestamp(d) - start).days)       # noqa: E731
    i_w0, i_w1 = day0(sc["window"][0]), day0(sc["window"][1])
    i_p0, i_p1 = day0(sc["peak_search"][0]), day0(sc["peak_search"][1])
    q0, q1 = day0(sc["quality_window"][0]), day0(sc["quality_window"][1])
    if i_w1 != n_days - 1 or not (i_w0 <= i_p0 < i_p1 <= i_w1):
        raise SystemExit("season dates: the peak search must lie inside the season window")
    n, D = V.shape
    raw = ~np.isnan(V)
    code = np.zeros(n, np.int8)
    code[raw.sum(1) == 0] = 1
    code[(code == 0) & (np.asarray(n_ef) == 0)] = 2
    used = raw & (1.0 >= cl["min_valid_fraction"])             # clear share of a pixel-date: 1 with NDVI, 0 without
    desp = np.zeros_like(used)
    act = np.flatnonzero(code == 0)
    day = np.asarray(obs_off, np.int64)
    while act.size:                                            # drop the worst negative spike, one at a time
        u, v = used[act], V[act]
        prev, nxt = neighbours(u)
        has = u & (prev >= 0) & (nxt < D)
        pi, ni = np.clip(prev, 0, D - 1), np.clip(nxt, 0, D - 1)
        vp, vn = np.take_along_axis(v, pi, 1), np.take_along_axis(v, ni, 1)
        with np.errstate(invalid="ignore"):
            gap = (vp + vn) / 2 - v
            spike = has & (v < np.minimum(vp, vn)) & (gap > cl["despike_drop"]) & \
                ((day[ni] - day[pi]) <= cl["despike_max_span_days"])
        hit = spike.any(1)
        if not hit.any():
            break
        k = np.argmax(np.where(spike, gap, -np.inf), axis=1)
        rows_, ks = act[hit], k[hit]
        used[rows_, ks] = False
        desp[rows_, ks] = True
        act = rows_
    n_used = used.sum(1)
    code[(code == 0) & (n_used < 2)] = 3
    ok = np.flatnonzero(code == 0)
    smooth = np.full((n, n_days), np.nan)
    amp = np.full(n, np.nan)
    if ok.size:
        y = np.full((ok.size, n_days), np.nan)
        y[:, day] = np.where(used[ok], V[ok], np.nan)
        lin = interp_rows(y, np.arange(n_days, dtype=float))
        sm = np.clip(savgol_filter(lin, cl["savgol_window_days"], cl["savgol_order"], mode="interp", axis=1), -0.1, 1.0)
        smooth[ok] = sm
        amp[ok] = sm[:, i_p0:i_p1 + 1].max(axis=1) - sm.min(axis=1)
    code[(code == 0) & (n_used < cl["min_used_obs"])] = 4
    code[(code == 0) & (amp < cl["min_amplitude"])] = 5
    ok = code == 0
    cols = np.arange(n_days)[None, :]
    seas = {}
    s = smooth
    with np.errstate(invalid="ignore"):
        pk = s[:, i_p0:i_p1 + 1]
        peak_i = i_p0 + np.argmax(np.where(ok[:, None], pk, -np.inf), axis=1)
        peak = np.take_along_axis(s, peak_i[:, None], 1)[:, 0]
        pre = (cols >= i_w0) & (cols <= peak_i[:, None])
        bs_i = np.argmin(np.where(pre & ok[:, None], s, np.inf), axis=1)
        post = cols >= peak_i[:, None]
        be_i = np.argmin(np.where(post & ok[:, None], s, np.inf), axis=1)
        bs, be = (np.take_along_axis(s, bs_i[:, None], 1)[:, 0], np.take_along_axis(s, be_i[:, None], 1)[:, 0])
        for p in fractions:
            thr_s, thr_e = bs + p * (peak - bs), be + p * (peak - be)
            rise = (cols >= bs_i[:, None]) & (cols <= peak_i[:, None]) & (s >= thr_s[:, None])
            em_i = np.argmax(rise, axis=1)
            below = post & (s <= thr_e[:, None])
            found = (peak - be >= cl["min_amplitude"]) & below.any(1)
            end_i = np.where(found, np.argmax(below, axis=1), i_w1)
            seas[round(float(p), 6)] = {
                "emergence": em_i, "season_end": end_i, "season_days": end_i - em_i + 1, "peak_i": peak_i,
                "peak_ndvi": peak, "base_start_i": bs_i, "base_start_ndvi": bs, "base_end_i": be_i, "base_end_ndvi": be,
                "amplitude": peak - np.maximum(bs, be), "threshold_start": thr_s, "threshold_end": thr_e,
                "start_ok": (bs_i > 0) & (em_i > 0), "peak_ok": (peak_i > i_p0) & (peak_i < i_p1), "end_ok": found}
    obs_q = (day >= q0) & (day <= q1)
    quality = {"clear_obs_quality_window": (used & obs_q[None, :]).sum(1), "despiked_dates": desp.sum(1),
               "low_valid_dates": (~raw).sum(1)}
    return code, n_used, amp, smooth, seas, quality, i_w0


def model_vec(V, EF, ETA, obs_off, ovp_off, ovp_sensor, cell_idx, W, cfg, combos, fractions, CLR=None):
    """The LUE model of steps 03-05 for pixels. V: pixels x dates NDVI; EF, ETA: pixels x overpasses (day offsets
    ovp_off from the series start, sensors ovp_sensor L8 / L9); CLR: Landsat clear share of the pixel square per
    overpass (None: 1 where EF or ETa has a value); cell_idx: row of every pixel in W (Weather).
    Returns code, n_used, amp, s (pixels x combinations; NaN for excluded pixels), central season, quality, n_ef."""
    sc = cfg["season"]
    n = V.shape[0]
    n_days = (pd.Timestamp(sc["window"][1]) - pd.Timestamp(sc["window"][0])).days + 1 + sc["lead_days"]
    if CLR is None:
        CLR = np.where(np.isnan(EF) & np.isnan(ETA), np.nan, 1.0)
    # field units (core.drivers.field_tables): an overpass with a clear share below ef.min_valid_fraction is missing
    low = np.nan_to_num(CLR, nan=0.0) < cfg["ef"]["min_valid_fraction"]
    EF = np.where(low, np.nan, EF)
    ETA = np.where(low, np.nan, ETA)
    n_ef = (~np.isnan(EF)).sum(1)
    code, n_used, amp, smooth, seas, quality, i_w0 = seasons_vec(V, obs_off, n_ef, cfg, fractions, n_days)
    s = np.full((n, len(combos)), np.nan)
    ok = np.flatnonzero(code == 0)
    cen = seas[round(float(sc["amplitude_fraction"]), 6)]
    if not ok.size:
        return code, n_used, amp, s, cen, quality, n_ef
    m_days = n_days - i_w0
    ci = cell_idx[ok]
    rs, tmean, eto = W.rs[ci], W.tmean[ci], W.eto[ci]
    ndvi = smooth[ok][:, i_w0:]
    rel = cfg["fapar"][cfg["fapar"]["relation_field_units"]]
    fapar = lue.fapar_from_ndvi(ndvi, rel["slope"], rel["intercept"])
    kc = lue.crop_coefficient(ndvi, cfg["crop_coefficient"])
    etc = kc * eto
    xw = np.arange(i_w0, n_days, dtype=float)                  # model days as day offsets of the series
    ovp_off, ovp_sensor = np.asarray(ovp_off, float), np.asarray(ovp_sensor)
    ef_o, eta_o = EF[ok], ETA[ok]
    avail = {q: (~np.isnan(ef_o[:, ovp_sensor == q])).any(1) for q in ("L8", "L9")}
    daily = {}
    for q in ("L8", "L9"):
        other = "L9" if q == "L8" else "L8"
        for src in (q, other):                                  # one sensor missing: use the other
            sel = ovp_sensor == src
            xo = ovp_off[sel]
            o = np.argsort(xo, kind="stable")
            xo, e, t = xo[o], ef_o[:, sel][:, o], eta_o[:, sel][:, o]
            m_o = (xo - i_w0).astype(int)
            inw = (m_o >= 0) & (m_o < m_days)
            etc_o = np.full(e.shape, np.nan)
            etc_o[:, inw] = etc[:, m_o[inw]]
            with np.errstate(invalid="ignore", divide="ignore"):
                ret_o = t / etc_o                               # overpass ETa vs ETc of that day
            daily[(q, src)] = (np.clip(interp_at(xo, e, xw), 0, 1), np.clip(interp_at(xo, ret_o, xw), 0, 1.5))
    ef_d, ret_d = {}, {}
    for q in ("L8", "L9"):
        other = "L9" if q == "L8" else "L8"
        use_q = avail[q][:, None]
        ef_d[q] = np.where(use_q, daily[(q, q)][0], daily[(q, other)][0])
        ret_d[q] = np.where(use_q, daily[(q, q)][1], daily[(q, other)][1])
    if cfg["temperature_scalar"]["topt"] == "casa_ndvi_peak_month":   # Topt of the pixel: its NDVI-peak month
        topt_px = W.topt[ci, W.month_of_day[cen["peak_i"][ok] - i_w0]]
    else:
        topt_px = None
    cols = np.arange(i_w0, n_days)[None, :]
    alpha = 0.5
    for k, cb in enumerate(combos.itertuples()):
        ss = seas[round(float(cb.season_p), 6)]
        mask = (cols >= ss["emergence"][ok][:, None]) & (cols <= ss["season_end"][ok][:, None])
        topt = topt_px[:, None] + cb.topt if topt_px is not None else cb.topt
        if cb.ws_form in ("ef", "casa_ef"):
            x = alpha * ef_d["L8"] + (1 - alpha) * ef_d["L9"]
        elif cb.ws_form == "relative_et":
            x = alpha * ret_d["L8"] + (1 - alpha) * ret_d["L9"]
        elif cb.ws_form == "none":
            x = np.ones_like(fapar)
        else:
            raise ValueError(cb.ws_form)
        v = lue.ts(tmean, topt) * lue.ws(x, cb.ws_form) * fapar * rs
        s[ok, k] = np.nansum(np.where(mask, v, 0.0), axis=1)
    return code, n_used, amp, s, cen, quality, n_ef


# ---------------------------------------------------------------- the same pixels through the field-unit code
def unit_check(cfg, idx, V, EF, ETA, dates, ovp, cells, e3, CLR=None):
    """Season sums and exclusion of pixels computed by the field-unit code of steps 03-05 (core.drivers.field_tables,
    core.season.unit_seasons, core.drivers.field_drivers, core.model.members) - each pixel a unit with clear share 1
    on its Sentinel-2 dates with a value, 0 on the others, and the Landsat clear share CLR of its square per overpass
    (None: 1 where EF or ETa has a value). Returns a table: pixel, reason, emergence, season_end, member fresh yields
    (unit, member, fresh_t_ha)."""
    from . import drivers as dv
    from . import model
    from . import season as ss
    sc, cl = cfg["season"], cfg["ndvi_cleaning"]
    names = [f"PX{i}" for i in idx]
    e1 = pd.DataFrame({"field_id": np.repeat(names, len(dates)), "date": np.tile(pd.DatetimeIndex(dates), len(idx)),
                       "ndvi": V.ravel(), "valid_fraction": (~np.isnan(V)).astype(float).ravel(), "n_scenes": 1})
    if CLR is None:
        CLR = (~np.isnan(EF) | ~np.isnan(ETA)).astype(float)
    e2 = pd.DataFrame({"field_id": np.repeat(names, len(ovp)), "date": np.tile(pd.DatetimeIndex(ovp.date), len(idx)),
                       "sensor": np.tile(ovp.sensor.to_numpy(), len(idx)), "ef": EF.ravel(), "eta_mm_day": ETA.ravel(),
                       "valid_fraction": np.asarray(CLR, float).ravel()})
    units = pd.DataFrame({"unit": names, "tehsil": "check", "cell": cells, "area_ha": 0.01, "lon": 0.0, "lat": 0.0})
    start = pd.Timestamp(sc["window"][0]) - pd.Timedelta(days=sc["lead_days"])
    fractions = sorted(set(sc["ensemble_fractions"]) | {sc["amplitude_fraction"]})
    obs, thermal, weather, _ = dv.field_tables(cfg, units, e1, e2, e3)
    nd = obs[(obs.date >= start) & (obs.date <= pd.Timestamp(sc["window"][1]))]
    has_th = thermal[thermal.ef.notna()].groupby("unit").size()
    _, daily, seasons, _, excl = ss.unit_seasons(nd, units.unit, has_th, sc, cl, fractions, start)
    reasons = {e["unit"]: e["reason"] for e in excl}
    mem = pd.DataFrame(columns=["unit", "member", "fresh_t_ha"])
    seas = pd.DataFrame(seasons)
    cen = pd.DataFrame(columns=["emergence", "season_end"])
    if seasons:
        seas["amplitude_fraction"] = seas.amplitude_fraction.round(2)
        cen = seas[np.isclose(seas.amplitude_fraction, sc["amplitude_fraction"])].set_index("unit")
        valid = units[units.unit.isin(cen.index)]
        d, _, _ = dv.field_drivers(cfg, valid, weather, thermal, pd.concat(daily), cen)
        mem, _ = model.members(d, seas, cfg)
    t = pd.DataFrame({"pixel": idx, "unit": names, "reason": [reasons.get(u, "") for u in names],
                      "emergence": [cen.emergence.get(u) if u in cen.index else pd.NaT for u in names],
                      "season_end": [cen.season_end.get(u) if u in cen.index else pd.NaT for u in names]})
    return t, mem[["unit", "member", "fresh_t_ha"]]


# ---------------------------------------------------------------- aggregation
def weather_cells(parcel_cells, e3_cells):
    """Weather cell of every parcel cell: itself when E3 has it, else the nearest cell of E3."""
    have = sorted(set(e3_cells))
    xy = np.array([[float(v) for v in c.split("_")] for c in have])
    out = {}
    for c in set(parcel_cells):
        if c in have:
            out[c] = c
        else:
            p = np.array([float(v) for v in c.split("_")])
            out[c] = have[int(np.argmin(((xy - p) ** 2).sum(axis=1)))]
    return out


def parcel_means(s, ok, parcel, n_parcels):
    """Mean of s (pixels x combinations) over the modelled pixels (ok) of every parcel; NaN for a parcel without
    one. Returns (means, modelled pixel count)."""
    cnt = np.bincount(parcel[ok], minlength=n_parcels).astype(float)
    out = np.full((n_parcels, s.shape[1]), np.nan)
    for k in range(s.shape[1]):
        tot = np.bincount(parcel[ok], weights=s[ok, k], minlength=n_parcels)
        out[:, k] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return out, cnt.astype(int)


def area_mean_by(values, area, groups):
    """Crop-area-weighted mean of the rows of values (rows x k, NaN rows left out) per group: DataFrame group x k."""
    v = np.asarray(values, float)
    ok = ~np.isnan(v).any(axis=1)
    a = np.asarray(area, float)[ok]
    df = pd.DataFrame(v[ok] * a[:, None]).assign(_a=a, _g=np.asarray(groups)[ok])
    g = df.groupby("_g").sum()
    return g.drop(columns="_a").div(g._a, axis=0), g._a


def fill_parcels(sp, area, cell, tehsil):
    """Parcels without a modelled pixel take the crop-area-weighted mean of the modelled parcels of their cell, else
    of their tehsil. Returns (filled values, status)."""
    out = sp.copy()
    status = np.where(~np.isnan(sp).any(axis=1), "modelled", "").astype(object)   # object: labels of any length
    for level, g in (("cell", cell), ("tehsil", tehsil)):
        means, _ = area_mean_by(np.where((status == "modelled")[:, None], sp, np.nan), area, g)
        need = status == ""
        have = need & pd.Series(g).isin(means.index).to_numpy()
        if have.any():
            out[have] = means.reindex(np.asarray(g)[have]).to_numpy()
            status[have] = f"{level} estimate"
    status[status == ""] = "no estimate"
    return out, status


def level_members(sp, area, groups, grid, km, level):
    """Member table of a level (06 format): level, member, biomass, dry, fresh + member grid columns; the members
    are aggregated first (crop-area-weighted mean of the parcels' season sums, then the member equations)."""
    means, a = area_mean_by(sp, area, groups)
    b, dry, fresh = yields_from_s(means.to_numpy(), grid, km)
    rows = pd.DataFrame({level: np.repeat(means.index.to_numpy(), len(grid)),
                         "member": np.tile(grid.member.to_numpy(), len(means)), "biomass_g_m2": b.ravel(),
                         "dry_t_ha": dry.ravel(), "fresh_t_ha": fresh.ravel()})
    return rows.merge(grid, on="member"), a
