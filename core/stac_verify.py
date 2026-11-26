"""Verification of the downloads of step 00c (used by 00d_verify.py).

Three kinds of checks, all on the real files and sources:
  completeness  what the STAC catalogues list now for the units and the period, against what was downloaded;
                every hour of every weather day; every unit and date in the export tables
  integrity     the stored GeoTIFFs are present, unchanged since they were written (sha256 of the file tables and
                records of step 00c) and open; a sample of stored pixels equals the source pixel by pixel
  sample values a random sample of export values recomputed from the sources with methods other than those of
                core/stac_ingest.py: pixel shares from a polygon rasterised on a finer grid (instead of the
                polygon overlay), bands read in one window, radiation from accumulated values (instead of hourly
                differences)
Nothing is corrected or filled here: a difference is reported.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import config
from core import stac_ingest as si

SUBPIXEL_BUDGET = 1.2e8        # largest fine grid of the pixel-share check (memory)


class Report:
    def __init__(self):
        self.rows = []

    def add(self, dataset, check, status, detail):
        self.rows.append({"dataset": dataset, "check": check, "status": status, "detail": detail})
        print(f"  [{status}] {dataset}: {check} - {str(detail)[:260]}", flush=True)

    def ok(self, dataset, check, passed, detail, warn=False):
        self.add(dataset, check, "PASS" if passed else ("WARN" if warn else "FAIL"), detail)

    def table(self):
        return pd.DataFrame(self.rows, columns=["dataset", "check", "status", "detail"])


def short(ids, n=5):
    ids = sorted(ids)
    return ", ".join(ids[:n]) + (f" ... ({len(ids)} in total)" if len(ids) > n else "")


# ---------------------------------------------------------------- pixel shares by a second method
def window_of(geom, ds):
    """Pixel window (col0, row0, width, height) of the raster grid of ds that holds the polygon; may reach outside."""
    t = ds.transform
    minx, miny, maxx, maxy = geom.bounds
    c0, c1 = int(np.floor((minx - t.c) / t.a)), int(np.ceil((maxx - t.c) / t.a))
    r0, r1 = int(np.floor((t.f - maxy) / t.a)), int(np.ceil((t.f - miny) / t.a))
    return c0, r0, c1 - c0, r1 - r0


def pixel_shares(geom, ds, win):
    """Share of every pixel of the window inside the polygon: the polygon rasterised (pixel-centre rule) on a grid
    k times finer, averaged per pixel. Returns (shares[rows, cols], k)."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    c0, r0, wd, h = win
    t = ds.transform
    k = int(max(2, min(50, np.floor(np.sqrt(SUBPIXEL_BUDGET / (wd * h))))))
    fine = rasterize([(geom, 1)], out_shape=(h * k, wd * k), fill=0, dtype="uint8",
                     transform=from_origin(t.c + c0 * t.a, t.f - r0 * t.a, t.a / k, t.a / k))
    return fine.reshape(h, k, wd, k).mean(axis=(1, 3)), k


def read_into(ds, win, fill, **kw):
    """Band 1 in the window, pixels outside the raster = fill. kw: out_shape / resampling for a coarser band."""
    from rasterio.windows import Window
    c0, r0, wd, h = win
    return ds.read(1, window=Window(c0, r0, wd, h), boundless=True, fill_value=fill, **kw)


# ---------------------------------------------------------------- Sentinel-2
def catalogue_now(cfg, fields, start, end, source):
    """Item ids that the STAC API lists now for the units and the period (same search as step 00c)."""
    c = cfg["stac"][source]
    bbox = [float(x) for x in fields.total_bounds]
    query = None
    if source == "landsat":
        query = {"platform": {"in": list(c["platforms"])}}
        if c.get("collection_category"):
            query["landsat:collection_category"] = {"eq": c["collection_category"]}
    items = si.search(c["api"], c["collection"], bbox, start, end, query=query, tries=cfg["stac"]["retries"])
    return {i.id: i for i in items if si.fields_in_item(i, fields)}


def stored_files(index):
    """The GeoTIFFs of a file table of step 00c (file relative to the table, bytes, sha256) against the files now:
    (listed, missing, changed, unreadable, bytes). changed = another sha256 than when the file was written;
    unreadable = the file does not open as a raster."""
    import rasterio
    t = pd.read_csv(index)
    missing, changed, unreadable, size = [], [], [], 0
    for n, r in enumerate(t.itertuples(), 1):
        p = index.parent / r.file
        if not p.exists():
            missing.append(r.file)
            continue
        size += p.stat().st_size
        if config.sha256(p) != r.sha256:
            changed.append(r.file)
        else:
            try:
                with rasterio.open(p) as ds:
                    ds.read(1, window=((0, min(8, ds.height)), (0, min(8, ds.width))))
            except Exception:  # noqa: BLE001 - a file that does not open is a finding
                unreadable.append(r.file)
        if n % 500 == 0:
            print(f"    {index.name}: {n}/{len(t)} files checked", flush=True)
    return len(t), missing, changed, unreadable, size


def stored_pixels_s2(cfg, store, items, n, seed):
    """n granules: the stored digital numbers of red, NIR and scene classification in the pixel window of one unit
    against the source, pixel by pixel. One row per granule and band."""
    import pystac_client
    import rasterio
    from rasterio.windows import Window
    c2 = cfg["stac"]["sentinel2"]
    a, sign = c2["assets"], si.signer(c2.get("sign"))
    col = pystac_client.Client.open(c2["api"]).get_collection(c2["collection"])
    recs = [r for r in (si.read_json(store / f"{i}.json") for i in items.item_id) if r and r.get("files")]
    rs, rows = np.random.RandomState(seed), []
    for j in rs.choice(len(recs), min(n, len(recs)), replace=False):
        rec = recs[j]
        fid = sorted(rec["units"])[rs.randint(len(rec["units"]))]
        it = si.retry(lambda: list(col.get_items(rec["item_id"])), cfg["stac"]["retries"], "granule")[0]
        with rasterio.Env(**si.GDAL_ENV):
            for name, k in (("red", 1), ("nir", 1), ("scl", rec["scl_factor"])):
                w = si.coarse(rec["units"][fid], k)
                u = [v // k for v in rec["window"]]
                with rasterio.open(sign(it.assets[a[name]].href)) as src, \
                        rasterio.open(store / rec["files"][name]["file"]) as loc:
                    x = si.retry(lambda: src.read(1, window=Window(*w)), cfg["stac"]["retries"], it.id)
                    y = loc.read(1, window=Window(w[0] - u[0], w[1] - u[1], w[2], w[3]))
                rows.append({"item_id": rec["item_id"], "field_id": fid, "band": a[name], "pixels": int(x.size),
                             "pixels_with_data": int((x != 0).sum()), "pixels_different": int((x != y).sum())})
        print(f"    stored Sentinel-2 pixels {rec['item_id']} {fid}: "
              f"{sum(r['pixels_different'] for r in rows[-3:])} of {sum(r['pixels'] for r in rows[-3:])} differ", flush=True)
    return pd.DataFrame(rows)


def stored_pixels_landsat(cfg, store, items, n, seed):
    """n stored Landsat bands (any band of a scene): a 512 x 512 pixel part of the scene window against the source,
    pixel by pixel. One row per band."""
    import pystac_client
    import rasterio
    from rasterio.windows import Window
    cl = cfg["stac"]["landsat"]
    sign = si.signer(cl.get("sign"))
    col = pystac_client.Client.open(cl["api"]).get_collection(cl["collection"])
    pairs = [(r, b) for r in (si.read_json(store / f"{i}.json") for i in items.item_id) if r and r.get("files")
             for b in sorted(r["files"])]
    rs, rows = np.random.RandomState(seed), []
    for j in rs.choice(len(pairs), min(n, len(pairs)), replace=False):
        rec, band = pairs[j]
        c0, r0, w, h = rec["window"]
        pw, ph = min(512, w), min(512, h)
        pc, pr = int(rs.randint(w - pw + 1)), int(rs.randint(h - ph + 1))
        it = si.retry(lambda: list(col.get_items(rec["item_id"])), cfg["stac"]["retries"], "scene")[0]
        with rasterio.Env(**si.GDAL_ENV), rasterio.open(sign(it.assets[band].href)) as src, \
                rasterio.open(store / rec["files"][band]["file"]) as loc:
            x = si.retry(lambda: src.read(1, window=Window(c0 + pc, r0 + pr, pw, ph)), cfg["stac"]["retries"], it.id)
            y = loc.read(1, window=Window(pc, pr, pw, ph))
        rows.append({"item_id": rec["item_id"], "band": band, "pixels": int(x.size),
                     "pixels_different": int((x != y).sum())})
        print(f"    stored Landsat pixels {rec['item_id']} {band}: {rows[-1]['pixels_different']} of {x.size} differ",
              flush=True)
    return pd.DataFrame(rows)


def pick(table, n, seed, clear_col="valid_fraction", lo=0.2):
    """n rows: half mostly clear (> 0.9), half partly clear (lo to 0.9) when there are any."""
    a = table[table[clear_col] > 0.9]
    b = table[(table[clear_col] > lo) & (table[clear_col] <= 0.9)]
    nb = min(n // 2, len(b))
    return pd.concat([a.sample(min(n - nb, len(a)), random_state=seed), b.sample(nb, random_state=seed)])


def ndvi_sample(cfg, fields, e1, items, n, seed):
    """Unit-dates of E1 recomputed from the Sentinel-2 assets (see module text). One row per unit-date."""
    import pystac_client
    import rasterio
    from rasterio.enums import Resampling
    c2 = cfg["stac"]["sentinel2"]
    a, sign = c2["assets"], si.signer(c2.get("sign"))
    col = pystac_client.Client.open(c2["api"]).get_collection(c2["collection"])
    rows = []
    for r in pick(e1[e1.ndvi.notna()], n, seed).itertuples():
        ids = items[items.date == r.date].item_id.tolist()
        g = fields[fields.field_id == r.field_id]
        nds, valids, w, win, k, lattice = [], [], None, None, None, None
        with rasterio.Env(**si.GDAL_ENV):
            for it in si.retry(lambda: list(col.get_items(*ids)), cfg["stac"]["retries"], "items of the date"):
                with rasterio.open(sign(it.assets[a["red"]].href)) as red, \
                        rasterio.open(sign(it.assets[a["nir"]].href)) as nir, \
                        rasterio.open(sign(it.assets[a["scl"]].href)) as scl:
                    t = red.transform
                    if w is None:
                        geom = g.to_crs(red.crs).geometry.iloc[0]
                        win = window_of(geom, red)
                        w, k = pixel_shares(geom, red, win)
                        lattice = (red.crs, t.a, t.c + win[0] * t.a, t.f - win[1] * t.a)
                    elif red.crs != lattice[0] or abs(t.a - lattice[1]) > 1e-9:
                        raise SystemExit(f"{it.id}: another pixel grid than the first granule of {r.date}")
                    c0, r0 = (lattice[2] - t.c) / t.a, (t.f - lattice[3]) / t.a
                    if abs(c0 - round(c0)) > 1e-6 or abs(r0 - round(r0)) > 1e-6:
                        raise SystemExit(f"{it.id}: pixel grid shifted against the first granule of {r.date}")
                    gw = (round(c0), round(r0), win[2], win[3])
                    x0, y1 = lattice[2], lattice[3]
                    sw = rasterio.windows.from_bounds(x0, y1 - gw[3] * t.a, x0 + gw[2] * t.a, y1, transform=scl.transform)
                    s = scl.read(1, window=sw, out_shape=(gw[3], gw[2]), resampling=Resampling.nearest, boundless=True,
                                 fill_value=0)
                    if not s.any():
                        continue                                  # unit outside the swath of this granule
                    valid = np.isin(s, c2["valid_scl"])
                    nd = np.full(s.shape, np.nan)
                    if valid.any():
                        rr, nn = read_into(red, gw, 0).astype(float), read_into(nir, gw, 0).astype(float)
                        offs, quant = si.s2_offsets(it, sign, c2)   # BOA_ADD_OFFSET of the product metadata
                        good = valid & (rr > 0) & (nn > 0)
                        rr = (rr + offs[c2["assets"]["red"]]) / quant
                        nn = (nn + offs[c2["assets"]["nir"]]) / quant
                        good &= (rr > 0) & (nn > 0)
                        nd[good] = (nn[good] - rr[good]) / (rr[good] + nn[good])
                    nds.append(nd)
                    valids.append(valid)
        row = {"field_id": r.field_id, "date": r.date, "area_ha": float(g.area_ha.iloc[0]) if "area_ha" in g else np.nan,
               "export_ndvi": r.ndvi, "check_ndvi": np.nan, "export_valid_fraction": r.valid_fraction,
               "check_valid_fraction": np.nan, "granules_of_the_date": len(ids), "granules_with_pixels": len(nds),
               "fine_grid_factor": k}
        if nds:
            cnt = np.sum([~np.isnan(x) for x in nds], axis=0)
            mean = np.where(cnt > 0, np.nansum(nds, axis=0) / np.maximum(cnt, 1), np.nan)
            ok = cnt > 0
            row["check_ndvi"] = float((w[ok] * mean[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else np.nan
            row["check_valid_fraction"] = float((w * np.any(valids, axis=0)).sum() / w.sum())
        rows.append(row)
        print(f"    NDVI sample {len(rows)}: {r.field_id} {r.date} export {r.ndvi:.4f} check {row['check_ndvi']:.4f}",
              flush=True)
    t = pd.DataFrame(rows)
    t["ndvi_diff"] = t.export_ndvi - t.check_ndvi
    t["valid_fraction_diff"] = t.export_valid_fraction - t.check_valid_fraction
    return t


# ---------------------------------------------------------------- Landsat
def clear_sample(cfg, fields, clear, n, seed):
    """Unit-scene clear shares recomputed from QA_PIXEL (see module text). One row per unit and scene."""
    import pystac_client
    import rasterio
    cl = cfg["stac"]["landsat"]
    sign = si.signer(cl.get("sign"))
    col = pystac_client.Client.open(cl["api"]).get_collection(cl["collection"])
    fill, bad = 1 << cl["fill_bit"], sum(1 << b for b in cl["masked_bits"])
    rows = []
    for r in pick(clear, n, seed, lo=0.05).itertuples():
        g = fields[fields.field_id == r.field_id]
        it = si.retry(lambda: list(col.get_items(r.item_id)), cfg["stac"]["retries"], "scene")[0]
        with rasterio.Env(**si.GDAL_ENV), rasterio.open(sign(it.assets[cl["qa_asset"]].href)) as qa:
            geom = g.to_crs(qa.crs).geometry.iloc[0]
            win = window_of(geom, qa)
            w, k = pixel_shares(geom, qa, win)
            q = read_into(qa, win, fill)
        data = (q & fill) == 0
        share = float((w * (data & ((q & bad) == 0))).sum() / w[data].sum()) if w[data].sum() > 0 else np.nan
        rows.append({"field_id": r.field_id, "date": r.date, "sensor": r.sensor, "item_id": r.item_id,
                     "export_valid_fraction": r.valid_fraction, "check_valid_fraction": share, "fine_grid_factor": k})
        print(f"    Landsat sample {len(rows)}: {r.field_id} {r.item_id} export {r.valid_fraction:.4f} check {share:.4f}",
              flush=True)
    t = pd.DataFrame(rows)
    t["valid_fraction_diff"] = t.export_valid_fraction - t.check_valid_fraction
    return t


# ---------------------------------------------------------------- ERA5-Land
def weather_hours(cfg, cells, raw, start, end, variables=None):
    """The hourly files as one dataset at the cell nodes, with the problems found: (dataset or None, time name,
    [problems], facts). Expected: every hour from (start - 1 day) 00 UTC to end 23 UTC, every variable, no gap.
    variables: {request name: file name}; default the weather variables (stac.era5land.variables)."""
    import xarray as xr
    v = variables or cfg["stac"]["era5land"]["variables"]
    tname = "valid_time"
    stems = si.era5_stems(raw) if raw.exists() else []
    problems, facts, good = [], {"files": len(stems) * len(v), "months": len(stems)}, []
    if not stems:
        return None, None, [f"no hourly GeoTIFF (record *.tif.json) in {raw}"], facts
    for stem in stems:
        rec = si.read_json(stem.with_name(stem.name + ".tif.json")) or {}
        tifs = si.era5_tifs(stem, v)
        bad = [p.name for n, p in tifs.items() if not p.exists()
               or config.sha256(p) != ((rec.get("files") or {}).get(n) or {}).get("sha256")]
        if bad:
            problems.append(f"{stem.name}: GeoTIFF missing or changed since it was written: {bad}")
            continue
        rq = stem.with_name(stem.name + ".request.json")
        if not rq.exists():
            problems.append(f"{stem.name}: no request file")
        else:
            q = json.loads(rq.read_text(encoding="utf-8"))
            if q.get("dataset") != cfg["stac"]["era5land"]["dataset"] or sorted(q.get("variable", [])) != sorted(v):
                problems.append(f"{stem.name}: request is for another dataset or variable list")
        good.append(stem)
    if not good:
        return None, None, problems, facts
    try:
        times, data, far = si.era5_hours(good, cells, v)
    except SystemExit as e:  # an hourly file that cannot serve the cells is a finding
        return None, None, problems + [str(e)], facts
    s = xr.Dataset({n: ((tname, "cell"), a) for n, a in data.items()}, coords={tname: times})
    s = s.drop_duplicates(tname).sortby(tname)
    if far > 1e-3:
        problems.append(f"grid nodes up to {far:.4f} degree away from the cell centres")
    want = pd.date_range(start - pd.Timedelta(days=1), end + pd.Timedelta(hours=23), freq="h")
    have = pd.DatetimeIndex(s[tname].values)
    gap = want.difference(have)
    if len(gap):
        problems.append(f"{len(gap)} of {len(want)} hours missing (first {gap[0]}, last {gap[-1]})")
    nan = {n: int(np.isnan(s[n].sel({tname: have.intersection(want)}).values).sum()) for n in v.values()}
    if sum(nan.values()):
        problems.append(f"missing values at the cell nodes: {nan}")
    facts.update({"hours_expected": len(want), "hours_present": int(len(have.intersection(want))),
                  "first_hour": str(have.min()), "last_hour": str(have.max()), "cells": len(cells),
                  "largest_node_distance_deg": far})
    return s.assign_coords(cell=("cell", cells.cell.to_numpy())), tname, problems, facts


def netcdf_check(raw, variables):
    """The hourly GeoTIFFs against the netCDF files of the Climate Data Store that are still in the folder (files of
    a download before v0.6.0; new downloads remove the netCDF after the conversion): (files compared, [differences])."""
    import xarray as xr
    n, diff = 0, []
    for nc in sorted(raw.glob("*.nc")):
        stem = nc.with_suffix("")
        with xr.open_dataset(nc, engine="netcdf4") as d:
            tname = "valid_time" if "valid_time" in d.dims else "time"
            d = d.sortby(tname).sortby("latitude", ascending=False).sortby("longitude")
            for name, p in si.era5_tifs(stem, variables).items():
                if not p.exists():
                    diff.append(f"{p.name} missing")
                    continue
                hours, a, _ = si.era5_tif(p)
                same = (np.array_equal(a, d[name].transpose(tname, "latitude", "longitude").values, equal_nan=True)
                        and hours.equals(pd.DatetimeIndex(d[tname].values)))
                n += 1
                if not same:
                    diff.append(f"{p.name} differs from {nc.name}")
    return n, diff


def elevation_sample(cfg, cells, n, seed):
    """Elevation of n cells recomputed from the tiles of the STAC DEM collection read at the source (the stored
    GeoTIFF is not used): mean of the pixels in the window of stac.dem.window_m at the cell centre, the edge row /
    column that two tiles share counted once. Returns cell, check_elev_m, check_n_pixels."""
    import math

    import pystac_client
    import rasterio
    from rasterio.windows import from_bounds
    cd, st = cfg["stac"]["dem"], cfg["stac"]
    pick = cells.sample(min(n, len(cells)), random_state=seed)
    half = cd["window_m"] / 2
    dlat = half / 111320.0
    bbox = [pick.lon.min() - 0.02, pick.lat.min() - 0.02, pick.lon.max() + 0.02, pick.lat.max() + 0.02]
    items = si.retry(lambda: list(pystac_client.Client.open(cd["api"]).search(collections=[cd["collection"]],
                                                                            bbox=bbox).items()),
                     st["retries"], f"search {cd['collection']}")
    sign, rows = si.signer(cd.get("sign")), []
    with rasterio.Env(**si.GDAL_ENV):
        open_ds = {i.id: rasterio.open(sign(i.assets[cd["asset"]].href)) for i in items}
        try:
            for c in pick.itertuples():
                dlon = half / (111320.0 * math.cos(math.radians(c.lat)))
                box = (c.lon - dlon, c.lat - dlat, c.lon + dlon, c.lat + dlat)
                px = {}
                for ds in open_ds.values():
                    b = ds.bounds
                    if box[0] >= b.right or box[2] <= b.left or box[1] >= b.top or box[3] <= b.bottom:
                        continue
                    win = from_bounds(*box, transform=ds.transform).round_offsets().round_lengths()
                    c0, r0 = int(win.col_off), int(win.row_off)
                    a = si.read_window(ds, c0, r0, int(win.width), int(win.height), fill=-32768).astype(float)
                    if ds.nodata is not None:
                        a[a == ds.nodata] = np.nan
                    a[a == -32768] = np.nan
                    rr, cc = np.nonzero(~np.isnan(a))
                    for r, k in zip(rr, cc):
                        x, y = ds.transform * (c0 + k + 0.5, r0 + r + 0.5)
                        px[(round(x / abs(ds.transform.a)), round(y / abs(ds.transform.e)))] = a[r, k]
                rows.append({"cell": c.cell, "check_elev_m": float(np.mean(list(px.values()))) if px else np.nan,
                             "check_n_pixels": len(px)})
        finally:
            for ds in open_ds.values():
                ds.close()
    return pd.DataFrame(rows)


def accumulated_sum(acc, t_s, t_b):
    """Sum from t_s (exclusive) to t_b (inclusive) of an ERA5-Land variable accumulated from 00 UTC (the value at
    00 UTC closes the previous day, the value at 01 UTC is the first hour; ECMWF ERA5-Land documentation):
    - acc(t_s) + acc(every 00 UTC in (t_s, t_b]) + acc(t_b). acc: function time -> accumulated value."""
    s = (0.0 if t_s.hour == 0 else -acc(t_s)) + (acc(t_b) if t_b.hour != 0 else 0.0)
    return s + sum(acc(m) for m in pd.date_range(t_s.floor("D") + pd.Timedelta(days=1), t_b, freq="D"))


def weather_sample(hours, tname, e3, n, seed, utc_offset_hours, precip=None):
    """Cell-days of E3 recomputed from the hourly values. Local day D = UTC + offset: the 24 hourly values whose
    local time falls in D; radiation (and precipitation) of the 24 hours ending in D (by the middle of each hour)
    from the values accumulated since 00 UTC (accumulated_sum). This is a second method: step 00c sums the
    hour-to-hour differences. precip: (hourly dataset at the cells, time name, variable name) or None."""
    off = pd.Timedelta(hours=utc_offset_hours)
    rows = []
    for r in e3.sample(min(n, len(e3)), random_state=seed).itertuples():
        day = pd.Timestamp(r.date)
        p = hours.sel(cell=r.cell)
        first = (day - off).ceil("h")                              # instantaneous values: local time in day D
        h = p.sel({tname: slice(first, first + pd.Timedelta(hours=23))})
        if h.sizes[tname] != 24:
            raise SystemExit(f"{r.cell} {day:%Y-%m-%d}: {h.sizes[tname]} of 24 hours in the hourly files")
        t_a = (day - off + pd.Timedelta(minutes=30)).ceil("h")     # first hour whose middle is in day D
        t_s, t_b = t_a - pd.Timedelta(hours=1), t_a + pd.Timedelta(hours=23)
        rs = accumulated_sum(lambda t: float(p.ssrd.sel({tname: t})), t_s, t_b)
        chk = {"rs_mj_m2": rs / 1e6, "tmin_c": float(h.t2m.min()) - 273.15, "tmax_c": float(h.t2m.max()) - 273.15,
               "tdew_c": float(h.d2m.mean()) - 273.15, "u10_ms": float(np.hypot(h.u10, h.v10).mean()),
               "sp_kpa": float(h.sp.mean()) / 1000}
        if precip is not None and "precip_mm" in e3:
            pp, pt, pn = precip
            q = pp.sel(cell=r.cell)
            chk["precip_mm"] = accumulated_sum(lambda t: float(q[pn].sel({pt: t})), t_s, t_b) * 1000.0   # m -> mm
        rows.append({"cell": r.cell, "date": f"{day:%Y-%m-%d}",
                     **{f"{k}_export": getattr(r, k) for k in chk}, **{f"{k}_check": v for k, v in chk.items()},
                     **{f"{k}_diff": getattr(r, k) - v for k, v in chk.items()}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- manifest
def manifest_files(p_manifest):
    """(listed, changed [paths], missing [paths]) of the outputs recorded by step 00c against the files now."""
    m = json.loads(p_manifest.read_text(encoding="utf-8"))
    listed, changed, missing = 0, [], []
    for entries in m.get("outputs", {}).values():
        for e in entries:
            listed += 1
            p = Path(e["path"])
            if not p.exists():
                missing.append(str(p))
            elif config.sha256(p) != e["sha256"]:
                changed.append(str(p))
    return listed, changed, missing, m
