"""Tests of the GeoTIFF storage of step 00c (v0.6.0) on synthetic rasters - no download, no real data:
stored pixels equal the source, the tables computed from the stored files equal a direct per-field computation on
the source, the field values step 11 reads back from the stored Landsat rasters equal those of step 00c, the hourly
weather survives the netCDF -> GeoTIFF conversion, and a JPG map is made of a GeoTIFF."""
import copy
import datetime
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config, stac_ingest as si  # noqa: E402

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

X0, Y0 = 600000.0, 2500000.0                    # upper left corner of the synthetic UTM grid (EPSG:32643)


def write(path, a, res, c0=0, r0=0, nodata=0):
    """Synthetic source raster: array on the UTM lattice, first pixel at column c0, row r0 (in pixels of `res`)."""
    with rasterio.open(path, "w", driver="GTiff", width=a.shape[1], height=a.shape[0], count=1, dtype=a.dtype.name,
                       crs="EPSG:32643", transform=from_origin(X0 + c0 * res, Y0 - r0 * res, res, res), nodata=nodata,
                       tiled=True, blockxsize=256, blockysize=256, compress="deflate") as ds:
        ds.write(a, 1)
    return str(path)


def fields(res):
    """Seven units on the lattice (pixels of `res` m): boxes and triangles, two of them next to each other, one
    reaching outside the first raster, one outside every raster."""
    def box(c, r, w, h):
        return Polygon([(X0 + c * res, Y0 - r * res), (X0 + (c + w) * res, Y0 - r * res),
                        (X0 + (c + w) * res, Y0 - (r + h) * res), (X0 + c * res, Y0 - (r + h) * res)])
    geoms = [box(100.3, 120.6, 40.2, 31.7), box(140.5, 120.6, 25, 60.4),
             Polygon([(X0 + 500 * res, Y0 - 400 * res), (X0 + 590.5 * res, Y0 - 410 * res), (X0 + 520 * res, Y0 - 505.5 * res)]),
             box(700.2, 600.9, 130.1, 90.3), box(1150.4, 900.2, 120, 150), box(880.7, 350.1, 12.3, 9.9),
             box(5000, 5000, 30, 30)]
    g = gpd.GeoDataFrame({"field_id": [f"F{i}" for i in range(len(geoms))], "tehsil": "t", "cell": "22.6_76.0"},
                         geometry=geoms, crs=32643).to_crs(4326)
    return g


def test_windows_and_strips():
    assert si.coarse([5, 7, 4, 3], 2) == [2, 3, 3, 2]
    assert si.union([[5, 7, 4, 3], [20, 1, 3, 3]], 2) == [4, 0, 20, 10]
    plan = si.strip_plan([[10, 500, 30, 30], [35, 520, 10, 10], [200, 0, 5, 5]], rows=512)
    assert plan == {0: [[10, 40], [200, 205]], 512: [[10, 45]]}        # per strip; ranges merged where they touch


def test_store_clip_keeps_source_values(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.integers(1, 6000, (1500, 2100), dtype=np.uint16)
    with rasterio.open(write(tmp_path / "src.tif", a, 10)) as src:
        whole = si.store_clip(src, tmp_path / "whole.tif", [100, 50, 1200, 1300])
        parts = [[120, 60, 30, 40], [900, 1100, 300, 200]]
        part = si.store_clip(src, tmp_path / "part.tif", [100, 50, 1200, 1300], parts, scale=1e-4, offset=-0.1)
    with rasterio.open(tmp_path / "whole.tif") as ds:
        assert np.array_equal(ds.read(1), a[50:1350, 100:1300]) and ds.transform.c == X0 + 1000 and ds.nodata == 0
    with rasterio.open(tmp_path / "part.tif") as ds:
        b = ds.read(1)
        assert ds.scales == (1e-4,) and ds.offsets == (-0.1,)
    want = np.zeros_like(b)
    for c, r, w, h in parts:
        want[r - 50:r - 50 + h, c - 100:c - 100 + w] = a[r:r + h, c:c + w]
    assert np.array_equal(b, want)                                     # only the parts are stored, the rest is no data
    assert whole["sha256"] == config.sha256(tmp_path / "whole.tif") and part["bytes"] < whole["bytes"] / 5
    assert not list(tmp_path.glob("*.part"))


@pytest.fixture
def s2(tmp_path, monkeypatch):
    """Two overlapping synthetic granules of one date, stored by the real s2_item (no network)."""
    rng = np.random.default_rng(2)
    cfg = copy.deepcopy(config.load("dewas_2025_26"))
    c2 = cfg["stac"]["sentinel2"]
    monkeypatch.setattr(si, "s2_offsets", lambda item, sign, c2, tries=4: ({"B04": -1000.0, "B08": -1000.0}, 10000.0))
    items = []
    for n, (c0, r0) in enumerate(((0, 0), (400, 300))):
        red = rng.integers(0, 4000, (1100, 1200), dtype=np.uint16)     # zeros (no data) and values below the offset
        nir = rng.integers(0, 6000, (1100, 1200), dtype=np.uint16)
        scl = rng.choice(np.array([0, 4, 4, 5, 8], np.uint8), (550, 600))
        href = {"B04": write(tmp_path / f"g{n}_B04.tif", red, 10, c0, r0), "B08": write(tmp_path / f"g{n}_B08.tif", nir, 10, c0, r0),
                "SCL": write(tmp_path / f"g{n}_SCL.tif", scl, 20, c0 // 2, r0 // 2)}
        items.append(SimpleNamespace(id=f"S2X_MSIL2A_20260105T05321{n}_R105_T43QFF_20260105T08401{n}",
                                     datetime=datetime.datetime(2026, 1, 5, 5, 32, n, tzinfo=datetime.timezone.utc),
                                     properties={"s2:processing_baseline": "05.11"},
                                     assets={k: SimpleNamespace(href=v) for k, v in href.items()}))
    f = fields(10)
    grid = si.FieldGrid(f)
    store = tmp_path / "granules"
    store.mkdir()
    recs = [si.s2_item(i, list(f.field_id), grid, lambda h: h, c2, store) for i in items]
    return cfg, c2, f, grid, items, recs, store, tmp_path


def test_s2_stored_pixels_equal_the_source(s2):
    cfg, c2, f, grid, items, recs, store, tmp = s2
    for item, rec in zip(items, recs):
        assert set(rec["files"]) == {"scl", "red", "nir"} and "F6" not in rec["units"]
        assert si.s2_stored(store, item.id, list(f.field_id), grid)["files"] == rec["files"]
        assert si.s2_stored(store, item.id, list(f.field_id)[:3], grid) is None      # other units: download again
        for name, asset, k in (("red", "B04", 1), ("nir", "B08", 1), ("scl", "SCL", 2)):
            u = [v // k for v in rec["window"]]
            with rasterio.open(item.assets[asset].href) as src, rasterio.open(store / rec["files"][name]["file"]) as loc:
                assert loc.transform == src.window_transform(rasterio.windows.Window(*u)) and loc.dtypes == src.dtypes
                for w in rec["units"].values():
                    w = si.coarse(w, k)
                    assert np.array_equal(src.read(1, window=rasterio.windows.Window(*w)),
                                          loc.read(1, window=rasterio.windows.Window(w[0] - u[0], w[1] - u[1], w[2], w[3])))


def test_s2_table_from_rasters_equals_per_field_computation(s2):
    cfg, c2, f, grid, items, recs, store, tmp = s2
    key = (int(recs[0]["grid"][0]), *recs[0]["grid"][1:])
    extent = grid.extent(key)
    plan = si.strip_plan([[c - extent[0], r - extent[1], w, h] for c, r, w, h in (grid.window(key, i) for i in f.field_id)])
    out = tmp / "ndvi"
    out.mkdir()
    rec = si.s2_date("run", "2026-01-05", recs, key, extent, plan, c2, store, out)
    assert set(rec["files"]) == {"clear", "red", "nir", "ndvi"}
    assert si.s2_date("run", "2026-01-05", recs, key, extent, plan, c2, store, out) == json.loads(json.dumps(rec))   # kept
    rows = pd.DataFrame(si.s2_rows("run", "2026-01-05", 2, f, grid, key, extent, out),
                        columns=["field_id", "date", "ndvi", "valid_fraction", "n_scenes"]).set_index("field_id")
    add, n_with = {"red": -1000.0, "nir": -1000.0}, 0
    for fid in f.field_id:                                             # the per-field method of v0.5.0, on the source
        gc, gr, w = grid.weights(key, fid)
        nds, valids = [], []
        for item, r in zip(items, recs):
            with rasterio.open(item.assets["SCL"].href) as scl, rasterio.open(item.assets["B04"].href) as red, \
                    rasterio.open(item.assets["B08"].href) as nir:
                nd, valid, seen, _ = si.s2_pixels(scl, red, nir, (gc - r["origin"][0], gr - r["origin"][1], w.shape[1],
                                                                  w.shape[0]), 2, c2, add, 10000.0)
            if seen.any():
                nds.append(nd)
                valids.append(valid)
        if not nds:
            assert np.isnan(rows.ndvi[fid]) and rows.valid_fraction[fid] == 0
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(np.stack(nds), axis=0)
        ok = ~np.isnan(m)
        assert rows.ndvi[fid] == pytest.approx((w[ok] * m[ok]).sum() / w[ok].sum(), abs=1e-12)
        assert rows.valid_fraction[fid] == pytest.approx((w * np.any(valids, axis=0)).sum() / w.sum(), abs=1e-12)
        n_with += 1
    assert n_with >= 5
    with rasterio.open(out / "run_red_2026-01-05.tif") as red, rasterio.open(out / "run_ndvi_2026-01-05.tif") as nd:
        r, n = red.read(1), nd.read(1)
        assert red.scales == (1e-4,) and ((r != 65535) == np.isfinite(n)).all() and r[r != 65535].max() <= 3000


def test_landsat_scene_from_stored_bands(tmp_path):
    rng = np.random.default_rng(3)
    cfg = copy.deepcopy(config.load("dewas_2025_26"))
    cl = {**cfg["stac"]["landsat"], "context_km": 0.3, "result_rasters": ["ef", "eta"], "download_assets": ["lwir11"]}
    qa = rng.choice(np.array([21824, 21824, 21824, 22280, 55052], np.uint16), (800, 900))   # clear, cloud, cirrus
    qa[:, :105] = 1                                                    # fill: outside the scene (cuts the first unit)
    red = rng.integers(7000, 20000, (800, 900), dtype=np.uint16)
    lw = rng.integers(40000, 50000, (800, 900), dtype=np.uint16)
    bands = {"scale": 2.75e-05, "offset": -0.2, "nodata": 0}
    item = SimpleNamespace(
        id="LC09_L2SP_146044_20260101_02_T1", datetime=datetime.datetime(2026, 1, 1, 5, 20, tzinfo=datetime.timezone.utc),
        properties={"platform": "landsat-9"},
        assets={"qa_pixel": SimpleNamespace(href=write(tmp_path / "qa.tif", qa, 30, nodata=1), extra_fields={}),
                "red": SimpleNamespace(href=write(tmp_path / "red.tif", red, 30), extra_fields={"raster:bands": [bands]}),
                "lwir11": SimpleNamespace(href=write(tmp_path / "lw.tif", lw, 30), extra_fields={
                    "raster:bands": [{"scale": 0.00341802, "offset": 149.0, "nodata": 0}]})})
    f = fields(30).iloc[:4]                                            # the first four units lie inside the scene
    grid = si.FieldGrid(f)
    scenes, ras = tmp_path / "scenes", tmp_path / "ras"
    scenes.mkdir()
    ras.mkdir()

    def ef_fn(scene):                                                  # not an EF model: bands passed through
        return {"ef": scene.read("red"), "eta": scene.read("lwir11")}
    rows = pd.DataFrame(si.landsat_item(item, list(f.field_id), grid, lambda h: h, cl, ef_fn, scenes, ras)).set_index("field_id")
    rec = si.read_json(scenes / f"{item.id}.json")
    c0, r0, w, h = rec["window"]
    assert set(rec["files"]) == {"qa_pixel", "lwir11", "red"}          # QA, the configured band, the band that was read
    for band, a in (("qa_pixel", qa), ("red", red), ("lwir11", lw)):
        with rasterio.open(scenes / rec["files"][band]["file"]) as ds:
            assert np.array_equal(ds.read(1), a[r0:r0 + h, c0:c0 + w])
    with rasterio.open(scenes / rec["files"]["red"]["file"]) as ds:
        assert ds.scales == (2.75e-05,) and ds.offsets == (-0.2,)
    key = (32643, 30.0, 0.0, 10.0)
    assert tuple(rec["source"]["grid"]) == key
    bad = sum(1 << b for b in cl["masked_bits"])
    for fid in f.field_id:
        gc, gr, wt = grid.weights(key, fid)
        oc, orow = rec["source"]["origin"]
        sl = (slice(gr - orow, gr - orow + wt.shape[0]), slice(gc - oc, gc - oc + wt.shape[1]))
        data = (qa[sl] & 1) == 0
        clear = data & ((qa[sl] & bad) == 0)
        assert rows.valid_fraction[fid] == pytest.approx((wt * clear).sum() / wt[data].sum(), abs=1e-12)
        assert rows.ef[fid] == pytest.approx((wt[clear] * (red[sl][clear] * 2.75e-05 - 0.2)).sum() / wt[clear].sum(), abs=1e-9)
    with rasterio.open(ras / f"{item.id}_clear.tif") as ds:
        m = ds.read(1)
        assert set(np.unique(m)) == {0, 1, 255} and (m == 255).sum() == ((qa[r0:r0 + h, c0:c0 + w] & 1) != 0).sum()
    assert (ras / f"{item.id}_ef.tif").exists() and (ras / f"{item.id}_eta.tif").exists()
    # step 11: the same field values read back from the stored rasters (clear mask, EF and ETa as float32)
    back = pd.DataFrame(si.landsat_rows(ras, item.id, "2026-01-01", "L9", f, grid)).set_index("field_id")
    assert list(back.index) == list(rows.index)
    for c in ("valid_fraction", "ef", "eta_mm_day"):
        np.testing.assert_allclose(back[c].to_numpy(float), rows[c].to_numpy(float), rtol=1e-6)


def test_hourly_netcdf_to_geotiff_and_back(tmp_path):
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(4)
    times = pd.date_range("2026-01-01", "2026-01-03 23:00", freq="h")
    lat, lon = np.array([22.5, 22.6, 22.7]), np.array([76.0, 76.1])     # latitude ascending: turned north-up
    v = {"2m_temperature": "t2m", "surface_solar_radiation_downwards": "ssrd"}
    data = {n: rng.normal(280, 5, (len(times), 3, 2)).astype(np.float32) for n in v.values()}
    data["t2m"][5, 0, 0] = np.nan
    nc = tmp_path / "era5land_hourly_2026_01_d01-03.nc"
    xr.Dataset({n: (("valid_time", "latitude", "longitude"), a) for n, a in data.items()},
               coords={"valid_time": times, "latitude": lat, "longitude": lon}).to_netcdf(nc)
    stem = nc.with_suffix("")
    rec = si.nc_to_tifs(nc, stem, v, 0.1, "synthetic")
    assert rec["hours"] == 72 and rec["grid"]["north"] == pytest.approx(22.75) and si.era5_stems(tmp_path) == [stem]
    cells = pd.DataFrame({"cell": ["22.5_76.0", "22.7_76.1"], "lat": [22.5, 22.7], "lon": [76.0, 76.1]})
    hours, got, far = si.era5_hours([stem], cells, v)
    assert list(hours) == list(times) and far < 1e-9
    for n in v.values():
        assert np.array_equal(got[n][:, 0], data[n][:, 0, 0], equal_nan=True)        # values unchanged, cell by cell
        assert np.array_equal(got[n][:, 1], data[n][:, 2, 1], equal_nan=True)
    h = si.hourly_frame([stem], cells, v)
    assert len(h) == 144 and list(h.columns) == ["cell", "time", "t2m", "ssrd"] and h.time.is_monotonic_increasing is False
    with pytest.raises(SystemExit):
        si.grid_nodes(si.era5_tif(si.era5_tifs(stem, v)["t2m"])[2], (3, 2), pd.DataFrame({"lat": [23.5], "lon": [76.0]}))


def test_weather_rasters(tmp_path):
    cfg = copy.deepcopy(config.load("dewas_2025_26"))
    days = pd.date_range("2026-01-01", "2026-01-04")
    e3 = pd.DataFrame([{"cell": c, "date": d, "rs_mj_m2": 15.0 + i, "tmin_c": 9.0, "tmax_c": 27.0 + k, "tdew_c": 10.0,
                        "u10_ms": 2.0, "sp_kpa": 97.0, "elev_m": 400.0 + 10 * k, "lat": float(c.split("_")[0]), "precip_mm": 0.0}
                       for k, c in enumerate(["22.5_76.0", "22.7_76.2"]) for i, d in enumerate(days)])
    out = si.weather_rasters(cfg, e3, tmp_path)
    with rasterio.open(out["raster_tmax_c"]) as ds:
        a = ds.read()
        assert a.shape == (4, 3, 3) and ds.descriptions[0] == "2026-01-01" and ds.transform.c == pytest.approx(75.95)
        assert a[:, 2, 0] == pytest.approx(27.0) and a[:, 0, 2] == pytest.approx(28.0) and np.isnan(a[:, 1, 1]).all()
    with rasterio.open(out["raster_eto_mm"]) as ds:
        assert 1 < np.nanmin(ds.read()) < np.nanmax(ds.read()) < 8      # FAO-56 ETo of a mild winter day
    with rasterio.open(out["raster_elev_m"]) as ds:
        assert ds.count == 1 and ds.read(1)[0, 2] == 410


def test_jpg_map_of_a_geotiff(tmp_path):
    pytest.importorskip("matplotlib")
    from core import quicklook
    cfg = config.load("dewas_2025_26")
    rng = np.random.default_rng(5)
    (tmp_path / "ndvi").mkdir()
    nd = rng.random((700, 900)).astype(np.float32)
    nd[:200] = np.nan
    cl = rng.choice(np.array([0, 1, 255], np.uint8), (700, 900))
    write(tmp_path / "ndvi" / "run_ndvi_2026-01-05.tif", nd, 10, nodata=np.nan)
    write(tmp_path / "ndvi" / "run_clear_2026-01-05.tif", cl, 10, nodata=255)
    a = quicklook.make_map(tmp_path / "ndvi" / "run_ndvi_2026-01-05.tif", tmp_path / "maps" / "ndvi.jpg", cfg)
    b = quicklook.make_map(tmp_path / "ndvi" / "run_clear_2026-01-05.tif", tmp_path / "maps" / "clear.jpg", cfg)
    assert a["kind"] == "value" and 0 <= a["value_min"] < a["value_max"] <= 1 and a["pixels_with_data_share"] == pytest.approx(5 / 7, abs=0.01)
    assert b["kind"] == "class" and b["classes"] == "clear; not clear"
    for p in ("ndvi.jpg", "clear.jpg"):
        assert (tmp_path / "maps" / p).read_bytes()[:3] == b"\xff\xd8\xff"          # a JPEG file
