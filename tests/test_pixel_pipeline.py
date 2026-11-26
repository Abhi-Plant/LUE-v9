"""End-to-end test of the pixel chain (v0.9.0) on a synthetic run in a temporary folder: crop map, Sentinel-2 NDVI
and clear rasters, Landsat clear / EF / ETa rasters on a 30 m grid 15 m off the 10 m grid (pixel squares straddle
Landsat pixels; two scenes of one overpass overlap), ERA5-Land table -> export px (step 00c), its verification (step
00d), the pixel yield model (step 05b, with its check against the field-unit code), the pixel yield map (step 11) and
the parcel-mean yield map (step 12). Landsat values of the pixel squares are checked against the field-export code E2
(stac_ingest.landsat_rows, overpass_rows) on every pixel and overpass. No project data or project folder is touched."""
import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
from core import config  # noqa: E402
from core import stac_ingest as si  # noqa: E402

RUN = "synth_2025_26"


def load_step(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").replace("0", "s0", 1), CODE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def write_tif(path, a, x0, y0, res, nodata):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", width=a.shape[1], height=a.shape[0], count=1, dtype=a.dtype.name,
                       crs="EPSG:32643", transform=from_origin(x0, y0, res, res), nodata=nodata) as ds:
        ds.write(a, 1)
    return si.file_record(path)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A synthetic run under a temporary ROOT: (root, cfg)."""
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import Polygon, box
    root = tmp_path_factory.mktemp("root")
    shutil.copytree(CODE / "config", root / "config")
    aoi = config.read_yaml(CODE / "config" / "aoi" / "dewas_2025_26.yaml")
    aoi["inputs"].update({"crop_mask": "data/synth/crop.gpkg", "crop_mask_layer": "potato",
                          "crop_mask_tehsil_field": "tehsil_en"})
    aoi["field_exports"] = {k: f"data/field_exports/{RUN}_{v}" for k, v in
                            (("e1_s2_perdate", "E1_s2_perdate.csv"), ("e2_landsat_perdate", "E2_landsat_perdate.csv"),
                             ("e3_weather_daily", "E3_era5land_daily.csv"))}
    import yaml
    (root / "config" / "aoi" / f"{RUN}.yaml").write_text(yaml.safe_dump(aoi), encoding="utf-8")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "ROOT", root)
    cfg = config.load(RUN)
    cfg["pixel_model"]["check_pixels"] = 12
    cfg["pixel_model"]["chunk_pixels"] = 150                     # several chunks
    # ---- crop map: parcels near 76.1 E, 22.9 N in UTM 43N (EPSG:32643)
    x0, y0 = Transformer.from_crs(4326, 32643, always_xy=True).transform(76.1, 22.9)
    x0, y0 = round(x0 / 30) * 30, round(y0 / 30) * 30
    geoms = [box(x0, y0, x0 + 120, y0 + 80), box(x0 + 200, y0, x0 + 290, y0 + 70),
             Polygon([(x0, y0 + 150), (x0 + 100, y0 + 150), (x0 + 50, y0 + 230)]),
             box(x0 + 300, y0 + 150, x0 + 420, y0 + 260),          # its pixels are always cloudy: cell estimate
             box(x0 + 151, y0 + 301, x0 + 155, y0 + 305)]          # 4 m square: representative point
    cm = gpd.GeoDataFrame({"parcel_id": [f"P{i}" for i in range(len(geoms))],
                           "tehsil_en": ["Alpha", "Alpha", "Beta", "Beta", "Alpha"], "source": "parcel",
                           "prob": 0.9, "review": False, "mixed": False,
                           "area_ha": [g.area / 1e4 for g in geoms]}, geometry=geoms, crs=32643)
    (root / "data" / "synth").mkdir(parents=True)
    cm.to_file(root / "data" / "synth" / "crop.gpkg", layer="potato", driver="GPKG")
    parcels = si.load_parcels(cfg)
    P = si.run_paths(cfg)
    rng = np.random.default_rng(2)
    # ---- Sentinel-2 NDVI rasters on the 10 m grid (origin 0, 0)
    key = (32643, 10.0, 0.0, 0.0)
    extent = si.FieldGrid(parcels).extent(key)
    gc, gr, w, h = extent
    xs = (gc + np.arange(w) + 0.5) * 10.0
    ys = -(gr + np.arange(h) + 0.5) * 10.0
    X, Y = np.meshgrid(xs, ys)
    nd_dir = P["rasters"] / "ndvi"
    nd_dir.mkdir(parents=True)
    start = pd.Timestamp(cfg["season"]["window"][0]) - pd.Timedelta(days=cfg["season"]["lead_days"])
    dates = pd.date_range(start, cfg["season"]["window"][1], freq="5D")
    cloudy = (X > x0 + 300) & (Y > y0 + 150) & (Y < y0 + 260)
    n1 = 70 + (X - x0) / 20.0
    for d in dates:
        t = (d - start).days
        a = 0.2 + 0.55 * (1 / (1 + np.exp(-0.1 * (t - n1))) + 1 / (1 + np.exp(0.08 * (t - 175))) - 1)
        a = (a + rng.normal(0, 0.01, a.shape)).astype(np.float32)
        a[(rng.random(a.shape) < 0.25) | cloudy] = np.nan
        name = f"{RUN}_ndvi_{d:%Y-%m-%d}.tif"
        f = write_tif(nd_dir / name, a, gc * 10.0, -gr * 10.0, 10.0, np.nan)
        write_tif(nd_dir / f"{RUN}_clear_{d:%Y-%m-%d}.tif", np.isfinite(a).astype(np.uint8), gc * 10.0, -gr * 10.0,
                  10.0, 255)                                      # clear mask (step 12 reads it as E1 does)
        si.write_json(nd_dir / f"{RUN}_ndvi_{d:%Y-%m-%d}.json",
                      {"date": f"{d:%Y-%m-%d}", "made_from": {"granules": {"G1": ["x"]}, "grid": list(key),
                                                              "extent": list(extent)}, "files": {"ndvi": f}})
    # ---- Landsat scenes on 30 m grids (origin 15 m off the 10 m grid); one overpass with two overlapping scenes
    ls_dir = P["rasters"] / "landsat"
    ls_dir.mkdir(parents=True)
    items = []
    ovp = [(d, "L8") for d in pd.date_range(start + pd.Timedelta(days=3), cfg["season"]["window"][1], freq="16D")] + \
          [(d, "L9") for d in pd.date_range(start + pd.Timedelta(days=11), cfg["season"]["window"][1], freq="16D")]
    for k, (d, sensor) in enumerate(ovp):
        for part in ((0,) if k != 4 else (0, 1)):
            iid = f"L{sensor[1]}_{d:%Y%m%d}_{part}"
            lx0, ly0 = x0 - 600 + 15 + part * 300, y0 + 900 + 15
            shp = (60, 60)
            clear = (rng.random(shp) > 0.2).astype(np.uint8)
            ef = rng.uniform(0.3, 0.9, shp).astype(np.float32)
            eta = (ef * rng.uniform(3, 5, shp)).astype(np.float32)
            files = {"clear": write_tif(ls_dir / f"{iid}_clear.tif", clear, lx0, ly0, 30.0, 255)}
            si.write_json(ls_dir / f"{iid}_clear.json", {"item_id": iid, "files": files})
            fe = {"ef": write_tif(ls_dir / f"{iid}_ef.tif", ef, lx0, ly0, 30.0, np.nan),
                  "eta": write_tif(ls_dir / f"{iid}_eta.tif", eta, lx0, ly0, 30.0, np.nan)}
            si.write_json(ls_dir / f"{iid}_ef.json", {"item_id": iid, "files": fe})
            items.append({"item_id": iid, "date": f"{d:%Y-%m-%d}", "sensor": sensor, "status": "read"})
    P["folder"]["landsat"].mkdir(parents=True)
    pd.DataFrame(items).to_csv(P["folder"]["landsat"] / "landsat_items.csv", index=False)
    # ---- ERA5-Land daily table of the parcels' cells (one cell missing: the nearest is used)
    days = pd.date_range(start, cfg["season"]["window"][1], freq="D")
    cells = sorted(parcels.cell.unique())
    e3 = pd.concat([pd.DataFrame({"cell": c, "date": days, "rs_mj_m2": rng.uniform(14, 22, len(days)),
                                  "tmin_c": 10 + rng.normal(0, 1, len(days)), "tmax_c": 26 + rng.normal(0, 1, len(days)),
                                  "eto_mm": rng.uniform(2, 5, len(days))}) for c in cells[:1]])
    P["exports"]["e3_weather_daily"].parent.mkdir(parents=True, exist_ok=True)
    e3.to_csv(P["exports"]["e3_weather_daily"], index=False, date_format="%Y-%m-%d")
    fdir = root / "outputs" / RUN / "02_fields"
    fdir.mkdir(parents=True)
    pd.DataFrame({"field_id": parcels.field_id[:2]}).to_csv(fdir / "fields.csv", index=False)
    yield root, cfg, P, parcels
    mp.undo()


def test_pixel_chain(run, monkeypatch):
    from core import pixels as px
    from core import stac_verify as sv
    root, cfg, P, parcels = run
    rec, f = px.build_stacks(cfg, P, parcels)
    assert f["parcels"] == 5 and f["pixel_rules"] == {"centre inside": 4, "representative point": 1}
    assert f["overpasses"] == len(pd.read_csv(P["folder"]["landsat"] / "landsat_items.csv")) - 1   # 2 scenes, 1 overpass
    rec2, _ = px.build_stacks(cfg, P, parcels)                    # unchanged inputs: kept
    assert rec2["made_at"] == rec["made_at"]
    st = px.Stacks(P)
    assert st.n == f["pixels"] and int(st.parcels.n_pixels.sum()) == st.n

    # ---- step 00d: verification of the stacks
    s00d = load_step("00d_verify.py")
    R = sv.Report()
    out = root / "outputs" / RUN / "00d_verify"
    out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "load", lambda r: cfg)
    s00d.pixel_stacks(R, cfg, P, out, 4, 1)
    t = R.table()
    assert len(t) >= 5 and (t.status == "PASS").all(), t.to_string()

    # ---- step 05b
    monkeypatch.setattr(sys, "argv", ["05b_pixel_model.py", "--run", RUN])
    load_step("05b_pixel_model.py").main()
    o5 = root / "outputs" / RUN / "05b_pixel_model"
    ck = pd.read_csv(o5 / "checks.csv")
    assert not (ck.status == "FAIL").any(), ck.to_string()
    assert ck.check.str.startswith("pixel model = field-unit code").any()
    assert (o5 / "manifest.json").exists()
    pv = pd.read_csv(o5 / "parcel_values.csv")
    assert pv.status.tolist()[3] == "cell estimate" and (pv.status != "no estimate").all()
    uc = pd.read_csv(o5 / "unit_check.csv")
    assert uc.same_reason.all() and uc.same_yield.all() and uc.same_season.all()
    md = pd.read_csv(o5 / "members_district.csv")
    assert len(md) == len(pd.read_csv(o5 / "member_grid.csv"))

    # ---- step 08 (stand-in: member factors) and step 09 (stand-in: the final district and tehsil values)
    grid = pd.read_csv(o5 / "member_grid.csv")
    fac = pd.Series(np.linspace(0.6, 0.9, len(grid)), index=grid.member, name="member_factor")
    cid = "placeholder_LUE-ENS_test0000"
    cdir = root / "outputs" / RUN / "08_calibrate" / cid
    cdir.mkdir(parents=True)
    fac.to_csv(cdir / "member_factors.csv")
    pd.DataFrame([{"product": "LUE-Y2-provisional", "factor": 0.75, "status": "placeholder"}]).to_csv(
        cdir / "calibration_factor.csv", index=False)
    (cdir.parent / "latest.json").write_text(json.dumps({"calibration_id": cid, "status": "placeholder"}))

    def final(mem, level):
        x = mem.assign(v=mem.fresh_t_ha * mem.member.map(fac))
        return x.groupby(level).apply(lambda y: np.average(y.v, weights=y.weight), include_groups=False)
    d9 = root / "outputs" / RUN / "09_aggregate"
    d9.mkdir(parents=True)
    fd = float(final(md, "district").iloc[0])
    pd.DataFrame([{"final_t_ha": fd, "stratified_estimate_t_ha": fd, "sampling_se_t_ha": 0.5,
                   "ci95_low": fd - 1, "ci95_high": fd + 1}]).to_csv(d9 / "district_final.csv", index=False)
    ft = final(pd.read_csv(o5 / "members_tehsil.csv"), "tehsil").rename("final_t_ha").reset_index()
    ft.to_csv(d9 / "tehsil_final.csv", index=False)

    # ---- step 11
    monkeypatch.setattr(sys, "argv", ["11_pixel_map.py", "--run", RUN])
    load_step("11_pixel_map.py").main()
    o11 = root / "outputs" / RUN / "11_pixel_map"
    ck = pd.read_csv(o11 / "checks.csv")
    assert not (ck.status == "FAIL").any(), ck.to_string()
    assert (o11 / "manifest.json").exists() and (o11 / f"yield_{RUN}.jpg").exists()
    import rasterio
    with rasterio.open(o11 / f"yield_{RUN}.tif") as ds:
        assert ds.count == len(cfg["pixel_map"]["yield_raster_bands"]) and ds.descriptions[0] == "final_t_ha"
        band = ds.read(1)
    assert np.isfinite(band).sum() >= st.n - 1                     # every crop pixel (a shared pixel shows one parcel)
    py_ = pd.read_csv(o11 / "parcel_yield.csv")
    assert py_.final_t_ha.notna().all() and np.isclose(py_.production_t.sum(),
                                                       float(pd.read_csv(o11 / "district_pixel_map.csv").production_t[0]))
    # the parcel value is the mean of its pixels (final = linear in the members)
    z = np.load(px.stack_folder(P) / "pixels.npz")
    i0 = py_.index[py_.status == "modelled"][0]
    sel = z["parcel"] == i0
    vals = band[z["row"][sel], z["col"][sel]]
    assert np.isclose(np.nanmean(vals), py_.final_t_ha[i0], rtol=1e-5)

    # ---- step 12: parcel-mean yield map next to the pixel-based parcel yields of step 11
    monkeypatch.setattr(sys, "argv", ["12_parcel_map.py", "--run", RUN])
    load_step("12_parcel_map.py").main()
    o12 = root / "outputs" / RUN / "12_parcel_map"
    ck = pd.read_csv(o12 / "checks.csv")
    assert not (ck.status == "FAIL").any(), ck.to_string()
    assert ck.check.str.startswith("parcels: pixel-based yield (step 11)").any()
    assert (o12 / "manifest.json").exists() and (o12 / f"parcel_mean_yield_{RUN}.jpg").exists()
    p12 = pd.read_csv(o12 / "parcel_yield.csv")
    assert len(p12) == len(parcels) and p12.final_t_ha.notna().all()
    assert np.allclose(p12.pixel_based_t_ha, p12.field_id.map(py_.set_index("field_id").final_t_ha))
    assert np.allclose(p12.pixel_minus_parcel_t_ha, p12.pixel_based_t_ha - p12.final_t_ha)
    assert np.isclose(p12.production_t.sum(), float(pd.read_csv(o12 / "district_parcel_map.csv").production_t[0]))
    with rasterio.open(o12 / f"parcel_mean_yield_{RUN}.tif") as ds:
        assert list(ds.descriptions) == list(cfg["parcel_map"]["yield_raster_bands"])


def test_landsat_of_pixel_squares_equals_the_field_export_code(run):
    """v0.9.0: EF, ETa and clear share of every pixel square and overpass (export px) = what the field export E2
    computes with the square as a field polygon (stac_ingest.landsat_rows per scene, overpass_rows per overpass)."""
    import geopandas as gpd
    from shapely import box
    from core import pixels as px
    root, cfg, P, parcels = run
    key, extent, _ = px.s2_grid(cfg, P)
    pix, *_ = px.pixel_index(parcels, key, extent)
    ovp = px.landsat_overpasses(cfg, P)
    ras = P["rasters"] / "landsat"
    h = key[1] / 2
    ids = [f"PX{i}" for i in range(len(pix["x"]))]
    sq = gpd.GeoDataFrame({"field_id": ids}, geometry=box(pix["x"] - h, pix["y"] - h, pix["x"] + h, pix["y"] + h),
                          crs=int(key[0]))
    fgrid = si.FieldGrid(sq)
    items = pd.read_csv(P["folder"]["landsat"] / "landsat_items.csv").set_index("item_id")
    cache, n_partial = {}, 0
    for o in ovp.itertuples():
        v, _ = px.landsat_merged(ras, o.item_ids, pix["x"], pix["y"], key, cache)
        rows = []
        for i in o.item_ids:
            rows += si.landsat_rows(ras, i, items.date[i], items.sensor[i], sq, fgrid)
        e2 = si.overpass_rows(pd.DataFrame(rows)).set_index("field_id").reindex(ids)
        for k, col in (("ef", "ef"), ("eta", "eta_mm_day"), ("clear", "valid_fraction")):
            a, b = v[k], e2[col].to_numpy(float)
            assert (np.isnan(a) == np.isnan(b)).all(), (o.date, k)
            assert np.nanmax(np.abs(a - b)) < 1e-9, (o.date, k)
        n_partial += int(((v["clear"] > 0) & (v["clear"] < 1)).sum())
    assert n_partial > 0                                          # some squares are partly clear: the rule matters
    w = cache[next(iter(cache))][2]
    assert np.allclose(w.sum(1), (key[1] / 30.0) ** 2)            # every square lies fully on the Landsat grid
    assert ((w > 0).sum(1) > 1).any()                             # and some squares straddle 30 m pixels


def test_stacks_of_an_earlier_version_are_refused(run, monkeypatch):
    """Pixel stacks without the v0.9.0 Landsat rule (value at the pixel centre) are not read: step 00c --only px."""
    root, cfg, P, parcels = run
    from core import pixels as px
    px.build_stacks(cfg, P, parcels)
    rec = si.read_json(px.stack_folder(P) / "record.json")
    old = copy.deepcopy(rec)
    old["scales"].pop("landsat_rule")
    si.write_json(px.stack_folder(P) / "record.json", old)
    with pytest.raises(SystemExit, match="earlier version"):
        px.Stacks(P)
    si.write_json(px.stack_folder(P) / "record.json", rec)


def test_stacks_stop_on_another_crop_map(run, monkeypatch):
    """Step 05b refuses pixel stacks of another crop map."""
    root, cfg, P, parcels = run
    from core import pixels as px
    px.build_stacks(cfg, P, parcels)
    rec = si.read_json(px.stack_folder(P) / "record.json")
    bad = copy.deepcopy(rec)
    bad["made_from"]["parcels"] = "0" * 64
    si.write_json(px.stack_folder(P) / "record.json", bad)
    monkeypatch.setattr(sys, "argv", ["05b_pixel_model.py", "--run", RUN])
    with pytest.raises(SystemExit, match="another crop map"):
        load_step("05b_pixel_model.py").main()
    si.write_json(px.stack_folder(P) / "record.json", rec)
