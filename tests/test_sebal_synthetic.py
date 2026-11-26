"""End-to-end test of core.ef_sebal.compute_ef on a synthetic Landsat scene with synthetic weather files: the
function runs, returns every output, EF lies within 0-1, the vegetated cold end is wet (EF near 1) and the bare hot
end dry (EF near 0), the energy balance closes and the anchor log is written. No download, no real data."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config, ef_sebal  # noqa: E402

xr = pytest.importorskip("xarray")
affine = pytest.importorskip("affine")

N = 60
WHEN = pd.Timestamp("2026-01-10 05:30:00", tz="UTC")


class Scene:
    """Synthetic scene: vegetation cover 0 (columns 0-9, bare and hot) rising to 1 (columns 50-59, green and cool);
    the extremes are blocks, so anchor candidates with a full clear 3 x 3 neighbourhood exist."""

    def __init__(self, context):
        f = np.tile(np.clip((np.arange(N) - 10) / 40, 0, 1), (N, 1))
        rng = np.random.default_rng(1)
        self.bands = {"red": 0.20 - 0.16 * f, "nir08": 0.25 + 0.17 * f, "blue": 0.06 + 0 * f, "green": 0.08 + 0 * f,
                      "swir16": 0.28 - 0.10 * f, "swir22": 0.22 - 0.12 * f,
                      "lwir11": 316.0 - 22.0 * f + rng.normal(0, 0.05, f.shape), "qa_pixel": np.zeros(f.shape, int)}
        self.shape, self.datetime, self.context = (N, N), WHEN.to_pydatetime(), context
        self.transform = affine.Affine(30.0, 0, 600000.0, 0, -30.0, 2500000.0)
        self.crs = "EPSG:32643"
        self.item = SimpleNamespace(id="SYNTHETIC_SCENE", properties={"view:sun_elevation": 45.0})

    def read(self, asset, scaled=True):
        return self.bands[asset]


@pytest.fixture
def context(tmp_path):
    import copy
    cfg = copy.deepcopy(config.load("dewas_2025_26"))
    cfg["sebal"]["hot_pixel"]["method"] = "zero"                   # SEBAL hot pixel; the water balance: own test
    cfg["sebal"]["anchors"]["sensitivity"] = cfg["sebal"]["anchors"]["sensitivity_rules"]   # diagnostic switched on
    raw = tmp_path / "raw"
    raw.mkdir()
    times = pd.date_range("2026-01-01", "2026-01-31 23:00", freq="h")
    lat, lon = np.array([22.5, 22.6]), np.array([76.0, 76.1])
    shape = (len(times), 2, 2)
    ds = xr.Dataset({"u10": (("valid_time", "latitude", "longitude"), np.full(shape, 2.0)),
                     "v10": (("valid_time", "latitude", "longitude"), np.full(shape, 1.0)),
                     "d2m": (("valid_time", "latitude", "longitude"), np.full(shape, 283.15)),
                     "sp": (("valid_time", "latitude", "longitude"), np.full(shape, 97000.0))},
                    coords={"valid_time": times, "latitude": lat, "longitude": lon})
    ds.to_netcdf(raw / "era5land_hourly_2026_01_d01-31.nc")
    ds.close()
    from core import stac_ingest                                 # the hourly weather is read from GeoTIFF (v0.6.0)
    stac_ingest.nc_to_tifs(raw / "era5land_hourly_2026_01_d01-31.nc", raw / "era5land_hourly_2026_01_d01-31",
                           {"10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
                            "2m_dewpoint_temperature": "d2m", "surface_pressure": "sp"}, 0.1, "synthetic test file")
    cells = pd.DataFrame({"cell": ["22.5_76.0"], "lat": [22.5], "lon": [76.0], "elev_m": [500.0]})
    cells.to_csv(tmp_path / "cell_elevation.csv", index=False)
    e3 = pd.DataFrame({"cell": "22.5_76.0", "date": pd.date_range("2026-01-01", "2026-01-31").strftime("%Y-%m-%d"),
                       "rs_mj_m2": 18.0, "tmin_c": 9.0, "tmax_c": 27.0, "tdew_c": 10.0, "u10_ms": 2.2,
                       "sp_kpa": 97.0, "elev_m": 500.0, "lat": 22.5, "precip_mm": 0.0})
    e3.to_csv(tmp_path / "e3.csv", index=False)
    return {"cfg": cfg, "era5_raw": raw, "e3_file": tmp_path / "e3.csv", "cell_elevation": tmp_path / "cell_elevation.csv",
            "utc_offset": 5.5, "anchor_log": tmp_path / "sebal_anchors.csv"}


def test_compute_ef_synthetic(context):
    out = ef_sebal.compute_ef(Scene(context))
    ef_sebal._open.clear()
    assert set(out) == {"ef", "eta", "rn_w_m2", "g_w_m2", "h_w_m2", "le_w_m2"}
    ef = out["ef"]
    assert np.isfinite(ef).all() and ef.min() >= 0 and ef.max() <= 1
    assert ef[:, -5:].mean() > 0.85 and ef[:, :5].mean() < 0.15          # wet cold end, dry hot end
    assert np.all(np.diff(ef.mean(axis=0))[10:50] > -0.02)              # EF rises with vegetation cover
    closure = out["rn_w_m2"] - out["g_w_m2"] - out["h_w_m2"] - out["le_w_m2"]
    assert np.abs(closure).max() < 1e-2                                   # LE is the residual (eq. 1)
    assert (out["eta"] >= 0).all()
    log = pd.read_csv(context["anchor_log"])
    assert len(log) == 1 and log.status.iloc[0].startswith("ok")
    assert bool(log.cold_homogeneous.iloc[0]) and bool(log.hot_homogeneous.iloc[0])
    assert log.cold_ts_k.iloc[0] < log.hot_ts_k.iloc[0]
    assert log[[c for c in log if c.endswith("_ef_median")]].notna().all(axis=None)   # anchor sensitivity


def test_compute_ef_with_terrain_datum(context, tmp_path):
    """Ts_datum (eq. 29): a DEM rising 300 m across the scene; the run uses it (log: DEM range, datum elevation)."""
    rasterio = pytest.importorskip("rasterio")
    import json
    scene = Scene(context)
    dem = np.tile(np.linspace(300, 600, N, dtype=np.float32), (N, 1))
    p = tmp_path / "dem.tif"
    with rasterio.open(p, "w", driver="GTiff", height=N, width=N, count=1, dtype="float32", crs="EPSG:32643",
                       transform=scene.transform, nodata=-32768) as dst:
        dst.write(dem, 1)
    bbox = [76.0, 22.5, 76.1, 22.6]
    p.with_suffix(".json").write_text(json.dumps({"bbox": bbox}), encoding="utf-8")
    context.update({"dem_file": p, "dem_bbox": bbox})
    out = ef_sebal.compute_ef(scene)
    ef_sebal._open.clear()
    assert np.isfinite(out["ef"]).all() and out["ef"].min() >= 0 and out["ef"].max() <= 1
    log = pd.read_csv(context["anchor_log"])
    assert log.dem_min_m.iloc[0] == pytest.approx(300, abs=6) and log.dem_max_m.iloc[0] == pytest.approx(600, abs=6)
    assert log.hot_ndvi.iloc[0] < 0.13                                    # hot pixel on bare soil


def test_compute_ef_hot_pixel_soil_water_balance(context):
    """METRIC hot pixel: bare-soil ET from the FAO-56 water balance. From field capacity on 1 Jan without rain the
    topsoil is in stage-2 drying on 10 Jan (0 < Ke < Kc max) and the hot pixel keeps some sensible heat. After 30 mm of
    rain the day before, the bare soil evaporates at stage 1 (Ke = Kc max) more than its daily net radiation: no
    pixel is dry, the hot anchor has no sensible heat and the scene gets no EF (interpolated from other dates)."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    soil = gpd.GeoDataFrame({"unit": ["T1"], "theta_fc": [0.38], "theta_wp": [0.22], "tew_mm": [27.0]},
                            geometry=[box(70, 15, 80, 30)], crs=4326)
    context["cfg"]["sebal"]["hot_pixel"]["method"] = "soil_water_balance"
    context["soil_units"] = soil
    out = ef_sebal.compute_ef(Scene(context))
    log = pd.read_csv(context["anchor_log"])
    assert log.hot_method.iloc[0] == "soil_water_balance" and 0 < log.hot_ke.iloc[0] < 1.2
    assert log.hot_de_mm.iloc[0] > 10                                    # De > REW: stage 2
    e3 = pd.read_csv(context["e3_file"])
    e3.loc[e3.date == "2026-01-09", "precip_mm"] = 30.0                # heavy rain the day before the overpass
    e3.to_csv(context["e3_file"], index=False)
    ef_sebal._wb.clear()
    context["anchor_log"].unlink()
    out2 = ef_sebal.compute_ef(Scene(context))
    ef_sebal._open.clear()
    ef_sebal._wb.clear()
    log2 = pd.read_csv(context["anchor_log"])
    assert 0 < log.hot_ef.iloc[0] < 1 and np.isfinite(out["ef"]).all()
    assert log2.status.iloc[0].startswith("no EF: hot pixel without sensible heat")
    assert np.isnan(out2["ef"]).all()
