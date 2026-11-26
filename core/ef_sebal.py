"""Landsat evaporative fraction EF = LE / (Rn - G) by SEBAL - the EF function of step 00c
(config stac.landsat.ef_function: "core.ef_sebal:compute_ef"; settings: config sebal).

Equations as published. Numbers "eq. N" are those of Allen R.G., Tasumi M., Trezza R. (2007) Satellite-based energy
balance for mapping evapotranspiration with internalized calibration (METRIC) - Model. J Irrig Drain Eng 133(4):
380-394 (references/Allen_2007_METRIC_model_JIrrigDrainEng.pdf). That paper sets out the SEBAL energy balance
(Bastiaanssen 1995; Bastiaanssen et al. 1998; Bastiaanssen 2000) on which METRIC is built and states where METRIC
departs from it. SEBAL as published is used where they differ:
  - cold pixel: dT = 0 (H = 0) "for Ts equal to the surface temperature of a local water body or in its absence, a
    well-vegetated field" (p. 387), not METRIC's LE_cold = 1.05 ETr lambda (eq. 48);
  - hot pixel: LE = 0, dT_hot from eq. 46 (no soil water balance);
  - 24 h: "In traditional applications of SEBAL, the evaporative fraction EF is used rather than ETrF ... EF is assumed
    to be the same at both the observation time and for the 24 h period" (p. 392);
  - incoming long-wave emissivity eq. 25 with the original SEBAL coefficients (Bastiaanssen 1995, as cited there);
  - horizontal surfaces (eq. 8, no slope or aspect terms) but Ts_datum in eq. 29: Ts adjusted to a common elevation
    with the 30 m NASADEM and the lapse rate 0.0065 K m-1 of FAO-56 Annex 3 eq. 3-2 (config sebal.terrain_datum);
  - hot pixel on bare soil, NDVI < 0.13 (Bastiaanssen & Ali 2003 p.327).

Per pixel of the scene window (step 00c, 30 m), clear pixels of QA_PIXEL as for the E2 clear share:
  NDVI eq. 23, SAVI eq. 19, LAI eq. 18 (0-6), albedo eq. 15 with the weights Wb of Table 1 (Landsat bands 1-5, 7 ->
  OLI blue, green, red, nir08, swir16, swir22) applied to the Collection 2 Level-2 surface reflectance (the
  at-surface reflectance rho_s,b of eqs. 11, 15); Ts = Collection 2 Level-2 surface temperature (ST_B10, K - the
  USGS product takes the place of eqs. 20-22); surface emissivity eq. 17 (water, NDVI < 0: 0.985, p. 384).
  Rs_in eq. 3 (cos theta from the sun elevation of the scene; 1/d^2 = dr, FAO-56 eq. 23), tau_sw eq. 4, P eq. 5,
  W eq. 6 (ea at the overpass from the ERA5-Land dewpoint, FAO-56 eq. 14). RL_out eq. 16; RL_in eq. 24 with eps_a
  eq. 25 and Ta = Ts of the cold pixel (p. 385). Rn eq. 2. G eq. 26 (G/Rn of Bastiaanssen 2000; water 0.5, p. 385).
  z0m eq. 33 (>= 0.005 m). u200 eq. 32 from the ERA5-Land 10 m wind at the overpass. u*, rah: eqs. 31, 30 (neutral),
  then eqs. 38, 39 with L eq. 40 and psi eqs. 41-45 (Paulson 1970; Webb 1970), iterated. rho_air eq. 37.
  dT = a + b Ts (eq. 29) through (Ts_cold, 0) and (Ts_hot, dT_hot). H eq. 28. LE = Rn - G - H (eq. 1).
Daily ETa (mm d-1) = EF * Rn24 / lambda: DERIVED from the EF definition (EF constant over the day, p. 392; no source
in references/ prints it as an equation), Rn24 = Rns - Rnl (FAO-56, Allen et al. 1998, eqs. 37-40: Rns with the pixel
albedo, eq. 38; Rnl eq. 39 from the daily ERA5-Land weather of E3), G24 = 0 (FAO-56 eq. 42), lambda = 2.45 MJ kg-1
(FAO-56). ETa is a diagnostic (ETa / ETc) and the 'relative_et' alternative; the yield model uses EF (Ws = EF:
Bastiaanssen & Ali 2003 eq. 7; Yuan et al. 2007 eq. 5).
Weather: the mean of the run's ERA5-Land cells inside the scene window (u200 and the atmosphere are constant over
the image, p. 386). The hot and cold pixels are chosen by the rule of config sebal.anchors - processing settings,
not literature values (Allen et al. 2007, p. 393: "highly reliant on the operator's ability to select appropriate
cold and hot pixels"). Among the candidates of the rule the pixel with the most uniform 3 x 3 surface temperature is
taken (homogeneous neighbourhood, all 9 pixels clear) - a setting. Every scene's choice, the elevation range of its
pixels and weather cells and the EF of alternative anchor rules
(config sebal.anchors.sensitivity) are written to data/stac/landsat/<run>/sebal_anchors.csv for review.
Outputs per pixel: EF (0-1), ETa (mm d-1) and the instantaneous Rn, G, H, LE (W m-2) at the overpass.
"""
import csv
import json
import math
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from . import eto
from . import soil_evaporation as se

GSC = 1367.0            # solar constant, W m-2 (eq. 3)
SIGMA = 5.67e-8         # Stefan-Boltzmann constant, W m-2 K-4 (eq. 16)
SIGMA_DAY = 4.903e-9    # Stefan-Boltzmann constant, MJ K-4 m-2 d-1 (FAO-56 eq. 39)
VK = 0.41               # von Karman's constant (eq. 30)
GRAV = 9.807            # gravitational acceleration, m s-2 (eq. 40)
R_AIR = 287.0           # specific gas constant, J kg-1 K-1 (eq. 37)
WB = {"blue": 0.254, "green": 0.149, "red": 0.147, "nir08": 0.311, "swir16": 0.103, "swir22": 0.036}   # Table 1, Wb

_lock = threading.Lock()
_dem_lock = threading.Lock()
_wb_lock = threading.Lock()
_wb = {}                # bare-soil water balance per (E3 file, cell, soil unit), built once per process
_open = {}              # hourly ERA5-Land GeoTIFFs read once per process


# ---------------------------------------------------------------- weather of the scene
def _context(scene):
    c = getattr(scene, "context", None)
    if not c:
        raise SystemExit("core.ef_sebal needs the run context of step 00c (scene.context) - run it through step 00c")
    for k in ("era5_raw", "e3_file", "cell_elevation"):
        if not Path(c[k]).exists():
            raise SystemExit(f"SEBAL needs the weather of E3 before E2: {c[k]} not found - run "
                             f"'python 00c_download.py --run {c['cfg']['run_id']} --only e3' first")
    return c


def _cells_in_window(scene, cells):
    """ERA5-Land cells of the run whose centre lies inside the scene window (all cells when none does)."""
    from pyproj import Transformer
    crs = scene.crs.to_wkt() if hasattr(scene.crs, "to_wkt") else scene.crs
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(cells.lon.to_numpy(), cells.lat.to_numpy())
    t, (h, w) = scene.transform, scene.shape
    x0, x1 = sorted([t.c, t.c + w * t.a])
    y0, y1 = sorted([t.f, t.f + h * t.e])
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return cells[inside] if inside.any() else cells


def _hourly(ctx, cells, hour):
    """ERA5-Land 10 m wind speed (m s-1), 2 m dewpoint (degC) and surface pressure (kPa) of one UTC hour, mean of the
    cells."""
    from core import stac_ingest as si
    v = ctx["cfg"]["stac"]["era5land"]["variables"]                  # CDS name -> netCDF name
    names = [v["10m_u_component_of_wind"], v["10m_v_component_of_wind"], v["2m_dewpoint_temperature"],
             v["surface_pressure"]]
    out = []
    for n in names:                                                  # hourly GeoTIFF of the month, one per variable
        for p in sorted(Path(ctx["era5_raw"]).glob(f"era5land_hourly_{hour.year}_{hour.month:02d}_d*_{n}.tif")):
            with _lock:
                if p not in _open:
                    _open[p] = si.era5_tif(p)                        # (hours, values[hour, row, col], transform)
                hours, a, tr = _open[p]
            i = hours.get_indexer([hour])[0]
            if i < 0:
                continue
            row, col, _ = si.grid_nodes(tr, a.shape[1:], cells, p.name)
            out.append(a[i, row, col].astype(float))
            break
        else:
            raise SystemExit(f"SEBAL: ERA5-Land hour {hour:%Y-%m-%d %H:%M} UTC not in {ctx['era5_raw']} - run E3 "
                             "for the period")
    u, vv, td, sp = out
    return {"wind_ms": float(np.mean(np.hypot(u, vv))), "tdew_c": float(np.mean(td) - 273.15),
            "sp_kpa": float(np.mean(sp) / 1000.0)}


def overpass_weather(ctx, cells, when_utc):
    """Weather at the overpass: linear in time between the two ERA5-Land hours around it."""
    h0 = when_utc.floor("h")
    w = (when_utc - h0) / pd.Timedelta(hours=1)
    a, b = _hourly(ctx, cells, h0), _hourly(ctx, cells, h0 + pd.Timedelta(hours=1))
    return {k: (1 - w) * a[k] + w * b[k] for k in a}


def daily_weather(ctx, cells, when_utc):
    """E3 daily weather of the local calendar day of the overpass, mean of the cells."""
    e3 = pd.read_csv(ctx["e3_file"], parse_dates=["date"])
    day = (when_utc + pd.Timedelta(hours=float(ctx["utc_offset"]))).floor("D")
    x = e3[(e3.date == day) & e3.cell.isin(cells.cell)]
    if x.empty:
        raise SystemExit(f"SEBAL: no E3 weather for {day:%Y-%m-%d} in {ctx['e3_file']}")
    return x[["rs_mj_m2", "tmin_c", "tmax_c", "tdew_c", "lat", "elev_m"]].mean(), day


def cos_zenith(scene, lat_deg, lon_deg, when_utc):
    """cos of the solar zenith angle over a horizontal surface: from the scene's sun elevation, else eq. 8 with the
    declination (FAO-56 eq. 24) and the hour angle from solar time (FAO-56 eqs. 31-33)."""
    el = scene.item.properties.get("view:sun_elevation")
    if el is not None:
        return math.sin(math.radians(float(el)))
    j = when_utc.dayofyear
    delta = 0.409 * math.sin(2 * math.pi / 365 * j - 1.39)                       # FAO-56 eq. 24
    b = 2 * math.pi * (j - 81) / 364                                             # FAO-56 eq. 33
    sc = 0.1645 * math.sin(2 * b) - 0.1255 * math.cos(b) - 0.025 * math.sin(b)  # FAO-56 eq. 32
    t_solar = when_utc.hour + when_utc.minute / 60 + lon_deg / 15 + sc           # local solar time, h
    omega = math.pi / 12 * (t_solar - 12)                                        # FAO-56 eq. 31
    phi = math.radians(lat_deg)
    return math.sin(delta) * math.sin(phi) + math.cos(delta) * math.cos(phi) * math.cos(omega)   # eq. 8


# ---------------------------------------------------------------- terrain: elevation of every pixel (eq. 29 Ts_datum)
def _build_dem_mosaic(ctx):
    """NASADEM mosaic (1 arc-second, STAC collection of config stac.dem) of the box ctx['dem_bbox'] (lon/lat), kept in
    ctx['dem_file'] with the box in a .json beside it; built once per run and used for every scene."""
    import rasterio
    from rasterio.merge import merge
    from core import config, stac_ingest as si
    cd, st = ctx["cfg"]["stac"]["dem"], ctx["cfg"]["stac"]
    path = Path(ctx["dem_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    items = si.retry(lambda: si.search(cd["api"], cd["collection"], ctx["dem_bbox"], pd.Timestamp("2000-01-01"),
                                       pd.Timestamp.now(), tries=st["retries"]), st["retries"], "DEM search")
    if not items:
        raise SystemExit(f"no {cd['collection']} items for the box {ctx['dem_bbox']}")
    sign = si.signer(cd.get("sign"))
    with rasterio.Env(**si.GDAL_ENV):
        srcs = [rasterio.open(sign(i.assets[cd["asset"]].href)) for i in items]
        try:
            arr, tr = merge(srcs, bounds=tuple(ctx["dem_bbox"]), nodata=-32768)
            prof = {"driver": "GTiff", "height": arr.shape[1], "width": arr.shape[2], "count": 1,
                    "dtype": arr.dtype, "crs": srcs[0].crs, "transform": tr, "nodata": -32768, "compress": "deflate"}
        finally:
            for s in srcs:
                s.close()

    def write(tmp):
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(arr)
    config.write_atomic(path, write)
    path.with_suffix(".json").write_text(json.dumps({"bbox": list(ctx["dem_bbox"]), "items": [i.id for i in items]}),
                                         encoding="utf-8")


def scene_dem(scene, ctx):
    """Terrain elevation (m) on the grid of the scene window: the run's NASADEM mosaic resampled bilinearly."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    path = Path(ctx["dem_file"])
    with _dem_lock:
        meta = path.with_suffix(".json")
        if not (path.exists() and meta.exists()
                and json.loads(meta.read_text(encoding="utf-8"))["bbox"] == list(ctx["dem_bbox"])):
            _build_dem_mosaic(ctx)
    out = np.full(scene.shape, np.nan, np.float32)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, src_nodata=src.nodata, dst_transform=scene.transform,
                  dst_crs=scene.crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


# ---------------------------------------------------------------- hot pixel: ET of a bare soil (METRIC, p.387)
def hot_pixel_et(ctx, lon, lat, day):
    """Bare-soil evaporation (FAO-56 ch. 7 water balance, core.soil_evaporation) on the local day of the overpass at
    the hot pixel: soil of the tehsil containing it (nearest tehsil when outside), rain and ETo of the nearest E3
    weather cell; the balance runs from the first day of the E3 export. Returns a dict (unit, cell, tew_mm, de_mm, kr,
    ke, e_mm)."""
    import geopandas as gpd
    from shapely.geometry import Point
    hp = ctx["cfg"]["sebal"]["hot_pixel"]
    su = ctx["soil_units"]
    pt = Point(lon, lat)
    inside = su[su.contains(pt)]
    if len(inside):
        u = inside.iloc[0]
    else:                                            # outside the district: nearest tehsil, distance in metres (UTM)
        utm = su.estimate_utm_crs()
        p_m = gpd.GeoSeries([pt], crs=4326).to_crs(utm).iloc[0]
        u = su.iloc[int(np.argmin(su.to_crs(utm).distance(p_m).to_numpy()))]
    with _wb_lock:
        key0 = ("e3", str(ctx["e3_file"]))
        if key0 not in _wb:
            e3 = pd.read_csv(ctx["e3_file"], parse_dates=["date"])
            if "precip_mm" not in e3:
                raise SystemExit(f"{ctx['e3_file']}: no precip_mm - run E3 again (step 00c --only e3)")
            e3["lon"] = e3.cell.str.split("_").str[1].astype(float)
            e3["eto_mm"] = eto.eto_fao56(e3.tmin_c, e3.tmax_c, e3.tdew_c, e3.rs_mj_m2, eto.wind_2m(e3.u10_ms),
                                         e3.sp_kpa, e3.lat, e3.elev_m, e3.date.dt.dayofyear)   # FAO-56 eq. 6
            _wb[key0] = e3
        e3 = _wb[key0]
        cells = e3.drop_duplicates("cell")
        cell = cells.cell.iloc[int(np.argmin((cells.lat - lat) ** 2 + (cells.lon - lon) ** 2))]
        key = (str(ctx["e3_file"]), cell, u.unit)
        if key not in _wb:
            w = e3[e3.cell == cell].sort_values("date")
            _wb[key] = se.bare_soil(w.date, w.precip_mm, w.eto_mm, float(u.tew_mm), hp["rew_mm"], hp["kc_max"],
                                    hp["kcb"], hp["few"], hp["de_start_mm"]).set_index("date")
        wb = _wb[key]
    if day not in wb.index:
        raise SystemExit(f"bare-soil water balance: {day:%Y-%m-%d} not in {ctx['e3_file']}")
    r = wb.loc[day]
    return {"unit": u.unit, "cell": cell, "tew_mm": float(u.tew_mm), "de_mm": float(r.de_start_mm),
            "kr": float(r.kr), "ke": float(r.ke), "e_mm": float(r.e_mm)}


# ---------------------------------------------------------------- stability corrections (eqs. 41-45)
def stability(L, zb=200.0, z2=2.0, z1=0.1):
    """psi_m(zb), psi_h(z2), psi_h(z1) for the Monin-Obukhov length L (Paulson 1970; Webb 1970); heights from config
    sebal (blending_height_m, z2_m, z1_m; the paper: 200, 2 and 0.1 m)."""
    pm, ph2, ph01 = np.zeros_like(L), np.zeros_like(L), np.zeros_like(L)
    un = np.isfinite(L) & (L < 0)
    if un.any():
        x200 = (1 - 16 * zb / L[un]) ** 0.25                                      # eq. 43a
        x2 = (1 - 16 * z2 / L[un]) ** 0.25                                        # eq. 43b
        x01 = (1 - 16 * z1 / L[un]) ** 0.25                                       # eq. 43c
        pm[un] = (2 * np.log((1 + x200) / 2) + np.log((1 + x200 ** 2) / 2) - 2 * np.arctan(x200)
                  + 0.5 * np.pi)                                                  # eq. 41
        ph2[un] = 2 * np.log((1 + x2 ** 2) / 2)                                   # eq. 42a
        ph01[un] = 2 * np.log((1 + x01 ** 2) / 2)                                 # eq. 42b
    st = np.isfinite(L) & (L > 0)
    if st.any():
        pm[st] = -5 * (z2 / L[st])                                                # eq. 44 (2 m, not 200 m: p. 387)
        ph2[st] = -5 * (z2 / L[st])                                               # eq. 45a
        ph01[st] = -5 * (z1 / L[st])                                              # eq. 45b
    return pm, ph2, ph01


def _anchor(ndvi, ts, ok, q_ndvi, q_ts, cold, local_sd=None):
    """(index, homogeneous) of the hot or cold pixel: NDVI above (cold) / below (hot) its quantile, Ts below (cold) /
    above (hot) its quantile among those (config sebal.anchors). local_sd given: the candidate with the smallest 3 x 3
    Ts standard deviation among those with a fully clear neighbourhood; else (or none such) the candidate with the
    median Ts."""
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return None, False
    if cold:
        c = idx[ndvi[idx] >= np.quantile(ndvi[idx], q_ndvi)]
        c = c[ts[c] <= np.quantile(ts[c], q_ts)]
    else:
        c = idx[ndvi[idx] <= np.quantile(ndvi[idx], q_ndvi)]
        c = c[ts[c] >= np.quantile(ts[c], q_ts)]
    if len(c) == 0:
        return None, False
    if local_sd is not None:
        h = c[np.isfinite(local_sd[c])]
        if len(h):
            return int(h[np.argmin(local_sd[h])]), True
    return int(c[np.argsort(ts[c])[len(c) // 2]]), False


def local_ts_sd(ts2d, ok2d, k):
    """Standard deviation of Ts in the k x k neighbourhood of every pixel; NaN where any of the k*k pixels is not
    clear (homogeneity of anchor candidates - a setting)."""
    from scipy.ndimage import uniform_filter
    m = ok2d.astype(np.float64)
    t = np.where(ok2d, ts2d, 0.0).astype(np.float64)
    n = uniform_filter(m, k, mode="constant") * k * k
    s1 = uniform_filter(t, k, mode="constant") * k * k
    s2 = uniform_filter(t * t, k, mode="constant") * k * k
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.maximum(s2 / n - (s1 / n) ** 2, 0.0)
    return np.where(np.isclose(n, k * k), np.sqrt(var), np.nan)


LOG_FIELDS = ["item_id", "datetime_utc", "status", "clear_pixels", "cells", "cell_elev_min_m", "cell_elev_max_m",
              "dem_min_m", "dem_max_m", "datum_elev_m",
              "u10_ms", "u200_ms", "tdew_c", "p_kpa", "cos_zenith", "tau_sw", "rs_in_w_m2",
              "cold_ts_k", "cold_ndvi", "cold_homogeneous", "cold_ts_sd_3x3_k",
              "hot_ts_k", "hot_ndvi", "hot_homogeneous", "hot_ts_sd_3x3_k",
              "hot_rn_minus_g_w_m2", "hot_method", "hot_unit", "hot_cell", "hot_tew_mm", "hot_de_mm", "hot_ke",
              "hot_et24_mm", "hot_ef", "hot_le_w_m2", "dT_hot_k", "a", "b", "iterations", "ef_clipped_pixels", "ef_median",
              "rn24_mj_m2", "local_day"] + [f"sens{i}_{k}" for i in (1, 2, 3)
                                            for k in ("rule", "ef_median", "abs_diff_median", "status")]


def _log(ctx, row):
    """One row per scene (hot / cold pixel, atmosphere, convergence). Step 00c gives a list (anchor_sink): the row is
    kept with the scene's cached result and sebal_anchors.csv is rebuilt from all scenes; else appended to
    ctx['anchor_log']."""
    if ctx.get("anchor_sink") is not None:
        ctx["anchor_sink"].append({k: row.get(k, "") for k in LOG_FIELDS})
        return
    p = ctx.get("anchor_log")
    if not p:
        return
    with _lock:
        new = not Path(p).exists()
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS, restval="")
            if new:
                w.writeheader()
            w.writerow(row)


# ---------------------------------------------------------------- SEBAL
def compute_ef(scene):
    """EF (0-1), ETa (mm d-1) and Rn, G, H, LE (W m-2, overpass) of every clear pixel of the scene window; NaN
    elsewhere."""
    ctx = _context(scene)
    cfg, sb = ctx["cfg"], ctx["cfg"]["sebal"]
    cl, an = cfg["stac"]["landsat"], sb["anchors"]
    shape = scene.shape
    nan = {k: np.full(shape, np.nan, np.float32) for k in ("ef", "eta", "rn_w_m2", "g_w_m2", "h_w_m2", "le_w_m2")}
    when = pd.Timestamp(scene.datetime)
    when = (when.tz_convert("UTC") if when.tzinfo else when.tz_localize("UTC")).tz_localize(None)
    row = {"item_id": scene.item.id, "datetime_utc": when.isoformat(), "status": ""}

    # clear pixels (QA_PIXEL, as for the E2 clear share) with physical surface reflectance and temperature
    qa = scene.read(cl["qa_asset"], scaled=False).astype(np.int64)
    bad = sum(1 << b for b in cl["masked_bits"])
    ok = ((qa & (1 << cl["fill_bit"])) == 0) & ((qa & bad) == 0)
    rho = {b: scene.read(b).astype(np.float32) for b in WB}
    ts = scene.read("lwir11").astype(np.float32)                                 # ST_B10, K
    for x in rho.values():
        ok &= np.isfinite(x) & (x >= 0) & (x <= 1)
    ok &= np.isfinite(ts) & (ts > 0)
    del qa
    n_ok = int(ok.sum())
    if n_ok < an["min_clear_pixels"]:
        _log(ctx, {**row, "status": f"no EF: {n_ok} clear pixels (< {an['min_clear_pixels']})"})
        return nan
    # Ts_datum (eq. 29): Ts adjusted to a common elevation with the DEM and a lapse rate, used for the dT function and
    # the anchor choice; every other equation uses Ts. The common elevation cancels (dT line fitted in the same datum).
    tsd2d = ts.astype(np.float64)
    if sb.get("terrain_datum") and ctx.get("dem_file"):
        dem = scene_dem(scene, ctx)
        ok &= np.isfinite(dem)
        n_ok = int(ok.sum())
        if n_ok < an["min_clear_pixels"]:
            _log(ctx, {**row, "status": f"no EF: {n_ok} clear pixels with elevation (< {an['min_clear_pixels']})"})
            return nan
        z_datum = float(np.median(dem[ok]))
        tsd2d = tsd2d + sb["lapse_rate_k_m"] * (dem - z_datum)
        row.update({"dem_min_m": round(float(dem[ok].min()), 1), "dem_max_m": round(float(dem[ok].max()), 1),
                    "datum_elev_m": round(z_datum, 1)})
        del dem
    sel = np.flatnonzero(ok)
    hk = an.get("homogeneity_window")
    sd_flat = local_ts_sd(tsd2d, ok, int(hk)).ravel()[sel] if hk else None
    tsd = tsd2d.ravel()[sel]
    del tsd2d
    red, nir = rho["red"].ravel()[sel], rho["nir08"].ravel()[sel]
    albedo = sum(w * rho[b].ravel()[sel] for b, w in WB.items())                  # eq. 15
    del rho
    ts = ts.ravel()[sel].astype(np.float64)
    ndvi = (nir - red) / (nir + red)                                              # eq. 23
    L_ = sb["savi_l"]
    savi = (1 + L_) * (nir - red) / (L_ + nir + red)                              # eq. 19
    with np.errstate(invalid="ignore", divide="ignore"):
        lai = -np.log((0.69 - savi) / 0.59) / 0.91                                # eq. 18
    lai = np.where(savi >= 0.687, 6.0, np.where(savi <= 0.1, 0.0, lai))          # limits of eq. 18 (p. 384)
    lai = np.clip(np.nan_to_num(lai, nan=6.0), 0.0, 6.0)
    water = ndvi < 0
    eps0 = np.where(lai < 3, 0.95 + 0.01 * lai, 0.98)                            # eq. 17
    eps0 = np.where(water, 0.985, eps0)                                           # NDVI < 0 (p. 384)

    # atmosphere of the scene: weather at the overpass, cells of the run inside the window
    cells = pd.read_csv(ctx["cell_elevation"])
    cells = _cells_in_window(scene, cells)
    row.update({"cells": len(cells), "cell_elev_min_m": round(float(cells.elev_m.min()), 1),
                "cell_elev_max_m": round(float(cells.elev_m.max()), 1)})
    wx = overpass_weather(ctx, cells, when)
    z = float(cells.elev_m.mean())
    cos_t = cos_zenith(scene, float(cells.lat.mean()), float(cells.lon.mean()), when)
    if cos_t <= 0.05:
        _log(ctx, {**row, "status": f"no EF: sun too low (cos zenith {cos_t:.3f})"})
        return nan
    p_kpa = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26                            # eq. 5
    ea = float(eto.svp(wx["tdew_c"]))                                             # FAO-56 eq. 14
    w_mm = 0.14 * ea * p_kpa + 2.1                                                # eq. 6
    tau = 0.35 + 0.627 * math.exp(-0.00146 * p_kpa / (sb["kt"] * cos_t) - 0.075 * (w_mm / cos_t) ** 0.4)   # eq. 4
    dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * when.dayofyear)                 # 1/d^2 (eq. 9)
    rs_in = GSC * cos_t * tau * dr                                                # eq. 3 (flat)
    ea_air = sb["emissivity_air"]["a"] * (-math.log(tau)) ** sb["emissivity_air"]["b"]   # eq. 25 (SEBAL coefficients)
    rl_out = eps0 * SIGMA * ts ** 4                                               # eq. 16
    g_rn = np.where(water, 0.5, (ts - 273.15) * (0.0038 + 0.0074 * albedo) * (1 - 0.98 * ndvi ** 4))  # eq. 26; water p. 385
    land = ~water
    z1, z2, zb, cp = sb["z1_m"], sb["z2_m"], sb["blending_height_m"], sb["cp_j_kg_k"]
    z0m = np.maximum(sb["z0m_lai"] * lai, sb["z0m_min"])                          # eq. 33
    zw, z0w = sb["wind_height_m"], sb["z0m_weather_m"]
    u200 = wx["wind_ms"] * math.log(zb / z0w) / math.log(zw / z0w)                # eq. 32

    # daily weather of the overpass day and Rn24 of every pixel (FAO-56; also used for ETa below)
    d, day = daily_weather(ctx, cells, when)
    ra = float(eto.extraterrestrial_radiation(d.lat, day.dayofyear))              # FAO-56 eq. 21
    rso = (0.75 + 2e-5 * d.elev_m) * ra                                           # FAO-56 eq. 37
    ea_d = float(eto.svp(d.tdew_c))                                               # FAO-56 eq. 14
    rnl = (SIGMA_DAY * ((d.tmax_c + 273.16) ** 4 + (d.tmin_c + 273.16) ** 4) / 2 * (0.34 - 0.14 * math.sqrt(ea_d))
           * (1.35 * min(d.rs_mj_m2 / rso, 1.0) - 0.35))                          # FAO-56 eq. 39
    rn24 = (1 - albedo) * d.rs_mj_m2 - rnl                                        # FAO-56 eqs. 38, 40
    hp = sb["hot_pixel"]
    hot_cache = {}
    to_lonlat = None
    if hp["method"] == "soil_water_balance":
        from pyproj import Transformer
        crs = scene.crs.to_wkt() if hasattr(scene.crs, "to_wkt") else scene.crs
        to_lonlat = Transformer.from_crs(crs, 4326, always_xy=True)

    def hot_ef(ih):
        """(EF of the hot pixel, record). SEBAL (method zero): LE_hot = 0, EF 0. METRIC (Allen et al. 2007 p.387):
        ET of a bare soil from the FAO-56 water balance - daily E (mm) -> EF_hot = E lambda / Rn24 (EF constant over
        the day, p.392); LE_hot = EF_hot (Rn - G) at the overpass and H_hot = (Rn - G) - LE_hot (eq. 47)."""
        if ih not in hot_cache:
            if hp["method"] == "zero":
                hot_cache[ih] = (0.0, {"hot_method": "zero (SEBAL)"})
            else:
                r, c = divmod(int(sel[ih]), shape[1])
                x, y = scene.transform * (c + 0.5, r + 0.5)
                lon, lat = to_lonlat.transform(x, y)
                wb = hot_pixel_et(ctx, lon, lat, day)
                rn24_h = float(rn24[ih])
                ef_h = min(1.0, wb["e_mm"] * sb["lambda_mj_kg"] / rn24_h) if rn24_h > 0 else 0.0
                hot_cache[ih] = (ef_h, {"hot_method": "soil_water_balance", **wb})
        return hot_cache[ih]

    def anchors(rule):
        """(cold index, hot index, cold homogeneous, hot homogeneous) of an anchor rule (config sebal.anchors keys;
        homogeneity_window_off: median-Ts candidate instead of the most uniform one)."""
        sd = None if rule.get("homogeneity_window_off") else sd_flat
        ic, hc = _anchor(ndvi, tsd, land, rule["cold_ndvi_quantile"], rule["cold_ts_quantile"], True, sd)
        bare = land & (ndvi > 0) & (ndvi < rule["hot_ndvi_max"])                 # bare soil (B&A 2003 p.327)
        ih, hh = _anchor(ndvi, tsd, bare, 1.0, rule["hot_ts_quantile"], False, sd)
        return ic, ih, hc, hh

    def solve(ic, ih):
        """Energy balance with the cold pixel ic and the hot pixel ih: (rn, g, h, le, ef unbounded, info) or None."""
        if ih is None:
            return None, f"no bare-soil pixel (0 < NDVI < {an['hot_ndvi_max']}) for the hot pixel"
        if ic is None or not tsd[ih] > tsd[ic]:
            return None, "cold pixel not found or Ts_hot <= Ts_cold (datum)"
        rl_in = ea_air * SIGMA * ts[ic] ** 4                                      # eq. 24, Ta = Ts of the cold pixel
        rn = (1 - albedo) * rs_in + rl_in - rl_out - (1 - eps0) * rl_in           # eq. 2
        g = g_rn * rn                                                             # eq. 26: G = (G/Rn) Rn
        avail = rn - g
        if not avail[ih] > 0:
            return None, "Rn - G <= 0 at the hot pixel"
        ef_h, hot = hot_ef(ih)
        h_hot = avail[ih] * (1.0 - ef_h)                                          # eq. 47: H = (Rn - G) - LE
        if not h_hot > 0:
            return None, f"hot pixel without sensible heat (bare-soil EF {ef_h:.2f})"
        u_star = VK * u200 / np.log(zb / z0m)                                     # eq. 31 (neutral)
        rah = np.log(z2 / z1) / (u_star * VK)                                     # eq. 30 (neutral)
        dT = np.zeros_like(ts)
        it_n, conv = 0, False
        for it_n in range(1, sb["iterations"]["max"] + 1):
            rho_air = 1000 * p_kpa / (1.01 * (ts - dT) * R_AIR)                   # eq. 37
            dT_hot = h_hot * rah[ih] / (rho_air[ih] * cp)                          # eqs. 46-47 (H_hot)
            b = dT_hot / (tsd[ih] - tsd[ic])                                      # eq. 29 through (Ts_cold, 0)
            a = -b * tsd[ic]                                                      # and (Ts_hot, dT_hot), datum Ts
            dT = a + b * tsd                                                      # eq. 29
            rho_air = 1000 * p_kpa / (1.01 * (ts - dT) * R_AIR)                   # eq. 37
            h = rho_air * cp * dT / rah                                           # eq. 28
            with np.errstate(divide="ignore", invalid="ignore"):
                L = -rho_air * cp * u_star ** 3 * ts / (VK * GRAV * h)             # eq. 40
            pm, ph2, ph01 = stability(L, zb, z2, z1)                              # eqs. 41-45
            u_star = VK * u200 / (np.log(zb / z0m) - pm)                          # eq. 38
            rah_new = (np.log(z2 / z1) - ph2 + ph01) / (u_star * VK)              # eq. 39
            change = abs(rah_new[ih] - rah[ih]) / rah[ih]
            rah = rah_new
            if change < sb["iterations"]["rah_tolerance"]:
                conv = True
                break
        rho_air = 1000 * p_kpa / (1.01 * (ts - dT) * R_AIR)                       # final H with the final rah
        dT_hot = h_hot * rah[ih] / (rho_air[ih] * cp)                              # eqs. 46-47
        b = dT_hot / (tsd[ih] - tsd[ic])
        a = -b * tsd[ic]
        dT = a + b * tsd                                                          # eq. 29 (Ts_datum)
        h = 1000 * p_kpa / (1.01 * (ts - dT) * R_AIR) * cp * dT / rah             # eqs. 37, 28
        le = avail - h                                                            # eq. 1
        with np.errstate(divide="ignore", invalid="ignore"):
            ef = np.where(avail > 0, le / avail, np.nan)                           # EF = LE / (Rn - G), p. 392
        return (rn, g, h, le, ef, {"dT_hot": dT_hot, "a": a, "b": b, "it_n": it_n, "conv": conv,
                                   "avail_hot": avail[ih], "ef_hot": ef_h, "le_hot": avail[ih] * ef_h, **hot}), "ok"

    # hot and cold pixels (config sebal.anchors: processing settings) and the energy balance
    ic, ih, hom_c, hom_h = anchors(an)
    res, why = solve(ic, ih)
    if res is None:
        _log(ctx, {**row, "status": f"no EF: {why}"})
        return nan
    rn, g, h, le, ef, info = res
    lo, hi = sb["ef_bounds"]                                                      # Ws 0-1 (Yuan et al. 2007 p. 192)
    n_out = int(np.sum((ef < lo) | (ef > hi)))
    ef = np.clip(ef, lo, hi)

    # anchor sensitivity: EF of alternative anchor rules (settings), compared on the clear land pixels
    sens = {}
    for i, alt in enumerate(an.get("sensitivity") or [], 1):
        rule = {**{k: an[k] for k in ("cold_ndvi_quantile", "cold_ts_quantile", "hot_ndvi_max",
                                      "hot_ts_quantile")}, **alt}
        r2, why2 = solve(*anchors(rule)[:2])
        sens[f"sens{i}_rule"] = ";".join(f"{k}={v}" for k, v in alt.items())
        sens[f"sens{i}_status"] = why2
        if r2 is not None:
            ef2 = np.clip(r2[4], lo, hi)
            sens[f"sens{i}_ef_median"] = round(float(np.nanmedian(ef2[land])), 3)
            sens[f"sens{i}_abs_diff_median"] = round(float(np.nanmedian(np.abs(ef2 - ef)[land])), 3)

    # daily ETa: EF constant over the day (SEBAL, p. 392), Rn24 by FAO-56 (above), G24 = 0 - derived, see module text
    eta = ef * np.maximum(rn24, 0.0) / sb["lambda_mj_kg"]                          # mm d-1

    out = {}
    for name, x in (("ef", ef), ("eta", eta), ("rn_w_m2", rn), ("g_w_m2", g), ("h_w_m2", h), ("le_w_m2", le)):
        a_ = np.full(shape[0] * shape[1], np.nan, np.float32)
        a_[sel] = x
        out[name] = a_.reshape(shape)
    sd_c = float(sd_flat[ic]) if sd_flat is not None and np.isfinite(sd_flat[ic]) else np.nan
    sd_h = float(sd_flat[ih]) if sd_flat is not None and np.isfinite(sd_flat[ih]) else np.nan
    _log(ctx, {**row, "status": "ok" if info["conv"] else f"ok, rah not converged in {info['it_n']} iterations",
               "clear_pixels": n_ok, "u10_ms": round(wx["wind_ms"], 3), "u200_ms": round(u200, 3),
               "tdew_c": round(wx["tdew_c"], 2), "p_kpa": round(p_kpa, 2), "cos_zenith": round(cos_t, 4),
               "tau_sw": round(tau, 4), "rs_in_w_m2": round(rs_in, 1),
               "cold_ts_k": round(float(ts[ic]), 2), "cold_ndvi": round(float(ndvi[ic]), 3),
               "cold_homogeneous": hom_c, "cold_ts_sd_3x3_k": round(sd_c, 3),
               "hot_ts_k": round(float(ts[ih]), 2), "hot_ndvi": round(float(ndvi[ih]), 3),
               "hot_homogeneous": hom_h, "hot_ts_sd_3x3_k": round(sd_h, 3),
               "hot_rn_minus_g_w_m2": round(float(info["avail_hot"]), 1), "hot_method": info["hot_method"],
               "hot_unit": info.get("unit", ""), "hot_cell": info.get("cell", ""),
               "hot_tew_mm": round(info["tew_mm"], 2) if "tew_mm" in info else "",
               "hot_de_mm": round(info["de_mm"], 2) if "de_mm" in info else "",
               "hot_ke": round(info["ke"], 3) if "ke" in info else "",
               "hot_et24_mm": round(info["e_mm"], 3) if "e_mm" in info else "",
               "hot_ef": round(float(info["ef_hot"]), 3), "hot_le_w_m2": round(float(info["le_hot"]), 1),
               "dT_hot_k": round(float(info["dT_hot"]), 3),
               "a": round(float(info["a"]), 4), "b": round(float(info["b"]), 5), "iterations": info["it_n"],
               "ef_clipped_pixels": n_out, "ef_median": round(float(np.nanmedian(ef)), 3),
               "rn24_mj_m2": round(float(np.nanmedian(rn24)), 2), "local_day": f"{day:%Y-%m-%d}", **sens})
    return out
