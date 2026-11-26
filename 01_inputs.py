"""Step 01 - input ingestion and quality control; writes the common input tables of the run units.

Tehsil units (config units: tehsil) read in place:
  - 16-day tehsil table (not made by step 00c): Sentinel-2 from every clear ~5-day date over all crop-map fields, Landsat 8/9
    EF and ETa, ERA5 - averaged per tehsil and period [start, end) in the export. Primary input of the method.
    EF / ETa of exactly 0 are no-data values and set to missing. The method document (step 8) prescribes yield per
    pixel / field then area-weighted aggregation - that is the field-unit route (exports E1-E3).
  - per-date Sentinel-2 NDVI of the tehsil crop core mask, and the daily weather workbook.
Field units (config units: field): the step-02 sample and the exports E1 (S2 per date), E2 (Landsat per date),
E3 (ERA5-Land daily per cell) written by step 00c from STAC APIs - format in README "Field exports".
Provenance: fingerprint (sha256), size, file date and CRS of every input vs the registered sha256 of the run config,
and the documented provenance fields (provenance.csv; undocumented items WARN).
Data QC: crop map (geometry, class, area, sources, parcel-id reuse, overlaps), duplicates, physical ranges, exact zeros
vs missing, clear-sky share and longest gap between clear dates, Landsat coverage, SL2P quality, energy-balance closure
(optional E2 components), ERA5-Land grid alignment. An S2_fAPAR_SL2P_mean column of the 16-day table replaces the
NDVI-based fAPAR.
Writes: units.csv, obs_ndvi.csv, obs_thermal.csv, weather_daily.csv, [period_inputs.csv, fapar_relation.csv],
provenance.csv, qc_checks.csv, manifest. Stops if a check FAILs.
"""
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core import config, sampling
from core import drivers as dv

WIDE = {"area_ha": "S2_potato_mask_area_ha_mean", "ndvi": "S2_NDVI_mean", "fapar": "S2_fAPAR_mean",
        "fapar_median": "S2_fAPAR_median", "ef_l8": "L8_EF_mean", "ef_l9": "L9_EF_mean",
        "rs": "ERA5_solar_MJ_m2_day_mean", "tmax": "ERA5_Tmax_C_mean", "tmin": "ERA5_Tmin_C_mean",
        "rn": "ERA5_Rn_MJ_m2_day_mean", "eta_l8": "L8_ETa_mm_day_mean", "eta_l9": "L9_ETa_mm_day_mean"}
SL2P_COL = "S2_fAPAR_SL2P_mean"                       # optional column of the 16-day table: SL2P fAPAR (used if present)
EB_COLS = ["le_w_m2", "rn_w_m2", "g_w_m2", "h_w_m2"]   # optional E2 energy-balance components at the overpass


def provenance(cfg, qc, inputs):
    """Fingerprint, size, file date and CRS of every input, compared with the registered sha256 of the run config,
    and the documented provenance fields (config provenance.required_fields). Returns the provenance table."""
    reg, prov = cfg["aoi"].get("input_sha256", {}), cfg["aoi"].get("provenance", {})
    req, und = cfg["provenance"]["required_fields"], cfg["provenance"]["undocumented_value"]
    rows, changed, missing_doc = [], [], []
    for key, p in inputs.items():
        p = Path(p)
        fp = config.fingerprint(p)
        crs = None
        if p.suffix.lower() in (".gpkg", ".shp"):
            try:
                import pyogrio
                info = pyogrio.read_info(p, layer=cfg["aoi"]["inputs"].get("crop_mask_layer")) if p.suffix.lower() == ".gpkg" \
                    else pyogrio.read_info(p)
                crs = info.get("crs")
            except Exception as e:                         # pragma: no cover - reported, not fatal
                crs = f"not read ({type(e).__name__})"
        r = {"input": key, "path": str(p), "sha256": fp, "registered_sha256": reg.get(key),
             "matches_registered": (fp == reg.get(key)) if reg.get(key) else None, "bytes": p.stat().st_size,
             "modified": datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
             "crs": crs}
        doc = prov.get(key)
        for f in req:
            v = None if doc is None else doc.get(f)
            r[f] = v if v is not None else und
            if doc is not None and (v is None or str(v).startswith(und)):
                missing_doc.append(f"{key}.{f}")
        if doc is None and key in reg:
            missing_doc.append(f"{key} (no provenance entry)")
        if reg.get(key) and fp != reg.get(key):
            changed.append(key)
        rows.append(r)
    unreg = [k for k in inputs if k in cfg["aoi"]["inputs"] and k not in reg]
    qc.add("inputs = registered versions (input_sha256 of the run config)", not changed and not unreg,
           (f"changed: {changed}; " if changed else "") + (f"not registered: {unreg}" if unreg else
                                                            f"{len(reg)} inputs match"), warn=True)
    qc.add("provenance documented (run config provenance)", not missing_doc,
           f"{len(missing_doc)} undocumented items: {missing_doc}" if missing_doc else "complete", warn=True)
    return pd.DataFrame(rows)


def crop_map_qc(cfg, qc, g):
    """Geometry, class, area, source, identifiers and overlaps of the crop-map parcels."""
    dq = cfg["data_qc"]
    qc.add("crop map: valid, non-empty geometries", bool(g.is_valid.all() and not g.is_empty.any()),
           f"{int((~g.is_valid).sum())} invalid, {int(g.is_empty.sum())} empty of {len(g):,}")
    if "crop" in g:
        qc.add(f"crop map: crop class = {dq['crop_class']}", bool((g.crop == dq["crop_class"]).all()),
               str(g.crop.value_counts().to_dict()))
    qc.add("crop map: parcel area > 0", bool((g.area_ha > 0).all()), f"{int((g.area_ha <= 0).sum())} parcels <= 0 ha")
    if "source" in g:
        unk = sorted(set(g.source) - set(dq["known_sources"]))
        qc.add("crop map: source classes known", not unk, f"unknown: {unk}" if unk else str(g.source.value_counts().to_dict()))
    if "prob" in g:
        qc.add("crop map: class probability within 0-1", bool(g.prob.dropna().between(0, 1).all()),
               f"{g.prob.min():.3f}-{g.prob.max():.3f}; {int(g.prob.isna().sum())} parcels without a probability")
    if "parcel_id" in g:
        dup = g[g.parcel_id.duplicated(keep=False)]
        dpar = int(g[g.source == "parcel"].parcel_id.duplicated().sum()) if "source" in g else int(g.parcel_id.duplicated().sum())
        qc.add("crop map: parcel_id unique among source = parcel (segments may share their parcel's id)", dpar == 0,
               f"{dpar} duplicated parcel ids among pure parcels; shared ids by source: "
               f"{dup.source.value_counts().to_dict() if 'source' in g else len(dup)}; the framework key is the row "
               "(field_id)", warn=True)
    import geopandas as gpd
    u = g[["geometry"]].to_crs(g.estimate_utm_crs()).reset_index(drop=True)
    u["i"] = np.arange(len(u))
    j = gpd.sjoin(u, u, predicate="intersects")
    j = j[j.i_left < j.i_right]
    ov = np.array([u.geometry.iloc[a].intersection(u.geometry.iloc[b]).area for a, b in zip(j.i_left, j.i_right)])
    big = ov > dq["overlap_tolerance_m2"]
    qc.add(f"crop map: overlapping parcels (> {dq['overlap_tolerance_m2']} m2)", not big.any(),
           f"{int(big.sum())} pairs, {ov[big].sum() / 1e4:.1f} ha counted twice "
           f"({100 * ov[big].sum() / 1e4 / g.area_ha.sum():.2f} % of the mapped area)", warn=True)


def date_qc(cfg, qc, obs, name, value_col, rng_key):
    """Per-date observations (unit, date, value, valid_fraction): duplicates, ranges, clear-sky share and the longest
    gap between clear dates in the season window."""
    dq, cl, sc = cfg["data_qc"], cfg["ndvi_cleaning"], cfg["season"]
    dup = int(obs.duplicated(["unit", "date"]).sum())
    qc.add(f"{name}: no duplicate unit-dates", dup == 0, f"{dup} duplicates")
    lo, hi = dq["ranges"][rng_key]
    v = obs[value_col]
    qc.add(f"{name}: {value_col} within {lo}-{hi}", bool(v.dropna().between(lo, hi).all()),
           f"{int((~v.dropna().between(lo, hi)).sum())} outside")
    vf = obs.valid_fraction
    qc.add(f"{name}: valid_fraction within 0-1", bool(vf.dropna().between(0, 1).all()), "")
    w0, w1 = pd.Timestamp(sc["window"][0]), pd.Timestamp(sc["window"][1])
    clear = obs[(vf >= cl["min_valid_fraction"]) & v.notna() & (obs.date >= w0) & (obs.date <= w1)]
    gaps = clear.sort_values("date").groupby("unit").date.apply(lambda d: d.diff().dt.days.max() if len(d) > 1 else np.nan)
    n_clear = clear.groupby("unit").size()
    worst = gaps.max()
    qc.add(f"{name}: longest gap between clear dates in the window <= {dq['max_s2_gap_days']} d",
           bool((gaps <= dq["max_s2_gap_days"]).all()),
           f"clear dates per unit {n_clear.min() if len(n_clear) else 0}-{n_clear.max() if len(n_clear) else 0} "
           f"(share of dates clear {100 * (vf >= cl['min_valid_fraction']).mean():.0f} %); longest gap {worst:.0f} d"
           + (f" ({gaps.idxmax()})" if len(gaps.dropna()) else ""), warn=True)


def energy_balance_qc(cfg, qc, e2):
    """Optional E2 components at the overpass: EF = LE / (Rn - G) and closure Rn - G - H - LE."""
    if not set(EB_COLS) <= set(e2):
        qc.add("E2 energy-balance components (le, rn, g, h)", False, "not delivered by the EF function of step 00c - "
               "EF cannot be checked against LE / (Rn - G)", warn=True)
        return
    eb = cfg["energy_balance"]
    x = e2.dropna(subset=EB_COLS + ["ef"])
    ef_c = (x.le_w_m2 / (x.rn_w_m2 - x.g_w_m2)).clip(0, 1)
    d = (x.ef - ef_c).abs()
    qc.add("E2: field EF (mean of pixel EF) vs LE / (Rn - G) of the field-mean fluxes", True,
           f"{len(x)} overpasses, median |diff| {d.median():.4f}, max {d.max():.4f} (differ by the order of averaging "
           f"and the 0-1 bound of pixel EF; {int((d > eb['ef_tolerance']).sum())} above {eb['ef_tolerance']})",
           info=True)
    res = (x.rn_w_m2 - x.g_w_m2 - x.h_w_m2 - x.le_w_m2).abs()
    qc.add(f"E2: energy balance closes within {eb['closure_tolerance_w_m2']} W m-2",
           bool((res <= eb["closure_tolerance_w_m2"]).all()), f"max residual {res.max():.1f} W m-2", warn=True)


def grid_alignment(cfg, qc, cells):
    """ERA5-Land cell ids (lat_lon) must be grid points of the native 0.1 deg grid."""
    gd = cfg["era5_land_grid"]
    ll = np.array([[float(v) for v in str(c).split("_")] for c in cells])
    off = np.abs((ll - gd["origin_deg"]) / gd["step_deg"] - np.round((ll - gd["origin_deg"]) / gd["step_deg"]))
    qc.add(f"ERA5-Land cells on the native {gd['step_deg']} deg grid", bool((off < 1e-6).all()),
           f"{len(ll)} cells, max offset {off.max() * gd['step_deg']:.2e} deg")


class QC:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail, warn=False, info=False):
        status = "INFO" if info else ("PASS" if ok else ("WARN" if warn else "FAIL"))
        self.rows.append({"check": name, "status": status, "detail": detail})

    def table(self):
        return pd.DataFrame(self.rows)


def weather_qc(cfg, qc, w, name):
    """Daily weather: duplicate unit-days, physical ranges, Tmin <= Tmax."""
    rg = cfg["data_qc"]["ranges"]
    dup = int(w.duplicated(["unit", "date"]).sum())
    qc.add(f"{name}: no duplicate unit-days", dup == 0, f"{dup} duplicates")
    out = {c: int((~w[c].dropna().between(*rg[r])).sum()) for c, r in (("rs", "rs_mj_m2"), ("tmin", "t_c"), ("tmax", "t_c"))}
    qc.add(f"{name}: Rs {rg['rs_mj_m2']} MJ m-2 d-1, T {rg['t_c']} degC", sum(out.values()) == 0, str(out))
    qc.add(f"{name}: Tmin <= Tmax", bool((w.tmin <= w.tmax).all()), f"{int((w.tmin > w.tmax).sum())} days with Tmin > Tmax")


def later_inputs(cfg):
    """Inputs read by later steps (boundaries; optional: comparison model, observed calendar): registered here."""
    out = {"boundaries": config.input_path(cfg, "boundaries")}
    if config.optional_input(cfg, "aquacrop_tehsil") is not None:
        out["aquacrop_tehsil"] = config.optional_input(cfg, "aquacrop_tehsil")
    if cfg["aoi"]["inputs"].get("observed_calendar"):
        out["observed_calendar"] = config.input_path(cfg, "observed_calendar")
    return out


def period_table(cfg, qc):
    """Tidy 16-day table (tehsil x period) from the wide table. EF / ETa of exactly 0 -> missing (no-data)."""
    inp = cfg["aoi"]["inputs"]
    p = config.input_path(cfg, "period_table")
    d = pd.read_csv(p, parse_dates=["period", "period_end"])
    miss = [c for c in list(WIDE.values()) + ["tehsil", "period", "period_end", "L8_EF_median", "L9_EF_median"]
            if c not in d]
    if miss:
        raise SystemExit(f"16-day table {p.name}: missing columns {miss}")
    t = pd.DataFrame({"tehsil": d.tehsil, "period": d.period, "period_end": d.period_end,
                      **{k: d[v] for k, v in WIDE.items()},
                      "ef_l8_median": d.L8_EF_median, "ef_l9_median": d.L9_EF_median})
    if SL2P_COL in d:                                   # SL2P fAPAR (method document) replaces the NDVI-based value
        t["fapar_ndvi_based"] = t.fapar
        t["fapar"] = d[SL2P_COL]
        qc.add("fAPAR source", True, f"SL2P ({SL2P_COL}); NDVI-based value kept as fapar_ndvi_based", info=True)
    else:
        qc.add("fAPAR source", False, f"NDVI-based value ({WIDE['fapar']}); SL2P column {SL2P_COL} not in the "
               "table - background fAPAR from soil counts as growth", warn=True)
    dup = int(d.duplicated(["tehsil", "period"]).sum())
    qc.add("16-day table: no duplicate tehsil-periods", dup == 0, f"{dup} duplicates")
    rg = cfg["data_qc"]["ranges"]
    for c, r in (("ndvi", "ndvi"), ("fapar", "fapar"), ("rs", "rs_mj_m2"), ("tmax", "t_c"), ("tmin", "t_c")):
        n = int(((t[c] < rg[r][0]) | (t[c] > rg[r][1])).sum())
        qc.add(f"16-day table {c} within {rg[r]}", n == 0, f"{n} values outside")
    zc = {c: int((t[c] == 0).sum()) for c in ("ndvi", "fapar", "rs", "eta_l8", "eta_l9", "ef_l8", "ef_l9")}
    qc.add("exact zeros per column (before the no-data rule)", True, f"{zc}; {cfg['data_qc']['zero_rule']}", info=True)
    zeros = []
    for c in ("ef_l8", "ef_l9", "eta_l8", "eta_l9", "ef_l8_median", "ef_l9_median"):
        if c.split("_")[0] in cfg["curve_fit"]["zero_as_missing"]:
            z = t[c] == 0
            zeros += [f"{r.tehsil} {r.period:%d %b} {c}" for r in t[z].itertuples()]
            t.loc[z, c] = np.nan
    qc.add("EF / ETa of exactly 0 set to missing (no-data value)", True,
           f"{len(zeros)}: {zeros}" if zeros else "none", info=True)
    t["mid"] = dv.period_mid(t.period, t.period_end)
    t["ef"] = t[["ef_l8", "ef_l9"]].mean(axis=1)                   # mean of the available sensors
    t["ef_median"] = t[["ef_l8_median", "ef_l9_median"]].mean(axis=1)
    t["tmean"] = (t.tmax + t.tmin) / 2
    return t.sort_values(["tehsil", "period"]).reset_index(drop=True), p


def crop_map_area(cfg, qc, table_area):
    """Crop area per tehsil from the classified crop map (attribute area_ha of the parcels), checked against the
    polygon geometry. The 16-day table area statistic has no documented derivation; it is reported, not used."""
    import geopandas as gpd
    inp = cfg["aoi"]["inputs"]
    p = config.input_path(cfg, "crop_mask")
    g = gpd.read_file(p, layer=inp["crop_mask_layer"])
    crop_map_qc(cfg, qc, g)
    geom = g.to_crs(g.estimate_utm_crs()).area / 1e4
    attr = g["area_ha"] if "area_ha" in g else geom
    qc.add("crop-map area attribute = polygon geometry area", abs(attr.sum() / geom.sum() - 1) < cfg["data_qc"]["area_attribute_tolerance"],
           f"{attr.sum():,.1f} ha (attribute) vs {geom.sum():,.1f} ha (geometry), {len(g):,} parcels")
    a = attr.groupby(g[inp["crop_mask_tehsil_field"]]).sum()
    miss = sorted(set(table_area.index) - set(a.index))
    if miss:
        raise SystemExit(f"tehsils without crop-map parcels: {miss}")
    ratio = table_area / a.reindex(table_area.index)
    qc.add("16-day table area vs crop-map area (information)", True,
           f"table / map {ratio.min():.2f}-{ratio.max():.2f} (table {table_area.sum():,.0f} ha, map {a.sum():,.0f} ha); "
           "the table statistic is undocumented - weights and production use the crop-map area", info=True)
    if "source" in g:
        sh = attr.groupby(g["source"]).sum() / attr.sum()
        qc.add("crop-map parcel sources (information)", True,
               "; ".join(f"{k} {100 * v:.0f} %" for k, v in sh.sort_values(ascending=False).items()), info=True)
    return a.reindex(table_area.index), p


def tehsil_units(cfg, qc):
    aoi, sc = cfg["aoi"], cfg["season"]
    m, p_period = period_table(cfg, qc)
    tehsils = sorted(m.tehsil.unique())
    qc.add("tehsils = expected", tehsils == sorted(aoi["expected_tehsils"]), f"{len(tehsils)}: {tehsils}")
    grid = m.groupby("tehsil").period.nunique()
    qc.add("complete tehsil x period grid, no duplicates",
           (not m.duplicated(["tehsil", "period"]).any()) and grid.nunique() == 1,
           f"{len(m)} rows, {grid.iloc[0]} periods {m.period.min():%d %b %Y} - {m.period_end.max():%d %b %Y}")
    contiguous = all((g.period.iloc[1:].to_numpy() == g.period_end.iloc[:-1].to_numpy()).all()
                     for _, g in m.groupby("tehsil"))
    qc.add("periods contiguous (next period starts on the previous end date)", contiguous,
           f"lengths {sorted(((m.period_end - m.period).dt.days).unique().tolist())} days; treated as [start, end)")
    cols = ["area_ha", "ndvi", "fapar", "rs", "tmax", "tmin"]
    qc.add("no missing values in S2 and ERA5 model columns", int(m[cols].isna().sum().sum()) == 0,
           f"{int(m[cols].isna().sum().sum())} missing")
    nmiss = {c: int(m[c].isna().sum()) for c in ("ef_l8", "ef_l9", "eta_l8", "eta_l9")}
    qc.add("Landsat periods without a clear overpass (filled by the daily curve)", True, str(nmiss), info=True)
    nmin = cfg["curve_fit"]["ef_eta"]["min_obs"]
    few = {f"{u} {s}": int(g[f"ef_{s}"].notna().sum()) for u, g in m.groupby("tehsil") for s in ("l8", "l9")
           if g[f"ef_{s}"].notna().sum() < nmin}
    qc.add(f"each tehsil and Landsat sensor has >= {nmin} periods with EF", not few, str(few) if few else "all", warn=True)
    rng = {k: tuple(v) for k, v in cfg["data_qc"]["period_ranges"].items()}      # settings
    bad = {c: int(((m[c] < lo) | (m[c] > hi)).sum()) for c, (lo, hi) in rng.items()}
    qc.add("physical ranges", sum(bad.values()) == 0, str(bad))
    qc.add("Tmin < Tmax", bool((m.tmin < m.tmax).all()), "16-day means")
    table_area = m.groupby("tehsil").area_ha.first()
    qc.add("16-day table area constant per tehsil", m.groupby("tehsil").area_ha.agg(np.ptp).max() < 1e-6, "")
    area, p_mask = crop_map_area(cfg, qc, table_area)            # v0.3.2: weights and production from the crop map
    ref = aoi["benchmark"].get("area_ha")
    qc.add("crop-map area vs official area", ref is None or abs(area.sum() / ref - 1) < cfg["data_qc"]["official_area_tolerance"],
           f"crop map {area.sum():,.0f} ha vs {ref:,} ha ({aoi['benchmark'].get('area_source')}): ratio "
           f"{area.sum() / ref:.2f} - affects production only, not t/ha", warn=True)
    x, y = m.ndvi.to_numpy(), (m.fapar_ndvi_based if "fapar_ndvi_based" in m else m.fapar).to_numpy()
    slope, icpt = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    ba = cfg["fapar"]["bastiaanssen_ali_2003"]
    ba_pred = ba["slope"] * x + ba["intercept"]
    rel = pd.DataFrame([{"relation": "fitted_on_input", "slope": slope, "intercept": icpt, "r2": r2,
                         "rmse": float(np.sqrt(np.mean((slope * x + icpt - y) ** 2))), "n": len(x)},
                        {"relation": "bastiaanssen_ali_2003", "slope": ba["slope"], "intercept": ba["intercept"],
                         "r2": np.corrcoef(ba_pred, y)[0, 1] ** 2, "rmse": float(np.sqrt(np.mean((ba_pred - y) ** 2))),
                         "n": len(x)}])
    qc.add("input NDVI-based fAPAR is a function of NDVI", r2 >= cfg["fapar"]["min_r2"],
           f"fAPAR = {slope:.3f} NDVI {icpt:+.3f}, R2 {r2:.4f}; RMSE of 1.257 NDVI - 0.161 (Bastiaanssen & Ali 2003 eq. 4) "
           f"{rel.rmse.iloc[1]:.4f}")
    pair = m.ef_l8.notna() & m.ef_l9.notna()
    qc.add("Landsat 8 vs 9 EF agreement",
           (m.ef_l8 - m.ef_l9)[pair].abs().mean() <= cfg["quality_flags"]["max_ef_l8_l9_abs_diff"],
           f"{int(pair.sum())} periods with both: r {m.ef_l8.corr(m.ef_l9):.2f}, mean |L8-L9| "
           f"{(m.ef_l8 - m.ef_l9)[pair].abs().mean():.3f}", warn=True)
    eta = m[["eta_l8", "eta_l9"]].mean(axis=1)
    ok = eta.notna() & m.ef.notna()
    est = m.ef * m.rn / cfg["sebal"]["lambda_mj_kg"]                       # lambda: FAO-56
    qc.add("ETa vs EF*Rn/lambda", True, f"r {np.corrcoef(est[ok], eta[ok])[0, 1]:.2f}: ETa is derived "
           "from EF - not an independent check; used only through ETa/ETc in the relative-ET water scalar", info=True)

    units = pd.DataFrame({"unit": area.index, "tehsil": area.index, "weight_ha": area.to_numpy()})

    p_nd = config.input_path(cfg, "ndvi_perdate")
    nd = pd.read_csv(p_nd)
    nd["date"] = pd.to_datetime(nd["date"], format="%m/%d/%Y")
    obs = pd.DataFrame({"unit": nd.SUB_DIST, "date": nd.date, "ndvi": nd["mean"],
                        "valid_fraction": nd["count"] / nd["potato_px"], "n_scenes": nd.n_scenes})
    units = units.merge(nd.groupby("SUB_DIST").SUBDIS_LGD.first().rename("lgd"), left_on="unit", right_index=True)
    qc.add("per-date NDVI tehsils = expected", sorted(obs.unit.unique()) == sorted(aoi["expected_tehsils"]),
           f"{obs.unit.nunique()} tehsils, {len(obs)} tehsil-dates, {obs.date.min():%Y-%m-%d} to {obs.date.max():%Y-%m-%d}")
    date_qc(cfg, qc, obs, "per-date NDVI", "ndvi", "ndvi")

    thermal = []
    for s in ("l8", "l9"):                      # 16-day means placed at mid-period; period bounds kept for ETc
        thermal.append(pd.DataFrame({"unit": m.tehsil, "date": m.mid, "sensor": s.upper(), "ef": m[f"ef_{s}"],
                                     "eta_mm_day": m[f"eta_{s}"], "valid_fraction": np.nan, "representative": "period",
                                     "period": m.period, "period_end": m.period_end}))
    thermal = pd.concat(thermal, ignore_index=True)

    p_w = config.input_path(cfg, "weather_daily")
    w = pd.read_excel(p_w, sheet_name=aoi["inputs"]["weather_sheet"])
    w["date"] = pd.to_datetime(w["date"])
    w0, w1 = pd.Timestamp(sc["window"][0]), pd.Timestamp(sc["window"][1])
    weather = pd.DataFrame({"unit": w.tehsil, "date": w.date, "rs": w.Rs_MJ, "tmin": w.Tmin_C, "tmax": w.Tmax_C,
                            "eto": w.ETo_mm, "lat": w.lat, "lon": w.lon})
    ws_ = weather[(weather.date >= w0) & (weather.date <= w1)]
    n_days = (w1 - w0).days + 1
    weather_qc(cfg, qc, ws_, "daily weather")
    qc.add("daily weather complete in the season window",
           ws_.groupby("unit").date.nunique().eq(n_days).all() and not ws_[["rs", "tmin", "tmax", "eto"]].isna().any().any(),
           f"{ws_.unit.nunique()} tehsils x {n_days} days, Rs, Tmin, Tmax, ETo (FAO-56)")
    js = []
    for u, g in ws_.groupby("unit"):
        mt = m[m.tehsil == u].reset_index(drop=True)
        k = dv.period_of_day(g.date, mt.period, mt.period_end)
        a = g.assign(k=k, tm=(g.tmin + g.tmax) / 2)
        a = a[a.k >= 0].groupby("k").agg(rs_d=("rs", "mean"), tm_d=("tm", "mean"), n=("rs", "size"))
        js.append(mt.loc[a.index].assign(rs_d=a.rs_d.to_numpy(), tm_d=a.tm_d.to_numpy(), n_days=a.n.to_numpy()))
    j = pd.concat(js)
    qc.add("daily Rs vs 16-day table", ((j.rs / j.rs_d) - 1).abs().max() <= cfg["data_qc"]["daily_rs_vs_table_max_rel"],
           f"ratio {(j.rs / j.rs_d).min():.3f}-{(j.rs / j.rs_d).max():.3f} over {len(j)} tehsil-periods with daily "
           f"weather (scaling applied in step 04)", warn=True)
    qc.add("daily Tmean vs 16-day table", (j.tmean - j.tm_d).abs().max() <= cfg["data_qc"]["daily_tmean_vs_table_max_c"],
           f"delta {(j.tmean - j.tm_d).min():+.2f} to {(j.tmean - j.tm_d).max():+.2f} degC (shift applied in step 04)",
           warn=True)
    after = m[m.period > w1].period.dt.strftime("%d %b").unique().tolist()
    qc.add("16-day periods after the season window", True, f"{after}: used only to support the curve fits "
           "(no daily weather after the window end)", info=True)
    inputs = {"period_table": p_period, "ndvi_perdate": p_nd, "weather_daily": p_w, "crop_mask": p_mask,
              **later_inputs(cfg)}
    return units, obs, thermal, weather, m, rel, inputs, "original"


def field_units(cfg, qc):
    fx = cfg["aoi"]["field_exports"]
    p_f = config.step_output(cfg, "02_fields", "fields.csv")
    f = pd.read_csv(p_f)
    if "design_weight_ha" not in f or (f.get("tehsil") == "not_assigned").all():
        raise SystemExit(f"{p_f}: these units come from step 00b (parts of an area polygon inside each ERA5-Land cell, "
                         "all land cover) - they have no crop area, design weight or tehsil. A yield run (steps 01-10) "
                         "needs the crop-map field sample of step 02: register inputs.crop_mask, crop_mask_layer, "
                         "crop_mask_tehsil_field, boundaries and boundary_district in the run config and run step 02. "
                         "Step 00b units serve the data download and its checks only (steps 00c, 00d).")
    cell_area = f.groupby("cell").design_weight_ha.sum()           # = mapped crop area of the cell
    units = pd.DataFrame({"unit": f.field_id, "tehsil": f.tehsil, "cell": f.cell, "area_ha": f.area_ha,
                          "weight_ha": sampling.ratio_weights(f.cell, f.area_ha, cell_area), "lon": f.lon, "lat": f.lat})
    qc.add("field weights: separate ratio estimator (cell area x parcel area / sampled area of the cell)",
           abs(units.weight_ha.sum() - cell_area.sum()) < 1e-6, f"{units.weight_ha.sum():,.1f} ha in {len(cell_area)} cells")
    paths = {k: config.path(v) for k, v in fx.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("field exports missing (run 00c_download.py; README 'Field exports'):\n  " + "\n  ".join(missing))
    e1 = pd.read_csv(paths["e1_s2_perdate"], parse_dates=["date"])
    e2 = pd.read_csv(paths["e2_landsat_perdate"], parse_dates=["date"])
    e3 = pd.read_csv(paths["e3_weather_daily"], parse_dates=["date"])
    e3, computed = dv.weather_eto(e3)                       # raw ERA5-Land variables -> FAO-56 ETo
    if computed:
        qc.add("E3 ETo computed (FAO-56 Penman-Monteith)", True, "from ERA5-Land Tmin, Tmax, Tdew, Rs, wind, pressure",
               info=True)
    for name, t, cols in (("E1", e1, ["field_id", "date", "ndvi", "valid_fraction"]),
                          ("E2", e2, ["field_id", "date", "sensor", "ef", "eta_mm_day", "valid_fraction"]),
                          ("E3", e3, ["cell", "date", "rs_mj_m2", "tmin_c", "tmax_c", "eto_mm"])):
        miss = [c for c in cols if c not in t]
        qc.add(f"{name} columns", not miss, f"missing {miss}" if miss else f"{len(t)} rows")
    for name, t in (("E1", e1), ("E2", e2)):         # a share is 0-1: sums of pixel weights can round past it
        n, top = dv.clip_shares(t)                     # floating-point rounding only; larger excess stays and FAILs
        qc.add(f"{name}: clear shares within 1e-9 of 0 or 1 set to the bound (floating-point rounding)", True,
               f"{n} rows; largest {top:.17g}", info=True)
    qc.add("E1 covers every sampled field", set(units.unit) <= set(e1.field_id),
           f"{len(set(units.unit) - set(e1.field_id))} fields without S2 data (excluded in step 03)", warn=True)
    qc.add("E2 covers every sampled field", set(units.unit) <= set(e2.field_id),
           f"{len(set(units.unit) - set(e2.field_id))} fields without Landsat data (excluded in step 03)", warn=True)
    qc.add("E3 covers every sampled cell", set(units.cell) <= set(e3.cell),
           f"{len(set(units.cell) - set(e3.cell))} cells without weather")
    import geopandas as gpd
    p_cm = config.input_path(cfg, "crop_mask")
    crop_map_qc(cfg, qc, gpd.read_file(p_cm, layer=cfg["aoi"]["inputs"]["crop_mask_layer"]))
    obs, thermal, weather, (n_low, n_ef) = dv.field_tables(cfg, units, e1, e2, e3)   # same tables as the check of step 05b
    date_qc(cfg, qc, obs, "E1 Sentinel-2", "ndvi", "ndvi")
    if "fapar_sl2p" in e1:
        obs["fapar_sl2p"] = e1.fapar_sl2p
        n_out = int((~e1.fapar_sl2p.dropna().between(*cfg["data_qc"]["ranges"]["fapar"])).sum())
        qc.add("E1 fapar_sl2p within 0-1", n_out == 0, f"{n_out} outside")
        if "fapar_sl2p_valid_fraction" in e1:          # share of the field's 20 m pixels with a valid SL2P retrieval
            bad = e1.fapar_sl2p_valid_fraction.fillna(0) < cfg["ndvi_cleaning"]["min_valid_fraction"]
            obs.loc[bad, "fapar_sl2p"] = np.nan
            qc.add("E1 SL2P retrievals with valid share below the minimum set to missing", True,
                   f"{int(bad.sum())} of {len(e1)} dates", info=True)
        qc.add("fAPAR source", True, "SL2P (E1 fapar_sl2p)", info=True)
    else:
        qc.add("fAPAR source", False, "E1 without fapar_sl2p: fAPAR = 1.257 NDVI - 0.161 (Bastiaanssen & Ali 2003 eq. 4); "
               "background fAPAR from soil counts as growth", warn=True)
    rg = cfg["data_qc"]["ranges"]
    dup2 = int(e2.duplicated(["field_id", "date", "sensor"]).sum())
    qc.add("E2: no duplicate field-date-sensor rows", dup2 == 0, f"{dup2} duplicates")
    sens = set(e2.sensor.str.upper())
    qc.add("E2: sensor is L8 or L9", sens <= {"L8", "L9"}, str(sorted(sens)))
    for c, r in (("ef", "ef"), ("eta_mm_day", "eta_mm_day"), ("valid_fraction", "valid_fraction")):
        n_out = int((~e2[c].dropna().between(*rg[r])).sum())
        qc.add(f"E2: {c} within {rg[r]}", n_out == 0, f"{n_out} outside")
    energy_balance_qc(cfg, qc, e2)
    dup3 = int(e3.duplicated(["cell", "date"]).sum())
    qc.add("E3: no duplicate cell-dates", dup3 == 0, f"{dup3} duplicates")
    w0, w1 = pd.Timestamp(cfg["season"]["window"][0]), pd.Timestamp(cfg["season"]["window"][1])
    n_days = (w1 - w0).days + 1
    cover = e3[(e3.date >= w0) & (e3.date <= w1)].groupby("cell").date.nunique()
    qc.add("E3: every day of the window for every cell", bool((cover.reindex(units.cell.unique()) == n_days).all()),
           f"{int((cover.reindex(units.cell.unique()).fillna(0) < n_days).sum())} cells incomplete ({n_days} days)")
    weather_qc(cfg, qc, e3.rename(columns={"cell": "unit", "rs_mj_m2": "rs", "tmin_c": "tmin", "tmax_c": "tmax"}), "E3")
    grid_alignment(cfg, qc, units.cell.unique())
    qc.add(f"E2 overpasses with clear share < {cfg['ef']['min_valid_fraction']}: EF and ETa set to missing", True,
           f"{n_low} of {n_ef} overpasses with EF", info=True)
    cov = thermal[thermal.ef.notna()].groupby(["unit", "sensor"]).size().unstack(fill_value=0).reindex(units.unit).fillna(0)
    qc.add("Landsat coverage: clear overpasses with EF per field", True,
           "; ".join(f"{s} median {cov[s].median():.0f} (min {cov[s].min():.0f})" for s in cov.columns), info=True)
    return units, obs, thermal, weather, None, None, {"fields": p_f, **paths, "crop_mask": p_cm, **later_inputs(cfg)}, \
        "field_exports"


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    out = config.output_dir(cfg, "01_inputs")
    config.check_writable(out)
    qc = QC()
    mode = config.units(cfg)
    units, obs, thermal, weather, m, rel, inputs, source = (tehsil_units if mode == "tehsil" else field_units)(cfg, qc)
    units.to_csv(out / "units.csv", index=False)
    obs.to_csv(out / "obs_ndvi.csv", index=False, date_format="%Y-%m-%d")
    thermal.to_csv(out / "obs_thermal.csv", index=False)      # full timestamp: mid-month can fall at noon
    weather.to_csv(out / "weather_daily.csv", index=False, date_format="%Y-%m-%d")
    if m is not None:
        m.to_csv(out / "period_inputs.csv", index=False, date_format="%Y-%m-%d %H:%M")
        rel.to_csv(out / "fapar_relation.csv", index=False)
    prov = provenance(cfg, qc, inputs)
    prov.to_csv(out / "provenance.csv", index=False)
    t = qc.table()
    t.to_csv(out / "qc_checks.csv", index=False)
    # a step with a FAIL check leaves no manifest: run.py then never treats it as complete
    if not (t.status == "FAIL").any():
        config.write_manifest(out, "01_inputs", inputs, {"folder": out}, {"units": mode, "period_table_source": source})
    pd.set_option("display.width", 250, "display.max_colwidth", 150)
    print(f"run {cfg['run_id']}: units {mode} ({len(units)}), inputs: {source}")
    print(t.to_string(index=False))
    print(f"\nwritten to {out}")
    if (t.status == "FAIL").any():
        raise SystemExit("step 01 stopped: FAIL checks above")


if __name__ == "__main__":
    main()
