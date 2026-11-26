"""Evaporation from a bare soil by the daily water balance of the surface layer - FAO-56 (Allen et al. 1998) chapter 7,
"Evaporation component" (https://www.fao.org/4/x0490e/x0490e0c.htm). Used for the ET of the SEBAL hot pixel, as
METRIC does: "a daily surface soil water balance such as that of the FAO56 publication ... is run for a bare soil
condition to confirm that ET = 0 or to supply a nonzero value for ET if there is residual evaporation ... from
antecedent precipitation" (Allen, Tasumi & Trezza 2007, p.387).

  TEW = 1000 (theta_FC - 0.5 theta_WP) Ze                       eq. 73 (mm)
  Kr  = 1 while De,i-1 <= REW; (TEW - De,i-1) / (TEW - REW) after   eq. 74 (stage 1 / stage 2 drying)
  Ke  = min(Kr (Kc_max - Kcb), few Kc_max)                      eq. 71
  E   = Ke ETo                                                  (mm d-1)
  De,i = De,i-1 - (P - RO) - I / fw + E / few + Tew + DPe, 0 <= De <= TEW   eqs. 77-79
Bare soil: Kcb = 0, so that Ke max = Kc max = 1.20 (FAO-56 Example 31 note); few = 1 for rain on bare soil
(Example 31); Tew = 0 (no transpiration); RO = 0 ("For general situations, ROi can be assumed to be zero");
I = 0 (rain only - irrigation of the bare field is not known: setting); De = 0 at the start ("the topsoil is near
field capacity following a heavy rain", start of the series in the monsoon: setting).
"""
import numpy as np
import pandas as pd


def tew(theta_fc, theta_wp, ze_m):
    """Total evaporable water, mm (eq. 73); theta in m3 m-3."""
    return 1000.0 * (theta_fc - 0.5 * theta_wp) * ze_m


def kr(de_prev, tew_mm, rew_mm):
    """Evaporation reduction coefficient (eq. 74): 1 in stage 1 (De <= REW), falling to 0 at De = TEW."""
    if de_prev <= rew_mm:
        return 1.0
    return float(np.clip((tew_mm - de_prev) / (tew_mm - rew_mm), 0.0, 1.0))


def bare_soil(dates, precip_mm, eto_mm, tew_mm, rew_mm, kc_max=1.2, kcb=0.0, few=1.0, de_start=0.0):
    """Daily water balance of the evaporation layer of a bare soil. Returns a table per day: de_start_mm (De,i-1),
    kr, ke, e_mm (evaporation), dp_mm (deep percolation), de_end_mm (De,i)."""
    if not 0 < rew_mm < tew_mm:
        raise ValueError(f"REW {rew_mm} must be between 0 and TEW {tew_mm}")
    rows, de = [], float(de_start)
    for d, p, eto in zip(dates, precip_mm, eto_mm):
        k_r = kr(de, tew_mm, rew_mm)
        ke = min(k_r * (kc_max - kcb), few * kc_max)                       # eq. 71
        e = ke * eto
        de_new = de - p + e / few                                          # eq. 77 (RO = 0, I = 0, Tew = 0)
        dp = max(0.0, -de_new)                                             # eq. 79: water above field capacity drains
        de_new = min(max(de_new, 0.0), tew_mm)                             # eq. 78
        rows.append({"date": d, "precip_mm": p, "eto_mm": eto, "de_start_mm": de, "kr": k_r, "ke": ke, "e_mm": e,
                     "dp_mm": dp, "de_end_mm": de_new})
        de = de_new
    return pd.DataFrame(rows)


def load_soil_units(cfg):
    """Soil of the surface layer per tehsil (run config inputs.soil_units: SoilGrids 2.0 summary workbook, sheet
    soil_units_sheet, column soil_units_field = tehsil) joined to the tehsil polygons (inputs.boundaries,
    boundary_district). theta_FC / theta_WP = water content at 33 / 1500 kPa (SoilGrids wv0033 / wv1500, vol %), mean of
    the 0-5 and 5-15 cm layers (= 0-10 cm, the evaporation layer Ze = 0.10 m); TEW by eq. 73.
    Returns a GeoDataFrame (EPSG:4326): unit, theta_fc, theta_wp, tew_mm, geometry."""
    import geopandas as gpd
    from core import config
    inp, hp = cfg["aoi"]["inputs"], cfg["sebal"]["hot_pixel"]
    for k in ("soil_units", "boundaries", "boundary_district"):
        if not inp.get(k):
            raise SystemExit(f"SEBAL hot pixel by soil water balance needs inputs.{k} in the run config "
                             "(or set sebal.hot_pixel.method: zero in config/project.yaml)")
    s = pd.read_excel(config.input_path(cfg, "soil_units"), sheet_name=inp.get("soil_units_sheet", "Mean_Wide"))
    col = lambda p, d: f"{p}_{d} (vol %)"  # noqa: E731
    s["theta_fc"] = (s[col("wv0033", "0-5cm")] + s[col("wv0033", "5-15cm")]) / 200.0
    s["theta_wp"] = (s[col("wv1500", "0-5cm")] + s[col("wv1500", "5-15cm")]) / 200.0
    s["tew_mm"] = tew(s.theta_fc, s.theta_wp, hp["ze_m"])
    s["unit"] = s[inp.get("soil_units_field", "tehsil")].astype(str)
    b = gpd.read_file(config.input_path(cfg, "boundaries"))
    b = b[b.DISTRICT == inp["boundary_district"]].to_crs(4326)
    g = b.merge(s[["unit", "theta_fc", "theta_wp", "tew_mm"]], left_on="SUB_DIST", right_on="unit", how="inner")
    if len(g) != len(b):
        raise SystemExit(f"soil units: {len(b) - len(g)} tehsils of the boundaries without soil values "
                         f"({sorted(set(b.SUB_DIST) - set(g.unit))})")
    if not (g.tew_mm > hp["rew_mm"]).all():
        raise SystemExit("soil units: TEW must exceed REW in every tehsil")
    return g[["unit", "theta_fc", "theta_wp", "tew_mm", "geometry"]].reset_index(drop=True)
