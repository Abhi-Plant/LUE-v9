"""Input parameters of a run (step 00a): the questions, the checks of the answers, the run config and the record of
every parameter. Used by 00a_parameters.py and run.py. No model value is set here."""
import datetime
import os
import re
import sys
from pathlib import Path

import pandas as pd

from core import config

ROOT = config.ROOT
# values offered in [brackets]: those of the Dewas 2025-26 run (config/aoi/dewas_2025_26.yaml)
DEWAS = {"window": ("2025-10-01", "2026-03-31"), "peak": ("2025-11-01", "2026-02-28"),
         "quality": ("2025-12-01", "2026-01-31"), "season": "rabi_2025_26", "utc_offset": "5.5"}

CONFIG = """# Run config written by step 00a (00a_parameters.py) on {today} for the area {aoi_file}.
# The units of this run are NOT crop fields but the parts of the area inside each ERA5-Land cell, all land cover
# (step 00b). No crop map or tehsil boundaries are registered: steps 01-10 cannot be run for it.
name: {name}
crop: {crop}                         # crop file config/crops/{crop}.yaml
season: {season}

# Season dates as answered in step 00a (the values in [brackets] there are those of the Dewas 2025-26 run)
season_dates:
  window: ["{w0}", "{w1}"]            # model days
  peak_search: ["{p0}", "{p1}"]       # canopy maximum searched here
  quality_window: ["{q0}", "{q1}"]    # clear Sentinel-2 dates counted here for the quality flag

calibration:
  input_file: {calibration}

# Local calendar day of the daily weather E3 (step 00c): UTC + this many hours
local_day_utc_offset_hours: {utc_offset}

inputs:
  boundary_file: "{aoi_file}"       # polygon file of the area, read in place
  era5_cell_deg: 0.1                # ERA5-Land grid

# Exports of step 00c (00c_download.py) for the step-00b units
field_exports:
  e1_s2_perdate: data/field_exports/{run}_E1_s2_perdate.csv          # field_id, date, ndvi, valid_fraction, n_scenes
  e2_landsat_perdate: data/field_exports/{run}_E2_landsat_perdate.csv # field_id, date, sensor, ef, eta_mm_day, valid_fraction
  e3_weather_daily: data/field_exports/{run}_E3_era5land_daily.csv    # cell, date, rs_mj_m2, tmin_c, tmax_c, tdew_c, u10_ms, sp_kpa, elev_m, lat
"""


# ---------------------------------------------------------------- checks of single answers
def a_date(s):
    try:
        return datetime.date.fromisoformat(s).isoformat()
    except ValueError:
        raise ValueError(f"'{s}' is not a date YYYY-MM-DD") from None


def a_file(s):
    p = Path(s.strip('"').strip("'")).expanduser()
    for q in ([p] if p.is_absolute() else [Path.cwd() / p, ROOT / p]):
        if q.is_file():
            return q.resolve()
    raise ValueError(f"file not found: {s}")


def read_aoi(p):
    """The polygon file as read; ValueError when it cannot serve as an area."""
    import geopandas as gpd
    try:
        g = gpd.read_file(p)
    except Exception as e:  # noqa: BLE001 - any read error is reported as the answer being wrong
        raise ValueError(f"{Path(p).name} cannot be read as a polygon file: {str(e)[:150]}") from None
    if g.empty:
        raise ValueError(f"{Path(p).name} has no feature")
    if g.crs is None:
        raise ValueError(f"{Path(p).name} has no coordinate reference system")
    if not g.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(f"{Path(p).name} needs polygons, has {g.geom_type.value_counts().to_dict()}")
    return g


def aoi_facts(g):
    b = g.to_crs(4326).total_bounds
    utm = g.estimate_utm_crs()
    return {"polygons": len(g), "crs": g.crs.to_string(), "area_ha": float(g.to_crs(utm).area.sum() / 1e4),
            "area_crs": utm.to_string(), "lon_min": float(b[0]), "lat_min": float(b[1]), "lon_max": float(b[2]),
            "lat_max": float(b[3])}


def an_aoi(s):
    p = a_file(s)
    f = aoi_facts(read_aoi(p))
    print(f"  {p.name}: {f['polygons']} polygon(s), {f['area_ha']:,.0f} ha, longitude {f['lon_min']:.4f} to "
          f"{f['lon_max']:.4f}, latitude {f['lat_min']:.4f} to {f['lat_max']:.4f}")
    return p


def a_name(s):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]*", s):
        raise ValueError("name: letters, digits, space, _ and - only, starting with a letter")
    return s


def a_run_id(s):
    s = s.lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", s):
        raise ValueError("run id: lower-case letters, digits and _ only")
    return s


def a_label(s):
    if not re.fullmatch(r"[A-Za-z0-9_]+", s):
        raise ValueError("season label: letters, digits and _ only")
    return s


def a_number(s):
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"'{s}' is not a number") from None


def crops_available():
    return sorted(p.stem for p in (ROOT / "config" / "crops").glob("*.yaml"))


def a_crop(s):
    s = s.strip().lower()
    if s not in crops_available():
        raise ValueError(f"no crop file config/crops/{s}.yaml (available: {', '.join(crops_available())})")
    return s


def a_count(s):
    if not s.isdigit() or int(s) < 1:
        raise ValueError(f"'{s}' is not a whole number of 1 or more")
    return int(s)


def exports(s):
    x = sorted({e.strip().lower() for e in s.split(",") if e.strip()})
    if not x or not set(x) <= {"e1", "e2", "e3", "px"}:
        raise ValueError("exports: e1, e2, e3, px separated by commas")
    return ",".join(x)


def ask(label, flag, given, check, default=None, optional=False):
    """(value, how it was given). The command-line value when given, otherwise the answer typed at the prompt
    (repeated until it is valid)."""
    if given is not None:
        try:
            return check(str(given)), f"command line {flag}"
        except ValueError as e:
            raise SystemExit(f"{flag}: {e}") from None
    hint = f" [{default}]" if default is not None else (" [none]" if optional else "")
    while True:
        try:
            s = input(f"{label}{hint}: ").replace("﻿", "").strip()
        except EOFError:
            raise SystemExit(f"\nno answer for '{label}' - give it on the command line: {flag}") from None
        how = "typed"
        if not s:
            if default is not None:
                s, how = str(default), "Enter: value in brackets (Dewas 2025-26 run or derived from the answers)"
            elif optional:
                return None, "Enter: none"
            else:
                print("  a value is needed")
                continue
        try:
            return check(s), how
        except ValueError as e:
            print(f"  {e}")


def inside(pair, window, what):
    if not pair[0] < pair[1]:
        raise SystemExit(f"{what}: {pair[0]} is not before {pair[1]}")
    if pair[0] < window[0] or pair[1] > window[1]:
        raise SystemExit(f"{what} {pair[0]} to {pair[1]} is not inside the season window {window[0]} to {window[1]}")


def same_area(p_cfg, aoi, w0, w1):
    """(same polygon, same window, stored boundary file, stored window) of an existing run config. The polygon is
    compared by content (sha256; shapefile: its parts), so the same file pasted from another folder is the same area."""
    old = config.read_yaml(p_cfg)
    old_bd = (old.get("inputs") or {}).get("boundary_file")
    old_win = [str(x) for x in (old.get("season_dates") or {}).get("window", [])]
    same_bd = False
    if old_bd is not None:
        q = Path(old_bd) if Path(old_bd).is_absolute() else ROOT / old_bd
        same_bd = q.resolve() == Path(aoi).resolve() or (q.is_file() and config.fingerprint(q) == config.fingerprint(aoi))
    return same_bd, old_win == [w0, w1], old_bd, old_win


def stored_path(p):
    """Path as written to the run config: relative to the framework folder when the file is inside it."""
    try:
        return Path(p).relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


# ---------------------------------------------------------------- command line and questions
def add_arguments(ap):
    ap.add_argument("--aoi", help="polygon file of the area (GeoJSON, GeoPackage, shapefile)")
    ap.add_argument("--name", help="name of the area, e.g. Guna")
    ap.add_argument("--window-start", help="season window start YYYY-MM-DD")
    ap.add_argument("--window-end", help="season window end YYYY-MM-DD")
    ap.add_argument("--run", help="run id = name of the run config and of the output folders, e.g. guna_2025_26; "
                    "alone, for a run config that exists: its parameters are checked and recorded, nothing is asked")
    ap.add_argument("--season", help="season label, e.g. rabi_2025_26")
    ap.add_argument("--crop", help="crop = name of a crop file in config/crops/, e.g. potato")
    ap.add_argument("--peak-start", help="peak search start YYYY-MM-DD")
    ap.add_argument("--peak-end", help="peak search end YYYY-MM-DD")
    ap.add_argument("--quality-start", help="quality window start YYYY-MM-DD")
    ap.add_argument("--quality-end", help="quality window end YYYY-MM-DD")
    ap.add_argument("--utc-offset", help="local time minus UTC in hours (India 5.5)")
    ap.add_argument("--calibration", help="file of benchmark or observed yields; 'none' when there is none")


def collect(a):
    """Asks what the command line does not give and writes the run config when it does not exist.
    Returns (run id, {parameter: how it was given})."""
    how = {}
    if a.run is not None and a.aoi is None:
        run = a_run_id(a.run)
        if (ROOT / "config" / "aoi" / f"{run}.yaml").exists():
            print(f"run config {run}.yaml exists: its parameters are used as they are (nothing is asked)")
            return run, how
    print("Enter takes the value in [brackets] - the values of the Dewas 2025-26 run, not values checked for "
          "your area.\n")
    aoi, how["aoi_file"] = ask("AOI polygon file (GeoJSON) - path", "--aoi", a.aoi, an_aoi)
    name, how["name"] = ask("Name of the area", "--name", a.name, a_name,
                            default=aoi.stem if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]*", aoi.stem) else None)
    w0, how["window_start"] = ask("Season window start (YYYY-MM-DD)", "--window-start", a.window_start, a_date,
                                  DEWAS["window"][0])
    w1, how["window_end"] = ask("Season window end (YYYY-MM-DD)", "--window-end", a.window_end, a_date,
                                DEWAS["window"][1])
    if not w0 < w1:
        raise SystemExit(f"season window: {w0} is not before {w1}")
    run, how["run_id"] = ask("Run id (names the run config and the output folders)", "--run", a.run, a_run_id,
                             a_run_id(re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + f"_{w0[:4]}_{w1[2:4]}"))
    p_cfg = ROOT / "config" / "aoi" / f"{run}.yaml"
    if p_cfg.exists():
        same_bd, same_win, old_bd, old_win = same_area(p_cfg, aoi, w0, w1)
        if same_bd and same_win:                     # same polygon (by content, wherever the file is) and window
            print(f"\nrun config {p_cfg.name} exists for this area and window: used as it is (polygon {aoi} has the "
                  f"same contents as its boundary file {old_bd}; its other values are not asked again)")
            return run, {k: v for k, v in how.items() if k in ("aoi_file", "window_start", "window_end", "run_id")}
        # another area or window: the existing run config and its outputs are kept; this one gets the next free id
        new = next(f"{run}_{i}" for i in range(2, 1000) if not (ROOT / "config" / "aoi" / f"{run}_{i}.yaml").exists())
        print(f"\nrun config {p_cfg.name} exists for {'another polygon' if not same_bd else 'the same polygon'} "
              f"({old_bd}) and window {old_win}; it is kept unchanged. This area ({aoi.name}, window {w0} to {w1}) "
              f"runs as {new}")
        run, how["run_id"] = new, f"{how['run_id']}; {p_cfg.name} exists for another area or window: next free id"
        p_cfg = ROOT / "config" / "aoi" / f"{run}.yaml"
    dflt = (w0, w1) == DEWAS["window"]
    season, how["season"] = ask("Season label", "--season", a.season, a_label, DEWAS["season"] if dflt else None)
    crop, how["crop"] = ask("Crop (a crop file in config/crops/: " + ", ".join(crops_available()) + ")", "--crop",
                            a.crop, a_crop, "potato" if "potato" in crops_available() else None)
    p0, how["peak_search_start"] = ask("Peak search start - canopy maximum is searched from", "--peak-start",
                                       a.peak_start, a_date, DEWAS["peak"][0] if dflt else None)
    p1, how["peak_search_end"] = ask("Peak search end", "--peak-end", a.peak_end, a_date,
                                     DEWAS["peak"][1] if dflt else None)
    inside((p0, p1), (w0, w1), "peak search")
    q0, how["quality_window_start"] = ask("Quality window start - clear Sentinel-2 dates are counted from",
                                          "--quality-start", a.quality_start, a_date,
                                          DEWAS["quality"][0] if dflt else None)
    q1, how["quality_window_end"] = ask("Quality window end", "--quality-end", a.quality_end, a_date,
                                        DEWAS["quality"][1] if dflt else None)
    inside((q0, q1), (w0, w1), "quality window")
    off, how["local_day_utc_offset_hours"] = ask(
        "Local time minus UTC in hours (the weather day is the local calendar day)", "--utc-offset", a.utc_offset,
        a_number, DEWAS["utc_offset"])
    if a.calibration == "none":
        cal, how["calibration_file"] = None, "command line --calibration none"
    else:
        cal, how["calibration_file"] = ask("Calibration file of benchmark or observed yields (used by step 08 only)",
                                           "--calibration", a.calibration, a_file, optional=True)
    cal_line = (stored_path(cal) if cal else
                f"data/calibration/{run}_calibration_inputs_NOT_PROVIDED.csv   # no file exists: none was given in step 00a")
    p_cfg.write_text(CONFIG.format(today=datetime.date.today().isoformat(), aoi_file=stored_path(aoi), name=name,
                                   season=season, crop=crop, w0=w0, w1=w1, p0=p0, p1=p1, q0=q0, q1=q1, calibration=cal_line,
                                   utc_offset=f"{off:g}", run=run), encoding="utf-8")
    print(f"\nrun config written: {p_cfg}")
    return run, how


# ---------------------------------------------------------------- record of all parameters
def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def record(run, how):
    """Checks the parameters of the run and writes outputs/<run>/00a_parameters/: parameters.csv (every input
    parameter with its source), checks.csv, manifest. Returns (checks table, output folder)."""
    cfg = config.load(run)
    aoi = cfg["aoi"]
    p_cfg = ROOT / "config" / "aoi" / f"{run}.yaml"
    out = config.output_dir(cfg, "00a_parameters")
    config.check_writable(out)
    rows, checks, inputs = [], [], {"run_config": p_cfg, "project_yaml": ROOT / "config" / "project.yaml"}

    def par(group, name, value, source):
        rows.append({"group": group, "parameter": name, "value": value, "source": source})

    def chk(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    src = lambda key: f"{p_cfg.name}" + (f" ({how[key]})" if key in how else "")  # noqa: E731
    par("run", "run_id", run, src("run_id"))
    par("run", "name", aoi.get("name"), src("name"))
    par("run", "season", aoi.get("season"), src("season"))
    par("run", "crop", aoi.get("crop"), src("crop"))
    sd = aoi["season_dates"]
    w = [str(x) for x in sd["window"]]
    par("run", "season_window", f"{w[0]} to {w[1]}", src("window_start"))
    par("run", "peak_search", " to ".join(str(x) for x in sd["peak_search"]), src("peak_search_start"))
    par("run", "quality_window", " to ".join(str(x) for x in sd["quality_window"]), src("quality_window_start"))
    off = aoi.get("local_day_utc_offset_hours")
    par("run", "local_day_utc_offset_hours", off, src("local_day_utc_offset_hours"))
    par("run", "calibration_file", aoi["calibration"]["input_file"], src("calibration_file"))
    lead = cfg["season"]["lead_days"]
    start = pd.Timestamp(w[0]) - pd.Timedelta(days=lead)
    end = pd.Timestamp(w[1])
    par("run", "download_period", f"{start:%Y-%m-%d} to {end:%Y-%m-%d}",
        f"season window start - season.lead_days ({lead}, {cfg['season']['lead_days_source']}) to the window end")
    for k, v in (aoi.get("field_exports") or {}).items():
        par("run", f"field_exports.{k}", v, p_cfg.name)

    chk("season window", "PASS" if w[0] < w[1] else "FAIL", f"{w[0]} to {w[1]}")
    for key in ("peak_search", "quality_window"):
        a0, a1 = (str(x) for x in sd[key])
        chk(f"{key} inside the season window", "PASS" if w[0] <= a0 < a1 <= w[1] else "FAIL", f"{a0} to {a1}")
    today = pd.Timestamp(datetime.date.today())
    chk("download period is over", "PASS" if end < today else "FAIL",
        f"period ends {end:%Y-%m-%d}, today {today:%Y-%m-%d}" + ("" if end < today else
                                                                  " - the data of the period cannot be complete yet"))
    chk("local day of the weather", "PASS" if isinstance(off, (int, float)) else "FAIL",
        f"UTC{off:+g} h" if isinstance(off, (int, float)) else "local_day_utc_offset_hours missing in the run config")
    fx = aoi.get("field_exports") or {}
    miss = [k for k in ("e1_s2_perdate", "e2_landsat_perdate", "e3_weather_daily") if k not in fx]
    chk("export files named in the run config", "PASS" if not miss else "FAIL",
        "E1, E2, E3" if not miss else f"missing field_exports: {miss}")

    inp = aoi.get("inputs") or {}
    bd = inp.get("boundary_file")
    if bd is not None:
        p = config.path(bd)
        par("area", "aoi_file", bd, src("aoi_file"))
        if p.is_file():
            try:
                f = aoi_facts(read_aoi(p))
                inputs["aoi_file"] = p
                par("area", "aoi_sha256", config.sha256(p), f"computed from {p.name}")
                for k in ("polygons", "crs", "area_ha", "area_crs", "lon_min", "lon_max", "lat_min", "lat_max"):
                    par("area", f"aoi_{k}", round(f[k], 4) if isinstance(f[k], float) else f[k], f"read from {p.name}")
                chk("AOI polygon file", "PASS", f"{p}: {f['polygons']} polygon(s), {f['area_ha']:,.1f} ha "
                    f"({f['area_crs']}), {f['crs']}")
            except ValueError as e:
                chk("AOI polygon file", "FAIL", str(e))
        else:
            chk("AOI polygon file", "FAIL", f"not found: {p}")
        par("area", "units", "parts of the area inside each ERA5-Land cell, all land cover - not crop fields",
            "step 00b (00b_units.py)")
    else:
        par("area", "units", "field sample of step 02 (crop map)", "step 02 (02_fields.py)")
        chk("AOI polygon file", "INFO", "none in the run config: the units are the step-02 field sample")
    par("area", "era5_cell_deg", inp.get("era5_cell_deg"), p_cfg.name)
    for k, v in inp.items():
        if k in ("boundary_file", "era5_cell_deg") or not isinstance(v, str) or not re.search(r"[\\/.]", v):
            continue
        found = config.path(v).exists()
        par("other input files of the run config", k, v, f"{p_cfg.name}; " + ("found" if found else
                                                                             "NOT on this machine"))
    cal = config.path(str(aoi["calibration"]["input_file"]).split("#")[0].strip())
    chk("calibration file (step 08 only)", "INFO", f"{cal.name}: " + ("found" if cal.is_file() else
                                                                    "no file - none is registered for this run"))

    crop = cfg["crop"]
    inputs["crop_file"] = Path(crop["file"])
    for r in config.crop_entries(crop):
        par(f"crop ({Path(crop['file']).name})", f"{r['section']}.{r['name']}",
            r["values"] if r["values"] is not None else r["value"], f"{r['status']}: {r['source']}")
    n_st = pd.Series([r["status"] for r in config.crop_entries(crop)]).value_counts().to_dict()
    chk("crop file: every value has a source and a status", "PASS", f"{Path(crop['file']).name}: {n_st}")
    par("method and data access (not asked)", "units", cfg.get("units"), "config/project.yaml")
    par("method and data access (not asked)", "season.lead_days", lead, cfg["season"]["lead_days_source"])
    par("method and data access (not asked)", "ndvi_cleaning.min_valid_fraction",
        cfg["ndvi_cleaning"]["min_valid_fraction"], "config/project.yaml")
    for k, v in flatten(cfg["stac"], "stac."):
        par("method and data access (not asked)", k, v, "config/project.yaml")
    ef = cfg["stac"]["landsat"].get("ef_function")
    chk("EF function for the Landsat export E2", "INFO" if ef else "BLOCKED",
        str(ef) if ef else "none configured (stac.landsat.ef_function): step 00c will not write EF and ETa; "
                           "steps 01-10 cannot run without them")
    key = (Path.home() / ".cdsapirc").exists() or bool(os.environ.get("CDSAPI_KEY"))
    par("environment", "python", sys.version.split()[0], sys.executable)
    par("environment", "climate_data_store_key", "found" if key else "not found", "~/.cdsapirc or CDSAPI_KEY")
    chk("Climate Data Store key for the weather export E3", "PASS" if key else "WARN",
        "found" if key else "no ~/.cdsapirc file and no CDSAPI_KEY: E3 cannot be downloaded")

    pt, ct = pd.DataFrame(rows), pd.DataFrame(checks)
    pt.to_csv(out / "parameters.csv", index=False)
    ct.to_csv(out / "checks.csv", index=False)
    config.write_manifest(out, "00a_parameters", inputs,
                          {"parameters": out / "parameters.csv", "checks": out / "checks.csv"},
                          {"answers": how})
    return pt, ct, out
