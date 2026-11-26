"""Configuration, paths, command line and run manifest, shared by all steps."""
import argparse
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def read_yaml(p):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


RUN_KEYS = {"season_dates": ("window", "peak_search", "quality_window"), "calibration": ("input_file",)}
LOADED = {"config_files": []}        # config files of the run loaded by this process (fingerprint in every manifest)

# Status of a crop value (config/crops/<crop>.yaml): every value must have a source and one of these
CROP_STATUS = ("verified", "verified_range", "adapted", "not_verified", "setting")
CROP_PARAMS = ("eps_max", "topt", "hi", "dm")
CROP_VALUE_SECTIONS = ("crop_coefficient", "season", "heat_stress")


def crop_entries(crop):
    """Every value of a crop file as rows: section, name, value, source, status."""
    rows = [{"section": "parameters", "name": k, "value": v.get("central"), "values": v.get("values"),
             "source": v.get("source"), "status": v.get("status")} for k, v in crop.get("parameters", {}).items()]
    for sec in CROP_VALUE_SECTIONS:
        for k, v in crop.get(sec, {}).items():
            v = v if isinstance(v, dict) else {"value": v}
            rows.append({"section": sec, "name": k, "value": v.get("value"), "values": None,
                         "source": v.get("source"), "status": v.get("status")})
    return rows


def load_crop(name):
    """config/crops/<name>.yaml, checked: every parameter of the model present, every value with a source and a status."""
    p = ROOT / "config" / "crops" / f"{str(name).lower()}.yaml"
    if not p.exists():
        raise SystemExit(f"no crop file {p} (config/crops/<crop>.yaml)")
    crop = read_yaml(p)
    miss = [k for k in CROP_PARAMS if k not in crop.get("parameters", {})]
    miss += [s for s in ("crop", "yield_basis", "calibration_expected") + CROP_VALUE_SECTIONS if s not in crop]
    if miss:
        raise SystemExit(f"crop file {p.name}: missing {miss}")
    bad = [f"{r['section']}.{r['name']}" for r in crop_entries(crop)
           if r["value"] is None or not str(r["source"] or "").strip() or r["status"] not in CROP_STATUS]
    if bad:
        raise SystemExit(f"crop file {p.name}: value, source or status ({'|'.join(CROP_STATUS)}) missing for {bad}")
    for k in CROP_PARAMS:
        v = crop["parameters"][k]
        if v["central"] not in v["values"] or not v["range"][0] <= min(v["values"]) <= max(v["values"]) <= v["range"][1]:
            raise SystemExit(f"crop file {p.name}: {k} central / values outside the stated range")
    crop["file"] = str(p)
    return crop


def load(run):
    """Method settings (config/project.yaml) + the run config (config/aoi/<run>.yaml: one district, one crop, one
    season) under 'aoi' + the crop file of the run (config/crops/<crop>.yaml) under 'crop'. Season dates, the
    calibration file and the crop values are merged into the method sections, so steps read cfg['season']['window'],
    cfg['parameters']['eps_max'], cfg['crop_coefficient']['kc_mid'] etc. Outputs go to outputs/<run>/."""
    cfg = read_yaml(ROOT / "config" / "project.yaml")
    p = ROOT / "config" / "aoi" / f"{run.lower()}.yaml"
    if not p.exists():
        raise SystemExit(f"no run config {p} (config/aoi/<district>_<season>.yaml)")
    aoi = read_yaml(p)
    for sec, keys in RUN_KEYS.items():
        miss = [k for k in keys if k not in aoi.get(sec, {})]
        if miss:
            raise SystemExit(f"run config {p.name}: missing {sec}: {miss}")
    if not aoi.get("crop"):
        raise SystemExit(f"run config {p.name}: 'crop' missing (name of a crop file in config/crops/)")
    crop = load_crop(aoi["crop"])
    LOADED["config_files"] = [ROOT / "config" / "project.yaml", p, Path(crop["file"])]
    cfg["season"].update({k: aoi["season_dates"][k] for k in RUN_KEYS["season_dates"]})
    cfg["season"]["lead_days_source"] = "config/project.yaml"
    if aoi["season_dates"].get("lead_days") is not None:       # optional per run; else season.lead_days of project.yaml
        ld = aoi["season_dates"]["lead_days"]
        if isinstance(ld, bool) or not isinstance(ld, int) or ld < 0:
            raise SystemExit(f"run config {p.name}: season_dates.lead_days must be a whole number of days >= 0 ({ld!r})")
        cfg["season"]["lead_days"], cfg["season"]["lead_days_source"] = ld, f"config/aoi/{p.name}"
    cfg["calibration"].update({"input_file": aoi["calibration"]["input_file"], "season": aoi["season"],
                               "expected": crop["calibration_expected"]})
    cfg["parameters"].update(crop["parameters"])
    if cfg.get("temperature_scalar", {}).get("topt") == "casa_ndvi_peak_month":
        # Topt comes from the data of each unit (step 04); the ensemble parameter becomes an offset fixed at 0
        cfg["parameters"]["topt"] = {
            "central": 0.0, "values": [0.0], "range": [0.0, 0.0], "unit": "degC offset from the CASA Topt of the unit",
            "status": "verified", "source": "CASA definition: Topt = mean air temperature of the month of maximum NDVI "
            "(Potter et al. 1993 p.816; Bastiaanssen & Ali 2003 p.324), per unit, step 04 topt_casa.csv"}
    elif cfg.get("temperature_scalar", {}).get("topt") not in (None, "crop_values"):
        raise SystemExit("config temperature_scalar.topt must be casa_ndvi_peak_month or crop_values")
    for sec in ("crop_coefficient", "heat_stress"):
        cfg[sec] = {k: v["value"] for k, v in crop[sec].items()}
    cfg["ndvi_cleaning"]["min_amplitude"] = crop["season"]["min_amplitude"]["value"]
    cfg["yield_basis"] = crop["yield_basis"]
    cfg["data_qc"]["crop_class"] = crop["crop"]["map_class"]
    cfg["crop"] = crop
    cfg["aoi"] = aoi
    cfg["run_id"] = run.lower()
    return cfg


def crop_label(cfg):
    """Crop name for texts and figure titles, e.g. 'potato'."""
    return cfg["crop"]["crop"]["name"]


def season_end_label(cfg):
    """Name of the detected season end for texts, e.g. 'haulm cutting' (potato)."""
    return cfg["crop"]["crop"]["season_end_event"]


def units(cfg):
    u = cfg.get("units", "tehsil")
    if u not in ("tehsil", "field"):
        raise SystemExit(f"config units must be tehsil or field, not {u}")
    return u


def biomass_method(cfg):
    """How the daily growth that is summed to biomass is obtained.
    Tehsil units: config biomass_method (fit_biomass = double logistic fitted to the 16-day growth rates).
    Field units: always fit_drivers - per-date data, fAPAR on each clear date and EF interpolated between clear dates
    to daily values, as the method document prescribes (steps 2 and Ws); growth-rate fits need 16-day periods."""
    m = cfg.get("biomass_method", "fit_drivers")
    if m not in ("fit_biomass", "fit_drivers"):
        raise SystemExit(f"biomass_method must be fit_biomass or fit_drivers, not {m}")
    return m if units(cfg) == "tehsil" else "fit_drivers"


def path(rel):
    return (ROOT / rel).resolve()


def input_path(cfg, key, section="inputs"):
    """Existing input file of the district, read in place."""
    p = path(cfg["aoi"][section][key])
    if not p.exists():
        raise SystemExit(f"input '{key}' not found: {p}")
    return p


def optional_input(cfg, key, section="inputs"):
    """Path of an optional input of the run config (e.g. an independent model's tehsil results for comparison);
    None when the run config does not register it (key absent or null). A registered file must exist."""
    if not (cfg["aoi"].get(section) or {}).get(key):
        return None
    return input_path(cfg, key, section)


def parse_args(description):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", "--district", dest="run", required=True,
                    help="run config in config/aoi/ = district and season (e.g. dewas_2025_26)")
    return ap.parse_args()


def output_dir(cfg, step):
    """Output folder of a step. Its manifest.json is removed when the step starts and written as the last file when
    it ends, so a step interrupted half-way (window closed, laptop off) is never taken as complete."""
    folder = ROOT / "outputs" / cfg["run_id"] / step
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").unlink(missing_ok=True)
    return folder


def write_atomic(path, write):
    """write(tmp_path) then rename to path: a file is either complete or absent, never half-written."""
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    write(tmp)
    tmp.replace(path)
    return path


def to_csv_atomic(df, path, **kw):
    return write_atomic(path, lambda p: df.to_csv(p, **kw))


def config_fingerprint(files=None):
    """sha256 of the config files of a run (project.yaml, run config, crop file): a changed setting gives a new
    fingerprint, so the steps that depend on it run again."""
    h = hashlib.sha256()
    for p in files if files is not None else LOADED["config_files"]:
        h.update(Path(p).name.encode() + b":" + sha256(p).encode())
    return h.hexdigest()


def step_complete(cfg, step):
    """(complete, reason) of a step of the run: its manifest exists (written last), the config fingerprint is the
    current one, every input file it read still exists with the same sha256 and every output file exists."""
    p = ROOT / "outputs" / cfg["run_id"] / step / "manifest.json"
    if not p.exists():
        return False, "not run or interrupted (no manifest)"
    try:
        m = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return False, "manifest unreadable"
    if m.get("config_sha256") != config_fingerprint():
        return False, "configuration changed"
    for k, files in m.get("inputs", {}).items():
        for e in files:
            q = Path(e["path"])
            if not q.exists() or sha256(q) != e["sha256"]:
                return False, f"input changed: {q.name}"
    for k, files in m.get("outputs", {}).items():
        for e in files:
            if not Path(e["path"]).exists():
                return False, f"output missing: {Path(e['path']).name}"
    return True, f"complete ({m.get('run_at')})"


def step_output(cfg, step, name, required=True):
    """Output file of an earlier step; stops with a clear message when that step has not been run."""
    p = ROOT / "outputs" / cfg["run_id"] / step / name
    if required and not p.exists():
        raise SystemExit(f"{p} missing - run step {step} first")
    return p


def check_writable(folder, patterns=("*.xlsx", "*.csv", "*.png", "*.gpkg", "*.shp")):
    """Stop before any work if output files are open in another program (Windows locks open files)."""
    locked = []
    for pat in patterns:
        for p in Path(folder).rglob(pat):
            if p.name.startswith("~$"):
                continue
            try:
                with open(p, "r+b"):
                    pass
            except PermissionError:
                locked.append(str(p))
    if locked:
        raise SystemExit("close these files (open in another program) and run again:\n  " + "\n  ".join(locked))


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


SHP_PARTS = (".shp", ".dbf", ".shx", ".prj", ".cpg")


def fingerprint(p):
    """sha256 of an input file; a shapefile is fingerprinted as the sha256 of its parts' hashes (.shp .dbf .shx .prj
    .cpg, those present), so a changed attribute table is detected too."""
    p = Path(p)
    if p.suffix.lower() != ".shp":
        return sha256(p)
    h = hashlib.sha256()
    for ext in SHP_PARTS:
        q = p.with_suffix(ext)
        if q.exists():
            h.update(f"{ext}:{sha256(q)}".encode())
    return h.hexdigest()


T0 = datetime.datetime.now()


def usage():
    """Run time of this step (s) and peak memory of the process (MB; psutil, None if unavailable)."""
    out = {"run_seconds": round((datetime.datetime.now() - T0).total_seconds(), 1), "peak_memory_mb": None}
    try:
        import psutil
        mi = psutil.Process().memory_info()
        out["peak_memory_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20, 1)   # Windows: peak working set
    except Exception:
        pass
    return out


def write_manifest(folder, step, inputs, outputs, extra=None):
    """What a step read and wrote (paths + sha256), software versions and run time."""
    import importlib.metadata as md

    def version(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    def entry(p):
        p = Path(p)
        files = sorted(x for x in p.rglob("*") if x.is_file() and x.name != "manifest.json") if p.is_dir() else [p]
        return [{"path": str(x), "sha256": sha256(x)} for x in files]

    m = {"step": step, "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
         "framework_version": read_yaml(ROOT / "config" / "project.yaml")["framework"]["version"],
         "python": sys.version.split()[0], "platform": platform.platform(),
         **{p: version(p) for p in ("numpy", "pandas", "scipy", "SALib", "geopandas", "shapely", "pyproj", "PyYAML")},
         "inputs": {k: entry(v) for k, v in inputs.items()},
         "outputs": {k: entry(v) for k, v in outputs.items()}, **usage(),
         "config_files": [str(p) for p in LOADED["config_files"]],
         "config_sha256": config_fingerprint() if LOADED["config_files"] else None, **(extra or {})}

    def dump(tmp):
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1, default=str)
    write_atomic(Path(folder) / "manifest.json", dump)          # the last file of a step: marks it complete
    return m
