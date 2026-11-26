"""Calibration helpers (Indore WOFOST step 11 logic): versioned input table, safeguards, calibration id,
ratio-of-means factor (Cochran 1977)."""
import hashlib
import json

import numpy as np
import pandas as pd

from . import config

REQUIRED = ["level", "unit_id", "crop", "season", "observed_yield_t_ha", "area_ha", "source", "status",
            "yield_basis", "units"]


def read_inputs(cfg):
    """Read and check the calibration input table; returns (table, path, status)."""
    cc = cfg["calibration"]
    p = config.path(cc["input_file"])
    if not p.exists():
        raise SystemExit(f"calibration input {p} not found")
    t = pd.read_csv(p)
    miss = [c for c in REQUIRED if c not in t]
    if miss:
        raise SystemExit(f"calibration input lacks columns {miss}")
    status = set(t.status)
    if len(status) != 1 or not status <= {"placeholder", "observed"}:
        raise SystemExit(f"status must be one of placeholder | observed and not mixed: {status}")
    status = status.pop()
    exp = cc["expected"]
    for k in ("crop", "yield_basis", "units"):
        if set(t[k]) != {exp[k]}:
            raise SystemExit(f"calibration input {k} {set(t[k])} differs from expected {exp[k]}")
    if not (t.observed_yield_t_ha > 0).all():
        raise SystemExit("observed yields must be positive")
    if set(t.level) - {"district", "tehsil"}:
        raise SystemExit(f"levels supported in this version: district, tehsil (not {set(t.level)})")
    if set(t.season) != {cc["season"]}:
        if not (status == "placeholder" and cc.get("allow_benchmark_season_mismatch")):
            raise SystemExit(f"season {set(t.season)} differs from {cc['season']} (allowed for placeholders only)")
    return t, p, status


def calibration_id(cfg, status, member_file):
    """status_product_hash8 from the input file, the calibration settings and the simulated members."""
    cc = cfg["calibration"]
    h = hashlib.sha256()
    h.update(cfg["framework"]["version"].encode())               # a code version change gives a new calibration
    h.update(config.sha256(config.path(cc["input_file"])).encode())
    h.update(json.dumps(cc, sort_keys=True, default=str).encode())
    h.update(config.sha256(member_file).encode())
    return f"{status}_{cc['base_product']}_{h.hexdigest()[:8]}"


def ratio_of_means(obs, sim, area=None):
    """factor = (weighted) mean observed / (weighted) mean simulated."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    w = np.ones_like(obs) if area is None or np.isnan(np.asarray(area, float)).any() else np.asarray(area, float)
    return float((w * obs).sum() / (w * sim).sum())


def latest(cfg):
    """The current calibration of step 08 (latest.json): id, status, folder, factor table row and member factors;
    None when step 08 has not run. Read by steps 09 and 11."""
    p = config.step_output(cfg, "08_calibrate", "latest.json", required=False)
    if not p.exists():
        return None
    info = json.load(open(p, encoding="utf-8"))
    folder = p.parent / info["calibration_id"]
    return {**info, "folder": folder, "factor": pd.read_csv(folder / "calibration_factor.csv").iloc[0],
            "member_factors": pd.read_csv(folder / "member_factors.csv", index_col=0).member_factor}
