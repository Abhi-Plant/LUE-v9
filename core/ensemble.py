"""Combined ensemble LUE-ENS: member grid with product weights, weighted statistics, exact variance partition
(functional ANOVA of a full factorial with product weights) and member-level aggregation - the logic of the
Indore WOFOST step 10b, written for the LUE factors."""
import itertools

import numpy as np
import pandas as pd

from . import stats

PARAM_KEYS = ["eps_max", "f_par", "topt", "hi", "dm"]
FACTORS = PARAM_KEYS + ["ws_form", "season_p"]


def member_grid(cfg):
    """All members (eps_max, f_par, topt, hi, dm, ws_form, season_p) with normalised product weights.
    Parameter values: equal weights; water-scalar forms and season fractions: config weights."""
    P = cfg["parameters"]
    wf = {k: float(v["weight"]) for k, v in cfg["water_stress"]["forms"].items()}
    sp = {float(k): float(v) for k, v in cfg["ensemble"]["season_weights"].items()}
    missing = set(cfg["season"]["ensemble_fractions"]) ^ set(sp)
    if missing:
        raise SystemExit(f"season_weights must cover season.ensemble_fractions, differs in {missing}")
    rows = []
    for vals in itertools.product(*[P[k]["values"] for k in PARAM_KEYS], wf, sp):
        w = 1.0
        for k, v in zip(PARAM_KEYS, vals[:5]):
            w *= 1.0 / len(P[k]["values"])
        rows.append(dict(zip(FACTORS, vals), weight=w * wf[vals[5]] / sum(wf.values()) * sp[vals[6]] / sum(sp.values())))
    g = pd.DataFrame(rows)
    g.insert(0, "member", range(len(g)))
    if abs(g.weight.sum() - 1) > cfg["ensemble"]["weight_tolerance"]:
        raise SystemExit(f"member weights sum to {g.weight.sum()}")
    return g


def member_stats(y, w, quantiles):
    """Weighted mean, SD, CV, quantiles, min, max and member count of one unit's members."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    w = w / w.sum()
    m = float((w * y).sum())
    sd = float(np.sqrt((w * (y - m) ** 2).sum()))
    out = {"mean": m, "sd": sd, "cv_pct": 100 * sd / m if m else np.nan, "min": y.min(), "max": y.max(), "n": len(y)}
    for q, v in zip(quantiles, stats.wquantile(y, w, quantiles)):
        out[f"p{int(round(100 * q))}"] = v
    return out


def variance_partition(df, value, weight="weight", factors=FACTORS):
    """Exact first-order shares of a full factorial with product weights: V_k = sum_l w_l (E[y|l] - mu)^2,
    interaction = V - sum V_k. Returns {factor: share %} incl. 'interaction' and 'total_variance'."""
    w = df[weight] / df[weight].sum()
    mu = float((w * df[value]).sum())
    v = float((w * (df[value] - mu) ** 2).sum())
    out = {}
    for k in factors:
        g = pd.DataFrame({"k": df[k], "w": w, "wy": w * df[value]}).groupby("k").sum()
        out[k] = float((g.w * (g.wy / g.w - mu) ** 2).sum())
    out["interaction"] = v - sum(out.values())
    shares = {k: (100 * x / v if v > 0 else 0.0) for k, x in out.items()}
    shares["total_variance"] = v
    return shares


def aggregate_members(members, unit_weight, by=None):
    """Member-level aggregation: for each member, the weighted mean of its units (unit_weight: Series by unit),
    within groups `by` (Series by unit, e.g. tehsil or cell) or over all units. Columns: [group,] member, value cols."""
    m = members.copy()
    m["_w"] = m.unit.map(unit_weight)
    if m._w.isna().any():
        raise SystemExit(f"units without a weight: {sorted(m[m._w.isna()].unit.unique())[:5]}")
    keys = ["member"] if by is None else ["group", "member"]
    if by is not None:
        m["group"] = m.unit.map(by)
    vals = [c for c in ("biomass_g_m2", "dry_t_ha", "fresh_t_ha") if c in m]
    for c in vals:
        m[c] = m[c] * m._w
    g = m.groupby(keys)[vals + ["_w"]].sum()
    for c in vals:
        g[c] = g[c] / g._w
    return g.drop(columns="_w").reset_index()
