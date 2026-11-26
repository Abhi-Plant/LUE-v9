"""Step 06 - combined ensemble LUE-ENS: statistics and exact variance shares per level (Indore step 10b logic).

Levels: run unit (tehsil or field), tehsil, ERA5-Land cell (field units), district. Every member is first aggregated
over the level's units (weights: crop area, or design weight for fields), then the statistics are taken across
members with the member weights: weighted mean (= LUE-ENS value), SD, CV, P10/P50/P90, min, max.
Variance shares (functional ANOVA of the full factorial, product weights) for
  absolute  - the uncalibrated fresh yield (what a calibration has to fix: dominated by the yield level), and
  relative  - the level's yield divided by the district yield of the same member (what limits the spatial
              pattern after a one-factor calibration: eps_max, f, HI and DM cancel exactly).
Writes: ensemble_<level>.csv, members_<level>.csv, variance_shares.csv, manifest.
"""
import pandas as pd

from core import config
from core import ensemble as en

VALUE = "fresh_t_ha"


def level_tables(mem, grid, units, by, q):
    """Member-level aggregation to `by` (Series unit -> group, or None for the district) and member statistics."""
    w = units.set_index("unit").weight_ha
    agg = en.aggregate_members(mem, w, by)
    if by is None:
        agg["group"] = "district"
    agg = agg.merge(grid, on="member")
    rows = []
    for g, x in agg.groupby("group"):
        r = {"group": g, "weight_ha": float(w[units.unit[(by == g).reindex(units.unit).to_numpy()] if by is not None
                                              else units.unit].sum())}
        for col in ("fresh_t_ha", "dry_t_ha"):
            s = en.member_stats(x[col], x.weight, q)
            r.update({f"{col}_{k}": v for k, v in s.items() if k != "n"})
        r["n_members"] = len(x)
        rows.append(r)
    return pd.DataFrame(rows), agg


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    out = config.output_dir(cfg, "06_ensemble")
    config.check_writable(out)
    mode = config.units(cfg)
    paths = {"members": config.step_output(cfg, "05_lue", "members.csv"),
             "member_grid": config.step_output(cfg, "05_lue", "member_grid.csv"),
             "units": config.step_output(cfg, "03_season", "units.csv")}
    mem = pd.read_csv(paths["members"])
    grid = pd.read_csv(paths["member_grid"])
    units = pd.read_csv(paths["units"])
    q = cfg["ensemble"]["quantiles"]
    by_unit = units.set_index("unit")
    levels = {"unit": by_unit.index.to_series(index=by_unit.index), "tehsil": by_unit.tehsil, "district": None}
    if mode == "field":
        levels["cell"] = by_unit.cell

    tables, aggs = {}, {}
    for name, by in levels.items():
        t, a = level_tables(mem, grid, units, by, q)
        tables[name], aggs[name] = t, a
        t.rename(columns={"group": name}).to_csv(out / f"ensemble_{name}.csv", index=False)
        a.rename(columns={"group": name}).to_csv(out / f"members_{name}.csv", index=False)

    dist = aggs["district"].set_index("member")[VALUE]
    shares = []
    for name in [k for k in levels if k != "unit" or mode == "field"]:      # tehsil units: unit level = tehsil level
        a = aggs[name]
        for g, x in a.groupby("group"):
            shares.append({"level": name, "group": g, "quantity": "absolute fresh yield",
                           **en.variance_partition(x, VALUE)})
            if name != "district":
                y = x.assign(rel=x[VALUE].to_numpy() / dist.reindex(x.member).to_numpy())
                shares.append({"level": name, "group": g, "quantity": "relative to district (pattern)",
                               **en.variance_partition(y, "rel")})
    sh = pd.DataFrame(shares)
    sh.to_csv(out / "variance_shares.csv", index=False)
    config.write_manifest(out, "06_ensemble", paths, {"folder": out},
                          {"spread_label": cfg["ensemble"]["spread_label"], "water_stress_status": cfg["water_stress"]["status"]})

    pd.set_option("display.width", 250)
    cols = ["fresh_t_ha_mean", "fresh_t_ha_p10", "fresh_t_ha_p50", "fresh_t_ha_p90", "fresh_t_ha_cv_pct", "dry_t_ha_mean"]
    print("district:\n", tables["district"][cols].round(2).to_string(index=False))
    print("\ntehsil:\n", tables["tehsil"].set_index("group")[cols].round(2).to_string())
    f = en.FACTORS + ["interaction"]
    show = sh[sh.level.isin(["district", "tehsil"])]
    print("\nvariance shares (%), district absolute and mean over tehsils:")
    print(show.groupby(["level", "quantity"])[f].mean().round(1).to_string())
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
