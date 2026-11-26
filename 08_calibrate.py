"""Step 08 - calibration: one common factor on LUE-ENS (ratio of means, Cochran 1977; Indore step 11 logic).

  member-wise  every LUE-ENS member m is calibrated with its own factor f_m = observed / simulated_m (ratio of
               means at the observed level); calibrated values = simulated_m x f_m. The level is then fixed by the
               observations, so eps_max, f, HI and DM cancel and the remaining spread (P10-P90) is the uncertainty
               of the spatial pattern (water-scalar form, season timing, Topt, sensor mix) - the calibrated product.
  common       one factor on the LUE-ENS mean (= Indore step 11), reported for comparison; because yield is
               proportional to eps_max it equals eps_max x factor: the implied eps_max is a physical plausibility
               check against the documented range.
Simulated values (v0.8.0): the pixel-based LUE-ENS members of the district and the tehsils (step 05b: every crop-map
pixel, aggregated pixel -> parcel -> tehsil, district); the sampled fields (step 06) are calibrated with the same
member factors for reference (calibrated_unit.csv).
Input: versioned table (config calibration.input_file): level, unit_id, crop, season, observed_yield_t_ha, area_ha,
source, status, yield_basis, units.
  placeholder (district average)  -> LUE-Y2-provisional; validation statistics not applicable (equal by design).
  observed (tehsil yields)        -> factor with leave-one-tehsil-out cross-validation, bootstrap over tehsils,
                                     RMSE, NRMSE, bias, MAE, Willmott d, R2 before / after; acceptance targets
                                     (frozen in config) checked on the cross-validated predictions -> LUE-Y2.
Each calibration writes a new folder outputs/<district>/08_calibrate/<status>_<product>_<hash8>/ (input file,
settings and simulated members hashed); an existing folder is never overwritten. latest.json names the current one.
Optional columns measured_hi and measured_dm_fraction (same basis as config yield_basis) are compared with the
parameter ranges in measured_parameters.csv - reported, never substituted.
Writes: calibration_factor.csv, calibrated_<level>.csv, inputs_used.csv, validation.csv, acceptance.csv,
measured_parameters.csv, manifest.
"""
import json

import numpy as np
import pandas as pd

from core import calibration as cal
from core import config, stats
from core import ensemble as en


def calibrated_stats(members, level, f_member, q):
    """Statistics across members of simulated_m x f_m per group of `level` (f_member: Series by member)."""
    rows = []
    x = members.assign(f=members.member.map(f_member))
    for g, y in x.groupby(level):
        r = {level: g}
        for col in ("fresh_t_ha", "dry_t_ha"):
            s = en.member_stats(y[col] * y.f, y.weight, q)
            r.update({f"{col}_{k}": v for k, v in s.items() if k != "n"})
        rows.append(r)
    return pd.DataFrame(rows)


def member_factors(obs, sim_matrix, area=None):
    """f_m = ratio of means per member: obs (n units), sim_matrix (members x n units)."""
    w = np.ones(len(obs)) if area is None else np.asarray(area, float)
    return (w * np.asarray(obs, float)).sum() / (sim_matrix * w[None, :]).sum(axis=1)


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    base = config.output_dir(cfg, "08_calibrate")
    cc, q, P = cfg["calibration"], cfg["ensemble"]["quantiles"], cfg["parameters"]
    t, p_in, status = cal.read_inputs(cfg)
    files = {lv: config.step_output(cfg, "05b_pixel_model", f"members_{lv}.csv") for lv in ("district", "tehsil")}
    files["unit"] = config.step_output(cfg, "06_ensemble", "members_unit.csv")      # the sampled fields (reference)
    ens = {lv: pd.read_csv(files[lv]) for lv in files}
    cid = cal.calibration_id(cfg, status, files["district"])
    out = base / cid

    def set_latest():                                     # only for a complete calibration folder
        with open(base / "latest.json", "w", encoding="utf-8") as f:
            json.dump({"calibration_id": cid, "status": status, "input_file": str(p_in)}, f, indent=1)

    if (out / "manifest.json").exists():
        set_latest()
        print(f"calibration {cid} exists (same input, settings, members and code version) - not overwritten")
        return
    out.mkdir(parents=True, exist_ok=True)

    def ens_mean(lv, key):
        x = ens[lv]
        x = x[x[lv] == key] if lv != "district" else x
        return float(np.average(x.fresh_t_ha, weights=x.weight))

    product = "LUE-Y2-provisional" if status == "placeholder" else "LUE-Y2"
    wm = ens["district"].set_index("member").weight
    val, acc = [], []
    if status == "placeholder":
        if set(t.level) != {"district"} or len(t) != 1:
            raise SystemExit("placeholder calibration expects one district row")
        obs = float(t.observed_yield_t_ha.iloc[0])
        sim = ens_mean("district", "district")
        factor, ci = obs / sim, (np.nan, np.nan)
        f_m = pd.Series(member_factors([obs], ens["district"].set_index("member")[["fresh_t_ha"]].to_numpy()),
                        index=ens["district"].member)
        val = [{"metric": k, "value": "not applicable - placeholder: the calibrated district value equals the "
                                     "benchmark by design"} for k in
               ("n", "bias", "mae", "rmse", "nrmse_pct", "willmott_d", "r2_pearson", "r2_determination")]
        acc = [{"target": k, "value": v, "result": "not tested (no independent observations)"}
               for k, v in cfg["acceptance_targets"].items() if k not in ("frozen_on", "nrmse_definition")]
    else:
        tt = t[t.level == "tehsil"].copy()
        known = set(ens["tehsil"].tehsil)
        if len(tt) < 3:
            raise SystemExit("observed calibration needs at least 3 tehsil rows (leave-one-out validation)")
        if set(tt.unit_id) - known:
            raise SystemExit(f"tehsils not in the run: {sorted(set(tt.unit_id) - known)}; known: {sorted(known)}")
        tt["sim"] = [ens_mean("tehsil", u) for u in tt.unit_id]
        area = tt.area_ha.to_numpy(float) if tt.area_ha.notna().all() else None
        S = ens["tehsil"].pivot(index="member", columns="tehsil", values="fresh_t_ha")[tt.unit_id].to_numpy()
        w = wm.reindex(ens["tehsil"].pivot(index="member", columns="tehsil", values="fresh_t_ha").index).to_numpy()
        obs_v = tt.observed_yield_t_ha.to_numpy(float)
        factor = cal.ratio_of_means(obs_v, tt.sim, area)
        f_m = pd.Series(member_factors(obs_v, S, area), index=ens["tehsil"].member.drop_duplicates().sort_values())
        rng = np.random.default_rng(cc["seed"])
        boot = [cal.ratio_of_means(obs_v[i], tt.sim.to_numpy()[i], None if area is None else area[i])
                for i in (rng.integers(0, len(tt), len(tt)) for _ in range(cc["bootstrap"]))]
        ci = tuple(np.percentile(boot, [2.5, 97.5]))      # 95 % percentile bootstrap interval of the factor
        loo = []
        for i in range(len(tt)):                          # member-wise factors without tehsil i -> predict i
            k = np.arange(len(tt)) != i
            f_i = member_factors(obs_v[k], S[:, k], None if area is None else area[k])
            loo.append(float(np.average(S[:, i] * f_i, weights=w)))
        tt["calibrated_loo"] = loo
        before = stats.metrics(tt.observed_yield_t_ha, tt.sim)
        after = stats.metrics(tt.observed_yield_t_ha, tt.calibrated_loo)
        val = [{"metric": k, "uncalibrated": before[k], "calibrated_leave_one_out": after[k]} for k in before]
        at = cfg["acceptance_targets"]
        ta = pd.read_csv(config.step_output(cfg, "05b_pixel_model", "ensemble_tehsil.csv")).set_index("tehsil").weight_ha
        a = tt.unit_id.map(ta).to_numpy(float)
        d_bias = 100 * ((a * tt.calibrated_loo).sum() / (a * tt.observed_yield_t_ha).sum() - 1)
        acc = [{"target": "tehsil_nrmse_pct", "value": at["tehsil_nrmse_pct"], "result": after["nrmse_pct"],
                "pass": after["nrmse_pct"] <= at["tehsil_nrmse_pct"]},
               {"target": "district_abs_bias_pct (area-weighted, leave-one-out predictions)",
                "value": at["district_abs_bias_pct"], "result": d_bias, "pass": abs(d_bias) <= at["district_abs_bias_pct"]},
               {"target": "district_nrmse_pct", "value": at["district_nrmse_pct"],
                "result": "needs several seasons (hindcast)", "pass": None},
               {"target": "benchmark", "value": at["benchmark"], "result": "needs the official multi-year series",
                "pass": None}]
        tt.to_csv(out / "tehsil_obs_vs_sim.csv", index=False)
        wt = np.ones(len(tt)) if area is None else area                # same weights as the factor
        obs, sim = float(np.average(obs_v, weights=wt)), float(np.average(tt.sim, weights=wt))

    grid_eps = P["eps_max"]
    fac = pd.DataFrame([{
        "calibration_id": cid, "product": product, "status": status, "base_product": cc["base_product"],
        "method": "member-wise ratio of means (product); common factor reported",
        "observed_t_ha": obs, "simulated_t_ha": sim, "factor": factor, "factor_ci95_low": ci[0], "factor_ci95_high": ci[1],
        "member_factor_p10": float(stats.wquantile(f_m, wm.reindex(f_m.index), [min(q)])[0]),   # ensemble quantiles
        "member_factor_p90": float(stats.wquantile(f_m, wm.reindex(f_m.index), [max(q)])[0]),
        "implied_eps_max_central": grid_eps["central"] * factor,
        "implied_eps_max_range": f"{min(grid_eps['values']) * factor:.2f}-{max(grid_eps['values']) * factor:.2f}",
        "documented_eps_max_range": f"{min(grid_eps['values'])}-{max(grid_eps['values'])}",
        "input_file": str(p_in), "input_sha256": config.sha256(p_in),
        "warning": cc["warning_placeholder"] if status == "placeholder" else "", "model_scope_note": cc["model_scope_note"]}])
    fac.to_csv(out / "calibration_factor.csv", index=False)
    f_m.rename("member_factor").to_csv(out / "member_factors.csv")
    for lv in ("district", "tehsil", "unit"):
        calibrated_stats(ens[lv], lv, f_m, q).assign(product=product, common_factor=factor) \
            .to_csv(out / f"calibrated_{lv}.csv", index=False)
    t.to_csv(out / "inputs_used.csv", index=False)
    # measured harvest index / dry matter of the harvested part (optional columns of the calibration file): compared, never substituted
    mp = []
    for col, key in (("measured_hi", "hi"), ("measured_dm_fraction", "dm")):
        if col in t and t[col].notna().any():
            v = t[col].dropna()
            lo_, hi_ = P[key]["range"]
            mp.append({"parameter": key, "n": len(v), "measured_mean": v.mean(), "measured_min": v.min(),
                       "measured_max": v.max(), "config_central": P[key]["central"], "config_range": f"{lo_}-{hi_}",
                       "mean_inside_range": lo_ <= v.mean() <= hi_,
                       "basis": cfg["yield_basis"]["harvest_index" if key == "hi" else "fresh_dm"]})
    pd.DataFrame(mp, columns=["parameter", "n", "measured_mean", "measured_min", "measured_max", "config_central",
                              "config_range", "mean_inside_range", "basis"]).to_csv(out / "measured_parameters.csv",
                                                                                     index=False)
    pd.DataFrame(val).to_csv(out / "validation.csv", index=False)
    pd.DataFrame(acc).to_csv(out / "acceptance.csv", index=False)
    config.write_manifest(out, "08_calibrate", {"input_file": p_in, **files}, {"folder": out},
                          {"calibration_id": cid, "calibration": cc})
    set_latest()
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(fac.drop(columns=["input_file", "input_sha256", "model_scope_note"]).T.to_string(header=False))
    print(pd.read_csv(out / "calibrated_tehsil.csv")[["tehsil", "fresh_t_ha_mean", "fresh_t_ha_p10", "fresh_t_ha_p90"]]
          .round(2).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
