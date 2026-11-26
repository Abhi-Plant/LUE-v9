"""Step 05b - pixel yield model: the LUE model of steps 03-05 for every pixel of the crop map, then pixel -> parcel ->
ERA5-Land cell, tehsil and district.

Pixels and their values: the pixel stacks of step 00c (export px, core/pixels.py) - every 10 m Sentinel-2 pixel
whose centre is inside a parcel, its NDVI per date and per overpass the Landsat EF, ETa and clear share of the 10 m
pixel square (area-weighted over the 30 m pixels it overlaps, as the field export E2; v0.9.0); weather: the daily ERA5-Land values (E3) of the parcel's cell (a cell without E3 takes the nearest cell
of E3, as step 11 of v0.7.0).
Model (core/pixels.py, vectorised over pixels): exactly the rules of steps 03-05 for field units - NDVI cleaning,
despiking, daily interpolation and Savitzky-Golay smoothing, Landsat overpasses with a clear share below
ef.min_valid_fraction left out, season dates at every amplitude fraction, exclusion of a
pixel without Sentinel-2 values, Landsat EF, enough clear dates or a crop cycle, daily fAPAR (Bastiaanssen & Ali 2003
eq. 4) and Kc from the smoothed NDVI, Landsat 8 / 9 EF and ETa/ETc linear in time between the overpasses (a missing
sensor takes the other), CASA Topt of the pixel's NDVI-peak month, season sum S = sum Ts Ws fAPAR Rs for every
(Topt, Ws form, season fraction) of the LUE-ENS members; member yield = HI eps_max f S x 0.01 / DM.
Check: a random sample of pixels (pixel_model.check_pixels) also runs through the field-unit code of steps 03-05
(core.season.unit_seasons, core.drivers.field_drivers, core.model.members): exclusion, season dates and every member
yield must agree.
Aggregation: parcel = mean of its modelled pixels (equal pixel areas); a pixel not modelled takes its parcel's value;
a parcel without a modelled pixel takes the crop-area-weighted mean of the modelled parcels of its cell, else of its
tehsil (status column). Cell, tehsil, district: crop-area-weighted mean of the parcels (area_ha of the crop map),
members aggregated first (then statistics across members, as step 06). Uncalibrated (LUE-ENS); steps 08 and 09
calibrate these tables.
Writes: pixel_s.npy (season sums, pixels x combinations; a pixel not modelled holds the value it takes),
pixel_code.npy (0 modelled, 1-5 reason not modelled: excluded_pixels.csv), pixel_source.npy (0 own value, 1 parcel,
2 cell, 3 tehsil, 4 none), pixel_season.npz (central emergence, season end and NDVI peak as days after the window
start, -1 = not modelled; peak NDVI), combos.csv, member_grid.csv, parcel_values.csv, excluded_pixels.csv,
members_<level>.csv and ensemble_<level>.csv (tehsil, cell, district), unit_check.csv, checks.csv, manifest.
"""
import time

import numpy as np
import pandas as pd

from core import config
from core import drivers as dv
from core import ensemble as en
from core import pixels as px
from core import stac_ingest as si

SOURCES = {0: "own value", 1: "parcel value", 2: "cell estimate", 3: "tehsil estimate", 4: "no estimate"}


def main():
    args = config.parse_args(__doc__)
    cfg = config.load(args.run)
    if config.units(cfg) != "field" or not cfg["aoi"]["inputs"].get("crop_mask"):
        raise SystemExit("step 05b needs field units (config units: field) and a crop map (inputs.crop_mask)")
    out = config.output_dir(cfg, "05b_pixel_model")
    config.check_writable(out)
    sc, cl, pm, q = cfg["season"], cfg["ndvi_cleaning"], cfg["pixel_model"], cfg["ensemble"]["quantiles"]
    P = si.run_paths(cfg)
    st = px.Stacks(P)
    paths = {"pixel_stacks_record": st.folder / "record.json", "e3_weather_daily": P["exports"]["e3_weather_daily"],
             "crop_mask": config.input_path(cfg, "crop_mask")}
    if not paths["e3_weather_daily"].exists():
        raise SystemExit(f"{paths['e3_weather_daily']} missing - run step 00c (export e3)")
    t_start = time.time()
    parcels = si.load_parcels(cfg)
    if px.parcels_fingerprint(parcels) != st.rec["made_from"]["parcels"]:
        raise SystemExit("the pixel stacks were made from another crop map - run step 00c --only px")
    e3, _ = dv.weather_eto(pd.read_csv(paths["e3_weather_daily"], parse_dates=["date"]))
    W = px.Weather(cfg, e3)
    pt = st.parcels.copy()
    pt["tehsil"] = pt.tehsil.fillna("(no tehsil)")
    pt["weather_cell"] = pt.cell.map(px.weather_cells(pt.cell, e3.cell))
    parcel = st.pix["parcel"].astype(np.int64)
    pix_cell = pt.weather_cell.map(W.index).to_numpy()[parcel]
    start = pd.Timestamp(sc["window"][0]) - pd.Timedelta(days=sc["lead_days"])
    n_days = (pd.Timestamp(sc["window"][1]) - start).days + 1
    obs_off = (st.dates - start).dt.days.to_numpy()
    ovp_off = (st.overpasses.date - start).dt.days.to_numpy()
    if obs_off.min() < 0 or obs_off.max() >= n_days:
        raise SystemExit("the Sentinel-2 dates of the pixel stacks lie outside the season of the run config - run "
                         "step 00c (e1, px) for this season")
    grid, combos, km, kc, c = px.combos_of(cfg)
    fractions = sorted(set(sc["ensemble_fractions"]) | {sc["amplitude_fraction"]})
    n, K = st.n, len(combos)
    print(f"run {cfg['run_id']}: {n:,} pixels of {len(pt):,} parcels, {len(st.dates)} Sentinel-2 dates, "
          f"{len(st.overpasses)} Landsat overpasses, {len(W.cells)} weather cells; {len(grid)} members, {K} season "
          f"sums per pixel", flush=True)

    # ---- the model of steps 03-05 for every pixel, in chunks
    s = np.full((n, K), np.nan)
    code, n_used = np.zeros(n, np.int8), np.zeros(n, np.int16)
    amp, peak_nd = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    em, en_, pk = (np.full(n, -1, np.int16) for _ in range(3))
    qual = {k: np.zeros(n, np.int16) for k in ("clear_obs_quality_window", "despiked_dates", "low_valid_dates")}
    n_ef = np.zeros(n, np.int16)
    step = int(pm["chunk_pixels"])
    for a in range(0, n, step):
        b = min(a + step, n)
        V, EF, ETA, CLR = st.chunk(a, b)
        cd, nu, am, ss_, cen, qu, ne = px.model_vec(V, EF, ETA, obs_off, ovp_off, st.overpasses.sensor.to_numpy(),
                                                    pix_cell[a:b], W, cfg, combos, fractions, CLR=CLR)
        ok = cd == 0
        s[a:b], code[a:b], n_used[a:b], amp[a:b], n_ef[a:b] = ss_, cd, nu, am, ne
        i_w0 = sc["lead_days"]
        em[a:b] = np.where(ok, cen["emergence"] - i_w0, -1)
        en_[a:b] = np.where(ok, cen["season_end"] - i_w0, -1)
        pk[a:b] = np.where(ok, cen["peak_i"] - i_w0, -1)
        peak_nd[a:b] = np.where(ok, cen["peak_ndvi"], np.nan)
        for k in qual:
            qual[k][a:b] = qu[k]
        if (a // step) % 10 == 9 or b == n:
            print(f"  pixels {b:,}/{n:,}: {int((code[:b] == 0).sum()):,} modelled, "
                  f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    ok = code == 0
    if not ok.any():
        raise SystemExit("no pixel could be modelled - check the pixel stacks of step 00c")

    # ---- check: a sample of pixels through the field-unit code of steps 03-05
    rng = np.random.default_rng(int(pm["check_seed"]))
    k_chk = int(pm["check_pixels"])
    pick = np.flatnonzero(ok)
    idx = list(rng.choice(pick, min(k_chk, len(pick)), replace=False))
    bad_px = np.flatnonzero(~ok)
    if len(bad_px):
        idx += list(rng.choice(bad_px, min(max(k_chk // 4, 1), len(bad_px)), replace=False))
    idx = np.sort(np.array(idx))
    Vc, EFc, ETAc, CLRc = st.take(idx)
    chk, mem_u = px.unit_check(cfg, idx, Vc, EFc, ETAc, st.dates, st.overpasses,
                               pt.weather_cell.to_numpy()[parcel[idx]], e3, CLR=CLRc)
    chk["reason_pixel_model"] = [px.reason_text(int(code[i]), n_used[i], float(amp[i]), cl) for i in idx]
    w0 = pd.Timestamp(sc["window"][0])
    chk["emergence_pixel_model"] = [w0 + pd.Timedelta(days=int(em[i])) if em[i] >= 0 else pd.NaT for i in idx]
    chk["season_end_pixel_model"] = [w0 + pd.Timedelta(days=int(en_[i])) if en_[i] >= 0 else pd.NaT for i in idx]
    fresh_v = px.yields_from_s(s[idx], grid, km)[2]
    vm = pd.DataFrame(fresh_v, index=[f"PX{i}" for i in idx], columns=grid.member)
    um = mem_u.pivot(index="unit", columns="member", values="fresh_t_ha") if len(mem_u) else pd.DataFrame()
    diff = (vm.reindex(um.index) - um.reindex(columns=vm.columns)).abs().max(axis=1) if len(um) else pd.Series(dtype=float)
    chk["max_member_diff_t_ha"] = chk.unit.map(diff)
    chk["same_reason"] = chk.reason == chk.reason_pixel_model
    chk["same_season"] = ((chk.emergence == chk.emergence_pixel_model) & (chk.season_end == chk.season_end_pixel_model)) \
        | (chk.reason != "")
    modelled_both = (chk.reason == "") & (chk.reason_pixel_model == "")
    chk["same_yield"] = ~modelled_both | (chk.max_member_diff_t_ha <= pm["check_tolerance_t_ha"])

    # ---- pixel -> parcel -> cell, tehsil, district
    area = pt.area_ha.to_numpy(float)
    sp, n_mod = px.parcel_means(s, ok, parcel, len(pt))
    spf, pstatus = px.fill_parcels(sp, area, pt.cell.to_numpy(), pt.tehsil.to_numpy())
    s_fill = np.where(ok[:, None], s, spf[parcel])
    src = {"modelled": 1, "cell estimate": 2, "tehsil estimate": 3, "no estimate": 4}
    source = np.where(ok, 0, pd.Series(pstatus).map(src).to_numpy()[parcel]).astype(np.int8)
    pt["n_modelled_pixels"] = n_mod
    pt["modelled_pixel_share"] = np.where(pt.n_pixels > 0, n_mod / np.maximum(pt.n_pixels, 1), 0.0)
    pt["status"] = pstatus
    for k in range(K):
        pt[f"s_{k}"] = spf[:, k]
    pt["central_t_ha"] = px.central_yield(spf, c, kc)
    pt["lue_ens_t_ha"] = px.yields_from_s(spf, grid, km)[2] @ (grid.weight.to_numpy() / grid.weight.sum())
    pxt = pd.DataFrame({"parcel": parcel[ok], "em": em[ok], "end": en_[ok], "pk": peak_nd[ok],
                        "q": qual["clear_obs_quality_window"][ok]}).groupby("parcel")
    med = pxt.agg(em=("em", "median"), end=("end", "median"), pk=("pk", "mean"), q=("q", "mean"))
    to_date = lambda v: pd.NaT if pd.isna(v) else w0 + pd.Timedelta(days=int(round(v)))   # noqa: E731
    pt["emergence_median"] = [to_date(v) for v in med.em.reindex(range(len(pt)))]
    pt["season_end_median"] = [to_date(v) for v in med.end.reindex(range(len(pt)))]
    pt["peak_ndvi_mean"] = med.pk.reindex(range(len(pt))).to_numpy()
    pt["clear_s2_dates_quality_window_mean"] = med.q.reindex(range(len(pt))).to_numpy()
    reasons = pd.DataFrame({"parcel": parcel[~ok], "code": code[~ok]})
    top = reasons.groupby("parcel").code.agg(lambda x: int(x.value_counts().idxmax()))
    pt["main_reason_not_modelled"] = ""
    lone = pt.index[(pt.n_modelled_pixels == 0) & pt.index.isin(top.index)]
    pt.loc[lone, "main_reason_not_modelled"] = [px.reason_label(top[i], cl) for i in lone]
    p_fields = config.step_output(cfg, "02_fields", "fields.csv", required=False)
    pt["sampled"] = pt.field_id.isin(pd.read_csv(p_fields).field_id) if p_fields.exists() else False

    levels = {"tehsil": pt.tehsil.to_numpy(), "cell": pt.cell.to_numpy(), "district": np.full(len(pt), "district")}
    tables = {}
    for level, g in levels.items():
        mem_l, a_l = px.level_members(spf, area, g, grid, km, level)
        mod_a = pd.Series(np.where(pstatus == "modelled", area, 0.0)).groupby(g).sum()
        rows = []
        for key, x in mem_l.groupby(level):
            r = {level: key, "weight_ha": float(a_l[key]), "modelled_area_ha": float(mod_a.get(key, 0.0))}
            for col in ("fresh_t_ha", "dry_t_ha"):
                s_ = en.member_stats(x[col], x.weight, q)
                r.update({f"{col}_{k}": v for k, v in s_.items() if k != "n"})
            r["n_members"] = len(x)
            rows.append(r)
        t = pd.DataFrame(rows)
        means, _ = px.area_mean_by(spf, area, g)
        t["central_t_ha"] = t[level].map(pd.Series(px.central_yield(means.to_numpy(), c, kc), index=means.index))
        t["parcels"] = t[level].map(pd.Series(g).value_counts())
        t["pixels"] = t[level].map(pd.Series(g[parcel]).value_counts())
        t["modelled_pixels"] = t[level].map(pd.Series(g[parcel][ok]).value_counts()).fillna(0).astype(int)
        tables[level] = (t, mem_l)
        t.to_csv(out / f"ensemble_{level}.csv", index=False)
        mem_l.to_csv(out / f"members_{level}.csv", index=False)

    # ---- checks
    ck = []
    add = lambda name, status, detail: ck.append({"check": name, "status": status, "detail": detail})   # noqa: E731
    n_chk = int(len(chk))
    good = bool(chk.same_reason.all() and chk.same_season.all() and chk.same_yield.all())
    add(f"pixel model = field-unit code of steps 03-05 ({n_chk} pixels)", "PASS" if good else "FAIL",
        f"{int(modelled_both.sum())} modelled by both: largest |member yield difference| "
        f"{chk.max_member_diff_t_ha.max() if modelled_both.any() else 0:.2e} t/ha (tolerance "
        f"{pm['check_tolerance_t_ha']}); exclusion reason equal for {int(chk.same_reason.sum())} of {n_chk}, season "
        f"dates equal for {int(chk.same_season.sum())} of {n_chk}; unit_check.csv")
    rc = pd.Series([px.reason_label(cc, cl) for cc in code[~ok]], dtype=object).value_counts()
    add("pixels modelled", "INFO", f"{int(ok.sum()):,} of {n:,} ({100 * ok.mean():.1f} %); not modelled: "
        f"{rc.to_dict() if len(rc) else 'none'} - such a pixel takes its parcel's value")
    stc = pd.Series(pstatus).value_counts().to_dict()
    a_mod = float(area[pstatus == "modelled"].sum())
    add("parcels: value from their own pixels", "PASS" if a_mod > 0 else "FAIL",
        f"{stc}; area with own pixels {a_mod:,.1f} of {area.sum():,.1f} ha ({100 * a_mod / area.sum():.1f} %)")
    add("parcels without an estimate", "PASS" if "no estimate" not in stc else "WARN",
        f"{stc.get('no estimate', 0)} parcels, {area[pstatus == 'no estimate'].sum():.1f} ha (left out of the totals)")
    rule = pt.pixel_rule.value_counts().to_dict()
    add("pixels per parcel", "INFO", f"{rule}; pixels per parcel: median {pt.n_pixels.median():.0f}, "
        f"max {int(pt.n_pixels.max())}")
    add("member weights sum to 1", "PASS" if abs(grid.weight.sum() - 1) < 1e-9 else "FAIL",
        f"{len(grid)} members, sum {grid.weight.sum():.12f}")
    td, tt = tables["district"][0], tables["tehsil"][0]
    md, mt = tables["district"][1], tables["tehsil"][1]
    rec = (mt.assign(av=mt.fresh_t_ha * mt.tehsil.map(tt.set_index("tehsil").weight_ha))
           .groupby("member").av.sum() / tt.weight_ha.sum())
    dd = float((rec - md.set_index("member").fresh_t_ha).abs().max())
    add("district members = crop-area-weighted mean of the tehsil members", "PASS" if dd < 1e-9 else "FAIL",
        f"max |difference| {dd:.2e} t/ha; area {td.weight_ha.iloc[0]:,.1f} ha = sum of tehsils "
        f"{tt.weight_ha.sum():,.1f} ha")
    p5 = config.step_output(cfg, "05_lue", "central_unit.csv", required=False)
    if p5.exists():
        c5 = pd.read_csv(p5).merge(pt[["field_id", "central_t_ha", "modelled_pixel_share"]], left_on="unit",
                                   right_on="field_id", suffixes=("_field", "_pixel"))
        c5 = c5.dropna(subset=["central_t_ha"])
        if len(c5):
            dlt = c5.central_t_ha - c5.fresh_t_ha
            add("sampled fields: pixel-based parcel yield vs field-unit yield of step 05 (central, uncalibrated)", "INFO",
                f"{len(c5)} fields: pixel-based {c5.central_t_ha.mean():.2f}, field-based {c5.fresh_t_ha.mean():.2f} "
                f"t/ha; difference mean {dlt.mean():+.2f}, mean |difference| {dlt.abs().mean():.2f} t/ha, r "
                f"{np.corrcoef(c5.central_t_ha, c5.fresh_t_ha)[0, 1]:.3f} (yield per pixel then mean vs mean inputs "
                "then yield: the model is not linear in NDVI and EF)")
    ck = pd.DataFrame(ck)

    # ---- outputs
    for name, arr in (("pixel_s.npy", s_fill), ("pixel_code.npy", code), ("pixel_source.npy", source)):
        config.write_atomic(out / name, lambda tmp, a=arr: _save(tmp, a))
    config.write_atomic(out / "pixel_season.npz", lambda tmp: _savez(tmp, emergence=em, season_end=en_, peak=pk,
                                                                     peak_ndvi=peak_nd, n_used=n_used, n_ef=n_ef))
    combos.to_csv(out / "combos.csv", index_label="combo")
    grid.to_csv(out / "member_grid.csv", index=False)
    for col in ("emergence_median", "season_end_median"):
        pt[col] = pd.to_datetime(pt[col]).dt.strftime("%Y-%m-%d")
    pt.to_csv(out / "parcel_values.csv", index=False)
    ex = pd.DataFrame({"code": code[~ok], "n_used": n_used[~ok], "amp": amp[~ok]})
    ex["reason"] = [px.reason_text(int(r.code), r.n_used, float(r.amp), cl) for r in ex.itertuples()] if len(ex) else []
    ex.groupby(["code", "reason"]).size().rename("pixels").reset_index().to_csv(out / "excluded_pixels.csv", index=False)
    chk.to_csv(out / "unit_check.csv", index=False, date_format="%Y-%m-%d")
    ck.to_csv(out / "checks.csv", index=False)
    if not (ck.status == "FAIL").any():           # a step with a FAIL check leaves no manifest (never complete)
        config.write_manifest(out, "05b_pixel_model", paths, {"folder": out},
                              {"pixels": n, "modelled_pixels": int(ok.sum()), "parcels": len(pt), "members": len(grid),
                               "combinations": combos.to_dict("records")})
    pd.set_option("display.width", 250, "display.max_colwidth", 200)
    print(ck.to_string(index=False))
    cols = ["tehsil", "parcels", "pixels", "modelled_pixels", "weight_ha", "fresh_t_ha_mean", "fresh_t_ha_p10",
            "fresh_t_ha_p90", "central_t_ha"]
    print("\n", tt[cols].round(2).to_string(index=False))
    print("\n", td[["weight_ha", "fresh_t_ha_mean", "fresh_t_ha_p10", "fresh_t_ha_p90", "central_t_ha"]].round(2)
          .to_string(index=False))
    print(f"\n{n:,} pixels, {(time.time() - t_start) / 60:.1f} min; written to {out}")
    if (ck.status == "FAIL").any():
        raise SystemExit("step 05b stopped: FAIL checks above")


def _save(tmp, a):
    with open(tmp, "wb") as f:
        np.save(f, a)


def _savez(tmp, **arrays):
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)


if __name__ == "__main__":
    main()
