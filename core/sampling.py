"""Field sample: stratified simple random sampling without replacement, strata = ERA5-Land 0.1 deg cells (Cochran
1977, stratified random sampling).

Frame of a cell = its crop-map parcels that meet every eligibility rule. If a cell has fewer than min_fields_per_cell
such parcels, the rules in relax_order are relaxed one at a time (to `relaxed_values` where given, else dropped) until
it has, so clean parcels are preferred. n_h = min(fields_per_cell, N_h) parcels are drawn at random from the N_h frame
parcels: inclusion probability pi_i = n_h / N_h, exact. Yield per hectare of a cell: separate ratio estimator
R_h = sum(a_i y_i / pi_i) / sum(a_i / pi_i) = sum(a_i y_i) / sum(a_i) (pi equal within the cell); district
sum_h W_h R_h with W_h = cell share of the mapped crop area. Parcels outside the frame (mixed segments, small
parcels) have pi = 0: the cell value is applied to them (stated assumption; frame coverage reported by step 02)."""
import numpy as np
import pandas as pd


def parcel_frame(cfg):
    """All parcels of the crop map of the run (steps 02, 00c, 11): (crop map with field_id, frame table, UTM CRS).
    field_id = first three letters of the area name + row number in the crop map (the ids of the step-02 sample);
    cell = nearest ERA5-Land grid point of the parcel's representative point."""
    import geopandas as gpd
    from . import config
    aoi, inp = cfg["aoi"], cfg["aoi"]["inputs"]
    step = inp["era5_cell_deg"]
    cm = gpd.read_file(config.input_path(cfg, "crop_mask"), layer=inp["crop_mask_layer"])
    cm["field_id"] = [f"{aoi['name'][:3].upper()}{i:06d}" for i in range(len(cm))]
    utm = cm.estimate_utm_crs()
    pt_utm = cm.to_crs(utm).geometry.representative_point()
    pt = pt_utm.to_crs(4326)
    frame = pd.DataFrame({"field_id": cm.field_id, "parcel_id": cm.parcel_id, "tehsil": cm[inp["crop_mask_tehsil_field"]],
                          "source": cm.source, "prob": cm.prob, "review": cm.review, "mixed": cm.mixed,
                          "area_ha": cm.area_ha, "lon": pt.x, "lat": pt.y, "x_m": pt_utm.x, "y_m": pt_utm.y})
    frame["cell_lat"] = (frame.lat / step).round() * step
    frame["cell_lon"] = (frame.lon / step).round() * step
    frame["cell"] = [f"{a:.1f}_{o:.1f}" for a, o in zip(frame.cell_lat, frame.cell_lon)]
    return cm, frame, utm


def eligibility(frame, el, relaxed):
    """Adds ok_<rule> (strict) and ok_<rule>_relaxed columns; returns the rule names."""
    f = frame
    f["ok_source"] = f.source.isin(el["source"])
    f["ok_source_relaxed"] = f.source.isin(relaxed.get("source", el["source"]))   # last-resort lists may include mixed segments
    f["ok_mixed"] = ~f.mixed.astype(bool) if el.get("mixed") is False else True
    f["ok_review"] = ~f.review.astype(bool) if el.get("review") is False else True
    f["ok_min_prob"] = f.prob.fillna(0) >= el.get("min_prob", 0)
    f["ok_min_area_ha"] = f.area_ha >= el["min_area_ha"]
    f["ok_min_area_ha_relaxed"] = f.area_ha >= relaxed.get("min_area_ha", el["min_area_ha"])
    return ["source", "mixed", "review", "min_prob", "min_area_ha"]


def select_srs(frame, rules, sc):
    """frame: field_id, cell, area_ha, ok_* columns. Returns (sample: field_id, rules_relaxed, inclusion_prob,
    frame_n; frames: cell, frame_n, frame_area_ha, sampled, rules_relaxed)."""
    n_cell, n_min = int(sc["fields_per_cell"]), int(sc.get("min_fields_per_cell", 1))
    rng = np.random.default_rng(int(sc["seed"]))
    picks, frames = [], []
    for cell in sorted(frame.cell.unique()):
        g = frame[frame.cell == cell].sort_values("field_id")
        cols = {k: f"ok_{k}" for k in rules}
        relax, done = list(sc.get("relax_order", [])), []
        while True:
            fr = g[np.logical_and.reduce([g[c].to_numpy(bool) for c in cols.values()])]
            if len(fr) >= n_min or not relax:
                break
            k = relax.pop(0)
            if f"ok_{k}_relaxed" in g:
                cols[k] = f"ok_{k}_relaxed"
            else:
                cols.pop(k, None)
            done.append(k)
        N = len(fr)
        n = min(n_cell, N)
        note = ";".join(done) or "none"
        frames.append({"cell": cell, "frame_n": N, "frame_area_ha": float(fr.area_ha.sum()), "sampled": n,
                       "rules_relaxed": note})
        if n == 0:
            continue
        for r in fr.iloc[np.sort(rng.choice(N, n, replace=False))].itertuples():
            picks.append({"field_id": r.field_id, "rules_relaxed": note, "inclusion_prob": n / N, "frame_n": N})
    return pd.DataFrame(picks), pd.DataFrame(frames)


def nearest_km(xy):
    """Distance of every sampled field to its nearest sampled neighbour (km) - spatial spread, reported only."""
    xy = np.asarray(xy, float)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1) / 1000


def stratified_estimate(y, cell, area_ha, cell_area, cell_frame_n=None):
    """Yield per hectare of the district: separate ratio estimator (Cochran 1977, ratio estimation in stratified
    sampling). Fields are drawn with equal probability within a cell, and the target is production / area, so the cell
    estimate is R_h = sum(a_i y_i) / sum(a_i) (a_i parcel area, y_i yield in t/ha), and the district value is
    sum_h W_h R_h with W_h the cell's share of the mapped crop area. Variance of R_h:
    (1 - n_h/N_h) / (n_h abar_h^2) * sum(a_i (y_i - R_h))^2 / (n_h - 1), N_h = frame parcels of the cell. Cells with
    one field use the pooled within-cell variance (flagged)."""
    df = pd.DataFrame({"y": np.asarray(y, float), "a": np.asarray(area_ha, float), "cell": np.asarray(cell)})
    df["ay"] = df.a * df.y
    g = df.groupby("cell").agg(ay=("ay", "sum"), a=("a", "sum"), abar=("a", "mean"), n=("y", "size"))
    g["R"] = g.ay / g.a
    df["d"] = df.a * (df.y - df.cell.map(g.R))
    g["s2"] = df.groupby("cell").d.apply(lambda d: (d ** 2).sum() / (len(d) - 1) if len(d) > 1 else np.nan) / g.abar ** 2
    W = cell_area.reindex(g.index) / cell_area.reindex(g.index).sum()
    pooled = float(np.nanmean(g.s2[g.n > 1])) if (g.n > 1).any() else np.nan
    s2 = g.s2.where(g.n > 1, pooled)
    fpc = 1 - g.n / cell_frame_n.reindex(g.index) if cell_frame_n is not None else 1.0
    se = float(np.sqrt((W ** 2 * fpc * s2 / g.n).sum()))
    return float((W * g.R).sum()), se, int((g.n == 1).sum())


def ratio_weights(cell, area_ha, cell_area):
    """Unit weights (ha) that make every area-weighted mean over the fields equal the separate ratio estimator:
    w_i = A_h * a_i / sum_{fields of cell h} a_j (sums to the cell's crop area A_h)."""
    a = pd.Series(np.asarray(area_ha, float), index=np.asarray(cell))
    return (np.asarray(cell_area.reindex(a.index), float) * a.to_numpy() / a.groupby(level=0).transform("sum").to_numpy())
