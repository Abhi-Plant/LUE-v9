"""Static figures (PNG): tehsil NDVI season panels, tehsil yield map, sensitivity bars.
One visual system: surface #fcfcfb, blue for the data series, orange for flagged/rejected marks, ink text."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9994", "#e6e5e1"
BLUE, ORANGE, SHADE = "#256abf", "#eb6834", "#cde2fb"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _axes_style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def season_panels(path, daily, obs, season, title):
    """Small multiples, one per tehsil: smoothed daily NDVI (line), used / rejected observations, season span."""
    tehsils = sorted(daily.tehsil.unique())
    ncol = 5
    nrow = int(np.ceil(len(tehsils) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.1 * nrow), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, t in zip(axes.flat, tehsils):
        _axes_style(ax)
        s = season[season.tehsil == t].iloc[0]
        ax.axvspan(s.emergence, s.season_end, color=SHADE, alpha=0.6, lw=0, zorder=0)
        g = daily[daily.tehsil == t]
        ax.plot(g.date, g.ndvi_smooth, color=BLUE, lw=2, zorder=3)
        o = obs[obs.tehsil == t]
        u, r = o[o.status == "used"], o[o.status != "used"]
        ax.scatter(u.date, u.ndvi, s=14, color=BLUE, edgecolors=SURFACE, linewidths=0.8, zorder=4)
        ax.scatter(r.date, r.ndvi, s=16, facecolors="none", edgecolors=ORANGE, linewidths=1.1, zorder=4)
        ax.set_title(t, loc="left", fontsize=9, color=INK)
        ax.text(0.02, 0.04, f"{s.emergence:%d %b} - {s.season_end:%d %b} ({s.season_days} d)",
                transform=ax.transAxes, fontsize=7, color=INK2)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_ylim(-0.1, 1.0)
    for ax in axes.flat[len(tehsils):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=BLUE, lw=2, label="NDVI, cleaned daily series"),
               Line2D([], [], marker="o", ls="", ms=5, mfc=BLUE, mec=SURFACE, label="observation used"),
               Line2D([], [], marker="o", ls="", ms=5, mfc="none", mec=ORANGE, label="observation rejected (cloud / fog / haze)"),
               Patch(facecolor=SHADE, alpha=0.6, label="season: emergence to season end")]
    fig.legend(handles=handles, loc="lower left", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def label_points(gdf, min_dist=0.12, clearance=0.10):
    """Label anchor per polygon; when two anchors are closer than min_dist (degrees), the larger polygon's label moves
    to the largest part of that polygon left after removing a clearance buffer around the smaller one (enclaves)."""
    from shapely.geometry import Point
    pts = [g.representative_point() for g in gdf.geometry]
    geoms = list(gdf.geometry)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i != j and pts[i].distance(pts[j]) < min_dist and geoms[i].area > geoms[j].area:
                rest = geoms[i].difference(geoms[j].buffer(clearance))
                inner = rest.buffer(-0.03)                    # keep the label off the boundary
                area = inner if not inner.is_empty else rest
                if area.is_empty:
                    continue
                x0, y0, x1, y1 = area.bounds                  # grid candidates inside: farthest from other labels
                cand = [Point(x, y) for x in np.arange(x0, x1, 0.01) for y in np.arange(y0, y1, 0.01)
                        if area.contains(Point(x, y))] or [area.representative_point()]
                others = [p for k, p in enumerate(pts) if k != i]
                pts[i] = max(cand, key=lambda c: min(c.distance(o) for o in others))
    return pts


def value_bins(values, step=2.0, max_classes=6):
    """Data-driven class edges (open-ended first and last class), `step` wide, at most `max_classes` classes."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    lo, hi = np.floor(v.min() / step) * step, np.ceil(v.max() / step) * step
    while (hi - lo) / step > max_classes:
        step *= 2
        lo, hi = np.floor(v.min() / step) * step, np.ceil(v.max() / step) * step
    inner = list(np.arange(lo + step, hi, step))
    return [-np.inf] + inner + [np.inf] if inner else [-np.inf, (lo + hi) / 2, np.inf]


def grid_map(path, tb, cells, frame, title, note, tehsil_values=None, points=None, point_label="",
             cell_value="cell_crop_ha", cell_bins=None, cell_unit="ha",
             cell_legend="ERA5-Land cell (0.1 deg): mapped crop area", cell_labels=False):
    """Map in the format of the WOFOST district maps: ERA5-Land 0.1 deg cells shaded by a cell value (default mapped
    crop area), tehsil and district boundaries, mapped crop parcels (frame: lon, lat), optional point markers
    (points: lon, lat), tehsil value boxes (tehsil_values {tehsil: (value line, detail line)}) or italic tehsil names,
    the value printed in each cell (cell_labels) and a note box at the lower left."""
    import matplotlib.patheffects as pe

    if cell_bins is None:
        top = max(2000.0, float(cells[cell_value].max()) + 1)
        bins, open_ends = [0, 100, 250, 500, 1000, top], False
    else:
        bins, open_ends = list(cell_bins), True
    ncls = len(bins) - 1
    blues = RAMP[:ncls]
    edges = [b if np.isfinite(b) else (-1e9 if b < 0 else 1e9) for b in bins]
    cmap, norm = ListedColormap(blues), BoundaryNorm(edges, ncls)

    fig, ax = plt.subplots(figsize=(11, 11.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    shaded = cells[cells[cell_value].notna()]
    if cells[cell_value].isna().any():
        cells[cells[cell_value].isna()].plot(ax=ax, color="#ecebe8", edgecolor=SURFACE, linewidth=2, zorder=1)
    shaded.plot(ax=ax, column=cell_value, cmap=cmap, norm=norm, edgecolor=SURFACE, linewidth=2, zorder=1)
    if cell_labels:
        for r in shaded.itertuples():
            ax.text(r.cell_lon, r.cell_lat - 0.012, f"{getattr(r, cell_value):.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=INK, zorder=10,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground=SURFACE)])
    ax.scatter(frame.lon, frame.lat, s=0.15, c=MUTED, alpha=0.35, linewidths=0, zorder=2, rasterized=True)
    tb.boundary.plot(ax=ax, color=INK2, linewidth=0.7, zorder=3)
    tb.dissolve().boundary.plot(ax=ax, color=INK, linewidth=1.6, zorder=4)
    if points is not None:
        ax.scatter(points.lon, points.lat, s=46, c=ORANGE, edgecolors=SURFACE, linewidths=1.5, zorder=7)
    for r, pt in zip(tb.itertuples(), label_points(tb, min_dist=0.22 if tehsil_values else 0.12)):
        if tehsil_values is None:
            ax.annotate(r.tehsil, (pt.x, pt.y), ha="center", va="center", fontsize=8, color=INK2, style="italic",
                        zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.7))
        elif r.tehsil in tehsil_values:
            value, detail = tehsil_values[r.tehsil]
            ax.annotate(f"{r.tehsil}\n{value}" + (f"\n{detail}" if detail else ""), (pt.x, pt.y), ha="center",
                        va="center", fontsize=8, color=INK, zorder=8, linespacing=1.25,
                        bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=INK2, lw=0.6, alpha=0.92))
    for r in cells.itertuples():
        ax.text(r.cell_lon - 0.047, r.cell_lat + 0.047, r.cell, fontsize=5.5, color=INK2, ha="left", va="top", zorder=5)
    ax.set_aspect(1 / np.cos(np.radians(float(cells.cell_lat.mean()))))
    ax.set_xlabel("Longitude (degrees E)", color=INK2, fontsize=9)
    ax.set_ylabel("Latitude (degrees N)", color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(MUTED)
    if open_ends:
        labels = [f"< {bins[1]:g} {cell_unit}"] + [f"{bins[i]:g}-{bins[i + 1]:g} {cell_unit}"
                                                  for i in range(1, ncls - 1)] + [f"> {bins[-2]:g} {cell_unit}"]
    else:
        labels = [f"{int(bins[i])}-{int(bins[i + 1])} {cell_unit}" for i in range(ncls - 1)] + \
                 [f"> {int(bins[-2])} {cell_unit}"]
    handles = [Patch(facecolor=c, edgecolor="none", label=lab) for c, lab in zip(blues, labels)]
    if cells[cell_value].isna().any():
        handles.append(Patch(facecolor="#ecebe8", edgecolor="none", label="no estimate"))
    if points is not None:
        handles.append(Line2D([], [], marker="o", ls="", ms=8, mfc=ORANGE, mec=SURFACE, mew=1.5,
                              label=f"{point_label} ({len(points):,})"))
    handles += [Line2D([], [], marker="o", ls="", ms=3, mfc=MUTED, mec="none",
                       label=f"Mapped crop parcel ({len(frame):,})"),
                Line2D([], [], color=INK2, lw=0.7, label="Tehsil boundary (LGD)"),
                Line2D([], [], color=INK, lw=1.6, label="District boundary (LGD)")]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True, title=cell_legend, title_fontsize=8)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor(MUTED)
    if note:                                             # free band below the map for the note box
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0 - 0.045 * (note.count("\n") + 1) * (y1 - y0) / 2.5, y1)
        ax.text(0.01, 0.01, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=INK, zorder=9,
                linespacing=1.4, bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=MUTED, lw=0.8))
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def sobol_bars(path, panels, title):
    """Sobol indices, one panel per output: total-order ST (blue) and first-order S1 (orange) per factor, with
    bootstrap confidence intervals. panels: list of (panel title, DataFrame factor, label, S1, S1_conf, ST, ST_conf)."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.6), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (ptitle, t) in zip(axes, panels):
        _axes_style(ax)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(t))
        ax.barh(y + 0.18, t.ST, height=0.34, xerr=t.ST_conf, color=BLUE, edgecolor=SURFACE,
                error_kw=dict(ecolor=INK2, lw=0.8, capsize=2), label="total effect ST")
        ax.barh(y - 0.18, t.S1.clip(lower=0), height=0.34, xerr=t.S1_conf, color=ORANGE, edgecolor=SURFACE,
                error_kw=dict(ecolor=INK2, lw=0.8, capsize=2), label="first-order effect S1")
        for yi, v, c in zip(y, t.ST, t.ST_conf):
            ax.text(min(v + c, 1.0) + 0.02, yi + 0.18, f"{v:.2f}", va="center", fontsize=7, color=INK2)
        ax.set_yticks(y, t.label, fontsize=8, color=INK)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Sobol index (share of output variance)", fontsize=8, color=INK2)
        ax.set_title(ptitle, loc="left", fontsize=9, color=INK)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def sensitivity_bars(path, table, central, title, xlabel, reference=None, reference_label=None):
    """Horizontal range bars (low-high result per factor) around the central value, largest swing on top;
    optional dashed reference line (e.g. an official benchmark)."""
    t = table.sort_values("swing", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(t) + 1.8), facecolor=SURFACE)
    _axes_style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(t))
    ax.barh(y, t.high_value - t.low_value, left=t.low_value, height=0.5, color=BLUE, edgecolor=SURFACE)
    ax.axvline(central, color=INK, lw=1)
    lo = min(t.low_value.min(), reference if reference is not None else np.inf)
    hi = t.high_value.max()
    ax.set_xlim(lo - 0.06 * (hi - lo), hi + 0.12 * (hi - lo))
    if reference is not None:
        ax.axvline(reference, color=MUTED, lw=1, ls="--")
        ax.text(reference, len(t) - 0.4, f" {reference_label}", fontsize=7, color=INK2, va="bottom", ha="left")
    ax.text(central, len(t) - 0.4, f" central {central:.1f}", fontsize=7, color=INK, va="bottom", ha="left")
    ax.set_yticks(y, t.label, fontsize=8, color=INK)
    for yi, (lo, hi) in enumerate(zip(t.low_value, t.high_value)):
        ax.text(hi + 0.2, yi, f"{lo:.1f}-{hi:.1f}", va="center", fontsize=7, color=INK2)
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


TEAL, TEAL2 = "#1f8a70", "#7cc4ae"


def curve_panels(path, daily, period, season, title):
    """v0.3 small multiples, one per tehsil: daily fAPAR (double logistic) and Landsat EF (smoothing splines) with
    the 16-day means they were fitted to (marks at mid-period; fAPAR bars span the period), season span."""
    tehsils = sorted(daily.tehsil.unique())
    ncol = 5
    nrow = int(np.ceil(len(tehsils) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.2 * nrow), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, t in zip(axes.flat, tehsils):
        _axes_style(ax)
        s = season[season.tehsil == t].iloc[0]
        ax.axvspan(s.emergence, s.season_end, color=SHADE, alpha=0.5, lw=0, zorder=0)
        g = daily[daily.tehsil == t]
        p = period[(period.tehsil == t) & (period.period <= g.date.max())]
        ax.hlines(p.fapar, p.period, p.period_end, color=BLUE, lw=1.2, alpha=0.6, zorder=2)
        ax.scatter(p.date, p.fapar, s=14, color=BLUE, edgecolors=SURFACE, linewidths=0.8, zorder=4)
        ax.plot(g.date, g.fapar, color=BLUE, lw=2, zorder=3)
        ax.scatter(p.date, p.ef_l8, s=12, marker="s", color=TEAL, edgecolors=SURFACE, linewidths=0.6, zorder=4)
        ax.scatter(p.date, p.ef_l9, s=12, marker="^", color=TEAL2, edgecolors=SURFACE, linewidths=0.6, zorder=4)
        ax.plot(g.date, g.ef_l8, color=TEAL, lw=1.5, zorder=3)
        ax.plot(g.date, g.ef_l9, color=TEAL2, lw=1.5, ls="--", zorder=3)
        ax.set_title(t, loc="left", fontsize=9, color=INK)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_ylim(0, 1.0)
    for ax in axes.flat[len(tehsils):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=BLUE, lw=2, marker="o", ms=4, label="fAPAR: double logistic fitted to 16-day means"),
               Line2D([], [], color=TEAL, lw=1.5, marker="s", ms=4, label="EF Landsat 8: smoothing spline"),
               Line2D([], [], color=TEAL2, lw=1.5, ls="--", marker="^", ms=4, label="EF Landsat 9: smoothing spline"),
               Patch(facecolor=SHADE, alpha=0.5, label="season (central threshold)")]
    fig.legend(handles=handles, loc="lower left", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- v0.3 biomass curve-fitting figures (approach 2)
# One hue (blue) for the data of interest, muted grey for context; one y-axis per chart; small multiples for tehsils.

def _months(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def _grid(n, ncol=5, h=3.0, sharey=True):
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, h * nrow), sharex=True, sharey=sharey, facecolor=SURFACE)
    return fig, np.atleast_1d(axes).ravel()


def _finish(fig, path, title, handles=None, ncol=4):
    if handles:
        fig.legend(handles=handles, loc="lower left", ncol=ncol, fontsize=8, frameon=False, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05 if handles else 0, 1, 0.95))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def period_inputs_panels(path, pdv, weights, title):
    """Step 1: the 16-day inputs of the growth rate - one panel per variable; tehsils grey, area-weighted district blue."""
    pdv = pdv.copy()
    pdv["ef"] = (pdv.ef_l8 + pdv.ef_l9) / 2
    pdv["w"] = pdv.unit.map(weights)
    spec = [("rs", "Solar radiation Rs (MJ m-2 d-1)"), ("tmean", "Mean air temperature (degC)"),
            ("fapar", "fAPAR (Sentinel-2)"), ("ef", "Evaporative fraction EF (Landsat 8/9, gaps filled)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), facecolor=SURFACE)
    for ax, (c, lab) in zip(axes, spec):
        _axes_style(ax)
        for _, g in pdv.groupby("unit"):
            ax.plot(g.mid, g[c], color=MUTED, lw=1, alpha=0.7, zorder=2)
        dm = pdv.groupby("mid").apply(lambda g: np.average(g[c], weights=g.w), include_groups=False)
        ax.plot(dm.index, dm.to_numpy(), color=BLUE, lw=2, marker="o", ms=4, mec=SURFACE, zorder=3)
        ax.set_title(lab, loc="left", fontsize=9, color=INK)
        _months(ax)
    handles = [Line2D([], [], color=MUTED, lw=1, label="each tehsil"),
               Line2D([], [], color=BLUE, lw=2, marker="o", ms=4, label="district (area-weighted)")]
    _finish(fig, path, title, handles, ncol=2)


def growth_fit_panels(path, daily, points, season, title):
    """Steps 2-4: 16-day growth rates (dots, bar = period span) and the fitted double-logistic daily curve."""
    tehs = sorted(daily.tehsil.unique())
    fig, axes = _grid(len(tehs))
    for ax, t in zip(axes, tehs):
        _axes_style(ax)
        s = season[season.tehsil == t].iloc[0]
        ax.axvspan(s.emergence, s.season_end, color=SHADE, alpha=0.45, lw=0, zorder=0)
        a = daily[daily.tehsil == t]
        p = points[(points.tehsil == t) & (points.mid <= a.date.max())]
        ax.hlines(p.growth_point_g_m2_d, p.period, p.period_end, color=INK2, lw=1.0, zorder=3)
        ax.scatter(p.mid, p.growth_point_g_m2_d, s=30, color=INK2, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        ax.plot(a.date, a.growth_curve_all_days_g_m2_d, color=BLUE, lw=2, zorder=5)
        pk = a.loc[a.growth_curve_all_days_g_m2_d.idxmax()]
        ax.set_title(t, loc="left", fontsize=9, color=INK)
        ax.text(0.02, 0.92, f"peak {pk.growth_curve_all_days_g_m2_d:.1f} on {pk.date:%d %b}", transform=ax.transAxes,
                fontsize=7, color=INK2)
        _months(ax)
    for ax in axes[len(tehs):]:
        ax.set_visible(False)
    top = max(float(daily.growth_curve_all_days_g_m2_d.max()), float(points.growth_point_g_m2_d.max()))
    axes[0].set_ylim(0, top * 1.22)                                  # headroom for the peak label (shared y)
    axes[0].set_ylabel("growth, g DM m-2 d-1", fontsize=8, color=INK2)
    handles = [Line2D([], [], color=INK2, marker="o", ls="-", lw=1, ms=5, label="16-day growth rate (bar = period)"),
               Line2D([], [], color=BLUE, lw=2, label="fitted double logistic, read daily"),
               Patch(facecolor=SHADE, alpha=0.45, label="season: emergence to season end (summed)")]
    _finish(fig, path, title, handles, ncol=3)


def fit_check_panels(path, points, title):
    """Step 3 check: curve mean over each period vs the 16-day growth rate (1:1), and residuals by period."""
    p = points.dropna(subset=["curve_mean_g_m2_d"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.4]})
    _axes_style(a1)
    a1.grid(axis="x", color=GRID, lw=0.6)
    hi = float(max(p.growth_point_g_m2_d.max(), p.curve_mean_g_m2_d.max())) * 1.05
    a1.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--", zorder=1)
    a1.scatter(p.growth_point_g_m2_d, p.curve_mean_g_m2_d, s=30, color=BLUE, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    r = p.growth_point_g_m2_d.corr(p.curve_mean_g_m2_d)
    rmse = float(np.sqrt(np.mean(p.difference ** 2)))
    a1.text(0.03, 0.95, f"r {r:.3f}   RMSE {rmse:.2f}   bias {p.difference.mean():+.2f} g m-2 d-1\nn = {len(p)} "
            "tehsil-periods", transform=a1.transAxes, fontsize=8, color=INK2, va="top")
    a1.set_xlim(0, hi)
    a1.set_ylim(0, hi)
    a1.set_xlabel("16-day growth rate, g DM m-2 d-1", fontsize=8, color=INK2)
    a1.set_ylabel("fitted curve averaged over the period", fontsize=8, color=INK2)
    a1.set_title("Fitted curve vs 16-day values (dashed = 1:1)", loc="left", fontsize=9, color=INK)
    _axes_style(a2)
    a2.axhline(0, color=MUTED, lw=1)
    a2.scatter(p.mid, p.difference, s=24, color=BLUE, edgecolors=SURFACE, linewidths=1, zorder=3)
    m = p.groupby("mid").difference.mean()
    a2.plot(m.index, m.to_numpy(), color=INK2, lw=1.5, zorder=4)
    a2.set_ylabel("curve - 16-day value, g DM m-2 d-1", fontsize=8, color=INK2)
    a2.set_title("Residual by period (dots: tehsils; line: mean)", loc="left", fontsize=9, color=INK)
    _months(a2)
    _finish(fig, path, title)


def cumulative_panels(path, daily, title):
    """Step 5: daily growth summed from emergence - cumulative biomass, t DM ha-1."""
    tehs = sorted(daily.tehsil.unique())
    fig, axes = _grid(len(tehs))
    for ax, t in zip(axes, tehs):
        _axes_style(ax)
        g = daily[(daily.tehsil == t) & daily.in_season]
        ax.fill_between(g.date, 0, g.biomass_cum_g_m2 * 0.01, color=SHADE, alpha=0.6, lw=0, zorder=1)
        ax.plot(g.date, g.biomass_cum_g_m2 * 0.01, color=BLUE, lw=2, zorder=3)
        last = g.iloc[-1]
        ax.scatter([last.date], [last.biomass_cum_g_m2 * 0.01], s=30, color=BLUE, edgecolors=SURFACE, linewidths=1.5,
                   zorder=4)
        ax.text(0.02, 0.92, f"{last.biomass_cum_g_m2 * 0.01:.1f} t DM/ha at season end ({last.date:%d %b})",
                transform=ax.transAxes, fontsize=7, color=INK2)
        ax.set_title(t, loc="left", fontsize=9, color=INK)
        _months(ax)
    for ax in axes[len(tehs):]:
        ax.set_visible(False)
    axes[0].set_ylabel("cumulative biomass, t DM ha-1", fontsize=8, color=INK2)
    _finish(fig, path, title)


def period_share_bars(path, per, title):
    """Share of season biomass formed in each half-month: district mean (bars) with the tehsil range (whisker)."""
    s = per.groupby("period").share_pct.agg(["mean", "min", "max"])
    lab = [f"{ts:%d %b}" for ts in s.index]
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURFACE)
    _axes_style(ax)
    x = np.arange(len(s))
    ax.bar(x, s["mean"], width=0.8, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.vlines(x, s["min"], s["max"], color=INK2, lw=1.2, zorder=3)
    for i, v in enumerate(s["mean"]):
        if v >= 10:
            ax.text(i + 0.12, v + 0.6, f"{v:.0f} %", ha="left", fontsize=8, color=INK2)
    ax.set_xticks(x, lab)
    ax.set_xlabel("16-day period starting", fontsize=8, color=INK2)
    ax.set_ylabel("share of season biomass, %", fontsize=8, color=INK2)
    _finish(fig, path, title)


def tehsil_biomass_yield(path, cen, title):
    """Season biomass and fresh yield per tehsil (central configuration) - two panels, one axis each."""
    c = cen.sort_values("biomass_g_m2")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, facecolor=SURFACE)
    y = np.arange(len(c))
    for ax, v, lab in ((a1, c.biomass_g_m2 * 0.01, "Season biomass, t DM ha-1"),
                       (a2, c.fresh_t_ha, "Fresh yield (uncalibrated), t ha-1")):
        _axes_style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.barh(y, v, height=0.7, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2)
        for i, val in enumerate(v):
            ax.text(val + v.max() * 0.01, i, f"{val:.1f}", va="center", fontsize=8, color=INK2)
        ax.set_title(lab, loc="left", fontsize=9, color=INK)
        ax.set_xlim(0, v.max() * 1.12)
    a1.set_yticks(y, c.unit)
    _finish(fig, path, title)


def method_comparison(path, table, central, title):
    """District fresh yield of the central configuration by calculation method (horizontal bars)."""
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(t) + 1.4), facecolor=SURFACE)
    _axes_style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    y = np.arange(len(t))
    col = [BLUE if m == central else MUTED for m in t.method]
    ax.barh(y, t.district_fresh_t_ha, height=0.7, color=col, edgecolor=SURFACE, linewidth=2, zorder=2)
    for i, (v, pc) in enumerate(zip(t.district_fresh_t_ha, t.change_pct)):
        ax.text(v + 0.3, i, f"{v:.1f} t/ha" + ("" if abs(pc) < 1e-9 else f"  ({pc:+.1f} %)"), va="center",
                fontsize=8, color=INK2)
    ax.set_yticks(y, t.method, fontsize=8)
    ax.set_xlim(0, t.district_fresh_t_ha.max() * 1.25)
    ax.set_xlabel("district fresh yield, central configuration, uncalibrated (t/ha)", fontsize=8, color=INK2)
    _finish(fig, path, title)
