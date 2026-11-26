"""JPG maps of the GeoTIFFs of step 00c (step 00e, 00e_maps.py) and of the yield GeoTIFFs of steps 11 and 12: one map per
GeoTIFF, for looking at the data.

A map shows the values of the file, reduced to at most MAX_PX pixels on its longer side (mean of the pixels with
data for values, most frequent class for class rasters); a file with several bands (hourly or daily weather) is shown
as the mean of its bands. Nothing is computed for the model here and no map is read by another step.
Colours: one blue ramp from light (low) to dark (high) for values; fixed colours for classes; white = no data.
"""
import math
import re
from pathlib import Path

import numpy as np

MAX_PX = 1100                                   # longer side of the map raster
SURFACE, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]                     # sequential ramp, light = low
CLASS_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]                        # class 1, 2, 3 in this order
SCL_VALID = {4: "vegetation (class 4)", 5: "not vegetated (class 5)"}     # Sentinel-2 scene classification

_fields = {}


def describe(path, cfg):
    """What a GeoTIFF of step 00c holds, from its name and folder: kind ('value' or 'class'), title, unit, colour
    range (None = 2nd to 98th percentile of the map) and, for class rasters, how its values map to classes."""
    name, parent = path.stem, path.parent.name
    cl = cfg["stac"]["landsat"]
    fill, bad = 1 << cl["fill_bit"], sum(1 << b for b in cl["masked_bits"])
    clear = {"classes": ["clear", "not clear"], "note": "white = not observed"}
    if parent == "granules":
        item, band = name.rsplit("_", 1)
        if band == cfg["stac"]["sentinel2"]["assets"]["scl"]:
            valid = cfg["stac"]["sentinel2"]["valid_scl"]
            labels = [SCL_VALID.get(v, f"class {v}") for v in valid[:2]] + ["other classes (cloud, shadow, water, snow ...)"]
            return {"kind": "class", "title": f"Sentinel-2 scene classification (SCL), {item[11:19]}", "sub": item,
                    "classes": labels, "code": lambda a: np.where(a == 0, 0, np.where(a == valid[0], 1,
                                                                  np.where(a == valid[1], 2, 3))),
                    "note": "white = no data or outside the pixel windows of the units"}
        return {"kind": "value", "title": f"Sentinel-2 {band} surface reflectance, {item[11:19]}", "sub": item,
                "unit": "surface reflectance = (DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE", "range": None,
                "note": "white = no data or outside the pixel windows of the units"}
    if parent == "scenes":
        m = re.match(r"(.+_T\d)_(.+)$", name)
        item, band = m.group(1), m.group(2)
        if band == cl["qa_asset"]:
            return {"kind": "class", "title": f"Landsat QA_PIXEL: clear pixels, {item.split('_')[3]}", "sub": item,
                    "classes": ["clear", f"not clear (bits {cl['masked_bits']})"],
                    "code": lambda a: np.where((a & fill) != 0, 0, np.where((a & bad) == 0, 1, 2)),
                    "note": "white = outside the scene"}
        kelvin = band.startswith("lwir")
        return {"kind": "value", "title": f"Landsat {band} {'surface temperature' if kelvin else 'surface reflectance'}, "
                                          f"{item.split('_')[3]}", "sub": item,
                "unit": "K" if kelvin else "surface reflectance", "range": None, "note": "white = outside the scene"}
    if parent in ("raw", "raw_precip"):
        var = name.rsplit("_", 1)[1]
        return {"kind": "value", "title": f"ERA5-Land hourly {var}, {name.split('_')[2]}-{name.split('_')[3]}",
                "sub": name, "unit": None, "range": None, "note": "pixel = grid node"}
    if parent == "ndvi":
        kind, date = name.rsplit("_", 2)[1:]
        if kind == "clear":
            return {"kind": "class", "title": f"Sentinel-2 clear mask, {date}", "sub": name, **clear,
                    "code": lambda a: np.where(a == 255, 0, np.where(a == 1, 1, 2))}
        if kind == "ndvi":
            return {"kind": "value", "title": f"Sentinel-2 NDVI, {date}", "sub": name, "unit": "NDVI",
                    "range": (0.0, 1.0), "note": "white = no valid pixel or outside the pixel windows of the units"}
        return {"kind": "value", "title": f"Sentinel-2 {kind} surface reflectance, {date}", "sub": name,
                "unit": "surface reflectance", "range": None,
                "note": "white = no valid pixel or outside the pixel windows of the units"}
    if parent == "landsat":
        m = re.match(r"(.+_T\d)_(.+)$", name)
        item, kind = m.group(1), m.group(2)
        day = item.split("_")[3]
        if kind == "clear":
            return {"kind": "class", "title": f"Landsat clear mask, {day}", "sub": item, **clear,
                    "code": lambda a: np.where(a == 255, 0, np.where(a == 1, 1, 2)), "note": "white = outside the scene"}
        unit = {"ef": "evaporative fraction EF", "eta": "ETa (mm per day)"}.get(kind, f"{kind.split('_')[0]} (W m-2)")
        return {"kind": "value", "title": f"Landsat {unit}, {day}", "sub": item, "unit": unit,
                "range": (0.0, 1.0) if kind == "ef" else None, "note": "white = not computed (not clear or outside the scene)"}
    if path.parent.parent.name == "dem":
        return {"kind": "value", "title": f"Terrain elevation ({cfg['stac']['dem']['collection']})", "sub": name,
                "unit": "elevation (m)", "range": None, "note": "white = no data"}
    if parent == "11_pixel_map":                                  # yield GeoTIFF of step 11: band 1 = final yield
        return {"kind": "value", "title": "Fresh yield per 10 m pixel, final product (t/ha)", "sub": name, "band": 1,
                "unit": "fresh yield (t/ha)", "range": None, "note": "white = no crop parcel of the crop map"}
    if parent == "12_parcel_map":                                 # yield GeoTIFF of step 12: band 1 = final yield
        return {"kind": "value", "title": "Fresh yield per parcel, parcel-mean model (t/ha)", "sub": name, "band": 1,
                "unit": "fresh yield (t/ha)", "range": None, "note": "white = no crop parcel of the crop map"}
    if parent == "weather":
        var = name.split(cfg["run_id"] + "_", 1)[-1]
        return {"kind": "value", "title": f"Daily weather: {var}", "sub": name, "unit": None, "range": None,
                "note": "pixel = weather cell of the run; white = cell not in the run"}
    return {"kind": "value", "title": name, "sub": name, "unit": None, "range": None, "note": "white = no data"}


def read_reduced(ds, classes, band=None):
    """(array reduced to at most MAX_PX on its longer side as a masked array, reduction factor, bands averaged).
    Values: mean of the pixels with data; classes: most frequent value of the pixels with data. band: that band only
    (else the mean of all bands)."""
    from rasterio.enums import Resampling
    f = max(1, math.ceil(max(ds.width, ds.height) / MAX_PX))
    shape = (max(1, math.ceil(ds.height / f)), max(1, math.ceil(ds.width / f)))
    res = Resampling.mode if classes else Resampling.average
    if ds.count == 1 or band:
        return ds.read(band or 1, out_shape=shape, masked=True, resampling=res), f, 1
    a = ds.read(out_shape=(ds.count, *shape), masked=True, resampling=res)
    return a.mean(axis=0), f, ds.count


def enlarge(a, steps=2):
    """Points of data in a mostly empty map, enlarged by `steps` pixels on every side so that they can be seen."""
    a = np.ma.array(a, copy=True)
    for _ in range(steps):
        src = a.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sh = np.ma.masked_all_like(src)
            ys = slice(max(dy, 0), src.shape[0] + min(dy, 0)), slice(max(-dy, 0), src.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), src.shape[1] + min(dx, 0)), slice(max(-dx, 0), src.shape[1] + min(-dx, 0))
            sh[ys[0], xs[0]] = src[ys[1], xs[1]]
            take = np.ma.getmaskarray(a) & ~np.ma.getmaskarray(sh)
            a[take] = sh[take]
    return a


def load_fields(gpkg):
    if gpkg and gpkg not in _fields:
        import geopandas as gpd
        _fields[gpkg] = gpd.read_file(gpkg, layer="fields")
    return _fields.get(gpkg)


def make_map(tif, jpg, cfg, fields_gpkg=None):
    """One GeoTIFF -> one JPG map. Returns a row for the map table (what is shown, value range, reduction)."""
    import rasterio
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    tif, jpg = Path(tif), Path(jpg)
    d = describe(tif, cfg)
    row = {"tif": tif.name, "jpg": jpg.name, "shown": d["title"], "kind": d["kind"]}
    with rasterio.open(tif) as ds:
        a, f, bands = read_reduced(ds, d["kind"] == "class", d.get("band"))
        b, crs, tags = ds.bounds, ds.crs, ds.tags()
        scale, offset = ds.scales[0], ds.offsets[0]
        geographic, res = bool(crs and crs.is_geographic), ds.res[0]
        size = (ds.width, ds.height)
    unit = d.get("unit") or " ".join(x for x in (tags.get("variable"), f"({tags['units']})" if tags.get("units") else "") if x)
    fig = Figure(figsize=(8, 8.8), dpi=150, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.12, 0.20, 0.78, 0.68])
    ax.set_facecolor(SURFACE)
    extent = (b.left, b.right, b.bottom, b.top)
    share = float((~np.ma.getmaskarray(a)).mean())
    big = 0 < share < 0.02                                       # scattered small units: points enlarged for the eye
    notes = [d["note"]]
    if d["kind"] == "class":
        code = np.ma.masked_equal(d["code"](np.ma.filled(a, 0) if a.dtype.kind in "iu" else np.ma.filled(a, 0).astype(int)), 0)
        code = np.ma.masked_where(np.ma.getmaskarray(a) | np.ma.getmaskarray(code), code)
        if big:
            code = enlarge(code)
        n = len(d["classes"])
        ax.imshow(code, extent=extent, cmap=ListedColormap(CLASS_COLOURS[:n]), norm=BoundaryNorm(np.arange(0.5, n + 1), n),
                  interpolation="nearest")
        fig.legend(handles=[Patch(facecolor=CLASS_COLOURS[i], edgecolor="none", label=t) for i, t in enumerate(d["classes"])],
                   loc="lower left", bbox_to_anchor=(0.12, 0.085), ncols=1 if n > 2 else 2, frameon=False, fontsize=8.5,
                   labelcolor=INK2, handlelength=1.2, handleheight=1.0)
        row.update(value_min=np.nan, value_max=np.nan, classes="; ".join(d["classes"]))
    else:
        v = np.ma.masked_invalid(a.astype(float))
        if scale != 1 or offset != 0:
            v = v * scale + offset                               # physical value from the stored digital numbers
        if big:
            v = enlarge(v)
        lo, hi = (float(v.min()), float(v.max())) if v.count() else (np.nan, np.nan)
        if d["range"]:
            vmin, vmax = d["range"]
        elif v.count():
            vmin, vmax = (float(x) for x in np.percentile(v.compressed(), [2, 98]))
            notes.append("colour scale: 2nd to 98th percentile of the map")
        else:
            vmin, vmax = 0.0, 1.0
        if not vmax > vmin:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        im = ax.imshow(v, extent=extent, cmap=LinearSegmentedColormap.from_list("blues", BLUES), vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        cax = fig.add_axes([0.12, 0.105, 0.52, 0.016])
        ext = {(True, True): "both", (True, False): "min", (False, True): "max"}.get((lo < vmin, hi > vmax), "neither")
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend=ext)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5, colors=MUTED, length=2, width=0.5)
        cb.set_label(unit + (f"  -  mean of {bands} bands" if bands > 1 else ""), fontsize=8.5, color=INK2, loc="left")
        row.update(value_min=lo, value_max=hi, classes="")
    if geographic:
        ax.set_aspect(1 / math.cos(math.radians((b.top + b.bottom) / 2)))
    fields = load_fields(fields_gpkg)
    if fields is not None and crs is not None:
        g = fields.to_crs(crs)
        bw = g.bounds
        if float(((bw.maxx - bw.minx) / (res * f)).median()) >= 8:   # units large enough to draw their outlines
            g.boundary.plot(ax=ax, linewidth=0.45, color=INK2, alpha=0.8)
            notes.append("lines: units of the run")
            ax.set_xlim(b.left, b.right)
            ax.set_ylim(b.bottom, b.top)
    if big:
        notes.append("small units enlarged by 2 map pixels to be visible")
    for s in ax.spines.values():
        s.set_color(AXIS)
        s.set_linewidth(0.6)
    ax.tick_params(labelsize=7, colors=MUTED, length=2.5, width=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_xlabel("longitude (degrees)" if geographic else f"easting (m), {crs.to_string() if crs else ''}", fontsize=7.5,
                  color=MUTED)
    ax.set_ylabel("latitude (degrees)" if geographic else "northing (m)", fontsize=7.5, color=MUTED)
    fig.text(0.12, 0.945, d["title"], fontsize=12, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.12, 0.918, f"{cfg['run_id']}  -  {d['sub']}", fontsize=8, color=INK2, ha="left", va="bottom")
    fig.text(0.12, 0.028, f"{tif.name}: {size[0]} x {size[1]} pixels of {res:g} {'degrees' if geographic else 'm'}"
             + (f", shown 1 : {f}" if f > 1 else "") + "\n" + "; ".join(notes), fontsize=7, color=MUTED, ha="left",
             va="bottom", linespacing=1.5)
    jpg.parent.mkdir(parents=True, exist_ok=True)
    tmp = jpg.with_name(jpg.name + ".part")
    fig.savefig(tmp, format="jpg", dpi=150, facecolor=SURFACE, pil_kwargs={"quality": 88})
    tmp.replace(jpg)
    row.update(pixels_with_data_share=round(share, 6), reduction=f, bands=bands)
    return row
