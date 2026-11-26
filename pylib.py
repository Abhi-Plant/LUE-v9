"""Install the Python libraries the whole pipeline needs (steps 00a-00e, 01-10 and run.py).

  python pylib.py             look for every library, install the missing ones with pip, then look again
  python pylib.py --check     only report what is installed and what is missing; installs nothing
  python pylib.py --dry-run   show what pip would install; installs nothing

Libraries are installed into the Python that runs this script (the one you will run the pipeline with).
A library that is already installed is left as it is, unless it is older than the minimum given below.
Not a library, so not handled here: the Climate Data Store key for the weather export (README_run.md).
"""
import argparse
import importlib
import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path

# (pip name, import name, minimum version or None, used by)
# A minimum is given only where the code calls something that older versions do not have.
LIBRARIES = [
    ("numpy", "numpy", None, "all steps"),
    ("pandas", "pandas", None, "all steps"),
    ("PyYAML", "yaml", None, "all steps (config files)"),
    ("scipy", "scipy", "1.10", "steps 03-05 (smoothing spline make_smoothing_spline, curve fits, Savitzky-Golay filter)"),
    ("matplotlib", "matplotlib", None, "figures of steps 02-05, 07, 09; JPG maps of step 00e"),
    ("geopandas", "geopandas", "1.0", "steps 00a-00d, 01, 02, 09, run.py (union_all needs 1.0)"),
    ("shapely", "shapely", "2.0", "steps 00b-00d, 02, 09 (pixel weights use the 2.0 functions)"),
    ("pyproj", "pyproj", None, "map projections (geopandas)"),
    ("pyogrio", "pyogrio", None, "reading and writing GeoJSON, GeoPackage, shapefile (geopandas)"),
    ("rasterio", "rasterio", None, "steps 00c, 00d, 00e (GeoTIFF storage of Sentinel-2, Landsat, weather and elevation)"),
    ("xarray", "xarray", None, "steps 00c, 00d (netCDF of the Climate Data Store, converted to GeoTIFF)"),
    ("netCDF4", "netCDF4", None, "steps 00c, 00d (netCDF reader of xarray)"),
    ("requests", "requests", None, "step 00c (Climate Data Store catalogue)"),
    ("pystac-client", "pystac_client", None, "steps 00c, 00d (STAC search)"),
    ("planetary-computer", "planetary_computer", None, "steps 00c, 00d (signed links of the Planetary Computer)"),
    ("cdsapi", "cdsapi", None, "step 00c (ERA5-Land retrieval)"),
    ("openpyxl", "openpyxl", None, "step 01 (weather workbook, tehsil units), step 09 (results workbook)"),
    ("SALib", "SALib", None, "step 07 (Sobol sensitivity)"),
]


def numbers(version):
    """Leading numeric parts of a version text: '2.1.2' -> (2, 1, 2), '2026.4.0' -> (2026, 4, 0)."""
    out = []
    for part in version.split("."):
        digits = ""
        for c in part:
            if not c.isdigit():
                break
            digits += c
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def state(pip_name, import_name, minimum):
    """('ok' | 'missing' | 'too old' | 'broken', version or error text)."""
    try:
        version = metadata.version(pip_name)
    except metadata.PackageNotFoundError:
        return "missing", ""
    try:
        importlib.import_module(import_name)
    except Exception as e:  # noqa: BLE001 - an installed library that cannot be imported is reported, not hidden
        return "broken", f"{version}: {type(e).__name__}: {str(e)[:120]}"
    if minimum and numbers(version) < numbers(minimum):
        return "too old", version
    return "ok", version


def report():
    rows = [(p, i, m, used, *state(p, i, m)) for p, i, m, used in LIBRARIES]
    width = max(len(r[0]) for r in rows)
    for p, _, m, used, st, detail in rows:
        need = f" (needs {m} or newer)" if m and st == "too old" else ""
        print(f"  {p:<{width}}  {st:<8} {detail}{need}  -  {used}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report; install nothing")
    ap.add_argument("--dry-run", action="store_true", help="show what pip would install; install nothing")
    a = ap.parse_args()
    print(f"Python {sys.version.split()[0]} at {sys.executable}\n")
    rows = report()
    todo = [(p, m) for p, _, m, _, st, _ in rows if st != "ok"]
    if not todo:
        print(f"\nall {len(rows)} libraries are installed")
    elif a.check:
        print(f"\n{len(todo)} of {len(rows)} to install: {', '.join(p for p, _ in todo)}  (python pylib.py installs them)")
        raise SystemExit(1)
    else:
        reqs = [f"{p}>={m}" if m else p for p, m in todo]
        cmd = [sys.executable, "-m", "pip", "install", *(["--dry-run"] if a.dry_run else []), *reqs]
        print("\n>>> " + " ".join(cmd), flush=True)
        code = subprocess.run(cmd).returncode
        if code != 0:
            raise SystemExit(f"\npip ended with exit code {code} - see its message above; nothing is reported as "
                             "installed that pip did not install")
        if a.dry_run:
            print("\ndry run: nothing was installed")
            raise SystemExit(1)
        importlib.invalidate_caches()
        print("\nafter the installation:")
        left = [r[0] for r in report() if r[4] != "ok"]
        if left:
            raise SystemExit(f"\nstill not usable: {', '.join(left)}")
        print(f"\nall {len(rows)} libraries are installed")
    if not (Path.home() / ".cdsapirc").exists():
        print("\nnote: no .cdsapirc file in your user folder - the weather export (e3) needs a Climate Data Store key "
              "(README_run.md, 'Before the first run')")


if __name__ == "__main__":
    main()
