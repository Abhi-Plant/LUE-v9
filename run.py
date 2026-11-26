"""Run the whole pipeline of a run step by step, continuing where it stopped.

  STEP 00a  input parameters   asks for the area, crop and season (new run) or checks an existing run config
  STEP 00b  units              area polygon without crop map: the parts of the area inside each ERA5-Land cell
  STEP 02   field sample       crop map: stratified sample of crop parcels per ERA5-Land cell (once per season)
  STEP 00c  data download      E1 Sentinel-2, E3 ERA5-Land weather and elevation, E2 Landsat 8/9 with SEBAL EF,
                               PX pixel stacks of every crop-map pixel (from the stored rasters)
  STEP 00d  verification       every dataset complete, intact and equal to a recomputation from the source
  STEP 00e  maps               a JPG map of every GeoTIFF of step 00c (pictures only, no step reads them)
  STEP 01-07 field sample      inputs, season, drivers, LUE, ensemble, sensitivity of the sampled fields (Sobol,
                               variance shares, design-based district estimate, checks)
  STEP 05b  pixel yield model  the model of steps 03-05 for every crop-map pixel, aggregated pixel -> parcel ->
                               cell, tehsil, district
  STEP 08-10                   calibration, final products (pixel-based), validation
  STEP 11   pixel map          pixel yield GeoTIFF, parcel GeoPackage, tehsil and district totals (calibrated)
  STEP 12   parcel-mean map    the model on each parcel's mean inputs (as the field sample), next to the pixel-based
                               parcel yield of step 11: the product to compare with plot / parcel yields (v0.9.0)

Continuing after an interruption (window closed, laptop switched off or asleep, network lost): run the same command
again. Nothing already done is done again:
  - downloads: every Sentinel-2 granule, Landsat band and scene result (SEBAL) and ERA5-Land month is stored as
    GeoTIFF, kept as soon as it is complete and used again (step 00c); elevation is kept for the same cells;
  - model steps: a step is skipped when its manifest exists (written as its last file), the configuration is the same
    and its input files are unchanged (sha256); else it runs again, and so do the steps after it;
  - every file is written to a temporary name and renamed when complete, so no half-written file is ever used.
A program cannot run while the laptop is off or asleep; the run continues when it is started again.

  python run.py                                    new run: asks the parameters, then every step
  python run.py --run dewas_2025_26                existing run: continues from the first step not complete
  python run.py --run dewas_2025_26 --detach       the same in the background: closing the window does not stop it;
                                                   output in logs/<run>_<date-time>.log
  python run.py --run X --from 03 --to 05          steps 03 to 05 only (complete ones are still skipped)
  python run.py --run X --rerun                    run every step again
Only steps 01-12 need a crop map (step 02 field sample) and the Landsat EF export of step 00c. The units of step 00b
are NOT crop fields: a run made from them stops after step 00e.
"""
import argparse
import atexit
import datetime
import json
import os
import subprocess
import sys

import pandas as pd

from core import config, params

INPUT = ["00a", "00b", "02", "00c", "00d", "00e"]
MODEL = ["01", "03", "04", "05", "05b", "06", "07", "08", "09", "10", "11", "12"]
STEPS = INPUT + MODEL
TITLE = {"00a": "input parameters", "00b": "units", "02": "field sample", "00c": "data download",
         "00d": "verification of the downloads", "00e": "maps of the GeoTIFFs (JPG)", "01": "inputs and QC",
         "03": "season dates", "04": "drivers",
         "05": "LUE biomass and yield (sample)", "05b": "pixel yield model", "06": "ensemble", "07": "sensitivity",
         "08": "calibration", "09": "final products", "10": "validation checks", "11": "pixel yield map",
         "12": "parcel-mean yield map"}
SCRIPT = {"00b": "00b_units.py", "02": "02_fields.py", "00c": "00c_download.py", "00d": "00d_verify.py",
          "00e": "00e_maps.py",
          "01": "01_inputs.py", "03": "03_season.py", "04": "04_drivers.py", "05": "05_lue.py",
          "05b": "05b_pixel_model.py", "06": "06_ensemble.py", "07": "07_sensitivity.py", "08": "08_calibrate.py",
          "09": "09_aggregate.py", "10": "10_validate.py", "11": "11_pixel_map.py",
          "12": "12_parcel_map.py"}
FOLDER = {"01": "01_inputs", "03": "03_season", "04": "04_drivers", "05": "05_lue", "05b": "05b_pixel_model",
          "06": "06_ensemble", "07": "07_sensitivity", "09": "09_aggregate", "10": "10_validate",
          "11": "11_pixel_map", "12": "12_parcel_map"}                                              # 08 keeps its own complete check


def banner(step, note=""):
    print(f"\n{'=' * 100}\nSTEP {step}  {TITLE[step]}{note}\n{'=' * 100}", flush=True)


def detach(a, argv):
    """Start this command again as a separate background process (not tied to the window) and return."""
    if a.run is None:
        raise SystemExit("--detach needs --run <run id> of an existing run config (a background run cannot ask)")
    child = [x for x in argv if x != "--detach"]
    defaults = {"--only": "e1,e2,e3,px", "--test-start": "none", "--workers": "4"}
    for k, v in defaults.items():                      # a background run cannot answer questions
        if k not in child:
            child += [k, v]
    logs = config.ROOT / "logs"
    logs.mkdir(exist_ok=True)
    log = logs / f"{a.run}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    kw = {"stdin": subprocess.DEVNULL, "stdout": open(log, "w", encoding="utf-8"), "stderr": subprocess.STDOUT,
          "cwd": config.ROOT, "env": {**os.environ, "PYTHONUNBUFFERED": "1"}}
    if os.name == "nt":
        kw["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                               | subprocess.CREATE_NO_WINDOW)
    else:
        kw["start_new_session"] = True
    p = subprocess.Popen([sys.executable, str(config.ROOT / "run.py"), *child], **kw)
    print(f"run {a.run} started in the background (process {p.pid}); closing this window does not stop it.\n"
          f"output: {log}\nfollow it:  Get-Content -Wait \"{log}\"   (PowerShell)\n"
          f"after a stop (laptop off): run the same command again - it continues where it stopped.")


def take_lock(run):
    """One run of a run id at a time: outputs/<run>/.run.lock holds the process id while the run works."""
    lock = config.ROOT / "outputs" / run / ".run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    running = None
    if lock.exists():
        try:
            import psutil
            pid = int(lock.read_text().split()[0])
            if pid != os.getpid() and psutil.pid_exists(pid) and "run.py" in " ".join(psutil.Process(pid).cmdline()):
                running = pid
        except Exception:                               # no psutil, unreadable lock, process gone: not running
            running = None
    if running:
        raise SystemExit(f"run {run} is already running (process {running}); wait for it or stop that process")
    lock.write_text(f"{os.getpid()} {datetime.datetime.now().isoformat(timespec='seconds')}")
    atexit.register(lambda: lock.unlink(missing_ok=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    params.add_arguments(ap)
    ap.add_argument("--only", help="step 00c/00d: exports to make and verify: e1,e2,e3,px")
    ap.add_argument("--workers", help="step 00c: parallel downloads")
    ap.add_argument("--test-start", help="steps 00c/00d: test period start YYYY-MM-DD ('none' = full period); a test "
                                         "period stops the run after step 00d")
    ap.add_argument("--test-end", help="steps 00c/00d: test period end YYYY-MM-DD")
    ap.add_argument("--passes", type=int, default=3, help="step 00c: how often the download is started again when "
                    "a granule, scene or weather request could not be retrieved (default 3)")
    ap.add_argument("--from", dest="first", choices=STEPS, default="00a", help="first step (default 00a)")
    ap.add_argument("--to", dest="last", choices=STEPS, default="12", help="last step (default 12)")
    ap.add_argument("--only-step", choices=STEPS, help="run this step only")
    ap.add_argument("--rerun", action="store_true", help="run every step again, also the complete ones")
    ap.add_argument("--detach", action="store_true", help="run in the background, not tied to this window")
    a = ap.parse_args()
    if a.detach:
        return detach(a, sys.argv[1:])
    todo = [a.only_step] if a.only_step else STEPS[STEPS.index(a.first):STEPS.index(a.last) + 1]
    result = {}

    # ---- step 00a: input parameters (always: also when the run starts later, its parameters are checked again)
    banner("00a")
    if "00a" not in todo and a.run is None:
        raise SystemExit(f"--run is needed to start at step {todo[0]}")
    run, how = params.collect(a)
    take_lock(run)
    pt, ct, out = params.record(run, how)
    pd.set_option("display.width", 250, "display.max_colwidth", 110)
    print(ct.to_string(index=False))
    print(f"\nrun {run}: parameters, checks and manifest in {out}")
    if (ct.status == "FAIL").any():
        raise SystemExit("\nstep 00a stopped: FAIL checks above - nothing was downloaded")
    result["00a"] = "done"
    cfg = config.load(run)
    inp = cfg["aoi"].get("inputs") or {}
    has_boundary, has_crop_map = "boundary_file" in inp, bool(inp.get("crop_mask"))

    # ---- options of the download and of its verification
    period = []
    if {"00c", "00d"} & set(todo):
        only, _ = params.ask("Exports to make (e1 Sentinel-2 NDVI, e2 Landsat, e3 weather, px pixel stacks)",
                             "--only", a.only, params.exports, "e1,e2,e3,px")
        t0 = None if a.test_start == "none" else params.ask(
            "Test period start YYYY-MM-DD (Enter = the full period of the season)", "--test-start", a.test_start,
            params.a_date, optional=True)[0]
        t1 = params.ask("Test period end YYYY-MM-DD", "--test-end", a.test_end, params.a_date)[0] if t0 else None
        if t0 and t1 < t0:
            raise SystemExit(f"test period: {t1} is before {t0}")
        period = ["--start", t0, "--end", t1] if t0 else []
    if "00c" in todo:
        workers, _ = params.ask("Parallel downloads (fewer need less memory)", "--workers", a.workers, params.a_count, 4)
        key = pt[(pt.parameter == "climate_data_store_key")].value.iloc[0]
        if "e3" in only and key != "found":
            raise SystemExit("\nE3 needs a Climate Data Store account: no ~/.cdsapirc file and no CDSAPI_KEY found. "
                             "Set it up (https://cds.climate.copernicus.eu/how-to-api) or leave e3 out of the exports.")

    def step(s, *args):
        cmd = [sys.executable, SCRIPT[s], "--run", run, *args]
        print(">>> python " + " ".join(cmd[1:]), flush=True)
        return subprocess.run(cmd, cwd=config.ROOT).returncode

    def earlier(s):
        """Status of a step this command did not run: from its manifest (written as a step's last file)."""
        if s == "00b" and not has_boundary:
            return "not needed (the units are the step-02 field sample)"
        out = config.ROOT / "outputs" / run
        gpkg = out / "02_fields" / "fields.gpkg"
        if s == "02" and gpkg.exists():                  # a sample is kept even without a manifest (earlier versions)
            t = datetime.datetime.fromtimestamp(gpkg.stat().st_mtime).isoformat(timespec="seconds")
            return f"kept - field sample exists (fields.gpkg, file date {t})"
        if s in FOLDER:                                 # model steps: same test as the skip rule
            ok, why = config.step_complete(cfg, FOLDER[s])
            return f"complete - earlier run ({why.split('(')[-1].rstrip(')')})" if ok else f"not run ({why})"
        name = {"00b": "00b_units", "02": "02_fields", "00c": "00c_download", "00d": "00d_verify",
                "00e": "00e_maps"}.get(s)
        if s == "08":                                    # one folder per calibration id: the latest one
            try:
                cid = json.load(open(out / "08_calibrate" / "latest.json", encoding="utf-8"))["calibration_id"]
                name = f"08_calibrate/{cid}"
            except (OSError, ValueError, KeyError):
                name = None
        p = out / name / "manifest.json" if name else None
        if p is None or not p.exists():
            return "not run"
        try:
            return f"complete - earlier run ({json.load(open(p, encoding='utf-8')).get('run_at')})"
        except (OSError, ValueError):
            return "not run (manifest unreadable)"

    def finish(code):
        print(f"\n{'=' * 100}\nRUN {run} - steps (this command: done / skipped; others from their manifest)")
        for s in STEPS:
            print(f"  STEP {s:<4} {TITLE[s]:<30} {result.get(s) or earlier(s)}")
        raise SystemExit(code)

    # ---- units: step 00b (area polygon) or step 02 (crop map, once per season)
    if "00b" in todo and has_boundary:
        banner("00b")
        if step("00b") != 0:
            result["00b"] = "FAILED"
            finish(1)
        result["00b"] = "done"
    if "02" in todo and not has_boundary:
        p_fields = config.ROOT / "outputs" / run / "02_fields" / "fields.gpkg"
        if p_fields.exists() and not a.rerun:
            result["02"] = "kept (field sample exists - the sample of a season is drawn once)"
        elif not has_crop_map:
            result["02"] = "FAILED: no field sample and no crop map in the run config"
            finish(1)
        else:
            banner("02")
            if step("02") != 0:
                result["02"] = "FAILED"
                finish(1)
            result["02"] = "done"

    # ---- step 00c: data download (resumes; started again when a download failed)
    if "00c" in todo:
        banner("00c", "  (what is already downloaded is used again)")
        for attempt in range(1, a.passes + 1):
            code = step("00c", "--only", only, "--workers", str(workers), *period)
            if code != 3:
                break
            print(f"\ndownload not complete after pass {attempt} of {a.passes}"
                  + (" - starting again, what is retrieved is kept\n" if attempt < a.passes else ""), flush=True)
        result["00c"] = {0: "done", 2: "done - E2 EF and ETa not written (no EF function)",
                         3: f"FAILED: download not complete after {a.passes} passes - run again"}.get(
            code, f"FAILED (exit code {code})")
        if code not in (0, 2):
            finish(1)

    # ---- step 00d: verification
    if "00d" in todo:
        banner("00d")
        code = step("00d", "--only", only, *period)
        result["00d"] = {0: "done - every dataset complete and verified",
                         2: "done - downloads verified; Landsat EF and ETa not produced",
                         1: "FAILED: a dataset is incomplete or wrong"}.get(code, f"FAILED (exit code {code})")
        if code != 0:
            finish(code)

    # ---- step 00e: JPG maps of the GeoTIFFs (maps that exist are kept)
    if "00e" in todo:
        banner("00e")
        code = step("00e", *period)
        result["00e"] = "done" if code == 0 else f"FAILED (exit code {code}): a map could not be made"
        if code != 0:
            finish(1)

    # ---- yield model: steps 01-12, a complete step with unchanged inputs and configuration is skipped
    model = [s for s in MODEL if s in todo]
    if model and (period or has_boundary):
        for s in model:
            result[s] = "not run: " + ("test period" if period else "units of step 00b are not crop fields")
        finish(0)
    redo = a.rerun
    for s in model:
        if not redo and s in FOLDER:
            ok, why = config.step_complete(cfg, FOLDER[s])
            if ok:
                result[s] = f"skipped - {why}"
                print(f"STEP {s}  {TITLE[s]}: skipped - {why}", flush=True)
                continue
            print(f"STEP {s}: {why}", flush=True)
        redo = True                                     # every step after a step that ran runs again
        banner(s)
        code = step(s)
        result[s] = "done" if code == 0 else f"FAILED (exit code {code})"
        if code != 0:
            finish(1)
    finish(0)


if __name__ == "__main__":
    main()
