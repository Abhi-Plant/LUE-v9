"""Step 00a - input parameters of a run.

Asks for the area (polygon file) and the other values of a run, checks them, writes the run config
config/aoi/<run>.yaml when it does not exist, and records every input parameter with its source.
  python 00a_parameters.py                       asks everything
  python 00a_parameters.py --run dewas_2025_26   run config exists: nothing is asked, its parameters are checked
A value given on the command line is not asked. A run config that exists is never overwritten.
Writes: config/aoi/<run>.yaml (new runs); outputs/<run>/00a_parameters/parameters.csv, checks.csv, manifest.
Next: step 00b (00b_units.py), or step 02 for a run with a crop map.
"""
import argparse

import pandas as pd

from core import params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    params.add_arguments(ap)
    a = ap.parse_args()
    run, how = params.collect(a)
    pt, ct, out = params.record(run, how)
    pd.set_option("display.width", 250, "display.max_colwidth", 110, "display.max_rows", 200)
    print(f"\ninput parameters of run {run}:")
    print(pt[pt.group != "method and data access (not asked)"].to_string(index=False))
    print(f"\n{int((pt.group == 'method and data access (not asked)').sum())} method and data-access settings of "
          "config/project.yaml are listed in parameters.csv\n")
    print(ct.to_string(index=False))
    print(f"\nparameters, checks and manifest: {out}")
    if (ct.status == "FAIL").any():
        raise SystemExit("step 00a stopped: FAIL checks above")


if __name__ == "__main__":
    main()
