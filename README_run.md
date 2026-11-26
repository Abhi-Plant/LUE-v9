# run.py – the input stage, step by step

`run.py` takes a polygon file of an area (GeoJSON) and runs the input steps of the pipeline in order. Each step
runs only when the step before it succeeded; step 00d checks that every dataset was downloaded completely and
correctly, and step 00e makes a JPG map of every stored GeoTIFF.

| Step | Script | What it does | Main result |
|---|---|---|---|
| 00a | `00a_parameters.py` | asks for the input parameters, checks them, writes the run config | `config/aoi/<run>.yaml`, `outputs/<run>/00a_parameters/parameters.csv` |
| 00b | `00b_units.py` | cuts the area into units, one per 0.1° ERA5-Land cell | `outputs/<run>/02_fields/fields.gpkg` |
| 00c | `00c_download.py` | downloads Sentinel-2 (E1), Landsat 8/9 (E2), ERA5-Land weather and elevation (E3) through STAC APIs, stores every dataset as GeoTIFF, writes the preprocessing results as GeoTIFF and computes the export tables from the stored files | `data/stac/<source>/<run>/*.tif`, `data/field_exports/<run>_rasters/`, `data/field_exports/<run>_E*.csv` |
| 00d | `00d_verify.py` | verifies every download: complete, the stored GeoTIFFs unchanged and equal to the source, the tables equal to a recomputation from the source | `outputs/<run>/00d_verify/summary.csv` |
| 00e | `00e_maps.py` | makes a JPG map of every GeoTIFF of step 00c (pictures for looking at the data; no step reads them) | `data/maps/<run>/<group>/*.jpg`, `outputs/<run>/00e_maps/maps.csv` |

No Google Earth Engine is used.

It does **not** estimate yield. Step 00c makes the Landsat evaporative fraction with SEBAL (main
[README](README.md), "SEBAL evaporative fraction") from the Landsat scenes and the weather export E3, which it
therefore makes first. The yield model (steps 01–11) also needs a crop map of the area (step 02 field sample).
**The units of step 00b are not crop fields: a unit is all land cover of the area inside one weather cell; they serve
the download and its checks only, and step 01 stops on them.**

## Before the first run

1. Python libraries: run `python pylib.py` once. It installs the libraries of the whole pipeline that are missing
   (`python pylib.py --check` only lists them).
2. For the weather (export e3): a Climate Data Store account with its key in the file `.cdsapirc` in your user
   folder (https://cds.climate.copernicus.eu/how-to-api), and the licence of the dataset "ERA5-Land hourly data"
   accepted on the Climate Data Store website. Without the key `run.py` stops before any download and says so; leave
   `e3` out of the exports to run without it.
3. Sentinel-2, Landsat and elevation come from the Planetary Computer and need no account.

## How to run

From the framework folder:

```
python run.py                                   a new area: asks the parameters, then every step
python run.py --run dewas_2025_26               an existing run: continues from the first step not complete
python run.py --run dewas_2025_26 --detach      the same in the background - closing the window does not stop it
python run.py --from 00c --run guna_2025_26     from the download on
python run.py --only-step 00d --run guna_2025_26   one step only (here: verify again)
```

**If the run stops** (window closed, laptop switched off or asleep, network lost): run the same command again.
Everything already downloaded or computed is used again – the stored GeoTIFFs of the granules, Landsat bands and
weather months, the SEBAL result of every scene, elevation, the maps, and every model step that is complete with
unchanged inputs and configuration. With
`--detach` the run continues when the window is closed; follow it with `Get-Content -Wait logs\<run>_<date-time>.log`
in PowerShell. A program cannot run while the laptop is off or asleep: start the command again afterwards.

Every step can also be run on its own, e.g. `python 00d_verify.py --run guna_2025_26`.

Answer the questions. Enter takes the value shown in [brackets]. A wrong answer (file not found, not a date) is
asked again.

**The values in brackets are those of the Dewas 2025-26 potato run. They are not checked for your area or crop –
type your own season dates when they differ.**

### Step 00a – input parameters

| # | Question | Value in brackets | Command-line option |
|---|---|---|---|
| 1 | AOI polygon file – path | none, must be given | `--aoi` |
| 2 | Name of the area | file name | `--name` |
| 3 | Season window start (YYYY-MM-DD) | 2025-10-01 | `--window-start` |
| 4 | Season window end | 2026-03-31 | `--window-end` |
| 5 | Run id | `<name>_<start year>_<end year>`, e.g. `guna_2025_26` | `--run` |
| 6 | Season label | rabi_2025_26 | `--season` |
| 6a | Crop (a file in `config/crops/`) | potato | `--crop` |
| 7 | Peak search start and end | 2025-11-01, 2026-02-28 | `--peak-start`, `--peak-end` |
| 8 | Quality window start and end | 2025-12-01, 2026-01-31 | `--quality-start`, `--quality-end` |
| 9 | Local time minus UTC in hours | 5.5 (India) | `--utc-offset` |
| 10 | Calibration file | none | `--calibration` (`none`) |

- **AOI polygon file** – GeoJSON, GeoPackage or shapefile with one or more polygons and a coordinate reference
  system. The file is read where it is and is not changed. All polygons together are the area.
- **Name** – the first three letters start the unit ids (`Guna` gives `GUN_24.5_77.3`).
- **Season window** – the model days of the season. Data are downloaded from 30 days before the window start
  (`lead_days` in `config/project.yaml`, 30 days; a run config may set its own
  `season_dates.lead_days`) to the window end.
- **Run id** – names the run config `config/aoi/<run>.yaml` and the output folders. Lower-case letters, digits, `_`.
- **Peak search, quality window** – read by the yield model later (step 03), not by the download. They must
  lie inside the season window. Without brackets when your window differs from the Dewas one: then type them.
- **Local time minus UTC** – a weather day is the local calendar day.
- **Calibration file** – benchmark or observed yields, used by step 08 only. Enter for none.
- **Crop** – the crop file `config/crops/<crop>.yaml` with every crop value, its source and status; step 00a lists
  them in `parameters.csv` and stops when a value has no source.

Questions 6–10 (and 6a) are asked only when the run config does not exist yet. With `--run <id>` alone and a run config that
exists, nothing is asked.

Step 00a then checks the parameters and stops the run before any download when one is wrong: season dates in order,
peak search and quality window inside the season window, the download period already over (data of a period that
has not ended cannot be complete), the polygon file readable, the export files named. It writes
`parameters.csv`: every input parameter with its value and where the value came from (typed, value in brackets,
command line, run config), the facts of the polygon file (checksum, area, extent) and the 38 method and data-access
settings of `config/project.yaml` (collections, cloud classes, variables) that are not asked.

### Steps 00c and 00d – options of the download and the verification

| # | Question | Value in brackets | Command-line option |
|---|---|---|---|
| 11 | Exports to make and verify | e1,e2,e3,px | `--only` |
| 12 | Test period start and end | none = the full period | `--test-start` (`none`), `--test-end` |
| 13 | Parallel downloads | 4 | `--workers` |

- **Exports** – `e1` Sentinel-2 NDVI, `e2` Landsat, `e3` weather, `px` pixel stacks of every crop-map pixel (made from
  the stored rasters of `e1` and `e2`, no download; needs a crop map); any combination, separated by commas.
- **Test period** – a short period to try the run first, e.g. 2026-01-03 to 2026-01-10. Its results go to separate
  test folders and never into the export files.
- **Parallel downloads** – more is faster on small areas; large areas need few (see "Time and memory").

`--passes` (default 3, not asked): when a granule, scene or weather request cannot be retrieved, step 00c is started
again up to this many times; everything already retrieved is kept.

### Without questions

Every answer can be given as an option; a value given this way is not asked.

```
python run.py --aoi Guna.geojson --name Guna --window-start 2025-10-01 --window-end 2026-03-31 --run guna_2025_26 ^
              --season rabi_2025_26 --peak-start 2025-11-01 --peak-end 2026-02-28 ^
              --quality-start 2025-12-01 --quality-end 2026-01-31 --utc-offset 5.5 --calibration none ^
              --only e1,e2,e3,px --workers 4 --test-start none
```

(`^` continues a line in the Windows command prompt; in PowerShell use a backtick.) `python run.py --help` lists the
options.

## What step 00d verifies

| Dataset | Completeness | Integrity | Sample recomputed from the source |
|---|---|---|---|
| Sentinel-2 NDVI (E1) | every granule the STAC catalogue lists now for the units and period was downloaded and read; every granule has its three GeoTIFFs and its record; one row per unit and date; every unit has NDVI | every stored band is present, has the sha256 it was written with and opens | 8 granules: stored red, NIR and SCL pixels of one unit against the source, pixel by pixel; 8 unit-dates: NDVI and clear share recomputed from the source (pixel shares from the polygon rasterised on a finer grid, bands read in one window) |
| Landsat scenes and clear shares (E2) | every scene the catalogue lists now was read and stored; every unit is inside a scene | every stored band is present, has the sha256 it was written with and opens | 8 stored bands: a 512 × 512 part of the scene window against the source, pixel by pixel; 8 unit-scenes from QA_PIXEL at the source, same second method |
| Landsat EF and ETa (E2) | every unit-overpass with a clear share has EF within 0–1; NOT PRODUCED when no EF function is configured | – | – |
| ERA5-Land weather (E3) | every hour of the period, every variable, every cell in the hourly GeoTIFFs; one row per cell and day, no missing value | every hourly GeoTIFF has the sha256 of its record; equal value by value to the netCDF file where one is still in the folder; physical ranges | 8 cell-days: radiation and rain from the accumulated values, other variables from the 24 hourly values |
| Elevation (E3) | every cell has an elevation, the same as in the weather export | the elevation GeoTIFF has the sha256 of its record | 8 cells recomputed from the source tiles (not from the stored GeoTIFF) |
| Preprocessing and raster results | – | every GeoTIFF of `<run>_rasters/raster_files.csv` is present, has its sha256 and opens | (E1 is computed from the NDVI and clear rasters, so its sample check covers them) |
| Pixel stacks (PX) | made from the current NDVI rasters, grid, crop map and Landsat scenes; one row per date / overpass, one column per pixel | every file has the sha256 of `record.json` | 8 dates and 8 overpasses at 2000 random pixels read again from the rasters: equal within half a storage step, same missing values |
| Download record | – | every file recorded by step 00c is present with the same sha256 | – |

Accepted differences between an export value and its recomputation (`TOLERANCE` in `00d_verify.py`; settings of the
check, not model values): NDVI 0.01, clear share 0.02, weather 0.001 in the unit of each variable. The two methods
weight the pixels on the edge of a unit differently, so small differences are expected. `--samples` and `--seed`
change the sample.

Result per dataset in `summary.csv`: VERIFIED, NOT PRODUCED, or INCOMPLETE OR WRONG. Nothing is corrected or filled
by step 00d; after INCOMPLETE OR WRONG run step 00c again (it continues where it stopped), then step 00d.

## What you get

| File | Content |
|---|---|
| `config/aoi/<run>.yaml` | the run config |
| `outputs/<run>/00a_parameters/parameters.csv`, `checks.csv` | every input parameter with its source; the parameter checks |
| `outputs/<run>/02_fields/fields.gpkg`, `fields.csv` | the units: `field_id`, `cell`, centre `lon`/`lat`, `area_ha`, `cell_share` (share of the cell inside the area); `checks.csv` |
| `data/field_exports/<run>_E1_s2_perdate.csv` | one row per unit and Sentinel-2 date: `field_id, date, ndvi, valid_fraction, n_scenes` |
| `data/field_exports/<run>_E3_era5land_daily.csv` | one row per cell and day: `cell, date, rs_mj_m2, tmin_c, tmax_c, tdew_c, u10_ms, sp_kpa, elev_m, lat` |
| `data/stac/landsat/<run>/landsat_field_overpass_clear.csv` | one row per unit and Landsat scene: `field_id, date, sensor, valid_fraction, item_id` |
| `data/field_exports/<run>_E2_landsat_perdate.csv` | one row per unit and overpass: `field_id, date, sensor, ef, eta_mm_day, rn_w_m2, g_w_m2, h_w_m2, le_w_m2, valid_fraction, item_id` (EF function configured: SEBAL) |
| `data/stac/landsat/<run>/sebal_anchors.csv` | one row per Landsat scene: hot and cold pixel (temperature, NDVI, 3 × 3 homogeneity), weather of the overpass, elevation range of the weather cells, convergence, clipped pixels, EF of three alternative anchor rules – review it after every E2 run |
| `data/stac/sentinel2/<run>/s2_items.csv`, `data/stac/landsat/<run>/landsat_items.csv` | every granule and scene used |
| `data/stac/sentinel2/<run>/granules/<granule>_B04.tif`, `_B08.tif`, `_SCL.tif`, `<granule>.json`; `s2_files.csv` | the Sentinel-2 bands as GeoTIFF: digital numbers of the source on its grid inside the pixel windows of the units (no data elsewhere); the record (offsets of the product metadata, windows, sha256); table of all files |
| `data/stac/landsat/<run>/scenes/<scene>_<band>.tif`, `<scene>.json`; `landsat_files.csv` | the Landsat bands that are read as GeoTIFF (QA_PIXEL; with SEBAL also blue, green, red, nir08, swir16, swir22, lwir11): digital numbers in the scene window (all units + 20 km); record and result of the scene; table of all files |
| `data/stac/era5land/<run>/raw/`, `raw_precip/` | hourly weather as GeoTIFF, one file per month and variable (band = hour, band description = UTC time), each month with its request (`.request.json`) and record (`.tif.json`) |
| `data/stac/dem/<run>/nasadem_cells.tif`, `cell_elevation.csv`; `nasadem_mosaic.tif` | elevation on the grid of the source tiles and the elevation of every cell computed from it; the mosaic SEBAL uses for the terrain datum |
| `data/field_exports/<run>_rasters/ndvi/` | per Sentinel-2 date on the 10 m grid: `<run>_clear_<date>.tif` (1 clear, 0 not, 255 not observed), `<run>_red_<date>.tif`, `<run>_nir_<date>.tif` (reflectance × 10000), `<run>_ndvi_<date>.tif` |
| `data/field_exports/<run>_rasters/landsat/` | per Landsat scene: `<scene>_clear.tif`, `<scene>_ef.tif`, `<scene>_eta.tif` (SEBAL) |
| `data/field_exports/<run>_rasters/weather/` | daily weather on the 0.1° cells, one GeoTIFF per variable (band = day): the E3 columns, `tmean_c`, `eto_mm`; `<run>_elev_m.tif` |
| `data/field_exports/<run>_rasters/raster_files.csv` | every raster result with bytes and sha256 |
| `data/field_exports/<run>_pixels/` | export px: `ndvi.npy` (Sentinel-2 dates × pixels), `landsat_ef.npy`, `landsat_eta.npy`, `landsat_clear.npy` (Landsat overpasses × pixels; the values of the 10 m pixel square, area-weighted as the field export E2, v0.9.0), int16 = value / scale (`config/project.yaml` section `pixels`), −32768 = no value; `pixels.npz` (parcel, row, column and centre of every pixel), `parcels.csv`, `dates.csv`, `overpasses.csv`, `record.json` (what the stacks were made from, sha256 of every file) |
| `data/maps/<run>/<group>/*.jpg`, `outputs/<run>/00e_maps/maps.csv` | a JPG map of every GeoTIFF above; the table says what each map shows, its value range and how much it was reduced |
| `outputs/<run>/00c_download/qc_checks.csv`, `manifest.json` | the checks of the download and the list of files with checksums |
| `outputs/<run>/00d_verify/summary.csv`, `verification.csv`, `sample_*.csv` | the verdict per dataset, every check, the recomputed samples |

`ndvi` is the mean over the clear pixels of the unit (scene classification vegetation or bare soil);
`valid_fraction` is the share of the unit that was clear. How each value is computed: main README, "How each export
is computed".

A test period writes to `data/_test/<run>_TEST_<dates>/`, `outputs/<run>/00c_download_TEST_<dates>/` and
`outputs/<run>/00d_verify_TEST_<dates>/` instead.

## Things to know

- **Exit codes.** `run.py` ends with 0 when every dataset is verified, 2 when the downloads are verified but the
  Landsat EF and ETa were not produced (expected while no EF function is configured), 1 when a step failed. The last
  lines list each step with its result.
- **An interrupted run continues.** Run `python run.py --from 00c --run <id>`: Sentinel-2 granules and weather
  months already downloaded are kept.
- **A run config is never overwritten.** For the same polygon file and season window the existing config is used as
  it is. For another file or window step 00a stops and asks for another run id. To change other values of an
  existing run, edit `config/aoi/<run>.yaml`.
- **Units that exist and are unchanged are left as they are.** When the polygon file changed, step 00b stops and
  asks to delete `data/stac/sentinel2/<run>/granules/` first (the stored granules hold the pixel windows of the old
  units).
- **Not GeoTIFF, and why.** The export tables E1–E3 (one row per unit and date) and all model outputs are tables
  (CSV); the units are polygons (GeoPackage); the records are JSON. The Climate Data Store delivers netCDF or GRIB
  only: the netCDF is converted to GeoTIFF at once, compared value by value and removed.
- **One job per run id.** Two jobs for the same run id at the same time write the same files.
- **Runs with a field sample** (Dewas, made by step 02 from a crop map): `python run.py --from 00c --run
  dewas_2025_26`. Step 00b is skipped; its field sample is never overwritten.
- **An area that spans two UTM zones** stops with a message: Sentinel-2 granules on two pixel grids are not
  supported.

## Time and memory

Figures measured on this machine, 2026-11-20:

- Weather: one request per calendar month; each took between 8 and 17 minutes in the Climate Data Store queue
  (two requests at a time).
- Guna district, 638,459 ha (88 units): 415 Sentinel-2 granules and 95 Landsat scenes for the season. With 4
  parallel downloads 395 granules were read in 165 minutes; the granule cache is 20 GB. Every granule in work holds
  the pixels of all its units in memory: with 8 parallel downloads the job used about 3.2 GB, which left too little
  free memory on this 16 GB machine, so it was run with 4.
- Dewas field sample (345 fields, most smaller than 1 ha): 365 granules in 56 minutes with 24 parallel downloads;
  cache 20 MB.

A district as units is a run of several hours; try a test period first.

## Messages

| Message | What to do |
|---|---|
| `file not found` | check the path; a path with spaces can be typed with or without quotes |
| `has no coordinate reference system` | save the polygon file with its CRS (for GeoJSON: longitude/latitude, EPSG:4326) |
| `needs polygons` | the file holds points or lines; give the area as polygons |
| `run config … exists for this area and window: used as it is` | the pasted polygon has the same contents as the run config's boundary file (any folder): the run continues with that run config |
| `run config … exists for another polygon … runs as <run id>_2` | a different polygon or season window under an existing run id: that run config and its outputs are kept, this area continues under the next free run id |
| `step 00a stopped: FAIL checks above` | a parameter is wrong; the check table above the message says which |
| `E3 needs a Climate Data Store account` | set up `.cdsapirc`, or leave `e3` out |
| `no answer for '…' - give it on the command line` | the script was started without a keyboard; pass the named option |
| `no … items for the fields from … to …` | no scene in the period (often a too-short test period): choose a longer one |
| `step 00c stopped, download not complete` | a granule, scene or weather request failed after 4 tries; `run.py` starts the step again by itself (`--passes`), afterwards run `--from 00c` again |
| `step 00c incomplete: E2 not written` | `stac.landsat.ef_function` is null in `config/project.yaml` (set it to `core.ef_sebal:compute_ef`) |
| `step 00d: FAIL` | a dataset is incomplete or wrong; `summary.csv` names the check – run step 00c, then 00d again |
