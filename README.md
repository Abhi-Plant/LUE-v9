# LUE crop yield framework – v0.9.0

Light-use-efficiency (Monteith) yield model driven by Sentinel-2, Landsat 8/9 and ERA5-Land, for one district, one
crop and one season per run:

    dry yield = HI × Σ_days [ εmax · Ts · Ws · fAPAR · f · Rs ]   (emergence → season end);   fresh yield = dry yield ÷ DM

The method follows the method document (`references/LUE_Potato_Yield_Methodology_method_document.pdf`) and the
SEBAL-based crop yield model of Bastiaanssen & Ali (2003). Every equation and coefficient is listed below with its
source (equation, page) and status. Data access is through STAC APIs (no Google Earth Engine).
Data chain: satellite data stored as GeoTIFF, weather from netCDF to GeoTIFF → preprocessing rasters (clear mask,
NDVI, EF, ETa, daily weather) → pixel stacks of every crop-map pixel (export PX) → yield model per 10 m pixel
(step 05b) → aggregation pixel → parcel → cell, tehsil, district → calibration and final products (steps 08, 09) →
spatial yield output (pixel yield GeoTIFF, parcel GeoPackage, tehsil and district totals, step 11) → next to it the
parcel-mean yield map (step 12: the model on each parcel's mean inputs, the product to compare with plot or parcel
yields). The field sample (steps 01–07) gives the Sobol and variance analysis, the design-based district estimate with
its sampling error and the checks. See "Storage: GeoTIFF", "Pixel level (steps 05b and 11)" and "Parcel-mean yield
map (step 12)".

**New in v0.9.0** (`framework.changes_0_9_0` in `config/project.yaml`; no equation or parameter value changed):
1. Landsat values of a 10 m pixel are those of the pixel **square**, computed exactly as the field export E2 computes
   them for a field: area-weighted over the clear 30 m Landsat pixels the square overlaps (one, two or four), with
   the clear share of the square; an overpass with a clear share below `ef.min_valid_fraction` is left out, as for
   field units. v0.8.0 took the value of the single 30 m pixel holding the pixel centre. Pixel stacks of v0.8.0 are
   refused: run `python 00c_download.py --run <run> --only px` (no download, made from the stored rasters).
2. New step 12 (`12_parcel_map.py`): the parcel-mean yield map, every parcel modelled as one unit on its mean inputs,
   next to the pixel-based parcel yield of step 11 and the difference.
How the steps, modules, equations and configuration connect: [ARCHITECTURE.md](ARCHITECTURE.md).

## Configuration: method, crop, run

| File | Holds | Same for every … |
|---|---|---|
| `config/project.yaml` | method settings: PAR fraction f, season detection, NDVI cleaning, curve fitting, water scalar, SEBAL, ensemble, Sobol, calibration, acceptance targets, data access (`stac`) | crop and run |
| `config/crops/<crop>.yaml` | crop values: εmax, HI, DM (ensemble values and ranges), Topt (used only with `temperature_scalar.topt: crop_values`), crop coefficient, NDVI amplitude of a crop cycle, heat-day flag, yield basis, expected calibration input; crop-map class name, harvested part, name of the season end | run of that crop |
| `config/aoi/<run>.yaml` | one district × crop × season: `crop`, season dates, input files (read in place), field sample, benchmark, registered sha256 and provenance of every input | – |

Every value of a crop file carries a `source` (citation with page, table or equation) and a `status`:

| Status | Meaning |
|---|---|
| `verified` | the value was seen on the printed page of the source |
| `verified_range` | the range is verified; the central value inside it is a framework choice |
| `adapted` | a published value used outside the purpose it was published for (stated in the source text) |
| `not_verified` | the cited source is not available; the value is not checked |
| `setting` | framework choice, no literature value |

`core/config.py` stops a run when a crop value has no source or no valid status. Step 00a lists every crop value with
its source and status in `outputs/<run>/00a_parameters/parameters.csv`.

**Another crop:** copy `config/crops/potato.yaml` to `config/crops/<crop>.yaml`, replace every value and its source
(εmax, HI and DM need crop-specific literature; Kc and crop height: FAO-56 Table 12; HIo: AquaCrop Annex I),
and set `crop: <crop>` in the run config. The equations, including the water scalar, are the same for every crop.

## Pipeline

Run from this folder; every step takes `--run <run id>` and writes `outputs/<run>/<step>/` with a `manifest.json`
(inputs and outputs with sha256, package versions, run time, peak memory).

| Order | Step | Script | What it does | Main outputs |
|---|---|---|---|---|
| 1 | 00a | `00a_parameters.py` | asks / checks the run parameters (area, crop, season dates, weather day), writes the run config, records every parameter and crop value with its source | `config/aoi/<run>.yaml`; `parameters.csv`, `checks.csv` |
| 2 | 02 | `02_fields.py` | stratified simple random sample of crop-map parcels per 0.1° ERA5-Land cell; inclusion probability n_h/N_h | `fields.csv`, `fields.gpkg`, `cell_strata.csv`, map |
| (2) | 00b | `00b_units.py` | area polygon without crop map: the parts of the area inside each ERA5-Land cell, all land cover – for the download and its checks only, not for yield (no crop area, no design weight) | `02_fields/fields.gpkg` |
| 3 | 00c | `00c_download.py` | E1 Sentinel-2, E3 ERA5-Land + elevation, then E2 Landsat 8/9 with SEBAL EF (E3 first: SEBAL reads its weather), then PX: the pixel stacks of every crop-map pixel from the stored rasters. Every dataset is stored as GeoTIFF and the tables are computed from the stored files | `data/stac/<source>/<run>/*.tif`; `data/field_exports/<run>_rasters/`; `data/field_exports/<run>_E1/E2/E3*.csv`; `data/field_exports/<run>_pixels/`; `data/stac/landsat/<run>/sebal_anchors.csv` |
| 4 | 00d | `00d_verify.py` | completeness against the STAC catalogue, integrity of the stored GeoTIFFs (sha256, sample of pixels against the source), sample values recomputed from the source by a second method; pixel stacks against the rasters | `summary.csv`, `verification.csv`, `sample_*.csv` |
| 5 | 00e | `00e_maps.py` | a JPG map of every GeoTIFF of step 00c (pictures only; no step reads them) | `data/maps/<run>/<group>/*.jpg`; `maps.csv` |
| 6 | 01 | `01_inputs.py` | field sample: input tables, provenance (sha256), data QC, energy-balance check of E2 | `units.csv`, `obs_ndvi.csv`, `obs_thermal.csv`, `weather_daily.csv`, `provenance.csv`, `qc_checks.csv` |
| 7 | 03 | `03_season.py` | field sample: season dates from the NDVI series (emergence, peak, season end) with censoring flags | `season.csv`, `season_grid.csv`, `season_quality.csv` |
| 8 | 04 | `04_drivers.py` | field sample: daily drivers: Rs, Tmean, fAPAR, EF (Ws), Kc, ETc; growth-curve hold-out | `drivers_daily.csv` |
| 9 | 05 | `05_lue.py` | field sample: biomass and yield of every ensemble member and the central configuration | `members.csv`, `central_unit.csv` |
| 10 | 05b | `05b_pixel_model.py` | the model of steps 03–05 for every crop-map pixel (vectorised; checked against the field-unit code on a sample of pixels), then pixel → parcel → cell, tehsil, district, members first | `parcel_values.csv`, `members_<level>.csv`, `ensemble_<level>.csv`, `pixel_s.npy`, `unit_check.csv`, `checks.csv` |
| 11 | 06 | `06_ensemble.py` | field sample: ensemble LUE-ENS at field, cell, tehsil, district level; variance shares | `ensemble_<level>.csv`, `variance_shares.csv` |
| 12 | 07 | `07_sensitivity.py` | field sample: Sobol indices, one-at-a-time, reported alternatives, Landsat 8 / 9 choice | `sobol_indices.csv`, `alternatives.csv` |
| 13 | 08 | `08_calibrate.py` | one calibration factor (ratio of means, member-wise) of the pixel-based district or tehsil values against benchmark or observed yields | `calibration_factor.csv`, `member_factors.csv`, calibrated yields |
| 14 | 09 | `09_aggregate.py` | tehsil, district and cell products from the pixel yield model; next to them the stratified district estimate of the field sample with its sampling error; maps; workbook | `tehsil_final.csv`, `district_final.csv`, `09_LUE_results.xlsx` |
| 15 | 10 | `10_validate.py` | consistency, reproducibility and plausibility checks (incl. the pixel yield model); validation statistics when observed yields exist | `validation_checks.csv`, `benchmarks.csv` |
| 16 | 11 | `11_pixel_map.py` | calibrated yield of every crop-map pixel and parcel: yield GeoTIFF on the 10 m grid, parcel GeoPackage, tehsil and district totals; checks against step 09 and the sample estimate | `yield_<run>.tif`, `parcel_yield.gpkg`, `tehsil_pixel_map.csv`, `district_pixel_map.csv`, `checks.csv` |
| 17 | 12 | `12_parcel_map.py` | parcel-mean yield map (v0.9.0): the field-unit model of steps 01–05 for every parcel on its area-weighted mean inputs (read from the stored rasters), calibrated with the factors of step 08; next to every parcel the pixel-based yield of step 11 and the difference; checks against the exports, step 05, step 09 and the sample estimate | `parcel_yield.gpkg`, `parcel_yield.csv`, `parcel_mean_yield_<run>.tif`, `tehsil_parcel_map.csv`, `district_parcel_map.csv`, `checks.csv` |

```
python pylib.py                                         # once: libraries (or pip install -r requirements.txt)
python run.py --run dewas_2025_26                       # whole pipeline; continues where it stopped
python run.py --run dewas_2025_26 --detach              # the same in the background (closing the window does not stop it)
python 00a_parameters.py --run dewas_2025_26            # or step by step: check and record the run parameters
python 00c_download.py --run dewas_2025_26 --only e3    # weather first (SEBAL needs it)
python 00c_download.py --run dewas_2025_26 --only e2    # Landsat with SEBAL EF
python 00c_download.py --run dewas_2025_26 --only px    # pixel stacks from the stored rasters (no download)
python 00c_download.py --run dewas_2025_26 --start 2026-01-01 --end 2026-01-07   # test week -> data/_test/
python 00d_verify.py --run dewas_2025_26
python 00e_maps.py --run dewas_2025_26                  # JPG maps of the GeoTIFFs (--only rasters,dem for a part)
python 01_inputs.py --run dewas_2025_26                 # ... then 03, 04, 05, 05b, 06 to 10 in order
python 05b_pixel_model.py --run dewas_2025_26           # yield model of every crop-map pixel
python 11_pixel_map.py --run dewas_2025_26              # pixel yield GeoTIFF and parcel yields (after step 08)
python 12_parcel_map.py --run dewas_2025_26             # parcel-mean yield map next to step 11 (after step 11)
python -m pytest -q tests                               # unit and synthetic tests
```

**Continuing after an interruption** (window closed, laptop off or asleep, network lost): run the same command again.
Every Sentinel-2 granule, Landsat band and scene result (SEBAL result and anchor row,
`data/stac/landsat/<run>/scenes/`), ERA5-Land month and the cell elevations are kept as soon as they are complete and
used again; a Landsat scene is computed again only when its fingerprint (EF function code, SEBAL and mask settings,
fields, daily weather) changed - from its stored bands, or with `stac.landsat.keep_source_bands: false` (the
default) after downloading them again.
Every file is written under a temporary name and renamed when complete. A step's `manifest.json` is removed when the
step starts and written as its last file; `run.py` skips a model step whose manifest exists with the same
configuration fingerprint (project, run and crop file) and unchanged input files, and runs every later step again
after one that ran. `--detach` writes the output to `logs/<run>_<date-time>.log`; `outputs/<run>/.run.lock` prevents
two runs of the same run id at the same time. A program cannot run while the laptop is off or asleep.

Questions, options and messages of the input stage: [README_run.md](README_run.md).

## Data in this repository and rerunning a run

Data folders are produced by the pipeline and are not versioned, except what is needed to rerun and check the runs
of `config/aoi/` (the `!` entries of `.gitignore`):

| Path | Content | Used by |
|---|---|---|
| `data/dewas_2025_26/` | potato crop map (GeoPackage), LGD sub-district boundaries of Dewas (extract of the Madhya Pradesh file: the 10 polygons of DISTRICT = DEWAS, unchanged), SoilGrids 2.0 tehsil workbook, AquaCrop tehsil yields (comparison only), tehsil NDVI and weather tables (tehsil units only) | steps 01, 02, 00c, 03, 09, 10 |
| `data/dewas_onion_2025_26/` | onion crop map (GeoPackage, prepared copy of the onion classification; provenance in the run config). Boundaries and soil are read from `data/dewas_2025_26/` | steps 01, 02, 00c, 09 |
| `data/calibration/` | benchmark yields (versioned) | step 08 |
| `outputs/dewas_2025_26/02_fields/`, `outputs/dewas_onion_2025_26/02_fields/` | the season's field sample of each run (drawn once, kept; the manifest records how it was drawn) | 00c, 01-12 |
| `outputs/dewas_2025_26/`, `outputs/dewas_onion_2025_26/` | every step output of the potato and onion runs (00a-12: tables, figures, workbook, maps, manifests) | review |
| `data/field_exports/dewas_2025_26_E1/E2/E3*.csv`, `data/field_exports/dewas_onion_2025_26_E1/E2/E3*.csv` | step 00c exports: Sentinel-2 NDVI, Landsat SEBAL EF and ETa, ERA5-Land weather | steps 01-10 |
| `data/stac/landsat/dewas_2025_26/sebal_anchors.csv`, `data/stac/landsat/dewas_onion_2025_26/sebal_anchors.csv` | per Landsat scene: anchors, hot-pixel water balance, convergence | review of E2 |
| `data/dewas_potato_2025_26/` | area polygon of the area run (step 00b units) | 00b, 00c |

Every input has its sha256 registered in the run config (`input_sha256`); step 01 compares them. Rerun the yield
model from the stored exports (about 10 minutes, no account needed):

```
pip install -r requirements.txt
python -m pytest -q tests
python run.py --run dewas_2025_26 --from 01          # potato
python run.py --run dewas_onion_2025_26 --from 01    # onion
```

`--from 01` leaves the exports and the field sample as they are. Steps 05b and 11 read the pixel stacks of step 00c, step 12 its rasters
(export px), made from its rasters, which are not versioned (several GB): they need step 00c first. Downloading again (`--from 00c`, about 2-3 hours)
needs a Climate Data Store account (`~/.cdsapirc`, README_run.md) and internet access to the Microsoft Planetary
Computer STAC API (no account); the Sentinel-2 granules, Landsat bands and ERA5-Land months are then stored as
GeoTIFF under `data/stac/`.

## Data (step 00c)

| Export | Source (STAC) | Variables | Processing |
|---|---|---|---|
| E1 | Sentinel-2 L2A, Planetary Computer `sentinel-2-l2a` | B04, B08 (10 m), SCL (20 m → 10 m nearest) | reflectance = (DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE from each product's metadata (ESA L2A definition, baseline ≥ 04.00); NDVI on pixels with SCL 4 (vegetation) or 5 (not vegetated); area-weighted field mean; clear share |
| E2 | Landsat 8/9 Collection 2 Level-2, Tier 1, `landsat-c2-l2` | SR_B2–B7 (scale 0.0000275, offset −0.2), ST_B10 (scale 0.00341802, offset 149.0 K), QA_PIXEL (30 m) | clear = QA_PIXEL bits 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow not set (USGS LSDS-1619 v6.0 Tables 6-1, 6-2); SEBAL per pixel in the scene window (fields + 20 km); area-weighted field means of EF, ETa, Rn, G, H, LE |
| PX | the rasters of E1 and E2 (no download) | NDVI per date; EF, ETa per overpass | every 10 m pixel of the NDVI grid with its centre inside a crop-map parcel (a parcel without one: the pixel of its representative point); NDVI of the pixel; EF, ETa and clear share of the 10 m pixel square as E2 computes them for a field (area-weighted over the clear 30 m Landsat pixels the square overlaps; mean of the scenes of one overpass, largest clear share; v0.9.0); stored as int16 (value / `pixels.*_scale`) |
| E3 | ERA5-Land hourly, Climate Data Store (`cdsapi`); NASADEM `nasadem` | 2 m T, 2 m dewpoint, 10 m u/v, surface pressure, ssrd, total precipitation (own folder `raw_precip/`); elevation | per 0.1° cell and local day: Rs = Σ hourly ssrd (accumulated from 00 UTC, ECMWF documentation), Tmin/Tmax of the hourly values, means of dewpoint, wind speed, pressure; elevation = mean of a 1 km window |

Export columns: E1 `field_id, date, ndvi, valid_fraction, n_scenes [, fapar_sl2p]`; E2 `field_id, date, sensor, ef,
eta_mm_day, rn_w_m2, g_w_m2, h_w_m2, le_w_m2, valid_fraction, item_id`; E3 `cell, date, rs_mj_m2, tmin_c, tmax_c,
tdew_c, u10_ms, sp_kpa, elev_m, lat, precip_mm`.

## Storage: GeoTIFF (v0.6.0)

Step 00c stores every satellite and weather dataset as GeoTIFF before anything is computed, and computes every
export value from the stored files. GeoTIFFs are tiled (512 pixels), DEFLATE-compressed (lossless) and keep the
grid, data type and no-data value of the source; blocks without data take no space.

| Dataset | Stored as | Content |
|---|---|---|
| Sentinel-2 | `data/stac/sentinel2/<run>/granules/<granule>_B04.tif`, `_B08.tif`, `_SCL.tif` | digital numbers of the source on its own grid (10 m, SCL 20 m) inside the pixel windows of the units – with `stac.sentinel2.coverage: all_parcels` (default) of every parcel of the crop map; no data elsewhere. The band scale and offset of the file give reflectance = (DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE |
| Landsat 8/9 | `data/stac/landsat/<run>/scenes/<scene>_<band>.tif` | digital numbers in the scene window (all units + `stac.landsat.context_km`): `qa_pixel`, the bands of `stac.landsat.download_assets`, and every band the EF function reads (SEBAL: blue, green, red, nir08, swir16, swir22, lwir11). Scale and offset of the file give reflectance or kelvin. With `stac.landsat.keep_source_bands: false` (default) the bands of a scene are deleted once its results are stored (field rows, clear mask, EF and ETa rasters): about 115 MB per scene in Dewas |
| ERA5-Land | `data/stac/era5land/<run>/raw/<month>_<variable>.tif`, `raw_precip/` | hourly values, one band per hour (band description = UTC time), pixel = grid node. The Climate Data Store delivers netCDF or GRIB, not GeoTIFF: the netCDF file is converted at once, the GeoTIFF is read again and compared value by value, and the netCDF is removed |
| Elevation | `data/stac/dem/<run>/nasadem_cells.tif` | NASADEM on the grid of its tiles over the weather cells (the elevation of a cell is computed from it); `nasadem_mosaic.tif` is the mosaic of the SEBAL terrain datum (v0.5.0, unchanged) |

Each granule, scene, month and raster has a record (`.json`) with the sha256 of its files; `s2_files.csv`,
`landsat_files.csv` and `raster_files.csv` list all of them.

**Preprocessing steps and their results** (`data/field_exports/<run>_rasters/`):

| Step | Result as GeoTIFF |
|---|---|
| Sentinel-2: scene classification to 10 m (nearest neighbour); valid pixel = class 4 or 5 | `ndvi/<run>_clear_<date>.tif`: 1 valid in a granule of the date, 0 not valid, 255 not observed |
| Sentinel-2: reflectance from the digital numbers and the offsets of the product metadata; pixels with red and NIR present and positive reflectance; mean of the overlapping granules of the date | `ndvi/<run>_red_<date>.tif`, `<run>_nir_<date>.tif`: reflectance × 10000 (`stac.sentinel2.reflectance_rasters`) |
| Sentinel-2: NDVI per granule on those pixels; mean of the overlapping granules | `ndvi/<run>_ndvi_<date>.tif` – the table E1 is computed from this raster and the clear raster |
| Landsat: QA_PIXEL bits to a clear mask | `landsat/<scene>_clear.tif`: 1 clear, 0 not clear, 255 outside the scene |
| Landsat: SEBAL per scene | `landsat/<scene>_ef.tif`, `<scene>_eta.tif` (`stac.landsat.result_rasters`; also possible: Rn, G, H, LE) |
| Weather: hourly to daily (local day), units converted | `weather/<run>_<variable>.tif`: one band per day on the 0.1° cells, for every E3 column |
| Weather: the two values step 01 derives from E3 (same functions, `core/eto.py`) | `weather/<run>_tmean_c.tif`, `<run>_eto_mm.tif` (FAO-56 Penman-Monteith); `<run>_elev_m.tif` |

**Maps (step 00e):** `data/maps/<run>/<group>/<name of the GeoTIFF>.jpg`, one per GeoTIFF: the file reduced to at
most 1100 pixels on its longer side (mean of the pixels with data; most frequent class for class rasters), one blue
ramp from light (low) to dark (high), white = no data; a file with several bands is shown as the mean of its bands.
Scattered small units are enlarged by 2 map pixels and the map says so. `outputs/<run>/00e_maps/maps.csv` lists every
map with its value range.

**Not GeoTIFF:** the export tables E1–E3 and the outputs of steps 01–10 are tables per field, parcel, tehsil or
district (CSV, one workbook); the pixel stacks and the pixel results of step 05b are numpy arrays (`.npy`, `.npz`)
with their records; units and boundaries are polygons (GeoPackage); records are JSON. The spatial yield products are
steps 11 (pixel-based, final) and 12 (parcel-mean, comparison).

## Pixel level (steps 05b and 11)

The yield model runs for every 10 m pixel of the crop map, then the pixel yields are aggregated: pixel → parcel →
ERA5-Land cell, tehsil, district (method document step 8: yield per pixel, then area-weighted aggregation).

| Part | What is done |
|---|---|
| Pixels | every pixel of the 10 m grid of the Sentinel-2 NDVI rasters whose centre is inside a parcel of the crop map; a pixel inside two overlapping parcels is a pixel of each; a parcel without such a pixel takes the pixel that holds its representative point (`pixel_rule`) |
| Pixel stacks (step 00c, export PX) | from the stored rasters, `data/field_exports/<run>_pixels/`: `ndvi.npy` (dates × pixels), `landsat_ef.npy`, `landsat_eta.npy`, `landsat_clear.npy` (overpasses × pixels), `pixels.npz` (parcel, row, column, centre), `parcels.csv`, `dates.csv`, `overpasses.csv`, `record.json` (what the stacks were made from, the Landsat rule, sha256 of every file). A pixel has NDVI only where its scene classification is valid, so its clear share of a date is 1 with NDVI and 0 without. Landsat (v0.9.0): the 10 m pixel square as a field of the export E2 (`stac_ingest.landsat_rows`, `overpass_rows`) - per scene the clear share of the square (clear area ÷ area inside the scene) and the area-weighted EF and ETa of its clear 30 m pixels with a value (`pixels.square_weights`: exact overlap of the square with each Landsat pixel; on another UTM zone the square is projected); over the scenes of one overpass the mean EF / ETa and the largest clear share. Stored as int16 = value / `pixels.ndvi_scale`, `ef_scale`, `eta_scale`, `clear_scale` (rounding at most half a step; step 00d checks a sample of squares against the E2 field code) |
| Weather | the daily E3 values of the parcel's 0.1° cell (a cell without E3: the nearest cell of E3) |
| Model (step 05b, `core/pixels.py`) | the rules of steps 03–05 for field units written for arrays of pixels: NDVI quality filter, despiking, daily linear interpolation and Savitzky–Golay smoothing, season dates at every amplitude fraction, Landsat overpasses with a clear share below `ef.min_valid_fraction` left out (as field units), exclusion rules (no Sentinel-2 value, no Landsat EF, fewer than 2 / `min_used_obs` clear dates, NDVI amplitude below `min_amplitude`), daily fAPAR and Kc from the smoothed NDVI, Landsat 8 / 9 EF and ETa/ETc linear in time between the overpasses (a missing sensor takes the other), CASA Topt of the pixel's NDVI-peak month, season sums S = Σ Ts · Ws · fAPAR · Rs for every (Topt, Ws form, season fraction) of the LUE-ENS members. Yield is linear in S (fresh = HI · εmax · f · S · 0.01 ÷ DM), so S gives every member's yield |
| Check of the model | `pixel_model.check_pixels` random pixels also run through the field-unit code of steps 03–05 (`core/season.py` `unit_seasons`, `core/drivers.py` `field_drivers`, `core/model.py` `members`): exclusion reason, season dates and every member yield must agree (`unit_check.csv`); a FAIL stops the step |
| Pixel → parcel | parcel = mean of its modelled pixels (all pixels have the same area); a pixel not modelled takes its parcel's value; a parcel without a modelled pixel takes the crop-area-weighted mean of the modelled parcels of its cell, else of its tehsil (`status`) |
| Parcel → cell, tehsil, district | crop-area-weighted mean of the parcels (`area_ha` of the crop map), members first; statistics across the members as step 06 (`members_<level>.csv`, `ensemble_<level>.csv`) |
| Calibration and products | step 08 calibrates the pixel-based district (placeholder benchmark) or tehsil (observed yields) members member-wise; step 09 reports the calibrated tehsil, district and cell values, with the stratified estimate of the field sample and its sampling error next to the district value |
| Pixel yield map (step 11) | every member of a pixel and of a parcel calibrated with its member factor, then the statistics across members; tehsil and district: production = Σ parcel yield × parcel crop area, yield = production ÷ area, P10–P90 by member-level aggregation |
| Checks of step 11 (`checks.csv`) | parcel production = tehsil sum = district total; district and tehsil values = step 09; the pixel-based district yield (model-based, every crop-map pixel) inside the 95 % interval of the stratified sample estimate of step 09 (design-based) |

Outputs of step 05b (`outputs/<run>/05b_pixel_model/`): `pixel_s.npy` (season sums, pixels × combinations of
`combos.csv`), `pixel_code.npy` (0 modelled, 1–5 reason not modelled), `pixel_source.npy` (0 own value, 1 parcel,
2 cell, 3 tehsil, 4 none), `pixel_season.npz` (central emergence, season end, NDVI peak, peak NDVI),
`parcel_values.csv` (per parcel: pixels, modelled pixels, status, season sums, uncalibrated LUE-ENS and central yield,
median season dates), `excluded_pixels.csv`, `members_<level>.csv`, `ensemble_<level>.csv` (tehsil, cell,
district), `member_grid.csv`, `unit_check.csv`, `checks.csv` (also the pixel-based yield of the sampled fields next to
their field-unit yield of step 05).

Outputs of step 11 (`outputs/<run>/11_pixel_map/`): `yield_<run>.tif` (bands `pixel_map.yield_raster_bands`, float32
on the 10 m grid of the Sentinel-2 rasters, no data outside the crop pixels) with its map `yield_<run>.jpg`;
`parcel_yield.gpkg` (layer `parcels`: polygon, tehsil, cell, crop area, pixels, status, season dates, `final_t_ha`,
`final_p10`, `final_p50`, `final_p90`, `lue_ens_t_ha`, `central_t_ha`, `production_t`) and the same table as
`parcel_yield.csv`; `tehsil_pixel_map.csv`, `district_pixel_map.csv` (with the step-09 values next to them).

Storage of a Dewas crop and season with the default settings: Sentinel-2 windows of every parcel and the date rasters
a few GB, the Landsat scene results about 1.5–2 GB, the pixel stacks about 0.7 GB (1.5 million pixels, 91 dates, 49
overpasses × 3 Landsat layers: 2 bytes per value), the weather and elevation under 0.1 GB.

## Parcel-mean yield map (step 12)

The final products model every 10 m pixel and then average (pixel → parcel, steps 05b, 09, 11). Step 12 models every
parcel as one unit on its mean inputs - exactly as the field sample of steps 01–05 - so that each parcel has two
values: the parcel-mean yield (step 12) and the pixel-based yield (step 11). Observed yields (crop cutting, farmer
yields) are plot or parcel values: the parcel-mean map is the product to compare with them. The difference
`pixel_minus_parcel_t_ha` shows where the pixel aggregation moves the yield: the model is not linear in NDVI and EF,
and a single pixel's NDVI series (edge pixels, residual clouds) gives season dates of its own. The final tehsil and
district values stay those of step 09.

| Part | What step 12 does |
|---|---|
| Inputs | the stored rasters of step 00c: Sentinel-2 NDVI and clear mask per date (`<run>_rasters/ndvi/`), Landsat clear mask, EF and ETa per scene (`<run>_rasters/landsat/`); the daily weather E3 per 0.1° cell (a parcel in a cell without E3 takes the nearest cell, flagged); the crop map; step 11 (pixel-based parcel yields) and step 09 |
| Values per parcel | as E1 (area-weighted NDVI, clear share per date) and E2 (clear share; EF, ETa of the clear pixels; one row per overpass) – the same functions (`core/stac_ingest.py` `s2_rows`, `landsat_rows`, `overpass_rows`) |
| Model | the code of steps 01, 03, 04 and 05 (`core/drivers.py` `field_tables`, `field_drivers`; `core/season.py` `unit_seasons`; `core/model.py`): season dates, daily drivers, every LUE-ENS member, the central configuration |
| Calibration | the member factors of the current calibration of step 08 (fitted on the pixel-based values), applied unchanged: the parcel-mean district value differs from the benchmark by the aggregation effect - reported, not corrected |
| Parcels not modelled | the rules of step 03 (no clear Sentinel-2 date, no Landsat EF, no clear NDVI cycle): the area-weighted yield of the modelled parcels of the cell, else of the tehsil (`status`) |
| Tehsil, district | production = Σ parcel yield × parcel crop area; yield = production ÷ area; P10–P90 by member-level aggregation over the modelled parcels (as step 06) |
| Checks (`checks.csv`) | the sampled fields: raster values = exports E1 and E2, central yield = step 05 (`parcel_map.sample_tolerance_t_ha`); parcel production = tehsil sum = district total; per parcel pixel-based (step 11) − parcel-mean yield: mean, median, P10–P90, correlation; district: step 09 (pixel-based) vs parcel-mean; the parcel-mean district yield inside the 95 % interval of the stratified sample estimate of step 09 |

Outputs (`outputs/<run>/12_parcel_map/`): `parcel_yield.gpkg` (layer `parcels`: polygon, tehsil, cell, crop area,
status, season dates, `final_t_ha`, `final_p10`, `final_p50`, `final_p90`, `lue_ens_t_ha`, `central_t_ha`,
`production_t`, quality counts, `pixel_based_t_ha`, `pixel_minus_parcel_t_ha`) and the same table as
`parcel_yield.csv`; `parcel_mean_yield_<run>.tif` (bands `parcel_map.yield_raster_bands`, the parcel value on the
10 m grid of the Sentinel-2 rasters, no data outside parcels) with its map `parcel_mean_yield_<run>.jpg`;
`tehsil_parcel_map.csv`, `district_parcel_map.csv` (with the values of steps 09 and 11 next to them). The step loops
over the parcels in Python (`parcel_map.chunk_parcels` at a time): it takes longer than step 05b.

## Equations and sources

Allen = Allen, Tasumi & Trezza 2007 (journal pages); B&A = Bastiaanssen & Ali 2003; FAO-56 = Allen et al. 1998
(online edition: several equations are printed as images, coefficients confirmed in the worked examples). PDFs are in
`references/` unless marked online.

### Yield model (`core/lue.py`, `core/model.py`)

| Quantity | Equation | Source | Status |
|---|---|---|---|
| PAR | f · Rs, f = 0.48 (0.50 in the ensemble) | B&A eq. 1 p.322; method document p.2; Potter et al. 1993 eq. 2 p.816 (0.5) | verified |
| fAPAR (no SL2P in E1) | −0.161 + 1.257 · NDVI, bounded 0–1 | B&A eq. 4 p.323 (f = 0–1 for NDVI 0.13–0.92, p.327); Jaafar & Mourad 2021 eq. 4 | verified. The method document prescribes SL2P (used when E1 carries `fapar_sl2p`) |
| APAR | fAPAR · PAR | B&A eq. 3; method document step 4 | verified |
| ε | εmax · Ts · Ws | B&A eq. 6 p.324; Potter et al. 1993 eq. 5 p.816; method document step 5 | verified |
| Ts | T1 · T2, bounded 0–1; T1 = 0.8 + 0.02 Topt − 0.0005 Topt²; T2 = 1.1814 / [1 + e^(0.2(Topt − 10 − T))] / [1 + e^(0.3(−Topt − 10 + T))]; T = daily mean air temperature | Potter et al. 1993 eqs. 6–7 p.816; B&A eqs. 8–9 p.324 | verified; daily instead of monthly T: adapted |
| Topt | per unit: mean daily air temperature of the calendar month of the unit's NDVI peak (step 03 peak, step 04 `topt_casa.csv`); same equation for every crop | Potter et al. 1993 p.816; B&A p.324 (`temperature_scalar.topt: casa_ndvi_peak_month`; `crop_values` = Topt of the crop file) | verified |
| Ws | EF = λE / (Rn − G), 0–1, interpolated daily between clear Landsat dates | B&A eq. 7 p.324; Yuan et al. 2007 eq. 5 p.192 (0–1); method document p.2 | verified. An LUE down-regulation scalar, not the FAO-56 soil-water stress coefficient Ks |
| Biomass B | Σ ε · APAR, emergence → season end (g m⁻²) | method document step 6 | verified |
| Yield | dry = HI · B; fresh = dry ÷ DM (t ha⁻¹ = g m⁻² × 0.01) | method document step 7 | verified |
| Tehsil / district | Σ (unit yield × area) ÷ Σ area; field units: separate ratio estimator | method document step 8; Cochran 1977 | verified / not seen |

Reported alternatives (step 07, never in LUE-ENS): Ws = 0.5 + 0.5 · min(1, ETa/ETc) and 0.5 + 0.5 · EF (Potter et al.
1993 eq. 8 p.818); Ws = 1; fAPAR from per-date NDVI; Landsat 8 or 9 only.

### SEBAL evaporative fraction (`core/ef_sebal.py`, step 00c)

| Quantity | Equation | Source |
|---|---|---|
| P, W, τsw, 1/d², Rs↓ | 101.3((293 − 0.0065z)/293)^5.26; 0.14 ea P + 2.1; 0.35 + 0.627 exp(−0.00146P/(Kt cosθ) − 0.075(W/cosθ)^0.4), Kt = 1; 1 + 0.033 cos(2π DOY/365); 1367 cosθ τsw/d² | Allen eqs. 5, 6, 4, 9, 3 (p.382); ea: FAO-56 eq. 14 |
| cosθ | sun elevation of the scene (STAC `view:sun_elevation`); else Allen eq. 8 with FAO-56 eqs. 24, 31–33 | Allen p.382 |
| α, NDVI, SAVI, LAI, ε0 | Σ Wb ρb (Wb 0.254, 0.149, 0.147, 0.311, 0.103, 0.036); (ρNIR − ρred)/(ρNIR + ρred); (1 + L)(ρNIR − ρred)/(L + ρNIR + ρred), L = 0.5; −ln((0.69 − SAVI)/0.59)/0.91, 0–6; 0.95 + 0.01 LAI (≤ 3), 0.98, water 0.985 | Allen eq. 15 + Table 1 p.383; eqs. 23, 19, 18, 17 p.384 |
| Ts | Landsat C2 L2 surface temperature ST_B10 (USGS single-channel algorithm, ASTER GED emissivity) – replaces Allen eqs. 20–22 | USGS LSDS-1619 v6.0 |
| εa, RL↓, RL↑, Rn | 1.08(−ln τsw)^0.265 (original SEBAL); εa σ Ta⁴, Ta = Ts of the cold pixel; ε0 σ Ts⁴; (1 − α)Rs↓ + RL↓ − RL↑ − (1 − ε0)RL↓ | Allen eqs. 25, 24, 16 (p.384), 2 (p.381) |
| G | Rn (Ts − 273.15)(0.0038 + 0.0074α)(1 − 0.98 NDVI⁴); water 0.5 Rn | Allen eq. 26, p.385 |
| z0m, u200, u*, rah | 0.018 LAI (≥ 0.005 m); u10 ln(200/z0m,w)/ln(10/z0m,w); k u200/ln(200/z0m); ln(z2/z1)/(u* k), z1 = 0.1, z2 = 2 m, k = 0.41 | Allen eqs. 33, 32, 31, 30 (p.386) |
| Ts,datum | Ts + 0.0065 (z − zdatum), z from the 30 m NASADEM, zdatum = median elevation of the window | Allen eq. 29 (p.385: Ts adjusted to a common elevation with a DEM and lapse rate); 0.0065 K m⁻¹: FAO-56 Annex 3 eq. 3-2 |
| ρair, dT, H | 1000P/(1.01(Ts − dT)287); dT = a + b Ts,datum through (Ts,datum cold, 0) and (Ts,datum hot, (Rn − G)rah/(ρair cp)); ρair cp dT/rah, cp = 1013 J kg⁻¹ K⁻¹ | Allen eqs. 37 (p.386), 29 (p.385), 46 (p.387), 28 (p.385); cp: FAO-56 eq. 8 |
| Stability | L = −ρair cp u*³ Ts/(k g H), g = 9.807; ψ from Paulson / Webb; u*, rah corrected and iterated | Allen eqs. 40–45, 38–39 (pp.386–387) |
| Hot pixel ET (METRIC) | bare-soil daily water balance of the surface layer: TEW = 1000(θFC − 0.5θWP)Ze, Ze = 0.10 m, θ from SoilGrids per tehsil; Kr = 1 (De ≤ REW) or (TEW − De)/(TEW − REW), REW = 10 mm; Ke = min(Kr(Kc,max − Kcb), few Kc,max), Kc,max = 1.2, Kcb = 0, few = 1; E = Ke ETo; De from ERA5-Land rain (eqs. 77–79, De = 0 on 1 Sep); EF_hot = E λ/Rn24; H_hot = (Rn − G)(1 − EF_hot) | Allen eq. 47 and p.387; FAO-56 ch. 7 eqs. 71–79, Table 19, Example 31 |
| LE, EF | Rn − G − H; LE/(Rn − G), bounded 0–1 | Allen eq. 1 p.381, p.392 (SEBAL EF); bounds: Yuan et al. 2007 p.192 |
| Daily ETa | EF · Rn24 / 2.45; Rn24 = (1 − α)Rs − Rnl (FAO-56 eqs. 21, 37–40), G24 = 0 (eq. 42) | derived from the EF definition (EF constant over the day, Allen p.392) |

SEBAL choices where METRIC differs (Allen p.387): cold pixel H = 0 (not 1.05 ETr), εa with the original SEBAL
coefficients, EF (not ETrF) constant over the day. The hot pixel follows METRIC: its ET comes from a FAO-56 bare-soil
water balance (`sebal.hot_pixel.method: soil_water_balance`; `zero` = SEBAL, LE = 0). A scene whose hot pixel has no
sensible heat left (wet bare soil after rain) gets no EF; its EF is interpolated from the neighbouring dates. Surfaces are treated as horizontal (eq. 8); the
dT function uses Ts reduced to a common elevation (eq. 29). Hot pixel on bare soil: NDVI < 0.13 (Bastiaanssen & Ali
2003 p.327).

**Settings (no literature value), `config/project.yaml` section `sebal`:** anchor rule (cold: NDVI ≥ 95th
percentile and Ts,datum ≤ 20th percentile of those; hot: bare soil and Ts,datum ≥ 80th percentile), the most
uniform 3 × 3 Ts neighbourhood among the candidates, at least 1,000 clear pixels, iteration tolerance 0.1 % of rah at
the hot pixel (≤ 30 iterations), z0m of the gridded wind (reference grass), 20 km scene window. The paper: anchor
selection is "highly reliant on the operator" (p.393). `sebal_anchors.csv` records per scene the anchors, their
homogeneity, the hot-pixel water balance, the atmosphere, convergence and clipped pixels; review it after every E2
run. Anchor sensitivity is a diagnostic: the three alternative anchor rules in `sebal.anchors.sensitivity_rules` are
solved per scene only when copied into `sebal.anchors.sensitivity` (empty in production; run it on a test period,
`--start` / `--end`), and their EF is then written to `sebal_anchors.csv`.

### Season, drivers, ETo (`core/season.py`, `core/drivers.py`, `core/eto.py`)

| Quantity | Method | Source | Status |
|---|---|---|---|
| Season dates | NDVI cleaned (clear share ≥ 0.5, despiking), daily linear interpolation, Savitzky-Golay (31 d, order 2); emergence / season end at base + p · (peak − base), p = 0.10, 0.20, 0.30 | Jönsson & Eklundh 2004; Chen et al. 2004 | papers not seen; all numbers settings |
| Daily EF, fAPAR (field units) | linear interpolation between clear observation dates | method document p.2 | verified |
| Curve fits (tehsil units) | double logistic (period-mean fit); smoothing spline with noise-based smoothing | Beck et al. 2006 eq. 3; Reinsch 1967; Morozov 1966 | Reinsch PDF; others not seen |
| ETo | FAO-56 Penman-Monteith, daily, G = 0, albedo 0.23, u2 from u10 (eq. 47) | FAO-56 eqs. 6–14, 21–25, 37–40, 47 | online; partly images |
| ETc (alternative only) | Kc · ETo; Kc = Kc,ini + (Kc,mid − Kc,ini) min(1, ML fc, fc^(1/(1+h))); fc = ((NDVI − NDVIsoil)/(NDVIfull − NDVIsoil))² | FAO-56 Table 12; Allen & Pereira 2009; Carlson & Ripley 1997 | see crop file |

### Ensemble, sensitivity, calibration, validation

LUE-ENS: full factorial of εmax × f × HI × DM × season threshold (Topt is data, not an ensemble factor) with product weights (equal unless stated);
weighted mean, SD, P10/P50/P90 and exact first-order variance shares – a scenario spread, not a probability interval.
Sobol indices: Saltelli sampling, N = 1024 (SALib; Saltelli et al. 2010). Calibration: one factor = mean observed ÷
mean simulated (Cochran 1977); it absorbs every model error without identifying it. Validation: bias, MAE, RMSE,
NRMSE, Willmott d, Pearson r² and coefficient of determination; acceptance targets frozen on 13 Nov 2026 (district
NRMSE 15 %, |bias| 5 %, tehsil NRMSE 20 %).

## Folders

| Folder | Content |
|---|---|
| `config/` | `project.yaml`, `crops/<crop>.yaml`, `aoi/<run>.yaml` |
| `core/` | model and data-access code |
| `data/field_exports/` | E1, E2, E3 (step 00c); `<run>_rasters/`: preprocessing and raster results as GeoTIFF; `<run>_pixels/`: pixel stacks (export PX; v0.9.0 adds `landsat_clear.npy`) |
| `data/stac/<source>/<run>/` | item and file tables, the stored GeoTIFFs (Sentinel-2 `granules/`, Landsat `scenes/`, ERA5-Land `raw/`, `raw_precip/`, elevation), their records, cell elevation, `sebal_anchors.csv` |
| `data/maps/<run>/` | JPG maps of the GeoTIFFs (step 00e) |
| `data/calibration/` | benchmark or observed yields (versioned) |
| `data/_test/` | test-period downloads (`--start`, `--end`); never read by the model steps |
| `outputs/<run>/<step>/` | step outputs with `manifest.json` |
| `references/` | sources (see `references/README.md`) |
| `tests/` | unit and synthetic tests (`python -m pytest -q tests`); CI: `.github/workflows/tests.yml` |

## Limitations

- εmax, HI and DM are literature values, not calibrated on local field data; without crop-cutting data the yield
  level rests on the calibration factor against official statistics.
- Ws = EF is a scene-calibrated energy-balance ratio: it is below 1 also without water stress (part of the available
  energy heats the air), falls with partial canopy cover, and depends on the anchor pixels.
- Landsat thermal pixels can mix the crop with other land in small parcels; Landsat gives only clear-sky days (fog in
  December–January leaves gaps that are interpolated). At pixel level a 10 m pixel takes the area-weighted EF of
  the 30 m Landsat pixels its square overlaps (v0.9.0); the thermal band is 100 m native, so neighbouring pixels still
  share most of their water scalar, and pixels at a parcel edge can carry the EF of the neighbouring land.
- The weather of a pixel is that of its 0.1° ERA5-Land cell (about 11 km); the yield pattern inside a cell comes from
  Sentinel-2 NDVI and Landsat EF only.
- Season dates are detected per pixel: a single pixel's NDVI is noisier than a parcel mean (mixed pixels at parcel
  edges, residual clouds), so more pixels can fail the season rules; such a pixel takes its parcel's value. The
  pixel-based parcel yield differs from the parcel-mean model (yield of the mean inputs) because the model is not
  linear in NDVI and EF; step 05b reports the difference for the sampled fields and step 12 for every parcel.
- Nutrients, pests, diseases and marketable losses are not simulated.
