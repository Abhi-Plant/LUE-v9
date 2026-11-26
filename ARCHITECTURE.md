# How the framework fits together

This file connects every step, module, equation, configuration value and output. The equations themselves, with their
sources, are in [README.md](README.md) ("Equations and sources"); the sources and checked pages in
[references/README.md](references/README.md).

## 1. The yield equation and where each term comes from

    fresh yield = HI · Σ_days [ εmax · Ts(T, Topt) · Ws · fAPAR · f · Rs ] / DM          (emergence → season end)

| Term | Meaning | Computed by | From | Configuration |
|---|---|---|---|---|
| Rs | daily global radiation, MJ m⁻² | `core/stac_ingest.era5land_daily` (step 00c, E3) | ERA5-Land `ssrd` | `stac.era5land` |
| f | PAR fraction of Rs | `core/lue.par` | Bastiaanssen & Ali 2003 eq. 1 | `parameters.f_par` (project) |
| fAPAR | absorbed PAR fraction | `core/lue.fapar_from_ndvi` (step 04) or SL2P from E1 | Sentinel-2 NDVI (E1), Bastiaanssen & Ali 2003 eq. 4 | `fapar` |
| Ts | temperature scalar 0–1 | `core/lue.ts` = `casa_t1 · casa_t2` | Potter et al. 1993 eqs. 6–7 | `temperature_scalar` |
| T | daily mean air temperature | step 04: (Tmin + Tmax)/2 | ERA5-Land `t2m` (E3) | – |
| Topt | optimum temperature per unit | `core/model.casa_topt` (step 04) | mean T of the NDVI-peak month (Potter p.816; B&A p.324) | `temperature_scalar.topt` |
| Ws | water scalar 0–1 = EF | `core/ef_sebal.compute_ef` (step 00c, E2); daily by `core/drivers.interpolate_dates` (step 04) | Landsat 8/9 SR + ST, ERA5-Land, NASADEM, SoilGrids | `sebal`, `water_stress` |
| εmax | maximum light-use efficiency | crop file | Kooman 1995 Table 3.1 (range) | `config/crops/<crop>.yaml` |
| HI, DM | harvest index, dry-matter fraction | crop file | AquaCrop Annex I | `config/crops/<crop>.yaml` |
| emergence, season end | summation window | `core/season.season_dates` (step 03) | Sentinel-2 NDVI series | `season`, `ndvi_cleaning` |

## 2. Pipeline: steps, modules, files

```
config/project.yaml ─┐
config/crops/<crop>.yaml ─┼─> core/config.load(run) ── every step
config/aoi/<run>.yaml ─┘

00a_parameters ── core/params ───────────────> outputs/<run>/00a_parameters/parameters.csv (every value + source)
02_fields ────── core/sampling ─────────────> outputs/<run>/02_fields/fields.gpkg, fields.csv (sample, weights)
   (00b_units: area polygon without crop map - download checks only)
00c_download ─── core/stac_ingest ───────────> data/stac/<source>/<run>/*.tif  (every dataset stored as GeoTIFF)
                  ├─ s2_item -> s2_date -> s2_rows ─> <run>_rasters/ndvi/*.tif -> <run>_E1_s2_perdate.csv (S2 NDVI)
                  │   (pixels of every parcel of the crop map: stac.sentinel2.coverage; E1 = the sample)
                  ├─ era5land_* , cell_elevation ─> <run>_E3_era5land_daily.csv              (weather, rain, elevation)
                  ├─ landsat_export ─> core/ef_sebal.compute_ef ─> <run>_E2_landsat_perdate.csv (EF, ETa, Rn, G, H, LE)
                  │                     │   and <run>_rasters/landsat/*_clear, _ef, _eta.tif (bands deleted after the
                  │                     │   scene results: stac.landsat.keep_source_bands)
                  │                     ├─ core/eto (FAO-56 Ra, ETo)
                  │                     ├─ core/soil_evaporation (FAO-56 bare-soil balance of the hot pixel)
                  │                     └─ NASADEM mosaic (Ts_datum)
                  └─ core/pixels.build_stacks (export px, from the rasters above) ─> <run>_pixels/ ndvi.npy (dates x
                                        pixels), landsat_ef.npy, landsat_eta.npy, landsat_clear.npy (overpasses x
                                        pixels; pixel square area-weighted as E2: square_weights, landsat_merged),
                                        pixels.npz, parcels.csv, record.json
00d_verify ───── core/stac_verify ───────────> outputs/<run>/00d_verify/summary.csv
00e_maps ─────── core/quicklook ─────────────> data/maps/<run>/<group>/*.jpg  (a map of every GeoTIFF; pictures only)
   field sample (steps 01-07: Sobol, variance shares, design-based estimate, checks)
01_inputs ────── core/drivers.field_tables, core/sampling ─> outputs/<run>/01_inputs/ units, obs_ndvi, obs_thermal, weather_daily, provenance, qc_checks
03_season ────── core/season.unit_seasons ───> outputs/<run>/03_season/ season.csv (emergence, peak, season end), season_grid.csv
04_drivers ───── core/drivers.field_drivers, core/lue, core/model ─> outputs/<run>/04_drivers/ drivers_daily.csv (Rs, T, Topt, fAPAR, EF), topt_casa.csv
05_lue ───────── core/lue, core/model ───────> outputs/<run>/05_lue/ members.csv, central_unit.csv, lue_daily_central.csv
   every crop-map pixel
05b_pixel_model ─ core/pixels (Stacks, model_vec = steps 03-05 vectorised; unit_check = the field-unit code on a
                  sample of pixels; parcel_means, fill_parcels, level_members) ─> outputs/<run>/05b_pixel_model/
                  pixel_s.npy, pixel_code.npy, pixel_source.npy, parcel_values.csv, members_<level>.csv,
                  ensemble_<level>.csv (tehsil, cell, district), unit_check.csv, checks.csv
06_ensemble ──── core/ensemble ──────────────> outputs/<run>/06_ensemble/ ensemble_<level>.csv, variance_shares.csv
07_sensitivity ─ core/model, core/lue (SALib) ─> outputs/<run>/07_sensitivity/ sobol_indices.csv, alternatives.csv
08_calibrate ─── core/calibration (members of 05b) ─> outputs/<run>/08_calibrate/<id>/ calibration_factor.csv, member_factors.csv, calibrated_*.csv
09_aggregate ─── core/sampling, core/ensemble (members of 05b; stratified estimate of the sample) ─> outputs/<run>/09_aggregate/ tehsil_final.csv, district_final.csv, cell_final.csv, workbook
10_validate ──── all model modules ──────────> outputs/<run>/10_validate/ validation_checks.csv
11_pixel_map ─── 05b pixel_s.npy, parcel_values.csv ─> calibration.latest (member factors of 08) ─> member
                  statistics per pixel and parcel ─> outputs/<run>/11_pixel_map/ yield_<run>.tif (+ .jpg),
                  parcel_yield.gpkg, tehsil_pixel_map.csv, district_pixel_map.csv, checks.csv
   every parcel on its mean inputs (comparison product, v0.9.0)
12_parcel_map ── core/stac_ingest (s2_rows, landsat_rows, overpass_rows from the rasters of 00c), core/drivers,
                  core/season, core/model (the field-unit code of 01-05), calibration.latest; 11 parcel_yield.csv
                  ─> outputs/<run>/12_parcel_map/ parcel_yield.gpkg (parcel-mean and pixel-based yield, difference),
                  parcel_mean_yield_<run>.tif (+ .jpg), tehsil_parcel_map.csv, district_parcel_map.csv, checks.csv
run.py ───────── runs the steps in this order, skips complete steps (core/config.step_complete), --detach
```

## 3. Modules

| Module | Role | Used by | Main functions |
|---|---|---|---|
| `core/config.py` | loads project + crop + run config (checks every crop value has a source and status), paths, atomic writes, manifests with config fingerprint, complete-step test | every step, `run.py` | `load`, `load_crop`, `crop_entries`, `output_dir`, `write_manifest`, `step_complete`, `to_csv_atomic` |
| `core/params.py` | questions and checks of step 00a, run-config template | `00a_parameters.py`, `run.py` | `collect`, `record` |
| `core/stac_ingest.py` | STAC search and download, GeoTIFF storage of every dataset (granules, Landsat bands, weather months, elevation) with records, preprocessing rasters, pixel weights of the units, E1 / E2 / E3 exports computed from the stored files; the parcels of the crop map; values per parcel read from the stored rasters | 00c, 00d, 00e, 12 | `store_clip`, `write_tif`, `s2_item`, `s2_date`, `s2_export`, `s2_rows`, `parse_s2_offsets`, `landsat_export`, `landsat_rows`, `overpass_rows`, `scene_fingerprint`, `load_parcels`, `nc_to_tifs`, `era5_hours`, `weather_export`, `weather_rasters` |
| `core/pixels.py` | pixel index (centre inside the parcel), raster values at pixel positions, pixel stacks (export px) and their reading; the model of steps 03-05 vectorised over pixels; the same pixels through the field-unit code (check); aggregation pixel -> parcel -> cell, tehsil, district | 00c, 00d, 05b, 10, 11, 12 | `pixel_index`, `raster_values`, `scene_positions`, `square_weights`, `landsat_merged`, `build_stacks`, `Stacks`, `seasons_vec`, `model_vec`, `unit_check`, `combos_of`, `yields_from_s`, `parcel_means`, `fill_parcels`, `level_members`, `member_stats_rows` |
| `core/quicklook.py` | JPG map of a GeoTIFF (reduced, one blue ramp for values, fixed colours for classes) | 00e, 11, 12 | `describe`, `make_map` |
| `core/ef_sebal.py` | SEBAL energy balance per Landsat scene: albedo, NDVI, SAVI, LAI, emissivity, Rn, G, Ts_datum, hot / cold pixel, iterated H, EF, ETa | `stac_ingest.landsat_export` (via `stac.landsat.ef_function`) | `compute_ef`, `hot_pixel_et`, `stability`, `scene_dem`, `_anchor`, `local_ts_sd` |
| `core/soil_evaporation.py` | FAO-56 ch. 7 bare-soil evaporation (TEW, Kr, Ke, daily balance); soil per tehsil from SoilGrids | `ef_sebal`, 00c | `tew`, `kr`, `bare_soil`, `load_soil_units` |
| `core/eto.py` | FAO-56 Ra, e°, ETo, wind at 2 m | `ef_sebal`, `drivers.weather_eto` | `extraterrestrial_radiation`, `svp`, `eto_fao56`, `wind_2m` |
| `core/stac_verify.py` | second-method recomputation of samples, completeness, integrity of the stored GeoTIFFs (sha256, pixels against the source) | 00d | `stored_files`, `stored_pixels_s2`, `stored_pixels_landsat`, `ndvi_sample`, `clear_sample`, `weather_hours`, `netcdf_check`, `elevation_sample`, `weather_sample` |
| `core/season.py` | NDVI cleaning, daily series, emergence / peak / season end; the season of every unit | 03, 05b, 12 | `clean_series`, `daily_series`, `season_dates`, `usable`, `unit_seasons` |
| `core/drivers.py` | interpolation, period helpers, double logistic, smoothing spline, sensor gap filling; input tables and daily drivers of field units (one implementation for the sample and every parcel) | 01, 04, 05, 05b, 10, 12 | `interpolate_dates`, `fit_double_logistic`, `fit_smoothing_spline`, `fill_sensors`, `field_tables`, `weather_eto`, `clip_shares`, `field_drivers` |
| `core/lue.py` | the LUE equations: PAR, fAPAR(NDVI), T1, T2, Ts, Ws forms, Kc, daily biomass, yield | 04, 05, 07, 10 | `par`, `fapar_from_ndvi`, `ts`, `ws`, `crop_coefficient`, `crop_yield` |
| `core/model.py` | season sums and yield of a configuration and of all ensemble members; CASA Topt | 04, 05, 05b, 07, 08, 10, 11, 12 | `casa_topt`, `topt_of`, `season_sum`, `run`, `members` |
| `core/ensemble.py` | member grid with weights, weighted statistics, variance partition, member aggregation | 06, 08, 09, 10, 11, 12 | `member_grid`, `member_stats`, `variance_partition`, `aggregate_members` |
| `core/sampling.py` | parcels of the crop map, stratified simple random sample, inclusion probabilities, ratio estimator | 02, 00c, 01, 09, 11 | `parcel_frame`, `select_srs`, `ratio_weights`, `stratified_estimate` |
| `core/calibration.py` | calibration input checks, id, ratio of means, the current calibration | 08, 09, 10, 11, 12 | `read_inputs`, `calibration_id`, `ratio_of_means`, `latest` |
| `core/stats.py` | weighted quantiles, validation metrics | 07, 08, 09, 10 | `wquantile`, `metrics` |
| `core/plots.py` | figures | 02, 03, 04, 05, 07, 09 | – |

## 4. The EF chain inside `core/ef_sebal.compute_ef`

```
Landsat C2 L2 scene window ──> clear pixels (QA_PIXEL bits 0-5)
  SR_B2..B7 ──> albedo (Allen eq. 15) ──────────────────────────────┐
  SR_B4, B5 ──> NDVI (23), SAVI (19), LAI (18), ε0 (17) ─────────────┤
  ST_B10 ─────> Ts ──> Ts_datum = Ts + 0.0065 (z − z_datum)  [NASADEM]│
ERA5-Land hour of overpass ──> P (5), ea (FAO-56 14), W (6), τsw (4), Rs↓ (3)
                                                                     ▼
  anchors: cold = high NDVI, low Ts_datum; hot = bare soil (NDVI < 0.13), high Ts_datum; most uniform 3 × 3
  Rn (2) with RL↓ (24, 25), RL↑ (16);  G (26)
  hot pixel ET: core/soil_evaporation.bare_soil(ERA5-Land rain, FAO-56 ETo, SoilGrids TEW) ──> EF_hot = E λ / Rn24
  H_hot = (Rn − G)(1 − EF_hot) (eq. 47) ──> dT_hot (46) ──> dT = a + b Ts_datum (29)
  z0m (33), u200 (32), u*, rah (31, 30) ──> iterate ρair (37), H (28), L (40), ψ (41-45), u* (38), rah (39)
  LE = Rn − G − H (1) ──> EF = LE / (Rn − G) (p. 392), 0-1 ──> ETa = EF · Rn24 / λ (FAO-56 Rn24)
  ──> E2 field means: ef, eta_mm_day, rn/g/h/le; sebal_anchors.csv: one row per scene
```

## 5. Configuration ↔ code

| Section | Read by | Holds |
|---|---|---|
| `parameters.f_par` + crop `parameters` | `core/ensemble.member_grid`, `core/model.central_config` | ensemble values of f, εmax, HI, DM (Topt: offset 0 under the CASA definition) |
| `temperature_scalar` | `core/config.load`, step 04 | Topt definition (CASA per unit or crop values) |
| `water_stress` | `core/ensemble`, `core/lue.ws` | Ws form in LUE-ENS (EF) and reported alternatives |
| `sebal` | `core/ef_sebal`, `core/soil_evaporation` | SEBAL constants with sources; hot-pixel water balance; anchor settings; terrain datum |
| `season`, `ndvi_cleaning` + crop `season` | `core/season`, step 03 | NDVI cleaning and threshold settings |
| `fapar`, `curve_fit`, `weather`, `biomass_method` | steps 01, 04, 05 | fAPAR relation, curve fitting, tehsil-unit options |
| crop `crop_coefficient`, `heat_stress`, `yield_basis` | steps 04, 09 | ETa/ETc alternative, heat flag, yield basis |
| `ensemble`, `sobol`, `calibration`, `acceptance_targets` | steps 06-10 | weights, sampling, calibration and validation settings |
| `data_qc`, `verification_tolerance`, `quality_flags`, `energy_balance` | steps 00c, 00d, 01, 09 | plausibility limits of the checks (settings) |
| `stac` | `core/stac_ingest`, `core/stac_verify` | collections, assets, masks, workers; storage settings (`sentinel2.coverage`, `landsat.keep_source_bands`) |
| `pixels` | `core/pixels`, steps 00c, 00d | storage of the pixel stacks (int16 scales of NDVI, EF, ETa, Landsat clear share) |
| `pixel_model` | step 05b | chunk size, number of pixels checked against the field-unit code, tolerance |
| `pixel_map` | step 11 | chunk size, bands of the yield GeoTIFF |
| `parcel_map` | step 12 | parcels per chunk, tolerance of the sample check, bands of the parcel-mean GeoTIFF |

## 6. Settings (values that are not literature values)

Every other number is an equation constant or a crop value with its source. These are framework choices; each is
labelled `setting` in its configuration file and none is hidden in code:

| Setting | Where | Value |
|---|---|---|
| SEBAL anchor quantiles, homogeneity window, minimum clear pixels, sensitivity rules | `sebal.anchors` | 0.95 / 0.20 (cold), 0.80 (hot Ts), 3 × 3, 1000 |
| Iteration stop | `sebal.iterations` | 0.1 % of rah, ≤ 30 |
| Wind roughness of the ERA5-Land wind | `sebal.z0m_weather_m` | reference grass (FAO-56 definition) |
| Hot-pixel water balance: REW within the Table 19 range, no irrigation, De = 0 on 1 Sep | `sebal.hot_pixel` | 10 mm; 0; 0 |
| Scene window margin | `stac.landsat.context_km` | 20 km |
| NDVI cleaning and season threshold | `ndvi_cleaning`, `season`, crop `season` | 0.5 clear share, despike 0.10 / 20 d, SG 31 d, p 0.10-0.30, amplitude 0.15 |
| fAPAR, EF daily: linear interpolation between clear dates | step 04 | method document |
| Ensemble weights, quantiles; Sobol sample size | `ensemble`, `sobol` | equal; 0.1 / 0.5 / 0.9; 1024 |
| QC plausibility limits and verification tolerances | `data_qc`, `verification_tolerance`, `quality_flags`, `energy_balance` | flags only |
| Heat-day flag | crop `heat_stress` | flag only, no yield effect |
| Acceptance targets | `acceptance_targets` | frozen 13 Nov 2026 |

## 7. Outputs that document a run

| File | Content |
|---|---|
| `outputs/<run>/00a_parameters/parameters.csv` | every parameter and crop value with its source and status |
| `outputs/<run>/<step>/manifest.json` | inputs and outputs with sha256, configuration fingerprint, package versions, run time, peak memory |
| `data/stac/landsat/<run>/sebal_anchors.csv` | per Landsat scene: anchors, homogeneity, hot-pixel water balance, atmosphere, terrain, convergence, anchor sensitivity |
| `outputs/<run>/04_drivers/topt_casa.csv` | CASA Topt per unit: NDVI-peak date and month, days, Topt |
| `outputs/<run>/01_inputs/provenance.csv`, `qc_checks.csv` | input fingerprints and documentation, data checks |
| `outputs/<run>/10_validate/validation_checks.csv` | equation, consistency, reproducibility and plausibility checks |
