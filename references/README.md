# Sources of the equations and values

"Checked" = the equation or value was read on the page given. "Not seen" = the source is not available here (paywall,
book or no open copy); what the framework uses from it is marked `not_verified` or cited only through a checked
source. Checked 23 Nov 2026.

## In this folder

| File | Source | Used for | Checked pages |
|---|---|---|---|
| `LUE_Potato_Yield_Methodology_method_document.pdf` | Method document: LUE-based potato yield estimation (2 pages) | core equation, steps 1–8, variable table | pp.1–2 |
| `Bastiaanssen_Ali_2003_AgricEcosystEnviron_LUE_SEBAL_yield.pdf` | Bastiaanssen W.G.M., Ali S. (2003) A new crop yield forecasting model based on satellite measurements applied across the Indus Basin, Pakistan. Agric Ecosyst Environ 94:321–340 | PAR = 0.48 K24 (eq. 1, p.322); f = −0.161 + 1.257 NDVI (eq. 4, p.323); ε = ε′T1T2W (eq. 6), W = Λ = λE/(Rn − G0) (eq. 7), T1, T2 (eqs. 8–9) (p.324); Λ and f constant between images (p.327); ε′ calibrated against district statistics within literature ranges (p.330, Table 2); Appendices A (ε) and B (HI) | 322–324, 327, 330, 337–339 |
| `Yuan_2007_EC-LUE_AgricForMeteorol_CC-BY.pdf` | Yuan W. et al. (2007) Deriving a light use efficiency model from eddy covariance flux data for predicting daily gross primary production across biomes. Agric For Meteorol 143:189–207 (open access, CC-BY) | Ws = EF = LE/(LE + H) (eq. 5); Ts and Ws between 0 and 1; ε = εmax min(Ts, Ws) (eq. 3) | 192 |
| `Allen_2007_METRIC_model_JIrrigDrainEng.pdf` | Allen R.G., Tasumi M., Trezza R. (2007) Satellite-based energy balance for mapping evapotranspiration with internalized calibration (METRIC) – Model. J Irrig Drain Eng 133(4):380–394 | SEBAL energy balance: eqs. 1–2 (p.381), 3–9 (p.382), Table 1 and eq. 15 (p.383), 16–25 (p.384), 26–29 (p.385), 30–38 (p.386), 39–46 (p.387); SEBAL EF constant over the day (p.392); anchor selection operator-dependent (p.393) | 381–387, 392–393 |
| `Potter_1993_CASA_GlobalBiogeochemCycles.pdf` | Potter C.S. et al. (1993) Terrestrial ecosystem production: a process model based on global satellite and surface data. Global Biogeochem Cycles 7(4):811–841 | IPAR with 0.5 (eq. 2), ε = Tε1 Tε2 Wε ε* (eq. 5), Tε1 (eq. 6), Tε2 (eq. 7); Topt = air temperature of the month of maximum NDVI (p.816); Wε = 0.5 + 0.5 EET/PET (eq. 8, p.818) | 816, 818 |
| `Jaafar_Mourad_2021_GYMEE_RemoteSens_secondary_fPAR_PAR.pdf` | Jaafar H., Mourad R. (2021) GYMEE. Remote Sens 13:773 | fPAR eq. 4 and PAR = 0.48 Ra τ (eq. 5) (p.4); pySEBAL, constant EF, advection factor (p.5); soil-moisture stress (pp.6–7) – not used | 4–7 |
| `Kooman_1995_PhD_potato_temperature_daylength.pdf` | Kooman P.L. (1995) Yielding ability of potato crops as influenced by temperature and daylength. PhD thesis, Wageningen | LUE 1.8–3.5 g MJ⁻¹ intercepted PAR (Table 3.1, p.41; PAR intercepted, p.37); HI 0.51–0.85 (p.61), "about 0.75" (p.102); optimum 18 / 21 °C for tuber allocation (p.86) | 37, 41, 61, 86, 102 |
| `Raes_AquaCrop_v7.3_Annexes_crop_parameters.pdf` | Raes D. et al. AquaCrop v7.3 Reference Manual, Annex I | potato HIo 70–85 %, dry-matter conversion 20–25 % (p.13) | 13 |
| `Reinsch_1967_smoothing_spline_NumerMath.pdf` | Reinsch C.H. (1967) Smoothing by spline functions. Numer Math 10:177–183 | smoothing spline (method) | – |

## Online (official documents)

| Source | Used for | Checked |
|---|---|---|
| Allen R.G. et al. (1998) FAO Irrigation and Drainage Paper 56, https://www.fao.org/4/x0490e/x0490e00.htm | ETo (eq. 6), constants (eq. 8: cp, λ), e° (eq. 11), ea (eq. 14), Ra (eqs. 21–25), Rso, Rns, Rnl, Rn (eqs. 37–40), G (eq. 42), u2 (eq. 47), z_om = 0.123 h (ch. 2 Box 4), Table 12 Kc and height; bare-soil evaporation ch. 7: Ke (eq. 71), Kc max (eq. 72, Example 31), TEW (eq. 73), Kr (eq. 74), few (eq. 75), De balance (eqs. 77–79), Table 19 (REW, TEW by soil type); Ks (ch. 8 eqs. 80–84); lapse rate 0.0065 K m⁻¹ (Annex 3 eq. 3-2) | text and worked examples; many equations are images in the online edition; potato Kc ini cell not readable |
| USGS (2024) Landsat 8-9 Collection 2 Level 2 Science Product Guide, LSDS-1619 v6.0 | SR and ST scale factors (Table 6-1), QA_PIXEL bits (Table 6-2), ST algorithm (single channel, ASTER GED, GEOS-5) | Tables 6-1, 6-2 |
| ESA SentiWiki, Sentinel-2 Products and Processing | L2A reflectance = (DN + BOA_ADD_OFFSET)/QUANTIFICATION_VALUE from baseline 04.00; SCL classes | text |
| ECMWF, ERA5-Land data documentation; CDS dataset page; Muñoz-Sabater et al. (2021) ESSD 13:4349–4383 | 0.1° hourly grid, units, accumulation of ssrd from 00 UTC | text |
| NASADEM STAC collection (Planetary Computer) | 1 arc-second elevation derived from SRTM (cell elevation; Ts_datum) | text |
| Poggio L. et al. (2021) SoilGrids 2.0, SOIL 7:217–240 (data: ISRIC, CC-BY 4.0; tehsil summary of the Dewas AquaCrop work) | water content at 33 / 1500 kPa of the topsoil (θFC, θWP for TEW) | data workbook |

## Not seen (cited, not available here)

Bastiaanssen 1995; Bastiaanssen et al. 1998; Bastiaanssen 2000 (SEBAL originals, used as set out by Allen 2007); USDA Soil Survey Manual (texture classes - not needed: TEW from SoilGrids water contents);
Allen et al. 2013 and Bhattarai et al. 2017 (automated anchor selection); Allen & Pereira 2009 (Kc from cover);
Carlson & Ripley 1997 (fc); Beck et al. 2006 (double logistic); Jönsson & Eklundh 2004 (TIMESAT); Chen et al. 2004
(Savitzky-Golay NDVI); Morozov 1966; Saltelli et al. 2010 (Sobol); Cochran 1977 (sampling); Willmott 1981
(index of agreement); Monteith 1977; Struik 2007; the printed FAO-56 (equation images, potato Kc ini); ESA SNAP
Sentinel-2 biophysical processor ATBD (SL2P network coefficients, needed to compute SL2P fAPAR in step 00c).
