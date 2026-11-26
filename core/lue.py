"""The documented LUE equations (method document p.1), the CASA stress scalars (Potter et al. 1993) and the crop
coefficient used by the relative-ET water scalar (FAO-56; Allen & Pereira 2009; Carlson & Ripley 1997).

    PAR   = f * Rs                                   MJ m-2 d-1
    APAR  = fAPAR * PAR                              MJ m-2 d-1
    eps   = eps_max * Ts * Ws                        g MJ-1
    B     = sum(eps * APAR), emergence -> season end     g m-2
    dry   = HI * B ;  fresh = dry / DM               g m-2 -> t/ha (x 0.01)
"""
import numpy as np

G_M2_TO_T_HA = 0.01


def par(rs, f):
    return f * np.asarray(rs, dtype=float)


def fapar_from_ndvi(ndvi, slope, intercept):
    """Linear NDVI -> fAPAR relation, bounded to 0-1."""
    return np.clip(slope * np.asarray(ndvi, dtype=float) + intercept, 0.0, 1.0)


def casa_t1(topt):
    """Potter et al. (1993) eq. 6: T1 = 0.8 + 0.02 Topt - 0.0005 Topt^2."""
    return 0.8 + 0.02 * np.asarray(topt, dtype=float) - 0.0005 * np.asarray(topt, dtype=float) ** 2


def casa_t2(t, topt):
    """Potter et al. (1993) eq. 7: asymmetric bell around Topt."""
    t, topt = np.asarray(t, dtype=float), np.asarray(topt, dtype=float)
    return 1.1814 / (1.0 + np.exp(0.2 * (topt - 10.0 - t))) / (1.0 + np.exp(0.3 * (-topt - 10.0 + t)))


def ts(tmean, topt):
    """Temperature scalar Ts = T1 * T2 (CASA), bounded to 0-1."""
    return np.clip(casa_t1(topt) * casa_t2(tmean, topt), 0.0, 1.0)


def ws(x, form):
    """Water scalar. 'ef': Ws = EF, x = EF = lambdaE / (Rn - G) (Bastiaanssen & Ali 2003 eq. 7; Yuan et al. 2007
    eq. 5; method document) - an LUE down-regulation scalar 0-1, not the FAO-56 soil-water stress coefficient Ks.
    'relative_et': Ws = 0.5 + 0.5 min(1, ETa/ETc) (Potter et al. 1993 eq. 8), x = ETa/ETc.
    'casa_ef': Ws = 0.5 + 0.5 EF (reported alternative), x = EF.  'none': Ws = 1 (upper bound)."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    if form == "ef":
        return x
    if form in ("relative_et", "casa_ef"):
        return 0.5 + 0.5 * x
    if form == "none":
        return np.ones_like(x)
    raise ValueError(f"unknown Ws form {form}")


def fractional_cover(ndvi, ndvi_soil, ndvi_full):
    """Carlson & Ripley (1997): fc = scaled NDVI squared, bounded to 0-1."""
    n = np.clip((np.asarray(ndvi, dtype=float) - ndvi_soil) / (ndvi_full - ndvi_soil), 0.0, 1.0)
    return n ** 2


def crop_coefficient(ndvi, cc):
    """Kc = Kc_ini + (Kc_mid - Kc_ini) Kd ; Kd = min(1, ML fc, fc^(1/(1+h))) (Allen & Pereira 2009);
    cc = config crop_coefficient."""
    fc = fractional_cover(ndvi, cc["ndvi_soil"], cc["ndvi_full"])
    kd = np.minimum.reduce([np.ones_like(fc), cc["ml"] * fc, fc ** (1.0 / (1.0 + cc["crop_height_m"]))])
    return cc["kc_ini"] + (cc["kc_mid"] - cc["kc_ini"]) * kd


def daily_biomass(eps_max, ts_, ws_, fapar, rs, f):
    """Daily dry-matter growth, g m-2 d-1 = eps_max * Ts * Ws * fAPAR * f * Rs."""
    return eps_max * ts_ * ws_ * fapar * par(rs, f)


def crop_yield(biomass_g_m2, hi, dm):
    """Dry and fresh yield in t/ha from seasonal dry biomass (g m-2)."""
    dry = hi * np.asarray(biomass_g_m2, dtype=float) * G_M2_TO_T_HA
    return dry, dry / dm
