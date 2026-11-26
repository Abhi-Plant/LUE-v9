"""FAO-56 Penman-Monteith reference evapotranspiration, daily (Allen et al. 1998, eq. 6 with G = 0), for the
field-unit weather export E3 when it carries ERA5-Land variables instead of ETo."""
import numpy as np


def extraterrestrial_radiation(lat_deg, doy):
    """Ra, MJ m-2 d-1 (FAO-56 eq. 21-25)."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    delta = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def svp(t):
    """Saturation vapour pressure, kPa (eq. 11)."""
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def eto_fao56(tmin, tmax, tdew, rs, u2, p_kpa, lat_deg, elev_m, doy):
    """Daily ETo (mm d-1). rs in MJ m-2 d-1, u2 wind at 2 m (m s-1), p_kpa surface pressure."""
    tmean = (tmin + tmax) / 2
    es = (svp(tmin) + svp(tmax)) / 2
    ea = svp(tdew)
    delta = 4098 * svp(tmean) / (tmean + 237.3) ** 2
    gamma = 0.000665 * p_kpa
    ra = extraterrestrial_radiation(lat_deg, doy)
    rso = (0.75 + 2e-5 * elev_m) * ra
    rns = (1 - 0.23) * rs
    rnl = 4.903e-9 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(ea)) \
        * (1.35 * np.clip(rs / rso, 0, 1) - 0.35)
    rn = rns - rnl
    return (0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))


def wind_2m(u10):
    """Wind at 2 m from 10 m (FAO-56 eq. 47)."""
    return u10 * 4.87 / np.log(67.8 * 10 - 5.42)
