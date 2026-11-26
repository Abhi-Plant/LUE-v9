"""Tests of the v0.5 parts: crop files, Sentinel-2 offsets from product metadata, SEBAL stability functions and anchor
selection. Synthetic data only; no download, no run of the steps."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config, ef_sebal, stac_ingest  # noqa: E402


# ---------------------------------------------------------------- crop files
def test_every_crop_file_loads_with_sources():
    for p in (config.ROOT / "config" / "crops").glob("*.yaml"):
        crop = config.load_crop(p.stem)
        rows = config.crop_entries(crop)
        assert rows and all(r["status"] in config.CROP_STATUS and r["source"] for r in rows)


def test_crop_file_without_source_is_refused(tmp_path, monkeypatch):
    crop = yaml.safe_load(open(config.ROOT / "config" / "crops" / "potato.yaml", encoding="utf-8"))
    del crop["parameters"]["hi"]["source"]
    (tmp_path / "config" / "crops").mkdir(parents=True)
    (tmp_path / "config" / "crops" / "nosource.yaml").write_text(yaml.safe_dump(crop), encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="parameters.hi"):
        config.load_crop("nosource")


def test_run_config_merges_crop_values():
    cfg = config.load("dewas_2025_26")
    crop = config.load_crop(cfg["aoi"]["crop"])
    assert cfg["parameters"]["eps_max"]["values"] == crop["parameters"]["eps_max"]["values"]
    assert cfg["crop_coefficient"]["kc_mid"] == crop["crop_coefficient"]["kc_mid"]["value"]
    assert cfg["calibration"]["expected"] == crop["calibration_expected"]
    assert "f_par" in cfg["parameters"]                       # PAR fraction stays a method value


# ---------------------------------------------------------------- Sentinel-2 offsets
XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-2A_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd">
 <General_Info><Product_Image_Characteristics>
  <QUANTIFICATION_VALUES_LIST><BOA_QUANTIFICATION_VALUE unit="none">10000</BOA_QUANTIFICATION_VALUE>
  </QUANTIFICATION_VALUES_LIST>
  <BOA_ADD_OFFSET_VALUES_LIST>
   <BOA_ADD_OFFSET band_id="3">-1000</BOA_ADD_OFFSET><BOA_ADD_OFFSET band_id="7">-1000</BOA_ADD_OFFSET>
   <BOA_ADD_OFFSET band_id="8">-999</BOA_ADD_OFFSET>
  </BOA_ADD_OFFSET_VALUES_LIST>
  <Spectral_Information_List>
   <Spectral_Information bandId="3" physicalBand="B4"/><Spectral_Information bandId="7" physicalBand="B8"/>
   <Spectral_Information bandId="8" physicalBand="B8A"/>
  </Spectral_Information_List>
 </Product_Image_Characteristics></General_Info>
</n1:Level-2A_User_Product>"""


def test_s2_offsets_from_metadata():
    offs, quant = stac_ingest.parse_s2_offsets(XML, ["B04", "B08", "B8A"])
    assert offs == {"B04": -1000.0, "B08": -1000.0, "B8A": -999.0} and quant == 10000.0


def test_s2_no_offset_list_means_zero():
    xml = XML.replace(b"<BOA_ADD_OFFSET_VALUES_LIST>", b"<X>").replace(b"</BOA_ADD_OFFSET_VALUES_LIST>", b"</X>")
    xml = xml.replace(b"BOA_ADD_OFFSET band_id", b"Y band_id").replace(b"</BOA_ADD_OFFSET>", b"</Y>")
    offs, _ = stac_ingest.parse_s2_offsets(xml, ["B04", "B08"])
    assert offs == {"B04": 0.0, "B08": 0.0}


# ---------------------------------------------------------------- SEBAL
def test_stability_unstable_matches_eqs_41_43():
    L = np.array([-50.0])
    pm, ph2, ph01 = ef_sebal.stability(L)
    x200, x2, x01 = ((1 - 16 * z / -50.0) ** 0.25 for z in (200, 2, 0.1))
    assert pm[0] == pytest.approx(2 * math.log((1 + x200) / 2) + math.log((1 + x200 ** 2) / 2)
                                  - 2 * math.atan(x200) + 0.5 * math.pi)
    assert ph2[0] == pytest.approx(2 * math.log((1 + x2 ** 2) / 2))
    assert ph01[0] == pytest.approx(2 * math.log((1 + x01 ** 2) / 2))


def test_stability_stable_uses_2_m_for_momentum():
    pm, ph2, ph01 = ef_sebal.stability(np.array([40.0]))
    assert pm[0] == pytest.approx(-5 * 2 / 40) and ph2[0] == pytest.approx(-5 * 2 / 40)
    assert ph01[0] == pytest.approx(-5 * 0.1 / 40)


def test_stability_neutral_is_zero():
    assert all(v[0] == 0 for v in ef_sebal.stability(np.array([np.inf])))


def test_local_sd_needs_full_clear_neighbourhood():
    ts = np.full((5, 5), 300.0)
    ok = np.ones((5, 5), bool)
    ok[0, 0] = False
    sd = ef_sebal.local_ts_sd(ts, ok, 3)
    assert np.isnan(sd[1, 1]) and sd[2, 2] == pytest.approx(0.0)


def test_anchor_prefers_homogeneous_candidate():
    ndvi = np.array([0.9, 0.9, 0.9, 0.2, 0.1])
    ts = np.array([290.0, 291.0, 292.0, 310.0, 315.0])
    sd = np.array([np.nan, 0.2, 0.8, 1.0, 1.0])
    i, hom = ef_sebal._anchor(ndvi, ts, np.ones(5, bool), 0.5, 1.0, True, sd)
    assert i == 1 and hom
    i, hom = ef_sebal._anchor(ndvi, ts, np.ones(5, bool), 0.5, 1.0, True, None)
    assert i == 1 and not hom                                  # median-Ts candidate of 290, 291, 292


# ---------------------------------------------------------------- CASA Topt
def test_casa_topt_is_mean_temperature_of_the_ndvi_peak_month():
    import pandas as pd
    from core import model
    days = pd.date_range("2025-11-01", "2026-01-31")
    d = pd.DataFrame({"unit": "u1", "date": days, "tmean": np.where(days.month == 12, 17.0, 25.0)})
    seasons = pd.DataFrame({"peak_date": [pd.Timestamp("2025-12-20")]}, index=["u1"])
    t = model.casa_topt(d, seasons)
    assert t.topt_c.iloc[0] == pytest.approx(17.0) and t.days.iloc[0] == 31 and t.peak_month.iloc[0] == "2025-12"


def test_topt_of_uses_unit_topt_plus_offset():
    import pandas as pd
    from core import model
    d = pd.DataFrame({"topt_unit": [16.5, 18.0]})
    assert list(model.topt_of(d, 0.0)) == [16.5, 18.0]
    assert model.topt_of(pd.DataFrame({"x": [1]}), 19.5) == 19.5      # crop-value Topt


def test_casa_mode_makes_topt_an_offset():
    cfg = config.load("dewas_2025_26")
    assert cfg["temperature_scalar"]["topt"] == "casa_ndvi_peak_month"
    assert cfg["parameters"]["topt"]["values"] == [0.0] and cfg["parameters"]["topt"]["range"] == [0.0, 0.0]


# ---------------------------------------------------------------- 00d: accumulated ERA5-Land values
def test_accumulated_sum_equals_sum_of_hourly_amounts():
    """ERA5-Land accumulates from 00 UTC: the value at 01 UTC is the first hour, the value at 00 UTC closes the day.
    The sum over any 24-hour window must equal the sum of the hourly amounts (second method of step 00d)."""
    import pandas as pd
    from core import stac_verify as sv
    times = pd.date_range("2025-12-01 00:00", "2025-12-04 00:00", freq="h")
    rng_ = np.random.default_rng(1)
    amount = pd.Series(rng_.uniform(0, 1e-3, len(times)), index=times)       # hourly amount ending at t
    day_of = (times - pd.Timedelta(hours=1)).floor("D")                     # accumulation day (00 UTC reset)
    acc = amount.groupby(day_of).cumsum()
    day = pd.Timestamp("2025-12-02")                                         # local day, UTC + 5.5 h
    t_a = (day - pd.Timedelta(hours=5.5) + pd.Timedelta(minutes=30)).ceil("h")
    t_s, t_b = t_a - pd.Timedelta(hours=1), t_a + pd.Timedelta(hours=23)
    got = sv.accumulated_sum(lambda t: float(acc[t]), t_s, t_b)
    assert got == pytest.approx(float(amount[t_a:t_b].sum()))
    assert sv.accumulated_sum(lambda t: float(acc[t]), pd.Timestamp("2025-12-02 00:00"),
                              pd.Timestamp("2025-12-03 00:00")) == pytest.approx(float(acc[pd.Timestamp("2025-12-03")]))


# ---------------------------------------------------------------- lead_days per run (optional)
def test_lead_days_default_from_project_and_per_run_override(tmp_path, monkeypatch):
    import shutil
    cfg = config.load("dewas_2025_26")                          # no lead_days in the run config
    assert cfg["season"]["lead_days"] == config.read_yaml(config.ROOT / "config" / "project.yaml")["season"]["lead_days"]
    assert cfg["season"]["lead_days_source"] == "config/project.yaml"
    for d in ("config/aoi", "config/crops"):
        (tmp_path / d).mkdir(parents=True)
    shutil.copy(config.ROOT / "config" / "project.yaml", tmp_path / "config")
    shutil.copy(config.ROOT / "config" / "crops" / "potato.yaml", tmp_path / "config" / "crops")
    run = yaml.safe_load(open(config.ROOT / "config" / "aoi" / "dewas_2025_26.yaml", encoding="utf-8"))
    run["season_dates"]["lead_days"] = 91
    (tmp_path / "config" / "aoi" / "x_2025_26.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    run["season_dates"]["lead_days"] = -5
    (tmp_path / "config" / "aoi" / "bad_2025_26.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    c = config.load("x_2025_26")
    assert c["season"]["lead_days"] == 91 and c["season"]["lead_days_source"] == "config/aoi/x_2025_26.yaml"
    import pandas as pd
    start, _ = stac_ingest.export_period(c)                     # download period starts 91 days before the window
    assert start == pd.Timestamp(c["season"]["window"][0]) - pd.Timedelta(days=91)
    with pytest.raises(SystemExit, match="lead_days"):
        config.load("bad_2025_26")
