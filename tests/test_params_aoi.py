"""Step 00a with an existing run config: a pasted polygon path is used (same contents -> same run; other polygon or
window -> next free run id, the existing run config untouched). Synthetic files only."""
import argparse
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config, params  # noqa: E402

POLY = '{"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"EPSG:4326"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[76.0,22.5],[76.2,22.5],[76.2,22.7],[76.0,22.7],[76.0,22.5]]]}}]}'


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "config" / "aoi").mkdir(parents=True)
    (tmp_path / "config" / "crops").mkdir()
    shutil.copy(config.ROOT / "config" / "crops" / "potato.yaml", tmp_path / "config" / "crops")
    (tmp_path / "data" / "area").mkdir(parents=True)
    (tmp_path / "data" / "area" / "Area.geojson").write_text(POLY, encoding="utf-8")
    monkeypatch.setattr(params, "ROOT", tmp_path)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path


def args(aoi, run="area_2025_26", w0="2025-10-01", w1="2026-03-31"):
    return argparse.Namespace(aoi=str(aoi), name="Area", window_start=w0, window_end=w1, run=run, season="rabi_2025_26",
                              crop="potato", peak_start="2025-11-01", peak_end="2026-02-28",
                              quality_start="2025-12-01", quality_end="2026-01-31", utc_offset="5.5",
                              calibration="none")


def test_new_run_config_is_written(root):
    run, _ = params.collect(args(root / "data" / "area" / "Area.geojson"))
    assert run == "area_2025_26" and (root / "config" / "aoi" / "area_2025_26.yaml").exists()


def test_same_polygon_pasted_from_another_folder_runs_with_the_existing_config(root, tmp_path_factory):
    params.collect(args(root / "data" / "area" / "Area.geojson"))
    before = (root / "config" / "aoi" / "area_2025_26.yaml").read_text(encoding="utf-8")
    elsewhere = tmp_path_factory.mktemp("old_folder") / "Area copy.geojson"
    elsewhere.write_text(POLY, encoding="utf-8")                       # same contents, other path
    run, _ = params.collect(args(elsewhere))
    assert run == "area_2025_26"
    assert (root / "config" / "aoi" / "area_2025_26.yaml").read_text(encoding="utf-8") == before


def test_other_polygon_or_window_runs_under_the_next_free_id(root, tmp_path_factory):
    params.collect(args(root / "data" / "area" / "Area.geojson"))
    before = (root / "config" / "aoi" / "area_2025_26.yaml").read_text(encoding="utf-8")
    other = tmp_path_factory.mktemp("other") / "Other.geojson"
    other.write_text(POLY.replace("76.2", "76.3"), encoding="utf-8")
    run, how = params.collect(args(other))
    assert run == "area_2025_26_2" and "next free id" in how["run_id"]
    assert str(other.resolve().as_posix()) in (root / "config" / "aoi" / "area_2025_26_2.yaml").read_text(encoding="utf-8")
    run, _ = params.collect(args(root / "data" / "area" / "Area.geojson", w0="2025-09-15"))   # same polygon, other window
    assert run == "area_2025_26_3"
    assert (root / "config" / "aoi" / "area_2025_26.yaml").read_text(encoding="utf-8") == before
