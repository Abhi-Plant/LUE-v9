"""Tests of continuing after an interruption: atomic files, the complete-step test (manifest, configuration and input
fingerprints) and the incomplete marker of a step that starts."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config  # noqa: E402


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    cfgfile = tmp_path / "config" / "project.yaml"
    cfgfile.write_text("framework: {version: test}\na: 1\n", encoding="utf-8")
    monkeypatch.setitem(config.LOADED, "config_files", [cfgfile])
    inp = tmp_path / "input.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(inp, index=False)
    return tmp_path, {"run_id": "r"}, cfgfile, inp


def complete_step(root, cfg, inp):
    out = config.output_dir(cfg, "01_inputs")
    config.to_csv_atomic(pd.DataFrame({"y": [3]}), out / "table.csv", index=False)
    config.write_manifest(out, "01_inputs", {"input": inp}, {"table": out / "table.csv"})
    return out


def test_atomic_write_leaves_no_partial_file(tmp_path):
    p = tmp_path / "x.csv"
    with pytest.raises(RuntimeError):
        config.write_atomic(p, lambda q: (q.write_text("half"), (_ for _ in ()).throw(RuntimeError("cut"))))
    assert not p.exists()
    config.to_csv_atomic(pd.DataFrame({"a": [1]}), p, index=False)
    assert p.read_text().splitlines() == ["a", "1"] and not list(tmp_path.glob("*.part"))


def test_complete_step_is_recognised(run_root):
    root, cfg, _, inp = run_root
    complete_step(root, cfg, inp)
    assert config.step_complete(cfg, "01_inputs")[0]


def test_changed_input_or_config_or_missing_output_reruns(run_root):
    root, cfg, cfgfile, inp = run_root
    out = complete_step(root, cfg, inp)
    pd.DataFrame({"x": [1, 5]}).to_csv(inp, index=False)
    assert config.step_complete(cfg, "01_inputs") == (False, "input changed: input.csv")
    out = complete_step(root, cfg, inp)
    cfgfile.write_text("framework: {version: test}\na: 2\n", encoding="utf-8")
    assert config.step_complete(cfg, "01_inputs") == (False, "configuration changed")
    out = complete_step(root, cfg, inp)
    (out / "table.csv").unlink()
    assert not config.step_complete(cfg, "01_inputs")[0]


def test_step_that_starts_is_incomplete_until_it_ends(run_root):
    root, cfg, _, inp = run_root
    complete_step(root, cfg, inp)
    config.output_dir(cfg, "01_inputs")                   # the step starts again ... and is interrupted
    assert config.step_complete(cfg, "01_inputs") == (False, "not run or interrupted (no manifest)")
