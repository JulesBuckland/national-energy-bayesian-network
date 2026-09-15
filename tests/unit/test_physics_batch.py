import pytest
import pandas as pd
import numpy as np
import math
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.physics.energyplus_batch import (
    _bc, calculate_occupants, build_idf_string, generate_task_from_row, generate_tasks_from_df,
    _parse_heating_kwh, run_single, run_lhs_batch, _run_single_axis, read_template_file, write_idf_file,
    check_placeholder_weather, _validate_weather_files
)

def test_bc():
    assert _bc(4, 1) == ("Outdoors", "SunExposed", "WindExposed")
    assert _bc(2, 3) == ("Adiabatic", "NoSun", "NoWind")

def test_calculate_occupants():
    occ1 = calculate_occupants(10.0)
    assert occ1 == 1.0
    occ2 = calculate_occupants(100.0)
    assert occ2 > 1.0

def test_build_idf_string():
    template = "Arch: @@ARCHETYPE_NAME@@, Occ: @@PEOPLE_COUNT@@, Axis: @@NORTH_AXIS@@"
    res = build_idf_string(template, "TestRun", 100.0, 2.1, 1.5, 0.15, 2, 4, 90.0, 2.5, 200.0, 300.0)
    assert "Arch: TestRun" in res
    assert "Occ: 2.50" in res
    assert "Axis: 90.0" in res

def test_generate_tasks():
    df = pd.DataFrame({
        "archetype": ["Test Arc"],
        "property_type": ["House"],
        "age_band": ["Pre-1900"],
        "floor_area": [100.0],
        "wall_u": [2.1],
        "ach": [1.5],
        "wwr": [0.15],
        "form_code": [1],
        "floors": [2],
        "exposed_walls": [4],
        "sample_id": [1],
        "city": ["Manchester"],
        "hdd": [2500.0]
    })
    tasks = generate_tasks_from_df(df, False)
    assert len(tasks) == 1
    t = tasks[0]
    assert t["run_id"] == "Test_Arc_s0001"
    assert t["city"] == "Manchester"
    assert t["resume"] is False

def test_generate_task_from_row_no_city():
    row = pd.Series({
        "archetype": "Test Arc",
        "property_type": "House",
        "age_band": "Pre-1900",
        "floor_area": 100.0,
        "wall_u": 2.1,
        "ach": 1.5,
        "wwr": 0.15,
        "form_code": 1,
        "floors": 2,
        "exposed_walls": 4,
        "sample_id": 1,
    })
    t = generate_task_from_row(row, True)
    assert "city" not in t
    assert t["resume"] is True

def test_parse_heating_kwh_success(tmp_path):
    html_content = """
    <html><body>
    <b>End Uses</b>
    <table>
    <tr><td>Heating</td><td>10.0</td><td>20.0</td></tr>
    </table>
    </body></html>
    """
    html_file = tmp_path / "eplustbl.htm"
    html_file.write_text(html_content)
    
    val = _parse_heating_kwh(tmp_path)
    assert np.isclose(val, 30.0 * 277.778)

def test_parse_heating_kwh_no_file(tmp_path):
    with pytest.raises(RuntimeError):
        _parse_heating_kwh(tmp_path)

def test_parse_heating_kwh_no_table(tmp_path):
    html_file = tmp_path / "eplustbl.htm"
    html_file.write_text("<html></html>")
    with pytest.raises(RuntimeError):
        _parse_heating_kwh(tmp_path)

def test_parse_heating_kwh_no_heating_row(tmp_path):
    html_content = """
    <html><body>
    <table>
    <tr><td>Cooling</td><td>10.0</td></tr>
    </table>
    </body></html>
    """
    html_file = tmp_path / "eplustbl.htm"
    html_file.write_text(html_content)
    with pytest.raises(RuntimeError):
        _parse_heating_kwh(tmp_path)

def test_read_write_idf(tmp_path):
    f = tmp_path / "test.idf"
    write_idf_file(f, "idf_content")
    assert read_template_file(f) == "idf_content"

def test_check_placeholder_weather(tmp_path):
    wf = tmp_path / "test.epw"
    wf.write_bytes(b'a' * 1546562)
    assert check_placeholder_weather(tmp_path) is True
    wf.unlink()
    wf.write_bytes(b'a' * 100)
    assert check_placeholder_weather(tmp_path) is False


def test_validate_weather_files_rejects_placeholder(tmp_path):
    wf = tmp_path / "Manchester_2030_ColdSnap.epw"
    wf.write_bytes(b'a' * 1546562)

    with pytest.raises(ValueError, match="Placeholder weather files"):
        _validate_weather_files(tmp_path, {"Manchester"})


def test_validate_weather_files_rejects_missing_requested_city(tmp_path):
    with pytest.raises(FileNotFoundError, match="London"):
        _validate_weather_files(tmp_path, {"London"})

def test_run_single_axis(monkeypatch):
    args = {
        "floor_area": 100.0,
        "wall_u": 2.1,
        "ach": 1.5,
        "wwr": 0.15,
        "floors": 1,
        "exposed_walls": 4,
    }
    monkeypatch.setattr("src.physics.energyplus_batch.write_idf_file", lambda p, s: None)
    monkeypatch.setattr("subprocess.run", MagicMock())
    monkeypatch.setattr("src.physics.energyplus_batch._parse_heating_kwh", lambda p: 100.0)
    monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)
    
    val = _run_single_axis(0, "test", args, "tmpl", Path("wf"), Path("od"))
    assert val == 100.0

def test_run_single_axis_exception(monkeypatch):
    args = {
        "floor_area": 100.0,
        "wall_u": 2.1,
        "ach": 1.5,
        "wwr": 0.15,
        "floors": 1,
        "exposed_walls": 4,
    }
    monkeypatch.setattr("src.physics.energyplus_batch.write_idf_file", lambda p, s: None)
    def raiser(*a, **kw): raise Exception("fail")
    monkeypatch.setattr("subprocess.run", raiser)
    monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)
    
    val = _run_single_axis(0, "test", args, "tmpl", Path("wf"), Path("od"))
    assert val == 0.0

def test_run_single(monkeypatch):
    args = {
        "run_id": "test",
        "floor_area": 100.0,
        "wall_u": 2.1,
        "ach": 1.5,
        "wwr": 0.15,
        "floors": 1,
        "exposed_walls": 4,
        "city": "Manchester",
        "resume": False
    }
    monkeypatch.setattr("src.physics.energyplus_batch.read_template_file", lambda p: "template")
    monkeypatch.setattr("src.physics.energyplus_batch._run_single_axis", lambda a, r, ar, ts, wf, od: 100.0)
    
    mock_open = MagicMock()
    monkeypatch.setattr("builtins.open", mock_open)
    
    res = run_single(args)
    assert res["status"] == "ok"
    assert res["T_h"] == 100.0

def test_run_single_resume(monkeypatch, tmp_path):
    args = {
        "run_id": "test",
        "resume": True
    }
    out_dir = tmp_path / "test"
    out_dir.mkdir()
    res_file = out_dir / "result.json"
    res_file.write_text('{"status": "ok", "T_h": 50.0}')
    
    monkeypatch.setattr("src.physics.energyplus_batch.SIM_DIR", tmp_path)
    res = run_single(args)
    assert res["T_h"] == 50.0

def test_run_lhs_batch_no_files(monkeypatch):
    monkeypatch.setattr("src.physics.energyplus_batch.LHS_DIR", MagicMock(glob=lambda p: []))
    run_lhs_batch()

def test_run_lhs_batch_no_exe(monkeypatch):
    mock_glob = [MagicMock()]
    monkeypatch.setattr("src.physics.energyplus_batch.LHS_DIR", MagicMock(glob=lambda p: mock_glob))
    monkeypatch.setattr("src.physics.energyplus_batch.EP_EXE", MagicMock(exists=lambda: False))
    run_lhs_batch()

def test_run_lhs_batch_check_completeness(monkeypatch):
    monkeypatch.setattr("src.physics.energyplus_batch.EP_EXE", MagicMock(exists=lambda: True))
    monkeypatch.setattr("src.physics.energyplus_batch.check_placeholder_weather", lambda p: False)
    
    mock_glob = [MagicMock()]
    monkeypatch.setattr("src.physics.energyplus_batch.LHS_DIR", MagicMock(glob=lambda p: mock_glob))
    
    monkeypatch.setattr(pd, "read_csv", lambda p: pd.DataFrame({
        "archetype": ["Test Arc"],
        "property_type": ["House"],
        "age_band": ["Pre-1900"],
        "floor_area": [100.0],
        "wall_u": [2.1],
        "ach": [1.5],
        "wwr": [0.15],
        "form_code": [1],
        "floors": [2],
        "exposed_walls": [4],
        "sample_id": [1],
    }))
    
    # completeness branch returns early
    run_lhs_batch(check_completeness=True)

def test_run_lhs_batch(monkeypatch):
    monkeypatch.setattr("src.physics.energyplus_batch.EP_EXE", MagicMock(exists=lambda: True))
    monkeypatch.setattr("src.physics.energyplus_batch._validate_weather_files", lambda p, c: None)
    
    mock_glob = [MagicMock()]
    monkeypatch.setattr("src.physics.energyplus_batch.LHS_DIR", MagicMock(glob=lambda p: mock_glob))
    
    monkeypatch.setattr(pd, "read_csv", lambda p: pd.DataFrame({
        "archetype": ["Test Arc"],
        "property_type": ["House"],
        "age_band": ["Pre-1900"],
        "floor_area": [100.0],
        "wall_u": [2.1],
        "ach": [1.5],
        "wwr": [0.15],
        "form_code": [1],
        "floors": [2],
        "exposed_walls": [4],
        "sample_id": [1],
    }))
    
    class DummyFuture:
        def exception(self): return None
        def result(self): return {"status": "ok", "archetype": "Test Arc"}
    
    class DummyExecutor:
        def __init__(self, max_workers): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def submit(self, fn, task): return DummyFuture()
        
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr("concurrent.futures.as_completed", lambda fs: fs)
    
    mock_to_csv = MagicMock()
    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)
    
    run_lhs_batch()
    mock_to_csv.assert_called()
