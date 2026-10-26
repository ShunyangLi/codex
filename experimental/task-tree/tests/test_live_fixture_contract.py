import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def test_employment_checker_uses_available_backend_and_rejects_wrong_result(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    module_spec = importlib.util.spec_from_file_location(
        "live_fixture", Path(__file__).with_name("real_smoke.py")
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE positions(id INTEGER, country VARCHAR, role_k1500_v2 VARCHAR, role_k17000_v3 VARCHAR)"
    )
    connection.executemany(
        "INSERT INTO positions VALUES (?,?,?,?)",
        [
            (1, "United States", "Artificial Intelligence Engineer", None),
            (2, "United States", "cashier", "machine learning intern"),
            (3, "United States", None, None),
            (4, "Canada", "machine learning", None),
            (5, "United States", "cashier", "Machine Learning Intern"),
        ],
    )
    connection.execute("COPY positions TO ? (FORMAT PARQUET)", [str(tmp_path / "position.parquet")])
    result = {
        "us_positions": 4,
        "occupation_counts": {
            "Artificial Intelligence Engineer": 1,
            "cashier": 2,
            "__MISSING__": 1,
        },
        "strict_ai_count": 3,
        "strict_ai_ids": ["1", "2", "5"],
    }
    (tmp_path / "employment_summary.json").write_text(json.dumps(result))
    (tmp_path / "analysis.py").write_text(
        'import json\ndef collect():\n    return json.load(open("employment_summary.json"))\n'
    )
    spec = module.employment_spec(tmp_path)
    for criterion in spec.acceptance:
        response = subprocess.run(
            criterion.check_spec.command, cwd=tmp_path, capture_output=True, text=True
        )
        assert response.returncode == 0, response.stderr
    result["strict_ai_count"] = 4
    (tmp_path / "employment_summary.json").write_text(json.dumps(result))
    response = subprocess.run(
        spec.acceptance[1].check_spec.command, cwd=tmp_path, capture_output=True, text=True
    )
    assert response.returncode != 0
