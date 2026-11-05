import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts.tabelas_s8_editorial import same_load

ROOT = Path(__file__).resolve().parents[1]


def table_cells(text, prefix):
    return next(line for line in text.splitlines() if line.startswith(prefix)).split(" & ")


def test_materialized_count_and_total_sensitivity_are_independent_columns():
    data = pd.read_csv(ROOT / "data/comparador_mesma_carga.csv")
    projections = pd.read_csv(ROOT / "data/comparador_mesma_carga_erro05.csv")
    scenarios = pd.read_csv(ROOT / "data/cenarios_sensibilidade_humana.csv")
    prefix = "Vigilância & O1 & "
    original = table_cells(same_load(data, projections, scenarios=scenarios), prefix)
    assert len(original) == 9
    mask = projections.modo.eq("vigilancia") & projections.ordenacao.eq("O1_SCORE") & projections.carga.eq("K1_humana")
    projections.loc[mask, "par2_topk_erro05"] = 13.37
    altered = table_cells(same_load(data, projections, scenarios=scenarios), prefix)
    assert altered[4] == "13{,}37"
    assert altered[5] == original[5]
    selected = data.modo.eq("vigilancia") & data.ordenacao.eq("O1_SCORE") & data.carga.eq("K1_humana")
    data.loc[selected, "sens_humano_05"] = .1234
    sensitivity_changed = table_cells(same_load(data, projections, scenarios=scenarios), prefix)
    assert sensitivity_changed[4] == "13{,}37"
    assert sensitivity_changed[5] == "12{,}34"
    p_mask = scenarios.modo.eq("vigilancia") & scenarios.estrategia.eq("P") & scenarios.s.eq(.95)
    scenarios.loc[p_mask, "sensibilidade_total"] = .2345
    p_row = table_cells(same_load(data, projections, scenarios=scenarios), "Vigilância & P & ")
    assert p_row[5] == "23{,}45"


@pytest.mark.parametrize("missing", ["COMPARADORSEMIDENT.csv", "pcal_ml_oof.csv", "pcal_ml_oof.json"])
def test_private_tests_skip_when_any_required_input_is_missing(tmp_path, missing):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("COMPARADORSEMIDENT.csv", "pcal_ml_oof.csv", "pcal_ml_oof.json"):
        if name != missing:
            (data / name).write_text("synthetic placeholder", encoding="utf-8")
    env = os.environ.copy()
    env["PHD_RESEARCH_DATA_ROOT"] = str(tmp_path)
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs",
                             "tests/test_subamostragem.py::test_private_reference_reproduces_published_oof",
                             "tests/test_subamostragem.py::test_private_regeneration_is_identical"],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 skipped" in result.stdout
    assert missing in result.stdout
    assert "FileNotFoundError" not in result.stdout + result.stderr
