"""Testes do script scripts/exp1_testes_t.py (testes t da ablação MACD)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
CSV_ON = RAIZ / "data" / "ablation_comprec_macd.csv"


def _carregar():
    spec = importlib.util.spec_from_file_location("exp1_testes_t", RAIZ / "scripts" / "exp1_testes_t.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not CSV_ON.exists(), reason="artefato de ablação ausente")
def test_exp1_t_pareado_reproduz_valores_do_texto():
    df = _carregar().comparar()
    f1 = df[df["metrica"] == "exp_f1"].set_index("modo")
    assert f1.loc["vigilancia", "t_pareado"] == pytest.approx(2.36, abs=0.01)
    assert f1.loc["vigilancia", "p_pareado"] == pytest.approx(0.078, abs=0.001)
    assert f1.loc["confirmacao", "t_pareado"] == pytest.approx(1.57, abs=0.01)
    assert f1.loc["confirmacao", "p_pareado"] == pytest.approx(0.191, abs=0.001)
    assert (df["gl_pareado"] == 4).all()


def test_poder_amostral_piloto():
    spec = importlib.util.spec_from_file_location("poder", RAIZ / "scripts" / "poder_amostral_piloto.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t = mod.tabela()
    assert list(t["n_por_grupo"]) == [485, 1997]
