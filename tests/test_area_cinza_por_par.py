from pathlib import Path

import pandas as pd
import pytest

from scripts.area_cinza_por_par import carregar, resumir
from scripts.sensibilidade_gzcmd_por_par import identificar
from scripts.sensibilidade_gzcmd_por_par import resumir as resumir_cascata


def extrato(tmp_path):
    path = tmp_path / "extrato.csv"
    path.write_text(
        "PAR;COMPREC,C,12,0;REFREC,C,12,0;PASSO;nota final\n"
        "1;a;r1;6;4,99\n1;b;r2;6;5\n2;c;r3;9;8,99\n"
        "2;d;r4;11;9\n0;e;r5;11;7\n", encoding="utf-8"
    )
    return path


def test_faixas_limites_e_par(tmp_path):
    table = resumir(carregar(extrato(tmp_path)))
    assert table.vinculos.tolist() == [1, 0, 1, 1, 0, 1]
    assert table.columns.tolist() == ["faixa", "PAR", "vinculos"]


def preparar(tmp_path):
    frame = identificar(carregar(extrato(tmp_path), identificadores=True))
    funnel = frame[["pair_id", "nota"]].copy()
    funnel["y"] = [1, 1, 1, 1, 0]
    funnel["final"] = ["NONMATCH", "MATCH", "DEFERRED", "MATCH", "MATCH"]
    metrics = pd.DataFrame([{
        "modo": "vigilancia", "versao": "v3.1", "VP_cadeia": 2,
        "FP_cadeia": 1, "FN_cadeia": 2, "VN_cadeia": 0,
        "sens_cadeia_observada": .5,
    }])
    return frame, funnel, metrics


def test_aceitacao_por_par_alinha_por_identificador(tmp_path):
    frame, funnel, metrics = preparar(tmp_path)
    result = resumir_cascata(frame, funnel.iloc[::-1], "vigilancia", metrics)
    assert result.aceitos_automaticamente.tolist() == [1, 1, 1, 0]
    assert result.vinculos.tolist() == [2, 2, 1, 1]
    assert "pair_id" not in result


@pytest.mark.parametrize("problem", ["duplicado", "ausente", "nota", "gabarito", "pendente", "metrica"])
def test_rejeita_inconsistencias(tmp_path, problem):
    frame, funnel, metrics = preparar(tmp_path)
    if problem == "duplicado":
        funnel = pd.concat([funnel, funnel.iloc[:1]])
    elif problem == "ausente":
        funnel = funnel.iloc[1:]
    elif problem == "nota":
        funnel.loc[0, "nota"] = 0
    elif problem == "gabarito":
        funnel.loc[0, "y"] = 0
    elif problem == "pendente":
        funnel.loc[0, "final"] = "SEM_RESPOSTA_LLM"
    else:
        metrics.loc[0, "VP_cadeia"] = 3
    with pytest.raises(ValueError):
        resumir_cascata(frame, funnel, "vigilancia", metrics)


@pytest.mark.parametrize("value", ["", "3", "texto"])
def test_rejeita_par_invalido(tmp_path, value):
    path = tmp_path / "invalido.csv"
    path.write_text(f"PAR;nota final\n{value};5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        carregar(path)


def test_extrato_local_quando_disponivel():
    path = Path(__file__).resolve().parents[1] / "data/COMPARADORSEMIDENT.csv"
    if not path.exists():
        pytest.skip("Extrato não versionado ausente")
    table = resumir(carregar(path))
    assert table.vinculos.tolist() == [2, 1, 123, 28, 86, 7]
