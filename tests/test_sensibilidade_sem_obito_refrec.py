from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd
import pytest

from scripts import sensibilidade_sem_obito_refrec as sens
from scripts import gerar_tabela_sensibilidade as tabela
from scripts.ablation_study import evaluate_configurations


def sintetico():
    df = pd.DataFrame({"REFREC": [f"R_{i}" for i in range(20) for _ in range(4)]})
    y = pd.Series([int(i % 4 == 0 and i < 40) for i in range(80)])
    X = pd.DataFrame(np.zeros((80, 56)), columns=[f"x{i}" for i in range(56)])
    X["obito_tb"] = 1
    X["obito_sinan"] = 1
    return df, X, y


def test_limpar_nan_recursivo_e_json_estrito():
    original = {"ausencias": [float("nan"), float("inf"), -float("inf"), pd.NA, pd.NaT],
                "aninhado": ({"float": np.float32(0.5), "int": np.int64(42)},
                             np.float64("nan"), np.bool_(True)),
                "preservado": ["óbito", None, False, 3, 0.25]}
    limpo = sens.limpar_nan(original)
    assert limpo == {"ausencias": [None] * 5,
                     "aninhado": [{"float": 0.5, "int": 42}, None, True],
                     "preservado": ["óbito", None, False, 3, 0.25]}
    assert type(limpo["aninhado"][0]["float"]) is float
    assert type(limpo["aninhado"][0]["int"]) is int
    assert json.loads(json.dumps(limpo, allow_nan=False)) == limpo


def test_remove_somente_obito_sem_mutar():
    _, X, _ = sintetico()
    reduzida = sens.remover_obito(X)
    assert X.shape[1] == 58 and reduzida.shape[1] == 56
    assert list(reduzida.columns) == list(X.columns[:-2])
    pd.testing.assert_frame_equal(reduzida, X.drop(columns=sens.OBITO))


def test_cenarios_indices_pareados_e_grupos_disjuntos():
    df, X, y = sintetico()
    cenarios = sens.montar_cenarios(df, X, y)
    assert list(cenarios) == ["A", "B", "C", "D"]
    assert [c[1].shape[1] for c in cenarios.values()] == [58, 56, 58, 56]
    for a, b in (("A", "B"), ("C", "D")):
        for pos in (2, 3):
            assert cenarios[a][pos].equals(cenarios[b][pos])
    for nome in ("C", "D"):
        _, _, tr, te = cenarios[nome]
        assert set(df.loc[tr].REFREC).isdisjoint(df.loc[te].REFREC)
        assert len(tr) + len(te) == len(df)
    repeticao = sens.montar_cenarios(df, X, y)
    assert repeticao["A"][3].equals(cenarios["A"][3])
    assert repeticao["C"][3].equals(cenarios["C"][3])


def test_impacto_distingue_recuperados_intersecao_e_liquido():
    r = sens.impacto([1, 1, 1, 1, 0], [0.8, 0.9, 0.1, 0.7, 0.9], [9, 6, 8, 7, 9])
    assert r == dict(recuperados=2, detectados_ambos=1, detectados_nota=2,
                     detectados_ml=3, perdidos_vs_nota=1, liquido=1, incremento_pct=50)
    assert sens.impacto([1], [0.5], [7])["incremento_pct"] is None


def test_carrega_strings_e_converte_numericos(tmp_path):
    p = tmp_path / "extrato.csv"
    p.write_text('"REFREC,C,12,0";PAR;nota final;C_SITUENCE\nR_001;1;8,5;3\n', encoding="utf-8")
    df = sens.carregar(p)
    assert df.REFREC.tolist() == ["R_001"]
    assert df.PAR.tolist() == [1]
    assert df["nota final"].tolist() == [8.5]
    assert df.C_SITUENCE.tolist() == [3]


def test_grade_original_tem_70_configuracoes_e_nove_categorias():
    p = {"RF": np.array([0.1, 0.8]), "RF+SMOTE": np.array([0.2, 0.9]),
         "GB": np.array([0.3, 0.7])}
    res = evaluate_configurations(np.array([0, 1]), np.array([2, 9]), np.array([0, 8]), p)
    assert len(res) == 70 and res.config.is_unique
    assert res.category.nunique() == 9
    assert ((res.tp + res.fn) == 1).all()
    assert ((res.fp + res.tn) == 1).all()


def test_parametros_xgb_iguais_ao_modal_exceto_dispositivo():
    arvore = ast.parse((sens.ROOT / "scripts/qw1_modal_unified_pipeline.py").read_text(encoding="utf-8"))
    funcao = next(n for n in arvore.body if isinstance(n, ast.FunctionDef) and n.name == "train_xgboost")
    chamada = next(n for n in ast.walk(funcao) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == "XGBClassifier")
    params = {k.arg: ast.literal_eval(k.value) for k in chamada.keywords}
    params["device"] = "cpu"
    assert params == sens.XGB_PARAMS


def test_artefatos_agregados_controle_e_tabela(tmp_path):
    res = pd.read_csv(sens.OUT)
    meta = json.loads(sens.META.read_text(encoding="utf-8"),
                      parse_constant=lambda valor: pytest.fail(f"JSON nao estrito: {valor}"))
    assert res.cenario.tolist() == ["A", "B", "C", "D"]
    assert res.n_atributos.tolist() == [58, 56, 58, 56]
    assert not {"pair_id", "REFREC", "COMPREC", "PAR"}.intersection(res.columns)
    assert meta["csv_sha256"] == sens.sha256(sens.OUT)
    for fonte, esperado in meta["fontes_sha256"].items():
        assert sens.sha256(sens.ROOT / fonte) == esperado
    assert meta["semente"] == 42
    assert meta["tempo_total_segundos"] > 0
    for nome in ("C", "D"):
        assert res.set_index("cenario").loc[nome, "n_REFREC_sobrepostos"] == 0
    sens.conferir_controle(pd.DataFrame(meta["cenarios"]["A"]["configuracoes_70"]), res.iloc[0])
    for nome in res.cenario:
        assert len(meta["cenarios"][nome]["configuracoes_70"]) == 70
        assert len(meta["cenarios"][nome]["melhor_por_categoria"]) == 9
    tex = tmp_path / "tabela.tex"
    tabela.gerar(res, tex)
    texto = tex.read_text(encoding="utf-8")
    assert texto == tabela.TEX.read_text(encoding="utf-8")
    assert r"\label{tab:sensibilidade-2x2}" in texto
    assert "0,931" in texto and "18.509" in texto
    assert not any(t in texto for t in ("—", "–", "---"))


@pytest.mark.skipif(not sens.EXTRATO.exists(), reason="extrato local ausente")
def test_csv_gerado_alinhado_ao_extrato_e_70_controles():
    df = sens.carregar(sens.EXTRATO)
    X, y = sens.engineer_features(df)
    cenarios = sens.montar_cenarios(df, X, y)
    res = pd.read_csv(sens.OUT).set_index("cenario")
    meta = json.loads(sens.META.read_text(encoding="utf-8"))
    for nome, (_, matriz, tr, te) in cenarios.items():
        r = res.loc[nome]
        assert (r.n_treino, r.n_teste, r.vinculos_teste, r.n_atributos) == (
            len(tr), len(te), int(y.loc[te].sum()), matriz.shape[1])
    assert meta["entrada_sha256"] == sens.sha256(sens.EXTRATO)
    sens.conferir_controle(pd.DataFrame(meta["cenarios"]["A"]["configuracoes_70"]), res.loc["A"])
