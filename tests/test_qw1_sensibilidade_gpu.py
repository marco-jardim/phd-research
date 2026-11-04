from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="dependencia opcional do experimento QW1")
import torch
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

from scripts import qw1_sensibilidade_gpu as sens


def sintetico():
    df = pd.DataFrame({"REFREC": [f"R{i}" for i in range(20) for _ in range(4)]})
    y = pd.Series([int(i % 4 == 0 and i < 40) for i in range(80)])
    X = pd.DataFrame(np.zeros((80, 56)), columns=[f"x{i}" for i in range(56)])
    X["obito_tb"] = 1
    X["obito_sinan"] = 1
    return df, X, y


def test_cenarios_reutilizam_particoes_e_removem_somente_obito():
    df, X, y = sintetico()
    cenarios = sens.montar_cenarios(df, X, y)
    assert [c[1].shape[1] for c in cenarios.values()] == [58, 56, 58, 56]
    for a, b in (("A", "B"), ("C", "D")):
        assert cenarios[a][2].equals(cenarios[b][2])
        assert cenarios[a][3].equals(cenarios[b][3])
        pd.testing.assert_frame_equal(
            cenarios[b][1], sens.remover_obito(cenarios[a][1])
        )
    for c in ("C", "D"):
        _, _, tr, te = cenarios[c]
        assert set(df.loc[tr, "REFREC"]).isdisjoint(df.loc[te, "REFREC"])


def test_agregacao_limiar_inclusivo_e_contagens():
    r = sens.agregar([1, 1, 0, 0], [0.5, 0.1, 0.6, 0.2])
    assert (r["vp"], r["fp"], r["fn"], r["vn"]) == (1, 1, 1, 1)
    assert r["f1"] == r["precisao"] == r["sensibilidade"] == 0.5
    assert r["n_teste"] == 4 and r["vinculos_teste"] == 2
    assert sens.agregar([0, 0], [0.1, 0.2])["auc"] is None
    with pytest.raises(ValueError, match="invalidas"):
        sens.agregar([1], [float("nan")])


def test_tabela_reproduz_numeros_sem_identificadores(tmp_path):
    rows = [
        dict(
            modelo=m,
            cenario=c,
            f1=0.875,
            precisao=0.9,
            sensibilidade=0.8,
            n_teste=18304,
            vinculos_teste=72,
            vp=63,
            fp=9,
        )
        for m in sens.MODELOS
        for c in "ABCD"
    ]
    meta = dict(
        gpu={"nome": "GPU de teste"},
        versoes={
            p: "1.0" for p in ("torch", "scikit-learn", "pytorch-tabnet", "xgboost")
        },
    )
    path = tmp_path / "tabela.tex"
    sens.gerar_tabela(pd.DataFrame(rows), meta, path)
    tex = path.read_text(encoding="utf-8")
    assert r"\label{tab:sensibilidade-qw1}" in tex
    assert tex.count("0,875") == 20 and tex.count("63/9") == 20
    assert "18.304" in tex and "18.509" in tex
    assert tex.count("MLP (128,64) &") == 1
    assert not any(t in tex for t in ("\u2014", "\u2013", "-" * 3))
    assert "pair_id" not in tex and b"\r\n" not in path.read_bytes()
    with pytest.raises(ValueError, match="quatro cenarios"):
        sens.gerar_tabela(pd.DataFrame(rows[:-1]), meta, path)


def test_controle_registra_divergencia_sem_forcar_resultado():
    res = pd.DataFrame([dict(modelo=m, cenario="A", f1=0.8) for m in sens.MODELOS])
    ref = pd.DataFrame([dict(model=m, f1=0.9) for m in sens.MODELOS])
    controles = sens.comparar_controle(res, ref)
    assert all(not r["dentro_tolerancia"] for r in controles.values())
    assert all(r["tolerancia"] == 0.01 for r in controles.values())


def test_cv_usa_somente_treino_e_preserva_grupos_REFREC():
    df, X, y = sintetico()
    cenarios = sens.montar_cenarios(df, X, y)
    for cenario in ("A", "D"):
        _, _, tr, te = cenarios[cenario]
        folds = list(sens.folds_treino(df, y, tr, cenario == "D"))
        assert [f for f, _, _ in folds] == list(range(1, 26))
        for _, a, b in folds:
            assert set(a).isdisjoint(b)
            assert set(a).union(b) == set(tr)
            assert set(a).union(b).isdisjoint(te)
            if cenario == "D":
                assert set(df.loc[a, "REFREC"]).isdisjoint(df.loc[b, "REFREC"])


def test_parametros_MLP_e_arvores_preservados():
    factories = sens._build_model_factories(58)
    for nome in ("RandomForest", "MLP (128,64)", "MLP (64,32)", "XGBoost"):
        original = factories[nome]().named_steps["classifier"].get_params()
        novo = sens.construir_modelo(nome, 58).named_steps["classifier"].get_params()
        if nome == "XGBoost":
            original.update(device="cuda", tree_method="hist")
        assert novo == original


@pytest.mark.skipif(
    not (sens.ROOT / "data/COMPARADORSEMIDENT.csv").exists(),
    reason="extrato local ausente",
)
def test_matriz_58_e_alvo_iguais_ao_carregador_historico():
    from scripts.feature_engineering import load_data

    path = sens.ROOT / "data/COMPARADORSEMIDENT.csv"
    X, y = sens.engineer_features(sens.carregar(path))
    historica, alvo = sens.engineer_features(load_data(path))
    pd.testing.assert_frame_equal(X.fillna(0), historica.fillna(0), check_dtype=False)
    pd.testing.assert_series_equal(y, alvo)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requer CUDA local")
def test_xgboost_nao_recua_para_CPU():
    X, y = make_classification(n_samples=120, n_features=8, random_state=42)
    model = sens.construir_modelo("XGBoost", 8)
    model.set_params(classifier__n_estimators=2)
    model.fit(X, y)
    cfg = json.loads(model.named_steps["classifier"].get_booster().save_config())
    assert cfg["learner"]["generic_param"]["device"].startswith("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requer CUDA local")
def test_mlp_cuda_equivalente_ao_sklearn_cpu():
    X, y = make_classification(n_samples=120, n_features=8, random_state=42)
    params = dict(
        hidden_layer_sizes=(8, 4),
        max_iter=8,
        random_state=42,
        batch_size=20,
        early_stopping=True,
    )
    cpu = MLPClassifier(**params).fit(X, y)
    gpu = sens.CudaMLPClassifier(**params).fit(X, y)
    np.testing.assert_allclose(cpu.predict_proba(X), gpu.predict_proba(X), atol=1e-10)
    for a, b in zip(cpu.coefs_, gpu.coefs_, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-10)


@pytest.mark.skipif(
    not (sens.OUT / "holdout.csv").exists(), reason="artefato local ausente"
)
def test_artefatos_agregados_totais_e_tabela(tmp_path):
    res = pd.read_csv(sens.OUT / "holdout.csv")
    meta = json.loads((sens.OUT / "_meta.json").read_text(encoding="utf-8"))
    assert len(res) == 20 and not res.duplicated(["cenario", "modelo"]).any()
    assert not {"pair_id", "REFREC", "COMPREC", "PAR"}.intersection(res.columns)
    for c in "ABCD":
        r = res.loc[res.cenario == c]
        assert set(r.modelo) == set(sens.MODELOS)
        assert (r.n_atributos == (58 if c in "AC" else 56)).all()
        assert (r.n_teste == (18509 if c in "AB" else 18304)).all()
        assert (r.vinculos_teste == (74 if c in "AB" else 72)).all()
        assert (r.vp + r.fn == r.vinculos_teste).all()
        assert (r.vp + r.fp + r.fn + r.vn == r.n_teste).all()
        assert (r.n_treino + r.n_teste == 61696).all()
        assert (r.limiar == 0.5).all() and (r.tempo_segundos > 0).all()
        if c in "CD":
            assert (r.n_REFREC_sobrepostos == 0).all()
    assert meta["csv_sha256"] == sens.sha256(sens.OUT / "holdout.csv")
    controle = pd.read_csv(sens.OUT / "controle_particao_original.csv")
    assert len(controle) == 5 and set(controle.modelo) == set(sens.MODELOS)
    assert (controle.n_teste == 18509).all()
    assert (controle.vinculos_teste == 74).all()
    assert (controle.vp + controle.fp + controle.fn + controle.vn == 18509).all()
    assert meta["controle_csv_sha256"] == sens.sha256(
        sens.OUT / "controle_particao_original.csv"
    )
    for path in sens.OUT.glob("*.csv"):
        dados = pd.read_csv(path)
        assert not {"pair_id", "REFREC", "COMPREC", "PAR"}.intersection(dados.columns)
        assert len(dados) <= 500  # somente modelo/cenario/fold, nunca um par por linha
    assert meta["gpu"]["nome"] and meta["versoes"]
    assert meta["semente"] == 42 and meta["limiar"] == 0.5
    assert meta["tempo_treinos_segundos"] == pytest.approx(
        res.tempo_segundos.sum() + controle.tempo_segundos.sum()
    )
    path = tmp_path / "tabela.tex"
    sens.gerar_tabela(res, meta, path)
    assert path.read_bytes() == sens.TEX.read_bytes()
