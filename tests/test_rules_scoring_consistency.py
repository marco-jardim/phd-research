"""A regra Modal deve ser a regra canônica, sem importar Modal nem treinar."""

import numpy as np
import pandas as pd
import pytest

from scripts.ablation_study import regras_alta_confianca
from scripts import qw1_modal_recompute_local as local
from scripts.generate_ablation_tables import generate, config_tex, pt


def modal_rule():
    return local.load_functions(local.PIPELINE.read_text(encoding="utf-8"),
                                ["regras_alta_confianca"])["regras_alta_confianca"]


@pytest.mark.parametrize("row,expected", [
    ({}, 0),
    ({"NOME qtd frag iguais": 0.95}, 3),
    ({"NOME qtd frag iguais": 0.85}, 2),
    ({"NOME qtd frag iguais": 0.849}, 0),
    ({"DTNASC dt iguais": 1, "DTNASC dt ap 1digi": 1}, 3),
    ({"DTNASC dt ap 1digi": 1}, 1.5),
    ({"NOMEMAE qtd frag iguais": 0.7}, 2),
    ({"NOMEMAE qtd frag iguais": 0.5}, 1),
    ({"CODMUNRES local igual": 1}, 1.5),
    ({"ENDERECO via igual": 1}, 1),
    ({"nota final": 9}, 2),
    ({"nota final": 8}, 1),
    ({"NOME qtd frag iguais": 1, "DTNASC dt iguais": 1,
      "NOMEMAE qtd frag iguais": 1, "CODMUNRES local igual": 1,
      "ENDERECO via igual": 1, "nota final": 9}, 12.5),
])
def test_limites_e_maximo(row, expected):
    assert regras_alta_confianca(pd.Series(row)) == expected
    assert modal_rule()(pd.Series(row)) == expected


def test_validacao_exata_rejeita_diferenca_minima():
    before = pd.DataFrame({"f1": [0.5]})
    with pytest.raises(AssertionError):
        local.assert_exact(before, pd.DataFrame({"f1": [np.nextafter(0.5, 1)]}))


def test_modal_e_ablacao_em_mil_linhas_do_teste():
    # A comparação integral é também obrigatória no recomputador local.
    if not (local.ROOT / "data/COMPARADORSEMIDENT.csv").exists():
        pytest.skip("extrato privado não disponível neste checkout")
    X_test, scored = local.load_holdout()
    sample = X_test.sample(n=1000, random_state=42)
    np.testing.assert_array_equal(sample.apply(modal_rule(), axis=1),
                                  sample.apply(regras_alta_confianca, axis=1))
    np.testing.assert_array_equal(X_test.apply(local.legacy_rules, axis=1), scored.rules_score)


def test_tabelas_reproduziveis_e_top10_real(tmp_path):
    abl = pd.read_csv(local.ROOT / "data/ablation_results.csv")
    generate(abl, tmp_path)
    for name in ("tab_ablation_best_category.tex", "tab_ablation_top10.tex"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert text == (local.ROOT / "text/latex/tables" / name).read_text(encoding="utf-8")
        assert "—" not in text and "–" not in text
    top = (tmp_path / "tab_ablation_top10.tex").read_text(encoding="utf-8")
    for i, row in enumerate(abl.sort_values("f1", ascending=False, kind="stable").head(10).itertuples(), 1):
        assert f"{i} & {config_tex(row.config)} & {pt(row.precision)} & {pt(row.recall)} & {pt(row.f1)}" in top


def test_validacao_bloqueante_nao_escreve(monkeypatch, tmp_path):
    X = pd.DataFrame({"nota final": [9.0]})
    # A regra histórica atribui 2 pontos, não os 3 persistidos neste exemplo.
    monkeypatch.setattr(local, "load_holdout", lambda: (X, pd.DataFrame({"rules_score": [3.0]})))
    monkeypatch.setattr(local, "META", tmp_path / "meta.json")
    monkeypatch.setattr(local.sys, "argv", ["qw1_modal_recompute_local.py"])
    with pytest.raises(AssertionError):
        local.main()
    assert not local.META.exists()
