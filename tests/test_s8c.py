import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from local_data import DATA_ROOT, requires_files, requires_reviewer_sources
from scripts.esforco_arranjo import generate as effort
from scripts.fp_arranjo_por_regiao import aggregate, generate as fp_regions
from scripts.tabelas_arranjo_editorial import replace_once
from scripts.tabelas_s8c_editorial import generate as editorial

ROOT = Path(__file__).resolve().parents[1]


def test_effort_regenerates_exactly_and_separates_human_from_LLM_budget(tmp_path):
    target = tmp_path / "effort.csv"
    table = effort(output=target)
    assert target.read_bytes() == (ROOT / "data/esforco_arranjo.csv").read_bytes()
    assert len(table) == 8
    counts = pd.read_csv(ROOT / "data/carga_abaixo_corte.csv")
    reference = counts.loc[counts.limiar.eq("por_passo") & counts.passo.eq("TOTAL"), "pares_acima_limiar"].item()
    minutes = pd.read_csv(ROOT / "data/carga_abaixo_corte_horas.csv").set_index("limiar").loc["por_passo", "minutos_por_par"]
    same_load = pd.read_csv(ROOT / "data/comparador_mesma_carga.csv")
    for row in table.itertuples():
        assert row.cenario and row.exploratorio_post_hoc and not row.horas_medidas
        assert row.horas == pytest.approx(row.revisoes * minutes / 60)
        assert row.revisoes_poupadas == reference - row.revisoes
        assert row.horas_poupadas == pytest.approx(row.revisoes_poupadas * minutes / 60)
        assert row.par2_recuperados + row.par2_nao_recuperados == 36
        assert row.horas_poupadas_por_vinculo_perdido == pytest.approx(row.horas_poupadas / row.par2_nao_recuperados)
        source = same_load.loc[same_load.modo.eq(row.modo) & same_load.carga.eq(row.carga)
                               & same_load.ordenacao.eq("O3_p_cal_ml")].iloc[0]
        assert row.orcamento_avaliacoes == source.k
        if row.braco == "P":
            assert row.revisoes == source.P_pares_humano
            assert row.par2_recuperados == source.P_par2_aceitos + source.P_par2_humano
            assert row.FP == source.P_fp
            assert row.horas_poupadas_por_fp == pytest.approx(row.horas_poupadas / row.FP)
        else:
            assert row.revisoes == source.k and row.FP == 0
            assert row.par2_recuperados == source.par2_topk
            assert pd.isna(row.horas_poupadas_por_fp)
    p = table.loc[table.braco.eq("P")].groupby("modo").revisoes.nunique()
    assert p.eq(1).all()  # K2 não transforma as chamadas do LLM em trabalho humano.


def test_FP_region_thresholds_are_inclusive_and_count_only_nonlinks():
    frame = pd.DataFrame({"passo": [6, 6, 9, 9, 9], "score": [-5.65, -6, -5, -4, 1],
                          "par": [0, 0, 0, 0, 2], "accepted": [True, True, True, True, True]})
    table = aggregate(frame, "test", -5.65, {6: -5.65, 9: -4})
    total = table.loc[table.passo.eq("TOTAL")].set_index(["limiar", "regiao_SCORE"]).FP
    assert total.to_dict() == {("global", "maior_ou_igual"): 3, ("global", "menor"): 1,
                               ("por_passo", "maior_ou_igual"): 2, ("por_passo", "menor"): 2}


def test_FP_region_versioned_counts_close_without_private_inputs():
    table = pd.read_csv(ROOT / "data/fp_arranjo_por_regiao.csv")
    assert len(table) == 32
    for (mode, version), group in table.groupby(["modo", "limiar"]):
        expected = 21 if mode == "vigilancia" else 4
        assert group.loc[group.passo.eq("TOTAL"), "FP"].sum() == expected
        assert group.loc[~group.passo.eq("TOTAL"), "FP"].sum() == expected
    assert not {"pair_id", "PAR", "COMPREC", "REFREC", "NOME"}.intersection(table.columns)


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv",
                ROOT / ".local/arranjo_corte/policy_vigilancia_P.csv",
                ROOT / ".local/arranjo_corte/policy_confirmacao_P.csv")
@requires_reviewer_sources()
def test_FP_regions_regenerate_exactly_from_private_inputs(tmp_path):
    target = tmp_path / "regions.csv"
    fp_regions(DATA_ROOT, target)
    assert target.read_bytes() == (ROOT / "data/fp_arranjo_por_regiao.csv").read_bytes()


def test_editorial_tables_regenerate_and_only_the_caption_changes_for_cascade(tmp_path):
    editorial(tmp_path)
    names = ("tab_equidade_v31.tex", "tab_equidade_v31_rejeicoes_cascata.tex",
             "tab_qw3_group_metrics.tex", "tab_cap8_cascata_v31.tex")
    for name in names:
        generated = (tmp_path / name).read_text(encoding="utf-8")
        assert generated == (ROOT / "text/latex/tables/editorial" / name).read_text(encoding="utf-8")
        assert "comparação" in generated
    from scripts.fechar_refrec_final import regenerate_cascade_table
    historical = tmp_path / "historical.tex"
    regenerate_cascade_table(output_path=historical)
    original = (ROOT / "text/latex/tables/tab_cap8_cascata_v31.tex").read_text(encoding="utf-8")
    assert historical.read_text(encoding="utf-8") == original
    expected = replace_once(original, "Cascata final observada por política e modo,",
                            "Cascata sobre a base inteira (comparação), por política e modo,")
    # Revisão 2 (guia 3.6): a legenda e as linhas de contagem trocam "vínculos" por "pares verdadeiros" (a classe
    # segundo a referência) e "falso vínculo" por "falso positivo". Nenhuma célula numérica muda; a igualdade
    # exata abaixo a garante, já que só estas substituições textuais e o glossário separam a tabela da histórica.
    for old, new in [("(vínculos verdadeiros)", "(pares verdadeiros)"),
                     ("Vínculos perdidos (NONMATCH)", "Pares verdadeiros perdidos (NONMATCH)"),
                     ("Vínculos encaminhados ao humano", "Pares verdadeiros encaminhados ao humano"),
                     ("mantém vínculos ao humano no denominador",
                      "mantém no denominador os pares verdadeiros encaminhados ao humano"),
                     ("ausência de falso vínculo automático", "ausência de falso positivo automático")]:
        expected = replace_once(expected, old, new)
    glossary = (r"\par\footnotesize Cascata: a mesma política aplicada a todos os potenciais pares. "
                "OOF: predições fora da amostra de treinamento; REFREC: registro do SIM. "
                "ALWAYS: guarda; AUTO: decisão por custo; MATCH: vínculo; NONMATCH: não aceito como vínculo; UNSURE: inconclusivo. "
                "v3.1 distingue revisão humana da revisão por modelo de linguagem; v4 são as instruções sem as saídas decisórias do motor.")
    expected = expected.replace(r"\end{longtable}", r"\end{longtable}" + "\n" + glossary + "\n")
    # Guia 3.6: MATCH e NONMATCH entre \texttt{} (só marcação; nenhuma célula numérica muda).
    from scripts.tabelas_persona_editorial import texttt_codigos
    expected = texttt_codigos(expected)
    expected = expected.replace("\\footnotesize\n\\setlength{\\tabcolsep}{3pt}", "\\scriptsize\n\\setlength{\\tabcolsep}{3pt}", 1)
    assert (tmp_path / names[-1]).read_text(encoding="utf-8") == expected
    for file in ("estudo4_gzcmd.tex", "estudo5_equidade.tex"):
        chapter = (ROOT / "text/latex" / file).read_text(encoding="utf-8")
        for name in names:
            assert r"\input{tables/" + name + "}" not in chapter


def test_editorial_generators_refuse_historical_output_directory():
    from scripts.tabelas_equidade_comparacao import generate
    for call in (lambda: editorial(ROOT / "text/latex/tables"),
                 lambda: generate(ROOT / "data", ROOT / "text/latex/tables")):
        with pytest.raises(ValueError, match="editorial"):
            call()


@pytest.mark.parametrize("caption", ["missing", "duplicate"])
def test_cascade_renderer_rejects_missing_or_ambiguous_caption(tmp_path, monkeypatch, caption):
    from scripts import fechar_refrec_final as renderer
    (tmp_path / "data").mkdir()
    tables = tmp_path / "text/latex/tables"
    tables.mkdir(parents=True)
    for name in ("funil_gzcmd_final.csv", "funil_gzcmd_final_metricas.csv"):
        (tmp_path / "data" / name).write_bytes((ROOT / "data" / name).read_bytes())
    text = (ROOT / "text/latex/tables/tab_cap8_cascata_v31.tex").read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith(r"\caption{"))
    text = replace_once(text, line, "" if caption == "missing" else line + "\n" + line)
    (tables / "tab_cap8_cascata_v31.tex").write_text(text, encoding="utf-8")
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Legenda histórica"):
        renderer.regenerate_cascade_table()


def test_M2_exact_pinned_comparator_and_rates_hashes_without_EOL_tolerance():
    meta = json.loads((ROOT / "data/arranjo_corte_metadata.json").read_text(encoding="utf-8"))
    source = "scripts/arranjo_corte_orl_gzcmd.py"
    assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == meta["code_sha256"][source]
    assert hashlib.sha256((ROOT / "data/arranjo_corte_taxas_crossfit.csv").read_bytes()).hexdigest() == meta["frozen_rates_sha256"]


def test_production_N_replicas_feed_actual_frozen_records_to_rate_estimator(tmp_path, monkeypatch):
    from gzcmd import arranjo_robustez as robust
    from gzcmd.arranjo_corte import crossfit_rates
    from scripts import arranjo_robustez as cli

    class Model:
        def fit(self, x, y):
            self.offset = float(np.mean(y)) * .01
        def predict_proba(self, x):
            p = .02 + .9 * x.signal.to_numpy() + self.offset
            return np.column_stack([1 - p, p])

    x = pd.DataFrame({"signal": np.arange(400) / 400, "nota final": 4.0})
    y = np.tile([0, 1, 0, 1], 100)
    par = np.where(y, 2, 0)
    ids = pd.Series([f"synthetic_{i}" for i in range(400)])
    groups = np.repeat(np.arange(100), 4)
    below = np.ones(400, dtype=bool)
    monkeypatch.setattr(robust, "replica_factory", lambda *_: lambda *_: Model())
    monkeypatch.setattr(cli, "LOCAL", tmp_path)
    monkeypatch.setattr(cli, "verified", lambda *_: ({}, "synthetic"))
    monkeypatch.setattr(cli, "identity_for", lambda *_: {})
    monkeypatch.setattr(cli, "inputs", lambda *_: (x, below, x, par, ids, groups))
    frozen = dict(zip(ids.iloc[::2], ["MATCH"] * 200))
    monkeypatch.setattr(cli, "frozen_responses", lambda *_: frozen)

    def forbidden(*args, **kwargs):
        raise AssertionError("Respostas novas não podem alimentar as taxas")

    monkeypatch.setattr(cli, "known_responses", forbidden)
    cfg = cli.load_config(cli.s1.CONFIG)
    bands = cli.BandAssigner.from_config(cfg).assign(x["nota final"])
    _, baseline_fold, _ = robust.run_replica(x, y, below, groups, par2=y.astype(bool), replica=robust.grid()[0])
    reference = pd.DataFrame({"pair_id": ids, "fold": baseline_fold, "band": bands, "y": y, "par": par})
    reference["decision"] = reference.pair_id.map(frozen)
    reference = reference.loc[reference.decision.notna()]
    expected_rates = crossfit_rates(reference, cfg.llm_review.error_rates_by_band, list(range(5)))
    rows = [{"modo": mode, "fold": fold, "banda": band, "e_fp": rate.e_fp,
             "e_fn": rate.e_fn, "e_unsure": rate.e_unsure}
            for mode in cli.s1.MODES for fold, values in expected_rates.items() for band, rate in values.items()]
    expected_path = tmp_path / "expected.csv"
    pd.DataFrame(rows)[cli.s1.RATE_COLUMNS].to_csv(expected_path, index=False)
    monkeypatch.setattr(cli, "input_path", lambda *_: expected_path)
    received = []

    def spy(records, fixed, folds):
        pd.testing.assert_frame_equal(records.reset_index(drop=True), reference.reset_index(drop=True))
        assert set(records.pair_id) == set(frozen)
        actual = crossfit_rates(records, fixed, folds)
        assert actual == expected_rates
        received.append(records.copy())
        return actual

    monkeypatch.setattr(cli, "crossfit_rates", spy)
    for index in (0, 2):
        cli.child_replica(tmp_path, "synthetic", index, False)
        table = pd.read_csv(tmp_path / robust.grid()[index].replica / "taxas.csv")
        pd.testing.assert_frame_equal(table, pd.read_csv(expected_path))
    assert len(received) == 4
