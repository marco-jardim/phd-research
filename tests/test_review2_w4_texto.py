"""Contratos W4: limites do argumento e fontes agregadas, sem dados por par."""
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.tabelas_s8a1_editorial import human_sensitivity

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"


def text(name):
    return (LATEX / name).read_text(encoding="utf-8")


def test_falsification_is_prospective_with_explicit_margin_not_point_differences():
    for source in (text("estudo4_gzcmd.tex"), text("conclusao.tex"),
                   (ROOT / "text/slides/defesa.tex").read_text(encoding="utf-8")):
        normalized = source.replace("{,}", ",")
        assert "prospectivo" in normalized
        assert "outra coorte ou período" in normalized
        assert "5,6" in normalized and "2/36" in normalized
        assert "réplicas predefinidas" in normalized or "réplicas de partição predefinidas" in normalized
        assert "referência cega" in normalized.lower()
        assert "nota calibrada" not in normalized
    assert "não foi testado" in text("conclusao.tex")
    assert "nenhuma dessas avaliações foi realizada" in text("estudo4_gzcmd.tex")


def test_dossier_coverage_matches_aggregate_and_no_auditability_is_claimed():
    with (ROOT / "data/dossie_campos_enviados.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 29
    assert sum(row["enviado"] == "sim" for row in rows) == 14
    for name in ("estudo4_gzcmd.tex", "conclusao.tex", "apendice_prompts.tex"):
        source = text(name)
        assert "14 dos 29 subescores" in source
        assert "penalidade por nome comum da mãe" in source
        assert "inversões de data" in source and "UF" in source
        assert "evidência ausente" in source or "evidências ausentes" in source
    assert r"\input{tables/editorial/tab_dossie_campos.tex}" in text("estudo4_gzcmd.tex")
    assert "política auditável" not in text("conclusao.tex")
    assert "não demonstra auditabilidade" in text("estudo4_gzcmd.tex")


def test_ablation_is_not_a_negative_control_or_identified_mechanism():
    rows = pd.read_csv(ROOT / "data/ablacao_dossie_taxa_base.csv")
    deepseek = rows.loc[rows.coorte.eq("opencode_go_sem_mo_v4")].iloc[0]
    haiku = rows.loc[rows.coorte.eq("anthropic_sem_mo_v4")].iloc[0]
    assert (deepseek.n, deepseek.NONMATCH, haiku.NONMATCH) == (1845, 0, 1582)
    study = text("estudo4_gzcmd.tex")
    assert "compatíveis com ancoragem ou abstenção generalizada" in study
    assert "1.845 revisões" in study and "1.582" in study
    assert "Não há controle negativo com não vínculos" in study
    assert "retira somente o grupo" in study
    assert "versão do prompt não está registrada" in study
    assert "proxies adicionais" not in study
    assert "unidade é a revisão" in study


def test_cardinality_counts_do_not_imply_an_implemented_rule():
    counts = json.loads((ROOT / "data/fp_cardinalidade.json").read_text())
    vigil = counts["modos"]["vigilancia"]
    conf = counts["modos"]["confirmacao"]
    assert (vigil["fp_total"], vigil["refrec_ja_aceito"], vigil["comprec_ja_aceito"],
            vigil["nenhum_ja_aceito"]) == (21, 1, 0, 20)
    assert (conf["fp_total"], conf["comprec_ou_refrec_ja_aceito"]) == (4, 0)
    source = text("estudo4_gzcmd.tex")
    assert "previstos, mas não implementados" in source
    assert "não avalia conflitos entre novas aceitações" in source
    assert "um compartilha REFREC" in source and "os outros 20" in source


def test_latency_cost_changes_order_only_conditionally_on_fixed_actions():
    rows = pd.read_csv(ROOT / "data/custos_equivalentes_exploratorios.csv")
    selected = rows.loc[rows.modo.eq("vigilancia")].set_index("braco")
    match = selected.loc["P_MATCH_ao_humano"]
    o3 = selected.loc["O3_recuperacao_equivalente"]
    threshold = (o3.revisoes_humanas - match.revisoes_humanas) / match.pares_llm
    c = round(17.5 / 1200, 4)
    assert threshold == pytest.approx(4 / 344)
    assert match.revisoes_humanas + c * match.pares_llm > o3.revisoes_humanas
    assert match.perda_total < o3.perda_total  # custo original de 0,005
    source = text("estudo4_gzcmd.tex")
    assert "0{,}0116" in source and "0{,}0146" in source
    assert "Não se reexecutou a política nesse ponto" in source


def test_filter_modes_are_replicates_and_registered_verdict_is_not_replaced():
    source = text("estudo4_gzcmd.tex")
    assert "classificou o resultado como ``ajuda''" in source
    assert "comparação posterior de perda ponderada" in source
    assert "réplicas estocásticas" in source
    assert "119 decisões coincidem e seis diferem" in source
    assert "22 e 18" in source and "pertence a uma réplica" in source


def test_confirmation_editorial_human_scenarios_use_its_own_workloads():
    scenarios = pd.read_csv(ROOT / "data/cenarios_sensibilidade_humana.csv")
    curve = pd.read_csv(ROOT / "data/comparador_mesma_carga_curva.csv")
    rendered = human_sensitivity(scenarios, curve)
    rows = [line.removeprefix('Conf. & O3 & ') for line in rendered.splitlines()
            if line.startswith('Conf. & O3 & ')]
    assert len(rows) == 3
    assert [row.split(" & ")[0] for row in rows] == ["17", "29", "58"]
    for k in (17, 29, 58):
        count = curve.loc[curve.ordenacao.eq("O3_p_cal_ml") & curve.k.eq(k), "par2_topk"].item()
        expected = [(211 + s * count) / 247 * 100 for s in (.95, .99, 1.)]
        formatted = [f"{value:.2f}".replace(".", "{,}") for value in expected]
        assert " & ".join([str(k), str(k), *formatted]) in rendered
    changed = curve.copy()
    changed.loc[changed.ordenacao.eq("O3_p_cal_ml") & changed.k.eq(17), "par2_topk"] += 1
    assert rendered != human_sensitivity(scenarios, changed)
    with pytest.raises(ValueError, match="ausente ou ambíguo"):
        human_sensitivity(scenarios, curve.loc[curve.k.ne(17)])


def test_script_answers_cover_the_new_review_without_delivery_metatext():
    source = (ROOT / "text/slides/roteiro_defesa.md").read_text(encoding="utf-8")
    for n, phrase in ((53, "14 dos 29"), (54, "abstenção generalizada"),
                      (55, "0,0146"), (56, "não foi exercitada"),
                      (57, "amostra independente"), (58, "um compartilha REFREC"),
                      (59, "réplicas estocásticas"), (60, "0,65"),
                      (61, "snapshots imutáveis"), (62, "O autor fixou"), (63, "[5,6)")):
        section = source.split(f"### {n}.")[1].split("### ")[0]
        assert phrase in section
    for forbidden in ("pq/banca-", "PDFs existentes não representam", "revisão pela orientadora foi dispensada"):
        assert forbidden not in source
