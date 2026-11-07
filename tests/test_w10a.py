"""Contratos editoriais W10a contra os agregados públicos, sem inferência ou API."""

import csv
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text" / "latex"
TABLES = LATEX / "tables" / "editorial"
APPROVED = (
    "Recomenda-se a fusão como ordenação da fila de revisão no lugar da ordenação "
    "pelo escore do OpenRecLink, a validar em coorte nova com referência cega; "
    "sua vantagem sobre a probabilidade calibrada não foi demonstrada."
)
MODELS = (
    ("claude-haiku-5-5", "Claude Haiku 5.5", "historico"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5", "historico"),
    ("claude-opus-5-5", "Claude Opus 5.5", "principal"),
    ("claude-fable-5-1", "Claude Fable 5.1", "principal"),
)


def text(name="estudo4_gzcmd.tex"):
    return (LATEX / name).read_text(encoding="utf-8")


def table(name):
    return (TABLES / f"tab_cap8_{name}.tex").read_text(encoding="utf-8")


def rows(name):
    with (ROOT / "data" / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def one(data, **criteria):
    found = [r for r in data if all(r[k] == v for k, v in criteria.items())]
    assert len(found) == 1, criteria
    return found[0]


def pt(value, decimals=0):
    return f"{float(value):,.{decimals}f}".translate(str.maketrans(",.", ".,"))


def math_int(value, signed=False):
    value = round(float(value))
    prefix = "+" if signed and value > 0 else ""
    return f"${prefix}{pt(value)}$"


def test_approved_recommendation_and_analysis_boundaries():
    body = text()
    assert APPROVED in body
    assert "entre as estratégias O1 a O4 e P, nenhuma ordenação domina" in body
    assert "posição de precaução diante do VPP observado e da incerteza da referência" in body
    assert "A referência operacional como fonte de erro" in body
    assert "Tamanho do modelo como revisor" in body
    for name in ("fusao_robustez", "vies_referencia", "tamanho_modelo"):
        assert f"\\input{{tables/editorial/tab_cap8_{name}.tex}}" in body
        assert r"\textit{post hoc}" in table(name)


def test_registered_primary_comparison_margin_and_criterion_are_preserved():
    body = text()
    assert "A comparação primária é entre a ordenação pela probabilidade calibrada do classificador (O3) e a ordenação pelo escore do OpenRecLink (O1)" in body
    assert r"pelo menos 25\% menos revisões que O1, pelo limite inferior do intervalo de 95\% entre partições predefinidas" in body
    assert "os 23 PAR~=~2 da vigilância e os nove da confirmação nesta coorte" in body


@pytest.mark.parametrize("comparison", ["F4-O1", "F4-O3"])
@pytest.mark.parametrize("target", ["9", "23", "36"])
def test_bootstrap_table_matches_csv(comparison, target):
    r = one(rows("fusao_robustez_bootstrap.csv"), comparacao=comparison, meta_nominal=target)
    expected = (
        f"{comparison.replace('-', ' menos ')} & {target} & "
        f"{math_int(r['diferenca_revisoes_mediana'])} & "
        f"{math_int(r['diferenca_revisoes_ic95_inf'])} a "
        f"{math_int(r['diferenca_revisoes_ic95_sup'], signed=True)}"
    )
    assert expected in table("fusao_robustez")
    assert int(r["replicas_validas"]) == 2000
    upper = float(r["diferenca_revisoes_ic95_sup"])
    assert (upper < 0) == (comparison == "F4-O1" and target in ("9", "36"))


@pytest.mark.parametrize("queue", ["O1", "O3", "F4", "F3"])
def test_within_band_effort_matches_csv(queue):
    data = rows("fusao_robustez_faixa.csv")
    counts = [one(data, fila=queue, meta=t)["revisoes"] for t in ("9", "23", "36")]
    assert f"{queue} & " + " & ".join(pt(n) for n in counts) in table("fusao_robustez")


def test_threshold_precision_and_ties_match_metadata():
    meta = json.loads((ROOT / "data/fusao_robustez_metadata.json").read_text(encoding="utf-8"))
    band = meta["faixa"]
    assert (band["operational_n"], band["operational_PAR2"]) == (2522, 36)
    assert (band["literal_n"], band["literal_PAR2"]) == (2495, 31)
    assert -5.6485 <= band["operational_min_score"] < -5.648
    body = text()
    assert body.count(r"\geq-5{,}6485") == 2
    assert r"$\geq-5{,}648$" not in body
    assert r"$-5{,}64846$" in body
    assert "2.495 candidatos e 31 PAR~=~2" in body
    topo = json.loads((ROOT / "data/fusao_robustez_topo.json").read_text(encoding="utf-8"))
    assert (topo["n"], topo["par2"]) == (17, 16)
    sensitivity = topo["desempate_aleatorio"]
    assert sensitivity["repeticoes"] == 200
    assert set(sensitivity["metricas"]["par2_top17"].values()) == {16.0}
    assert "16 PAR~=~2 entre os 17 primeiros" in body
    assert "regra híbrida das metas muda o estimando" in body


def test_f3_recovers_25_at_96():
    r = one(rows("fusao_filas.csv"), fila="F3", metrica="par2_top_96")
    assert float(r["media"]) == 25
    assert "F3 recupera 25 PAR~=~2 com 96 revisões" in text()


@pytest.mark.parametrize("model,name,passada", MODELS)
def test_model_counts_vpp_confirmation_and_replica(model, name, passada):
    data = rows("probe_frontier_agregado.csv")
    s1 = one(data, model=model, modo="vigilancia", estrato="S1", passada=passada)
    s2 = one(data, model=model, modo="vigilancia", estrato="S2", passada=passada)
    cells = [pt(s1[k]) for k in ("n", "MATCH", "FP")]
    cells += [pt(float(s1["VPP"]) * 100, 1), pt(s2["n"])]
    cells += ["/".join(pt(s2[k]) for k in ("PAR2_recuperados", "PAR2_descartados", "PAR2_unsure"))]
    assert f"{name} & " + " & ".join(cells) in table("tamanho_modelo")
    confirm = one(data, model=model, modo="confirmacao", estrato="C", passada=passada)
    total = one(data, model=model, modo="vigilancia", estrato="TOTAL_AMOSTRA", passada=passada)
    kappas = "/".join(pt(r["kappa_replica"], 3) if r["kappa_replica"] else "ND" for r in (total, confirm))
    costs = rows("probe_frontier_custos.csv")
    probe = one(costs, model=model, escopo="probe")
    historic = [r for r in costs if r["model"] == model and r["escopo"] == "coorte_historica"]
    cost_probe = pt(probe["usd_por_par_revisado"], 4) if probe["usd_por_par_revisado"] else "ND"
    cost_historic = pt(historic[0]["usd_por_par_revisado"], 4) if historic else "ND"
    cells = [pt(confirm["n"]), pt(float(confirm["VPP"]) * 100, 1), kappas, cost_probe, cost_historic]
    assert f"{name} & " + " & ".join(cells) in table("tamanho_modelo")
    if model == "claude-sonnet-5-5":
        assert (float(s1["n"]), float(s2["n"]), float(confirm["n"])) == (21, 7, 33)


@pytest.mark.parametrize("mode,n,agreement,kappa", [
    ("vigilancia", 1410, "87,0", "0,555"),
    ("confirmacao", 435, "85,5", "0,659"),
])
def test_complete_historical_cohort(mode, n, agreement, kappa):
    r = one(rows("probe_haiku_sonnet_coorte.csv"), coorte="original", model="claude-haiku-5-5", modo=mode, faixa="TOTAL")
    assert int(r["n"]) == n
    assert pt(float(r["agreement"]) * 100, 1) == agreement
    assert pt(r["kappa"], 3) == kappa
    for value in (pt(n), agreement, kappa):
        assert value in text()


def test_bias_turning_points_and_expected_losses():
    turning = rows("vies_referencia_viradas.csv")
    c1 = one(turning, modo="vigilancia", conclusao="C1_VPP_menor_50", eixo="fp_uniforme", par2_nao_vinculo="0.0")
    assert float(c1["fracao_virada"]) == 0.5
    assert pt((0.5 * 25 - 4) / 21 * 100, 1) == "40,5"
    assert "40,5\\%" in table("vies_referencia")
    data = rows("vies_referencia.csv")
    for mode, losses in (("vigilancia", "10,4"), ("confirmacao", "21,6")):
        r = one(data, modo=mode, fp_abaixo="0.0", fp_acima="0.0", par2_nao_vinculo="0.2", metrica="perdidos")
        assert pt(r["valor_esperado"], 1) == losses
        assert losses in table("vies_referencia")
        c2 = one(turning, modo=mode, conclusao="C2_perdidos_13_27", eixo="par2_nao_vinculo", fp_abaixo="0.0", fp_acima="0.0")
        assert float(c2["fracao_virada"]) == 0.05
    c3 = [r for r in data if r["modo"] == "vigilancia" and r["metrica"] == "O3-O1_margem25_64"]
    assert len(c3) == 100
    assert all(float(r["mediana"]) < 0 for r in c3)
    assert sum(float(r["p975"]) > 0 for r in c3) == 1
    c4 = [r for r in turning if r["conclusao"].startswith("C4_")]
    assert c4 and all(r["status"] == "sem_virada_na_grade" for r in c4)


def test_strong_identifiers_and_truncation_are_not_positive_evidence():
    data = rows("vies_referencia_identificadores.csv")
    assert all(float(r["concordancia_maxima_simultanea"]) == 0 for r in data)
    assert int(one(data, modo="vigilancia", grupo="FP_P", regiao_limiar_passo="menor")["n"]) == 10
    cost = one(rows("probe_frontier_custos.csv"), model="claude-haiku-5-5", escopo="probe")
    assert int(cost["respostas_truncadas"]) == 17
    data = rows("probe_frontier_agregado.csv")
    missing = sum(float(one(data, model="claude-haiku-5-5", modo=m, estrato="TOTAL_AMOSTRA", passada="replica_historico")["n_sem_resultado"]) for m in ("vigilancia", "confirmacao"))
    assert missing == 16
    assert "17 respostas truncadas afetaram 16 pares" in text()
    assert "nenhum grupo" in text()


def test_owned_tex_has_no_dashes_and_new_references_exist():
    files = [LATEX / "estudo4_gzcmd.tex", *[p for p in LATEX.glob("apendice_*.tex") if p.name != "apendice_busca.tex"]]
    files += [TABLES / f"tab_cap8_{name}.tex" for name in ("fusao_robustez", "vies_referencia", "tamanho_modelo")]
    for path in files:
        source = path.read_text(encoding="utf-8")
        # As instruções históricas transcritas são dados, não prosa editorial.
        source = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", source, flags=re.S)
        prose = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
        assert not re.search("[—–]", prose), path
    bib = text("referencias_abnt.bib")
    for key in ("Lam2026clerical", "Ather2026llmlinkage"):
        assert re.search(r"@\w+\{" + key + ",", bib)
    for name in ("fusao_robustez", "vies_referencia", "tamanho_modelo"):
        source = table(name)
        assert source.count(r"\begin{table}") == source.count(r"\end{table}") == 1
        assert source.count(r"\begin{tabular}") == source.count(r"\end{tabular}")


@pytest.mark.parametrize("name", ["fusao_robustez", "vies_referencia", "tamanho_modelo"])
def test_new_numeric_tables_have_matching_column_counts(name):
    for specification, content in re.findall(r"\\begin\{tabular\}\{([lr]+)\}(.*?)\\end\{tabular\}", table(name), flags=re.S):
        for row in content.splitlines():
            if " & " in row and r"\multicolumn" not in row:
                assert row.count(" & ") + 1 == len(specification), row
