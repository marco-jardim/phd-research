"""Rastreabilidade editorial W9a, somente leitura de agregados públicos."""
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"
KEYS = {
    "Ather2026llmlinkage": "10.1177/18747655261422068",
    "Rahman2019bandits": "10.1145/3308558.3313675",
    "Fisch2024stratppi": "10.48550/arxiv.2406.04291",
    "Alaofi2026llmrelevance": "10.1145/3788872",
    "Gienapp2026relevance": "10.1145/3805712.3809713",
    "Callaghan2020stopping": "10.1186/s13643-020-01521-4",
    "Lewis2021certifying": "10.1145/3459637.3482415",
    "Lam2026clerical": "10.23889/ijpds.v11i5.3595",
    "DiNunzio2026stopping": "10.1145/3774905.3795598",
}


def pt_number(value, places=0):
    rounded = Decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f"{rounded:,.{places}f}".translate(str.maketrans({",": ".", ".": ","}))


def study():
    return (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")


def expected_ablation_appendix():
    """Preserva igualdade integral; W13a incorporou as glosas ao gerador."""
    from scripts.tabelas_review2_w3 import appendix_ablation

    expected = appendix_ablation()
    for state in ("ativado", "desativado"):
        for mode, error in (("confirmação", "positivo"), ("vigilância", "negativo")):
            caption = ("Ablação restrita aos subescores e à nota final, com partição por COMPREC (registro do Sinan), "
                       f"MACD {state} (Medidas Contínuas de Diferença de Datas de nascimento), modo {mode} "
                       f"(maior custo para falso {error}): métricas esperadas, sem chamadas a modelo de linguagem.")
            assert expected.count(caption) == 1
    gloss = ("MACD designa as Medidas Contínuas de Diferença de Datas de nascimento, atributos derivados dessas datas; "
             "v3 identifica a política histórica de três ações (aceitar como vínculo, rejeitar e revisão por modelo de linguagem).\n\n")
    assert expected.startswith(gloss)
    return expected


def test_sensibilidade_text_matches_csv_by_variant_and_metric():
    paragraph = next(p for p in study().split("\n\n")
                     if p.startswith(r"Na sensibilidade exploratória \textit{post hoc}"))
    with (ROOT / "data/sensibilidade_o3_macd.csv").open(encoding="utf-8", newline="") as handle:
        values = {(r["variante"], r["metrica"]): r["valor"] for r in csv.DictReader(handle)}
    triplets = []
    for variant in ("O3_MACD", "O3_controle", "O1_SCORE"):
        nums = [pt_number(values[variant, f"revisoes_{n}"]) for n in (9, 23, 36)]
        triplets.append(f"{nums[0]}, {nums[1]} e {nums[2]}")
    assert f"respectivamente, {triplets[0]} revisões, contra {triplets[1]} de O3 original e {triplets[2]} de O1" in paragraph
    ap = [pt_number(values[v, "ap_abaixo"], 3) for v in ("O3_controle", "O3_MACD")]
    auc = [pt_number(values[v, "auc_abaixo"], 3) for v in ("O3_controle", "O3_MACD")]
    assert f"(AP) abaixo do corte caiu de {ap[0]} para {ap[1]}" in paragraph
    assert f"(AUC) aumentou de {auc[0]} para {auc[1]}" in paragraph
    assert "encurtou a cauda, mas piorou o topo" in paragraph
    assert "não modifica o calibrador de ASR nem a hipótese fixada" in paragraph


@pytest.mark.parametrize("queue", ["F1", "F2", "F3", "F4", "A1", "A2"])
def test_fusion_table_matches_csv_means_and_quantiles(queue):
    source = study()
    table = source.split(r"\label{tab:cap8-fusao-posthoc}", 1)[1].split(r"\end{table}", 1)[0]
    row = next(line for line in table.splitlines() if line.startswith(queue + " ("))
    cells = [cell.strip().removesuffix(r"\\").strip() for cell in row.split("&")[1:]]
    with (ROOT / "data/fusao_filas.csv").open(encoding="utf-8", newline="") as handle:
        values = {(r["fila"], r["metrica"]): r for r in csv.DictReader(handle)}
    expected = []
    for n in (9, 23, 36):
        metric = values[queue, f"revisoes_{n}"]
        if queue.startswith("A"):
            expected.append(f"{pt_number(metric['media'], 2)} [{pt_number(metric['p10'], 2)}; {pt_number(metric['p90'], 2)}]")
        else:
            expected.append(pt_number(metric["media"]))
    assert cells == expected


@pytest.mark.parametrize("name", ["referencias.bib", "referencias_abnt.bib"])
def test_nine_keys_are_unique_have_exact_dois_and_are_cited(name):
    bib = (LATEX / name).read_text(encoding="utf-8")
    blocks = re.findall(r"@\w+\{([^,]+),([\s\S]*?)(?=\n@|\Z)", bib)
    for key, doi in KEYS.items():
        entries = [entry for name, entry in blocks if name == key]
        assert len(entries) == 1, key
        assert bib.lower().count(doi.lower()) == 1, (key, "DOI duplicado")
        assert re.search(r"doi\s*=\s*\{" + re.escape(doi) + r"\}", entries[0])
        # W11a separou o registro da v4 das propostas de continuidade da conclusão.
        # Conclusão reescrita: Fisch2024stratppi deixou de ser citada em qualquer capítulo; a entrada
        # (DOI exato, única) segue verificada no .bib, e a citação só é exigida das demais chaves.
        if key == "Fisch2024stratppi":
            continue
        assert re.search(r"\\cite(?:online)?\{[^}]*\b" + key + r"\b", study())
    fisch = next(entry for key, entry in blocks if key == "Fisch2024stratppi")
    assert "preprint" in fisch and "arXiv" in fisch
    gienapp = next(entry for key, entry in blocks if key == "Gienapp2026relevance")
    assert re.search(r"year\s*=\s*\{2026\}", gienapp)  # chave e ano coerentes com o DOI


def test_primary_criterion_is_preserved_and_secondary_is_registered_candidate():
    paragraph = next(p for p in study().split("\n\n")
                     if p.startswith("Como critério prospectivo de falsificação"))
    assert r"pelo menos 25\% menos revisões que O1" in paragraph
    assert r"limite inferior do intervalo de 95\% entre partições predefinidas" in paragraph
    assert "os 23 PAR~=~2 da vigilância e os nove da confirmação" in paragraph
    assert "Nenhuma dessas avaliações foi realizada" in paragraph
    for phrase in ("A comparação primária é entre a ordenação pela probabilidade calibrada do classificador (O3)",
                   "e a ordenação pelo escore do OpenRecLink (O1)",
                   "Para a coorte nova, o registro da v4 candidata",
                   r"\path{gzcmd/M_v4_candidata.md}",
                   r"\path{data/v4_candidata_registro.json}",
                   "fixa F4 e CM como filas secundárias",
                   "exclui LLM como evidência, dossiê v5 e encaminhamento conformal",
                   "mantendo referência cega por amostragem estratificada",
                   "o método de parada com garantia de recuperação permanece pendente de escolha antes da coorte"):
        assert phrase in paragraph
    for key in ("Callaghan2020stopping", "Lewis2021certifying",
                "DiNunzio2026stopping", "Lam2026clerical"):
        assert key in paragraph
    # Conclusão reescrita: o parágrafo de continuidade com Fisch2024stratppi foi removido por concisão; a
    # conclusão mantém a referência cega por amostragem estratificada e a recomendação sem fixar o registro da v4.
    conclusion = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    assert "Fisch2024stratppi" not in conclusion
    assert r"A referência da coorte nova pode ser obtida por amostragem estratificada \cite{Lam2026clerical}" in conclusion


def test_ablation_keeps_generated_values_with_exact_editorial_glosses():
    assert (LATEX / "apendice_ablacao.tex").read_text(encoding="utf-8") == expected_ablation_appendix()


def test_fusion_interpretation_keeps_f1_exception_and_all_six_variants():
    paragraph = next(p for p in study().split("\n\n")
                     if p.startswith(r"Outra sensibilidade exploratória \textit{post hoc}"))
    # Âncora C12: F4 foi escolhida entre seis variantes, sem correção para multiplicidade.
    for phrase in ("F1 não superou O3",
                   "F4 foi escolhida entre seis variantes avaliadas nos mesmos 36 PAR~=~2, sem correção para multiplicidade",
                   "sem conhecimento prévio dos desfechos", "não constitui pré-registro",
                   "viés de verificação favorável a O1 na cauda", "200 repetições"):
        assert phrase in paragraph
    assert "sem selecionar a melhor" not in paragraph


def test_census_first_mentions_and_captions_are_self_contained():
    text = study()
    union = next(p for p in text.split("\n\n") if r"na união $\mathrm{ASR}\cup B1$" in p)
    assert "B1, ponte que mantém a probabilidade original" in union
    first_k2 = next(p for p in text.split("\n\n") if re.search(r"\bK2\b(?!\.\d)", p))
    assert "K2, soma das avaliações humanas e LLM" in first_k2
    # Revisão 2: a tabela do Experimento 2 foi para o Apêndice do Estudo 4 (A.4).
    appendix = (LATEX / "apendice_gzcmd.tex").read_text(encoding="utf-8")
    for label, source, phrases in (
        ("tab:cap8-revisao-desempenho", appendix, ("Experimento 2 (Exp2, revisão assistida)",
                                                   "incluindo as saídas decisórias do motor", "versão histórica que solicita")),
        ("tab:cap8-robustez-arranjo", text, ("ASR-Platt (análise de sensibilidade de ASR com", "ASR (arranjo sequencial", "reexecuções com outra")),
        ("fig:cap8-curva-esforco", text, ("O1 ordena", "O2, pela nota", "O3, pela probabilidade", "O4, pela probabilidade",
                                         "ASR é o arranjo sequencial", "sem falso positivo humano")),
    ):
        before = source.split(r"\label{" + label + "}", 1)[0]
        caption = before.rsplit(r"\caption{", 1)[1]
        for phrase in phrases:
            assert phrase in caption
    assert r"\footnote{$H$ é a quantidade de revisões humanas.}" not in text  # H também denota ação, não contagem
