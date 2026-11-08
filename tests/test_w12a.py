"""Conformidade W12a: contratos editoriais e rastreabilidade dos agregados."""

import csv
import re
from pathlib import Path

import pytest

from latex_fontes import fontes_tex
from scripts.tabela_tamanho_modelo_editorial import OUTPUT, generate, render


ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"


def text(name):
    return (LATEX / name).read_text(encoding="utf-8")


def test_generated_model_table_matches_file_bytes(tmp_path):
    destination = tmp_path / "modelos.tex"
    assert generate(destination) == destination
    assert destination.read_bytes() == OUTPUT.read_bytes()
    assert b"\r" not in destination.read_bytes()


def test_generator_uses_aggregate_values_and_rejects_ambiguous_rows(tmp_path):
    for name in ("probe_frontier_agregado.csv", "probe_frontier_custos.csv", "probe_frontier_metadata.json"):
        (tmp_path / name).write_bytes((ROOT / "data" / name).read_bytes())
    path = tmp_path / "probe_frontier_agregado.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    row = next(row for row in rows if (row["model"], row["modo"], row["estrato"], row["passada"])
               == ("claude-haiku-5-5", "vigilancia", "S1", "historico"))
    row.update(MATCH="20", FP="16", VPP="0.2")

    def write_rows():
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_rows()
    assert "Claude Haiku 5.5 & 25 & 20 & 16 & 20,0 & 13 & 0/6/7" in render(tmp_path)
    rows.append(row.copy())
    write_rows()
    with pytest.raises(ValueError, match="Esperada uma linha"):
        render(tmp_path)


def test_model_strata_are_editorial_and_all_four_are_defined():
    body = text("estudo4_gzcmd.tex")
    section = body.split(r"\subsection{Tamanho do modelo como revisor}")[1].split(r"\subsection{Limitações}")[0]
    assert not re.search(r"\bS[1-4]\b", section.replace("etapas S1 a S3 do protocolo", ""))
    for token in ("V1 reúne as 25 aceitações", "V2, os 13 PAR~=~2", "V3, os 24 PAR~=~0", "V4, a amostra de 88 dos 282"):
        assert token in section
    assert "V1: $n$" in render() and "V2: $n$" in render()
    assert "V3: 24 PAR~=~0" in render() and "V4: amostra de 88 dos 282" in render()
    # A renomeação não deve alterar as etapas do protocolo ou as chaves de origem.
    assert "simulação sem API (S1)" in body and "chamadas ao LLM (S2)" in body
    assert "avaliação com os \\textit{labels} (S3)" in body
    with (ROOT / "data/probe_frontier_agregado.csv").open(encoding="utf-8", newline="") as stream:
        assert {"S1", "S2", "S3", "S4"} <= {row["estrato"] for row in csv.DictReader(stream)}


def test_bayesian_evidence_compares_original_and_posterior_counts():
    with (ROOT / "data/llm_evidencia.csv").open(encoding="utf-8", newline="") as stream:
        rows = {(row["variante"], row["metrica"]): float(row["valor"])
                for row in csv.DictReader(stream) if row["modo"] == "vigilancia" and row["tipo"] == "aceitacao"}
    body = text("estudo4_gzcmd.tex")
    for variant in ("O3", "O3-post"):
        vp, n = rows[variant, "VP"], rows[variant, "aceitacoes"]
        vpp = f"{rows[variant, 'VPP'] * 100:.1f}".replace(".", ",")
        assert f"{vpp}\\% ({vp:.0f} vínculos em {n:.0f} aceitações)" in body
        assert rows[variant, "VPP"] == pytest.approx(vp / n)
    assert "16,0\\% quando o LLM aceita sozinho" in body


@pytest.mark.parametrize("name", ["estudo1_comparador.tex", "apendice_sensibilidade.tex"])  # estudo2 revisado (revisão 2): Experimento 1 fica no estudo1 e no apêndice
def test_experiment_one_has_study_scope_and_separating_comma(name):
    body = text(name)
    expansion = ("Medidas Contínuas de Diferença de Datas de nascimento, MACD, "
                 if name == "estudo1_comparador.tex" else "MACD ")
    assert "Experimento~1 do Estudo~4 (ablação das " + expansion + "no GZ-CMD)" in body
    assert not re.search(r"isotônica(?:\)| \(Capítulo~\\ref\{cap:estudo4\}\)) (?:utilizam|que não utilizam)", body)


def test_operational_reference_is_explicit_and_bias_direction_is_not_identified():
    body = text("capitulo_dados.tex")
    assert "revisão que originou PAR" not in body and "revisão clerical que originou PAR" not in body
    assert "se a revisão tivesse seguido o escore do OpenRecLink, a verificação favoreceria O1" in body
    assert "a direção dominante não é identificável" in body


def test_phonetic_blocking_description_does_not_infer_sex_frequency():
    body = text("estudo5_equidade.tex")
    assert "Soundex do primeiro e do último fragmentos do nome" in body
    assert "por exemplo após mudança de estado civil" in body
    assert "mais frequentes entre mulheres" not in body
    assert "quando a alteração modifica a chave fonética" in body


def test_version_glosses_are_not_repeated_inside_a_section():
    body = text("estudo4_gzcmd.tex")
    for section in re.split(r"(?=\\section\*?\{)", body):
        for version in ("v3.1", "v3", "v4"):
            assert section.count(version + " identifica") <= 1
    for phrase in ("v3.1 identifica a política de quatro ações", "v4 identifica as instruções do LLM"):
        assert phrase in body
    # As notas metodológicas não são glosas descartáveis.
    assert "Artefatos \\texttt{carga\\_abaixo\\_corte}" in body


def test_consensus_label_and_appendix_prose_are_unambiguous():
    search = text("apendice_busca.tex")
    assert "Consensus & CS1," in search
    assert not re.search(r"\bC[1-4]\b", search)
    prompts = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", text("apendice_prompts.tex"), flags=re.S)
    assert not re.search(r"\bbanda\b", prompts)
    assert "maior prefixo de tamanho $k$" in text("apendice_formalizacao.tex")


@pytest.mark.parametrize("name", ["referencias.bib", "referencias_abnt.bib"])
def test_bibliography_keys_and_protocol_are_consistent(name):
    bib = text(name)
    for removed in ("Doidge2018demystifying", "Paixao2017linkageevaluation", "DuVall2010extending", "Gienapp2025relevance"):
        assert removed not in bib
        assert all(removed not in path.read_text(encoding="utf-8") for path in fontes_tex(LATEX, recursivo=True))
    assert bib.count("@inproceedings{Gienapp2026relevance,") == 1
    protocol = re.search(r"(?m)^@book\{MS2017ProtocoloObitoTB,\n[\s\S]*?^\}", bib)
    source = re.search(r"(?m)^@book\{MS2017ProtocoloObitoTB,\n[\s\S]*?^\}", text("referencias.bib"))
    assert protocol and source and protocol.group() == source.group()
