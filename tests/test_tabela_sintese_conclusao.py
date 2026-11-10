"""Quadro-síntese da conclusão (tab:concl-sintese), gerado por scripts/tabela_sintese_conclusao.py.

Protege: (a) a tabela versionada é a regenerada; (b) todo número do quadro bate com o artefato
de origem e a divergência interrompe a geração; (c) as afirmações sem número (IC que inclui ou não
o zero, ordem entre arranjos) continuam sustentadas pelos CSV; (d) rótulo único, remissões
válidas, sem \\resizebox nem travessão, sem formas terminológicas proibidas; (e) a conclusão insere
o quadro logo após a frase de resposta, na seção de recomendações.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from scripts import tabela_sintese_conclusao as ts

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text" / "latex"
TABELA = LATEX / "tables" / "editorial" / "tab_concl_sintese.tex"
CONCLUSAO = LATEX / "conclusao.tex"
INPUT_QUADRO = r"\input{tables/editorial/tab_concl_sintese.tex}"
# Nenhuma frase da conclusão é fixada por igualdade literal (regra do registro): o parágrafo de abertura da
# seção de recomendações tem de conter estes termos, no padrão das âncoras.
TERMOS_ABERTURA = (
    "dominou em todas as cargas", "hipótese de trabalho", "validar em coorte nova", "preservar o corte",
    "O3", "F4; F3 quando houver as MACD", "curva de esforço", r"\textit{post hoc}", "revisor simulado",
    "não reduziram a revisão manual além da ordenação", "cascatas v3 e v3.1", r"\ref{tab:concl-sintese}",
)


def _abertura(texto: str) -> str:
    """Parágrafo que vem logo após o rótulo da seção de recomendações."""
    depois = texto.split(r"\label{sec:concl-recomendacoes}", 1)[1]
    return [p for p in depois.split("\n\n") if p.strip()][0].strip()


@pytest.fixture(scope="module")
def fontes():
    return ts.carregar()


def _tex(caminho: Path) -> str:
    return caminho.read_text(encoding="utf-8", errors="replace")


def test_tabela_versionada_igual_a_regenerada(fontes):
    assert TABELA.read_text(encoding="utf-8") == ts.render(fontes)


def test_numeros_do_quadro_batem_com_os_artefatos(fontes):
    valores = ts.conferir_numeros(fontes)
    assert set(valores) == ts.chaves_usadas(), "número definido e não usado, ou usado e não definido"
    assert all(n.fonte.strip() for n in ts.NUMEROS), "todo número tem fonte"
    assert all(n.calcular(fontes) == n.esperado for n in ts.NUMEROS)


def test_afirmacoes_sem_numero_sao_sustentadas_pelos_artefatos(fontes):
    ts.conferir_afirmacoes(fontes)


def test_numero_divergente_do_artefato_interrompe_a_geracao(fontes):
    alterado = copy.deepcopy(fontes)
    d = alterado["mesma_carga"]
    mascara = (d.ordenacao == "O3_p_cal_ml") & (d.modo == "vigilancia") & (d.carga == "k_min_equivalente_P")
    assert mascara.sum() == 1
    d.loc[mascara, "k"] = 999
    with pytest.raises(ValueError, match="o3_v"):
        ts.render(alterado)


def test_equidade_e_llm_divergentes_interrompem_a_geracao(fontes):
    alterado = copy.deepcopy(fontes)
    alterado["vpp_llm"].loc[(alterado["vpp_llm"].modo == "vigilancia") & (alterado["vpp_llm"].braco == "P"), "FP"] = 7
    with pytest.raises(ValueError, match="llm_fp"):
        ts.render(alterado)


def test_afirmacao_sem_numero_divergente_interrompe_a_geracao(fontes):
    alterado = copy.deepcopy(fontes)
    alterado["boot_o3_o1"]["diferenca_revisoes_ic95_sup"] = -1.0
    with pytest.raises(ValueError, match="IC de O3 menos O1 inclui zero"):
        ts.conferir_afirmacoes(alterado)


def test_rotulo_unico_na_arvore_de_latex():
    total = sum(_tex(c).count(r"\label{" + ts.LABEL + "}") for c in LATEX.rglob("*.tex"))
    assert total == 1


def test_sem_resizebox_sem_travessao_e_em_footnotesize():
    texto = TABELA.read_text(encoding="utf-8")
    assert r"\resizebox" not in texto
    assert r"\footnotesize" in texto
    for travessao in ("\u2014", "\u2013", "--"):
        assert travessao not in texto


def test_cinco_colunas_em_todas_as_linhas_do_corpo():
    texto = TABELA.read_text(encoding="utf-8")
    cabecalho, corpo = texto.split(r"\endlastfoot", 1)
    corpo = corpo.split(r"\end{longtable}", 1)[0]
    linhas = [l for l in corpo.splitlines() if l.rstrip().endswith(r"\\")]
    assert len(linhas) == sum(len(g) for _, g in ts.GRUPOS)
    assert all(len(l.rstrip()[:-2].split(" & ")) == 5 for l in linhas)
    # cabeçalho da primeira página e o repetido nas seguintes: as mesmas cinco colunas
    esperado = ["Abordagem", "Pergunta", "Resultado principal", "Uso recomendado", "Onde"]
    cabecalhos = [l for l in cabecalho.splitlines() if l.startswith("Abordagem & ")]
    assert len(cabecalhos) == 2
    assert all([c.strip() for c in l.rstrip()[:-2].split(" & ")] == esperado for l in cabecalhos)


def test_estrutura_longtable_sem_float_e_com_nota_fora_da_tabela():
    texto = TABELA.read_text(encoding="utf-8")
    assert texto.count(r"\begin{longtable}") == texto.count(r"\end{longtable}") == 1
    for proibido in (r"\begin{table}", r"\begin{tabular}", r"\begin{minipage}", r"\parbox", r"\centering"):
        assert proibido not in texto, proibido
    ordem = [r"\endfirsthead", r"\endhead", r"\endfoot", r"\bottomrule" + "\n" + r"\endlastfoot",
             r"\end{longtable}", r"\noindent Quando há dois valores", r"\endgroup"]
    posicoes = [texto.index(t) for t in ordem]
    assert posicoes == sorted(posicoes)
    assert texto.index(r"\begingroup") < texto.index(r"\setlength{\tabcolsep}{3pt}") < texto.index(r"\begin{longtable}")
    assert texto.index(r"\caption[") < posicoes[0] and r"}\label{" + ts.LABEL + r"}\\" in texto
    assert r"\multicolumn{5}{l}{\textit{(continuação)}} \\" in texto
    assert r"\multicolumn{5}{r}{\textit{(continua)}} \\" in texto
    assert texto.rstrip().endswith(r"\endgroup")
    assert ts.NUM_COLUNAS == 5
    # legenda na largura do texto (o padrão do longtable é de 4 polegadas)
    capwidth = r"\setlength{\LTcapwidth}{\linewidth}"
    assert texto.index(r"\begingroup") < texto.index(capwidth) < texto.index(r"\begin{longtable}")


def test_remissoes_do_quadro_e_da_conclusao_resolvem():
    rotulos = set()
    for caminho in LATEX.rglob("*.tex"):
        rotulos.update(re.findall(r"\\label\{([^}]+)\}", _tex(caminho)))
    for caminho in (TABELA, CONCLUSAO):
        faltam = sorted(set(re.findall(r"\\ref\{([^}]+)\}", _tex(caminho))) - rotulos)
        assert not faltam, f"{caminho.name}: remissões sem rótulo: {faltam}"


def test_terminologia_do_guia_no_quadro():
    texto = TABELA.read_text(encoding="utf-8")
    proibidas = ["vínculo", "área cinza", "zona cinz", "revisão clerical", "padrão-ouro", "pareamento",
                 "não-par", "PPV", "SINAN", "bancos de dados", "record linkage", "regressão RL"]
    assert [p for p in proibidas if p in texto] == []
    assert re.findall(r"precis[ãa]o(?!-sensibilidade)", texto) == []
    assert "regressão logística" in texto
    assert r"\textit{grey area}" in texto
    assert "CM: O3 com atributos de contexto por óbito e MACD" in texto


def test_conclusao_responde_qual_o_melhor_arranjo_antes_do_quadro():
    texto = _tex(CONCLUSAO)
    assert texto.count(INPUT_QUADRO) == 1
    assert texto.count(r"\ref{" + ts.LABEL + "}") == 1
    secao = texto.index(r"\label{sec:concl-recomendacoes}")
    abertura = _abertura(texto)
    frase = texto.index(abertura)
    remissao = texto.index(r"\ref{" + ts.LABEL + "}")
    quadro = texto.index(INPUT_QUADRO)
    recomenda = texto.index("Recomenda-se, a validar em coorte nova com referência cega")
    assert secao < frase < remissao < quadro < recomenda
    assert texto[frase:quadro].strip() == abertura, "o quadro vem logo após o parágrafo de abertura"
    faltam = [t for t in TERMOS_ABERTURA if t not in abertura]
    assert not faltam, f"termos ausentes da abertura: {faltam}"
    secao_recomendacoes = texto.split(r"\label{sec:concl-recomendacoes}", 1)[1].split(r"\section{Validação", 1)[0]
    assert secao_recomendacoes.lower().count("nenhuma ordenação dominou em todas as cargas") == 1


def test_abertura_nao_traz_numero_novo_e_cita_rotulos_existentes():
    # só os nomes v3, v3.1, F3, F4 e O3 e os rótulos têm dígitos: nenhum número novo na abertura
    abertura = _abertura(_tex(CONCLUSAO))
    sem_comandos = re.sub(r"\\(?:ref|label)\{[^}]*\}", "", abertura)
    sem_nomes = re.sub(r"\b(?:v3(?:\.1)?|F[34]|O3)\b", "", sem_comandos)
    assert re.findall(r"\d", sem_nomes) == []
    rotulos = set()
    for caminho in LATEX.rglob("*.tex"):
        rotulos.update(re.findall(r"\\label\{([^}]+)\}", _tex(caminho)))
    assert {"sec:e4-arranjo", "tab:cap8-referencia", ts.LABEL} <= rotulos


def test_numeros_novos_da_revisao_batem_com_os_artefatos(fontes):
    valores = ts.conferir_numeros(fontes)
    esperado = {"o1_96": "19", "o3_96": "22", "asr_perd_v": "13", "asr_perd_c": "27", "asr_hum_v": "96",
                "asr_hum_c": "17", "lr_aucpr": "0,967", "rf_aucpr": "0,956", "rfs_aucpr": "0,973",
                "fil_H": "22", "fil_H_c": "18"}
    assert {k: valores[k] for k in esperado} == esperado
    texto = TABELA.read_text(encoding="utf-8")
    for trecho in ("com as 96 revisões da vigilância (O1: 19; O3: 22)", "13 e 27 pares verdadeiros perdidos",
                   "96 e 17 pares encaminhados ao revisor humano", "(0,967 na regressão logística contra 0,956 no Random Forest)",
                   "a carga humana de 22 é a de uma réplica (18 na outra)"):
        assert trecho in texto, trecho


def test_arredondamento_meia_unidade_para_cima():
    assert ts.inteiro(0.5) == "1" and ts.inteiro(2.5) == "3" and ts.inteiro(1234567) == "1.234.567"
    assert ts.percentual(0.0005) == "0,1" and ts.percentual(0.125) == "12,5"
    assert ts.decimal3(0.0005) == "0,001" and ts.decimal3(0.9726) == "0,973"
