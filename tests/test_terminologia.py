"""Testes do auxiliar de normalização terminológica (guia de escrita, seção 3.6)."""

import pytest

from terminologia import (
    GREY_AREA,
    LINKAGE,
    NAO_PAR,
    PAR,
    REVISAO_MANUAL,
    SIH_SUS,
    SINAN,
    VPP,
    conta,
    contem,
    normalizar_terminologia as norm,
)

CLASSES = [
    (PAR, ["vínculo", "vínculos", "par verdadeiro", "pares verdadeiros"]),
    (NAO_PAR, ["não vínculo", "não vínculos", "não-vínculo", "não par", "não pares", "não-par"]),
    (GREY_AREA, ["área cinza", "áreas cinza", r"\textit{grey area}", "grey areas",
                 r"\textit{grey area} (área cinza)", r"\textit{grey areas} (áreas cinza)"]),
    (REVISAO_MANUAL, ["revisão clerical", "revisão manual", "revisões clericais", "revisões manuais"]),
    (VPP, ["precisão", "VPP", "valor preditivo positivo", "valor preditivo positivo (VPP)"]),
    (LINKAGE, ["relacionamento de registros", "vinculação de registros", "record linkage",
               r"\textit{linkage}", r"\textit{linkage} (relacionamento de registros)"]),
    (SINAN, ["SINAN", "Sinan"]),
    (SIH_SUS, ["SIH-SUS", "SIH/SUS"]),
]


@pytest.mark.parametrize("marcador,formas", CLASSES)
def test_cada_classe_converge_para_um_marcador(marcador, formas):
    for forma in formas:
        assert norm(forma) == marcador, forma


@pytest.mark.parametrize("marcador,formas", CLASSES)
def test_formas_da_mesma_classe_sao_intercambiaveis_nos_dois_sentidos(marcador, formas):
    for antiga in formas:
        for nova in formas:
            frase_a = f"Há 5 casos de {antiga} no período."
            frase_b = f"Há 5 casos de {nova} no período."
            assert contem(frase_a, frase_b) and contem(frase_b, frase_a)


def test_classes_distintas_nao_se_confundem():
    assert norm("vínculo") != norm("não vínculo")
    assert norm("área cinza") != norm("revisão manual")
    assert not contem("pares verdadeiros", "não pares")


def test_mais_longa_primeiro_nao_vinculo():
    assert norm("não vínculo") == NAO_PAR
    assert norm("os não vínculos") == f"os {NAO_PAR}"
    assert norm("um vínculo e um não-vínculo") == f"um {PAR} e um {NAO_PAR}"
    assert PAR not in norm("não vínculos")


def test_glosa_vira_um_unico_marcador():
    assert norm(r"A \textit{grey area} (área cinza) da nota") == f"a {GREY_AREA} da nota"
    assert norm(r"O \textit{linkage} (relacionamento de registros) é") == f"o {LINKAGE} é"
    assert norm("área cinza") == GREY_AREA == norm("grey area (área cinza)")
    assert conta("grey area (área cinza), depois área cinza", "área cinza") == 2


def test_zona_cinzenta_passa_intacta():
    assert norm("zona cinzenta") == "zona cinzenta"
    assert GREY_AREA not in norm("zona cinzenta")
    assert not contem("zona cinzenta", "área cinza")
    assert not contem("área cinza", "zona cinzenta")


def test_revisao_manual_de_origem_e_clerical_de_origem():
    assert norm("revisão manual de origem") == norm("revisão clerical de origem")
    assert contem("pela revisão manual de origem", "pela revisão clerical de origem")


def test_vpp_e_precisao():
    assert contem("sensibilidade de 0,932 e VPP de 0,885", "sensibilidade de 0,932 e precisão de 0,885")


def test_trivia_latex():
    assert norm(r"42{,}3\% de PAR~=~1") == "42,3% de par = 1"
    assert contem(r"PAR~=~1", "PAR = 1")
    assert contem(r"\textit{post mortem}", "post mortem")
    assert contem("a  b\n\nc", "a b c")
    assert norm("*grey area*") == GREY_AREA


def test_acentos_e_caixa():
    assert norm("VÍNCULOS") == PAR
    assert norm("Não Vínculo") == NAO_PAR
    assert norm("ÁREA CINZA") == GREY_AREA
    assert norm("vinculo") == "vinculo"  # sem acento não é a forma do texto


def test_fronteira_de_palavra():
    assert norm("vinculados") == "vinculados"
    assert norm("revisão manualmente") == "revisão manualmente"
    assert norm("não participou") == "não participou"
    assert norm("linkages") == "linkages"


def test_presenca_ausencia_e_contagem():
    texto = "Dois vínculos, três pares verdadeiros e um não par."
    assert contem(texto, "dois pares verdadeiros")
    assert contem(texto, "um não vínculo")
    assert not contem(texto, "quatro vínculos")
    assert conta(texto, "vínculos") == 2  # "vínculos" e "pares verdadeiros" são a mesma classe
    assert conta(texto, "par verdadeiro") == 2
    assert conta(texto, "não vínculo") == 1
    assert conta("de dois a quatro vínculos a mais; de dois a quatro pares verdadeiros a mais",
                 "de dois a quatro vínculos a mais") == 2


def test_idempotencia_e_numeros_preservados():
    base = r"Nos 211 vínculos (85{,}4\%) da \textit{grey area} (área cinza)."
    assert norm(norm(base)) == norm(base)
    assert "211" in norm(base) and "85,4%" in norm(base)
