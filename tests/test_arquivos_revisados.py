"""Porta de lint para os arquivos já revisados (tests/data/arquivos_revisados.yaml).

Um capítulo listado tem de estar sem erros R02 (formas proibidas), R03 (glosa de primeira
menção), R04 (linkage fora de itálico) e R08 (nomes oficiais dos sistemas).

A regra R03 depende da ordem de leitura da tese (Random Forest é glosado uma vez, na primeira
menção de toda a tese). Por isso a árvore inteira, a partir de `tese.tex`, é lintada uma única
vez por módulo, e as violações são filtradas pelo arquivo revisado.
"""

import pytest

import terminologia
from scripts import linter_prosa as lp

REGRAS_EXIGIDAS = ("R02", "R03", "R04", "R08")
REVISADOS = terminologia.arquivos_revisados()


@pytest.fixture(scope="module")
def arvore_da_tese():
    return lp.carregar_arvore(lp.TESE_PADRAO)


@pytest.fixture(scope="module")
def violacoes_da_tese(arvore_da_tese):
    """Lint da árvore inteira, uma vez por módulo."""
    return lp.lintar(arvore_da_tese, regras=REGRAS_EXIGIDAS)


def erros_do_arquivo(violacoes, caminho):
    rel = lp._relativo(caminho, lp.REPO.resolve())
    return [v for v in violacoes if v.severidade == "erro" and v.arquivo == rel]


def erros_exigidos(caminho):
    """Lint isolado de um arquivo avulso (fora da tese), para testar a própria porta."""
    arvore = lp.carregar_arvore(caminho)
    return [
        v for v in lp.lintar(arvore, regras=REGRAS_EXIGIDAS)
        if v.severidade == "erro"
    ]


def test_lista_de_revisados_aponta_para_arquivos_existentes():
    ausentes = [str(c) for c in REVISADOS if not c.is_file()]
    assert not ausentes, ausentes


@pytest.mark.parametrize("caminho", REVISADOS, ids=lambda c: c.name)
def test_arquivo_revisado_sem_erros_de_terminologia(caminho, arvore_da_tese, violacoes_da_tese):
    assert str(caminho) in arvore_da_tese.arquivos, f"{caminho} não é incluído por tese.tex"
    erros = erros_do_arquivo(violacoes_da_tese, caminho)
    assert not erros, "\n".join(
        f"{v.regra} {v.arquivo}:{v.linha}: {v.mensagem} [{v.trecho}]" for v in erros
    )


def test_porta_detecta_forma_proibida(tmp_path):
    ruim = tmp_path / "ruim.tex"
    ruim.write_text("\\chapter{Teste}\n\nUm não vínculo foi observado no SINAN.\n", encoding="utf-8")
    regras = {v.regra for v in erros_exigidos(ruim)}
    assert "R02" in regras


def _tese_com_dois_capitulos(tmp_path, primeiro, segundo):
    (tmp_path / "a.tex").write_text(primeiro, encoding="utf-8")
    (tmp_path / "b.tex").write_text(segundo, encoding="utf-8")
    raiz = tmp_path / "tese.tex"
    raiz.write_text("\\input{a}\n\\input{b}\n", encoding="utf-8")
    return lp.carregar_arvore(raiz, base=tmp_path)


def test_random_forest_glosado_em_outro_capitulo_nao_gera_r03(tmp_path):
    arvore = _tese_com_dois_capitulos(
        tmp_path,
        "\\chapter{Um}\n\nUsa-se Random Forest (floresta aleatória) como classificador.\n",
        "\\chapter{Dois}\n\nO Random Forest foi treinado de novo.\n",
    )
    violacoes = lp.lintar(arvore, regras=("R03",))
    assert [v for v in violacoes if v.arquivo == "b.tex"] == []
    assert [v for v in violacoes if v.arquivo == "a.tex"] == []

    # Lintado isoladamente, o segundo capítulo vira "primeira menção" e gera o R03 falso.
    isolado = lp.lintar(lp.carregar_arvore(tmp_path / "b.tex", base=tmp_path), regras=("R03",))
    assert any("Random Forest" in v.mensagem for v in isolado)


def test_random_forest_sem_glosa_em_toda_a_tese_gera_r03_so_na_primeira_menção(tmp_path):
    arvore = _tese_com_dois_capitulos(
        tmp_path,
        "\\chapter{Um}\n\nUsa-se Random Forest como classificador.\n",
        "\\chapter{Dois}\n\nO Random Forest foi treinado de novo.\n",
    )
    violacoes = [v for v in lp.lintar(arvore, regras=("R03",)) if "Random Forest" in v.mensagem]
    assert [v.arquivo for v in violacoes] == ["a.tex"]


def test_filtro_por_arquivo_usa_o_caminho_relativo_a_raiz_do_repositorio():
    caminho = lp.TESE_PADRAO.parent / "estudo4_gzcmd.tex"
    assert lp._relativo(caminho, lp.REPO.resolve()) == "text/latex/estudo4_gzcmd.tex"
