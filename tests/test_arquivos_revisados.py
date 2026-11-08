"""Porta de lint para os arquivos já revisados (tests/data/arquivos_revisados.yaml).

Um capítulo listado tem de estar sem erros R02 (formas proibidas), R03 (glosa de primeira
menção), R04 (linkage fora de itálico) e R08 (nomes oficiais dos sistemas).
"""

import pytest

import terminologia
from scripts import linter_prosa as lp

REGRAS_EXIGIDAS = ("R02", "R03", "R04", "R08")
REVISADOS = terminologia.arquivos_revisados()


def erros_exigidos(caminho):
    arvore = lp.carregar_arvore(caminho)
    return [
        v for v in lp.lintar(arvore, regras=REGRAS_EXIGIDAS)
        if v.severidade == "erro"
    ]


def test_lista_de_revisados_aponta_para_arquivos_existentes():
    ausentes = [str(c) for c in REVISADOS if not c.is_file()]
    assert not ausentes, ausentes


@pytest.mark.parametrize("caminho", REVISADOS, ids=lambda c: c.name)
def test_arquivo_revisado_sem_erros_de_terminologia(caminho):
    erros = erros_exigidos(caminho)
    assert not erros, "\n".join(
        f"{v.regra} {v.arquivo}:{v.linha}: {v.mensagem} [{v.trecho}]" for v in erros
    )


def test_porta_detecta_forma_proibida(tmp_path):
    ruim = tmp_path / "ruim.tex"
    ruim.write_text("\\chapter{Teste}\n\nUm não vínculo foi observado no SINAN.\n", encoding="utf-8")
    regras = {v.regra for v in erros_exigidos(ruim)}
    assert "R02" in regras
