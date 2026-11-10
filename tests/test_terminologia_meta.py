"""Meta-teste dos pinos de terminologia (guia de escrita, seção 3.6).

O modo estrito de `tests/terminologia.py` só vale quando `contem`, `conta` ou
`normalizar_terminologia` recebem `arquivo=`. Sem isso, um capítulo listado em
`tests/data/arquivos_revisados.yaml` continuaria aceitando as formas antigas. Este teste
varre `tests/test_*.py` e falha se alguma chamada vier sem o argumento.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import terminologia

TESTS = Path(__file__).resolve().parent
HELPERS = ("contem", "conta", "normalizar_terminologia")
# test_terminologia.py testa os dois modos do próprio módulo e chama os helpers sem arquivo de propósito.
ISENTOS = {"test_terminologia.py", Path(__file__).name}


def _nomes_locais(arvore: ast.AST) -> set[str]:
    """Nomes sob os quais os helpers são visíveis no módulo (inclui `import ... as ...`)."""
    nomes: set[str] = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.ImportFrom) and no.module in ("terminologia", "tests.terminologia"):
            for alias in no.names:
                if alias.name in HELPERS:
                    nomes.add(alias.asname or alias.name)
    return nomes


def _modulos_locais(arvore: ast.AST) -> set[str]:
    """Nomes sob os quais o módulo `terminologia` é visível."""
    modulos: set[str] = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            for alias in no.names:
                if alias.name in ("terminologia", "tests.terminologia"):
                    modulos.add(alias.asname or alias.name.split(".")[-1])
    return modulos


def chamadas_sem_arquivo(fonte: str) -> list[tuple[int, str]]:
    """(linha, nome) de cada chamada a um helper de terminologia sem `arquivo=` (ou com `arquivo=None`)."""
    arvore = ast.parse(fonte)
    nomes = _nomes_locais(arvore)
    modulos = _modulos_locais(arvore)
    achados = []
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Call):
            continue
        funcao = no.func
        if isinstance(funcao, ast.Name) and funcao.id in nomes:
            nome = funcao.id
        elif (isinstance(funcao, ast.Attribute) and funcao.attr in HELPERS
              and isinstance(funcao.value, ast.Name) and funcao.value.id in modulos):
            nome = funcao.attr
        else:
            continue
        argumento = next((k for k in no.keywords if k.arg == "arquivo"), None)
        vazio = argumento is None or (
            isinstance(argumento.value, ast.Constant) and argumento.value.value is None
        )
        if vazio:
            achados.append((no.lineno, nome))
    return sorted(achados)


def _arquivos_de_teste() -> list[Path]:
    return sorted(p for p in TESTS.glob("test_*.py") if p.name not in ISENTOS)


def test_ha_pinos_de_terminologia_para_varrer():
    total = sum(
        len(_nomes_locais(ast.parse(p.read_text(encoding="utf-8"))))
        + len(_modulos_locais(ast.parse(p.read_text(encoding="utf-8"))))
        for p in _arquivos_de_teste()
    )
    assert total > 0


@pytest.mark.parametrize("caminho", _arquivos_de_teste(), ids=lambda p: p.name)
def test_todo_pino_de_terminologia_passa_arquivo(caminho):
    achados = chamadas_sem_arquivo(caminho.read_text(encoding="utf-8"))
    assert not achados, "\n".join(
        f"{caminho.name}:{linha}: {nome}() sem arquivo=" for linha, nome in achados
    )


def test_varredura_detecta_chamada_sem_arquivo():
    fonte = (
        "from terminologia import contem, conta as c, normalizar_terminologia\n"
        "import terminologia as t\n"
        "contem('a', 'b')\n"
        "c('a', 'b')\n"
        "normalizar_terminologia('a', arquivo=None)\n"
        "t.contem('a', 'b')\n"
        "contem('a', 'b', arquivo='text/latex/x.tex')\n"
        "t.conta('a', 'b', arquivo='text/latex/x.tex')\n"
        "ri.contem('a', 'b')\n"
    )
    assert chamadas_sem_arquivo(fonte) == [
        (3, "contem"), (4, "c"), (5, "normalizar_terminologia"), (6, "contem"),
    ]


def test_arquivo_revisado_deixa_de_aceitar_a_forma_antiga(monkeypatch):
    revisado = "text/latex/x_revisado.tex"
    livre = "text/latex/x_livre.tex"
    monkeypatch.setattr(
        terminologia,
        "_revisados",
        lambda _caminho: frozenset({(terminologia.REPO / revisado).resolve()}),
    )
    assert terminologia.modo_estrito(revisado)
    assert not terminologia.modo_estrito(livre)

    pares = [
        ("há não vínculos", "há não par"),
        ("há revisão clerical", "há revisão manual"),
        ("há relacionamento de registros", "há linkage"),
        ("precisão de 0,9", "VPP de 0,9"),
        ("dois vínculos", "dois pares verdadeiros"),
    ]
    for antiga_ou_equivalente, canonica in pares:
        # Transicional (arquivo fora da lista): as duas formas são equivalentes.
        assert terminologia.contem(antiga_ou_equivalente, canonica, arquivo=livre), (antiga_ou_equivalente, canonica)
        assert terminologia.contem(canonica, antiga_ou_equivalente, arquivo=livre), (canonica, antiga_ou_equivalente)
        # Estrito (arquivo revisado): a forma antiga deixa de ser aceita nos dois sentidos.
        assert not terminologia.contem(antiga_ou_equivalente, canonica, arquivo=revisado), (antiga_ou_equivalente, canonica)
        assert not terminologia.contem(canonica, antiga_ou_equivalente, arquivo=revisado), (canonica, antiga_ou_equivalente)

    # A forma canônica continua casando consigo mesma no modo estrito.
    assert terminologia.contem("há não par aqui", "não par", arquivo=revisado)
    assert terminologia.conta("um vínculo e outro vínculo", "vínculo", arquivo=revisado) == 2
