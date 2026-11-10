"""Proteção dos números das tabelas da tese (tests/data/tabelas_tese.yaml).

Tabelas ficam fora da reescrita por LLM e só mudam quando o gerador muda:

(a) toda tabela que a árvore de ``tese.tex`` inclui está no manifesto, e todo item do manifesto
    existe em ``text/latex/tables/``;
(b) tabela ``sha256`` (sem teste que regenere o arquivo inteiro) tem de manter o hash do
    manifesto; mudança exige aprovação da autora e atualização do hash;
(c) tabela ``regeneracao`` aponta para um teste que existe e que cita a tabela (ou o gerador);
(d) o manifesto não tem duplicatas.

O hash é do conteúdo com quebras de linha normalizadas para LF.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from scripts import linter_prosa as lp

REPO = lp.REPO
TESTES = REPO / "tests"
SCRIPTS = REPO / "scripts"
PASTA_TABELAS = REPO / "text" / "latex" / "tables"
PREFIXO_TABELAS = "text/latex/tables/"
MANIFESTO = TESTES / "data" / "tabelas_tese.yaml"
PROTECOES = ("regeneracao", "sha256")
RE_SHA256 = re.compile(r"[0-9a-f]{64}")


def carregar_manifesto() -> list[dict]:
    dados = yaml.safe_load(MANIFESTO.read_text(encoding="utf-8"))
    return dados["tabelas"]


def sha256_lf(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def tabelas_da_arvore() -> set[str]:
    """Caminhos (relativos a text/latex/tables/) das tabelas que a árvore de tese.tex inclui."""
    arvore = lp.carregar_arvore(lp.TESE_PADRAO, base=REPO)
    return {
        arq.rel[len(PREFIXO_TABELAS):]
        for arq in arvore.arquivos.values()
        if arq.rel.startswith(PREFIXO_TABELAS)
    }


MANIFESTO_ITENS = carregar_manifesto()
ITENS_SHA256 = [e for e in MANIFESTO_ITENS if e.get("protecao") == "sha256"]
ITENS_REGENERACAO = [e for e in MANIFESTO_ITENS if e.get("protecao") == "regeneracao"]


def _id(entrada: dict) -> str:
    return entrada["arquivo"]


# ---------------------------------------------------------------------------
# (a) cobertura
# ---------------------------------------------------------------------------


def test_toda_tabela_incluida_esta_no_manifesto():
    no_manifesto = {e["arquivo"] for e in MANIFESTO_ITENS}
    incluidas = tabelas_da_arvore()
    assert incluidas, "a árvore de tese.tex não incluiu nenhuma tabela de text/latex/tables/"
    faltam = sorted(incluidas - no_manifesto)
    assert not faltam, (
        "tabelas incluídas por tese.tex e ausentes de tests/data/tabelas_tese.yaml "
        "(classifique como regeneracao ou sha256): " + ", ".join(faltam)
    )


def test_todo_item_do_manifesto_existe():
    ausentes = sorted(e["arquivo"] for e in MANIFESTO_ITENS if not (PASTA_TABELAS / e["arquivo"]).is_file())
    assert not ausentes, "itens do manifesto sem arquivo em text/latex/tables/: " + ", ".join(ausentes)


def test_manifesto_bem_formado():
    erros: list[str] = []
    for e in MANIFESTO_ITENS:
        nome = e.get("arquivo", "<sem arquivo>")
        if e.get("protecao") not in PROTECOES:
            erros.append(f"{nome}: protecao deve ser uma de {PROTECOES}, veio {e.get('protecao')!r}")
        elif e["protecao"] == "regeneracao":
            if not e.get("teste"):
                erros.append(f"{nome}: protecao regeneracao exige 'teste'")
        else:
            if not RE_SHA256.fullmatch(str(e.get("sha256", ""))):
                erros.append(f"{nome}: 'sha256' ausente ou inválido (64 hexadecimais minúsculos)")
            if not str(e.get("motivo", "")).strip():
                erros.append(f"{nome}: protecao sha256 exige 'motivo'")
    assert not erros, "\n".join(erros)


# ---------------------------------------------------------------------------
# (b) tabelas travadas por hash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entrada", ITENS_SHA256, ids=_id)
def test_tabela_sha256_nao_mudou(entrada):
    caminho = PASTA_TABELAS / entrada["arquivo"]
    assert caminho.is_file(), f"tabela ausente: {caminho}"
    atual = sha256_lf(caminho)
    assert atual == entrada["sha256"], (
        f"{entrada['arquivo']} mudou sem gerador (hash {atual} != manifesto {entrada['sha256']}). "
        "Tabelas sem teste de regeneração não são editadas à mão nem por reescrita de texto: "
        "a autora precisa aprovar a mudança e atualizar o hash em tests/data/tabelas_tese.yaml."
    )


# ---------------------------------------------------------------------------
# (c) tabelas regeneradas por teste
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entrada", ITENS_REGENERACAO, ids=_id)
def test_tabela_regenerada_tem_teste_que_a_cita(entrada):
    arquivo_teste, _, no = entrada["teste"].partition("::")
    caminho = TESTES / arquivo_teste
    assert caminho.is_file(), f"{entrada['arquivo']}: arquivo de teste inexistente: tests/{arquivo_teste}"
    texto = caminho.read_text(encoding="utf-8")
    if no:
        assert re.search(rf"(?m)^\s*def {re.escape(no)}\(", texto), (
            f"{entrada['arquivo']}: tests/{arquivo_teste} não define {no}"
        )
    basename = Path(entrada["arquivo"]).stem
    cita_tabela = re.search(rf"(?<![A-Za-z0-9_]){re.escape(basename)}(?![A-Za-z0-9_])", texto)
    gerador = entrada.get("gerador")
    if gerador:
        assert (SCRIPTS / f"{gerador}.py").is_file(), f"{entrada['arquivo']}: scripts/{gerador}.py inexistente"
        cita_gerador = re.search(rf"(?<![A-Za-z0-9_]){re.escape(gerador)}(?![A-Za-z0-9_])", texto)
        assert cita_tabela or cita_gerador, (
            f"{entrada['arquivo']}: tests/{arquivo_teste} não cita a tabela nem o gerador {gerador}"
        )
    else:
        assert cita_tabela, (
            f"{entrada['arquivo']}: tests/{arquivo_teste} não cita {basename}; "
            "se o teste regenera em laço, informe 'gerador' no manifesto"
        )


# ---------------------------------------------------------------------------
# (d) duplicatas
# ---------------------------------------------------------------------------


def test_manifesto_sem_duplicatas():
    contagem = Counter(e["arquivo"] for e in MANIFESTO_ITENS)
    repetidas = sorted(nome for nome, n in contagem.items() if n > 1)
    assert not repetidas, "itens duplicados em tests/data/tabelas_tese.yaml: " + ", ".join(repetidas)
