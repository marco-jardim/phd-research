"""Normalização terminológica para os pinos de texto da tese (guia de escrita, seção 3.6).

Os testes que fixam trechos do texto usam `contem`/`conta` para aceitar, ao mesmo tempo,
a redação anterior e a canônica. Cada classe de formas equivalentes vira um único
marcador; o texto e o literal passam pela mesma normalização, então a comparação é simétrica.

Classes (formas antigas e novas, singular e plural, mesmo marcador):
- PAR: "vínculo(s)", "par(es) verdadeiro(s)" (a classe do par segundo a referência;
  "vínculo" continua válido para o vínculo aceito).
- NAO_PAR: "não vínculo(s)", "não-vínculo(s)", "não par(es)", "não-par(es)".
- GREY_AREA: "área cinza", "\\textit{grey area}", "grey area (área cinza)" (glosa).
- REVISAO_MANUAL: "revisão clerical", "revisão manual" ("de origem" permanece como texto).
- VPP: "precisão" (no sentido de métrica), "VPP", "valor preditivo positivo (VPP)".
- LINKAGE: "relacionamento de registros", "vinculação de registros", "record linkage",
  "\\textit{linkage}", "linkage (relacionamento de registros)".
- SINAN: "SINAN", "Sinan".
- SIH_SUS: "SIH-SUS", "SIH/SUS".

"zona cinzenta" não pertence a nenhuma classe: continua proibida e passa intacta.

Dois modos, escolhidos pelo argumento opcional `arquivo` de `contem`/`conta`/`normalizar_terminologia`:
- transicional (padrão, e arquivos fora de tests/data/arquivos_revisados.yaml): as classes acima.
- estrito (arquivo listado em arquivos_revisados.yaml): só as formas canônicas viram marcador;
  as formas antigas passam intactas ("não vínculo(s)", "não-par", "revisão clerical",
  "relacionamento de registros", "vinculação (de registros)", "record linkage", "precisão", "SIH-SUS"),
  e "vínculo(s)" (VINCULO) deixa de ser equivalente a "par(es) verdadeiro(s)" (PAR).
  Um capítulo entra na lista quando a sua revisão é encerrada.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ARQUIVOS_REVISADOS_YAML = Path(__file__).resolve().parent / "data" / "arquivos_revisados.yaml"

PAR = "⟦par⟧"
VINCULO = "⟦vinculo⟧"  # só no modo estrito: o vínculo aceito, distinto do par verdadeiro
NAO_PAR = "⟦nao_par⟧"
GREY_AREA = "⟦grey_area⟧"
REVISAO_MANUAL = "⟦revisao_manual⟧"
VPP = "⟦vpp⟧"
LINKAGE = "⟦linkage⟧"
SINAN = "⟦sinan⟧"
SIH_SUS = "⟦sih_sus⟧"

# Cada classe: (marcador, alternativas de regex já em minúsculas, da mais longa para a mais curta).
_CLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (NAO_PAR, (r"não[ -]vínculos?", r"não[ -]par(?:es)?")),
    (PAR, (r"par(?:es)? verdadeiros?", r"vínculos?")),
    (
        GREY_AREA,
        (
            r"grey areas \(áreas cinza\)",
            r"grey area \(área cinza\)",
            r"grey areas",
            r"grey area",
            r"áreas cinza",
            r"área cinza",
        ),
    ),
    (
        REVISAO_MANUAL,
        (r"revisões clericais", r"revisões manuais", r"revisão clerical", r"revisão manual"),
    ),
    (VPP, (r"valor preditivo positivo \(vpp\)", r"valor preditivo positivo", r"precisão", r"vpp")),
    (
        LINKAGE,
        (
            r"linkage \(relacionamento de registros\)",
            r"relacionamento de registros",
            r"vinculação de registros",
            r"record linkage",
            r"vinculação",
            r"linkage",
        ),
    ),
    (SINAN, (r"sinan",)),
    (SIH_SUS, (r"sih[-/]sus",)),
)

# Modo estrito (arquivos já revisados, ver tests/data/arquivos_revisados.yaml): só as formas
# canônicas viram marcador. "vínculo" (vínculo aceito) e "par verdadeiro" (classe segundo a
# referência) têm marcadores distintos. Marcador `None` = forma proibida: passa intacta, de modo
# que o literal antigo só casa com o texto antigo e o pino falha se o texto o reintroduzir.
_CLASSES_ESTRITO: tuple[tuple[str | None, tuple[str, ...]], ...] = (
    (
        None,
        (
            r"não[ -]vínculos?",
            r"não-par(?:es)?",
            r"revisões clericais",
            r"revisão clerical",
            r"relacionamento de registros",
            r"vinculação de registros",
            r"record linkage",
            r"vinculação",
            r"precisão",
            r"sih-sus",
        ),
    ),
    (NAO_PAR, (r"não par(?:es)?",)),
    (PAR, (r"par(?:es)? verdadeiros?",)),
    (VINCULO, (r"vínculos?",)),
    (
        GREY_AREA,
        (
            r"grey areas \(áreas cinza\)",
            r"grey area \(área cinza\)",
            r"grey areas",
            r"grey area",
            r"áreas cinza",
            r"área cinza",
        ),
    ),
    (REVISAO_MANUAL, (r"revisões manuais", r"revisão manual")),
    (VPP, (r"valor preditivo positivo \(vpp\)", r"valor preditivo positivo", r"vpp")),
    (LINKAGE, (r"linkage \(relacionamento de registros\)", r"linkage")),
    (SINAN, (r"sinan",)),
    (SIH_SUS, (r"sih/sus",)),
)


def _compilar(classes) -> tuple[re.Pattern, dict[str, str | None]]:
    nomes = {"c%d" % i: marcador for i, (marcador, _) in enumerate(classes)}
    padrao = re.compile(
        # ⟦ e ⟧ delimitam os marcadores: excluí-los das fronteiras torna a normalização idempotente.
        r"(?<![\w⟦])(?:"
        + "|".join("(?P<c%d>%s)" % (i, "|".join(alts)) for i, (_, alts) in enumerate(classes))
        + r")(?![\w⟧])"
    )
    return padrao, nomes


_PADRAO, _NOMES = _compilar(_CLASSES)
_PADRAO_ESTRITO, _NOMES_ESTRITO = _compilar(_CLASSES_ESTRITO)
_ESTILO = re.compile(r"\\(?:textit|emph|textsc)\{([^{}]*)\}")


def _trivia(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    anterior = None
    while anterior != texto:
        anterior = texto
        texto = _ESTILO.sub(r"\1", texto)
    texto = texto.replace("{,}", ",").replace("\\%", "%").replace("~", " ").replace("*", "")
    return " ".join(texto.casefold().split())


@functools.lru_cache(maxsize=None)
def _revisados(caminho: Path) -> frozenset[Path]:
    dados = yaml.safe_load(caminho.read_text(encoding="utf-8")) or []
    if not isinstance(dados, list):
        raise ValueError(f"{caminho}: esperada uma lista de caminhos")
    # Chaves "arquivo#secao" (ex.: tese.tex#resumo) valem só para o registro de invariantes
    # (scripts/registro_invariantes.py); não são arquivos nem ativam o modo estrito.
    return frozenset(
        (REPO / str(item)).resolve() for item in dados if "#" not in str(item)
    )


def arquivos_revisados() -> list[Path]:
    """Caminhos absolutos dos arquivos da lista `tests/data/arquivos_revisados.yaml`."""
    return sorted(_revisados(ARQUIVOS_REVISADOS_YAML))


def modo_estrito(arquivo: str | Path | None) -> bool:
    """True quando `arquivo` consta da lista de arquivos revisados (caminho relativo à raiz do repositório)."""
    if arquivo is None:
        return False
    return (REPO / Path(arquivo)).resolve() in _revisados(ARQUIVOS_REVISADOS_YAML)


def normalizar_terminologia(texto: str, arquivo: str | Path | None = None) -> str:
    """Normaliza trivia de LaTeX e troca cada forma equivalente pelo marcador da classe.

    Sem `arquivo`, ou com arquivo fora da lista de revisados, vale a equivalência transicional.
    Com arquivo revisado, só as formas canônicas viram marcador (modo estrito).
    """
    if modo_estrito(arquivo):
        padrao, nomes = _PADRAO_ESTRITO, _NOMES_ESTRITO
    else:
        padrao, nomes = _PADRAO, _NOMES
    return padrao.sub(lambda m: nomes[m.lastgroup] or m.group(0), _trivia(texto))


def contem(texto: str, literal: str, arquivo: str | Path | None = None) -> bool:
    """Presença do literal no texto, aceitando as formas equivalentes do modo do `arquivo`."""
    return normalizar_terminologia(literal, arquivo) in normalizar_terminologia(texto, arquivo)


def conta(texto: str, literal: str, arquivo: str | Path | None = None) -> int:
    """Número de ocorrências (sem sobreposição) do literal, aceitando as formas equivalentes do modo do `arquivo`."""
    return normalizar_terminologia(texto, arquivo).count(normalizar_terminologia(literal, arquivo))
