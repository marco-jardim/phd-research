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
"""

from __future__ import annotations

import re
import unicodedata

PAR = "⟦par⟧"
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

_NOMES = {
    "c%d" % i: marcador for i, (marcador, _) in enumerate(_CLASSES)
}
_PADRAO = re.compile(
    # ⟦ e ⟧ delimitam os marcadores: excluí-los das fronteiras torna a normalização idempotente.
    r"(?<![\w⟦])(?:"
    + "|".join(
        "(?P<c%d>%s)" % (i, "|".join(alts)) for i, (_, alts) in enumerate(_CLASSES)
    )
    + r")(?![\w⟧])"
)
_ESTILO = re.compile(r"\\(?:textit|emph|textsc)\{([^{}]*)\}")


def _trivia(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    anterior = None
    while anterior != texto:
        anterior = texto
        texto = _ESTILO.sub(r"\1", texto)
    texto = texto.replace("{,}", ",").replace("\\%", "%").replace("~", " ").replace("*", "")
    return " ".join(texto.casefold().split())


def normalizar_terminologia(texto: str) -> str:
    """Normaliza trivia de LaTeX e troca cada forma equivalente pelo marcador da classe."""
    return _PADRAO.sub(lambda m: _NOMES[m.lastgroup], _trivia(texto))


def contem(texto: str, literal: str) -> bool:
    """Presença do literal no texto, aceitando as formas equivalentes."""
    return normalizar_terminologia(literal) in normalizar_terminologia(texto)


def conta(texto: str, literal: str) -> int:
    """Número de ocorrências (sem sobreposição) do literal, aceitando as formas equivalentes."""
    return normalizar_terminologia(texto).count(normalizar_terminologia(literal))
