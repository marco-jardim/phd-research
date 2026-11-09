"""Mapeamento único dos rótulos de apresentação do arranjo sequencial recalibrado.

No registro interno (``8c16a41``), nos dados e nos identificadores de scripts o arranjo se chama ``P``.
Na camada de apresentação (texto da tese, tabelas editoriais, figura, pinos de teste) o nome passa a ser
``ASR`` (decisão da autora de 2026-11-08, ``docs/plans/PLANO_RENOMEAR_P.md``). Este módulo é a única fonte do
mapeamento: ``scripts/renomear_sigla.py`` o importa, e os geradores o importarão no lote 3 do plano.

Nada aqui altera dados, chaves de seleção (``braco == "P"``), ``\\label``/``\\ref``, caminhos ou tabelas
históricas: só o rótulo exibido ao leitor.
"""

from __future__ import annotations

NOME_ANTIGO = "P"
NOME_NOVO = "ASR"

# Rótulos em prosa, legenda, célula, ``\item[...]`` e título.
ROTULO_BRACO: dict[str, str] = {
    "P": "ASR",
    "P-Platt": "ASR-Platt",
    "P-PAR2-grey": "ASR-PAR2-faixa",  # decisão da autora de 2026-11-09: "grey" não designa a faixa [5,9)
    "P_MATCH_ao_humano": r"ASR, \texttt{MATCH} ao humano",
}

# Forma em modo matemático (``$P\cup B1$`` vira ``$\mathrm{ASR}\cup B1$``).
ROTULO_MATH: dict[str, str] = {"P": r"\mathrm{ASR}"}
ROTULO_UNIAO_B1_MATH = r"$\mathrm{ASR}\cup B1$"

# Tokens que a renomeação reconhece (variantes A1 a A3; A5 troca só o P inicial).
TOKENS_ALVO: tuple[str, ...] = ("P", "P-Platt", "P-PAR2-grey")


def rotulo_braco(chave: str) -> str:
    """Rótulo de apresentação de uma chave de ``ROTULO_BRACO`` (``P``, ``P-Platt``, ...)."""
    return ROTULO_BRACO[chave]


def rotulo_math(chave: str = "P") -> str:
    """Rótulo em modo matemático."""
    return ROTULO_MATH[chave]


def substituicao_token(token: str, em_math: bool = False) -> str:
    """Texto que substitui ``token`` na camada de apresentação.

    Em modo matemático só ``P`` tem forma própria; outra variante em math é erro de classificação.
    """
    if em_math:
        if token not in ROTULO_MATH:
            raise ValueError(f"sem forma matemática para o token {token!r}")
        return ROTULO_MATH[token]
    if token not in TOKENS_ALVO:
        raise ValueError(f"token fora do alvo da renomeação: {token!r}")
    return ROTULO_BRACO[token]
