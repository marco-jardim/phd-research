from __future__ import annotations

import os

import pytest
from _limite_cpu import aplicar_limite_de_cpu

# Limite de CPU da suite (ver tests/_limite_cpu.py e README): aplicado já na importação
# deste conftest, que cada worker do xdist também carrega, antes dos módulos de teste.
aplicar_limite_de_cpu()

# Exceção temporária autorizada pela autora em 2026-11-10 ("eu ainda vou mudar o deck de
# apresentação, então este deck não está válido mesmo; pode fazer uma exceção momentaneamente
# para o deck"). Os testes marcados com ``deck_qualificacao`` conferem o conteúdo do deck da
# qualificação (text/slides/defesa.tex, text/slides/roteiro_defesa.md, text/slides/figures/*)
# e ficam pulados enquanto o deck é reescrito; nenhuma asserção foi apagada nem alterada.
# Interruptor único: DECK_QUALIFICACAO=1 reativa estes testes e reinclui text/slides/ no portão
# de fidelidade (scripts/portao_fidelidade.py, _deck_em_excecao). Para encerrar a exceção, remover
# este bloco, os marcadores ``deck_qualificacao`` e o bloco correspondente do portão.
MOTIVO_DECK_QUALIFICACAO = (
    "Exceção temporária autorizada pela autora em 2026-11-10: deck da qualificação em "
    "reescrita; reativar com DECK_QUALIFICACAO=1 ou removendo a exceção"
)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DECK_QUALIFICACAO") == "1":
        return
    pular = pytest.mark.skip(reason=MOTIVO_DECK_QUALIFICACAO)
    for item in items:
        if item.get_closest_marker("deck_qualificacao") is not None:
            item.add_marker(pular)
