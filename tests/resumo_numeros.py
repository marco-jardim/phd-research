"""Contagem de números de RESULTADO no Resumo e no Abstract de tese.tex.

Decisão da autora de 2026-11-08: no Resumo e no Abstract, as âncoras do registro prevalecem sobre o teto
antigo de 18 números (qualquer número). O teto passa a ser 24, aplicado só a números de resultado.

Não contam: identificadores (O1 a O4, F1 a F4, K1 a K3, S1 a S3, V1 a V4, B1, CM, P-Platt; PAR~=~1/2;
v3, v3.1, v4, v5; "passo 6/9/11" e "step 11" quando o dígito rotula o passo), anos (19xx/20xx), o nível do
IC ("IC de 95%", "95% CI", "intervalo de confiança de 95%", "95% confidence interval") e números por extenso.
"""
import re

TETO_NUMEROS_RESULTADO = 24

_NUMERO = r'\d+(?:[.,]\d+)*'
_NIVEL_IC = r'95\\?%'
_EXCLUSOES = (
    # nível do IC, nas duas ordens (PT e EN)
    r'(?:IC|CI|intervalo\s+de\s+confiança|confidence\s+interval)\s*(?:\([^)]*\)\s*)?(?:de\s+)?' + _NIVEL_IC,
    _NIVEL_IC + r'\s*(?:IC|CI|confidence\s+interval|intervalo\s+de\s+confiança)',
    # PAR~=~2, PAR=1
    r'PAR\s*~?=~?\s*\d',
    # rótulo de passo (passo 11, step 11)
    r'(?:passos?|steps?)\s+\d+',
    # identificadores de fila, fusão, carga, estrato, variante, braço
    r'(?<![\w.])[OFKSVB][1-4](?![\w])',
    r'(?<![\w.])CM(?![\w])',
    # versões
    r'(?<![\w.])v\d+(?:[.,]\d+)*',
    # anos
    r'(?<!\d)(?:19|20)\d{2}(?!\d)',
)


def numeros_de_resultado(text):
    """Lista os números de resultado de `text`, depois de excluir identificadores, anos e nível do IC."""
    for pattern in _EXCLUSOES:
        text = re.sub(pattern, ' ', text)
    return re.findall(_NUMERO, text)
