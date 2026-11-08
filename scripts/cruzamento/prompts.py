"""Textos dos prompts do cruzamento (sistema em cache e blocos por requisição).

O bloco de sistema de cada etapa é fixo dentro de uma execução (âncoras, taxonomia e terminologia entram
nele), de modo que o prefixo em cache é idêntico byte a byte entre as requisições.
"""

from __future__ import annotations

import json

from .comum import (
    CONFLITOS_PINO,
    POLARIDADES,
    RELACOES,
    TERMINOLOGIA,
    TIPOS_AFIRMACAO,
    TIPOS_INCONSISTENCIA,
    TOPICOS,
    VEREDITOS,
    bloco_ancoras,
    numerar,
)

ESQUEMA_MAPEAR = (
    '{"relacoes":[{"ancora_id":"C01","relacao":"afirma","trecho":"...","linha":123,"justificativa":"..."}],'
    '"afirmacoes":[{"trecho":"...","linha":123,"tipo":"resultado","topicos":["P_vigilancia"],'
    '"numeros":["23"],"polaridade":"afirma","escopo":"vigilância, meta 23, abaixo do corte"}]}'
)


def secao_ficha(ficha: str, regra: str) -> str:
    """Seção do bloco de sistema com a ficha de valores conferidos (vazia se não há ficha)."""
    if not ficha:
        return ""
    return (
        "\nFICHA DE VALORES CONFERIDOS (números da tese recalculados dos artefatos; cada linha traz o id da "
        "ficha, a descrição, o nome do valor, o texto em português e a fonte):\n"
        f"{ficha}\n\n{regra}\n"
    )


REGRA_FICHA_MAPEAR = (
    "Um número do texto que divirja da ficha para a mesma grandeza e o mesmo escopo também é contradição: "
    'registre uma relação "contradiz" com a âncora mais próxima do assunto e inicie a justificativa com o id '
    'da ficha (ex.: "N10: o texto diz 214, a ficha diz 215"). Não aponte números de grandezas ou escopos '
    "diferentes."
)
REGRA_FICHA_CRUZAR = (
    "Um número de uma afirmação que divirja da ficha para a mesma grandeza e o mesmo escopo também é "
    "inconsistência (numero_divergente): aponte-a com outra afirmação que traga o valor da ficha ou, se não "
    "houver, com ela mesma (a e b iguais); a justificativa cita o id da ficha (ex.: N10)."
)
REGRA_FICHA_ADJUDICAR = (
    "Um número do trecho apontado que divirja da ficha para a mesma grandeza e o mesmo escopo é inconsistência "
    "real (CONFIRMADO), salvo se o contexto mostrar que se trata de outra grandeza ou de outro escopo; a "
    "justificativa cita o id da ficha (ex.: N10)."
)


def sistema_mapear(ancoras: list[dict], ficha: str = "") -> str:
    return f"""\
Você audita a coerência interna de uma tese de doutorado em Saúde Coletiva (linkage de registros de saúde, \
com revisão manual dos pares abaixo do corte), de seus slides e do roteiro da defesa. Cada requisição traz \
uma fatia do texto (LaTeX ou Markdown) com o número da linha do arquivo antes de cada linha ("  123| texto"). \
Você faz duas tarefas sobre a fatia.

TAREFA 1, relações com as âncoras (afirmações centrais da tese). Liste só as relações NÃO neutras. Uma \
relação liga um trecho da fatia a uma âncora:
- afirma: o trecho sustenta a âncora.
- contradiz: o trecho diz o oposto da âncora ou um número incompatível com ela.
- enfraquece: o trecho reduz ou relativiza a âncora sem contradizê-la.
- exagera: o trecho afirma mais do que a âncora permite (superlativo, generalização de escopo, certeza maior).
- omite_ressalva: o trecho enuncia o conteúdo da âncora sem a ressalva que a acompanha (use o campo RESSALVA).
Se a fatia não toca uma âncora, não a mencione.

ÂNCORAS:
{bloco_ancoras(ancoras)}

TAREFA 2, afirmações substantivas da fatia (resultados, cautelas, recomendações, escolhas de método, \
definições). Ignore transições, listas de capítulos e referências cruzadas sem conteúdo.
- tipo: {" | ".join(TIPOS_AFIRMACAO)}.
- topicos: de 1 a 3 itens da lista fechada abaixo; use "outro" só se nenhum servir.
- numeros: lista de strings, cada uma exatamente como aparece no trecho (ex.: "23", "12,4%", "0,87").
- polaridade: {" | ".join(POLARIDADES)}.
- escopo: descrição curta (até 12 palavras) de onde a afirmação vale (ex.: "vigilância, meta 23, abaixo do corte").

TAXONOMIA DE TÓPICOS (lista fechada): {", ".join(TOPICOS)}.

TERMINOLOGIA (só para interpretar o texto; não avalie estilo):
{TERMINOLOGIA}
{secao_ficha(ficha, REGRA_FICHA_MAPEAR)}
REGRAS
- "trecho" é uma cópia LITERAL e contígua do texto da fatia, sem o prefixo de número de linha, com uma ou duas \
frases (até cerca de 60 palavras). Não parafraseie, não corrija, não resuma; mantenha comandos LaTeX, til e \
aspas como estão. Trechos que não existirem literalmente na fatia serão marcados como inválidos.
- "linha" é o número da linha do arquivo em que o trecho começa.
- relacao: {" | ".join(RELACOES)}. ancora_id: um id da lista de âncoras. justificativa: uma frase curta.
- Responda apenas com um objeto JSON, sem cercas de código e sem texto antes ou depois, neste formato:
{ESQUEMA_MAPEAR}
- Sem relações ou sem afirmações: use listas vazias.
"""


def usuario_mapear(fatia: dict) -> str:
    return (
        f"Fatia {fatia['id']}\n"
        f"arquivo: {fatia['arquivo']} (linhas {fatia['linha_inicio']} a {fatia['linha_fim']})\n"
        f"seção: {fatia['secao']}\n\n"
        "Texto da fatia:\n"
        f"{numerar(fatia)}\n"
    )


def sistema_cruzar(ficha: str = "") -> str:
    return f"""\
Você confere a coerência entre afirmações de uma tese de doutorado em Saúde Coletiva (linkage de registros de \
saúde, com revisão manual dos pares abaixo do corte), de seus slides e do roteiro da defesa. Cada requisição \
traz afirmações de UM tópico, cada uma com rótulo (A1, A2, ... ou B1, B2, ...), arquivo:linha, tipo, polaridade, \
escopo, números e o trecho literal. Aponte pares de afirmações INCONSISTENTES entre si, de arquivos ou seções \
diferentes ou da mesma fatia. Tipos:
- numero_divergente: números diferentes para a mesma grandeza, no mesmo escopo.
- polaridade_oposta: uma afirma e a outra nega a mesma coisa.
- escopo_trocado: o mesmo resultado atribuído a escopos diferentes (população, passo, meta, regime, arranjo).
- definicao_divergente: o mesmo termo definido de formas incompatíveis.
- superlativo_sem_suporte: superlativo ou certeza que outro trecho não sustenta.
Afirmações que apenas tratam de aspectos diferentes, ou números de grandezas ou escopos diferentes, NÃO são \
inconsistência. Seja conservador: um outro modelo adjudicará cada par com os textos completos.

TERMINOLOGIA (só para interpretar):
{TERMINOLOGIA}
{secao_ficha(ficha, REGRA_FICHA_CRUZAR)}
Quando houver grupos A e B, aponte somente pares com um item de A e outro de B. Responda apenas com um objeto \
JSON, sem cercas de código e sem texto antes ou depois:
{{"pares":[{{"a":"A3","b":"B17","tipo":"{TIPOS_INCONSISTENCIA[0]}","justificativa":"frase curta"}}]}}
tipo: {" | ".join(TIPOS_INCONSISTENCIA)}. Sem inconsistências: {{"pares":[]}}.
"""


def _linha_afirmacao(rotulo: str, af: dict) -> str:
    numeros = "; ".join(af.get("numeros") or []) or "-"
    return (
        f"[{rotulo}] {af['arquivo']}:{af['linha']} | {af['tipo']} | {af['polaridade']} | "
        f"escopo: {af.get('escopo') or '-'} | números: {numeros} | \"{' '.join(af['trecho'].split())}\""
    )


def bloco_afirmacoes(grupo: str, afirmacoes: list[dict]) -> str:
    linhas = [_linha_afirmacao(f"{grupo}{i}", af) for i, af in enumerate(afirmacoes, 1)]
    return f"Grupo {grupo}:\n" + "\n".join(linhas) + "\n"


def usuario_cruzar_topico(topico: str) -> str:
    return f"Tópico: {topico}\n"


def sistema_adjudicar(ficha: str = "") -> str:
    return f"""\
Você é o adjudicador final de apontamentos de incoerência numa tese de doutorado em Saúde Coletiva (linkage \
de registros de saúde, com revisão manual dos pares abaixo do corte), nos slides e no roteiro da defesa. Cada \
requisição traz as fatias COMPLETAS (com número de linha) onde um modelo menor achou um problema e o \
apontamento: um par de trechos que ele julgou inconsistentes, ou uma relação de contradição ou exagero entre \
um trecho e uma âncora (afirmação central da tese). Leia o contexto completo e decida:
- CONFIRMADO: há inconsistência real que o autor precisa corrigir.
- FALSO_ALARME: o contexto completo resolve a aparente incoerência (escopos explícitos, grandezas diferentes, \
trecho mal lido).
- DEPENDE_DE_ESCOPO: os dois lados só são compatíveis se escopos diferentes forem pressupostos e o texto não \
os deixa explícitos ao leitor.
"correcao" é a correção mínima sugerida (até 40 palavras, indicando o arquivo:linha a alterar); vazia no \
FALSO_ALARME. "justificativa" tem uma ou duas frases.

TERMINOLOGIA (só para interpretar):
{TERMINOLOGIA}
{secao_ficha(ficha, REGRA_FICHA_ADJUDICAR)}
Responda apenas com um objeto JSON, sem cercas de código e sem texto antes ou depois:
{{"veredito":"{VEREDITOS[0]}","correcao":"...","justificativa":"..."}}
veredito: {" | ".join(VEREDITOS)}.
"""


def bloco_fatia(fatia: dict) -> str:
    return (
        f"Fatia {fatia['id']} | {fatia['arquivo']} (linhas {fatia['linha_inicio']} a {fatia['linha_fim']})\n"
        f"seção: {fatia['secao']}\n{numerar(fatia)}\n"
    )


def bloco_apontamento_par(par: dict) -> str:
    def lado(rotulo: str, item: dict) -> str:
        return f"{rotulo}: {item['arquivo']}:{item['linha']} \"{' '.join(item['trecho'].split())}\""

    return (
        f"APONTAMENTO (par, tópico {par['topico']}, tipo {par['tipo']})\n"
        f"{lado('Trecho 1', par['a'])}\n{lado('Trecho 2', par['b'])}\n"
        f"Justificativa do modelo menor: {par.get('justificativa') or '-'}\n"
    )


def bloco_apontamento_relacao(rel: dict, ancora: dict) -> str:
    ressalva = f"\nRessalva da âncora: {ancora['ressalva']}" if ancora.get("ressalva") else ""
    return (
        f"APONTAMENTO (relação {rel['relacao']} com a âncora {ancora['id']})\n"
        f"Âncora: {ancora['texto']}{ressalva}\n"
        f"Trecho: {rel['arquivo']}:{rel['linha']} \"{' '.join(rel['trecho'].split())}\"\n"
        f"Justificativa do modelo menor: {rel.get('justificativa') or '-'}\n"
    )


ESQUEMA_PINOS = (
    '{"pinos":[{"pino_id":"...","afirmacao_travada":"frase curta","duplica_ancora":null,'
    f'"conflito":"{CONFLITOS_PINO[0]}","ancora_conflito":null,"central_sem_ancora":false,'
    '"justificativa":"frase curta"}]}'
)
LITERAL_PINO_MAX = 400


def sistema_pinos(ancoras: list[dict]) -> str:
    return f"""\
Você revisa os "pinos" de uma tese de doutorado em Saúde Coletiva (linkage de registros de saúde, com revisão \
manual dos pares abaixo do corte). Um pino é uma asserção num teste automatizado que trava um trecho do texto \
da tese: o teste falha se a frase, a negação ou a ressalva mudar. Cada requisição traz uma lista de pinos, \
cada um com pino_id, arquivo do teste e linha, tipo, categoria (frase, negacao ou ressalva), alvo (arquivos da \
tese que o teste lê) e o literal travado. Para cada pino, decida:
- afirmacao_travada: a afirmação que o literal trava, numa frase curta.
- duplica_ancora: o id da âncora cujo conteúdo o pino repete, ou null.
- conflito: {" | ".join(CONFLITOS_PINO)}.
  - forte_demais: o literal trava uma formulação mais forte do que a âncora permite (superlativo, \
generalização de escopo, certeza maior).
  - contradiz_ancora: o literal trava o oposto da âncora ou um número incompatível com ela.
  - cautela_solta: o literal trava uma cautela ou ressalva que a âncora correspondente não traz, ou a \
âncora tem uma ressalva que nenhum pino protege.
  - nenhum: sem conflito.
- ancora_conflito: o id da âncora em conflito, ou null se conflito é nenhum.
- central_sem_ancora: true se o pino trava uma afirmação central da tese (resultado, escolha de método, \
cautela de interpretação) que nenhuma âncora cobre; false caso contrário.
- justificativa: uma frase curta.
Seja conservador: marque conflito só quando o literal e a âncora forem comparáveis (mesma grandeza e mesmo \
escopo). Pinos sobre formatação, nomes de arquivo ou estrutura não são centrais.

ÂNCORAS:
{bloco_ancoras(ancoras)}

TERMINOLOGIA (só para interpretar; não avalie estilo):
{TERMINOLOGIA}

Responda apenas com um objeto JSON, sem cercas de código e sem texto antes ou depois, com exatamente um item \
por pino da requisição, na mesma ordem, copiando o pino_id:
{ESQUEMA_PINOS}
"""


def linha_pino(pino: dict) -> str:
    literal = " ".join(str(pino["literal"]).split())
    if len(literal) > LITERAL_PINO_MAX:
        literal = literal[: LITERAL_PINO_MAX - 1] + "…"
    return (
        f"[{pino['pino_id']}] {pino['arquivo_teste']}:{pino['linha']} ({pino['funcao']}) | {pino['tipo']} | "
        f"{pino['categoria']} | alvo: {pino['alvo'] or '-'} | literal: {json.dumps(literal, ensure_ascii=False)}"
    )


def usuario_pinos(lote: list[dict]) -> str:
    return "Pinos:\n" + "\n".join(linha_pino(p) for p in lote) + "\n"
