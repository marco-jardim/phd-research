"""Glosas editoriais W13a, sem alterar células numéricas ou fontes congeladas.

Os geradores editoriais chamam ``transform`` antes de gravar. ``generate``
produz apenas as versões editoriais adicionais a partir das tabelas históricas.
Não lê o extrato, não ajusta modelos e não executa chamadas externas.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rotulos_apresentacao import ROTULO_BRACO  # noqa: E402

TABLES = ROOT / "text/latex/tables"
SUPPLEMENTAL = (
    "tab_cap8_revisores", "tab_e4_desempenho_faixas_oof", "tab_cap8_reruns",
    "tab_cap8_ablacao_dossie", "tab_qw3_fairness_rejections", "tab_qw3_fairness_rejections_corte6",
    "tab_sensibilidade_qw1", "tab_sensibilidade_2x2",
)
EDITORIAL_TEMPLATES = ("tab_cap8_fusao_robustez", "tab_cap8_vies_referencia")

P = "ASR: arranjo sequencial do registro interno, que preserva o corte do OpenRecLink e aplica a política com recalibração isotônica abaixo dele. "
PAR2 = "PAR~=~2: pares verdadeiros recuperados na revisão manual de origem abaixo do corte. "
REFREC = "REFREC identifica o registro do SIM. "
# Tabelas de equidade sem coluna ou linha de ASR: não recebem a glosa de ASR nem a de REFREC.
SEM_ASR = ("tab_equidade_v31", "tab_equidade_v31_rejeicoes_cascata", "tab_qw3_group_metrics",
           "tab_qw3_fairness_rejections", "tab_qw3_fairness_rejections_corte6")
# OR do corte 6,0 com duas casas (valores completos em data/equidade_v31_pairwise_tests.csv);
# OR abaixo de 0,005 aparece como <0,01.
CORTE6_OR_DUAS_CASAS = (
    ("$-$13{,}0 & 0{,}025 \\\\", "$-$13{,}0 & 0{,}03 \\\\"),
    ("$-$13{,}3 & 0{,}005 \\\\", "$-$13{,}3 & $<$0{,}01 \\\\"),
    ("$-$13{,}4 & 0{,}017 \\\\", "$-$13{,}4 & 0{,}02 \\\\"),
    ("$-$15{,}1 & 0{,}094 \\\\", "$-$15{,}1 & 0{,}09 \\\\"),
    ("$+$78{,}3 & 224{,}7 \\\\", "$+$78{,}3 & 224{,}68 \\\\"),
    ("$+$76{,}6 & 113{,}2 \\\\", "$+$76{,}6 & 113{,}23 \\\\"),
    ("$+$76{,}8 & 121{,}2 \\\\", "$+$76{,}8 & 121{,}20 \\\\"),
    ("$+$75{,}8 & 91{,}9 \\\\", "$+$75{,}8 & 91{,}95 \\\\"),
)
FUSION = ("F4: fusão por posição recíproca de O1 (escore do OpenRecLink) e O3 "
          "(probabilidade calibrada); F3: mesma combinação acrescida de O3 com MACD "
          "(Medidas Contínuas de Diferença de Datas de nascimento). ")


def add_note(text, note):
    """Acrescenta uma nota autossuficiente em cada tabela, sem duplicá-la."""
    note = terminologia_revisao2(note)
    previous_marker = r"\par\footnotesize " + note + "\n"
    marker = r"\par\footnotesize " + note.rstrip() + "\n"
    text = text.replace(previous_marker, marker)
    if marker in text:
        return text
    if r"\end{table}" in text:
        return text.replace(r"\end{table}", marker + r"\end{table}")
    if r"\end{longtable}" in text:
        return text.replace(r"\end{longtable}", r"\end{longtable}" + "\n" + marker)
    # Fragmentos tabulares são incluídos em floats definidos pelo capítulo.
    return text.rstrip() + "\n" + marker


# Terminologia da revisão 2 (guia 3.6; decisões da autora de 2026-11-08). Só strings de
# texto: nenhuma célula numérica muda. Idempotente; aplicada ao final de ``transform``.
# Ordem importa: as formas mais longas vêm antes das mais curtas.
TERMINOLOGIA_REVISAO2 = (
    # P é o braço do registro interno, não o resultado principal (decisão D2 da autora, 2026-11-08)
    ("P: arranjo principal, que preserva", "ASR: arranjo sequencial do registro interno, que preserva"),
    ("P permanece o braço principal", "ASR permanece o braço do registro interno"),
    # FP de P reclassificados na referência: a classe é "par verdadeiro"
    ("FP reclassificados uniformemente como vínculos", "FP reclassificados uniformemente como pares verdadeiros"),
    # revisão manual (sem "revisão clerical")
    ("revisão clerical de origem", "revisão manual de origem"),
    ("revisão clerical abaixo do corte", "revisão manual de origem abaixo do corte"),
    ("revisão clerical", "revisão manual"),
    # [5,9) da nota é a "faixa intermediária da nota final"; \textit{grey area} é só a região
    # abaixo do corte do OpenRecLink
    ("Vigilância: área cinza da nota $[5,9)$", "Vigilância: faixa intermediária da nota final $[5,9)$"),
    ("Confirmação: área cinza da nota $[5,9)$", "Confirmação: faixa intermediária da nota final $[5,9)$"),
    ("agrupa as faixas da área cinza da nota", "agrupa as faixas da nota final em $[5,9)$"),
    ("revisão humana da área cinza da nota", "revisão humana da faixa intermediária da nota final"),
    ("A revisão da área cinza da nota do CdR", "A revisão da faixa intermediária da nota final do CdR"),
    ("A área cinza da nota não é a região", "A faixa intermediária da nota final não é a região"),
    ("Área cinza da nota: nota do CdR em", "Faixa intermediária da nota final: nota final do CdR em"),
    ("na área cinza da nota do CdR (", "na faixa intermediária da nota final do CdR ("),
    ("Calibração na área cinza definida pela nota do comparador",
     "Calibração na faixa intermediária definida pela nota final do comparador"),
    ("Área cinza: $5", "Faixa intermediária: $5"),
    ("Área cinza &", "Faixa intermediária &"),
    # "vínculo" é só o resultado da decisão; a classe da referência é "par verdadeiro"
    ("PAR~=~2: vínculos recuperados", "PAR~=~2: pares verdadeiros recuperados"),
    ("vínculos PAR~=~2", "pares verdadeiros PAR~=~2"),
    ("vínculos verdadeiros", "pares verdadeiros"),
    ("cada célula apresenta potenciais pares (vínculos)", "cada célula apresenta potenciais pares (pares verdadeiros)"),
    ("Inconcl. (vínculos)", "Inconcl. (pares verdadeiros)"),
    ("classe positiva (vínculo)", "classe positiva (par verdadeiro)"),
    ("os vínculos da referência", "os pares verdadeiros da referência"),
    ("referência operacional de vínculos", "referência operacional de pares verdadeiros"),
    ("dividida pelo total de vínculos", "dividida pelo total de pares verdadeiros"),
    ("mantém vínculos ao humano no denominador",
     "mantém no denominador os pares verdadeiros encaminhados ao humano"),
    ("todos os vínculos não aceitos automaticamente", "todos os pares verdadeiros não aceitos automaticamente"),
    ("inclui vínculos encaminhados ao revisor humano", "inclui pares verdadeiros encaminhados ao revisor humano"),
    ("são os vínculos na sensibilidade", "são os pares verdadeiros na sensibilidade"),
    ("com um vínculo, sem idade", "com um par verdadeiro, sem idade"),
    ("com vínculos $V_H$ entre parênteses", "com pares verdadeiros $V_H$ entre parênteses"),
    ("Vínculos perdidos", "Pares verdadeiros perdidos"),
    ("vínculos perdidos", "pares verdadeiros perdidos"),
    ("Vínculos encaminhados ao humano", "Pares verdadeiros encaminhados ao humano"),
    ("por vínculo perdido", "por par verdadeiro perdido"),
    ("2, vínculo recuperado na", "2, par verdadeiro recuperado na"),
    # não vínculo / falso vínculo
    ("os classificados como não-vínculo", "os classificados como NONMATCH"),
    ("decisões de não vínculo", "decisões NONMATCH"),
    (r"\texttt{NONMATCH} (não-vínculo)", r"\texttt{NONMATCH} (não aceito como vínculo)"),
    ("NONMATCH: decisão de não vínculo", "NONMATCH: decisão de não aceitar como vínculo"),
    ("NONMATCH: não vínculo", "NONMATCH: não aceito como vínculo"),
    ("ausência de falso vínculo automático", "ausência de falso positivo automático"),
    # VPP, não "precisão"
    ("Prec.: precisão", "VPP: valor preditivo positivo"),
    ("Prec.", "VPP"),
    ("Precisão", "VPP"),
    (r"Precis\~ao", "VPP"),
    ("para sensibilidade e precisão", "para sensibilidade e VPP"),
    ("F$_1$, precisão e sensibilidade", "F$_1$, VPP e sensibilidade"),
    ("estratégia de máxima precisão", "estratégia de confirmação"),
    ("ordenados por precisão decrescente", "ordenados por VPP decrescente"),
    ("maximização do VPP", "maximização do VPP"),
    (r"no plano precisão $\times$ sensibilidade", r"no plano VPP $\times$ sensibilidade"),
)
# Contagens da classe da referência (teste, extrato, faixa [5,9), PAR = 2, metas, seleção da ablação).
# Não inclui 211, 127, 123 etc., que são vínculos aceitos pela decisão.
CONTAGEM_CLASSE = re.compile(r"(?<![\d.,])(74|72|247|151|36|23|18) vínculos\b")
# Cabeçalho de coluna que conta a classe da referência.
CABECALHO_CLASSE = re.compile(r"(?<=& )Vínculos(?= &| \\\\)")


def texttt_codigos(text):
    """Guia 3.6: MATCH e NONMATCH só em \texttt{} (não altera AUTO\_MATCH nem o que já está em \texttt{})."""
    return re.sub(r"(?<![\\\w{_])((?:NON)?MATCH)(?![\w}])", r"\\texttt{\1}", text)


# Renomeação P -> ASR (decisão da autora de 2026-11-08; ``docs/plans/PLANO_RENOMEAR_P.md``). No registro
# interno e nos dados o arranjo continua ``P``; só o rótulo exibido muda, e só em contextos ancorados.
# Nunca ``str.replace("P", ...)``: probabilidade, ``\label{...-P-...}``, caminhos e códigos de dados ficam.
# Idempotente: ``ASR`` não casa com nenhuma regra. Variantes mais longas vêm antes das mais curtas.
_ASR = ROTULO_BRACO["P"]
_FIM_TOKEN = r"(?![\w-])"
ROTULOS_BRACO_REGRAS = tuple((re.compile(padrao), novo) for padrao, novo in (
    # variantes A2 e A3
    (r"(?<![\w\\-])P-PAR2-grey" + _FIM_TOKEN, ROTULO_BRACO["P-PAR2-grey"]),
    (r"(?<![\w\\-])P-Platt" + _FIM_TOKEN, ROTULO_BRACO["P-Platt"]),
    # A5: ``P, \texttt{MATCH} ao humano`` troca só o P inicial
    (r"(?<![\w\\-])P(?=, \\texttt\{MATCH\} ao humano)", _ASR),
    # A1 em célula: ``P & ...`` no início da linha, ``... & P & ...`` e ``... & P \\``
    (r"(?m)(?:^|(?<= & ))P(?= &| \\\\)", _ASR),
    # A1 em rótulo de linha ou de grupo e em prosa de nota/legenda
    (r"(\bArranjo sequencial )P" + _FIM_TOKEN, r"\1" + _ASR),
    (r"(\bArranjo )P" + _FIM_TOKEN, r"\1" + _ASR),
    (r"(\bbraço )P" + _FIM_TOKEN, r"\1" + _ASR),
    (r"(?<![\w\\-])P(?= (?:vigil|confirm)\.)", _ASR),
    (r"(?<![\w\\-])P(?=: arranjo sequencial)", _ASR),
    (r"(\bsobre )P" + _FIM_TOKEN, r"\1" + _ASR),
    (r"(\b(?:VPP|PAR~=~2) de )P" + _FIM_TOKEN, r"\1" + _ASR),
    (r"(\bsensibilidade de )P(?= com\b)", r"\1" + _ASR),
))


def rotulos_braco(text):
    """Troca o rótulo do arranjo (P) pelo de apresentação (ASR) só em contextos ancorados."""
    for regra, novo in ROTULOS_BRACO_REGRAS:
        text = regra.sub(novo, text)
    return text


def terminologia_revisao2(text):
    for old, new in TERMINOLOGIA_REVISAO2:
        text = text.replace(old, new)
    text = rotulos_braco(text)
    text = CONTAGEM_CLASSE.sub(r"\1 pares verdadeiros", text)
    text = texttt_codigos(text)
    return CABECALHO_CLASSE.sub(lambda _: r"\shortstack{Pares\\verdadeiros}", text)


_CAPTION = re.compile(r"\\caption(?=[\[{])")
_CAMINHO_FRAGIL = re.compile(r"(?<!\\protect)\\(path|url)\{")


def proteger_caminhos_legenda(text):
    r"""\path e \url são frágeis em argumento móvel: dentro de \caption[...]{...} recebem \protect.

    Só muda o comando, nunca o caminho; idempotente (``\protect\path`` não casa de novo).
    """
    out, pos = [], 0
    for match in _CAPTION.finditer(text):
        if match.start() < pos:
            continue
        i = match.end()
        for opening, closing in (("[", "]"), ("{", "}")):
            if i >= len(text) or text[i] != opening:
                continue
            depth, i = 1, i + 1
            while depth and i < len(text):
                if text[i] == "\\":
                    i += 2
                    continue
                depth += (text[i] == opening) - (text[i] == closing)
                i += 1
        out += [text[pos:match.start()], _CAMINHO_FRAGIL.sub(r"\\protect\\\1{", text[match.start():i])]
        pos = i
    return "".join(out + [text[pos:]])


def _trocar_uma_vez(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Layout de tab_cap8_reruns exige count == 1: {old!r}, count={text.count(old)}")
    return text.replace(old, new, 1)


def legible_reruns_layout(text):
    r"""Só apresentação de tab_cap8_reruns: sem \resizebox, nomes que quebram linha e fonte legível.

    Colunas p{} em frações de \linewidth (o preâmbulo não carrega tabularx): soma de 0,85 mais o
    preenchimento de 9 * 2 * 3 pt de \tabcolsep, cerca de 0,12 da largura útil. As células de
    sensibilidade e especificidade trazem o IC de Wilson na mesma linha (cerca de 70 pt em 8 pt), de modo
    que a tabela usa \scriptsize (a fonte anterior, com \resizebox, ficava em 7,6 pt); a legenda e as
    notas mantêm \footnotesize. Células, ordem e números intactos. Idempotente.
    """
    if r"\resizebox" not in text:
        return text
    right = r">{\raggedleft\arraybackslash}"
    left = r">{\raggedright\arraybackslash}"
    columns = (left + r"p{0.14\linewidth}" + left + r"p{0.115\linewidth}"
               + (right + r"p{0.038\linewidth}") * 3 + right + r"p{0.08\linewidth}"
               + (right + r"p{0.16\linewidth}") * 2 + right + r"p{0.08\linewidth}")
    text = _trocar_uma_vez(
        text, "\\label{tab:cap8-reruns}\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{llrrrrrrr}",
        "\\label{tab:cap8-reruns}\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{" + columns + "}")
    text = _trocar_uma_vez(text, "\\end{tabular}%\n}\n", "\\end{tabular}\n")
    text = _trocar_uma_vez(text, r"& Espec. (\%) & UNSURE (\%) \\",
                           r"& Espec. (\%) & \shortstack[r]{UNSURE\\(\%)} \\")
    return _trocar_uma_vez(
        text, "FN: falsos negativos. MO:",
        "FN: falsos negativos; Sens.: sensibilidade; Espec.: especificidade. MO:")


def transform(name, text):
    name = Path(name).stem
    # Aplica a terminologia antes das notas, para que add_note reconheça notas já presentes
    # na fonte histórica com a redação antiga e não as duplique.
    text = terminologia_revisao2(text)
    if name == "tab_cap8_cascata_v31":
        # Os rótulos da revisão 2 (pares verdadeiros, \texttt{}) alargam a primeira coluna; fonte menor só aqui.
        text = text.replace("\\footnotesize\n\\setlength{\\tabcolsep}{3pt}", "\\scriptsize\n\\setlength{\\tabcolsep}{3pt}", 1)
    text = text.replace("grupo A menos grupo B", "grupo 1 menos grupo 2")
    text = text.replace("Grupo A & Grupo B", "Grupo 1 & Grupo 2")
    if name in ("tab_ablation_top10", "tab_ablation_best_category", "tab_qw1_deep_learning", "tab_cv_5fold",
                "tab_imbalance_sensitivity", "tab_pareto_frontier"):
        text = re.sub(r"\bF1\b", r"F$_1$", text)
    if name in ("tab_ablation_top10", "tab_ablation_best_category"):
        text = re.sub(r"teste interno QW1(?! \(comparação de classificadores tabulares\))",
                      "teste interno QW1 (comparação de classificadores tabulares)", text)
    if name in ("tab_cap8_revisores", "tab_e4_desempenho_faixas_oof",
                "tab_cap8_ancoras_decisao", "tab_cap8_ancoras"):
        text = text.replace("$s", "$v").replace(" s<", " v<").replace(r" s\geq", r" v\geq")
        text = text.replace("nota final $v$ do comparador", "nota final $v$ do Comparador de Registros (CdR)")
        text = text.replace("nota final do CdR ($v$)", "nota final do Comparador de Registros (CdR), $v$")
        text = text.replace("área cinza ($5", "faixa intermediária da nota final do CdR ($5")
    text = text.replace("sensibilidade com \\textit{Platt scaling}", "análise de sensibilidade de ASR com \\textit{Platt scaling}")
    text = text.replace("Numérico ($k/20$)", "Numérico")
    text = text.replace("Partição ($k/20$)", "Partição")
    text = text.replace("O3: \\texttt{p\\_cal\\_ml}", "O3: probabilidade calibrada do classificador na base inteira")
    text = text.replace("O2: nota do CdR;", "O2: nota final do CdR;")
    text = text.replace("As chaves S1 a S4 nos artefatos são históricas, não as etapas do protocolo. ", "")
    text = text.replace("P97,5", "percentil 97,5").replace("P10 a P90", "percentis 10 a 90")
    text = text.replace("meta nominal de 64\\%", "meta de 23 dos 36 PAR~=~2 (64\\%)")
    if name == "tab_cap8_arranjo_corte":
        text = text.replace("O \\textit{a posteriori} reutiliza a cascata sobre a base inteira sem recalibração.",
                            "O braço \\textit{a posteriori} preserva o corte e aplica a cascata sobre a base inteira abaixo dele, sem recalibração.")
        text = add_note(text, P + PAR2 +
            "B1: mantém a base inteira e a probabilidade original, substituindo apenas as proporções de erro do revisor. "
            "A3: restringe também o treinamento do classificador à região abaixo do corte. "
            "ASR-PAR2-faixa: estima a omissão do revisor somente nos pares verdadeiros PAR~=~2 e agrupa as faixas da nota final em $[5,9)$. "
            "ASR-Platt: análise de sensibilidade de ASR com \\textit{Platt scaling} abaixo do corte. "
            "Braço \\textit{a posteriori}: cascata sobre a base inteira aplicada abaixo do corte, sem recalibração. "
            "Viés de incorporação: uso, na construção da referência, das mesmas evidências avaliadas pelo método.")
    notes = {
        "tab_cap8_mesma_carga": P + PAR2 + "$k$: número de revisões humanas na ordenação. CdR: Comparador de Registros.",
        "tab_cap8_esforco": P + PAR2 + "O3: ordenação pela probabilidade calibrada do classificador na base inteira. A meta de recuperar os 36 pares verdadeiros é cenário, não carga observada.",
        "tab_cap8_fusao_robustez": FUSION,
        "tab_cap8_robustez_arranjo": P + "ASR-Platt: análise de sensibilidade de ASR com \\textit{Platt scaling} abaixo do corte. "
            "Numérico e Partição: réplicas com dominância de ASR-Platt sobre ASR, em 20 reexecuções de cada bloco. "
            "Inversa: réplicas com dominância de ASR sobre ASR-Platt nos blocos numérico e de partição, respectivamente.",
        "tab_cap8_filtro_llm": PAR2 + "NONMATCH: decisão de não aceitar como vínculo; MATCH: vínculo; UNSURE: inconclusivo. O3, mesmo $H$: ordenação pela probabilidade calibrada com o mesmo número de revisões humanas após o filtro. O topo de 440 pares não teve todas as respostas.",
        "tab_cap8_subamostragem": "A ordenação usa a probabilidade calibrada do classificador. A linha Sem subamostragem corresponde ao treino sem subamostragem, não à referência operacional de pares verdadeiros.",
        "tab_cap8_ancoras": "Conjuntos âncora: potenciais pares de nota extrema usados para calibrar. CdR: Comparador de Registros.",
        "tab_cap8_ancoras_decisao": "Conjuntos âncora: potenciais pares de nota extrema usados para calibrar. CdR: Comparador de Registros; $v$: sua nota final. A faixa intermediária da nota final não é a região abaixo do corte do OpenRecLink.",
        "tab_cap8_referencia": P + PAR2 + "CdR: Comparador de Registros. Faixa intermediária da nota final: nota final do CdR em $[5,9)$.",
        "tab_cap8_tamanho_modelo": "PAR é a classificação da referência operacional: 1, vínculo aceito pelo corte; 2, par verdadeiro recuperado na revisão manual de origem abaixo do corte; 0, demais potenciais pares.",
        "tab_cap8_revisores": "Dossiê completo: inclui as saídas decisórias do motor. Instruções v3: solicitam essas saídas. CdR: Comparador de Registros; $v$: sua nota final.",
        "tab_cap8_cascata_v31": "Cascata: a mesma política aplicada a todos os potenciais pares. OOF: predições fora da amostra de treinamento; REFREC: registro do SIM. ALWAYS: guarda; AUTO: decisão por custo; MATCH: vínculo; NONMATCH: não aceito como vínculo; UNSURE: inconclusivo. v3.1 distingue revisão humana da revisão por modelo de linguagem; v4 são as instruções sem as saídas decisórias do motor.",
        "tab_e4_desempenho_faixas_oof": REFREC + "OOF: probabilidades produzidas fora da amostra de treinamento. CdR: Comparador de Registros; $v$: sua nota final.",
        "tab_cap8_ablacao_dossie": "Ablação: retirada conjunta das saídas decisórias do motor do dossiê, documento estruturado com os subescores disponíveis para a revisão.",
        "tab_dossie_campos": "Dossiê: documento estruturado com os subescores disponíveis para a revisão por modelo de linguagem. CdR: Comparador de Registros.",
        "tab_cv_5fold": "QW1: comparação de classificadores tabulares. Validação cruzada: avaliações com alternância das partições de treinamento e validação.",
        "tab_qw1_deep_learning": "QW1: comparação de classificadores tabulares. A0: controle com a partição original do QW1.",
        "tab_nb02_recall": "NB02: estratégia de vigilância (maximização da sensibilidade); RF: Random Forest; MLP: perceptron multicamada.",
        "tab_nb03_precision": "NB03: estratégia de confirmação (maximização do VPP); consenso: unanimidade entre modelos; empilhamento: combinação das predições por um meta-aprendiz.",
        "tab_pareto_frontier": "RF: Random Forest. SMOTE: sobreamostragem sintética da classe minoritária. Fluxo Modal: execução em nuvem que acrescentou o XGBoost à ablação (remoção ou isolamento de componentes).",
        "tab_perfil_recuperados": "ML: aprendizado de máquina. Recuperados: vínculos detectados pelo Random Forest com SMOTE e não pelo limiar 8 da nota final do Comparador de Registros (CdR).",
        "tab_sensibilidade_qw1": "QW1: comparação de classificadores tabulares. A e B: partição por linha, respectivamente com e sem indicadores de óbito. C e D: partição agrupada por REFREC, respectivamente com e sem esses indicadores. " + REFREC,
        "tab_sensibilidade_2x2": REFREC + "A e B: partição por linha, com e sem indicadores de óbito; C e D: partição por REFREC, com e sem esses indicadores.",
        "tab_cap8_vies_referencia": P + PAR2 + "C1: VPP das aceitações do LLM abaixo do corte inferior a 50\\%; C2: perdas de PAR~=~2 de ASR; C3: margem de O3 frente a O1 na meta de 23 pares verdadeiros; C4: vantagem de F4 frente a O1. O1: escore do OpenRecLink; O3: probabilidade calibrada; F4: fusão por posição recíproca de O1 e O3. ",
    }
    if name == "tab_cap8_reruns":
        text = text.replace("Avaliação dos revisores no Exp2 segundo o dossiê e a versão do prompt.",
            "Avaliação dos revisores no Experimento 2 (revisão assistida por LLM), segundo o dossiê (completo ou sem as saídas do motor, MO) e as instruções (v3, que solicitam essas saídas, ou v4, que não as solicitam), eixos independentes: há combinações como \\textit{Sem MO/v3}.")
        text = text.replace("MO: bloco \\texttt{model\\_outputs}.", "MO: bloco de saídas decisórias do motor.")
        text = add_note(text, REFREC + "QW2: registro das execuções de revisão por modelo de linguagem.")
        text = legible_reruns_layout(text)
    if name == "tab_sensibilidade_qw1":
        text = text.replace("Classificadores do QW1 nos quatro cenários de sensibilidade, limiar 0,5.",
            "Classificadores do QW1 (comparação de classificadores tabulares) nos cenários A a D (partição por linha ou por REFREC, com ou sem indicadores de óbito), limiar 0,5.")
    if "equidade" in name or name.startswith("tab_qw3_"):
        text = text.replace("corte 6{,}0 da nota (", "corte 6{,}0 da nota final do CdR (")
        text = text.replace("corte 6,0,", "corte 6,0 da nota final do CdR,")
        text = re.sub(r"corte 6\{,\}0(?! da nota final do CdR)", "corte 6{,}0 da nota final do CdR", text)
        base_name = name.removesuffix(".tex")
        if base_name in ("tab_qw3_fairness_rejections", "tab_qw3_fairness_rejections_corte6"):
            # Sem ASR, REFREC nem cascata: só CdR e OR aparecem nestas tabelas.
            text = add_note(text, "CdR: Comparador de Registros. OR é razão de chances.")
        elif base_name in SEM_ASR:
            # Tabelas da comparação (corte 6,0, XGBoost, cascata sobre a base inteira): sem ASR nem REFREC.
            text = add_note(text, "CdR: Comparador de Registros. "
                "Cascata v3.1: política de quatro ações aplicada à base inteira, incluindo revisão humana.")
        else:
            text = add_note(text, P + "CdR: Comparador de Registros. " + REFREC +
                "Cascata v3.1: política de quatro ações aplicada à base inteira, incluindo revisão humana.")
        if base_name == "tab_qw3_fairness_rejections_corte6":
            text = text.replace("Encerramento   &", r"Encerramento da notifica\c{c}\~ao &")
            # Com o rótulo editorial a tabela excede a largura do texto em 40 pt: fonte e espaçamento menores.
            text = text.replace("\\small\n\\begin{tabular}",
                                "\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}", 1)
            for old, new in CORTE6_OR_DUAS_CASAS:
                if text.count(old) == 0 and text.count(new) == 1:
                    continue  # já transformada: a função é idempotente
                if text.count(old) != 1:
                    raise ValueError(f"Linha do corte 6 divergente: {old!r}")
                text = text.replace(old, new)
    if name in notes:
        text = add_note(text, notes[name])
    return proteger_caminhos_legenda(terminologia_revisao2(text))


def generate(output=TABLES / "editorial"):
    if output.resolve() == TABLES.resolve():
        raise ValueError("Use a versão editorial; as tabelas históricas são imutáveis")
    output.mkdir(parents=True, exist_ok=True)
    for name in SUPPLEMENTAL:
        source = TABLES / (name + ".tex")
        (output / source.name).write_text(transform(name, source.read_text(encoding="utf-8")),
                                          encoding="utf-8", newline="\n")
    # Estas duas tabelas já eram versões editoriais sem fonte histórica irmã.
    # Conservam-se seus modelos tabulares, atualizando somente a apresentação.
    for name in EDITORIAL_TEMPLATES:
        source = TABLES / "editorial" / (name + ".tex")
        (output / source.name).write_text(transform(name, source.read_text(encoding="utf-8")),
                                          encoding="utf-8", newline="\n")


if __name__ == "__main__":
    generate()
