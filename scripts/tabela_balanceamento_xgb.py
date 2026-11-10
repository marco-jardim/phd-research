"""Gera a tabela do apêndice de sensibilidade com o tratamento do desbalanceamento no RF e no XGBoost.

Lê apenas ``data/balanceamento_xgb.csv`` e ``data/balanceamento_xgb_metadata.json`` (agregados por
estimador, estratégia e métrica, produzidos por ``scripts/balanceamento_xgb.py``). Não lê o extrato, não
ajusta modelos e não chama serviços externos. A saída é determinística: números com vírgula decimal e
três casas, arredondados meia unidade para cima (``ROUND_HALF_UP``), sem data nem versão no arquivo.

Uso: ``python scripts/tabela_balanceamento_xgb.py``
"""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "balanceamento_xgb.csv"
META = ROOT / "data" / "balanceamento_xgb_metadata.json"
SAIDA = ROOT / "text" / "latex" / "tables" / "editorial"
NOME = "tab_balanceamento_xgb.tex"
LABEL = "tab:balanceamento-xgb"

ESTIMADORES = (("rf", "Random Forest"), ("xgb", "XGBoost"))
BASE = "No balancing"
METRICAS = ("VPP", "Sensibilidade", "F1", "AUC-PR")
NOMES_METRICAS = {"VPP": "VPP", "Sensibilidade": "Sensibilidade", "F1": "F$_1$", "AUC-PR": "AUC-PR"}
# Painel A: médias das quatro métricas; painel B: diferença em relação ao treino sem balanceamento e IC 95%.
CABECALHO_A = ("Estratégia",) + tuple(NOMES_METRICAS[m] for m in METRICAS)
CABECALHO_B = ("Estratégia",) + tuple(r"$\Delta$" + NOMES_METRICAS[m] for m in METRICAS)
COLUNAS = len(CABECALHO_A)
# Só apresentação, sem \resizebox: estratégia em coluna que quebra linha e quatro colunas numéricas à direita.
# O preâmbulo não carrega tabularx; 0,21 + 4 x 0,175 = 0,91 de \linewidth mais 10 \tabcolsep de 3 pt (cerca de 0,07).
ESPECIFICACAO_COLUNAS = (r">{\raggedright\arraybackslash}p{0.21\linewidth}"
                         + (r">{\raggedleft\arraybackslash}p{0.175\linewidth}") * (COLUNAS - 1))
# Rótulos iguais aos da tabela histórica (editorial/tab_imbalance_sensitivity.tex); não vêm de ``label_pt``.
ROTULOS = {
    "No balancing": "Sem balanceamento",
    "Class weight=balanced": "Pesos de classe balanceados",
    "SMOTE 0.3": "SMOTE 0,3",
    "SMOTE 0.5": "SMOTE 0,5",
    "SMOTE 1.0": "SMOTE 1,0",
    "BorderlineSMOTE 0.3": "BorderlineSMOTE 0,3",
    "ADASYN 0.3": "ADASYN 0,3",
    "SMOTETomek 0.3": "SMOTETomek 0,3",
    "Class weight + SMOTE 0.3": "Pesos de classe + SMOTE 0,3",
}


def arredondar(valor):
    """Três casas, meia unidade para cima; ``repr`` do float evita o ruído binário do ``Decimal(float)``."""
    return Decimal(repr(float(valor))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def numero(valor):
    """Vírgula decimal e três casas; negativo em modo matemático; -0,000 vira 0,000."""
    q = arredondar(valor)
    texto = f"{abs(q):.3f}".replace(".", ",")
    return "$-$" + texto if q < 0 else texto


def milhar(valor):
    return f"{int(valor):,}".replace(",", ".")


def carregar(csv=CSV, meta=META):
    return pd.read_csv(csv), json.loads(Path(meta).read_text(encoding="utf-8"))


def validar(df, meta):
    """Interrompe se o artefato não for o esperado; nada é completado ou trocado por zero."""
    if not meta["historical_reproduction"]["passou"]:
        raise ValueError("a primeira repetição não reproduz data/imbalance_sensitivity.csv")
    # A nota da tabela afirma isto; os metadados têm de sustentar a afirmação.
    if "ml_oof" not in meta["estimators"]["xgb"]["factory"] or "30 atributos" not in meta["features"]["note"]:
        raise ValueError("os metadados não sustentam a descrição do XGBoost (estimador de O3, 30 atributos de O3)")
    if "n_neg/n_pos" not in meta["estimators"]["xgb"]["weighted"]:
        raise ValueError("os metadados não descrevem scale_pos_weight como negativos/positivos")
    if "1,0" not in meta["estimators"]["xgb"]["unweighted"]:
        raise ValueError("os metadados não descrevem scale_pos_weight igual a 1 sem pesos de classe")
    if set(meta["strategies"]) != set(ROTULOS):
        raise ValueError("as estratégias dos metadados diferem das que têm rótulo da tabela histórica")
    n = meta["partition"]["k"] * meta["partition"]["repetitions"]
    if not (df["n_particoes"] == n).all():
        raise ValueError(f"o CSV tem linhas com n_particoes diferente de {n}")
    dif = df[df["tipo"] == "dif_sem_balanceamento"]
    if not (dif["gl"] == n - 1).all():
        raise ValueError(f"o CSV tem diferenças com graus de liberdade diferentes de {n - 1}")
    for usadas in (df.loc[df["tipo"] == "desempenho", ["media"]],
                   dif[["diferenca_media", "ic95_inf", "ic95_sup"]]):
        if not np.isfinite(usadas.to_numpy(dtype=float)).all():
            raise ValueError("o CSV tem valor indefinido ou infinito nas colunas usadas pela tabela")
    return n


def indexar(df):
    desempenho = {(r.estimador, r.estrategia, r.metrica): r.media
                  for r in df[df["tipo"] == "desempenho"].itertuples()}
    diferencas = {(r.estimador, r.estrategia, r.metrica): r
                  for r in df[df["tipo"] == "dif_sem_balanceamento"].itertuples()}
    return desempenho, diferencas


def ic_diferenca(linha):
    """IC 95% entre colchetes, para a segunda linha de tabela da estratégia.

    Os colchetes vão entre chaves: ``{[}`` não é lido como argumento opcional de ``\\\\`` ou de ``\\\\*``.
    """
    return r"{[}" + f"{numero(linha.ic95_inf)}; {numero(linha.ic95_sup)}" + r"{]}"


def linhas_medias(codigo, estrategias, desempenho):
    """Painel A: média das quatro métricas, uma linha por estratégia (inclui o treino sem balanceamento)."""
    return [" & ".join([ROTULOS[chave]] + [numero(desempenho[(codigo, chave, m)]) for m in METRICAS]) + r" \\"
            for chave in estrategias]


def linhas_diferencas(codigo, estrategias, diferencas):
    """Painel B: duas linhas de tabela por estratégia, sem a linha de referência.

    A primeira traz o nome e as quatro diferenças e termina em ``\\\\*`` (não quebra a página depois dela); a
    segunda traz a célula de nome vazia e os quatro IC 95%. O valor fica na linha de base do nome.
    """
    linhas = []
    for chave in estrategias:
        if chave == BASE:
            continue
        dif = [diferencas[(codigo, chave, m)] for m in METRICAS]
        linhas.append(" & ".join([ROTULOS[chave]] + [numero(d.diferenca_media) for d in dif]) + r" \\*")
        linhas.append(" & ".join([""] + [ic_diferenca(d) for d in dif]) + r" \\")
    return linhas


def legenda(n, meta):
    return (r"Tratamento do desbalanceamento no Random Forest e no XGBoost: média das métricas em "
            f"{n} partições (validação cruzada $5 \\times 5$, limiar de 0,5) e diferença, com IC de "
            "95\\%, em relação ao treino sem balanceamento.")


def nota(n, meta):
    k, r = meta["partition"]["k"], meta["partition"]["repetitions"]
    universo = meta["universe"]
    tolerancia = f"{meta['historical_reproduction']['tolerancia']:.4f}".replace(".", ",")
    return (
        r"\par\smallskip\noindent Fonte: \path{data/balanceamento_xgb.csv}, produzida por "
        r"\path{scripts/balanceamento_xgb.py}; metadados em \path{data/balanceamento_xgb_metadata.json}. "
        f"Cada valor do painel A é a média nas {n} partições ({r} repetições de {k} partições estratificadas por linha, "
        f"sem agrupar por REFREC, iguais para os dois estimadores) dos {milhar(universo['rows'])} "
        f"potenciais pares, com {milhar(universo['positives'])} pares verdadeiros. "
        "Reamostragem só no treinamento; AUC-PR calculada sobre a probabilidade, sem limiar. "
        r"Painel B, $\Delta$: média das diferenças pareadas, partição a partição, entre a estratégia e o treino sem "
        r"balanceamento do mesmo estimador; entre colchetes, IC de 95\% pelo teste $t$ corrigido para "
        r"validação cruzada repetida \cite{Nadeau2003inference}. Os IC não têm correção para comparações "
        "múltiplas e são descritivos."
        "\n\n"
        r"\noindent "
        f"O XGBoost usa a configuração do estimador de O3 (\\path{{gzcmd/ml_oof.py}}) sobre os mesmos "
        f"{meta['features']['n']} atributos do Estudo 2, não sobre os 30 atributos de O3. "
        f"Os {meta['features']['n']} atributos incluem os indicadores de encerramento por óbito, ausentes de O3 "
        r"(\texttt{obito\_tb} e \texttt{obito\_\allowbreak sinan}). "
        r"O \texttt{scale\_pos\_weight} vale 1 sem pesos de classe; com pesos de classe, é a razão entre "
        "negativos e positivos do treino da partição (após a reamostragem, na combinação com SMOTE). "
        "A primeira repetição reproduz, no Random Forest, as médias e os desvios de VPP, sensibilidade e "
        r"F$_1$ da Tabela~\ref{tab:imbalance-sensitivity}, dentro de " + tolerancia + "."
    )


def render(df, meta):
    n = validar(df, meta)
    desempenho, diferencas = indexar(df)
    estrategias = meta["strategies"]
    if next(iter(estrategias)) != BASE:
        raise ValueError("a primeira estratégia dos metadados tem de ser o treino sem balanceamento")
    cabecalho_a = " & ".join(CABECALHO_A) + r" \\"
    cabecalho_b = " & ".join(CABECALHO_B) + r" \\"
    # longtable: a tabela quebra entre páginas; nas páginas seguintes, linha de continuação e cabeçalho genérico
    # das colunas (as métricas são as mesmas nos painéis A e B, que trazem o próprio cabeçalho no corpo).
    tex = [
        "% Gerada por scripts/tabela_balanceamento_xgb.py; não editar manualmente.",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\begin{longtable}{" + ESPECIFICACAO_COLUNAS + "}",
        r"\caption{" + legenda(n, meta) + r"}\label{" + LABEL + r"}\\",
        r"\toprule",
        r"\endfirsthead",
        r"\multicolumn{" + str(COLUNAS) + r"}{l}{\textit{(continuação)}} \\",
        r"\toprule",
        cabecalho_a,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(COLUNAS) + r"}{r}{\textit{(continua)}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for i, (codigo, titulo) in enumerate(ESTIMADORES):
        if i:
            tex.append(r"\midrule")
        tex += [r"\multicolumn{" + str(COLUNAS) + r"}{l}{\textit{" + titulo + r"}} \\",
                r"\multicolumn{" + str(COLUNAS) + r"}{l}{Painel A: média nas " + str(n) + r" partições} \\",
                cabecalho_a, r"\cmidrule(lr){1-" + str(COLUNAS) + "}"]
        tex += linhas_medias(codigo, estrategias, desempenho)
        tex += [r"\multicolumn{" + str(COLUNAS) + r"}{l}{Painel B: diferença em relação ao treino sem "
                r"balanceamento [IC 95\%]} \\",
                cabecalho_b, r"\cmidrule(lr){1-" + str(COLUNAS) + "}"]
        tex += linhas_diferencas(codigo, estrategias, diferencas)
    tex += [r"\end{longtable}", nota(n, meta), r"\par", r"\endgroup", ""]
    return "\n".join(tex)


def generate(output=SAIDA, csv=CSV, meta=META):
    df, metadados = carregar(csv, meta)
    destino = Path(output)
    destino.mkdir(parents=True, exist_ok=True)
    alvo = destino / NOME
    alvo.write_text(render(df, metadados), encoding="utf-8", newline="\n")
    return alvo


def main():
    print(generate())


if __name__ == "__main__":
    main()
