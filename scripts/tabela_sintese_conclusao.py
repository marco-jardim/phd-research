"""Quadro-síntese da conclusão (tab:concl-sintese).

Gera text/latex/tables/editorial/tab_concl_sintese.tex. O conteúdo do quadro está nas estruturas
GRUPOS (texto com marcadores <<chave>>) e NUMEROS (cada número, com o valor esperado, a fonte
e a função que o calcula a partir do artefato). Ao gerar, todo número é recalculado dos
artefatos em data/ (e de tab_equivalencia_java.tex) e comparado com o valor esperado, que é o
que consta do texto da tese; qualquer divergência interrompe a geração. As afirmações sem
número (IC que inclui ou não o zero, ordem entre arranjos) são conferidas em AFIRMACOES.

Uso:
    python scripts/tabela_sintese_conclusao.py            # regenera a tabela
    python scripts/tabela_sintese_conclusao.py --fontes   # lista cada número, valor e fonte
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EQUIVALENCIA_JAVA = ROOT / "text/latex/tables/tab_equivalencia_java.tex"
DESTINO = ROOT / "text/latex/tables/editorial/tab_concl_sintese.tex"
LABEL = "tab:concl-sintese"

CSVS = {
    "referencia": "referencia_comparador_pos_orl.csv",
    "arranjo": "arranjo_corte_metricas.csv",
    "funil": "funil_gzcmd_final_metricas.csv",
    "mesma_carga": "comparador_mesma_carga.csv",
    "fusao_filas": "fusao_filas.csv",
    "boot_o3_o1": "ordenacao_bootstrap_o3_o1.csv",
    "boot_fusao": "fusao_robustez_bootstrap.csv",
    "boot_contexto": "contexto_robustez_bootstrap.csv",
    "equidade": "equidade_arranjo_group_metrics.csv",
    "vpp_llm": "arranjo_corte_vpp_llm.csv",
    "filtro": "filtro_llm.csv",
    "balanceamento": "imbalance_sensitivity.csv",
}


# --------------------------------------------------------------------------- formatação

def _arredondar(valor: Decimal, casas: int) -> Decimal:
    """Arredondamento de meia unidade para cima (não o 'banker's rounding' de round())."""
    return valor.quantize(Decimal(1).scaleb(-casas), rounding=ROUND_HALF_UP)


def _decimal(valor) -> Decimal:
    """Decimal a partir da representação mais curta do número, sem ruído binário."""
    return Decimal(str(float(valor)))


def inteiro(valor) -> str:
    """Inteiro (meia unidade para cima) com ponto como separador de milhar."""
    return f"{int(_arredondar(_decimal(valor), 0)):,}".replace(",", ".")


def percentual(valor) -> str:
    """Fração em porcentagem com uma casa e vírgula decimal (sem o sinal de %)."""
    return f"{_arredondar(_decimal(valor) * 100, 1):.1f}".replace(".", ",")


def decimal3(valor) -> str:
    return f"{_arredondar(_decimal(valor), 3):.3f}".replace(".", ",")


# --------------------------------------------------------------------------- leitura

def carregar(data_dir: Path = DATA) -> dict:
    fontes = {nome: pd.read_csv(data_dir / arquivo) for nome, arquivo in CSVS.items()}
    fontes["nb01"] = json.loads((data_dir / "nb01_results.json").read_text(encoding="utf-8"))
    fontes["equivalencia_java"] = EQUIVALENCIA_JAVA.read_text(encoding="utf-8")
    return fontes


def _unica(df: pd.DataFrame, **filtros) -> pd.Series:
    """Linha única que satisfaz os filtros; qualquer outra contagem é erro."""
    mascara = pd.Series(True, index=df.index)
    for coluna, valor in filtros.items():
        mascara &= df[coluna] == valor
    achadas = df[mascara]
    if len(achadas) != 1:
        raise ValueError(f"filtro {filtros} devolveu {len(achadas)} linhas, esperada 1")
    return achadas.iloc[0]


# --------------------------------------------------------------------------- números

@dataclass(frozen=True)
class Numero:
    chave: str
    esperado: str          # como aparece no texto da tese (vírgula decimal, ponto de milhar)
    fonte: str             # arquivo e filtro/coluna de origem
    calcular: Callable[[dict], str]


def _coluna(chave, esperado, nome, filtros, coluna, formato, transformar=None) -> Numero:
    """Número lido de uma coluna de um CSV de data/, na linha única definida por `filtros`."""
    descricao = ", ".join(f"{k}={v}" for k, v in filtros.items())
    fonte = f"data/{CSVS[nome]}; {descricao}; coluna {coluna}"

    def calcular(fontes: dict) -> str:
        valor = _unica(fontes[nome], **filtros)[coluna]
        return formato(transformar(valor) if transformar else valor)

    return Numero(chave, esperado, fonte, calcular)


def _perdidos(chave, esperado, estrategia) -> Numero:
    """Pares verdadeiros perdidos como na Tabela cap8-referencia: total menos aceitos menos ao humano."""
    fonte = (f"data/{CSVS['referencia']}; estrategia={estrategia}; "
             "total_vinculos menos vp_automaticos menos vinculos_ao_humano")

    def calcular(fontes: dict) -> str:
        linha = _unica(fontes["referencia"], estrategia=estrategia)
        return inteiro(linha.total_vinculos - linha.vp_automaticos - linha.vinculos_ao_humano)

    return Numero(chave, esperado, fonte, calcular)


def _nb01(chave, esperado, modelo, coluna) -> Numero:
    fonte = f"data/nb01_results.json; all_results, Modelo={modelo}; campo {coluna}"

    def calcular(fontes: dict) -> str:
        achados = [r for r in fontes["nb01"]["all_results"] if r["Modelo"] == modelo]
        if len(achados) != 1:
            raise ValueError(f"modelo {modelo!r} aparece {len(achados)} vezes em nb01_results.json")
        return decimal3(achados[0][coluna])

    return Numero(chave, esperado, fonte, calcular)


def _balanceamento(chave, esperado, funcao) -> Numero:
    fonte = f"data/{CSVS['balanceamento']}; {funcao} de f1_mean entre as estratégias"

    def calcular(fontes: dict) -> str:
        return decimal3(getattr(fontes["balanceamento"]["f1_mean"], funcao)())

    return Numero(chave, esperado, fonte, calcular)


def _equivalencia_java(chave, esperado) -> Numero:
    fonte = "text/latex/tables/tab_equivalencia_java.tex; linha Nota final, coluna de nota final idêntica (%)"

    def calcular(fontes: dict) -> str:
        achado = re.search(r"(?m)^Nota final & ([0-9,]+) & ", fontes["equivalencia_java"])
        if not achado:
            raise ValueError("linha 'Nota final' ausente de tab_equivalencia_java.tex")
        return achado.group(1)

    return Numero(chave, esperado, fonte, calcular)


_MC = "mesma_carga"
_META = "k_min_equivalente_P"
_V, _C = "vigilancia", "confirmacao"


def _mc(chave, esperado, ordenacao, modo, carga, coluna="k") -> Numero:
    return _coluna(chave, esperado, _MC, dict(ordenacao=ordenacao, modo=modo, carga=carga), coluna, inteiro)


def _ff(chave, esperado, fila, metrica) -> Numero:
    return _coluna(chave, esperado, "fusao_filas", dict(fila=fila, metrica=metrica), "media", inteiro)


NUMEROS: tuple[Numero, ...] = (
    # Estudo 1
    _equivalencia_java("cdr_igual", "11,2"),
    _coluna("cdr9_sens", "37,7", "referencia", dict(estrategia="pos_orl_corte_nota_9"), "sens_automatica", percentual),
    _coluna("cdr9_fp", "2", "referencia", dict(estrategia="pos_orl_corte_nota_9"), "fp_automaticos", inteiro),
    _perdidos("cdr9_perd", "154", "pos_orl_corte_nota_9"),
    # Estudos 2 e 3
    _nb01("rf_f1", "0,910", "Random Forest", "F1-Score"),
    _nb01("lr_f1", "0,485", "Logistic Regression", "F1-Score"),
    _nb01("rfs_aucpr", "0,973", "RF + SMOTE", "AUC-PR"),
    _nb01("lr_aucpr", "0,967", "Logistic Regression", "AUC-PR"),
    _nb01("rf_aucpr", "0,956", "Random Forest", "AUC-PR"),
    _balanceamento("bal_min", "0,880", "min"),
    _balanceamento("bal_max", "0,918", "max"),
    # Estudo 4, referência operacional (Tabela cap8-referencia)
    _coluna("corte_acei", "211", "referencia", dict(estrategia="orl_corte_servico"), "vp_automaticos", inteiro),
    _coluna("tot", "247", "referencia", dict(estrategia="orl_corte_servico"), "total_vinculos", inteiro),
    _coluna("corte_sens", "85,4", "referencia", dict(estrategia="orl_corte_servico"), "sens_automatica", percentual),
    _perdidos("corte_perd", "36", "orl_corte_servico"),
    _coluna("manual_rev", "36", "referencia", dict(estrategia="orl_corte_servico_mais_revisao_manual"),
            "vinculos_ao_humano", inteiro),
    _coluna("manual_sens", "100,0", "referencia", dict(estrategia="orl_corte_servico_mais_revisao_manual"),
            "sens_com_humano", percentual),
    # Estudo 4, ASR (braço P, faixa TODAS) e cascatas
    _coluna("asr_sens_v", "87,0", "arranjo", dict(braco="P", faixa="TODAS", modo=_V), "sens_auto", percentual),
    _coluna("asr_sens_c", "88,7", "arranjo", dict(braco="P", faixa="TODAS", modo=_C), "sens_auto", percentual),
    _coluna("asr_fp_v", "21", "arranjo", dict(braco="P", faixa="TODAS", modo=_V), "fp", inteiro),
    _coluna("asr_fp_c", "4", "arranjo", dict(braco="P", faixa="TODAS", modo=_C), "fp", inteiro),
    _coluna("asr_perd_v", "13", "arranjo", dict(braco="P", faixa="TODAS", modo=_V), "perdidos", inteiro),
    _coluna("asr_perd_c", "27", "arranjo", dict(braco="P", faixa="TODAS", modo=_C), "perdidos", inteiro),
    _coluna("asr_hum_v", "96", "arranjo", dict(braco="P", faixa="TODAS", modo=_V), "pares_humano", inteiro),
    _coluna("asr_hum_c", "17", "arranjo", dict(braco="P", faixa="TODAS", modo=_C), "pares_humano", inteiro),
    _coluna("v31_sens_v", "78,5", "funil", dict(versao="v3.1", modo=_V), "sens_cadeia_observada", percentual),
    _coluna("v31_sens_c", "53,4", "funil", dict(versao="v3.1", modo=_C), "sens_cadeia_observada", percentual),
    _coluna("v31_fp_v", "34", "funil", dict(versao="v3.1", modo=_V), "FP_cadeia", inteiro),
    _coluna("v31_fp_c", "0", "funil", dict(versao="v3.1", modo=_C), "FP_cadeia", inteiro),
    _coluna("v3_sens_v", "58,3", "funil", dict(versao="v3", modo=_V), "sens_cadeia_observada", percentual),
    _coluna("v3_sens_c", "40,5", "funil", dict(versao="v3", modo=_C), "sens_cadeia_observada", percentual),
    _coluna("v3_fp_v", "35", "funil", dict(versao="v3", modo=_V), "FP_cadeia", inteiro),
    _coluna("v3_fp_c", "6", "funil", dict(versao="v3", modo=_C), "FP_cadeia", inteiro),
    # Estudo 4, ordenações na mesma carga (metas: pares verdadeiros recuperados por ASR)
    _mc("asr_par2_v", "23", "O1_SCORE", _V, "K1_humana", "P_par2_recuperados"),
    _mc("asr_par2_c", "9", "O1_SCORE", _C, "K1_humana", "P_par2_recuperados"),
    _mc("o1_v", "501", "O1_SCORE", _V, _META),
    _mc("o1_c", "51", "O1_SCORE", _C, _META),
    _mc("o1_36", "2.521", "O1_SCORE", _V, "k_min_36"),
    _mc("o2_v", "4.163", "O2_CdR", _V, _META),
    _mc("o2_c", "11", "O2_CdR", _C, _META),
    _mc("o3_v", "125", "O3_p_cal_ml", _V, _META),
    _mc("o3_c", "29", "O3_p_cal_ml", _C, _META),
    _mc("o4_v", "106", "O4_P_recalibrada", _V, _META),
    _mc("o4_c", "27", "O4_P_recalibrada", _C, _META),
    _mc("o3_36", "23.443", "O3_p_cal_ml", _V, "k_min_36"),
    # Estudo 4, fusões e contexto (pares verdadeiros entre os 96 primeiros, a carga humana de ASR na vigilância)
    _ff("o1_96", "19", "O1", "par2_top_96"),
    _ff("o3_96", "22", "O3", "par2_top_96"),
    _ff("f3_96", "25", "F3", "par2_top_96"),
    _ff("f3_36", "1.777", "F3", "revisoes_36"),
    _ff("f4_96", "18", "F4", "par2_top_96"),
    _ff("f4_36", "1.387", "F4", "revisoes_36"),
    _coluna("f4_m9", "38", "boot_fusao", dict(comparacao="F4-O1", meta_nominal=9),
            "diferenca_revisoes_mediana", inteiro, abs),
    _coluna("f4_m36", "1.203", "boot_fusao", dict(comparacao="F4-O1", meta_nominal=36),
            "diferenca_revisoes_mediana", inteiro, abs),
    _coluna("cm_m36", "1.296", "boot_contexto", dict(comparacao="CM-O1", meta_nominal=36),
            "diferenca_revisoes_mediana", inteiro, abs),
    # Estudo 5
    _coluna("eq11", "42,5", "equidade",
            dict(decisao="arranjo_P_vigilancia", tipo="operacional", dimension="passo", group="passo_11"),
            "sens", percentual),
    _coluna("eq6", "95,0", "equidade",
            dict(decisao="arranjo_P_vigilancia", tipo="operacional", dimension="passo", group="passo_6"),
            "sens", percentual),
    _coluna("eq9", "96,1", "equidade",
            dict(decisao="arranjo_P_vigilancia", tipo="operacional", dimension="passo", group="passo_9"),
            "sens", percentual),
    # LLM
    _coluna("llm_acc", "25", "vpp_llm", dict(modo=_V, braco="P"), "aceitacoes_llm_abaixo", inteiro),
    _coluna("llm_vp", "4", "vpp_llm", dict(modo=_V, braco="P"), "VP", inteiro),
    _coluna("llm_fp", "21", "vpp_llm", dict(modo=_V, braco="P"), "FP", inteiro),
    _coluna("llm_vpp", "16,0", "vpp_llm", dict(modo=_V, braco="P"), "VPP", percentual),
    _coluna("llm_inf", "6,4", "vpp_llm", dict(modo=_V, braco="P"), "ic95_inf", percentual),
    _coluna("llm_sup", "34,7", "vpp_llm", dict(modo=_V, braco="P"), "ic95_sup", percentual),
    _coluna("fil_H", "22", "filtro", dict(modo=_V, k=96), "H", inteiro),
    _coluna("fil_H_c", "18", "filtro", dict(modo=_C, k=96), "H", inteiro),
    _coluna("fil_rec", "13", "filtro", dict(modo=_V, k=96), "par2_recuperados", inteiro),
    _coluna("fil_o3", "6", "filtro", dict(modo=_V, k=96), "o3_par2_mesma_carga", inteiro),
    _coluna("fil_desc", "9", "filtro", dict(modo=_V, k=96), "par2_descartados", inteiro),
    _coluna("fil_top", "22", "filtro", dict(modo=_V, k=96), "par2_topk", inteiro),
)


def conferir_numeros(fontes: dict) -> dict[str, str]:
    """Recalcula cada número do artefato e exige igualdade com o valor esperado."""
    chaves = [n.chave for n in NUMEROS]
    if len(chaves) != len(set(chaves)):
        raise ValueError("chaves de números repetidas em NUMEROS")
    valores, divergencias = {}, []
    for numero in NUMEROS:
        calculado = numero.calcular(fontes)
        if calculado != numero.esperado:
            divergencias.append(f"{numero.chave}: artefato {calculado!r}, esperado {numero.esperado!r} ({numero.fonte})")
        valores[numero.chave] = calculado
    if divergencias:
        raise ValueError("números do quadro divergem dos artefatos:\n" + "\n".join(divergencias))
    return valores


# --------------------------------------------------------------------------- afirmações

def conferir_afirmacoes(fontes: dict) -> None:
    """Afirmações sem número que o quadro faz; cada uma falha se o artefato disser outra coisa."""
    falhas: list[str] = []

    def exigir(descricao: str, condicao) -> None:
        if not bool(condicao):
            falhas.append(descricao)

    metas = (9, 23, 36)
    b = fontes["boot_o3_o1"]
    exigir("metas nominais do bootstrap são 9, 23 e 36", sorted(b.meta_nominal) == list(metas))
    exigir("IC de O3 menos O1 inclui zero nas três metas",
           len(b) == 3 and ((b.diferenca_revisoes_ic95_inf < 0) & (b.diferenca_revisoes_ic95_sup > 0)).all())

    fb = fontes["boot_fusao"]
    for comparacao in ("F4-O1", "F3-O1"):
        sup = {m: _unica(fb, comparacao=comparacao, meta_nominal=m).diferenca_revisoes_ic95_sup for m in metas}
        exigir(f"{comparacao}: IC inteiramente abaixo de zero nas metas de 9 e 36, não na de 23",
               sup[9] < 0 and sup[36] < 0 and sup[23] >= 0)
    for comparacao in ("F4-O3", "F3-O3"):
        exigir(f"{comparacao}: nenhum IC inteiramente abaixo de zero",
               (fb[fb.comparacao == comparacao].diferenca_revisoes_ic95_sup >= 0).all()
               and len(fb[fb.comparacao == comparacao]) == 3)

    cb = fontes["boot_contexto"]
    sup = {m: _unica(cb, comparacao="CM-O1", meta_nominal=m).diferenca_revisoes_ic95_sup for m in metas}
    exigir("CM-O1: IC inteiramente abaixo de zero só na meta de 36", sup[36] < 0 and sup[9] >= 0 and sup[23] >= 0)
    exigir("CM-F4: nenhum IC inteiramente abaixo de zero",
           (cb[cb.comparacao == "CM-F4"].diferenca_revisoes_ic95_sup >= 0).all() and len(cb[cb.comparacao == "CM-F4"]) == 3)

    for modo in (_V, _C):
        asr = _unica(fontes["arranjo"], braco="P", faixa="TODAS", modo=modo).sens_auto
        for versao in ("v3.1", "v3"):
            cascata = _unica(fontes["funil"], versao=versao, modo=modo).sens_cadeia_observada
            exigir(f"sensibilidade automática de ASR maior que a da cascata {versao} ({modo})", asr > cascata)

    mc = fontes["mesma_carga"]
    for modo, carga_humana in ((_V, 96), (_C, 17)):
        k1 = mc[(mc.carga == "K1_humana") & (mc.modo == modo)]
        exigir(f"a carga K1_humana é de {carga_humana} revisões na {'vigilância' if modo == _V else 'confirmação'}, "
               "em todas as ordenações e igual aos pares ao humano de ASR",
               len(k1) == 4 and (k1.k == carga_humana).all() and (k1.P_pares_humano == carga_humana).all())
    ff = fontes["fusao_filas"]
    for fila, ordenacao in (("O1", "O1_SCORE"), ("O3", "O3_p_cal_ml")):
        exigir(f"{fila}: pares verdadeiros entre os 96 primeiros iguais em fusao_filas e em comparador_mesma_carga",
               _unica(ff, fila=fila, metrica="par2_top_96").media
               == _unica(mc, ordenacao=ordenacao, modo=_V, carga="K1_humana").par2_topk)
    exigir("filtro do LLM: carga humana em k=96 é a de uma réplica (22 na vigilância, 18 na confirmação)",
           _unica(fontes["filtro"], modo=_V, k=96).H != _unica(fontes["filtro"], modo=_C, k=96).H)
    exigir("O3 exige menos revisões que O1 nas metas de ASR e mais na recuperação integral",
           all(_unica(mc, ordenacao="O3_p_cal_ml", modo=m, carga=_META).k < _unica(mc, ordenacao="O1_SCORE", modo=m, carga=_META).k
               for m in (_V, _C))
           and _unica(mc, ordenacao="O3_p_cal_ml", modo=_V, carga="k_min_36").k
           > _unica(mc, ordenacao="O1_SCORE", modo=_V, carga="k_min_36").k)
    exigir("O2 alcança a meta da confirmação com menos revisões que O3",
           _unica(mc, ordenacao="O2_CdR", modo=_C, carga=_META).k < _unica(mc, ordenacao="O3_p_cal_ml", modo=_C, carga=_META).k)

    ref = fontes["referencia"]
    corte = _unica(ref, estrategia="orl_corte_servico")
    manual = _unica(ref, estrategia="orl_corte_servico_mais_revisao_manual")
    exigir("211 aceitos pelo corte mais 36 da revisão manual de origem somam 247",
           corte.vp_automaticos + manual.vinculos_ao_humano == corte.total_vinculos)

    exigir("o Random Forest tem o maior F1 no limiar de 0,5 entre os classificadores da comparação inicial",
           max(fontes["nb01"]["all_results"], key=lambda r: r["F1-Score"])["Modelo"] == "Random Forest")
    exigir("nove estratégias de balanceamento", len(fontes["balanceamento"]) == 9)

    exigir("o filtro recupera mais pares verdadeiros que O3 na carga de 22 revisões",
           _unica(fontes["filtro"], modo=_V, k=96).par2_recuperados
           > _unica(fontes["filtro"], modo=_V, k=96).o3_par2_mesma_carga)

    if falhas:
        raise ValueError("afirmações do quadro que os artefatos não sustentam:\n" + "\n".join(falhas))


# Afirmações do quadro sem artefato em data/, conferidas só no texto da tese (documentação).
AFIRMACOES_TEXTO = (
    "Cascata v3.1: a diferença de sensibilidade automática para ASR vem dos pares verdadeiros que o corte já aceita "
    "(estudo4_gzcmd.tex, Seção sec:e4-arranjo: a cascata não aceitou automaticamente 27 e 87 dos 211 PAR=1)",
    "Random Forest sem diferença por IC 95% do Random Forest com SMOTE, do Gradient Boosting e do MLP no F1 "
    "(estudo2_classificadores.tex, comparação inicial; data/nb01_results.json traz só as estimativas pontuais)",
    "Balanceamento: sem ordenar os métodos pelo F1 médio (estudo2_classificadores.tex, Seção de balanceamento)",
    "CdR em Python: equivalência ao protótipo em Java não demonstrada (estudo1_comparador.tex)",
    "100,0% do corte mais revisão manual de origem decorre da definição de PAR (nota b de tab:cap8-referencia)",
    "Equidade: paridade demográfica não demonstrada; passo é estrato do processo (estudo5_equidade.tex)",
)


# --------------------------------------------------------------------------- conteúdo

@dataclass(frozen=True)
class Linha:
    abordagem: str
    resultado: str
    uso: str
    onde: str


# Cada grupo: (pergunta, linhas). A pergunta aparece só na primeira linha do grupo.
GRUPOS: tuple[tuple[str, tuple[Linha, ...]], ...] = (
    ("Qual comparador?", (
        Linha("CdR em Python",
              r"Não é equivalente ao protótipo em Java: nota final idêntica em <<cdr_igual>>\% dos pares de um lote sintético, valor que não estima a concordância no extrato",
              "sim, como código aberto; não como equivalente ao protótipo em Java",
              r"Capítulo~\ref{cap:estudo1}"),
        Linha("CdR, corte da nota final em 9 (cenário)",
              r"Sensibilidade automática de <<cdr9_sens>>\%, <<cdr9_fp>> falsos positivos e <<cdr9_perd>> pares verdadeiros perdidos",
              "só como comparação",
              r"Tabela~\ref{tab:cap8-referencia}"),
    )),
    ("Qual classificador e qual balanceamento?", (
        Linha("Random Forest (com ou sem SMOTE), Gradient Boosting e MLP, frente aos demais",
              r"Com limiar de 0,5, $F_1$ de <<rf_f1>> no Random Forest, sem diferença por IC 95\% para os outros três do grupo, e <<lr_f1>> na regressão logística; pela AUC-PR a ordem se inverte pontualmente (<<lr_aucpr>> na regressão logística contra <<rf_aucpr>> no Random Forest), e o Random Forest com SMOTE tem <<rfs_aucpr>>; estimativas otimistas (vazamento de alvo, partição por linha)",
              "não como base de adoção",
              r"Capítulo~\ref{cap:estudo2}"),
        Linha("Nove estratégias de balanceamento",
              r"$F_1$ médio de <<bal_min>> a <<bal_max>>, sem ordenar os métodos",
              "sem ganho demonstrado",
              r"Capítulo~\ref{cap:estudo2}"),
    )),
    ("Como decidir abaixo do corte?", (
        Linha("Corte do OpenRecLink, isolado",
              r"Aceita <<corte_acei>> dos <<tot>> pares verdadeiros (<<corte_sens>>\%); <<corte_perd>> ficam abaixo do corte",
              "sim, preservar o corte",
              r"Tabela~\ref{tab:cap8-referencia}"),
        Linha("Corte mais revisão manual de origem",
              r"<<corte_acei>> mais <<manual_rev>> pares verdadeiros; sensibilidade de <<manual_sens>>\% por definição da referência, não por acurácia observada",
              "referência operacional, não arranjo avaliado",
              r"Tabela~\ref{tab:cap8-referencia}"),
        Linha("Arranjo ASR",
              r"Sensibilidade automática de <<asr_sens_v>>\% e <<asr_sens_c>>\%, com <<asr_fp_v>> e <<asr_fp_c>> falsos positivos, <<asr_perd_v>> e <<asr_perd_c>> pares verdadeiros perdidos e <<asr_hum_v>> e <<asr_hum_c>> pares encaminhados ao revisor humano",
              "só como comparação",
              r"Tabela~\ref{tab:cap8-referencia}"),
        Linha("Cascatas v3.1 e v3, sobre a base inteira",
              r"Sensibilidade automática de <<v31_sens_v>>\% e <<v31_sens_c>>\% (v3.1, com <<v31_fp_v>> e <<v31_fp_c>> falsos positivos) e de <<v3_sens_v>>\% e <<v3_sens_c>>\% (v3, com <<v3_fp_v>> e <<v3_fp_c>>); a diferença da v3.1 para ASR vem dos pares verdadeiros aceitos pelo corte",
              "só como comparação",
              r"Tabela~\ref{tab:cap8-referencia}; Seção~\ref{sec:e4-arranjo}"),
    )),
    (r"Em que ordem revisar a \textit{grey area}?", (
        Linha("O1, escore do OpenRecLink",
              r"<<o1_v>> e <<o1_c>> revisões para recuperar os <<asr_par2_v>> e <<asr_par2_c>> pares verdadeiros de ASR; <<o1_36>> para os 36",
              "como comparador (linha de base)",
              r"Seção~\ref{sec:e4-mesma-carga}"),
        Linha("O2, nota final do CdR",
              r"<<o2_v>> e <<o2_c>> revisões para os mesmos pares",
              "só como comparação",
              r"Seção~\ref{sec:e4-mesma-carga}"),
        Linha("O3, probabilidade calibrada; O4, recalibrada de ASR",
              r"O3: <<o3_v>> e <<o3_c>> revisões (O4: <<o4_v>> e <<o4_c>>); <<o3_36>> para os 36, contra <<o1_36>> de O1; IC de O3 menos O1 inclui zero nas três metas",
              "O3, sim, escolhida pela curva de esforço; O4, comparação secundária no registro da coorte nova",
              r"Seção~\ref{sec:e4-mesma-carga}"),
        Linha("F3, fusão de O3, O1 e O3 com MACD",
              r"<<f3_96>> pares verdadeiros com as <<asr_hum_v>> revisões da vigilância (O1: <<o1_96>>; O3: <<o3_96>>) e <<f3_36>> revisões para os 36; frente a O1, IC inteiramente abaixo de zero nas metas de 9 e de 36; frente a O3, nenhum",
              "sim, quando houver as MACD",
              r"Seção~\ref{sec:e4-mesma-carga}"),
        Linha("F4, fusão de O3 e O1",
              r"<<f4_96>> pares verdadeiros com as <<asr_hum_v>> revisões da vigilância (O1: <<o1_96>>; O3: <<o3_96>>) e <<f4_36>> revisões para os 36; frente a O1, mediana de <<f4_m9>> e <<f4_m36>> revisões a menos nas metas de 9 e de 36, IC inteiramente abaixo de zero; frente a O3, nenhum",
              "sim, escolhida pela curva de esforço",
              r"Seção~\ref{sec:e4-mesma-carga}; Tabela~\ref{tab:cap8-fusao-robustez}"),
        Linha("CM, O3 com contexto por óbito e MACD",
              r"IC inteiramente abaixo de zero frente a O1 só na meta de 36 (mediana de <<cm_m36>> revisões a menos); sem vantagem demonstrada sobre F4",
              "comparação secundária no registro da coorte nova",
              r"Seção~\ref{sec:e4-mesma-carga}"),
    )),
    ("A decisão é equânime por estrato?", (
        Linha("Decisões de ASR, por passo",
              r"Sensibilidade automática de <<eq11>>\% no passo 11 (vigilância), contra <<eq6>>\% no passo 6 e <<eq9>>\% no passo 9, disparidade herdada do corte; a equidade da ordenação recomendada não foi examinada",
              "diagnóstico; paridade demográfica não demonstrada",
              r"Capítulo~\ref{cap:estudo5}"),
    )),
    ("O LLM reduz a revisão manual?", (
        Linha("LLM como aceitador (Claude Haiku 5.5, em ASR)",
              r"<<llm_acc>> aceitações abaixo do corte na vigilância, com <<llm_vp>> pares verdadeiros e <<llm_fp>> falsos positivos; VPP de <<llm_vpp>>\% (IC 95\% de <<llm_inf>> a <<llm_sup>>)",
              "não como aceitador",
              r"Seção~\ref{sec:e4-discussao}"),
        Linha(r"LLM como filtro de \texttt{NONMATCH}",
              r"Na vigilância, com <<fil_H>> revisões, <<fil_rec>> pares verdadeiros contra <<fil_o3>> de O3, mas <<fil_desc>> dos <<fil_top>> pares verdadeiros do topo da fila descartados; a carga humana de <<fil_H>> é a de uma réplica (<<fil_H_c>> na outra), e o filtro perdeu para O3 na unidade de custo da tese",
              "só como hipótese exploratória",
              r"Seção~\ref{sec:e4-filtro}"),
    )),
)

LEFT = r">{\raggedright\arraybackslash}"
LARGURAS = ("0.16", "0.12", "0.40", "0.13", "0.11")
NUM_COLUNAS = len(LARGURAS)
COLUNAS = "".join(LEFT + rf"p{{{largura}\linewidth}}" for largura in LARGURAS)
CABECALHO = r"Abordagem & Pergunta & Resultado principal & Uso recomendado & Onde \\"
LEGENDA = (r"Síntese das abordagens avaliadas nos cinco estudos, agrupadas por pergunta, "
           r"com o resultado principal, o uso recomendado e a remissão ao texto de origem.")
NOTA = (r"Quando há dois valores, o primeiro é da vigilância e o segundo, da confirmação. "
        r"As revisões e os pares verdadeiros recuperados supõem revisor humano simulado sem erros, "
        r"em análise \textit{post hoc}, frente à referência operacional. "
        r"Meta: número de pares verdadeiros a recuperar (9, 23 ou 36; 36 é a totalidade dos pares verdadeiros abaixo do corte). "
        r"CdR: Comparador de Registros. IC: intervalo de confiança de 95\%, por \textit{bootstrap} agrupado por óbito, salvo o do VPP do LLM, de Wilson. "
        r"SMOTE: sobreamostragem sintética da classe minoritária. MLP: perceptron multicamadas. "
        r"ASR: arranjo que preserva o corte do OpenRecLink e decide abaixo dele pela política de quatro ações. "
        r"Cascatas v3 e v3.1: políticas de três e de quatro ações aplicadas à base inteira, sem preservar o corte. "
        r"O1 a O4: filas ordenadas pelo escore do OpenRecLink, pela nota final do CdR, "
        r"pela probabilidade calibrada e pela recalibrada de ASR. "
        r"F3 e F4: fusões por posição recíproca. "
        r"MACD: Medidas Contínuas de Diferença de Datas de nascimento. "
        r"CM: O3 com atributos de contexto por óbito e MACD. "
        r"AUC-PR: área sob a curva precisão-sensibilidade. "
        r"Fonte: tabelas e seções da coluna Onde, com os números conferidos contra os agregados de "
        r"\path{data/} por \path{scripts/tabela_sintese_conclusao.py}.")


def _substituir(texto: str, valores: dict[str, str]) -> str:
    def trocar(achado: re.Match) -> str:
        return valores[achado.group(1)]

    return re.sub(r"<<(\w+)>>", trocar, texto)


def chaves_usadas() -> set[str]:
    textos = [NOTA, LEGENDA]
    for pergunta, linhas in GRUPOS:
        textos.append(pergunta)
        for linha in linhas:
            textos += [linha.abordagem, linha.resultado, linha.uso, linha.onde]
    return {chave for texto in textos for chave in re.findall(r"<<(\w+)>>", texto)}


def render(fontes: dict | None = None) -> str:
    fontes = carregar() if fontes is None else fontes
    valores = conferir_numeros(fontes)
    conferir_afirmacoes(fontes)
    n = str(NUM_COLUNAS)
    # longtable: o quadro quebra entre páginas; nunca dentro de float ou minipage.
    linhas = [
        r"% Gerada por scripts/tabela_sintese_conclusao.py; não editar manualmente.",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\begin{longtable}{" + COLUNAS + "}",
        r"\caption[Síntese das abordagens avaliadas]{" + LEGENDA + r"}\label{" + LABEL + r"}\\",
        r"\toprule",
        CABECALHO,
        r"\endfirsthead",
        r"\multicolumn{" + n + r"}{l}{\textit{(continuação)}} \\",
        r"\toprule",
        CABECALHO,
        r"\endhead",
        r"\midrule",
        r"\multicolumn{" + n + r"}{r}{\textit{(continua)}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for pergunta, grupo in GRUPOS:
        linhas.append(r"\midrule")
        for i, linha in enumerate(grupo):
            celulas = [linha.abordagem, pergunta if i == 0 else "", linha.resultado, linha.uso, linha.onde]
            linhas.append(" & ".join(_substituir(c, valores) for c in celulas) + r" \\")
    linhas += [
        r"\end{longtable}",
        r"\par\smallskip",
        r"\noindent " + _substituir(NOTA, valores),
        r"\par",
        r"\endgroup",
        "",
    ]
    return "\n".join(linhas)


def listar_fontes(fontes: dict | None = None) -> str:
    fontes = carregar() if fontes is None else fontes
    valores = conferir_numeros(fontes)
    saida = ["chave\tvalor\tfonte"]
    saida += [f"{n.chave}\t{valores[n.chave]}\t{n.fonte}" for n in NUMEROS]
    saida.append("")
    saida += ["Afirmações apenas de texto:"] + [f"- {a}" for a in AFIRMACOES_TEXTO]
    return "\n".join(saida)


def main(argv: list[str] | None = None) -> int:
    analisador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analisador.add_argument("--fontes", action="store_true", help="lista cada número com valor e fonte")
    args = analisador.parse_args(argv)
    if args.fontes:
        print(listar_fontes())
        return 0
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(render(), encoding="utf-8", newline="\n")
    print(DESTINO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
