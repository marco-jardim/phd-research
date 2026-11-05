"""Gera a comparação operacional da tese apenas com CSVs agregados."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def inteiro(value):
    return f"{int(value):,}".replace(",", ".")


def percentual(value):
    return f"{100 * value:.1f}".replace(".", ",")


def linhas(referencia, arranjo, cascata):
    """Cada coluna numérica, inclusive perdas, deriva de um agregado de origem."""
    rows = []
    for key, label in [
        ("orl_corte_servico", "OpenRecLink, corte isolado"),
        ("orl_corte_servico_mais_revisao_manual", "Corte + revisão manual do serviço"),
    ]:
        r = referencia.loc[referencia.estrategia.eq(key)].iloc[0]
        rows.append(("Referência operacional", label, "ambos", "n.d.", "0",
                     percentual(r.sens_automatica), inteiro(r.fp_automaticos),
                     inteiro(r.total_vinculos - r.vp_automaticos - r.vinculos_ao_humano),
                     "n.a." if pd.isna(r.sens_com_humano) else percentual(r.sens_com_humano)))
    for arm, group, label in [
        ("humano_integral", "Referência operacional", "Corte + revisão humana integral (cenário)"),
        ("P", "Resultado principal", "Arranjo sequencial P"),
    ]:
        for mode in ("vigilancia", "confirmacao"):
            if arm == "humano_integral" and mode == "confirmacao":
                continue
            r = arranjo.loc[arranjo.braco.eq(arm) & arranjo.modo.eq(mode)
                            & arranjo.faixa.eq("TODAS")].iloc[0]
            h = inteiro(r.pares_humano)
            if arm == "humano_integral":
                h = "até " + h
            rows.append((group, label, "ambos" if arm == "humano_integral" else
                         ("vigilância" if mode == "vigilancia" else "confirmação"),
                         h, inteiro(r.pares_llm), percentual(r.sens_auto), inteiro(r.fp),
                         inteiro(r.perdidos), percentual(r.sens_humano_05)))
    for version in ("v3.1", "v3"):
        for mode in ("vigilancia", "confirmacao"):
            r = cascata.loc[cascata.versao.eq(version) & cascata.modo.eq(mode)].iloc[0]
            rows.append(("Comparação sobre a base inteira", "Cascata " + version,
                         "vigilância" if mode == "vigilancia" else "confirmação",
                         inteiro(r.pares_ao_humano), inteiro(r.pares_ao_llm),
                         percentual(r.sens_cadeia_observada), inteiro(r.FP_cadeia),
                         inteiro(r.FN_nonmatch_observado), percentual(r.sens_com_humano_erro_05)))
    for key, label in [
        ("pos_orl_corte_nota_9", "CdR, corte da nota em 9"),
        ("pos_orl_limiar_nota_8", "CdR, limiar da nota em 8"),
        ("pos_orl_revisao_humana_area_cinza", "CdR, revisão humana da área cinza"),
    ]:
        r = referencia.loc[referencia.estrategia.eq(key)].iloc[0]
        rows.append(("Cenários hipotéticos do CdR", label, "ambos", inteiro(r.pares_ao_humano),
                     "0", percentual(r.sens_automatica), inteiro(r.fp_automaticos),
                     inteiro(r.total_vinculos - r.vp_automaticos - r.vinculos_ao_humano),
                     "n.a." if pd.isna(r.sens_com_humano) else percentual(r.sens_com_humano)))
    return rows


def render(referencia, arranjo, cascata):
    tex = [r"% Gerada por scripts/tabela_referencia_arranjo.py; não editar manualmente.",
           r"\begin{table}[!ht]", r"\centering\footnotesize",
           r"\caption{Referência operacional, arranjo sequencial (resultado principal), cascatas sobre a base inteira (comparação) e cenários do Comparador de Registros (CdR), nos 61.696 potenciais pares e 247 vínculos. PAR~=~1 é definido pelo próprio corte do OpenRecLink (viés de verificação).}",
           r"\label{tab:cap8-referencia}", r"\resizebox{\linewidth}{!}{%",
           r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"Estratégia & Modo & Pares ao humano & Pares ao LLM & Sens. automática (\%) & FP automáticos & Vínculos perdidos & Sens. com humano (\%) \\"]
    previous = None
    for group, *cells in linhas(referencia, arranjo, cascata):
        if group != previous:
            tex += [r"\midrule", r"\multicolumn{8}{l}{\textit{" + group + r"}} \\"]
        tex.append(" & ".join(cells) + r" \\")
        previous = group
    tex += [r"\bottomrule", r"\end{tabular}}", r"\par\smallskip",
            r"\parbox{\linewidth}{\footnotesize (a) O corte aceita 211 vínculos; a revisão manual acrescenta 36. O volume examinado e o limite inferior da revisão do serviço não estão documentados (n.d.). (b) Zero FP acima do corte é consequência do rótulo, não medida de acurácia. Em $[5,9)$ há 151 vínculos: 123 pelo corte e 28 pela revisão. Nessa faixa, o arranjo aceita 127 nos dois modos, 123 pelo corte e 4 com PAR~=~2; a cascata sobre a base inteira aceita 113 e 65. (c) A revisão integral abaixo do corte é cenário de até 61.485 potenciais pares, não carga histórica medida. Sua sensibilidade é $(211+0{,}95\times36)/247$. (d) O arranjo corresponde ao braço P do registro prévio versionado no repositório, com duas emendas e um suplemento, não a registro público. Seus FP estão abaixo do corte. Vínculos perdidos são apenas os classificados como não-vínculo, excluindo encaminhamentos pendentes. (e) A revisão da área cinza do CdR é cenário com aceitação de notas $\geq9$, descarte de notas $<5$ e sensibilidade $(93+0{,}95\times151)/247$. Todas as sensibilidades com humano, salvo a revisão observada do serviço, usam erro hipotético de 5\%, não medido. n.a.: não se aplica. Fontes: \path{data/referencia_comparador_pos_orl.csv}, \path{data/arranjo_corte_metricas.csv} e \path{data/funil_gzcmd_final_metricas.csv} \cite{Jardim2026phdrepo}.",
            r"}", r"\end{table}", ""]
    return "\n".join(tex)


def main():
    sources = [pd.read_csv(ROOT / "data" / name) for name in (
        "referencia_comparador_pos_orl.csv", "arranjo_corte_metricas.csv",
        "funil_gzcmd_final_metricas.csv")]
    target = ROOT / "text/latex/tables/tab_cap8_referencia.tex"
    target.write_text(render(*sources), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
