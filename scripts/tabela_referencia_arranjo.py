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
        rows.append(("Referência operacional", label, "ambos", "0" if key == "orl_corte_servico" else "n.d.", "0",
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


def notas(referencia, arranjo):
    """Resultados numéricos das notas seguem os mesmos agregados das linhas."""
    cut = referencia.loc[referencia.estrategia.eq("orl_corte_servico")].iloc[0]
    manual = referencia.loc[referencia.estrategia.eq("orl_corte_servico_mais_revisao_manual")].iloc[0]
    grey = referencia.loc[referencia.estrategia.eq("pos_orl_revisao_humana_area_cinza")].iloc[0]
    subset = arranjo.loc[arranjo.faixa.eq("[5,9)")]
    row = lambda arm, mode="vigilancia": subset.loc[subset.braco.eq(arm) & subset.modo.eq(mode)].iloc[0]
    kept = int(row("corte_sozinho").aceitos)
    total = int(grey.total_vinculos)
    human = arranjo.loc[arranjo.braco.eq("humano_integral") & arranjo.faixa.eq("TODAS")].iloc[0]
    recovered = 1 - grey.erro_humano_hipotese
    recovered_tex = f"{recovered:.2f}".replace(".", "{,}")
    return (
        r"\parbox{\linewidth}{\footnotesize "
        f"(a) O corte aceita {inteiro(cut.vp_automaticos)} vínculos; a revisão manual acrescenta {inteiro(manual.vinculos_ao_humano)}. "
        "O volume examinado e o limite inferior da revisão do serviço não estão documentados (n.d.). "
        f"(b) Os {percentual(manual.sens_com_humano)}\\% da linha corte + revisão manual decorrem da definição de PAR, não de acurácia observada. "
        "Zero FP acima do corte também decorre do rótulo. "
        f"Em $[5,9)$ há {inteiro(grey.vinculos_area_cinza)} vínculos: {kept} pelo corte e {int(grey.vinculos_area_cinza)-kept} pela revisão. "
        f"Nessa faixa, o arranjo aceita {inteiro(row('P').aceitos)} e {inteiro(row('P', 'confirmacao').aceitos)} nos dois modos, "
        f"incluindo {kept} pelo corte em cada modo; a cascata sobre a base inteira aceita {inteiro(row('publicada').aceitos)} e {inteiro(row('publicada', 'confirmacao').aceitos)}. "
        f"(c) A revisão integral abaixo do corte é cenário de até {inteiro(human.pares_humano)} potenciais pares, não carga histórica medida. "
        f"Sua sensibilidade é $({inteiro(cut.vp_automaticos)}+{recovered_tex}\\times{inteiro(manual.vinculos_ao_humano)})/{total}$. "
        "(d) O arranjo corresponde ao braço P do registro prévio versionado no repositório, com emendas e suplemento, não a registro público. "
        "Seus FP estão abaixo do corte. Vínculos perdidos são apenas os classificados como não-vínculo, excluindo encaminhamentos pendentes. "
        r"(e) A revisão da área cinza do CdR é cenário com aceitação de notas $\geq9$ e descarte de notas $<5$, "
        f"com sensibilidade $({inteiro(grey.vp_automaticos)}+{recovered_tex}\\times{inteiro(grey.vinculos_area_cinza)})/{total}$. "
        f"Todas as sensibilidades com humano, salvo a revisão do serviço incorporada ao rótulo, usam erro hipotético de {percentual(grey.erro_humano_hipotese)}\\%, não medido. "
        r"n.a.: não se aplica. Fontes: \path{data/referencia_comparador_pos_orl.csv}, \path{data/arranjo_corte_metricas.csv} e \path{data/funil_gzcmd_final_metricas.csv} \cite{Jardim2026phdrepo}."
    )


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
            notas(referencia, arranjo),
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
