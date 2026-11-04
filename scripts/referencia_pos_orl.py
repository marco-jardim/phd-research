"""Comparador de referência pós-ORL: contagens por faixa de nota e hipótese de erro humano.

Lê o extrato anonimizado (data/COMPARADORSEMIDENT.csv, não versionado), conta pares e
vínculos (PAR em {1, 2}) por faixa da nota final do Comparador de Registros e grava
data/referencia_comparador_pos_orl.csv, sem linhas por par.

Estratégias:
  * corte do OpenRecLink aplicado pelo serviço (referência operacional): PAR = 1 são os pares aceitos
    pelo ponto de corte do escore probabilístico; PAR = 2 são os vínculos recuperados por revisão manual
    abaixo do corte. O corte é inferido por passo de blocagem (menor SCORE entre os PAR = 1 do passo) e
    a coerência é verificada no extrato. Como o rótulo PAR = 1 é definido pelo próprio corte, a
    precisão do corte é 100% por construção (viés de verificação);
  * revisão humana integral da área cinza [5, 9), com aceitação automática dos pares com nota >= 9;
  * corte da nota em 9 (nada vai ao humano);
  * limiar >= 8 (opcional, referência do corte convencional da tese).

A sensibilidade com humano usa a hipótese de erro humano simétrico de 5%: um vínculo encaminhado
ao humano é recuperado com probabilidade 0,95, e um não-vínculo encaminhado é aceito por engano
com probabilidade 0,05. Não há decisão humana observada.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ERRO_HUMANO = 0.05
COL_COMPREC = "COMPREC"
COL_REFREC = "REFREC"


def carregar(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, sep=";", dtype=str)
    # Cabeçalho do extrato: "NOME,C,12,0" (formato DBF); mantém só o nome do campo.
    df.columns = [c.split(",")[0] for c in df.columns]
    nota = pd.to_numeric(df["nota final"].str.replace(",", ".", regex=False), errors="raise")
    par = pd.to_numeric(df["PAR"], errors="raise")
    score = pd.to_numeric(df["SCORE"].str.replace(",", ".", regex=False), errors="raise")
    return pd.DataFrame(
        {
            "nota": nota,
            "vinculo": par.isin([1, 2]),
            "par": par,
            "passo": df["PASSO"],
            "score_orl": score,
            "comprec": df[COL_COMPREC],
            "refrec": df[COL_REFREC],
            "ano_obito": df["R_DTOBITO"].str[-4:],
        }
    )


def linha(estrategia: str, humano: pd.DataFrame, auto: pd.DataFrame, total_vinculos: int) -> dict:
    vp = int(auto["vinculo"].sum())
    fp = int((~auto["vinculo"]).sum())
    v_h = int(humano["vinculo"].sum())
    n_h = len(humano)
    sens_auto = vp / total_vinculos
    sens_humano = (vp + (1 - ERRO_HUMANO) * v_h) / total_vinculos if n_h else float("nan")
    return {
        "estrategia": estrategia,
        "pares_ao_humano": n_h,
        "vinculos_ao_humano": v_h,
        "pares_aceitos_automaticamente": len(auto),
        "vp_automaticos": vp,
        "fp_automaticos": fp,
        "total_vinculos": total_vinculos,
        "sens_automatica": sens_auto,
        "erro_humano_hipotese": ERRO_HUMANO if n_h else float("nan"),
        "sens_com_humano": sens_humano,
        "fp_humanos_esperados_hipotese": ERRO_HUMANO * (n_h - v_h) if n_h else float("nan"),
    }


def cortes_orl(d: pd.DataFrame) -> pd.Series:
    """Corte do OpenRecLink por passo: menor SCORE entre os pares PAR = 1 do passo."""
    return d[d.par == 1].groupby("passo")["score_orl"].min()


def linha_orl(d: pd.DataFrame, total_vinculos: int) -> tuple[list[dict], dict]:
    """Linhas da referência operacional (corte do ORL) e verificação de coerência do corte."""
    cortes = cortes_orl(d)
    acima = d["score_orl"] >= d["passo"].map(cortes)
    auto = d[acima]
    abaixo = d[~acima]
    revisados = abaixo[abaixo.par == 2]
    # Coerência: dentro do passo, PAR = 1 se e somente se SCORE >= corte inferido.
    incoerentes = int((acima != (d.par == 1)).sum())
    base = linha("orl_corte_servico", d.iloc[0:0], auto, total_vinculos)
    base["pares_ao_humano"] = float("nan")  # número de pares revisados manualmente não documentado
    com_revisao = linha("orl_corte_servico_mais_revisao_manual", revisados, auto, total_vinculos)
    com_revisao["pares_ao_humano"] = float("nan")  # idem: só os 36 vínculos são observados
    com_revisao["erro_humano_hipotese"] = 0.0  # decisão humana observada (PAR = 2), sem hipótese de erro
    com_revisao["sens_com_humano"] = (int(auto["vinculo"].sum()) + len(revisados)) / total_vinculos
    com_revisao["fp_humanos_esperados_hipotese"] = float("nan")
    meta = {
        "n_orl_aceitos_pelo_corte": len(auto),
        "n_orl_vinculos_revisao_manual": len(revisados),
        "n_orl_nao_vinculos_acima_corte": int((~auto["vinculo"]).sum()),
        "n_orl_pares_incoerentes_com_corte_por_passo": incoerentes,
        "orl_score_min_par2": float(d[d.par == 2]["score_orl"].min()),
        "orl_score_max_par2": float(d[d.par == 2]["score_orl"].max()),
    }
    for passo, valor in cortes.items():
        maximo_nao_aceito = float(d[(d.passo == passo) & (d.par != 1)]["score_orl"].max())
        meta[f"orl_passo{passo}_min_par1"] = float(valor)
        meta[f"orl_passo{passo}_max_nao_par1"] = maximo_nao_aceito
    return [base, com_revisao], meta


def resumo_extrato(d: pd.DataFrame) -> pd.DataFrame:
    """Contagens agregadas do extrato (sem linhas por par, sem datas individuais)."""
    anos = pd.to_numeric(d["ano_obito"], errors="coerce")
    itens = {
        "n_pares_candidatos": len(d),
        "n_notificacoes_comprec_distintos": d["comprec"].nunique(),
        "n_obitos_refrec_distintos": d["refrec"].nunique(),
        "n_par0": int((d.par == 0).sum()),
        "n_par1": int((d.par == 1).sum()),
        "n_par2": int((d.par == 2).sum()),
        "n_vinculos_pares_distintos": len(d[d.vinculo][["comprec", "refrec"]].drop_duplicates()),
        "ano_obito_min": int(anos.min()),
        "ano_obito_max": int(anos.max()),
        "n_sem_data_obito": int(anos.isna().sum()),
    }
    for ano, n in anos.value_counts().sort_index().items():
        itens[f"n_pares_ano_obito_{int(ano)}"] = int(n)
    return pd.DataFrame({"metrica": list(itens), "valor": list(itens.values())})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--extrato",
        "--csv",
        dest="csv",
        type=Path,
        default=Path(os.environ.get("EXTRATO_CSV", ROOT / "data/COMPARADORSEMIDENT.csv")),
        help="extrato anonimizado (padrão: variável EXTRATO_CSV ou data/COMPARADORSEMIDENT.csv)",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data/referencia_comparador_pos_orl.csv")
    ap.add_argument("--resumo", type=Path, default=ROOT / "data/extrato_resumo.csv")
    args = ap.parse_args()

    d = carregar(args.csv)
    total = int(d["vinculo"].sum())
    cinza = d[(d.nota >= 5) & (d.nota < 9)]
    alto = d[d.nota >= 9]
    ge8 = d[d.nota >= 8]
    vazio = d.iloc[0:0]
    linhas_orl, meta_orl = linha_orl(d, total)
    linhas = linhas_orl + [
        linha("pos_orl_revisao_humana_area_cinza", cinza, alto, total),
        linha("pos_orl_corte_nota_9", vazio, alto, total),
        linha("pos_orl_limiar_nota_8", vazio, ge8, total),
    ]
    papel = {
        "orl_corte_servico": "operacional",
        "orl_corte_servico_mais_revisao_manual": "operacional",
    }
    for r in linhas:
        r["referencia"] = papel.get(r["estrategia"], "hipotetica")
    faixas = {
        "n_nota_lt5": int((d.nota < 5).sum()),
        "vinculos_nota_lt5": int(d[d.nota < 5]["vinculo"].sum()),
        "n_area_cinza": len(cinza),
        "vinculos_area_cinza": int(cinza["vinculo"].sum()),
        "n_nota_ge9": len(alto),
        "vinculos_nota_ge9": int(alto["vinculo"].sum()),
        "n_total": len(d),
    }
    out = pd.DataFrame(linhas)
    for k, v in {**faixas, **meta_orl}.items():
        out[k] = v
    out.to_csv(args.out, index=False)
    resumo = resumo_extrato(d)
    resumo.to_csv(args.resumo, index=False)
    print(out.T.to_string())
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()
