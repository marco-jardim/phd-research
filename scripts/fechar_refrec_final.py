"""Reestimate once, freeze rates, export any newly selected pairs, finalize offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gzcmd.config import load_config
from gzcmd.provenance import sha256_file_normalized
from gzcmd.runner import run_v3
from scripts.funil_gzcmd_v2 import HAIKU_V4, artifact_paths, load_responses, safe_funnel
from scripts.tabelas_arranjo_editorial import replace_once


def command(script, *args):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, check=True)


def rates_digest():
    return sha256_file_normalized(ROOT / "gzcmd/error_rates_by_band_v4.yaml")


def reestimate_and_export():
    command("avaliar_reruns_exp2.py")
    metadata = {"rates_sha256": rates_digest(), "default_split_by": "refrec", "seed": 42,
                "rates_note": "Reestimated from historical + row supplement + initial REFREC supplement, then frozen before closure responses"}
    for mode in ("vigilancia", "confirmacao"):
        result, _ = run_v3(input_csv=ROOT / "data/COMPARADORSEMIDENT.csv", config_path=ROOT / "gzcmd/gzcmd_v3_config.yaml",
                            mode=mode, p_cal="ml_xgb", policy_version="v3.1", reuse_ml_oof=True, policy_kind="cost")
        frame = safe_funnel(result)
        responses, _ = load_responses(mode, HAIKU_V4, include_v31=True)
        pending = frame.loc[frame.action.eq("LLM_REVIEW") & ~frame.index.isin(responses),
                            ["nota", "p_cal", "band", "decisao_motor"]].rename(columns={"p_cal": "p_cal_ml", "band": "banda"}).reset_index()
        pending["motivo"] = "selecao_apos_reestimar_taxas"
        pending.to_csv(ROOT / f"data/lote_llm_v31_refrec_{mode}_fechamento.csv", index=False)
        metadata[mode] = {"llm": int(frame.action.eq("LLM_REVIEW").sum()), "extra_pending": len(pending)}
        print("REFREC_POST_RATE_PENDING", mode, len(pending), flush=True)
    (ROOT / "data/refrec_fechamento_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def regenerate_cascade_table(output_path=None):
    table = pd.read_csv(ROOT / "data/funil_gzcmd_final.csv")
    metrics = pd.read_csv(ROOT / "data/funil_gzcmd_final_metricas.csv")
    order = [(mode, version) for mode in ("vigilancia", "confirmacao") for version in ("v3", "v3.1")]
    path = ROOT / "text/latex/tables/tab_cap8_cascata_v31.tex"
    lines = path.read_text(encoding="utf-8").splitlines()
    captions = [line for line in lines if line.startswith(r"\caption{")]
    if len(captions) != 1 or not captions[0].startswith(r"\caption{Cascata final observada por política e modo,"):
        raise ValueError("Legenda histórica da cascata ausente ou ambígua")

    def number(value, decimals=0):
        return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")

    def cell(mode, version, band, level, stage):
        p = table.loc[table.modo.eq(mode) & table.versao.eq(version) & table.faixa.eq(band)
                      & table.nivel.eq(level) & table.etapa.eq(stage)]
        if len(p) != 1:
            raise ValueError(f"Ambiguous cascade cell: {mode}/{version}/{band}/{stage}")
        return f"{number(p.iloc[0].n)} ({number(p.iloc[0].vinculos)})"

    metric_rows = {
        "VP aceitos sem revisão humana": ("VP_cadeia", 0, False),
        "FP aceitos sem revisão humana": ("FP_cadeia", 0, False),
        "Vínculos perdidos (NONMATCH)": ("FN_resolvidos", 0, False),
        "Vínculos encaminhados ao humano": ("vinculos_ao_humano", 0, False),
        r"Especificidade automática (\%)": ("espec_cadeia_observada", 3, True),
        "Pares submetidos ao LLM": ("pares_ao_llm", 0, False),
        "Pares encaminhados ao humano": ("pares_ao_humano", 0, False),
        r"Sens. com humano, erro 5,0\% (\%)": ("sens_com_humano_erro_05", 1, True),
        r"Custo LLM estimado (US\$)": ("custo_llm_usd", 2, False)}
    band = None
    for index, line in enumerate(lines):
        label = line.split(" & ")[0]
        values = None
        if line.startswith(r"\caption{") and "REFREC" not in line:
            lines[index] = replace_once(line, "Cascata final observada por política e modo,", "Cascata final observada por política e modo, com OOF por REFREC na v3.1,")
            continue
        if label == "Universo avaliado":
            values = [cell(m, v, "TODAS", 0, "pares_revisados") for m, v in order]
        elif label.startswith(r"\textbf{Nota"):
            band = "<5" if "$<5$" in label else "[5,9)" if "$[5,9)$" in label else ">=9"
            values = [cell(m, v, band, 1, "faixa") for m, v in order]
        elif label.startswith(r"\qquad ") and band:
            stage = label.removeprefix(r"\qquad ")
            values = [cell(m, v, band, 3, "LLM_" + stage) for m, v in order]
        elif label.startswith(r"\quad ") and band:
            stage = label.removeprefix(r"\quad ").replace(r"\_", "_")
            stage = "HUMANO_DIRETO" if stage == "Humano direto" else stage
            values = [cell(m, v, band, 2, stage) for m, v in order]
        elif label == "Subtotal humano" and band:
            values = [cell(m, v, band, 4, "HUMANO_TOTAL") for m, v in order]
        elif label.startswith("Sensibilidade automática"):
            values = []
            for mode, version in order:
                row = metrics.loc[metrics.modo.eq(mode) & metrics.versao.eq(version)].iloc[0]
                values.append(f"{number(100 * row.sens_cadeia_observada, 1)} ({number(100 * row.sens_ci95_low, 1)}; {number(100 * row.sens_ci95_high, 1)})")
        elif label in metric_rows:
            field, decimals, percent = metric_rows[label]
            values = [number(metrics.loc[metrics.modo.eq(m) & metrics.versao.eq(v)].iloc[0][field] * (100 if percent else 1), decimals) for m, v in order]
        if values is not None:
            lines[index] = label + " & " + " & ".join(values) + r" \\"
    (output_path or path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize():
    frozen = json.loads((ROOT / "data/refrec_fechamento_metadata.json").read_text(encoding="utf-8"))
    if frozen["rates_sha256"] != rates_digest():
        raise ValueError("Reliability changed after closure selection")
    command("regenerar_gzcmd_v4.py")
    command("auditar_funil_adversarial.py")
    regenerate_cascade_table()
    metrics = pd.read_csv(ROOT / "data/funil_gzcmd_final_metricas.csv")
    row = pd.read_csv(ROOT / "data/funil_gzcmd_final_metricas_row.csv")
    comparison = pd.concat([row.assign(split_by="row"), metrics.assign(split_by="refrec")], ignore_index=True)
    if "sens_com_humano_erro_05" not in row:
        selected = comparison.split_by.eq("row")
        comparison.loc[selected, "sens_com_humano_erro_05"] = (comparison.loc[selected, "VP_cadeia"] + .95 * comparison.loc[selected, "vinculos_ao_humano"]) / (comparison.loc[selected, "VP_cadeia"] + comparison.loc[selected, "FN_cadeia"])
    comparison.to_csv(ROOT / "data/funil_gzcmd_linha_vs_refrec.csv", index=False)
    usage = []
    for directory in ("anthropic_sem_mo_v4_lote31_refrec", "anthropic_sem_mo_v4_lote31_refrec_fechamento"):
        for mode in ("vigilancia", "confirmacao"):
            path = ROOT / "data/sprint3b" / directory / f"exp2_llm_sync_{mode}.json"
            if not path.exists():
                continue
            run = json.loads(path.read_text(encoding="utf-8"))
            if not run["complete"]:
                raise ValueError("Incomplete paid cohort")
            usage.append({"lote": directory, "modo": mode, "n": len(run["results"]), "elapsed_s": run["elapsed_s"],
                          "cost_estimate_usd": run["cost_estimate_usd"], **{k: v for k, v in run["transport"].items() if not isinstance(v, dict)}})
    use_table = pd.DataFrame(usage)
    use_table.to_csv(ROOT / "data/refrec_rerun_consumo.csv", index=False)
    profile = yaml.safe_load((ROOT / "gzcmd/error_rates_by_band_v4.yaml").read_text(encoding="utf-8"))
    bands = pd.DataFrame([{"modo": mode, "band": "grey_low", **profile["by_mode"][mode]["models"]["claude-haiku-5-5"]["estimates_by_band"]["grey_low"]} for mode in ("vigilancia", "confirmacao")])
    bands.to_csv(ROOT / "data/refrec_taxas_grey_low.csv", index=False)
    assert metrics.llm_sem_resposta.eq(0).all()
    report = ["# Fechamento observado v3.1 por REFREC", "", "Padrão REFREC adotado por decisão do usuário. OOF aninhado, seed 42; perfil histórico v3/Platt preservado.",
              "Taxas Haiku v4 reestimadas uma vez incluindo os suplementos por linha e REFREC inicial; congeladas antes de quaisquer chamadas adicionais de fechamento.",
              "Classificador e calibração são OOF por grupo. As taxas LLM são estimadas em coortes selecionadas adaptativamente; o replay observado não é validação independente das taxas.",
              "Todas as decisões LLM da cascata final são observadas. UNSURE é encaminhamento humano. Sensibilidade após revisão humana é projeção com erro hipotético de 5%, não resultado humano observado.",
              "FN_resolvidos são vínculos classificados NONMATCH. FN_cadeia inclui ainda vínculos encaminhados ao humano, não aceitos automaticamente.",
              "Custo em US$ aplica a tarifa histórica aos tokens registrados por par; não é fatura. O custo alocado à cascata não precisa coincidir com o gasto total dos lotes executados.",
              "A comparação linha/REFREC combina mudança do split e reestimação das taxas; não isola causalmente apenas o split. A comparação com taxas fixas permanece no relatório OOF por grupo."]
    cols = ["split_by", "modo", "versao", "VP_cadeia", "FP_cadeia", "FN_resolvidos", "pares_ao_humano", "pares_ao_llm", "sens_cadeia_observada", "sens_ci95_low", "sens_ci95_high", "sens_com_humano_erro_05", "custo_llm_usd"]
    for title, table in (("Comparação por linha e REFREC", comparison[cols]), ("Taxas grey_low", bands), ("Consumo completo dos reruns", use_table)):
        report.extend(["\n## " + title, "```", table.to_string(index=False), "```"])
    report.extend(["\nArtefatos por linha preservados com sufixo _row e manifesto SHA256 em data/arquivos_preservados_row.json.",
                   "Nenhum capítulo foi editado. Os CSVs de pendentes não contêm rótulos, grupos ou campos pessoais.",
                   "Reprodução offline após os runs: python scripts/fechar_refrec_final.py --finalize. Não reestimar novamente após coletar o fechamento sem avaliar a nova seleção."])
    (ROOT / "data/funil_gzcmd_refrec_final_relatorio.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reestimate", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.reestimate:
        reestimate_and_export()
    if args.finalize:
        finalize()
