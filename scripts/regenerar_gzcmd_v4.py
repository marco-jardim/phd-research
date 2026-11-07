"""Preserve M3v3 outputs, regenerate against Haiku v4 only, report deltas offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.artifacts import preserve_m3v3_artifacts
from gzcmd.reliability import reference_rates_model
from scripts.funil_gzcmd_v2 import artifact_paths, HAIKU_V4


def summarize() -> None:
    metrics = pd.read_csv(ROOT / "data/funil_gzcmd_metricas.csv")
    losses = pd.read_csv(ROOT / "data/perda_total_por_modelo.csv")
    before = pd.read_csv(ROOT / "data/varredura_custos_antes_depois_m3v3.csv")
    after = pd.read_csv(ROOT / "data/varredura_custos_antes_depois.csv")
    fields = ["lote_llm", "humano_direto", "AUTO_MATCH", "AUTO_NONMATCH", "vinculos_perdidos_auto_nonmatch",
              "FP_automaticos", "sens_esperada_automatica", "sens_esperada_com_humano", "perda_esperada_total"]
    rows = []
    for mode in ("vigilancia", "confirmacao"):
        old = before[before.modo.eq(mode) & before.cenario.eq("depois")].iloc[0]
        new = after[after.modo.eq(mode) & after.cenario.eq("depois")].iloc[0]
        for field in fields:
            rows.append({"modo": mode, "metrica": field, "m3v3": old[field], "haiku_v4": new[field],
                         "diferenca_v4_m3v3": new[field] - old[field]})
    delta = pd.DataFrame(rows)
    delta.to_csv(ROOT / "data/comparacao_taxas_m3v3_v4.csv", index=False)
    batches, gold_rows = [], []
    for mode in ("vigilancia", "confirmacao"):
        batch = pd.read_csv(ROOT / f"data/lote_llm_v31_{mode}.csv")
        pending = pd.read_csv(ROOT / f"data/lote_llm_v31_{mode}_pendentes.csv")
        batches.append({"modo": mode, "lote_llm_v31": len(batch), "respondidos_haiku_v4": len(batch) - len(pending),
                        "pendentes_haiku_v4": len(pending)})
        old_funnel = pd.read_csv(ROOT / f"data/funil_gzcmd_antes_{mode}.csv", low_memory=False)
        reviewed = old_funnel.loc[old_funnel.action.eq("LLM_REVIEW")]
        gold_rows.append({"modo": mode, "n": len(reviewed),
                          "VP": int((reviewed.llm_decision.eq("MATCH") & reviewed.y.eq(1)).sum()),
                          "FP": int((reviewed.llm_decision.eq("MATCH") & reviewed.y.eq(0)).sum()),
                          "FN": int((reviewed.llm_decision.eq("NONMATCH") & reviewed.y.eq(1)).sum()),
                          "UNSURE": int(reviewed.llm_decision.eq("UNSURE").sum())})
    batches = pd.DataFrame(batches)
    batches.to_csv(ROOT / "data/lote_llm_v31_resumo.csv", index=False)
    gold_table = pd.DataFrame(gold_rows)
    gold_table.to_csv(ROOT / "data/haiku_v4_gabarito_lote_antigo.csv", index=False)
    sources = []
    for mode in ("vigilancia", "confirmacao"):
        for path in artifact_paths(mode, HAIKU_V4, include_v31=True):
            sources.append({"mode": mode, "path": path.relative_to(ROOT).as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    reference = losses[losses.revisor_referencia & losses.versao.eq("v3.1")]
    if reference.rates_sha256.nunique() != 1 or not reference.rates_model.eq("claude-haiku-5-5").all():
        raise ValueError("Regeneration mixes reference reliability sources")
    metadata = {"seed": 42, "rates_yaml": reference.rates_yaml.iloc[0],
                "rates_sha256": reference.rates_sha256.iloc[0], "rates_model": "claude-haiku-5-5",
                "reference_runs": sources,
                "oof_sha256": hashlib.sha256((ROOT / "data/pcal_ml_oof.csv").read_bytes()).hexdigest(),
                "archive_manifest_sha256": hashlib.sha256((ROOT / "data/arquivos_preservados_m3v3.json").read_bytes()).hexdigest(),
                "approved_costs": {"llm_review": .005, "human_review": 1., "human_error": .05, "human_budget": None}}
    (ROOT / "data/regeneracao_gzcmd_v4_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = ["# Reavaliação com Claude Haiku 5.5, prompt v4 e dossiê sem model_outputs", "",
              "A referência é somente by_mode.<modo>.models.claude-haiku-5-5 do YAML v4. O pooled misto com DeepSeek V4.1 Flash não foi usado.",
              "Custos aprovados preservados: C_llm=0,005; C_humano=1; e_h=0,05; humano direto ilimitado.",
              "OOF, guardas e custos permanecem fixos. O perfil v3 mantém Platt/taxas fixas/custo LLM=0,25 históricos; v3.1 usa as taxas Haiku v4.",
              "Kimi K2.5 e Claude Haiku 5.5 v3 com MO são comparadores históricos. Suas pendências usam taxas históricas próprias, identificadas no CSV; nunca alimentam o motor novo.",
              "A referência não é substituída pelo comparador de menor perda. Scores do motor no dossiê antigo tornam essa comparação potencialmente contaminada.",
              "A perda mistura erros conhecidos e tempo de encaminhamento observados com perdas esperadas dos pendentes e do revisor humano imperfeito.",
              "A política v3.1 tem cobertura Haiku v4 integral após incorporar o lote complementar. Não há inferência de respostas LLM pendentes na referência.",
              "A projeção anterior foi preservada em colunas pre_lote31. O cenário com revisor humano continua sendo hipótese de erro de 5%, separado do desfecho automático observado.",
              "Os deltas abaixo usam expectativas da varredura, sem replay, no mesmo ponto C_llm=0,005/e_h=0,05/orçamento infinito. Frente ao arquivo M3v3, podem combinar alterações de taxas e de perfil OOF; não isolam um efeito causal.",
              "Arquivos anteriores preservados com sufixo _m3v3 e hashes em arquivos_preservados_m3v3.json; nenhuma resposta LLM foi gerada.", ""]
    tables = [
        ("Gabarito do Haiku v4 no lote antigo, sem decisões automáticas do motor", gold_table),
        ("Funil v3 e v3.1, referência Haiku v4", metrics[["modo", "versao", "modelo", "LLM", "AUTO_MATCH", "AUTO_MATCH_FP", "AUTO_NONMATCH_vinc_cinza", "DEFERRED", "UNSURE", "sem_resposta_LLM", "sens_observada", "sens_esperada", "sens_esperada_com_humano", "perda"]]),
        ("Perda por revisor", losses[["modo", "versao", "modelo", "FP", "FN", "UNSURE", "DEFERRED", "sem_resposta_LLM", "perda_observada", "perda_esperada", "perda", "revisor_referencia"]]),
        ("Lotes para revisão", batches), ("Diferenças frente às taxas anteriores", delta),
        ("Confirmação NP", pd.read_csv(ROOT / "data/comparacao_np_confirmacao.csv")),
    ]
    final_path = ROOT / "data/funil_gzcmd_final_metricas.csv"
    if final_path.exists():
        tables.append(("Desfecho final da aceitação automática", pd.read_csv(final_path)))
    for title, table in tables:
        print(f"\n{title}\n{table.to_string(index=False)}")
        report.extend([f"## {title}", "```", table.to_string(index=False), "```", ""])
    report.extend(["## Fontes v4", "```json", json.dumps(sources, indent=2), "```",
                   "Relatórios detalhados: funil_gzcmd_relatorio.md, varredura_custos_relatorio.md e np_confirmacao_relatorio.md.",
                   "A viabilidade NP deve ser lida na tabela recalculada. A alteração do perfil OOF pode mudar o risco das guardas, embora trocar apenas taxas LLM não o altere."])
    (ROOT / "data/regeneracao_gzcmd_v4_relatorio.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates-yaml", type=Path, default=ROOT / "gzcmd/error_rates_by_band_v4.yaml")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if reference_rates_model(args.rates_yaml) != "claude-haiku-5-5":
        raise ValueError("This regeneration requires the Haiku v4 reference block")
    archive = preserve_m3v3_artifacts(ROOT)
    print("PRESERVADOS M3V3", len(archive["files"]))
    if not args.summary_only:
        calls = [("scripts/funil_gzcmd_v2.py", ["--reuse-oof", "--policy-kind", "cost"]),
                 ("scripts/fig_varredura_custos.py", ["--policy-kind", "cost"]),
                 ("scripts/fig_varredura_custos.py", ["--policy-kind", "np"])]
        for script, options in calls:
            subprocess.run([sys.executable, script, "--rates-yaml", str(args.rates_yaml), *options], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "scripts/fig_cascata_v31.py"], cwd=ROOT, check=True)
    summarize()


if __name__ == "__main__":
    main()
