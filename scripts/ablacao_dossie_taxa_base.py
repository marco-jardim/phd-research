"""Whole-cohort decision rates beside the selected 18-pair ablation arms.

The complete historical cohort is 1,410 vigilance and 435 confirmation reviews.
Supplementary cohorts are deliberately excluded. The historical sem_pcal name
denotes removal of model_outputs only, preserving every other dossier field.
The selected arms use the original prompt; their aggregates do not record its
version. Whole-cohort rates are descriptive, not a negative control.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ("MATCH", "NONMATCH", "UNSURE")
COHORTS = ("anthropic", "anthropic_sem_mo", "anthropic_sem_mo_v4",
           "opencode_go", "opencode_go_sem_mo", "opencode_go_sem_mo_v4")


def cohort_counts(directory: Path) -> tuple[dict, dict]:
    counts = Counter()
    hashes = {}
    metadata = []
    for mode, expected in (("vigilancia", 1410), ("confirmacao", 435)):
        path = directory / f"exp2_llm_sync_{mode}.json"
        raw = path.read_bytes()
        run = json.loads(raw)
        rows = run["results"]
        ids = {row["pair_id"] for row in rows}
        if len(rows) != expected or len(ids) != expected:
            raise ValueError(f"Historical cohort is incomplete or internally duplicated: {directory.name}/{mode}")
        # A pair may be reviewed under both modes. The denominator is reviews
        # (mode, pair), not unique pairs pooled across modes.
        counts.update(row["decision"] for row in rows)
        hashes[f"{directory.name}/{path.name}"] = hashlib.sha256(raw).hexdigest()
        metadata.append((run["model"], run.get("prompt_version", "v3")))
    if len(set(metadata)) != 1 or set(counts) - set(DECISIONS):
        raise ValueError("Inconsistent cohort metadata/decisions")
    return {"model": metadata[0][0], "prompt": metadata[0][1], "n": sum(counts.values()),
            **{d: counts[d] for d in DECISIONS}}, hashes


def generate(data: Path, output: Path | None = None) -> pd.DataFrame:
    rows = []
    base = data / "sprint3b"
    paired = json.loads((base / "ablacao/resumo.json").read_text(encoding="utf-8"))["models"]
    summaries = list(paired.values()) if isinstance(paired, dict) else paired
    for cohort in COHORTS:
        counts, hashes = cohort_counts(base / cohort)
        aggregate = base / "ablacao" / f"{counts['model']}_controle_agregado.json"
        control = json.loads(aggregate.read_text(encoding="utf-8")) if aggregate.exists() else None
        ablated_path = base / "ablacao" / f"{counts['model']}_sem_pcal_agregado.json"
        ablated = json.loads(ablated_path.read_text(encoding="utf-8")) if ablated_path.exists() else None
        row = {"coorte": cohort, "modelo": counts["model"], "prompt": counts["prompt"],
               "unidade": "revisão (modo, par); modos podem compartilhar pares",
               "campos_motor": "sem model_outputs" if "sem_mo" in cohort else "com model_outputs",
               "n": counts["n"], "fontes_sha256": json.dumps(hashes, sort_keys=True)}
        for decision in DECISIONS:
            row[decision] = counts[decision]
            row[f"taxa_{decision}"] = counts[decision] / counts["n"]
            row[f"controle_18_{decision}"] = control["decisions"][decision] if control else None
            row[f"sem_pcal_18_{decision}"] = ablated["decisions"][decision] if ablated else None
        row["comparabilidade"] = "Taxa-base descritiva, não controle negativo; sem_pcal é nome histórico da retirada de model_outputs, demais campos preservados, prompt original; versão do prompt dos 18 não registrada nos agregados"
        summary = next((s for s in summaries if s["model"] == counts["model"]), None)
        for key in ("control_fn_to_match", "control_fn_to_unsure", "gained_match", "lost_match"):
            row[key] = summary[key] if summary else None
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output or data / "ablacao_dossie_taxa_base.csv", index=False, lineterminator="\n")
    print(result.drop(columns=["fontes_sha256", "comparabilidade"]).to_string(index=False))
    return result


if __name__ == "__main__":
    generate(ROOT / "data")
