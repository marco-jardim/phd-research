"""Publica somente agregados do probe D, sem inferência ou dados individuais."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "model", "modo", "estrato", "passada", "n_comparados", "n_originais",
    "MATCH", "NONMATCH", "UNSURE", "VP", "FP", "VPP", "truncadas",
    "n_sem_resultado", "PAR2_recuperados",
)
ADDED = [
    f"{group} qtd frag {suffix}"
    for group in ("NOME", "NOMEMAE")
    for suffix in ("raros", "comuns", "muito parec", "abrev")
] + [
    "DTNASC dt ap 1digi", "DTNASC dt inv dia", "DTNASC dt inv mes",
    "DTNASC dt inv ano", "CODMUNRES uf igual", "CODMUNRES uf prox",
    "CODMUNRES local prox",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utility(rows):
    def one(stratum, arm):
        found = [r for r in rows if r["modo"] == "vigilancia"
                 and r["estrato"] == stratum and r["passada"] == arm]
        if len(found) != 1:
            raise ValueError(f"Estrato/braço ausente ou duplicado: {stratum}/{arm}")
        return found[0]

    s1, paired = one("S1", "v5"), one("S1", "v4_pareado")
    s2, old = one("S2", "v5"), one("S2", "v4_pareado")
    removed = int(paired["FP"]) - int(s1["FP"])
    recovered = int(s2["PAR2_recuperados"]) - int(old["PAR2_recuperados"])
    vpp = float(s1["VPP"]) if s1["VPP"] else None
    return {
        "criterion": "FP S1 retirados >=5 OU PAR=2 S2 adicionais >=3 OU VPP S1 >=0.35",
        "FP_S1_retirados_validos": removed,
        "PAR2_S2_adicionais": recovered,
        "VPP_S1": vpp,
        "useful": removed >= 5 or recovered >= 3 or (vpp is not None and vpp >= .35),
        "missing_rule": "ausentes não contam como correções; truncadas não são UNSURE",
    }


def generate(source, output):
    with (source / "resultados_agregados.csv").open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for item in raw:
        if item["ponderado"].lower() != "false":
            continue
        if item["passada"] not in ("v5", "v4_pareado", "v4_todos"):
            raise ValueError("Braço não previsto no probe D")
        row = {k: item[k] for k in FIELDS if k not in ("n_comparados", "n_originais")}
        row.update(n_comparados=item["n"], n_originais=item["n_expected"])
        n = int(row["n_comparados"])
        if n > int(row["n_originais"]) or n != sum(int(row[k]) for k in ("MATCH", "NONMATCH", "UNSURE")):
            raise ValueError("Denominadores inconsistentes")
        if int(row["MATCH"]) != int(row["VP"]) + int(row["FP"]):
            raise ValueError("VP/FP inconsistentes")
        rows.append(row)
    transport = json.loads((source / "transporte_agregado.json").read_text(encoding="utf-8"))
    gate = utility(rows)
    if gate["useful"] != transport["gate"]["useful"]:
        raise ValueError("Critério de utilidade diverge do transporte")
    totals = [r for r in rows if r["estrato"] == "TOTAL_AMOSTRA" and r["passada"] == "v5"]
    late = sum(s["calls"] for s in transport["stages"] if s["modo"] == "vigilancia"
               and s["stage"] == "AB" and s["transport"] == "batch")
    metadata = {
        "analysis": "exploratória post hoc; probe D",
        "model": "Claude Haiku 5.5",
        "source_sha256": {name: sha256(source / name) for name in (
            "resultados_agregados.csv", "transporte_agregado.json", "PREREG.md", "prompt_v5.md")},
        "prompt_v5_sha256": sha256(source / "prompt_v5.md"),
        "campos_acrescentados": ADDED,
        "contexto_acrescentado": ["contagem REFREC", "contagem COMPREC", "posição pela nota",
                                  "diferença para a melhor nota", "diferença para a segunda nota"],
        "MACD": "nove campos já presentes no v4, mantidos",
        "cost_usd": transport["spent_usd"],
        "cost_basis": "usage retornado, não fatura; inclui piloto e truncadas",
        "transport_by_stage": transport["stages"],
        "cancellation": {
            "mode": "vigilancia", "stage": "AB", "canceled_without_generation": 1,
            "late_success_responses_used": late,
            "rule": "cancelamento por prazo do crédito; sucessos tardios usados e não repetidos; somente cancelada sem geração teve fallback direto",
            "source": "PREREG.md, adendo de transporte; reconciliação do relatório D",
        },
        "utility": {**gate, "resultado": "atingido" if gate["useful"] else "não atingido"},
        "opus_executado": False,
        "truncadas_passada_principal": sum(int(r["truncadas"]) for r in totals),
        "pares_sem_decisao": sum(int(r["n_sem_resultado"]) for r in totals),
        "truncation_note": "respostas não equivalem a pares; piloto separado",
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "probe_dossie_v5_agregado.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "probe_dossie_v5_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("D:/git/phd-research/.local/probe_dossie_v5"))
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    meta = generate(args.source, args.output)
    print(f"Probe D: {meta['utility']['resultado']}; custo USD {meta['cost_usd']:.8f}")


if __name__ == "__main__":
    main()
