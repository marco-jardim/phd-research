"""Audita somente metadados locais de uso, sem imprimir respostas ou IDs.

Não chama APIs nem modelos. Ausência de thinking_tokens não é contagem zero
de raciocínio. Não há amostragem ou componente aleatório nesta agregação.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORTS = (
    "anthropic_sem_mo_v4",
    "anthropic_sem_mo_v4_lote31",
    "anthropic_sem_mo_v4_lote31_refrec",
)
MODES = ("vigilancia", "confirmacao")


def thinking_values(value):
    """Extrai somente os contadores numéricos, nunca texto das respostas."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "thinking_tokens":
                if isinstance(item, bool) or not isinstance(item, (int, float)) or item < 0:
                    raise ValueError("Contador thinking_tokens inválido")
                yield item
            else:
                yield from thinking_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from thinking_values(item)


def token_summary(values):
    if not values:
        return {
            "status": "não registrado",
            "n_contadores_registrados": 0,
            "n_chamadas_thinking_tokens_maior_zero": None,
            "soma_thinking_tokens": None,
            "media_thinking_tokens": None,
        }
    return {
        "status": "contadores registrados; verificar unidade antes de interpretar como chamadas",
        "n_contadores_registrados": len(values),
        "n_contadores_maior_zero": sum(value > 0 for value in values),
        "soma_thinking_tokens": sum(values),
        "media_thinking_tokens": sum(values) / len(values),
    }


def audit(root=ROOT):
    data_root = root / "data/sprint3b"
    rows = []
    all_values = []
    scanned = set()
    for cohort in COHORTS:
        for mode in MODES:
            path = data_root / cohort / f"exp2_llm_sync_{mode}.json"
            checkpoint = json.loads(path.read_text(encoding="utf-8-sig"))
            values = list(thinking_values(checkpoint))
            all_values.extend(values)
            scanned.add(path)
            rows.append({
                "coorte": cohort,
                "modo": mode,
                "fonte": path.relative_to(root).as_posix(),
                "chamadas_transport": checkpoint["transport"]["calls"],
                **token_summary(values),
            })
    # Inclui sidecars e journals, também quando ignorados pelo git.
    journal_files = []
    for path in data_root.rglob("*.transport.json"):
        parts = path.relative_to(data_root).parts
        if any(cohort in parts or any(part.startswith(cohort + "_journal")
                                     or any(part.startswith(cohort + "_" + mode + "_journal")
                                            for mode in MODES)
                                     for part in parts)
               for cohort in COHORTS):
            journal_files.append(path)
    for cohort in COHORTS:
        scanned.update((data_root / cohort).rglob("*.json"))
    scanned.update(journal_files)
    additional_values = []
    checkpoint_paths = {data_root / cohort / f"exp2_llm_sync_{mode}.json"
                        for cohort in COHORTS for mode in MODES}
    for path in sorted(scanned - checkpoint_paths):
        additional_values.extend(thinking_values(json.loads(path.read_text(encoding="utf-8-sig"))))
    all_values.extend(additional_values)
    return {
        "escopo": list(COHORTS),
        "fontes": rows,
        "arquivos_json_examinados": len(scanned),
        "journals_transport_examinados": len(journal_files),
        "chamadas_transport_total": sum(row["chamadas_transport"] for row in rows),
        "contadores_adicionais": len(additional_values),
        "agregado": token_summary(all_values),
        "nota": "Sem contador registrado, n de chamadas com raciocínio, soma e média são desconhecidos, não zero.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = json.dumps(audit(), ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")


if __name__ == "__main__":
    main()
