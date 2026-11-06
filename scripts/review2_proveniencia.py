"""Record changed current sources beside frozen historical fingerprints.

The four historical manifests are not mutable current-source manifests: their
fingerprints record the executed experiment. The existing provenance-only
refresh for sensibilidade_sem_obito_refrec does not include llm_dossier.py.
This supplemental audit never rewrites pre-registrations or historical metadata.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGED = ("gzcmd/llm_dossier.py", "scripts/run_exp2_ablacao_dossie.py",
           "gzcmd/M_arranjo_corte.md", "gzcmd/gzcmd_v3_config.yaml")


def references(value, target, trail=""):
    if isinstance(value, dict):
        if value.get("caminho") == target and "sha256" in value:
            yield trail, value["sha256"]
        for key, item in value.items():
            path = f"{trail}/{key}"
            if key == target and isinstance(item, str) and len(item) == 64:
                yield path, item
            yield from references(item, target, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from references(item, target, f"{trail}/{index}")


def generate(data: Path) -> dict:
    records = []
    for path in sorted(data.glob("*meta*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        for target in CHANGED:
            current = hashlib.sha256((ROOT / target).read_bytes()).hexdigest()
            for field, historical in references(value, target):
                records.append({"manifesto": path.name, "campo": field, "fonte": target,
                                "sha256_historico": historical, "sha256_atual": current,
                                "identico": historical == current})
    output = {"fontes": records, "manifestos_historicos_regravados": [],
              "preregistros_alterados": False,
              "procedimento_existente": "sensibilidade_sem_obito_refrec.py --refresh-provenance-only não inclui estas quatro fontes; não se aplica",
              "interpretacao": "Fingerprints históricos preservados. As correções atuais não são os bytes registrados da execução histórica; esta auditoria não equivale a reproduzir S1."}
    (data / "review2_proveniencia.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8", newline="\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    generate(ROOT / "data")
