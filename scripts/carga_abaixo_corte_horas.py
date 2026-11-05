"""Converte o limite inferior de pares em horas sob a hipótese histórica."""
from decimal import Decimal
from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "gzcmd/gzcmd_v3_config.yaml"


def generate(root=ROOT):
    config = (root / SOURCE).read_text(encoding="utf-8")
    minutes = {Decimal(value) for value in re.findall(r"(\d+) min", config)}
    if len(minutes) != 1:
        raise ValueError("Hipótese temporal ausente ou ambígua na configuração histórica")
    minutes_per_pair = minutes.pop()
    with (root / "data/carga_abaixo_corte.csv").open(encoding="utf-8") as stream:
        totals = [row for row in csv.DictReader(stream) if row["passo"] == "TOTAL"]
    target = root / "data/carga_abaixo_corte_horas.csv"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["limiar", "pares", "minutos_por_par", "horas",
                                                    "horas_inteiras", "minutos_restantes", "fonte_hipotese"],
                                lineterminator="\n")
        writer.writeheader()
        for row in totals:
            pairs = int(row["pares_acima_limiar"])
            total_minutes = pairs * minutes_per_pair
            writer.writerow(dict(limiar=row["limiar"], pares=pairs, minutos_por_par=minutes_per_pair,
                                 horas=f"{total_minutes / 60:.12f}", horas_inteiras=int(total_minutes // 60),
                                 minutos_restantes=total_minutes % 60, fonte_hipotese=SOURCE))
    return target


if __name__ == "__main__":
    print(generate())
