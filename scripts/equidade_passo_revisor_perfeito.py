"""Upper-bound sensitivity by blocking step and comparisons excluded by den30.

Human referrals are not observed correct decisions. Recovering all true links
referred to humans is a hypothetical upper bound, not measured performance.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sensitivity(groups: pd.DataFrame) -> pd.DataFrame:
    result = groups.loc[groups.dimension.eq("passo")].copy()
    if not (result.tp + result.fn).eq(result.n_true).all():
        raise ValueError("True-link accounting does not close")
    if not result.n_humano_true.between(0, result.fn).all():
        raise ValueError("Human true links must be a subset of nonaccepted true links")
    result["sens_automatica"] = result.tp / result.n_true
    result["sens_revisor_perfeito"] = (result.tp + result.n_humano_true) / result.n_true
    result["fn_revisor_perfeito"] = result.fn - result.n_humano_true
    return result[["decisao", "group", "n_true", "tp", "fn", "n_humano_true",
                   "fn_revisor_perfeito", "sens_automatica", "sens_revisor_perfeito"]].rename(columns={"group": "passo"})


def excluded(tests: pd.DataFrame) -> pd.DataFrame:
    result = tests.loc[tests.n1.lt(30) | tests.n2.lt(30)].copy()
    result["motivo_exclusao"] = "n1<30 ou n2<30; antes de BH"
    return result[["decisao", "tipo", "dimension", "metric", "group_a", "group_b",
                   "n1", "n2", "num_a", "num_b", "rate_a", "rate_b", "motivo_exclusao"]]


def generate(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = sensitivity(pd.read_csv(data / "equidade_arranjo_group_metrics.csv"))
    outside = excluded(pd.read_csv(data / "equidade_arranjo_pairwise_tests.csv"))
    steps.to_csv(data / "equidade_passo_revisor.csv", index=False, lineterminator="\n")
    outside.to_csv(data / "equidade_arranjo_den30_excluidos.csv", index=False, lineterminator="\n")
    print(steps.to_string(index=False))
    print(outside.loc[outside.dimension.eq("passo") & outside.metric.eq("ppv")].to_string(index=False))
    return steps, outside


if __name__ == "__main__":
    generate(ROOT / "data")
