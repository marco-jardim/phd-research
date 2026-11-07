"""Fusão exploratória post hoc abaixo do corte: CPU, sem LLM, só agregados."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import sha256  # noqa: E402
from gzcmd.ml_oof import nested_oof_probabilities  # noqa: E402
from scripts.comparador_mesma_carga import numeric  # noqa: E402
from scripts.sensibilidade_o3_macd import (  # noqa: E402
    estimate_minutes,
    prepare,
    queue,
)

TARGETS = (9, 23, 30, 36)
TOP_K = (17, 58, 96, 125, 440, 2522, 3040)
BASES = ("O1", "O3", "O3_MACD")
VARIANTS = BASES + ("F1", "F2", "F3", "F4", "A1", "A2")
METRICS = (
    tuple(f"revisoes_{n}" for n in TARGETS)
    + tuple(f"par2_top_{k}" for k in TOP_K)
    + ("auc_recuperacao_3040",)
)
# Fixado ANTES de calcular/observar qualquer curva fundida. Não é pré-registro.
DEFINITIONS = {
    "F1": {"method": "round_robin", "arms": ["O3", "O1"]},
    "F2": {"method": "round_robin", "arms": ["O3", "O1", "O3_MACD"]},
    "F3": {"method": "RRF", "k": 60, "arms": ["O3", "O1", "O3_MACD"]},
    "F4": {"method": "RRF", "k": 60, "arms": ["O3", "O1"]},
    "A1": {"method": "Thompson", "arms": list(BASES), "window": 200},
    "A2": {"method": "Thompson", "arms": list(BASES), "window": None},
}


def validate_order(order, n):
    if len(order) != n or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("Fila não é permutação completa: duplicação/omissão")


def round_robin(queues):
    n = len(queues[0])
    seen = np.zeros(n, dtype=bool)
    pointers = [0] * len(queues)
    result = []
    while len(result) < n:
        for arm, q in enumerate(queues):
            p = pointers[arm]
            while p < n and seen[q[p]]:
                p += 1
            if p < n:
                item = q[p]
                seen[item] = True
                result.append(item)
                p += 1
            pointers[arm] = p
    order = np.asarray(result, dtype=np.int64)
    validate_order(order, n)
    return order


def rrf(queues):
    n = len(queues[0])
    score = np.zeros(n)
    for q in queues:
        score[q] += 1.0 / (60 + np.arange(1, n + 1))
    # Índices são a ordem lexical dos hashes: desempate estável, sem usar PAR.
    order = np.lexsort((np.arange(n), -score))
    validate_order(order, n)
    return order


def thompson(queues, labels, seed, window):
    """Apenas o braço selecionado recebe o resultado da revisão corrente.

    Os demais braços só avançam sobre pares globalmente já revisados; não
    recebem recompensa nem informação PAR desses pares. Todas as três amostras
    são sorteadas a cada passo; empate de amostras favorece O1, O3, MACD.
    """
    rng = np.random.default_rng(seed)
    n, arms = len(labels), len(queues)
    seen = np.zeros(n, dtype=bool)
    pointers = [0] * arms
    successes = np.zeros(arms, dtype=np.int64)
    trials = np.zeros(arms, dtype=np.int64)
    history = [deque() for _ in range(arms)]
    order = np.empty(n, dtype=np.int64)
    for step in range(n):
        proposals = []
        for arm, q in enumerate(queues):
            p = pointers[arm]
            while p < n and seen[q[p]]:
                p += 1
            pointers[arm] = p
            if p == n:
                raise ValueError("Braço esgotado antes de cobrir o universo")
            proposals.append(q[p])
        samples = rng.beta(1 + successes, 1 + trials - successes)
        arm = int(np.argmax(samples))
        item = proposals[arm]
        order[step] = item
        seen[item] = True
        reward = int(labels[item])  # Único acesso à referência neste passo.
        successes[arm] += reward
        trials[arm] += 1
        if window is not None:
            history[arm].append(reward)
            if len(history[arm]) > window:
                successes[arm] -= history[arm].popleft()
                trials[arm] -= 1
    validate_order(order, n)
    return order


def metrics(counts):
    if np.any(np.diff(counts) < 0) or counts[-1] != 36:
        raise ValueError("Curva inválida")
    result = {f"revisoes_{n}": float(np.searchsorted(counts, n) + 1) for n in TARGETS}
    result.update({f"par2_top_{k}": float(counts[k - 1]) for k in TOP_K})
    # Integral da curva em degraus C(k)/36, constante em [k,k+1), C(0)=0.
    result["auc_recuperacao_3040"] = float(counts[:3039].sum() / (3040 * 36))
    return result


def verify_baselines(curves):
    """Bloqueio obrigatório ANTES de qualquer fusão, contra ambos os CSVs."""
    reference = pd.read_csv(ROOT / "data/sensibilidade_o3_macd.csv")
    full = pd.read_csv(ROOT / "data/sensibilidade_o3_macd_curva.csv")
    comp = pd.read_csv(ROOT / "data/comparador_mesma_carga.csv")
    comp_curve = pd.read_csv(ROOT / "data/comparador_mesma_carga_curva.csv")
    names = {"O1": "O1_SCORE", "O3": "O3_controle", "O3_MACD": "O3_MACD"}
    expected = {
        "O1": [51, 501, 2521],
        "O3": [29, 125, 23443],
        "O3_MACD": [46, 118, 3040],
    }
    for name, c in curves.items():
        if len(c) != 61485 or c[-1] != 36:
            raise ValueError("Universo não reproduziu 61.485/36")
        m = metrics(c)
        if [m[f"revisoes_{t}"] for t in (9, 23, 36)] != expected[name]:
            raise ValueError(f"{name}: revisões de referência não reproduzidas")
        old = reference.loc[reference.variante.eq(names[name])]
        for row in old.itertuples():
            if row.metrica in m and m[row.metrica] != row.valor:
                raise ValueError(f"{name}: métrica difere de sensibilidade")
        old_curve = full.loc[full.variante.eq(names[name])].sort_values("k")
        if not np.array_equal(old_curve.k, np.arange(1, 61486)) or not np.array_equal(
            c, old_curve.par2_acumulados.to_numpy()
        ):
            raise ValueError(f"{name}: curva individual difere de sensibilidade")
        if name == "O3_MACD":
            continue
        comp_name = "O1_SCORE" if name == "O1" else "O3_p_cal_ml"
        for row in comp.loc[comp.ordenacao.eq(comp_name)].itertuples():
            target = 9 if row.modo == "confirmacao" else 23
            if (
                c[row.k - 1] != row.par2_topk
                or m["revisoes_36"] != row.k_min_36
                or m[f"revisoes_{target}"] != row.k_min_equivalente_P
            ):
                raise ValueError(f"{name}: resumo difere de comparador")
        old_curve = comp_curve.loc[comp_curve.ordenacao.eq(comp_name)].sort_values("k")
        if not np.array_equal(old_curve.k, np.arange(1, 2523)) or not np.array_equal(
            c[:2522], old_curve.par2_topk.to_numpy()
        ):
            raise ValueError(f"{name}: curva difere de comparador")
    return {"passed": True, "full_curves": True, "expected_reviews_9_23_36": expected}


def load_queues(data_root):
    import sklearn
    import xgboost

    data = prepare(data_root)  # Inclui align_original e verificações de proveniência.
    fingerprint = data["original_meta"]["fingerprint"]
    if (
        sklearn.__version__ != fingerprint["sklearn"]
        or xgboost.__version__ != fingerprint["xgboost"]
    ):
        raise ValueError("Versões sklearn/XGBoost divergem do original")
    estimate = estimate_minutes(data)
    print(
        "Estimativa de fits (conservadora, 60 fits): " + json.dumps(estimate),
        flush=True,
    )
    pm, folds = nested_oof_probabilities(
        data["x"]["O3_MACD"], data["y"], split_by="refrec", groups=data["groups"]
    )
    if not np.array_equal(folds, data["original"].fold.to_numpy()):
        raise ValueError("Folds MACD divergem do original")
    ids = data["ids"]
    below = data["below"]
    lexical = np.sort(ids[below])
    if len(lexical) != 61485 or len(np.unique(lexical)) != 61485:
        raise ValueError("Universo não é de 61.485 pares únicos")
    par = numeric(data["raw"].PAR).to_numpy()
    if not np.isin(par[below], [0, 2]).all():
        raise ValueError("Rótulos abaixo do corte diferentes de 0/2")
    labels = pd.Series(par[below] == 2, index=ids[below]).loc[lexical].to_numpy()
    scores = (
        numeric(data["raw"].SCORE).to_numpy(),
        data["original"].p_cal_ml.to_numpy(),
        pm,
    )
    queues = {
        name: np.searchsorted(lexical, queue(ids, score, below))
        for name, score in zip(BASES, scores, strict=True)
    }
    for q in queues.values():
        validate_order(q, len(labels))
    return queues, labels, data, estimate


def aggregate(name, runs):
    values = np.array([[metrics(c)[m] for m in METRICS] for c in runs])
    stats = {
        "media": values.mean(axis=0),
        "p10": np.quantile(values, 0.1, axis=0, method="linear"),
        "p90": np.quantile(values, 0.9, axis=0, method="linear"),
    }
    return [
        {"fila": name, "metrica": metric, **{s: float(v[j]) for s, v in stats.items()}}
        for j, metric in enumerate(METRICS)
    ]


def write_report(table, curves, metadata):
    wide = table.pivot(index="fila", columns="metrica", values="media").loc[
        list(VARIANTS)
    ]
    lines = [
        "# Fusão exploratória post hoc de filas abaixo do corte",
        "",
        "CPU, sem LLM. Mesmos cortes e filas do comparador. Revisor perfeito "
        "simulado por PAR=2; 61.485 pares únicos, 36 PAR=2. "
        "Não é validação independente "
        "nem altera a hipótese registrada O3 versus O1, margem de 25%.",
        "",
        "Os 36 PAR=2 estão nos 2.522 candidatos SCORE ≥ −5,648: provável viés de "
        "verificação a favor de O1. Nenhuma política teve acesso "
        "a resultados futuros. "
        "A1/A2 usam apenas as respostas já observadas pelo braço escolhido.",
        "",
        "## Definições fixadas antes das curvas fundidas",
        "",
        "F1: alternar O3/O1; F2: alternar O3/O1/MACD; F3: RRF(60) O3/O1/MACD; "
        "F4: RRF(60) O3/O1. A1: Thompson Beta(1,1), "
        "janela de 200 revisões do braço; "
        "A2: sem janela. Deduplicação global. Não houve seleção de parâmetros. "
        "Fixação a priori para esta execução post hoc, não pré-registro.",
        "",
        f"{metadata['repetitions']} repetições por política, "
        f"seeds 0–{metadata['repetitions'] - 1}. "
        "Métricas adaptativas são média [P10; P90] das métricas por repetição; "
        "não são inversões da curva média. "
        "Quantis lineares, não IC inferenciais.",
        "",
        "## Tabela (todas as métricas)",
        "",
        "| Fila | " + " | ".join(METRICS) + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for name in VARIANTS:
        rows = table.loc[table.fila.eq(name)].set_index("metrica")
        cells = []
        for metric in METRICS:
            r = rows.loc[metric]
            fmt = ".6f" if metric.startswith("auc") else ".2f"
            cell = format(r.media, fmt)
            if name.startswith("A"):
                cell += f" [{format(r.p10, fmt)}; {format(r.p90, fmt)}]"
            cells.append(cell)
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "AUC: integral da recuperação em degraus, C(0)=0 e C(k)/36 em "
        "[k,k+1), até 3.040; normalização por 3.040. Empates individuais e RRF: "
        "hash lexical crescente, exatamente k sem expansão; intercalação segue "
        "a ordem dos braços; empate nas amostras Thompson: O1, O3, MACD.",
        "",
        "## Curva resumida e melhores recuperações por orçamento",
        "",
        "| k | " + " | ".join(VARIANTS) + " | Melhor(es), pela média |",
        "|---:|" + "---:|" * len(VARIANTS) + "---|",
    ]
    for k in TOP_K:
        at = {name: float(curves[name][k - 1]) for name in VARIANTS}
        winners = ", ".join(
            name for name, value in at.items() if value == max(at.values())
        )
        lines.append(
            f"| {k} | "
            + " | ".join(f"{at[n]:.2f}" for n in VARIANTS)
            + f" | {winners} |"
        )
    lines += ["", "## Leitura curta", ""]
    for target in TARGETS:
        column = wide[f"revisoes_{target}"]
        winners = ", ".join(column.index[column.eq(column.min())])
        lines.append(
            f"- Recuperar {target} PAR=2: menor esforço médio em {winners} "
            f"({column.min():.2f} revisões)."
        )
    dominating = [
        n
        for n in VARIANTS
        if n not in BASES
        and all(
            wide.loc[n, f"revisoes_{t}"]
            <= min(wide.loc[b, f"revisoes_{t}"] for b in ("O1", "O3"))
            for t in (9, 23, 36)
        )
    ]
    lines.append(
        "- Dominação simultânea de O1 e O3 em 9/23/36 (não pior em todos): "
        + (", ".join(dominating) if dominating else "nenhuma fusão/política")
        + "."
    )
    lines += [
        "- Menor esforço e maior recuperação pontual não implicam "
        "dominância em toda a curva. Comparações adaptativas acima "
        "usam médias; a dispersão consta da tabela.",
        "",
        "## Reprodução e proveniência",
        "",
        "As curvas individuais reproduziram "
        "sensibilidade_o3_macd_curva.csv; O1/O3 também reproduziram "
        "comparador_mesma_carga_curva.csv e os resumos, antes de qualquer "
        "fusão. O3 usa p_cal_ml OOF original validado por align_original; "
        "MACD é recalculada com prepare/nested_oof_probabilities importados, "
        "mesmos folds, versões e atributos. Nenhum código existente "
        "foi copiado ou alterado.",
        "",
        "Estimativas, tempos, versões, SHA-256 e definições constam de "
        "data/fusao_filas_metadata.json. CSVs contêm somente agregados, "
        "sem identificadores ou valores individuais.",
    ]
    path = ROOT / "docs/plans/RELATORIO_FUSAO_2026-11-07.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(data_root):
    import sklearn
    import xgboost

    start = time.perf_counter()
    # Registra a especificação antes de carregar rótulos ou calcular curvas.
    fixed_at = datetime.now(UTC).isoformat()
    queues, labels, data, fit_estimate = load_queues(data_root)
    curves = {name: labels[q].astype(int).cumsum() for name, q in queues.items()}
    reproduction = verify_baselines(curves)  # STOP se divergir; nenhuma fusão antes.
    print("Referências individuais reproduzidas integralmente.", flush=True)
    ordered_arms = [queues[name] for name in BASES]
    pilot = {}
    for name in ("A1", "A2"):
        t = time.perf_counter()
        order = thompson(ordered_arms, labels, 0, DEFINITIONS[name]["window"])
        pilot[name] = {
            "seconds": time.perf_counter() - t,
            "curve": labels[order].astype(int).cumsum(),
        }
    estimate_200 = sum(p["seconds"] for p in pilot.values()) * 200 * 1.5 / 60
    repetitions = 50 if estimate_200 > 30 else 200
    print(
        f"Estimativa A1+A2 (200 cada, fator 1.5): {estimate_200:.2f} min; "
        f"selecionadas {repetitions} repetições.",
        flush=True,
    )
    summary = []
    for name in BASES:
        summary.extend(aggregate(name, [curves[name]]))
    for name in ("F1", "F2", "F3", "F4"):
        arms = [queues[a] for a in DEFINITIONS[name]["arms"]]
        order = round_robin(arms) if name in ("F1", "F2") else rrf(arms)
        curves[name] = labels[order].astype(int).cumsum()
        summary.extend(aggregate(name, [curves[name]]))
    for name in ("A1", "A2"):
        runs = [pilot[name]["curve"]]
        for seed in range(1, repetitions):
            order = thompson(ordered_arms, labels, seed, DEFINITIONS[name]["window"])
            runs.append(labels[order].astype(int).cumsum())
            if seed % 25 == 0:
                print(f"{name}: {seed + 1}/{repetitions}", flush=True)
        summary.extend(aggregate(name, runs))
        curves[name] = np.mean(runs, axis=0)
    table = pd.DataFrame(summary)
    curve_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "fila": name,
                    "k": np.arange(1, 61486),
                    "par2_acumulados": curves[name],
                }
            )
            for name in VARIANTS
        ],
        ignore_index=True,
    )
    inputs = (
        "data/COMPARADORSEMIDENT.csv",
        "data/pcal_ml_oof.csv",
        "data/pcal_ml_oof.json",
    )
    refs = (
        "data/sensibilidade_o3_macd.csv",
        "data/sensibilidade_o3_macd_curva.csv",
        "data/sensibilidade_o3_macd_metadata.json",
        "data/comparador_mesma_carga.csv",
        "data/comparador_mesma_carga_curva.csv",
        "data/ablation_comprec_allowlist_metadata.json",
    )
    codes = (
        "scripts/fusao_filas.py",
        "scripts/sensibilidade_o3_macd.py",
        "scripts/comparador_mesma_carga.py",
        "gzcmd/ml_oof.py",
        "gzcmd/loader.py",
        "gzcmd/arranjo_corte.py",
        "tests/test_fusao_filas.py",
    )
    metadata = {
        "analysis": "exploratory_post_hoc",
        "registered": False,
        "cpu_only": True,
        "llm_calls": 0,
        "date_utc": datetime.now(UTC).isoformat(),
        "fixed_before_fused_curves_utc": fixed_at,
        "fixed_a_priori_for_this_run": True,
        "definitions": DEFINITIONS,
        "prior": {"alpha": 1, "beta": 1},
        "repetitions": repetitions,
        "seeds": list(range(repetitions)),
        "rng": "numpy.default_rng PCG64",
        "quantiles": "linear P10/P90 of run metrics",
        "below_cutoff": 61485,
        "par2": 36,
        "targets": TARGETS,
        "top_k": TOP_K,
        "deduplication": "global; complete permutation validated for every run",
        "ordering": "score descending, lexical pair hash ascending; "
        "RRF same tie-break; round-robin arm order as definitions; "
        "Thompson sample ties O1/O3/MACD",
        "reward": "PAR=2 only after selected review; selected arm only, "
        "no shared rewards",
        "auc": "sum C(k), k=0..3039, divided by 3040*36 (left-step integral)",
        "original_O3": "validated original p_cal_ml; no refit, "
        "no substitute calibration",
        "baseline_reproduction": reproduction,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "source_sha256": sha256(data["source"]),
        "input_sha256": {
            **{n: sha256(Path(data_root) / n) for n in inputs},
            **{n: sha256(ROOT / n) for n in refs},
        },
        "code_sha256": {n: sha256(ROOT / n) for n in codes},
        "estimate": {
            "fits": fit_estimate,
            "adaptive_pilot_seconds": {n: p["seconds"] for n, p in pilot.items()},
            "adaptive_200_minutes": estimate_200,
            "safety_factor": 1.5,
            "reduction_rule": "50 if combined 200-run estimate exceeds 30 min",
        },
        "elapsed_seconds": time.perf_counter() - start,
        "command": " ".join([sys.executable, *sys.argv]),
        "limitation": "Verification bias favors O1; post hoc; "
        "registered O3 vs O1 25% margin unchanged.",
    }
    table.to_csv(ROOT / "data/fusao_filas.csv", index=False, lineterminator="\n")
    curve_table.to_csv(
        ROOT / "data/fusao_filas_curva.csv", index=False, lineterminator="\n"
    )
    metadata["output_sha256"] = {
        n: sha256(ROOT / "data" / n)
        for n in ("fusao_filas.csv", "fusao_filas_curva.csv")
    }
    (ROOT / "data/fusao_filas_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(table, curves, metadata)
    print(table.to_string(index=False), flush=True)
    print(f"Concluído em {time.perf_counter() - start:.1f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    args = parser.parse_args()
    generate(args.data_root)
