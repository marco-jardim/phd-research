"""Robustez post hoc das filas com contexto: OOF fixo, bootstrap REFREC, CPU."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.ml_oof import N_SPLITS, SEED as FIT_SEED, _classifier, nested_oof_probabilities
from scripts import contexto_obito as contexto
from scripts import fusao_robustez as fusao
from scripts.fusao_filas import sha256, validate_order
from scripts.sensibilidade_o3_macd import prepare, queue

SEED = fusao.SEED
TARGETS = fusao.TARGETS
FRACTIONS = fusao.FRACTIONS
COMPARISONS = (("CM", "O1"), ("CM", "O3"), ("CM", "F4"), ("C", "O1"), ("C", "O3"))
VARIANTS = {"C": "O3_contexto", "CM": "O3_contexto_MACD", "O3_MACD": "O3_MACD"}
EXPECTED = {"C": [20, 75, 9801], "CM": [14, 74, 1298], "O3_MACD": [46, 118, 3040]}
NAMES = (*fusao.NAMES, "C", "CM")
PREFIX = "contexto_robustez"
CPU_LIMIT_SECONDS = 60 * 60
SCIENTIFIC_CODE = (
    "scripts/contexto_obito.py", "scripts/sensibilidade_o3_macd.py",
    "scripts/fusao_filas.py", "scripts/fusao_robustez.py", "gzcmd/ml_oof.py",
    "gzcmd/loader.py", "gzcmd/arranjo_corte.py", "scripts/comparador_mesma_carga.py",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def replace_retry(source, destination):
    """Checkpoint local com backoff para bloqueios temporários no Windows."""
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * 2 ** attempt)


def validate_cache(ids, probabilities, folds, expected_ids, expected_folds):
    if not np.array_equal(ids, expected_ids) or not np.array_equal(folds, expected_folds):
        raise ValueError("Cache OOF desalinhado ou folds divergentes")
    if probabilities.shape != expected_ids.shape or not np.isfinite(probabilities).all():
        raise ValueError("Probabilidades OOF incompletas")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilidades OOF fora de [0, 1]")


def probabilities_once(data, cache_dir, cpu_start):
    """Uma regeneração por variante; retomadas reutilizam checkpoints validados."""
    import sklearn
    import xgboost

    cache_dir = Path(cache_dir).resolve()
    if not cache_dir.is_relative_to((ROOT / ".local").resolve()):
        raise ValueError("Cache por par deve ficar exclusivamente em .local/ deste worktree")
    cache_dir.mkdir(parents=True, exist_ok=True)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__,
                "xgboost": xgboost.__version__}
    expected_versions = json.loads((ROOT / "data/contexto_obito_metadata.json").read_text(encoding="utf-8"))["versions"]
    if versions != expected_versions:
        raise ValueError("Versões divergem da execução original de contexto")
    fingerprint = {
        "versions": versions, "source_sha256": sha256(data["source"]),
        "seed": FIT_SEED, "inner_seeds": list(range(FIT_SEED + 1, FIT_SEED + N_SPLITS + 1)),
        "folds": N_SPLITS, "split_by": "refrec", "calibration": "nested isotonic 5x5",
        "code_sha256": {name: sha256(ROOT / name) for name in SCIENTIFIC_CODE},
    }
    result, provenance = {}, {}
    for name, variant in VARIANTS.items():
        path = cache_dir / f"{variant}.npz"
        meta_path = path.with_suffix(".json")
        variant_fingerprint = {**fingerprint, "variant": variant, "features": list(data["x"][variant])}
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["fingerprint"] != variant_fingerprint or meta["npz_sha256"] != sha256(path):
                raise ValueError("Checkpoint divergiu; não regenerar silenciosamente")
            with np.load(path, allow_pickle=False) as saved:
                p, folds = saved["probabilities"], saved["folds"]
                validate_cache(saved["ids"], p, folds, data["ids"], data["original"].fold.to_numpy())
            mode = "reused_private_checkpoint"
        else:
            if path.exists() or meta_path.exists():
                raise ValueError("Checkpoint incompleto; preservar e examinar antes de regenerar")

            def check_budget(*_):
                if time.process_time() - cpu_start >= CPU_LIMIT_SECONDS:
                    raise RuntimeError("Teto de 60 min de CPU atingido durante regeneração OOF")

            def budgeted_classifier(y, seed):
                check_budget()
                return _classifier(y, seed)

            wall, cpu = time.perf_counter(), time.process_time()
            p, folds = nested_oof_probabilities(
                data["x"][variant], data["y"], split_by="refrec", groups=data["groups"],
                model_factory=budgeted_classifier, audit=check_budget)
            validate_cache(data["ids"], p, folds, data["ids"], data["original"].fold.to_numpy())
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, ids=data["ids"].astype(str), probabilities=p, folds=folds)
            replace_retry(temporary, path)
            meta = {"fingerprint": variant_fingerprint, "npz_sha256": sha256(path),
                    "date_utc": datetime.now(UTC).isoformat(),
                    "wall_seconds": time.perf_counter() - wall,
                    "cpu_seconds": time.process_time() - cpu, "model_fits": 30}
            temporary_meta = meta_path.with_suffix(".tmp.json")
            write_json(temporary_meta, meta)
            replace_retry(temporary_meta, meta_path)
            mode = "regenerated_once"
        result[name] = p
        provenance[name] = {"mode": mode, "private_path": path.relative_to(ROOT).as_posix(),
                            "sha256": meta["npz_sha256"], "wall_seconds": meta["wall_seconds"],
                            "cpu_seconds": meta["cpu_seconds"], "model_fits": meta["model_fits"]}
        print(f"OOF {name}: {mode}; CPU original do checkpoint {meta['cpu_seconds']:.2f} s", flush=True)
    return result, provenance, versions


def orders(queues):
    return {**fusao.fused_orders(queues), "C": queues["C"], "CM": queues["CM"]}


def verify_context(data, probabilities):
    """Resumos e curvas completos, antes de qualquer reamostragem."""
    scores = {VARIANTS[name]: p for name, p in probabilities.items()}
    actual, curves = contexto.summarize(data, scores)
    reference = pd.read_csv(ROOT / "data/contexto_obito.csv")
    reference_curves = pd.read_csv(ROOT / "data/contexto_obito_curva.csv")
    reproduction = {}
    for name, variant in VARIANTS.items():
        a = actual.loc[actual.variante.eq(variant)].set_index("metrica").valor
        b = reference.loc[reference.variante.eq(variant)].set_index("metrica").valor
        if set(a.index) != set(b.index):
            raise ValueError(f"Métricas divergentes em {name}")
        for metric in a.index:
            # Contagens exatas; AUC/AP toleram só a serialização de float do CSV.
            tolerance = 2e-16 if metric.startswith(("auc", "ap")) else 0
            if not np.isclose(a[metric], b[metric], rtol=0, atol=tolerance):
                raise ValueError(f"Resumo não reproduzido: {name}/{metric}")
        got = [int(a[f"revisoes_{target}"]) for target in TARGETS]
        if got != EXPECTED[name]:
            raise ValueError(f"Revisões não reproduzidas: {name}")
        c = curves.loc[curves.variante.eq(variant)].sort_values("k")
        old = reference_curves.loc[reference_curves.variante.eq(variant)].sort_values("k")
        if not np.array_equal(c.k, old.k) or not np.array_equal(c.par2_acumulados, old.par2_acumulados):
            raise ValueError(f"Curva completa não reproduzida: {name}")
        reproduction[name] = {"reviews": got, "all_summary_metrics": True, "full_curve": True}
    return reproduction


def bootstrap_once(queues, labels, group_codes, group_count, rng):
    """Mesma expansão, recálculo RRF e regra híbrida da F4, mais C e CM fixas."""
    draws = rng.integers(0, group_count, size=group_count)
    counts = np.bincount(draws, minlength=group_count)
    sampled, original = fusao.resampled_queues(queues, counts[group_codes])
    sampled_labels = labels[original]
    total = int(sampled_labels.sum())
    if not total:
        return None
    targets = TARGETS if total >= 36 else tuple(int(np.ceil(f * total)) for f in FRACTIONS)
    return fusao.reviews(orders(sampled), sampled_labels, targets), total, len(original)


def summarize_bootstrap(runs):
    rows = []
    for variant, baseline in COMPARISONS:
        difference = np.array([r[0][variant] - r[0][baseline] for r in runs])
        for j, target in enumerate(TARGETS):
            lo, median, hi = np.quantile(difference[:, j], [0.025, 0.5, 0.975], method="linear")
            rows.append({"comparacao": f"{variant}-{baseline}", "meta_nominal": target,
                         "fracao_se_menos_36": FRACTIONS[j], "replicas_validas": len(runs),
                         "diferenca_revisoes_mediana": float(median),
                         "diferenca_revisoes_ic95_inf": float(lo),
                         "diferenca_revisoes_ic95_sup": float(hi),
                         "proporcao_diferenca_menor_zero": float((difference[:, j] < 0).mean())})
    return pd.DataFrame(rows)


def faixa_table(queues, labels, scores):
    baseline = fusao.faixa_table(queues, labels, scores)
    reference = pd.read_csv(ROOT / "data/fusao_robustez_faixa.csv")
    for column in baseline.columns:
        if column == "fila":
            equal = baseline[column].equals(reference[column])
        else:
            equal = np.allclose(baseline[column], reference[column], rtol=0, atol=2e-12)
        if not equal:
            raise ValueError(f"Faixa F4 original divergiu: {column}")
    mask = fusao.reviewable_mask(scores)
    restricted, original = fusao.resampled_queues(queues, mask.astype(int))
    effort = fusao.reviews(orders(restricted), labels[original], TARGETS)
    rows = []
    for name in NAMES:
        for j, target in enumerate(TARGETS):
            row = {"fila": name, "meta": target, "revisoes": int(effort[name][j])}
            for base in ("O1", "O3", "F4"):
                row[f"diferenca_{base}"] = int(effort[name][j] - effort[base][j])
                row[f"reducao_percentual_{base}"] = float(100 * (1 - effort[name][j] / effort[base][j]))
            rows.append(row)
    return pd.DataFrame(rows)


def report(table, faixa, metadata):
    def fmt(value, precision=2):
        return f"{value:,.{precision}f}".replace(",", "_").replace(".", ",").replace("_", ".")

    def metas(values):
        return ", ".join(map(str, values)) or "nenhuma"

    lines = ["# B: robustez das filas com contexto por óbito", "",
             "Análise exploratória post hoc, CPU, sem LLM. C = O3+contexto; CM = O3+contexto+MACD. "
             "Revisor perfeito simulado por PAR=2. A hipótese registrada O3 versus O1 permanece inalterada.", "",
             "## Procedimento e reprodução", "",
             "Autorizada uma regeneração das probabilidades OOF de C, CM e O3_MACD. "
             "Importadas as funções originais, com os mesmos atributos, seeds, folds, hiperparâmetros "
             "e calibração isotônica aninhada 5×5 por REFREC. Checkpoints por par somente em .local/. "
             "C reproduziu 20/75/9.801 revisões; CM, 14/74/1.298; O3_MACD, 46/118/3.040, "
             "além dos demais resumos e das curvas completas. O1/O3/F3/F4 também reproduziram suas referências.", "",
             f"Bootstrap agrupado por REFREC: {metadata['replicas']} réplicas, seed {SEED}, "
             f"{metadata['grupos_REFREC']} grupos abaixo do corte. Probabilidades OOF e rótulos fixos, "
             "sem retreino nas réplicas. Amostragem de grupos com reposição, todos os candidatos de cada "
             "óbito recebem sua multiplicidade; cópias contam como revisões. Posições e RRF são "
             "recalculadas em cada réplica pelas funções importadas da F4; C/CM conservam sua ordem relativa.", "",
             "Correção da descrição inicial do despacho: usa-se exatamente a regra híbrida histórica da F4, "
             "não frações em todas as réplicas. Com pelo menos 36 PAR=2, metas fixas 9/23/36; abaixo disso, "
             "teto de 25%/64%/100% dos positivos da réplica. A regra muda o estimando entre réplicas e "
             "não equivale a sensibilidade fixa. Réplicas sem positivos são excluídas e contadas nos metadados.", "",
             "IC95% percentil, quantis lineares. Diferença = variante menos base, negativa favorece a variante. "
             "A proporção de diferenças negativas é descritiva, não um valor-p. Critério descritivo de "
             "vantagem consistente: limite superior do IC95% estritamente negativo, sem ajuste de multiplicidade.", "",
             "## Bootstrap", "", "| Comparação | Meta nominal | Mediana Δ [IC95%] | P(Δ < 0) |",
             "|---|---:|---:|---:|"]
    for r in table.itertuples():
        lines.append(f"| {r.comparacao} | {r.meta_nominal} | {fmt(r.diferenca_revisoes_mediana)} "
                     f"[{fmt(r.diferenca_revisoes_ic95_inf)}; {fmt(r.diferenca_revisoes_ic95_sup)}] "
                     f"| {fmt(r.proporcao_diferenca_menor_zero, 4)} |")
    lines += ["", "## Faixa com todos os 36 PAR=2", "",
              "Mesmo critério de data/fusao_robustez_faixa.csv: round(SCORE, 3) ≥ −5,648, "
              "2.522 candidatos e 36 PAR=2. O mínimo observado é −5,64846544302 (aproximadamente −5,6485). "
              "O filtro literal ≥ −5,648 excluiria 27 candidatos e cinco positivos. "
              "Recalculam-se as filas e a RRF no subconjunto, sem apenas filtrar a fusão já pronta.", "",
              "| Fila | Meta | Revisões | Δ O1 | Δ O3 | Δ F4 |", "|---|---:|---:|---:|---:|---:|"]
    for r in faixa.itertuples():
        lines.append(f"| {r.fila} | {r.meta} | {r.revisoes} | {r.diferenca_O1} | {r.diferenca_O3} | {r.diferenca_F4} |")
    lines += ["", "## Leitura em cinco linhas", ""]
    for i, base in enumerate(("O1", "O3", "F4"), 1):
        selected = table.loc[table.comparacao.eq(f"CM-{base}") & table.diferenca_revisoes_ic95_sup.lt(0)]
        lines.append(f"{i}. CM apresenta IC95% inteiramente favorável contra {base} nas metas nominais: "
                     f"{metas(selected.meta_nominal)}; nas demais, não há esse suporte.")
    c_robust = table.loc[table.comparacao.str.startswith("C-")].groupby("meta_nominal").diferenca_revisoes_ic95_sup.max()
    cm_band = faixa.loc[faixa.fila.eq("CM")]
    survived = cm_band.loc[cm_band[["diferenca_O1", "diferenca_O3", "diferenca_F4"]].lt(0).all(axis=1), "meta"]
    band_detail = "; ".join(f"{base}: {metas(cm_band.loc[cm_band[f'diferenca_{base}'].lt(0), 'meta'])}"
                            for base in ("O1", "O3", "F4"))
    lines += [f"4. Na faixa, CM mantém vantagem pontual sobre as três bases simultaneamente nas metas: "
              f"{metas(survived)} ({band_detail}); isso descreve ordenação interna, não um novo IC.",
              f"5. C tem IC95% inteiramente favorável contra O1 e O3 simultaneamente nas metas: "
              f"{metas(c_robust.index[c_robust.lt(0)])}; seleção post hoc, viés de verificação e incerteza "
              "de treino/revisão não são resolvidos pelo bootstrap condicional.", "",
              "## Limites e proveniência", "",
              "Os 36 positivos foram identificados pelo serviço, provavelmente com priorização por SCORE. "
              "O contexto usa o universo completo sem rótulos; COMPREC pode atravessar folds por REFREC. "
              "A análise não estima generalização para novas bases, não elimina viés de verificação, "
              "não cobre a incerteza do treinamento e não constitui teste confirmatório.", "",
              f"CPU desta execução: {fmt(metadata['cpu_seconds'] / 60)} min; "
              f"parede: {fmt(metadata['elapsed_seconds'] / 60)} min. Teto de 60 min de CPU, "
              f"redução para 500 réplicas: {'sim' if metadata['reduced_to_500'] else 'não aplicada'}. "
              "Estimativa anterior ao bootstrap, tempos e checkpoints constam dos metadados.", "",
              "Artefatos agregados: data/contexto_robustez_bootstrap.csv, "
              "data/contexto_robustez_faixa.csv e data/contexto_robustez_metadata.json. "
              "Metadados contêm versões, seeds e SHA-256 de entradas, código, cache privado e saídas. "
              "Nenhuma saída em data/ contém identificadores ou valores por par.", "",
              "Comando: `" + metadata["command"] + "`."]
    path = ROOT / "docs/plans/RELATORIO_B_CONTEXTO_ROBUSTEZ_2026-11-07.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def generate(data_root, cache_dir=ROOT / ".local/contexto_robustez_oof"):
    started, cpu_started = time.perf_counter(), time.process_time()
    print("Estimativa preliminar: 90 fits OOF únicos, cerca de 10 min de CPU segundo "
          "os tempos anteriores; bootstrap será estimado por 10 pilotos. Teto: 60 min de CPU.", flush=True)
    data = prepare(data_root)
    contexto.augment(data)
    probabilities, cache, versions = probabilities_once(data, cache_dir, cpu_started)
    reproduction = verify_context(data, probabilities)
    below = data["below"]
    aligned = np.argsort(data["ids"][below])
    lexical = data["ids"][below][aligned]
    labels = contexto.numeric(data["raw"].PAR).to_numpy()[below][aligned] == 2
    scores = contexto.numeric(data["raw"].SCORE).to_numpy()[below][aligned]
    q_scores = {"O1": contexto.numeric(data["raw"].SCORE).to_numpy(),
                "O3": data["original"].p_cal_ml.to_numpy(), **probabilities}
    queues = {name: np.searchsorted(lexical, queue(data["ids"], p, below)) for name, p in q_scores.items()}
    for q in queues.values():
        validate_order(q, len(labels))
    baseline_reproduction = fusao.verify_reference(
        {n: queues[n] for n in ("O1", "O3", "O3_MACD")}, labels)
    faixa = faixa_table(queues, labels, scores)
    print("Resumos e curvas C/CM/MACD e referências O1/O3/F3/F4 reproduzidos; faixa F4 também.", flush=True)
    group_codes, groups = pd.factorize(data["groups"][below][aligned], sort=True)
    rng = np.random.default_rng(SEED)
    pilot_wall, pilot_cpu = time.perf_counter(), time.process_time()
    runs = [bootstrap_once(queues, labels, group_codes, len(groups), rng) for _ in range(10)]
    pilot_cpu = time.process_time() - pilot_cpu
    pilot_wall = time.perf_counter() - pilot_wall
    estimated_total_cpu = time.process_time() - cpu_started + pilot_cpu / 10 * 2000 * 1.5
    repetitions = 500 if estimated_total_cpu > CPU_LIMIT_SECONDS else 2000
    print(f"Piloto 10 réplicas: CPU {pilot_cpu:.3f}s, parede {pilot_wall:.3f}s; "
          f"projeção total conservadora {estimated_total_cpu / 60:.2f} min CPU; {repetitions} réplicas.", flush=True)
    for i in range(10, repetitions):
        runs.append(bootstrap_once(queues, labels, group_codes, len(groups), rng))
        if (i + 1) % 250 == 0:
            print(f"Bootstrap {i + 1}/{repetitions}; CPU {(time.process_time() - cpu_started) / 60:.2f} min", flush=True)
        if time.process_time() - cpu_started >= CPU_LIMIT_SECONDS:
            repetitions = 500
            if len(runs) >= 500:
                break
    attempted = len(runs)
    runs = runs[:repetitions]
    valid = [r for r in runs if r is not None]
    if not valid:
        raise ValueError("Nenhuma réplica com PAR=2")
    table = summarize_bootstrap(valid)
    # Verifica também a reprodução do bootstrap F4 histórico para as mesmas réplicas.
    old_bootstrap = pd.read_csv(ROOT / "data/fusao_robustez_bootstrap.csv")
    base_bootstrap = fusao.summarize_bootstrap(valid)
    if repetitions == 2000:
        for col in ("diferenca_revisoes_mediana", "diferenca_revisoes_ic95_inf", "diferenca_revisoes_ic95_sup", "proporcao_fusao_menor"):
            if not np.allclose(base_bootstrap[col], old_bootstrap[col], rtol=0, atol=2e-12):
                raise ValueError(f"Bootstrap F4 histórico não reproduzido: {col}")
    out = ROOT / "data"
    table.to_csv(out / f"{PREFIX}_bootstrap.csv", index=False, lineterminator="\n")
    faixa.to_csv(out / f"{PREFIX}_faixa.csv", index=False, lineterminator="\n")
    refs = ("data/contexto_obito.csv", "data/contexto_obito_curva.csv", "data/contexto_obito_metadata.json",
            "data/contexto_obito_spec.json", "data/fusao_filas.csv", "data/fusao_filas_metadata.json",
            "data/fusao_robustez_faixa.csv", "data/fusao_robustez_bootstrap.csv",
            "data/fusao_robustez_metadata.json", "data/ablation_comprec_allowlist_metadata.json")
    inputs = {"data/COMPARADORSEMIDENT.csv": data["source"],
              **{n: Path(data_root) / n for n in ("data/pcal_ml_oof.csv", "data/pcal_ml_oof.json")},
              **{n: ROOT / n for n in refs}}
    metadata = {
        "status": "completed", "analysis": "B_contexto_robustez_exploratory_post_hoc",
        "date_utc": datetime.now(UTC).isoformat(), "registered": False, "cpu_only": True, "llm_calls": 0,
        "seed": SEED, "fit_seed": FIT_SEED, "inner_seeds": list(range(43, 48)), "folds": N_SPLITS,
        "split_by": "refrec", "calibration": "nested isotonic 5x5", "variants": VARIANTS,
        "comparisons": [f"{a}-{b}" for a, b in COMPARISONS],
        "targets": TARGETS, "fractions_if_less_36": FRACTIONS,
        "targets_rule": "fixed 9/23/36 if total>=36; otherwise ceil(25%/64%/100% of total); zero-positive replicates excluded",
        "dispatch_correction": "Hybrid F4 rule explicitly authorized; fractions are NOT used in all replicates",
        "bootstrap": "REFREC resampled with replacement below cutoff; expanded copies count as reviews; rerank arms and RRF; fixed OOF scores, no replicate refits",
        "confidence_interval": "percentile 2.5/97.5, linear quantiles; conditional on fixed OOF scores and labels",
        "rng": "numpy.default_rng PCG64", "replicas": repetitions, "replicas_tentadas": attempted,
        "replicas_sem_PAR2": len(runs) - len(valid), "replicas_menos_36": sum(r[1] < 36 for r in valid),
        "grupos_REFREC": len(groups), "below_cutoff": len(labels), "PAR2": int(labels.sum()),
        "PAR2_por_replica": fusao.distribution([r[1] for r in valid]),
        "pares_por_replica": fusao.distribution([r[2] for r in valid]),
        "faixa": {"rule": "round(SCORE, 3) >= -5.648", "n": int(fusao.reviewable_mask(scores).sum()),
                  "PAR2": int(labels[fusao.reviewable_mask(scores)].sum()),
                  "minimum_score": float(scores[fusao.reviewable_mask(scores)].min()),
                  "baseline_F4_table_reproduced": True},
        "context_reproduction": reproduction, "baseline_reproduction": baseline_reproduction,
        "F4_bootstrap_reproduced": repetitions == 2000,
        "oof_authorization": "One regeneration per C, CM, O3_MACD; fixed across bootstrap; private checkpoints only",
        "oof_cache": cache, "bootstrap_model_fits": 0,
        "pilot_seconds_10": pilot_wall, "pilot_cpu_seconds_10": pilot_cpu,
        "estimated_total_cpu_2000_minutes_safety_1_5": estimated_total_cpu / 60,
        "cpu_limit_minutes": 60, "reduced_to_500": repetitions == 500,
        "reduction_rule": "500 if projected total or actual CPU reaches 60 minutes; first 500 draws preserved",
        "versions": versions, "input_sha256": {n: sha256(p) for n, p in inputs.items()},
        "code_sha256": {n: sha256(ROOT / n) for n in (*SCIENTIFIC_CODE, "scripts/contexto_robustez.py", "tests/test_contexto_robustez.py")},
        "output_sha256": {f"{PREFIX}_{s}.csv": sha256(out / f"{PREFIX}_{s}.csv") for s in ("bootstrap", "faixa")},
        "elapsed_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
        "command": " ".join([sys.executable, *sys.argv]),
        "limitation": "Post hoc, verification bias, full-universe unlabeled context and shared COMPREC across REFREC folds; no training or reviewer uncertainty; no independent validation",
    }
    write_json(out / f"{PREFIX}_metadata.json", metadata)
    report(table, faixa, metadata)
    print(table.to_string(index=False), flush=True)
    print(faixa.to_string(index=False), flush=True)
    print(f"Concluído: {repetitions} réplicas; CPU {metadata['cpu_seconds']:.2f}s; parede {metadata['elapsed_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".local/contexto_robustez_oof")
    args = parser.parse_args()
    generate(args.data_root, args.cache_dir)
