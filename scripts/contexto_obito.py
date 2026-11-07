"""Contexto de concorrência, post hoc, CPU e saídas exclusivamente agregadas."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import sha256
from gzcmd.loader import resolve_prefixed_column
from gzcmd.ml_oof import N_SPLITS, SEED, _classifier, nested_oof_probabilities
from scripts.comparador_mesma_carga import cumulative_counts, numeric
from scripts.fusao_filas import rrf
from scripts.sensibilidade_o3_macd import estimate_minutes, fit_control, prepare, queue

PREFIX = "contexto_obito"
TARGETS = (9, 23, 36)
TOP_K = (17, 58, 96, 125, 440, 2522)
VARIANTS = ("O1", "O3", "O3_MACD", "F4", "O3_contexto", "O3_contexto_MACD", "F4_contexto")
METRICS = (tuple(f"revisoes_{n}" for n in TARGETS)
           + tuple(f"par2_top_{k}" for k in TOP_K) + ("auc_abaixo", "ap_abaixo"))
CONTEXT_SUFFIXES = ("n_candidatos", "posicao", "delta_melhor", "delta_segundo", "eh_melhor")
CONTEXT_FEATURES = tuple(f"ctx_{group}_{suffix}" for group in ("refrec", "comprec")
                         for suffix in CONTEXT_SUFFIXES)
DEFINITION = {
    "features": list(CONTEXT_FEATURES),
    "score": "nota final (ou nota_final), mesma imputação zero de numeric_features",
    "grouping_keys_only": ["REFREC", "COMPREC"],
    "numeric_inputs": "somente nota final; subscores permanecem nos 30 atributos originais",
    "forbidden_feature_inputs": ["SCORE", "PAR", "C_SITUENCE", "identificadores"],
    "n_candidatos": "número de linhas candidatas no grupo, incluindo o próprio par",
    "posicao": "rank decrescente mínimo, base 1; empates compartilham posição",
    "delta_melhor": "nota do par menos a maior nota do grupo (não positivo)",
    "delta_segundo": "nota do par menos a segunda nota em ordem decrescente, com multiplicidade",
    "singleton": "segunda nota igual à primeira; ambos os deltas zero",
    "eh_melhor": "1 para todos os empatados na maior nota; 0 nos demais",
    "context_scope": "universo completo de candidatos, antes do corte e dos folds; sem rótulos",
    "rrf": {"k": 60, "arms": ["O1", "O3_contexto"], "implementation": "scripts.fusao_filas.rrf"},
    "tuning": "nenhum; definições e atributos fixos antes dos fits e das curvas",
}


def context_features(x, groups):
    """Recebe só a matriz permitida e chaves, sem acesso a rótulo ou SCORE."""
    score_columns = [name for name in ("nota final", "nota_final") if name in x]
    if len(score_columns) != 1:
        raise ValueError("Necessária exatamente uma coluna de nota final")
    score = x[score_columns[0]].astype("float64")
    if not np.isfinite(score.to_numpy()).all():
        raise ValueError("Nota final não finita")
    result = pd.DataFrame(index=x.index)
    for name in ("refrec", "comprec"):
        key = groups[name]
        if not key.index.equals(x.index) or key.isna().any() or key.astype(str).str.strip().eq("").any():
            raise ValueError("Chaves de contexto ausentes ou desalinhadas")
        grouped = score.groupby(key, sort=False)
        best = grouped.transform("max")
        # nlargest mantém a multiplicidade: dois melhores empatados têm a mesma nota.
        second = grouped.agg(lambda s: s.nlargest(2).iloc[-1])
        prefix = f"ctx_{name}_"
        result[prefix + "n_candidatos"] = grouped.transform("size")
        result[prefix + "posicao"] = grouped.rank(ascending=False, method="min")
        result[prefix + "delta_melhor"] = score - best
        result[prefix + "delta_segundo"] = score - key.map(second)
        result[prefix + "eh_melhor"] = score.eq(best).astype(int)
    result = result.loc[:, list(CONTEXT_FEATURES)].astype("float64")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Contexto não finito")
    return result


def augment(data):
    groups = {}
    for name in ("refrec", "comprec"):
        column = resolve_prefixed_column(data["raw"], name.upper())
        if column is None:
            raise ValueError(f"Chave de agrupamento {name} ausente")
        groups[name] = data["raw"][column].str.strip()
    context = context_features(data["x"]["O3_controle"], groups)
    data["x"]["O3_contexto"] = pd.concat([data["x"]["O3_controle"], context], axis=1)
    data["x"]["O3_contexto_MACD"] = pd.concat([data["x"]["O3_MACD"], context], axis=1)


def strict_control(data):
    p, folds, reproduction = fit_control(data)
    reproduction["tolerance"] = float(np.finfo(float).eps)
    if reproduction["max_abs_difference"] > reproduction["tolerance"]:
        raise RuntimeError("Controle O3 diverge além da precisão de máquina")
    return p, folds, reproduction


def fused_scores(ids, below, scores):
    """Reutiliza a permutação RRF e preserva seus empates nas AUC/AP."""
    lexical = np.sort(ids[below])
    if len(np.unique(lexical)) != len(lexical):
        raise ValueError("Hashes duplicados na fila")
    arms = [np.searchsorted(lexical, queue(ids, p, below)) for p in scores]
    order = rrf(arms)
    values = np.zeros(len(lexical))
    for arm in arms:
        values[arm] += 1.0 / (60 + np.arange(1, len(lexical) + 1))
    if not np.array_equal(order, np.lexsort((np.arange(len(lexical)), -values))):
        raise ValueError("Escores RRF não reproduzem a função importada")
    # Só as posições abaixo do corte são utilizadas; não são probabilidades.
    output = np.full(len(ids), np.nan)
    output[below] = values[np.searchsorted(lexical, ids[below])]
    return output


def summarize(data, scores):
    rows, curves = [], []
    below = data["below"]
    labels = numeric(data["raw"].PAR).eq(2).to_numpy()
    if not np.isin(numeric(data["raw"].PAR).to_numpy()[below], [0, 2]).all():
        raise ValueError("Rótulo abaixo do corte diferente de 0/2")
    for variant, p in scores.items():
        counts = cumulative_counts(data["ids"][below], p[below], labels[below])
        if len(counts) != 61485 or counts[-1] != 36:
            raise ValueError("Universo abaixo do corte divergiu de 61.485/36")
        values = {f"revisoes_{t}": int(np.searchsorted(counts, t) + 1) for t in TARGETS}
        values.update({f"par2_top_{k}": int(counts[k - 1]) for k in TOP_K})
        values["auc_abaixo"] = float(roc_auc_score(labels[below], p[below]))
        values["ap_abaixo"] = float(average_precision_score(labels[below], p[below]))
        rows.extend({"variante": variant, "metrica": m, "valor": v} for m, v in values.items())
        curves.append(pd.DataFrame({"variante": variant, "k": np.arange(1, len(counts) + 1),
                                    "par2_acumulados": counts}))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def verify_references(table, curves):
    fusion = pd.read_csv(ROOT / "data/fusao_filas.csv")
    sensitivity = pd.read_csv(ROOT / "data/sensibilidade_o3_macd.csv")
    old_curves = pd.read_csv(ROOT / "data/fusao_filas_curva.csv")
    mapping = {"O1": "O1_SCORE", "O3": "O3_controle", "O3_MACD": "O3_MACD"}
    for name in ("O1", "O3", "O3_MACD", "F4"):
        actual = table.loc[table.variante.eq(name)].set_index("metrica").valor
        for row in fusion.loc[fusion.fila.eq(name)].itertuples():
            if row.metrica in actual and actual[row.metrica] != row.media:
                raise ValueError(f"{name}: resumo difere de fusao_filas")
        curve = curves.loc[curves.variante.eq(name)].sort_values("k")
        old = old_curves.loc[old_curves.fila.eq(name)].sort_values("k")
        if not np.array_equal(curve.k, old.k) or not np.array_equal(curve.par2_acumulados, old.par2_acumulados):
            raise ValueError(f"{name}: curva difere de fusao_filas")
        if name in mapping:
            for row in sensitivity.loc[sensitivity.variante.eq(mapping[name])].itertuples():
                if row.metrica in actual and not np.isclose(actual[row.metrica], row.valor, rtol=0, atol=2e-16):
                    raise ValueError(f"{name}: resumo difere de sensibilidade_o3_macd")
    return {"passed": True, "full_curves": ["O1", "O3", "O3_MACD", "F4"],
            "summaries": ["data/fusao_filas.csv", "data/sensibilidade_o3_macd.csv"]}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def record_estimate_stop(out, spec, estimate, *, report_path=None):
    """Registra interrupção, sem apresentar pilotos como resultados da análise."""
    metadata = {
        "status": "stopped_runtime_estimate", "analysis": "B_contexto_obito_exploratory_post_hoc",
        "registered": False, "cpu_only": True, "llm_calls": 0,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "definition": DEFINITION, "specification": spec, "estimate": estimate,
        "limit_minutes": 45, "completed_nested_fits": 0,
        "context_features": list(CONTEXT_FEATURES), "seed": SEED, "folds": N_SPLITS,
        "split_by": "refrec", "targets": TARGETS, "top_k": TOP_K,
        "command": " ".join([sys.executable, *sys.argv]),
        "output_sha256": {f"{PREFIX}_spec.json": sha256(out / f"{PREFIX}_spec.json")},
        "code_sha256": {n: sha256(ROOT / n) for n in ("scripts/contexto_obito.py", "tests/test_contexto_obito.py")},
        "limitation": "Nenhuma métrica estimada: somente quatro fits-piloto; autorização necessária para superar 45 min.",
    }
    write_json(out / f"{PREFIX}_metadata.json", metadata)
    report_path = report_path or ROOT / "docs/plans/RELATORIO_B_CONTEXTO_2026-11-07.md"
    lines = ["# B: atributos de contexto por óbito", "", "## Situação: interrompida pelo limite de tempo", "",
             "Análise exploratória post hoc, CPU, sem LLM. A execução respeitou a regra de parar "
             "quando a estimativa ultrapassasse 45 minutos. Não foram iniciados os ajustes OOF completos, "
             "não foi verificada a reprodução de O3 e não foram calculadas métricas das novas variantes.", "",
             "## Estimativa antes da execução completa", "",
             "| Matriz do piloto | Segundos |", "|---|---:|"]
    for name, seconds in zip(("O3_controle", "O3_MACD", "O3_contexto", "O3_contexto_MACD"), estimate["pilot_seconds"], strict=True):
        lines.append(f"| {name} | {seconds:.6f} |".replace(".", ","))
    lines += ["", f"Estimativa conservadora: {estimate['estimated_minutes']:.6f} minutos "
              "(maior piloto × 120 fits × fator de segurança 3 / 60).".replace(".", ","), "",
              "São 120 fits aninhados previstos, mais os quatro pilotos concluídos. "
              "Os demais trabalhos do worktree podem afetar a duração, mas não se repetiu o piloto "
              "nem se reduziu o fator de segurança para contornar o limite.", "",
              "## Definição fixada", "",
              "Cinco atributos por REFREC e os mesmos cinco por COMPREC: número de candidatos, "
              "posição pela nota final, nota menos a melhor nota, nota menos a segunda nota "
              "e indicador de melhor candidato. Empates usam rank mínimo; segunda nota inclui "
              "multiplicidade; grupos unitários têm deltas zero. Identificadores servem apenas "
              "como chaves, sem codificação como preditores. O contexto não consulta SCORE, PAR ou "
              "C_SITUENCE. Universo completo de candidatos, sem rótulos, antes dos folds.", "",
              "## Perguntas ainda sem resposta", "",
              "| Comparação | Topo | Cauda | Fusão |", "|---|---|---|---|",
              "| O3 versus O3+contexto | Não calculado | Não calculado | Não se aplica |",
              "| O3+MACD versus O3+contexto+MACD | Não calculado | Não calculado | Não se aplica |",
              "| F4 versus F4' | Não calculado | Não calculado | Não calculado |", "",
              "Não há evidência desta execução sobre melhora do topo, da cauda ou da fusão. "
              "Nenhum CSV de métricas foi criado. A hipótese original O3 versus O1 permanece inalterada.", "",
              "## Artefatos e continuação", "",
              "Scripts e testes novos: scripts/contexto_obito.py e tests/test_contexto_obito.py. "
              "Especificação anterior ao carregamento/fits: data/contexto_obito_spec.json. "
              "Estimativa e interrupção: data/contexto_obito_metadata.json. "
              "É necessária autorização explícita para ampliar o limite antes de executar a análise completa.", "",
              "Comando interrompido: `" + metadata["command"] + "`"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return metadata


def write_report(table, metadata):
    available = [name for name in VARIANTS if name in set(table.variante)]
    wide = table.pivot(index="variante", columns="metrica", values="valor").loc[available]
    lines = ["# B: atributos de contexto por óbito", "", "## Delimitação", "",
             "Análise exploratória post hoc, em CPU, sem LLM e sem ajuste de hiperparâmetros. "
             "Não altera a hipótese O3 versus O1 nem sua margem de 25%. Revisor perfeito, "
             "61.485 potenciais pares abaixo do corte e 36 PAR=2. O provável viés de verificação "
             "do serviço favorece O1; recuperação destes PAR=2 não é validação independente de sensibilidade.", "",
             "## Atributos fixados antes dos fits", "",
             "Cinco atributos por REFREC e os mesmos cinco por COMPREC: número de candidatos, "
             "posição pela nota final, nota menos a melhor nota, nota menos a segunda nota e indicador "
             "de melhor candidato. Ranking mínimo em empates, segunda nota com multiplicidade; "
             "em grupos unitários, deltas zero. Os identificadores são somente chaves de agrupamento, "
             "nunca valores preditores. SCORE, PAR e C_SITUENCE não entram no contexto.", "",
             "O contexto usa o universo completo de candidatos, inclusive acima do corte, sem rótulos. "
             "É informação transdutiva disponível antes da revisão, não estatística ajustada com os "
             "rótulos dos folds. COMPREC pode ser compartilhado entre folds por REFREC, como no desenho "
             "original; não se estima aqui generalização a candidatos ou bases inteiramente novos. "
             "As MACD preservam a definição original, inclusive seus insumos temporais.", "",
             "## Resultados", "", "F4_contexto corresponde a F4': RRF(60) de O1 e O3_contexto.", "",
             "| Variante | " + " | ".join(METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
    for name, row in wide.iterrows():
        cells = [f"{row[m]:.6f}".replace(".", ",") if m.startswith(("auc", "ap"))
                 else f"{int(row[m]):,}".replace(",", ".") for m in METRICS]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += ["", "## Leitura curta", ""]
    for before, after in (("O3", "O3_contexto"), ("O3_MACD", "O3_contexto_MACD"), ("F4", "F4_contexto")):
        if after not in wide.index:
            lines.append(f"- {after}: não executada, conforme a regra de projeção do orçamento de CPU.")
            continue
        a, b = wide.loc[before], wide.loc[after]
        top = "; ".join(f"k={k}: {int(a[f'par2_top_{k}'])} para {int(b[f'par2_top_{k}'])}" for k in (17, 58, 96, 125))
        tail = "; ".join(f"{t} PAR=2: {int(a[f'revisoes_{t}']):,} para {int(b[f'revisoes_{t}']):,}".replace(",", ".") for t in TARGETS)
        changes = [b[f"par2_top_{k}"] - a[f"par2_top_{k}"] for k in (17, 58, 96, 125)]
        verdict = ("melhora sem perdas nos quatro orçamentos iniciais" if min(changes) >= 0 and max(changes) > 0
                   else "piora sem ganhos nos quatro orçamentos iniciais" if max(changes) <= 0 and min(changes) < 0
                   else "empata nos quatro orçamentos iniciais" if not any(changes) else "tem efeito misto no topo")
        tail_verdict = "melhora" if b.revisoes_36 < a.revisoes_36 else "piora" if b.revisoes_36 > a.revisoes_36 else "empata"
        lines.append(f"- {before} para {after}: {verdict}; {top}. Na cauda, {tail_verdict}. Revisões: {tail}.")
    lines += ["", "AUC e AP usam os escores originais, preservando empates; na fusão, usam a soma "
              "RRF, não uma probabilidade. Na fila, os empates são resolvidos pelo hash lexical, "
              "exatamente k, sem expansão. Não há teste confirmatório nem seleção posterior de atributos.", "",
              "## Reprodução e proveniência", "",
              "Mesmas funções importadas de sensibilidade_o3_macd e fusao_filas, partições aninhadas "
              "5×5 por REFREC, seed 42, seeds internas 43 a 47 e calibração isotônica. "
              "O controle foi reajustado e validado antes das variantes. O1, O3 e F4 "
              "reproduziram as curvas completas existentes. O3+MACD foi importada dos agregados "
              "de referência, sem reajuste; todos os resumos foram comparados aos dois CSVs de referência.", "",
              "```json", json.dumps(metadata["control_reproduction"], indent=2), "```", "",
              "Artefatos: data/contexto_obito.csv, data/contexto_obito_curva.csv, "
              "data/contexto_obito_spec.json e data/contexto_obito_metadata.json. "
              "Somente agregados, sem identificadores ou valores individuais. A especificação foi gravada "
              "antes do carregamento dos dados e dos fits; isso não constitui pré-registro.", "",
              "Comando: `" + metadata["command"] + "`", "",
              f"Estimativa conservadora de parede: {metadata['estimate']['estimated_minutes']:.2f} min; "
              f"CPU projetada: {metadata['estimate']['projected_total_cpu_minutes']:.2f} min. "
              f"Tempo real de parede: {metadata['elapsed_seconds']:.1f} s; "
              f"CPU real: {metadata['cpu_seconds']:.1f} s; teto autorizado: {metadata['cpu_limit_minutes']:.0f} min.", "",
              "Ordem: controle O3, O3+contexto, checkpoint agregado com F4', e somente depois "
              "O3+contexto+MACD quando couber no orçamento. O checkpoint da primeira variante consta "
              "de data/contexto_obito_etapa1.json. Hashes, versões, tempos por variante e decisão "
              "de orçamento constam dos metadados."]
    path = ROOT / "docs/plans/RELATORIO_B_CONTEXTO_2026-11-07.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def estimate_plan(data, cpu_limit_minutes):
    """Pilotos originais, com medida explícita de CPU de cada matriz."""
    wall, cpu = {}, {}
    for name, x in data["x"].items():
        start = time.process_time()
        original = estimate_minutes({"x": {name: x}, "y": data["y"], "groups": data["groups"]})
        cpu[name] = time.process_time() - start
        wall[name] = original["pilot_seconds"][0]
    planned = ("O3_controle", "O3_contexto", "O3_contexto_MACD")
    projected_cpu = sum(cpu[name] for name in planned) * 30 * 3 / 60 + sum(cpu.values()) / 60
    return {"pilot_seconds": list(wall.values()), "pilot_wall_by_matrix": wall,
            "pilot_cpu_by_matrix": cpu, "pilot_fits": len(cpu), "model_fits": 90,
            "safety_factor": 3, "estimated_minutes": max(wall.values()) * 90 * 3 / 60,
            "projected_total_cpu_minutes": projected_cpu,
            "include_context_macd": projected_cpu <= cpu_limit_minutes}


def stored_macd():
    """Referência agregada já calculada, sem refit nem cache de valores por par."""
    reference = pd.read_csv(ROOT / "data/sensibilidade_o3_macd.csv")
    fusion = pd.read_csv(ROOT / "data/fusao_filas.csv")
    values = reference.loc[reference.variante.eq("O3_MACD")].set_index("metrica").valor.to_dict()
    values["par2_top_125"] = float(fusion.loc[
        fusion.fila.eq("O3_MACD") & fusion.metrica.eq("par2_top_125"), "media"].item())
    rows = pd.DataFrame([{"variante": "O3_MACD", "metrica": m, "valor": values[m]} for m in METRICS])
    curves = pd.read_csv(ROOT / "data/fusao_filas_curva.csv")
    curves = curves.loc[curves.fila.eq("O3_MACD"), ["fila", "k", "par2_acumulados"]].rename(columns={"fila": "variante"})
    return rows, curves


def save_outputs(out, table, curves, metadata):
    table.to_csv(out / f"{PREFIX}.csv", index=False, lineterminator="\n")
    curves.to_csv(out / f"{PREFIX}_curva.csv", index=False, lineterminator="\n")
    metadata["output_sha256"] = {f"{PREFIX}{suffix}": sha256(out / f"{PREFIX}{suffix}")
                                 for suffix in (".csv", "_curva.csv", "_spec.json")}
    if (out / f"{PREFIX}_etapa1.json").exists():
        metadata["output_sha256"][f"{PREFIX}_etapa1.json"] = sha256(out / f"{PREFIX}_etapa1.json")
    write_json(out / f"{PREFIX}_metadata.json", metadata)


def generate(data_root=ROOT, out=ROOT / "data", *, estimate_only=False, cpu_limit_minutes=45):
    import sklearn
    import xgboost

    start = time.perf_counter()
    cpu_start = time.process_time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / f"{PREFIX}_spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec["definition"] != DEFINITION:
            raise RuntimeError("Especificação anterior divergiu; não substituir após observar resultados")
    else:
        spec = {"fixed_before_data_and_fits_utc": datetime.now(timezone.utc).isoformat(),
                "registered": False, "definition": DEFINITION}
        write_json(spec_path, spec)
    data = prepare(data_root)
    expected = data["original_meta"]["fingerprint"]
    if sklearn.__version__ != expected["sklearn"] or xgboost.__version__ != expected["xgboost"]:
        raise RuntimeError("Versões divergem de O3 original")
    augment(data)
    estimate = estimate_plan(data, cpu_limit_minutes)
    print("Estimativa antes da execução completa: " + json.dumps(estimate), flush=True)
    if estimate_only:
        return None
    timings = {}
    t, c = time.perf_counter(), time.process_time()
    p, folds, reproduction = strict_control(data)
    timings["O3_controle"] = {"wall_seconds": time.perf_counter() - t, "cpu_seconds": time.process_time() - c}
    scores = {"O1": numeric(data["raw"].SCORE).to_numpy(), "O3": p}
    scores["F4"] = fused_scores(data["ids"], data["below"], [scores["O1"], p])
    macd_table, macd_curves = stored_macd()

    def summarize_with_reference():
        table, curves = summarize(data, scores)
        return (pd.concat([table, macd_table], ignore_index=True),
                pd.concat([curves, macd_curves], ignore_index=True))

    baseline_table, baseline_curves = summarize_with_reference()
    references = verify_references(baseline_table, baseline_curves)
    print("O1/O3/F4 reproduzidos; referência agregada O3_MACD verificada, sem refit.", flush=True)
    inputs = ("data/COMPARADORSEMIDENT.csv", "data/pcal_ml_oof.csv", "data/pcal_ml_oof.json")
    refs = ("data/sensibilidade_o3_macd.csv", "data/fusao_filas.csv", "data/fusao_filas_curva.csv",
            "data/ablation_comprec_allowlist_metadata.json")
    codes = ("scripts/contexto_obito.py", "tests/test_contexto_obito.py", "scripts/sensibilidade_o3_macd.py",
             "scripts/fusao_filas.py", "scripts/comparador_mesma_carga.py", "gzcmd/ml_oof.py",
             "gzcmd/loader.py", "gzcmd/arranjo_corte.py")
    metadata = {
        "status": "control_verified",
        "analysis": "B_contexto_obito_exploratory_post_hoc", "registered": False,
        "date_utc": datetime.now(timezone.utc).isoformat(), "cpu_only": True, "llm_calls": 0,
        "definition": DEFINITION, "specification": spec,
        "features": {name: list(x) for name, x in data["x"].items()},
        "seed": SEED, "inner_seeds": list(range(SEED + 1, SEED + N_SPLITS + 1)),
        "folds": N_SPLITS, "split_by": "refrec", "calibration": "nested isotonic 5x5",
        "classifier_parameters": {k: "NaN" if isinstance(v, float) and np.isnan(v) else v
                                  for k, v in _classifier(data["y"], SEED).get_params().items()},
        "scale_pos_weight": "recalculado em cada treinamento pela função original",
        "missing_values": {"original_30": 0.0, "macd": -1.0},
        "control_reproduction": reproduction, "baseline_reproduction": references,
        "below_cutoff": 61485, "par2": 36, "targets": TARGETS, "top_k": TOP_K,
        "ordering": "score descending, lexical pair hash ascending, exactly k",
        "auc_ap": "raw score, isotonic probability or actual RRF sum, ties preserved",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        "input_sha256": {**{n: sha256(Path(data_root) / n) for n in inputs},
                         **{n: sha256(ROOT / n) for n in refs}},
        "code_sha256": {n: sha256(ROOT / n) for n in codes},
        "estimate": estimate, "elapsed_seconds": time.perf_counter() - start,
        "cpu_seconds": time.process_time() - cpu_start, "cpu_limit_minutes": cpu_limit_minutes,
        "runtime_authorization": "Continuação autorizada além de 45 min, teto de 150 min de CPU",
        "timings": timings, "execution_order": [], "checkpoints": [],
        "reference_macd": "stored aggregate metrics and curve, no refit",
        "command": " ".join([sys.executable, *sys.argv]),
        "limitation": "Verification bias favors O1; full-universe unlabeled context; COMPREC can span REFREC folds; post hoc.",
    }
    reserve = max(estimate["pilot_cpu_by_matrix"].values()) * 3

    def check_budget(*_):
        if time.process_time() - cpu_start + reserve >= cpu_limit_minutes * 60:
            raise RuntimeError("Orçamento de CPU esgotado; checkpoint anterior preservado")

    for variant in ("O3_contexto", "O3_contexto_MACD"):
        if variant == "O3_contexto_MACD":
            projection = ((time.process_time() - cpu_start)
                          + estimate["pilot_cpu_by_matrix"][variant] * 30 * 3) / 60
            metadata["context_macd_projection_cpu_minutes"] = projection
            if not estimate["include_context_macd"] or projection > cpu_limit_minutes:
                metadata["skipped_variants"] = {variant: "Projeção total superior ao teto de CPU"}
                break
        check_budget()
        metadata["execution_order"].append({"variant": variant, "started_utc": datetime.now(timezone.utc).isoformat()})
        t, c = time.perf_counter(), time.process_time()

        def budgeted_classifier(y, seed):
            check_budget()
            return _classifier(y, seed)

        scores[variant], current_folds = nested_oof_probabilities(
            data["x"][variant], data["y"], split_by="refrec", groups=data["groups"],
            model_factory=budgeted_classifier, audit=check_budget)
        if not np.array_equal(folds, current_folds):
            raise ValueError("Partições das variantes divergem")
        timings[variant] = {"wall_seconds": time.perf_counter() - t, "cpu_seconds": time.process_time() - c}
        if variant == "O3_contexto":
            scores["F4_contexto"] = fused_scores(data["ids"], data["below"], [scores["O1"], scores[variant]])
        table, curves = summarize_with_reference()
        metadata.update(status="partial_context_completed", date_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu_start)
        if variant == "O3_contexto":
            checkpoint = {"completed_utc": metadata["date_utc"], "variant": variant,
                          "control_reproduction": reproduction, "metrics": table.to_dict(orient="records"),
                          "cpu_seconds": metadata["cpu_seconds"], "elapsed_seconds": metadata["elapsed_seconds"]}
            write_json(out / f"{PREFIX}_etapa1.json", checkpoint)
            metadata["checkpoints"].append({"path": f"{PREFIX}_etapa1.json", "completed_utc": checkpoint["completed_utc"]})
        save_outputs(out, table, curves, metadata)
        print(f"Checkpoint {variant}: parede={timings[variant]['wall_seconds']:.1f} s; "
              f"CPU={timings[variant]['cpu_seconds']:.1f} s", flush=True)
    metadata.update(status="completed", date_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu_start)
    save_outputs(out, table, curves, metadata)
    write_report(table, metadata)
    print(table.pivot(index="variante", columns="metrica", values="valor").to_string(), flush=True)
    print(f"Concluído em {metadata['elapsed_seconds']:.1f} s", flush=True)
    return table, curves, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--cpu-limit-minutes", type=float, default=45)
    args = parser.parse_args()
    generate(args.data_root, args.out, estimate_only=args.estimate_only, cpu_limit_minutes=args.cpu_limit_minutes)
