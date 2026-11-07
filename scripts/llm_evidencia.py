"""Análise C post hoc, offline: resposta do LLM como evidência, só agregados.

Faixas, Laplace=1 e RRF=60 fixados no código antes de observar os resultados.
Não é um pré-registro nem uma validação independente das políticas históricas.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import MODEL, REMOVED_FIELDS, below_cutoff, sha256
from gzcmd.ml_oof import N_SPLITS, SEED, numeric_features, pair_ids
from gzcmd.loader import resolve_prefixed_column
from scripts.comparador_mesma_carga import align_original, load_raw, numeric
from scripts.fusao_filas import rrf

DECISIONS = ("MATCH", "NONMATCH", "UNSURE")
MODES = ("vigilancia", "confirmacao")
EDGES = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1.0])
ALPHA = 1.0
TARGETS = (9, 23, 36)
TOP_K = (17, 58, 96, 125, 440)
REPORT = "RELATORIO_C_LLM_EVIDENCIA_2026-11-07.md"


def resolve(name, data_root):
    local = ROOT / name
    return local if local.exists() else Path(data_root) / name


def select_oldest(runs):
    """Mais antigo por timestamp do run, caminho lexical e posição da linha.

    Checkpoints antigos não têm horário por resposta. Não usar mtime do checkout.
    """
    selected, observed = {}, {}
    for stamp, name, rows in sorted(runs, key=lambda item: (item[0], item[1])):
        for row in rows:
            pid, decision = row["pair_id"], row["decision"].strip().upper()
            if decision not in DECISIONS:
                raise ValueError("Decisão inválida")
            selected.setdefault(pid, decision)
            observed.setdefault(pid, set()).add(decision)
    return selected, sum(len(values) > 1 for values in observed.values())


def collect_responses(data_root):
    roots = (ROOT / "data/sprint3b", Path(data_root) / "data/sprint3b")
    directories = {"anthropic_sem_mo_v4", "anthropic_sem_mo_v4_lote31",
                   "anthropic_sem_mo_v4_lote31_refrec", "anthropic_sem_mo_v4_arranjo_corte",
                   "anthropic_sem_mo_v4_arranjo_corte_supl",
                   "anthropic_sem_mo_v4_arranjo_robustez", "anthropic_sem_mo_v4_filtro_llm"}
    for base in roots:
        directories.update(p.name for p in base.glob("anthropic_sem_mo_v4_arranjo_corte*") if p.is_dir())
    prompt = sha256(ROOT / "gzcmd/gzcmd_v4_llm_prompt.md")
    result, audit, missing = {}, [], []
    for mode in MODES:
        runs = []
        for directory in sorted(directories):
            name = f"data/sprint3b/{directory}/exp2_llm_sync_{mode}.json"
            path = resolve(name, data_root)
            if not path.exists():
                missing.append(name)
                continue
            run = json.loads(path.read_text(encoding="utf-8"))
            if (run.get("model") != MODEL or run.get("mode") != mode
                    or run.get("prompt_version") != "v4" or run.get("prompt_sha256") != prompt
                    or set(run.get("removed_fields", [])) != set(REMOVED_FIELDS)
                    or run.get("complete") is not True or run.get("n_errors") != 0):
                raise ValueError(f"Proveniência inválida: {name}")
            stamp = datetime.fromisoformat(run["timestamp"])
            if stamp.tzinfo is None:
                raise ValueError("Timestamp sem fuso")
            rows = run["results"]
            if run["n_reviewed"] != len(rows):
                raise ValueError("Contagem do checkpoint divergiu")
            runs.append((stamp, name, rows))
            audit.append({"source": name, "resolved_path": str(path), "sha256": sha256(path),
                          "modo": mode, "timestamp": stamp.isoformat(), "n": len(rows)})
        if not runs:
            raise ValueError(f"Nenhuma resposta disponível: {mode}")
        selected, conflicts = select_oldest(runs)
        result[mode] = selected
        audit.append({"modo": mode, "unique_responses": len(selected), "conflicting_pairs": conflicts})
    return result, audit, missing


def probability_bands(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilidades fora de [0,1]")
    return np.searchsorted(EDGES[1:-1], p, side="right")


def crossfit_update(p, y, decisions, folds, groups, audit=None):
    """RV=P(decisão|vínculo,faixa)/P(decisão|não vínculo,faixa).

    Somente respostas nos outros folds entram nos denominadores. Sem resposta
    preserva exatamente O3. Sem nenhuma das classes na faixa, RV=1 (não há
    contraste identificável), mesmo com Laplace; a ocorrência fica registrada.
    """
    p = np.asarray(p, dtype=float)
    y, decisions = np.asarray(y), np.asarray(decisions)
    folds, groups = np.asarray(folds), np.asarray(groups)
    if not (len(p) == len(y) == len(decisions) == len(folds) == len(groups)):
        raise ValueError("Vetores desalinhados")
    if not np.isin(y, [0, 1]).all() or not np.isin(decisions, ["", *DECISIONS]).all():
        raise ValueError("Rótulos ou decisões inválidos")
    if pd.isna(groups).any() or pd.isna(folds).any() or len(np.unique(folds)) < 2:
        raise ValueError("Grupos/folds inválidos")
    bands = probability_bands(p)
    posterior, rows = p.copy(), []
    responded = decisions != ""
    for fold in np.unique(folds):
        test, train = folds == fold, folds != fold
        if not set(groups[test]).isdisjoint(groups[train]):
            raise ValueError("REFREC atravessa folds")
        if audit is not None:
            audit(np.flatnonzero(train & responded), np.flatnonzero(test))
        for band in range(len(EDGES) - 1):
            fit = train & responded & (bands == band)
            pos, neg = fit & (y == 1), fit & (y == 0)
            npos, nneg = int(pos.sum()), int(neg.sum())
            fallback = npos == 0 or nneg == 0
            for decision in DECISIONS:
                npos_d = int((pos & (decisions == decision)).sum())
                nneg_d = int((neg & (decisions == decision)).sum())
                prob_pos = (npos_d + ALPHA) / (npos + len(DECISIONS) * ALPHA)
                prob_neg = (nneg_d + ALPHA) / (nneg + len(DECISIONS) * ALPHA)
                lr = 1.0 if fallback else prob_pos / prob_neg
                target = test & (bands == band) & (decisions == decision)
                # Forma algébrica das odds, preserva p=0 e p=1 sem clipping.
                posterior[target] = p[target] * lr / (1 - p[target] + p[target] * lr)
                rows.append({"fold": int(fold), "faixa": band, "p_inf": EDGES[band],
                             "p_sup": EDGES[band + 1], "decisao": decision,
                             "n_vinculos": npos, "n_nao_vinculos": nneg,
                             "n_decisao_vinculos": npos_d, "n_decisao_nao_vinculos": nneg_d,
                             "p_decisao_vinculo_laplace": prob_pos,
                             "p_decisao_nao_vinculo_laplace": prob_neg,
                             "RV": lr, "fallback_neutro": fallback,
                             "n_aplicados": int(target.sum())})
    return posterior, pd.DataFrame(rows)


def ranking_metrics(order, labels):
    counts = np.cumsum(np.asarray(labels)[order])
    result = {f"revisoes_{target}": int(np.searchsorted(counts, target) + 1)
              for target in TARGETS}
    result.update({f"par2_top_{k}": int(counts[k - 1]) for k in TOP_K})
    return result


def acceptance(p, y, below):
    accepted = below & (p >= 0.5)
    n = int(accepted.sum())
    tp = int((accepted & (y == 1)).sum())
    return {"aceitacoes": n, "VP": tp, "FP": n - tp, "VPP": tp / n if n else None}


def write_report(summary, meta, report):
    def fmt(value):
        if pd.isna(value):
            return "não definido"
        if float(value).is_integer():
            return f"{int(value):,}".replace(",", ".")
        return f"{value:.4f}".replace(".", ",")

    lines = ["# C: resposta do LLM como evidência bayesiana", "",
             "Análise exploratória post hoc, somente CPU, sem chamadas de API. Claude Haiku 5.5, prompt v4, dossiê sem saídas do motor. Não altera o perfil histórico nem substitui a hipótese O3 versus O1.", "",
             "## Cobertura e fontes", "",
             "ATENÇÃO: " + ("checkpoint(s) ausente(s); execução parcial das fontes solicitadas." if meta["missing_sources"] else "todas as fontes solicitadas disponíveis."), ""]
    for mode, cov in meta["coverage"].items():
        lines.append(f"* {mode}: {cov['abaixo_com_resposta']:,} de 61.485 abaixo do corte; {cov['par2_com_resposta']} de 36 PAR=2 ({100 * cov['fracao_par2']:.1f}%); {cov['par1_com_resposta']} de 211 PAR=1.".replace(",", ".").replace(f"{100 * cov['fracao_par2']:.1f}%", f"{100 * cov['fracao_par2']:.1f}%".replace(".", ",")))
    lines += ["", "Ausências (busca no worktree e no diretório de dados alternativo):", ""]
    lines.extend("* " + name for name in meta["missing_sources"])
    lines += ["", "A cobertura das respostas não é aleatória: o LLM viu pares escolhidos pelas políticas. A ordenação depende de quais pares têm resposta. Não extrapolar o efeito aos pares não respondidos.", "",
              "Duplicatas: vence o timestamp mais antigo do run, depois caminho lexical e posição da linha. Timestamp é do checkpoint, não de cada chamada; não se usa mtime. Fontes, hashes e conflitos constam dos metadados.", "",
              "## Ordenação abaixo do corte", "",
              "Revisor perfeito, 61.485 potenciais pares e 36 PAR=2. Desempate pelo hash lexical em memória, sem exportar identificadores. RRF com constante 60; F4 combina O1 e O3; RRF-post combina O1 e O3-post.", ""]
    metrics = [*(f"revisoes_{n}" for n in TARGETS), *(f"par2_top_{k}" for k in TOP_K)]
    lines += ["| Modo | Fila | " + " | ".join(metrics) + " |", "|---|---|" + "---:|" * len(metrics)]
    for (mode, variant), part in summary[summary.tipo.eq("ordenacao")].groupby(["modo", "variante"], sort=False):
        values = part.set_index("metrica").valor
        lines.append("| " + mode + " | " + variant + " | " + " | ".join(fmt(values[m]) for m in metrics) + " |")
    lines += ["", "## Aceitação hipotética: probabilidade ≥ 0,5", "",
              "| Modo | Regra | Aceitações | VP | FP | VPP |", "|---|---|---:|---:|---:|---:|"]
    for (mode, variant), part in summary[summary.tipo.eq("aceitacao")].groupby(["modo", "variante"], sort=False):
        values = part.set_index("metrica").valor
        vpp = "não definido" if pd.isna(values.VPP) else f"{100 * values.VPP:.1f}%".replace(".", ",")
        lines.append(f"| {mode} | {variant} | {fmt(values.aceitacoes)} | {fmt(values.VP)} | {fmt(values.FP)} | {vpp} |")
    lines += ["", "P-LLM é referência histórica, não comparação com mesma cobertura: em vigilância, 25 aceitações, 4 VP, 21 FP e VPP de 16,0%; em confirmação, 12 aceitações, 8 VP, 4 FP e VPP de 66,7%. Fonte: data/arranjo_corte_vpp_llm.csv.", "",
              "## Leitura curta", ""]
    for mode in MODES:
        values = summary[summary.modo.eq(mode)].pivot(index="variante", columns="metrica", values="valor")
        lines.append(f"* {mode}: para os 36 PAR=2, O3 requer {fmt(values.loc['O3', 'revisoes_36'])} revisões, O3-post {fmt(values.loc['O3-post', 'revisoes_36'])}, F4 {fmt(values.loc['F4', 'revisoes_36'])} e RRF-post {fmt(values.loc['RRF-post', 'revisoes_36'])}. O3-post produz {fmt(values.loc['O3-post', 'FP'])} FP na aceitação hipotética, contra {fmt(values.loc['O3', 'FP'])} no O3 com o mesmo limiar.")
        vpp_post = f"{100 * values.loc['O3-post', 'VPP']:.1f}%".replace(".", ",")
        vpp_p = f"{100 * values.loc['P-LLM', 'VPP']:.1f}%".replace(".", ",")
        lines.append(f"  Para 9/23 PAR=2, O3-post requer {fmt(values.loc['O3-post', 'revisoes_9'])}/{fmt(values.loc['O3-post', 'revisoes_23'])} revisões, contra {fmt(values.loc['O3', 'revisoes_9'])}/{fmt(values.loc['O3', 'revisoes_23'])} no O3. VPP de O3-post: {vpp_post}, contra {vpp_p} na referência histórica P-LLM. A redução de FP em relação a O3 não elimina aceitações erradas.")
    lines += ["", "Uma melhora no topo ou menos FP não implica melhora uniforme, nem garante ausência de aceitações erradas. Os FP observados e a cobertura parcial impedem conclusão de segurança operacional.", "",
              "## Método e limites", "",
              "RV por decisão e faixa de O3: [0;0,01), [0,01;0,05), [0,05;0,1), [0,1;0,5), [0,5;1]. Faixas fixas antes da execução, não quintis ajustados aos resultados nem necessariamente as faixas da tese. Laplace α=1 nas três categorias; se faltar uma classe na faixa de treinamento, RV neutra 1, com registro explícito. Estimação por modo, com todos os respondidos dos outros quatro folds (incluindo PAR=1). Vínculo é PAR em {1,2}. Posterior = p×RV/(1−p+p×RV); sem resposta mantém O3 exatamente. Valores 0 e 1 não são truncados: evidência finita não os move.", "",
              "Mesmos cinco folds externos de O3, agrupados por REFREC, seed 42, reconstruídos com numeric_features e conferidos contra o cache alinhado. Nenhum rótulo do fold avaliado entra diretamente nos contadores da RV. Ressalva: os escores OOF de outros folds foram produzidos por modelos que podem ter usado rótulos do fold avaliado; reutilizar esse cache não constitui validação aninhada integral do empilhamento. Seleção adaptativa das coortes e viés de verificação do serviço também permanecem. As probabilidades atualizadas não têm garantia de calibração: o uso bayesiano é heurístico, condicionado a faixas grosseiras e à amostra selecionada.", "",
              "Comando: `" + meta["command"] + "`", "",
              "Artefatos agregados: data/llm_evidencia.csv, data/llm_evidencia_rv.csv e data/llm_evidencia_metadata.json. Metadados incluem fontes ausentes, hashes, versões, cobertura, regras e verificações. Nenhum identificador de par é exportado."]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def generate(data_root, out, report):
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold

    source = resolve("data/COMPARADORSEMIDENT.csv", data_root)
    original_path = resolve("data/pcal_ml_oof.csv", data_root)
    raw = load_raw(source)
    below = below_cutoff(raw)
    original = align_original(raw, original_path, source)
    y = numeric(raw.PAR).isin([1, 2]).astype(int).to_numpy()
    groups = raw[resolve_prefixed_column(raw, "REFREC")].str.strip().to_numpy()
    folds = original.fold.to_numpy(dtype=int)
    recreated = np.full(len(raw), -1)
    for fold, (_, test) in enumerate(StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED).split(numeric_features(raw), y, groups)):
        recreated[test] = fold
    if not np.array_equal(folds, recreated):
        raise ValueError("Folds reconstruídos divergem do O3")
    p = original.p_cal_ml.to_numpy()
    ids = pair_ids(raw).to_numpy()
    responses, audit, missing = collect_responses(data_root)
    # Índices abaixo do corte em ordem lexical tornam o desempate idêntico a F4.
    indices = np.flatnonzero(below)
    indices = indices[np.argsort(ids[below])]
    labels = numeric(raw.PAR).to_numpy()[indices] == 2
    o1 = np.argsort(-numeric(raw.SCORE).to_numpy()[indices], kind="stable")
    o3 = np.argsort(-p[indices], kind="stable")
    rows, rv_parts, coverage = [], [], {}
    def add(mode, variant, kind, values):
        rows.extend({"modo": mode, "variante": variant, "tipo": kind, "metrica": k, "valor": v}
                    for k, v in values.items())
    reference_path = resolve("data/arranjo_corte_vpp_llm.csv", data_root)
    references = pd.read_csv(reference_path)
    for mode in MODES:
        selected = responses[mode]
        if set(selected) - set(ids):
            raise ValueError("Resposta fora do universo")
        decisions = np.array([selected.get(pid, "") for pid in ids])
        responded = decisions != ""
        cov = {"abaixo_com_resposta": int((responded & below).sum()),
               "par2_com_resposta": int((responded & below & (y == 1)).sum()),
               "par1_com_resposta": int((responded & ~below).sum())}
        cov["fracao_par2"] = cov["par2_com_resposta"] / 36
        coverage[mode] = cov
        add(mode, "cobertura", "cobertura", cov)
        post, rv = crossfit_update(p, y, decisions, folds, groups)
        rv.insert(0, "modo", mode)
        rv_parts.append(rv)
        qpost = np.argsort(-post[indices], kind="stable")
        orders = {"O1": o1, "O3": o3, "F4": rrf([o1, o3]),
                  "O3-post": qpost, "RRF-post": rrf([o1, qpost])}
        for variant, order in orders.items():
            add(mode, variant, "ordenacao", ranking_metrics(order, labels))
        if [ranking_metrics(o3, labels)[f"revisoes_{t}"] for t in TARGETS] != [29, 125, 23443]:
            raise ValueError("O3 não reproduziu 29/125/23443")
        if [ranking_metrics(o1, labels)[f"revisoes_{t}"] for t in TARGETS] != [51, 501, 2521]:
            raise ValueError("O1 não reproduziu 51/501/2521")
        for variant, probs in (("O3", p), ("O3-post", post)):
            add(mode, variant, "aceitacao", acceptance(probs, y, below))
        ref = references[references.modo.eq(mode) & references.braco.eq("P")].iloc[0]
        add(mode, "P-LLM", "aceitacao", {"aceitacoes": int(ref.aceitacoes_llm_abaixo),
                                         "VP": int(ref.VP), "FP": int(ref.FP), "VPP": float(ref.VPP)})
    summary, rv_frame = pd.DataFrame(rows), pd.concat(rv_parts, ignore_index=True)
    meta = {"analysis": "C_exploratory_post_hoc", "registered": False,
            "timestamp": datetime.now(timezone.utc).isoformat(), "cpu_only": True, "api_calls": 0,
            "command": " ".join([sys.executable, *sys.argv]),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "sklearn": sklearn.__version__},
            "seed": SEED, "n_folds": N_SPLITS, "folds_reproduced": True, "split_by": "REFREC",
            "laplace_alpha": ALPHA, "edges": EDGES.tolist(), "rrf_k": 60,
            "definitions_fixed_before_outcomes": True,
            "endpoints": "finite RV leaves p=0 and p=1 unchanged, no clipping",
            "universe": {"below_cutoff": 61485, "PAR2": 36, "PAR1": 211},
            "empty_class_fallback": "RV=1", "unanswered": "exact original O3",
            "deduplication": "oldest run timestamp; lexical source; first row; no per-response timestamp",
            "training_population": "all answered PAR in {1,2} vs others; exclude evaluated fold",
            "coverage": coverage, "sources": audit, "missing_sources": missing,
            "source_set_complete": not missing,
            "limitations": ["policy-selected nonrandom coverage", "post hoc", "service verification bias",
                            "reused OOF training scores are not fully nested stacking validation",
                            "updated probabilities are not independently calibrated"],
            "input_sha256": {str(path): sha256(path) for path in (source, original_path,
                original_path.with_suffix('.json'), reference_path, ROOT / 'gzcmd/gzcmd_v4_llm_prompt.md')},
            "code_sha256": {name: sha256(ROOT / name) for name in (
                'scripts/llm_evidencia.py', 'scripts/comparador_mesma_carga.py',
                'scripts/fusao_filas.py', 'gzcmd/ml_oof.py', 'gzcmd/arranjo_corte.py')}}
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "llm_evidencia.csv", index=False, lineterminator="\n")
    rv_frame.to_csv(out / "llm_evidencia_rv.csv", index=False, lineterminator="\n")
    meta["output_sha256"] = {name: sha256(out / name) for name in ('llm_evidencia.csv', 'llm_evidencia_rv.csv')}
    (out / "llm_evidencia_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    write_report(summary, meta, report)
    print("COVERAGE", json.dumps(coverage), flush=True)
    print("MISSING_SOURCES", len(missing), flush=True)
    print(summary[summary.tipo.ne('cobertura')].pivot(index=['modo', 'variante'], columns='metrica', values='valor').to_string(), flush=True)
    return summary, rv_frame, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/plans" / REPORT)
    args = parser.parse_args()
    generate(args.data_root, args.out, args.report)
