"""Análise 4: viés de incorporação da referência operacional sobre o NB01 (Estudo 2).

Análise quantitativa de viés (QBA), exploratória e post hoc, só CPU, sem LLM.
Mesmo enquadramento de scripts/vies_referencia.py (Estudo 4): cenários de
perturbação da referência com predições FIXAS; os cenários NÃO estimam a
frequência real de erro da referência nem a corrigem como se fosse verdade
conhecida. Respondem a: quanto, e em que direção, a referência precisaria
errar para mudar as conclusões da comparação de classificadores do Estudo 2.

Mecanismo modelado. A revisão que produziu PAR via a nota final do
CdR e seus subescores; o viés de incorporação tem direção previsível:
  (a) pares de nota final ALTA (>= 9) tendem a ser confirmados como par mesmo
      quando não são pares verdadeiros; no cenário, uma fração `a` dos pares
      verdadeiros do teste com nota >= 9 é reclassificada como não par;
  (b) não pares com nota na faixa intermediária [5, 9) são, no cenário,
      reclassificados como pares verdadeiros, em número `b` expresso como
      fração dos pares verdadeiros do teste (74).
Nos dois cenários a referência acompanha a nota mais do que a verdade
acompanharia.

Desenho.
- Reexecuta o NB01 por scripts/nb01_bootstrap_ic.run_nb01 (importado, sem cópia)
  e confere contra data/nb01_results.json (tolerância 1e-4). Modelo que não
  reproduz fora de DIVERGENCIA_AMBIENTE (regressão logística e Stacking)
  aborta sem gravar nada. Decisão da autora (2026-11-09): pontos publicados no
  texto; predições da REEXECUÇÃO (a coluna ponto_publicado traz os publicados).
- Positivo = PAR em {1, 2} (rótulo do NB01); "PAR=2" no pedido = esse conjunto.
- Estratos no teste: A = pares verdadeiros com nota >= 9; B = não pares com
  5 <= nota < 9. Linhas sem nota ficam fora dos dois.
- Seleção aleatória sem reposição dentro do estrato, que NÃO usa predição de
  nenhum modelo (cell_selections não recebe predições). Variante de
  sensibilidade: b escolhido entre os não pares de B com as notas mais altas
  (empates sorteados), cenário mais adverso para quem aprendeu os subescores.
- Número de casos trocados: k_a = arredonda(a * |A|), k_b = arredonda(b * 74)
  (metades para cima); a semente de cada célula depende de (seed, variante,
  k_a, k_b), então acrescentar células à grade não muda as demais.
- Por réplica, rótulos perturbados, predições e escores fixos, VPP,
  sensibilidade, F1 (limiar 0,5, como no NB01) e AUC-PR dos 7 modelos.
  Mediana e P2,5 a P97,5 entre réplicas são dispersão do sorteio dentro do
  cenário, não intervalo de confiança inferencial.
- Conclusões do texto avaliadas por réplica: RF com maior F1 pontual; RF+SMOTE
  com maior AUC-PR; grupo de cima (RF, RF+SMOTE, Gradient Boosting, MLP) acima
  do de baixo (SVM, regressão logística, Stacking) em F1 (menor F1 do grupo de
  cima > maior F1 do de baixo). Ponto de virada = menor perturbação (menor
  k_a + k_b) com proporção de réplicas em que a conclusão vale < 0,5 (critério
  "mediana") ou < 1 (critério "qualquer_replica"); também por eixo.
- Saída só agregada: sem linhas do extrato, sem pair_id, sem REFREC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.nb01_bootstrap_ic import (  # noqa: E402
    DECISAO, DIVERGENCIA_AMBIENTE, EXTRATO, INPUT, JSON_KEYS, MODELS, TOLERANCE,
    ReproducaoFalhou, atomic_write, classify_reproduction, compare_published,
    environment, prepare, run_nb01, sha256, weighted_metrics,
)

REPLICAS = 200
SEED = 20261109
GRADE_A = (0.0, 0.05, 0.10, 0.20, 0.30)
GRADE_B = (0.0, 0.05, 0.10, 0.20, 0.30)
NOTA_ALTA = 9.0
NOTA_INTERMEDIARIA = (5.0, 9.0)  # [5, 9)
VARIANTS = ("aleatoria", "nota_mais_alta")
METRICS = ("VPP", "Sensibilidade", "F1", "AUC-PR")
RANK_METRICS = ("F1", "AUC-PR")
TOP_GROUP = ("Random Forest", "RF + SMOTE", "Gradient Boosting", "MLP Neural Network")
BOTTOM_GROUP = ("SVM (RBF)", "Logistic Regression", "Stacking Ensemble")
CONCLUSIONS = ("RF_maior_F1", "RFSMOTE_maior_AUCPR", "grupo_cima_acima_baixo_F1")
CRITERIA = {"mediana": 0.5, "qualquer_replica": 1.0}
OUT_CSV = "data/vies_incorporacao_nb01.csv"
OUT_META = "data/vies_incorporacao_nb01_metadata.json"
COLUMNS = (
    "tipo", "variante", "a", "b", "k_a", "k_b", "n_estrato_a", "n_estrato_b", "status",
    "classificador", "metrica", "conclusao", "mediana", "p2_5", "p97_5",
    "ponto_sem_perturbacao_rerun", "ponto_publicado", "proporcao_replicas", "texto",
    "empates", "replicas", "seed",
)
LIMITACOES = (
    "Os cenários perturbam a referência com predições fixas: não estimam a frequência real "
    "de erro da referência nem a corrigem como se fosse verdade conhecida.",
    "Mediana e P2,5 a P97,5 entre réplicas são dispersão do sorteio dentro do cenário, não "
    "intervalo de confiança inferencial.",
    "A partição do NB01 é por linha (train_test_split estratificado, seed 42); óbitos podem "
    "ter linhas no treino e no teste. Não há reajuste dos classificadores sob os rótulos "
    "perturbados: o treino segue com a referência original.",
    "Regressão logística e Stacking usam as predições da reexecução (divergência de ambiente "
    "registrada); os pontos publicados no texto vão na coluna ponto_publicado.",
    "Só dois mecanismos de erro são modelados (pares de nota >= 9 tomados como não pares; "
    "não pares de nota em [5, 9) tomados como pares), com seleção aleatória ou pelas maiores "
    "notas da faixa; outras formas de erro da referência não são cobertas.",
    "A grade é discreta; não se interpola entre pontos. O ponto de virada é o primeiro ponto "
    "da grade com a conclusão alterada.",
)


# --------------------------------------------------------------------------
# Estratos, seleção e rótulos perturbados (nada aqui lê predições)
# --------------------------------------------------------------------------

def strata_from(y, nota) -> dict:
    """Índices (posição no teste) dos estratos A e B e contagens de apoio."""
    y = np.asarray(y).astype(bool)
    nota = np.asarray(nota, dtype=float)
    if len(y) != len(nota):
        raise ValueError("y e nota com tamanhos diferentes")
    known = ~np.isnan(nota)
    with np.errstate(invalid="ignore"):
        in_a = y & known & (nota >= NOTA_ALTA)
        in_b = (~y) & known & (nota >= NOTA_INTERMEDIARIA[0]) & (nota < NOTA_INTERMEDIARIA[1])
    idx_a, idx_b = np.flatnonzero(in_a), np.flatnonzero(in_b)
    return {"idx_a": idx_a, "idx_b": idx_b, "nota_b": nota[idx_b],
            "n_pos": int(y.sum()), "n_total": int(len(y)),
            "n_sem_nota": int((~known).sum()),
            "n_pos_sem_nota": int((y & ~known).sum())}


def n_to_flip(fraction: float, base: int) -> int:
    """Arredondamento do número de casos trocados (metades para cima)."""
    return int(np.floor(fraction * base + 0.5 + 1e-9))


def cell_selections(strata: dict, k_a: int, k_b: int, variant: str, replicas: int, seed: int):
    """Gera, por réplica, (posições de A reclassificadas como não par, posições de B como par).

    Depende só dos estratos (rótulo e nota), da variante, de k_a, k_b e da
    semente: nenhuma predição entra aqui, para não favorecer modelo algum.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variante desconhecida: {variant}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, VARIANTS.index(variant), k_a, k_b]))
    idx_a, idx_b, nota_b = strata["idx_a"], strata["idx_b"], strata["nota_b"]
    for _ in range(replicas):
        sel_a = rng.choice(idx_a, size=k_a, replace=False) if k_a else idx_a[:0]
        if not k_b:
            sel_b = idx_b[:0]
        elif variant == "aleatoria":
            sel_b = rng.choice(idx_b, size=k_b, replace=False)
        else:  # as maiores notas da faixa; empates sorteados
            perm = rng.permutation(len(idx_b))
            order = perm[np.argsort(-nota_b[perm], kind="stable")]
            sel_b = idx_b[order[:k_b]]
        yield sel_a, sel_b


def perturbed_labels(y, sel_a, sel_b) -> np.ndarray:
    """Rótulos perturbados: sel_a vira não par, sel_b vira par verdadeiro; o resto fica igual."""
    y_new = np.asarray(y).astype(bool).copy()
    y_new[sel_a] = False
    y_new[sel_b] = True
    return y_new


# --------------------------------------------------------------------------
# Métricas com predições fixas
# --------------------------------------------------------------------------

def with_labels(prep: dict, y_new) -> dict:
    """prepare() com outros rótulos; ordenação por escore e predições inalteradas."""
    return {**prep, "y": y_new, "y_sorted": y_new[prep["order"]].astype(float)}


def metrics_fixed(prep: dict, y_new, ones) -> dict:
    return weighted_metrics(with_labels(prep, y_new), ones)


def ranks(values: np.ndarray) -> np.ndarray:
    """Posição por réplica (1 = maior); empate compartilha a melhor posição."""
    return 1 + (values[:, None, :] > values[:, :, None]).sum(axis=2)


def conclusion_flags(metric_arrays: dict, names) -> dict:
    """Booleano por réplica para cada conclusão do texto."""
    names = list(names)
    f1 = np.column_stack([metric_arrays[n]["F1"] for n in names])
    ap = np.column_stack([metric_arrays[n]["AUC-PR"] for n in names])

    def strictly_best(matrix, name):
        col = names.index(name)
        others = np.delete(matrix, col, axis=1)
        return (matrix[:, [col]] > others).all(axis=1)

    top = f1[:, [names.index(n) for n in TOP_GROUP]]
    bottom = f1[:, [names.index(n) for n in BOTTOM_GROUP]]
    return {"RF_maior_F1": strictly_best(f1, "Random Forest"),
            "RFSMOTE_maior_AUCPR": strictly_best(ap, "RF + SMOTE"),
            "grupo_cima_acima_baixo_F1": top.min(axis=1) > bottom.max(axis=1)}


def quantiles(values):
    lo, hi = np.quantile(values, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def run_cell(preps: dict, y, strata: dict, k_a: int, k_b: int, variant: str,
             replicas: int, seed: int):
    """Métricas dos modelos em todas as réplicas de uma célula (arrays por modelo e métrica)."""
    y = np.asarray(y).astype(bool)
    ones = np.ones(len(y))
    out = {n: {m: np.empty(replicas) for m in METRICS} for n in preps}
    for r, (sel_a, sel_b) in enumerate(cell_selections(strata, k_a, k_b, variant, replicas, seed)):
        y_new = perturbed_labels(y, sel_a, sel_b)
        for name, prep in preps.items():
            for metric, value in metrics_fixed(prep, y_new, ones).items():
                out[name][metric][r] = value
    return out


# --------------------------------------------------------------------------
# Grade completa e tabela agregada
# --------------------------------------------------------------------------

def _row(**kw) -> dict:
    row = dict.fromkeys(COLUMNS)
    row.update(kw)
    return row


def run_grid(preps: dict, y, nota, grade_a=GRADE_A, grade_b=GRADE_B, variants=VARIANTS,
             replicas=REPLICAS, seed=SEED, published=None, progress=None) -> pd.DataFrame:
    """Tabela longa agregada de todas as células; sem dados por linha do extrato."""
    published = published or {}
    names = list(preps)
    strata = strata_from(y, nota)
    n_a, n_b, n_pos = len(strata["idx_a"]), len(strata["idx_b"]), strata["n_pos"]
    y_arr = np.asarray(y).astype(bool)
    ones = np.ones(len(y_arr))
    point = {n: metrics_fixed(p, y_arr, ones) for n, p in preps.items()}
    rows, flags_by_cell = [], {}
    for variant in variants:
        for a in grade_a:
            for b in grade_b:
                k_a, k_b = n_to_flip(a, n_a), n_to_flip(b, n_pos)
                base = dict(variante=variant, a=a, b=b, k_a=k_a, k_b=k_b,
                            n_estrato_a=n_a, n_estrato_b=n_b, replicas=replicas, seed=seed)
                if k_a > n_a or k_b > n_b:
                    rows.append(_row(tipo="celula", status="ignorada_estrato_insuficiente", **base))
                    continue
                if n_pos - k_a + k_b <= 0:
                    rows.append(_row(tipo="celula", status="ignorada_sem_pares_verdadeiros", **base))
                    continue
                if progress:
                    progress(variant, a, b)
                arrays = run_cell(preps, y_arr, strata, k_a, k_b, variant, replicas, seed)
                rows.append(_row(tipo="celula", status="calculada", **base))
                for name in names:
                    for metric in METRICS:
                        v = arrays[name][metric]
                        lo, hi = quantiles(v)
                        rows.append(_row(
                            tipo="metrica", status="calculada", classificador=name, metrica=metric,
                            mediana=float(np.median(v)), p2_5=lo, p97_5=hi,
                            ponto_sem_perturbacao_rerun=point[name][metric],
                            ponto_publicado=published.get(name, {}).get(metric), **base))
                for metric in RANK_METRICS:
                    matrix = np.column_stack([arrays[n][metric] for n in names])
                    rk = ranks(matrix)
                    for j, name in enumerate(names):
                        lo, hi = quantiles(rk[:, j])
                        rows.append(_row(tipo="posicao", status="calculada", classificador=name,
                                         metrica=metric, mediana=float(np.median(rk[:, j])),
                                         p2_5=lo, p97_5=hi, **base))
                    medians = np.median(matrix, axis=0)
                    order = sorted(range(len(names)), key=lambda j: (-medians[j], names[j]))
                    rows.append(_row(tipo="ordem", status="calculada", metrica=metric,
                                     texto=" > ".join(names[j] for j in order), **base))
                flags = conclusion_flags(arrays, names)
                flags_by_cell[(variant, a, b)] = (k_a, k_b, {c: float(f.mean()) for c, f in flags.items()})
                for c, f in flags.items():
                    rows.append(_row(tipo="conclusao", status="calculada", conclusao=c,
                                     proporcao_replicas=float(f.mean()), **base))
    rows += turning_points(flags_by_cell, variants, n_a, n_b, replicas, seed)
    return pd.DataFrame(rows, columns=list(COLUMNS))


def turning_points(flags_by_cell: dict, variants, n_a: int, n_b: int, replicas: int, seed: int):
    """Menor perturbação com a conclusão invertida, por eixo e conjunta, por critério."""
    rows = []
    for variant in variants:
        cells = [(a, b, ka, kb, props) for (v, a, b), (ka, kb, props) in flags_by_cell.items()
                 if v == variant]
        for conclusion in CONCLUSIONS:
            baseline = [p[conclusion] for a, b, ka, kb, p in cells if ka == 0 and kb == 0]
            for criterion, threshold in CRITERIA.items():
                base = dict(tipo="virada", variante=variant, conclusao=conclusion,
                            n_estrato_a=n_a, n_estrato_b=n_b, replicas=replicas, seed=seed)
                if not baseline or baseline[0] < threshold:
                    rows.append(_row(status="nao_vale_no_baseline", texto=f"todos|{criterion}", **base))
                    continue
                for axis in ("a", "b", "conjunto"):
                    if axis == "a":
                        pool = sorted((c for c in cells if c[3] == 0 and c[2] > 0), key=lambda c: c[2])
                    elif axis == "b":
                        pool = sorted((c for c in cells if c[2] == 0 and c[3] > 0), key=lambda c: c[3])
                    else:
                        pool = sorted((c for c in cells if c[2] + c[3] > 0), key=lambda c: (c[2] + c[3], c[0], c[1]))
                    hit = [c for c in pool if c[4][conclusion] < threshold]
                    if not hit:
                        rows.append(_row(status="sem_virada_na_grade", texto=f"{axis}|{criterion}", **base))
                        continue
                    first = hit[0]
                    tied = [c for c in hit if c[2] + c[3] == first[2] + first[3]] if axis == "conjunto" else [first]
                    rows.append(_row(
                        status="virada_na_grade", texto=f"{axis}|{criterion}",
                        a=first[0], b=first[1], k_a=first[2], k_b=first[3],
                        proporcao_replicas=first[4][conclusion],
                        empates=";".join(f"a={c[0]:g},b={c[1]:g}" for c in tied) if axis == "conjunto" else None,
                        **base))
    return rows


# --------------------------------------------------------------------------
# Reprodução do NB01 (aborta fora de DIVERGENCIA_AMBIENTE)
# --------------------------------------------------------------------------

def check_reproduction(captured: dict, reference: dict, divergence=DIVERGENCIA_AMBIENTE):
    """Confere o rerun com o publicado; levanta ReproducaoFalhou fora das divergências aceitas."""
    records, structural = compare_published(captured, reference)
    status, fatal = classify_reproduction(records, divergence)
    if structural or fatal:
        lines = [f"{k}: obtido={a} esperado={b}" for k, a, b in structural]
        lines += [f"{r['classificador']} {r['metrica']}: obtido={r['rerun']} "
                  f"esperado={r['publicado']}" for r in fatal]
        raise ReproducaoFalhou("\n".join(lines))
    return status, records


def published_points(reference: dict) -> dict:
    ref = {r["Modelo"]: r for r in reference["all_results"]}
    return {n: {m: float(ref[n][JSON_KEYS[m]]) for m in METRICS} for n in MODELS if n in ref}


# --------------------------------------------------------------------------
# E/S e estimativa de tempo
# --------------------------------------------------------------------------

def estimate_time(replicas: int, grade_a, grade_b, variants, rows=18509, positives=74):
    rng = np.random.default_rng(0)
    y = np.zeros(rows, dtype=bool)
    y[rng.choice(rows, positives, replace=False)] = True
    nota = np.where(y, rng.uniform(5, 15, rows), rng.uniform(0, 10, rows))
    preps = {}
    for name in MODELS:
        score = np.where(y, rng.beta(5, 1.5, rows), rng.beta(1, 8, rows))
        preps[name] = prepare(y, score >= 0.5, score)
    strata = strata_from(y, nota)
    k_a, k_b = min(5, len(strata["idx_a"])), min(5, len(strata["idx_b"]))
    probe = 40
    start = time.perf_counter()
    run_cell(preps, y, strata, k_a, k_b, VARIANTS[0], probe, SEED)
    per = (time.perf_counter() - start) / probe
    cells = len(grade_a) * len(grade_b) * len(variants)
    print(f"Cenários: {per * 1000:.1f} ms por réplica (7 classificadores, dados sintéticos); "
          f"{cells} células x {replicas} réplicas: cerca de {per * replicas * cells / 60:.1f} min "
          "(limite superior: células ignoradas não custam).")
    print("Reexecução do NB01 (não medida aqui): dominada por SVM com probability=True e "
          "Stacking com cv=5; esperar de ~10 a ~40 min conforme a CPU.")


def parse_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(x) for x in text.split(",") if x.strip())
    if not values or any(v < 0 or v > 1 for v in values):
        raise argparse.ArgumentTypeError("frações em [0, 1], separadas por vírgula")
    return values


def generate(args) -> int:
    started = time.perf_counter()
    reference = json.loads(INPUT.read_text(encoding="utf-8"))
    captured = run_nb01(args.extrato)
    try:
        status, records = check_reproduction(captured, reference)
    except ReproducaoFalhou as exc:
        print(f"NB01 NÃO reproduzido (tolerância {TOLERANCE}) fora das divergências de "
              "ambiente aceitas; nada foi gravado:", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1
    diverging = [n for n, s in status.items() if not s["reproduz"]]
    print(f"Reproduzidos dentro de {TOLERANCE}: {', '.join(n for n in status if n not in diverging)}."
          f" Divergência de ambiente aceita: {', '.join(diverging) or 'nenhuma'}.", flush=True)

    X_test, y_test = captured["split"][1], captured["split"][3]
    y = np.asarray(y_test).astype(bool)
    nota = X_test["nota final"].to_numpy(dtype=float)
    if any(not np.array_equal(np.asarray(captured["preds"][n][0]).astype(bool), y) for n in MODELS):
        print("Rótulos capturados diferem do y do teste; nada foi gravado.", file=sys.stderr)
        return 1
    preps = {n: prepare(*captured["preds"][n]) for n in MODELS}
    strata = strata_from(y, nota)
    print(f"Teste: {strata['n_total']} linhas, {strata['n_pos']} pares verdadeiros; "
          f"estrato A (par verdadeiro, nota >= 9): {len(strata['idx_a'])}; "
          f"estrato B (não par, nota em [5, 9)): {len(strata['idx_b'])}; "
          f"sem nota: {strata['n_sem_nota']}.", flush=True)

    def progress(variant, a, b):
        print(f"  célula {variant} a={a:g} b={b:g}", flush=True)

    table = run_grid(preps, y, nota, args.grade_a, args.grade_b, args.variantes, args.replicas,
                     args.seed, published_points(reference), progress)
    csv_text = table.to_csv(index=False, lineterminator="\n")
    saida = args.saida
    out = saida / Path(OUT_CSV).name
    input_paths = {"data/COMPARADORSEMIDENT.csv": args.extrato, "data/nb01_results.json": INPUT,
                   "scripts/run_nb01_uniform.py": ROOT / "scripts/run_nb01_uniform.py",
                   "scripts/feature_engineering.py": ROOT / "scripts/feature_engineering.py"}
    skipped = table[(table.tipo == "celula") & (table.status != "calculada")]
    metadata = {
        "analysis": "vies_incorporacao_nb01", "cpu_only": True, "llm_calls": 0,
        "autorizacao": "autora, 2026-11-09 (análise 4)",
        "date_utc": datetime.now(UTC).isoformat(), "seed": args.seed, "replicas": args.replicas,
        "rng": "numpy.default_rng(SeedSequence([seed, indice_variante, k_a, k_b])) por célula",
        "grade_a": list(args.grade_a), "grade_b": list(args.grade_b),
        "variantes": list(args.variantes),
        "estratos": {
            "linhas_teste": strata["n_total"], "pares_verdadeiros_teste": strata["n_pos"],
            "A_pares_verdadeiros_nota_ge_9": int(len(strata["idx_a"])),
            "B_nao_pares_nota_5_a_9": int(len(strata["idx_b"])),
            "linhas_sem_nota": strata["n_sem_nota"],
            "pares_verdadeiros_sem_nota": strata["n_pos_sem_nota"],
            "faixas": "A: nota >= 9; B: 5 <= nota < 9 (faixa intermediária da nota final)",
        },
        "regra_de_selecao": "aleatória sem reposição dentro do estrato, sem usar predição de "
                            "modelo algum; variante nota_mais_alta: b escolhido entre as maiores "
                            "notas de B, empates sorteados",
        "arredondamento": "k_a = arredonda(a * |A|) e k_b = arredonda(b * pares verdadeiros do "
                          "teste), metades para cima",
        "celulas_ignoradas": [
            {"variante": r.variante, "a": r.a, "b": r.b, "k_a": int(r.k_a), "k_b": int(r.k_b),
             "motivo": r.status} for r in skipped.itertuples()],
        "classificadores": list(MODELS), "metricas": list(METRICS), "limiar": 0.5,
        "grupo_cima": list(TOP_GROUP), "grupo_baixo": list(BOTTOM_GROUP),
        "conclusoes": {
            "RF_maior_F1": "Random Forest com maior F1 (estritamente)",
            "RFSMOTE_maior_AUCPR": "RF + SMOTE com maior AUC-PR (estritamente)",
            "grupo_cima_acima_baixo_F1": "menor F1 do grupo de cima > maior F1 do grupo de baixo"},
        "ponto_de_virada": "primeira célula da grade, em ordem de k_a + k_b, com proporção de "
                           "réplicas em que a conclusão vale < 0,5 (mediana) ou < 1 (qualquer_replica)",
        "decisao": DECISAO + "; predições da reexecução, pontos publicados na coluna ponto_publicado",
        "reproducao": {"tolerancia": TOLERANCE, "status": status, "registros": records,
                       "divergencia_ambiente": DIVERGENCIA_AMBIENTE,
                       "nao_reproduzidos": diverging},
        "limitacoes": list(LIMITACOES),
        **environment(),
        "input_sha256": {n: sha256(p) for n, p in input_paths.items()},
        "code_sha256": {n: sha256(ROOT / n) for n in
                        ("scripts/vies_incorporacao_nb01.py", "tests/test_vies_incorporacao_nb01.py")
                        if (ROOT / n).exists()},
        "output_sha256": {out.name: hashlib.sha256(csv_text.encode("utf-8")).hexdigest()},
        "elapsed_seconds": time.perf_counter() - started,
        "command": " ".join([sys.executable, *sys.argv]),
    }
    meta_text = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    atomic_write(out, csv_text)
    atomic_write(saida / Path(OUT_META).name, meta_text)
    summary(table)
    print(f"Concluído em {metadata['elapsed_seconds']:.1f}s", flush=True)
    return 0


def summary(table: pd.DataFrame) -> None:
    turns = table[table.tipo == "virada"]
    cols = ["variante", "conclusao", "texto", "status", "a", "b", "k_a", "k_b", "proporcao_replicas"]
    print(turns[cols].to_string(index=False), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=REPLICAS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--grade-a", type=parse_grid, default=GRADE_A,
                        help="frações dos pares verdadeiros com nota >= 9 reclassificados como não par")
    parser.add_argument("--grade-b", type=parse_grid, default=GRADE_B,
                        help="não pares de nota [5, 9) tornados par, como fração dos pares verdadeiros")
    parser.add_argument("--variantes", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--saida", type=Path, default=ROOT / "data")
    parser.add_argument("--extrato", type=Path, default=EXTRATO)
    parser.add_argument("--estimar-tempo", action="store_true",
                        help="só estima o tempo dos cenários e sai")
    args = parser.parse_args(argv)
    if args.estimar_tempo:
        estimate_time(args.replicas, args.grade_a, args.grade_b, args.variantes)
        return 0
    args.saida.mkdir(parents=True, exist_ok=True)
    return generate(args)


if __name__ == "__main__":
    sys.exit(main())
