"""Incerteza da comparação inicial de classificadores (NB01, seção 6.1.1).

Reproduz o NB01 chamando scripts/run_nb01_uniform.py sem copiá-lo (mesmos sete
classificadores, 58 atributos, partição estratificada por linha com seed 42,
limiar 0,5) e confere VPP, sensibilidade, F1, AUC-ROC e AUC-PR contra
data/nb01_results.json (tolerância 1e-4). Modelo que não reproduz faz o script
sair com código 1 sem gravar nada, salvo os listados em DIVERGENCIA_AMBIENTE
(regressão logística e Stacking): para esses, publicado e rerun vão lado a lado,
com reproduz=False e o motivo. Decisão da autora (2026-11-09): pontos
publicados no texto; IC do rerun no ambiente atual.

Estima IC 95% por bootstrap agrupado por óbito (REFREC) dentro do conjunto de
teste, sobre as predições do rerun, para cada classificador e para a diferença
pareada entre pares de classificadores (mesma réplica para os dois). Para a
regressão logística, registra a faixa entre otimizadores (lbfgs, liblinear,
newton-cholesky): mesmo X e mesmos hiperparâmetros, só o solver muda.

A partição do NB01 é por linha; o bootstrap é por óbito dentro do teste. As
predições do teste ficam só em memória; os artefatos têm apenas agregados.
Os artefatos de run_nb01_uniform.py não são tocados: a execução do NB01 grava
em diretório temporário.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

REPLICAS = 2000
SEED = 20261109
TOLERANCE = 1e-4
MODELS = (
    "Logistic Regression", "Random Forest", "Gradient Boosting", "SVM (RBF)",
    "MLP Neural Network", "RF + SMOTE", "Stacking Ensemble",
)
POINT_METRICS = ("VPP", "Sensibilidade", "F1", "AUC-PR")
DIFF_METRICS = ("F1", "AUC-PR")
JSON_KEYS = {"VPP": "Precision", "Sensibilidade": "Recall", "F1": "F1-Score",
             "AUC-ROC": "AUC-ROC", "AUC-PR": "AUC-PR"}
COLUMNS = (
    "tipo", "classificador_ou_par", "metrica", "ponto_rerun", "ic95_inf",
    "ic95_sup", "ponto_publicado", "diferenca_rerun_menos_publicado",
    "reproduz", "motivo", "proporcao_replicas_diferenca_maior_0",
    "replicas", "replicas_validas", "seed",
)
LR_NAME = "Logistic Regression"
LR_SOLVERS = ("lbfgs", "liblinear", "newton-cholesky")
DIAGNOSTICO = ".local/diag_nb01.json e .local/diag_nb01.log (locais, não versionados; 2026-11-09)"
# Modelos que podem não reproduzir sem abortar; valor = justificativa (motivo).
# Qualquer outro modelo que não reproduza continua abortando.
DIVERGENCIA_AMBIENTE = {
    "Logistic Regression": (
        "divergência de ambiente: regressão logística sobre atributos não padronizados, "
        "mal condicionada; lbfgs determinístico e convergido (n_iter < max_iter), mas o "
        "ponto de parada depende do ambiente (sklearn/scipy/BLAS); X idêntico bit a bit nas "
        "duas cargas e padrões antigos do sklearn sem efeito (diagnóstico .local/diag_nb01.json, "
        "2026-11-09)"),
    "Stacking Ensemble": (
        "divergência de ambiente: contém a mesma regressão logística sem padronização como "
        "estimador de base (lbfgs); causa provável idêntica, não isolada no diagnóstico "
        "(.local/diag_nb01.json, 2026-11-09)"),
}
DECISAO = "pontos publicados no texto; IC do rerun no ambiente atual (decisão da autora, 2026-11-09)"
OUT_CSV = "data/nb01_bootstrap_ic.csv"
OUT_META = "data/nb01_bootstrap_ic_metadata.json"
INPUT = ROOT / "data" / "nb01_results.json"
EXTRATO = ROOT / "data" / "COMPARADORSEMIDENT.csv"


# --------------------------------------------------------------------------
# Métricas ponderadas (peso = multiplicidade do óbito na réplica)
# --------------------------------------------------------------------------

def prepare(y_true, y_pred, score) -> dict:
    """Pré-computa a ordenação por escore, comum a todas as réplicas."""
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_pred).astype(bool)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="stable")
    ss = score[order]
    # último índice de cada grupo de escores iguais (limiares distintos)
    ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    return {"y": y, "pred": pred, "scores": score, "order": order, "ends": ends,
            "y_sorted": y[order].astype(float)}


def weighted_metrics(prep: dict, w) -> dict:
    """VPP, sensibilidade, F1 e AUC-PR com pesos inteiros por linha.

    Pesos = multiplicidades equivalem a duplicar as linhas (escores iguais ficam
    no mesmo limiar). Convenção zero_division=0 do NB01 para VPP, sensibilidade
    e F1. Exige pelo menos um par verdadeiro com peso (o chamador exclui o resto).
    """
    w = np.asarray(w, dtype=float)
    y, pred = prep["y"], prep["pred"]
    tp = float(w[y & pred].sum())
    fp = float(w[~y & pred].sum())
    fn = float(w[y & ~pred].sum())
    vpp = tp / (tp + fp) if tp + fp > 0 else 0.0
    sens = tp / (tp + fn) if tp + fn > 0 else 0.0
    den = 2 * tp + fp + fn
    f1 = 2 * tp / den if den > 0 else 0.0
    ws = w[prep["order"]]
    tpc = np.cumsum(ws * prep["y_sorted"])[prep["ends"]]
    cnt = np.cumsum(ws)[prep["ends"]]
    total = tpc[-1]
    valid = cnt > 0
    prec = np.where(valid, tpc / np.where(valid, cnt, 1.0), 0.0)
    rec = tpc / total
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return {"VPP": vpp, "Sensibilidade": sens, "F1": f1, "AUC-PR": ap}


def point_estimates(prep: dict) -> dict:
    """Estimativas pontuais na amostra de teste original (pesos 1)."""
    return weighted_metrics(prep, np.ones(len(prep["y"])))


def group_weights(rng, codes, group_count) -> np.ndarray:
    """Sorteia óbitos com reposição; peso da linha = multiplicidade do seu óbito."""
    draws = rng.integers(0, group_count, size=group_count)
    return np.bincount(draws, minlength=group_count)[codes]


def bootstrap(preps: dict, group_codes, group_count, replicas=REPLICAS, seed=SEED):
    """Bootstrap agrupado: sorteia óbitos com reposição, todas as linhas de cada um.

    Retorna (reps, n_validas): reps[nome][métrica] = array das réplicas válidas.
    Réplica sem par verdadeiro (rara; 74 positivos) é descartada para todos os
    classificadores, mantendo o pareamento. Cada réplica consome uma chamada
    rng.integers.
    """
    codes = np.asarray(group_codes)
    y = next(iter(preps.values()))["y"]
    rng = np.random.default_rng(seed)
    out = {n: {m: [] for m in POINT_METRICS} for n in preps}
    for _ in range(replicas):
        w = group_weights(rng, codes, group_count)
        if not w[y].sum():
            continue
        for name, prep in preps.items():
            for metric, value in weighted_metrics(prep, w).items():
                out[name][metric].append(value)
    reps = {n: {m: np.array(v) for m, v in d.items()} for n, d in out.items()}
    valid = len(next(iter(reps.values()))["F1"])
    return reps, valid


def summarize(point: dict, reps: dict, replicas: int, valid: int, seed: int,
              published: dict | None = None, status: dict | None = None):
    """Tabela longa: IC por classificador e por diferença pareada (A menos B).

    ponto_rerun é a estimativa pontual do rerun (base do IC); ponto_publicado vem
    de data/nb01_results.json (4 casas). Numa diferença pareada, o publicado é a
    diferença dos publicados e reproduz exige que os dois modelos reproduzam.
    """
    published, status = published or {}, status or {}
    rows = []

    def quant(values):
        lo, hi = np.quantile(values, [0.025, 0.975], method="linear")
        return float(lo), float(hi)

    def pub(name, metric):
        return float(published.get(name, {}).get(metric, float("nan")))

    def ok(name):
        return bool(status.get(name, {}).get("reproduz", True))

    def why(name):
        return status.get(name, {}).get("motivo", "")

    names = list(reps)
    for name in names:
        for metric in POINT_METRICS:
            lo, hi = quant(reps[name][metric])
            rerun, p = point[name][metric], pub(name, metric)
            rows.append(("classificador", name, metric, rerun, lo, hi, p, rerun - p,
                         ok(name), why(name), float("nan"), replicas, valid, seed))
    for a, b in itertools.combinations(names, 2):
        for metric in DIFF_METRICS:
            diff = reps[a][metric] - reps[b][metric]
            lo, hi = quant(diff)
            rerun = point[a][metric] - point[b][metric]
            p = pub(a, metric) - pub(b, metric)
            diverging = [n for n in (a, b) if not ok(n)]
            motivo = f"não reproduz: {', '.join(diverging)} (ver linha do classificador)" \
                if diverging else ""
            rows.append(("diferenca_pareada", f"{a} - {b}", metric, rerun, lo, hi, p,
                         rerun - p, ok(a) and ok(b), motivo, float((diff > 0).mean()),
                         replicas, valid, seed))
    return pd.DataFrame(rows, columns=list(COLUMNS))


# --------------------------------------------------------------------------
# NB01 reproduzido via run_nb01_uniform (sem cópia) e conferência
# --------------------------------------------------------------------------

def load_extract(path: Path) -> pd.DataFrame:
    """Extrato lido com dtype=str e convertido com to_numeric (AGENTS.md §4).

    Substitui feature_engineering.load_data, que só converte colunas de dtype
    object: no pandas 3 o texto vira string Arrow e escapa da conversão, o que
    quebra engineer_features. Aqui toda coluna passa por to_numeric sobre array
    object do NumPy (vírgula decimal trocada por ponto, errors="coerce"), o que
    devolve int64/float64 do NumPy, como load_data devolvia quando o texto era
    object. Nomes de coluna limpos com o mesmo _clean_col.
    """
    from feature_engineering import _clean_col

    raw = pd.read_csv(path, sep=";", dtype=str, low_memory=False)
    columns = {}
    for i in range(raw.shape[1]):
        text = raw.iloc[:, i].str.replace(",", ".", regex=False)
        columns[i] = pd.to_numeric(np.asarray(text, dtype=object), errors="coerce")
    df = pd.DataFrame(columns, index=raw.index)
    df.columns = [_clean_col(c) for c in raw.columns]
    return df


def run_nb01(extrato: Path = EXTRATO) -> dict:
    """Executa run_nb01_uniform.main() capturando partição e predições em memória.

    Só a carga do extrato é trocada (load_extract); o resto é o código original.
    Também guarda em memória a partição (X e y) e a regressão logística ajustada
    pelo NB01, para a faixa entre otimizadores; o construtor é só embrulhado
    para registrar as instâncias, sem mudar parâmetros.
    """
    from scripts import run_nb01_uniform as nb

    captured = {"preds": {}, "test_index": None}
    originals = (nb.evaluate, nb.split_data, nb.load_data, nb.RESULTS_DIR,
                 nb.LogisticRegression)
    built = []

    def evaluate(name, y_true, y_pred, y_proba):
        captured["preds"][name] = (np.asarray(y_true), np.asarray(y_pred),
                                   np.asarray(y_proba))
        return originals[0](name, y_true, y_pred, y_proba)

    def split_data(X, y, *a, **k):
        parts = originals[1](X, y, *a, **k)
        captured["split"] = parts
        captured["test_index"] = parts[1].index.to_numpy()
        captured["n_test"] = len(parts[1])
        captured["pos_test"] = int(parts[3].sum())
        return parts

    def logistic_regression(*a, **k):
        estimator = originals[4](*a, **k)
        built.append(estimator)
        return estimator

    with tempfile.TemporaryDirectory() as tmp:
        nb.evaluate, nb.split_data = evaluate, split_data
        nb.load_data = lambda: load_extract(extrato)
        nb.RESULTS_DIR = Path(tmp)
        nb.LogisticRegression = logistic_regression
        try:
            nb.main()
        finally:
            (nb.evaluate, nb.split_data, nb.load_data, nb.RESULTS_DIR,
             nb.LogisticRegression) = originals
    # O Stacking clona seus estimadores; só a regressão logística isolada é ajustada.
    fitted = [e for e in built if hasattr(e, "coef_")]
    if len(fitted) != 1:
        raise RuntimeError(f"Esperada 1 regressão logística ajustada, achadas {len(fitted)}")
    captured["lr_estimator"] = fitted[0]
    return captured


def groups_of_test_rows(extrato: Path, test_index) -> np.ndarray:
    """REFREC (identificador do óbito) das linhas do teste; dtype=str."""
    from feature_engineering import _clean_col

    ref = pd.read_csv(extrato, sep=";", dtype=str,
                      usecols=lambda c: _clean_col(c) == "REFREC")
    return ref.iloc[:, 0].to_numpy()[test_index]


class ReproducaoFalhou(RuntimeError):
    """NB01 não reproduzido fora das divergências de ambiente aceitas."""


def rerun_metrics(y_true, y_pred, proba) -> dict:
    """VPP, sensibilidade, F1, AUC-ROC e AUC-PR sem arredondar (AUC-PR conferida no sklearn)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    point = point_estimates(prepare(y_true, y_pred, proba))
    sk_ap = float(average_precision_score(y_true, proba))
    if abs(sk_ap - point["AUC-PR"]) > 1e-9:
        raise AssertionError("AUC-PR próprio difere do sklearn")
    point["AUC-PR"] = sk_ap
    point["AUC-ROC"] = float(roc_auc_score(y_true, proba))
    return point


def compare_published(captured, reference: dict, tol=TOLERANCE):
    """Publicado e rerun lado a lado, por modelo e métrica.

    Retorna (registros, falhas_estruturais); falha estrutural (partição diferente
    ou modelo ausente) sempre aborta.
    """
    structural = [(k, captured[k], reference[k]) for k in ("n_test", "pos_test")
                  if captured[k] != reference[k]]
    ref = {r["Modelo"]: r for r in reference["all_results"]}
    records = []
    for name in MODELS:
        if name not in captured["preds"] or name not in ref:
            structural.append((name, "ausente", "ausente"))
            continue
        rerun = rerun_metrics(*captured["preds"][name])
        for metric, key in JSON_KEYS.items():
            pub = float(ref[name][key])
            records.append({"classificador": name, "metrica": metric, "publicado": pub,
                            "rerun": rerun[metric],
                            "diferenca_rerun_menos_publicado": rerun[metric] - pub,
                            "reproduz_metrica": bool(abs(rerun[metric] - pub) <= tol)})
    return records, structural


def classify_reproduction(records, divergence=DIVERGENCIA_AMBIENTE):
    """Status por modelo e falhas que abortam (modelo fora da lista que não reproduz)."""
    status, fatal = {}, []
    for name in dict.fromkeys(r["classificador"] for r in records):
        failing = [r for r in records if r["classificador"] == name and not r["reproduz_metrica"]]
        listed = name in divergence
        status[name] = {"reproduz": not failing, "na_lista_divergencia": listed,
                        "motivo": divergence[name] if failing and listed else ""}
        if failing and not listed:
            fatal.extend(failing)
    return status, fatal


def analyze(captured, reference: dict, groups_raw, replicas: int, seed: int,
            divergence=DIVERGENCIA_AMBIENTE) -> dict:
    """Conferência com o publicado, depois bootstrap sobre as predições do rerun.

    Levanta ReproducaoFalhou se a partição divergir, se faltar modelo ou se um
    modelo fora de `divergence` não reproduzir.
    """
    records, structural = compare_published(captured, reference)
    status, fatal = classify_reproduction(records, divergence)
    if structural or fatal:
        lines = [f"{k}: obtido={a} esperado={b}" for k, a, b in structural]
        lines += [f"{r['classificador']} {r['metrica']}: obtido={r['rerun']} "
                  f"esperado={r['publicado']}" for r in fatal]
        raise ReproducaoFalhou("\n".join(lines))
    preps = {n: prepare(*captured["preds"][n]) for n in MODELS}
    codes, uniques = pd.factorize(pd.Series(groups_raw), sort=True)
    if (codes < 0).any():
        raise ValueError("REFREC ausente em linhas do teste")
    point = {n: point_estimates(p) for n, p in preps.items()}
    reps, valid = bootstrap(preps, codes, len(uniques), replicas, seed)
    if not valid:
        raise ValueError("Nenhuma réplica com par verdadeiro")
    published = {}
    for r in records:
        published.setdefault(r["classificador"], {})[r["metrica"]] = r["publicado"]
    return {"table": summarize(point, reps, replicas, valid, seed, published, status),
            "records": records, "status": status, "valid": valid, "groups": len(uniques),
            "vpp_zero": {n: int((reps[n]["VPP"] == 0).sum()) for n in reps}}


def _json_safe(value):
    if isinstance(value, (str, bool, int, type(None))):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    return repr(value)


def optimizer_range(lr, X_train, y_train, X_test, y_test, rerun_proba=None,
                    solvers=LR_SOLVERS) -> dict:
    """Faixa entre otimizadores da regressão logística do NB01.

    clone(lr) mantém todos os hiperparâmetros do NB01; só `solver` muda. Mesmo
    X e mesma partição; limiar 0,5 via predict, como no NB01.
    """
    from sklearn.base import clone

    y = np.asarray(y_test).astype(bool)
    per_solver = {}
    for solver in solvers:
        model = clone(lr).set_params(solver=solver).fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = np.asarray(model.predict(X_test)).astype(bool)
        metrics = rerun_metrics(y, pred, proba)
        metrics.update(TP=int((y & pred).sum()), FP=int((~y & pred).sum()),
                       FN=int((y & ~pred).sum()), n_iter=int(np.max(model.n_iter_)))
        if rerun_proba is not None:
            metrics["max_abs_dif_proba_vs_rerun"] = float(
                np.max(np.abs(proba - np.asarray(rerun_proba))))
        per_solver[solver] = metrics
    faixa = {m: {"minimo": min(per_solver[s][m] for s in solvers),
                 "maximo": max(per_solver[s][m] for s in solvers)}
             for m in (*POINT_METRICS, "FP")}
    params = {k: _json_safe(v) for k, v in lr.get_params().items() if k != "solver"}
    return {"parametros_fixos_do_nb01": params, "solvers": per_solver, "faixa": faixa}


# --------------------------------------------------------------------------
# E/S
# --------------------------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str, retries=8):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * 2 ** attempt)


def estimate_time(replicas: int, rows=18509, positives=74, groups=17000):
    """Mede o bootstrap em dados sintéticos do mesmo tamanho e extrapola."""
    rng = np.random.default_rng(0)
    y = np.zeros(rows, dtype=bool)
    y[rng.choice(rows, positives, replace=False)] = True
    codes = rng.integers(0, groups, rows)
    preps = {}
    for name in MODELS:
        score = np.where(y, rng.beta(5, 1.5, rows), rng.beta(1, 8, rows))
        preps[name] = prepare(y, score >= 0.5, score)
    probe = 50
    start = time.perf_counter()
    bootstrap(preps, codes, groups, probe, SEED)
    per = (time.perf_counter() - start) / probe
    print(f"Bootstrap: {per * 1000:.1f} ms por réplica (7 classificadores, dados "
          f"sintéticos); B={replicas}: cerca de {per * replicas / 60:.1f} min.")
    print("Treino do NB01 (não medido aqui): dominado por SVM com probability=True "
          "e Stacking com cv=5; esperar de ~10 a ~40 min conforme a CPU.")


def environment() -> dict:
    """Versões e BLAS (sem caminhos de arquivo), para registrar o ambiente do rerun."""
    import imblearn
    import scipy
    import sklearn
    import threadpoolctl

    return {
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "imblearn": imblearn.__version__,
                     "threadpoolctl": threadpoolctl.__version__},
        "plataforma": platform.platform(),
        "threadpools": [{k: p.get(k) for k in ("user_api", "internal_api", "version", "num_threads")}
                        for p in threadpoolctl.threadpool_info()],
    }


def generate(replicas: int, seed: int, saida: Path, extrato: Path = EXTRATO):
    started = time.perf_counter()
    reference = json.loads(INPUT.read_text(encoding="utf-8"))
    captured = run_nb01(extrato)
    groups_raw = groups_of_test_rows(extrato, captured["test_index"])
    try:
        result = analyze(captured, reference, groups_raw, replicas, seed)
    except ReproducaoFalhou as exc:
        print(f"NB01 NÃO reproduzido (tolerância {TOLERANCE}) fora das divergências de "
              "ambiente aceitas; nada foi gravado:", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1
    status = result["status"]
    diverging = [n for n, s in status.items() if not s["reproduz"]]
    print(f"Reproduzidos dentro de {TOLERANCE}: "
          f"{', '.join(n for n in status if n not in diverging)}.", flush=True)
    for r in result["records"]:
        if r["classificador"] in diverging:
            print(f"  divergência aceita {r['classificador']} {r['metrica']}: "
                  f"publicado={r['publicado']} rerun={r['rerun']:.6f}", flush=True)

    X_train, X_test, y_train, y_test = captured["split"]
    lr_range = optimizer_range(captured["lr_estimator"], X_train, y_train, X_test, y_test,
                               rerun_proba=captured["preds"][LR_NAME][2])
    table, valid = result["table"], result["valid"]

    out = saida / Path(OUT_CSV).name
    csv_text = table.to_csv(index=False, lineterminator="\n")
    input_paths = {"data/COMPARADORSEMIDENT.csv": extrato,
                   "data/nb01_results.json": INPUT,
                   "scripts/run_nb01_uniform.py": ROOT / "scripts/run_nb01_uniform.py",
                   "scripts/feature_engineering.py": ROOT / "scripts/feature_engineering.py"}
    metadata = {
        "analysis": "nb01_bootstrap_ic", "cpu_only": True, "llm_calls": 0,
        "date_utc": datetime.now(UTC).isoformat(), "seed": seed,
        "replicas": replicas, "replicas_validas": valid,
        "replicas_sem_par_verdadeiro": replicas - valid,
        "rng": "numpy.default_rng PCG64; uma chamada rng.integers por réplica",
        "grupos_REFREC_no_teste": result["groups"],
        "linhas_teste": captured["n_test"], "pares_verdadeiros_teste": captured["pos_test"],
        "classificadores": list(MODELS),
        "metricas": list(POINT_METRICS), "diferencas_pareadas": list(DIFF_METRICS),
        "limiar": 0.5, "tolerancia_reproducao": TOLERANCE,
        "decisao": DECISAO,
        "base_do_ic": "predições do rerun no ambiente atual, para todos os modelos e pares "
                      "(coluna ponto_rerun); ponto_publicado vem de data/nb01_results.json",
        "divergencia_ambiente": {
            "modelos": DIVERGENCIA_AMBIENTE, "diagnostico": DIAGNOSTICO,
            "regra": "modelo da lista que não reproduz não aborta (publicado e rerun lado a lado, "
                     "reproduz=False, motivo); modelo fora da lista que não reproduz aborta",
            "nao_reproduzidos": diverging,
        },
        "reproducao": {"tolerancia": TOLERANCE, "status": status,
                       "registros": result["records"]},
        "lr_faixa_otimizadores": lr_range,
        "carga_extrato": "load_extract: dtype=str e to_numeric em todas as colunas (vírgula decimal), "
                         "no lugar de feature_engineering.load_data, que no pandas 3 deixa texto como string Arrow",
        "replicas_com_VPP_zero_por_classificador": result["vpp_zero"],
        "nota_particao": "A partição do NB01 é por linha (train_test_split estratificado, seed 42); "
                         "o bootstrap é por óbito (REFREC) dentro do teste. Óbitos podem ter linhas "
                         "no treino e no teste; o IC é condicional aos modelos já treinados "
                         "(não reajusta os classificadores).",
        "bootstrap": "REFREC reamostrado com reposição dentro do teste; multiplicidades viram pesos das linhas",
        "diferenca": "A menos B; mesma réplica para os dois classificadores",
        "zero_division": "VPP, sensibilidade e F1 = 0 quando o denominador é zero (como no NB01)",
        "replicas_sem_par_verdadeiro_regra": "descartadas para todos os classificadores",
        "confidence_interval": "percentil 2,5/97,5 (quantil linear)",
        **environment(),
        "input_sha256": {n: sha256(p) for n, p in input_paths.items()},
        "code_sha256": {n: sha256(ROOT / n) for n in
                        ("scripts/nb01_bootstrap_ic.py", "tests/test_nb01_bootstrap_ic.py")
                        if (ROOT / n).exists()},
        "output_sha256": {out.name: hashlib.sha256(csv_text.encode("utf-8")).hexdigest()},
        "elapsed_seconds": time.perf_counter() - started,
        "command": " ".join([sys.executable, *sys.argv]),
    }
    # Serializa antes de gravar: se o JSON falhar, nenhum artefato é escrito.
    meta_text = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    atomic_write(out, csv_text)
    atomic_write(saida / Path(OUT_META).name, meta_text)
    print(table.drop(columns=["motivo"]).to_string(index=False), flush=True)
    print("Faixa da regressão logística entre otimizadores "
          f"({', '.join(LR_SOLVERS)}):", flush=True)
    for metric, span in lr_range["faixa"].items():
        print(f"  {metric}: {span['minimo']:.4g} a {span['maximo']:.4g}", flush=True)
    print(f"Concluído em {metadata['elapsed_seconds']:.1f}s", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=REPLICAS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--saida", type=Path, default=ROOT / "data",
                        help="diretório de saída dos artefatos (padrão: data/)")
    parser.add_argument("--extrato", type=Path, default=EXTRATO)
    parser.add_argument("--estimar-tempo", action="store_true",
                        help="só estima o tempo do bootstrap e sai")
    args = parser.parse_args(argv)
    if args.estimar_tempo:
        estimate_time(args.replicas)
        return 0
    args.saida.mkdir(parents=True, exist_ok=True)
    return generate(args.replicas, args.seed, args.saida, args.extrato)


if __name__ == "__main__":
    sys.exit(main())
