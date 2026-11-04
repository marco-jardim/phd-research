"""Sensibilidade QW1 em GPU local; publica exclusivamente agregados."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import LOSS_FUNCTIONS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import validate_data

from gzcmd.splitting import SplitSpec, split_train_test_indices
from scripts.feature_engineering import engineer_features
from scripts.qw1_deep_learning_comparison import _build_model_factories
from scripts.sensibilidade_sem_obito_refrec import (
    carregar,
    limpar_nan,
    montar_cenarios,
    remover_obito,
    sha256,
)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/qw1_sensibilidade"
TEX = ROOT / "text/latex/tables/tab_sensibilidade_qw1.tex"
SEED = 42
LIMIAR = 0.5
TOLERANCIA_F1 = 0.01
MODELOS = ("RandomForest", "MLP (128,64)", "MLP (64,32)", "TabNet", "XGBoost")


class CudaMLPClassifier(MLPClassifier):
    """Mesma MLP sklearn, com produtos matriciais e retropropagacao em CUDA.

    Preserva inicializacao, embaralhamento, Adam, lote 200, regularizacao e
    parada por acuracia do sklearn. Adam e controle de treino ficam na CPU;
    nao e uma nova arquitetura PyTorch nem altera a funcao objetivo. Usa
    float64, sem TF32. A equivalencia numerica e coberta por teste CUDA.
    A dependencia da API privada do sklearn e registrada nas versoes do run.
    """

    def _cuda_activations(self, X):
        if self.activation != "relu" or self.out_activation_ != "logistic":
            raise ValueError("Backend restrito a ReLU e saida binaria logistica")
        weights = [torch.as_tensor(w, device="cuda") for w in self.coefs_]
        biases = [torch.as_tensor(b, device="cuda") for b in self.intercepts_]
        acts = [torch.as_tensor(np.asarray(X), device="cuda")]
        for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
            z = acts[-1] @ w + b
            acts.append(torch.sigmoid(z) if i == len(weights) - 1 else z.relu())
        return acts, weights

    def _forward_pass_fast(self, X, check_input=True):
        if check_input:
            X = validate_data(self, X, reset=False)
        with torch.no_grad():
            acts, _ = self._cuda_activations(X)
        return acts[-1].cpu().numpy()

    def _backprop(
        self, X, y, sample_weight, activations, deltas, coef_grads, intercept_grads
    ):
        with torch.no_grad():
            acts, weights = self._cuda_activations(X)
            proba = acts[-1].cpu().numpy()
            sw_sum = len(X) if sample_weight is None else sample_weight.sum()
            loss = LOSS_FUNCTIONS["binary_log_loss"](y, proba, sample_weight)
            loss += 0.5 * self.alpha * sum(np.vdot(w, w) for w in self.coefs_) / sw_sum
            delta = acts[-1] - torch.as_tensor(y, device="cuda", dtype=acts[-1].dtype)
            if sample_weight is not None:
                delta *= torch.as_tensor(sample_weight[:, None], device="cuda")
            for i in reversed(range(len(weights))):
                coef_grads[i] = (
                    ((acts[i].T @ delta + self.alpha * weights[i]) / sw_sum)
                    .cpu()
                    .numpy()
                )
                intercept_grads[i] = (delta.sum(dim=0) / sw_sum).cpu().numpy()
                if i:
                    delta = (delta @ weights[i].T) * (acts[i] > 0)
        return loss, coef_grads, intercept_grads


class CudaTabNet:
    """Configuracao do comparativo local, sem fallback e sem ajuste no teste."""

    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetClassifier

        self.imputer_ = SimpleImputer(strategy="median")
        x = self.imputer_.fit_transform(X).astype(np.float64)
        self.model_ = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.5,
            seed=SEED,
            verbose=0,
            device_name="cuda",
        )
        # O script comparativo nao fornece eval_set. Patience nao interrompe
        # o treino nesse protocolo; nao introduzir validacao diferente aqui.
        self.model_.fit(
            x,
            np.asarray(y, dtype=np.int64),
            max_epochs=100,
            patience=15,
            batch_size=1024,
            num_workers=0,
            pin_memory=False,
        )
        assert self.model_.device.type == "cuda"
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self.imputer_.transform(X).astype(np.float64))


def construir_modelo(nome, n_atributos):
    """Reutiliza parametros originais, alterando somente backend/dispositivo."""
    if nome.startswith("MLP"):
        camadas = (128, 64) if "128" in nome else (64, 32)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    CudaMLPClassifier(
                        hidden_layer_sizes=camadas,
                        max_iter=500,
                        early_stopping=True,
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if nome == "TabNet":
        return CudaTabNet()
    model = _build_model_factories(n_atributos)[nome]()
    if nome == "XGBoost":
        model.set_params(classifier__tree_method="hist", classifier__device="cuda")
    return model


def agregar(y, proba):
    """Somente contagens e metricas, com limiar predefinido inclusivo de 0,5."""
    y, proba = np.asarray(y), np.asarray(proba)
    if y.shape != proba.shape or not np.isfinite(proba).all():
        raise ValueError("Probabilidades invalidas ou desalinhadas")
    pred = proba >= LIMIAR
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        f1=float(f1_score(y, pred, zero_division=0)),
        precisao=float(precision_score(y, pred, zero_division=0)),
        sensibilidade=float(recall_score(y, pred, zero_division=0)),
        auc=float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else None,
        ap=float(average_precision_score(y, proba)) if y.sum() else None,
        vp=int(tp),
        fp=int(fp),
        fn=int(fn),
        vn=int(tn),
        n_teste=len(y),
        vinculos_teste=int(y.sum()),
        limiar=LIMIAR,
    )


def avaliar(nome, X, y, tr, te, cenario, avaliacao="holdout", fold=0):
    inicio = time.perf_counter()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    modelo = construir_modelo(nome, X.shape[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        modelo.fit(X.loc[tr], y.loc[tr])
        proba = modelo.predict_proba(X.loc[te])[:, 1]
    if nome == "XGBoost":
        cfg = json.loads(modelo.named_steps["classifier"].get_booster().save_config())
        assert cfg["learner"]["generic_param"]["device"].startswith("cuda")
    row = dict(
        cenario=cenario,
        modelo=nome,
        avaliacao=avaliacao,
        fold=fold,
        n_atributos=X.shape[1],
        n_treino=len(tr),
        **agregar(y.loc[te], proba),
        tempo_segundos=time.perf_counter() - inicio,
        dispositivo="cpu" if nome == "RandomForest" else "cuda",
        avisos=" | ".join(sorted({str(w.message) for w in caught})),
    )
    print(
        json.dumps(
            {
                k: row[k]
                for k in (
                    "cenario",
                    "modelo",
                    "avaliacao",
                    "fold",
                    "f1",
                    "tempo_segundos",
                )
            }
        ),
        flush=True,
    )
    return row


def numero(valor, casas=3):
    return f"{valor:.{casas}f}".replace(".", ",")


def gerar_tabela(res, meta, path=TEX):
    if set(zip(res.modelo, res.cenario, strict=True)) != {
        (m, c) for m in MODELOS for c in "ABCD"
    } or len(res) != 20:
        raise ValueError("Tabela requer exatamente cinco modelos nos quatro cenarios")
    linhas = [
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\caption{Classificadores do QW1 nos quatro cenários de sensibilidade, "
        r"limiar 0,5.}",
        r"\label{tab:sensibilidade-qw1}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{A} & \multicolumn{2}{c}{B} & "
        r"\multicolumn{2}{c}{C} & \multicolumn{2}{c}{D} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Modelo & $F_1$ & VP/FP & $F_1$ & VP/FP & $F_1$ & VP/FP & $F_1$ & VP/FP \\",
        r"\midrule",
    ]
    for nome in MODELOS:
        label = "Random Forest" if nome == "RandomForest" else nome
        valores = []
        for r in res.loc[res.modelo == nome].sort_values("cenario").itertuples():
            valores.extend((numero(r.f1), f"{r.vp}/{r.fp}"))
        linhas.append(label + " & " + " & ".join(valores) + r" \\")
    versoes = meta["versoes"]
    nota = (
        "Fonte: execução local, agregados em "
        r"\path{data/qw1_sensibilidade/holdout.csv}. "
        "A/B: linha, 58/56 atributos, 18.509 potenciais pares e 74 vínculos no teste; "
        "C/D: REFREC, 58/56 atributos, 18.304 potenciais pares e 72 vínculos. "
        "VP/FP: verdadeiros/falsos positivos; limiar 0,5 fixo. "
        f"GPU {meta['gpu']['nome']} (Random Forest em CPU; "
        "MLP com álgebra CUDA e Adam em CPU). "
        f"PyTorch {versoes['torch']}, scikit-learn {versoes['scikit-learn']}, "
        f"pytorch-tabnet {versoes['pytorch-tabnet']} e XGBoost {versoes['xgboost']}. "
        "Em relação ao QW1 histórico na L4, diferem hardware, versões e partição "
        "por linha do script comparativo. XGBoost com 300 árvores, não as 100 do fluxo "
        "Modal da ablação; TabNet com dimensão 16 e 100 épocas, sem validação interna. "
        "Limiar fixo, sem seleção no teste nesta reexecução ou no comparativo QW1; "
        "a seleção histórica no teste refere-se à ablação."
    )
    linhas.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\textwidth}\footnotesize " + nota,
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(linhas) + "\n", encoding="utf-8", newline="\n")


def salvar_json(path, obj):
    path.write_text(
        json.dumps(limpar_nan(obj), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def comparar_controle(res, referencia):
    """Nao mascara divergencias: retorna cada comparacao e a tolerancia."""
    mapa = {"TabNet (fallback=FT-Transformer)": "TabNet"}
    ref = referencia.assign(modelo=referencia.model.replace(mapa)).set_index("modelo")
    controle = res.loc[res.cenario == "A"].set_index("modelo")
    assert set(controle.index) == set(MODELOS)
    return {
        nome: dict(
            f1_historico=float(ref.loc[nome, "f1"]),
            f1_A=float(controle.loc[nome, "f1"]),
            diferenca=float(controle.loc[nome, "f1"] - ref.loc[nome, "f1"]),
            tolerancia=TOLERANCIA_F1,
            dentro_tolerancia=bool(
                abs(controle.loc[nome, "f1"] - ref.loc[nome, "f1"]) <= TOLERANCIA_F1
            ),
        )
        for nome in MODELOS
    }


def folds_treino(df, y, tr, agrupado):
    for repeticao in range(5):
        if agrupado:
            cv = StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=SEED + repeticao
            )
            splits = cv.split(np.zeros(len(tr)), y.loc[tr], df.loc[tr, "REFREC"])
        else:
            # Mesma sequencia 5x5 do comparativo historico.
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
            splits = cv.split(np.zeros(len(tr)), y.loc[tr])
        for fold, (itr, ite) in enumerate(splits, 1):
            a, b = tr[itr], tr[ite]
            if agrupado:
                assert set(df.loc[a, "REFREC"]).isdisjoint(df.loc[b, "REFREC"])
            yield (repeticao * 5 + fold if agrupado else fold), a, b
        if not agrupado:
            break


def executar(args):
    inicio = time.perf_counter()
    assert torch.cuda.is_available(), "CUDA obrigatoria para esta execucao"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    csv = args.output / "holdout.csv"
    cv_path = args.output / "cv_folds.csv"
    meta_path = args.output / "_meta.json"
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    entrada_sha256 = sha256(args.data)
    if meta and meta.get("entrada_sha256") != entrada_sha256:
        raise ValueError("Entrada diferente do checkpoint; use outro diretorio")
    versoes = {
        p: importlib.metadata.version(p)
        for p in (
            "torch",
            "pytorch-tabnet",
            "xgboost",
            "scikit-learn",
            "imbalanced-learn",
            "pandas",
            "numpy",
            "scipy",
        )
    }
    if meta and meta.get("versoes") != versoes:
        raise ValueError("Versoes diferentes do checkpoint; use outro diretorio")
    execucoes = meta.get("execucoes", [])
    if meta and "fontes_holdout_sha256" not in meta:
        meta["fontes_holdout_sha256"] = meta["fontes_sha256"].copy()
    if not execucoes and "tempo_processo_segundos" in meta:
        execucoes.append(
            {
                "fase": "holdout",
                "tempo_processo_segundos": meta["tempo_processo_segundos"],
                "script_sha256": meta["fontes_sha256"][
                    "scripts/qw1_sensibilidade_gpu.py"
                ],
            }
        )
    smi = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    meta.update(
        inicio_utc=meta.get("inicio_utc", datetime.now(UTC).isoformat()),
        gpu=dict(
            nome=torch.cuda.get_device_name(0),
            nvidia_smi=smi,
            cuda_torch=torch.version.cuda,
        ),
        versoes=versoes,
        python=platform.python_version(),
        semente=SEED,
        limiar=LIMIAR,
        entrada_sha256=entrada_sha256,
        protocolo="holdout 70/30; A/B split_data; C/D REFREC; somente treino na CV",
        mlp_backend=(
            "float64 CUDA forward/backprop; Adam, inicializacao "
            "e early stopping sklearn CPU"
        ),
        tabnet="n_d=n_a=16, 5 passos, 100 epocas, sem eval_set (comparativo local)",
        xgboost="300 estimadores, hist CUDA; nao as 100 arvores da ablacao Modal",
        limiares_historicos="QW1: predict (0.5); ablacao: selecionados no teste",
        fontes_sha256={
            p: sha256(ROOT / p)
            for p in (
                "scripts/qw1_sensibilidade_gpu.py",
                "scripts/qw1_deep_learning_comparison.py",
                "scripts/sensibilidade_sem_obito_refrec.py",
                "scripts/feature_engineering.py",
                "gzcmd/splitting.py",
            )
        },
        hiperparametros={
            nome: construir_modelo(nome, 58).named_steps["classifier"].get_params()
            for nome in MODELOS
            if nome != "TabNet"
        },
    )
    meta["hiperparametros"]["TabNet"] = dict(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        seed=SEED,
        device_name="cuda",
        max_epochs=100,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128,
        eval_set=None,
        num_workers=0,
        pin_memory=False,
        demais="defaults de pytorch-tabnet 4.1.0, iguais ao comparativo local",
    )
    salvar_json(meta_path, meta)
    df = carregar(args.data)
    X, y = engineer_features(df)
    X = X.fillna(0)
    assert (len(df), int(y.sum()), df.REFREC.nunique()) == (61696, 247, 1515)
    assert X.shape[1] == 58 and remover_obito(X).shape[1] == 56
    cenarios = montar_cenarios(df, X, y)
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    for cenario, (particao, matriz, tr, te) in cenarios.items():
        for nome in MODELOS:
            if any(r["cenario"] == cenario and r["modelo"] == nome for r in rows):
                continue
            row = avaliar(nome, matriz, y, tr, te, cenario)
            row.update(
                particao=particao,
                n_REFREC_sobrepostos=len(
                    set(df.loc[tr, "REFREC"]) & set(df.loc[te, "REFREC"])
                ),
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv, index=False, lineterminator="\n")
    res = pd.DataFrame(rows)
    assert len(res) == 20 and not res.duplicated(["cenario", "modelo"]).any()
    ref_path = args.reference_dir / "holdout_metrics.csv"
    referencia = pd.read_csv(ref_path)
    meta["referencia_sha256"] = sha256(ref_path)
    cv_ref = args.reference_dir / "cv_summary.csv"
    meta["historico_cv"] = pd.read_csv(cv_ref)[
        ["model", "f1_mean", "f1_std", "n_folds"]
    ].to_dict("records")
    meta["historico_cv_sha256"] = sha256(cv_ref)
    meta["controle_A"] = comparar_controle(res, referencia)
    meta["investigacao"] = (
        "A/B usam split_data sklearn, conforme sensibilidade; comparativo local usa "
        "split_train_test_indices(row), baseado em default_rng. Mesma semente nao "
        "implica mesmas linhas. A0 abaixo replica a particao do comparativo original. "
        "Nao alterar resultados historicos nem forcar tolerancia."
    )
    salvar_json(meta_path, meta)
    # Controle adicional A0 isola a troca de particao do efeito CUDA/versoes.
    legado = args.output / "controle_particao_original.csv"
    legacy_rows = pd.read_csv(legado).to_dict("records") if legado.exists() else []
    tr0, te0 = split_train_test_indices(
        df, y, spec=SplitSpec(split_by="row", seed=SEED)
    )
    comuns = X.index[te0].intersection(cenarios["A"][3])
    meta["particao_controle_A0"] = dict(
        n_treino=len(tr0),
        n_teste=len(te0),
        linhas_teste_em_comum_com_A=len(comuns),
        vinculos_teste_em_comum_com_A=int(y.loc[comuns].sum()),
    )
    for nome in MODELOS:
        if any(r["modelo"] == nome for r in legacy_rows):
            continue
        row = avaliar(nome, X, y, X.index[tr0], X.index[te0], "A0")
        ref_nome = "TabNet (fallback=FT-Transformer)" if nome == "TabNet" else nome
        f1_ref = float(referencia.set_index("model").loc[ref_nome, "f1"])
        row.update(
            f1_historico=f1_ref,
            diferenca_f1=row["f1"] - f1_ref,
            dentro_tolerancia=abs(row["f1"] - f1_ref) <= TOLERANCIA_F1,
        )
        legacy_rows.append(row)
        pd.DataFrame(legacy_rows).to_csv(legado, index=False, lineterminator="\n")
    meta["controle_particao_original"] = legacy_rows
    meta["tolerancia_f1"] = TOLERANCIA_F1
    if not args.holdout_only:
        cv_rows = pd.read_csv(cv_path).to_dict("records") if cv_path.exists() else []
        # Medicao inicial em A antes de decidir a abrangencia da CV.
        _, matriz, tr, _ = cenarios["A"]
        piloto = next(
            (
                r
                for r in cv_rows
                if r["cenario"] == "A" and r["modelo"] == "TabNet" and r["fold"] == 1
            ),
            None,
        )
        if piloto is None:
            fold, a, b = next(folds_treino(df, y, tr, False))
            piloto = avaliar("TabNet", matriz, y, a, b, "A", "cv", fold)
            cv_rows.append(piloto)
            pd.DataFrame(cv_rows).to_csv(cv_path, index=False, lineterminator="\n")
        estimativa = 25 * piloto["tempo_segundos"]
        cv_cenarios = ("A", "D") if estimativa > 1800 else tuple(cenarios)
        meta["cv"] = dict(
            cenarios=cv_cenarios,
            folds=5,
            repeticoes=5,
            piloto_A_segundos=piloto["tempo_segundos"],
            estimativa_TabNet_por_cenario_segundos=estimativa,
            criterio="se estimativa TabNet > 1800 s, CV somente A e D",
            grupos="REFREC externo e interno em C/D; linha em A/B",
            sementes_REFREC=list(range(42, 47)),
            semente_linha=42,
        )
        salvar_json(meta_path, meta)
        for cenario in cv_cenarios:
            _, matriz, tr, _ = cenarios[cenario]
            for nome in MODELOS:
                for fold, a, b in folds_treino(df, y, tr, cenario in ("C", "D")):
                    if any(
                        r["cenario"] == cenario
                        and r["modelo"] == nome
                        and r["fold"] == fold
                        for r in cv_rows
                    ):
                        continue
                    cv_rows.append(avaliar(nome, matriz, y, a, b, cenario, "cv", fold))
                    pd.DataFrame(cv_rows).to_csv(
                        cv_path, index=False, lineterminator="\n"
                    )
        resumo = (
            pd.DataFrame(cv_rows)
            .groupby(["cenario", "modelo"], sort=False)
            .agg(
                f1_media=("f1", "mean"),
                f1_dp=("f1", lambda v: v.std(ddof=0)),
                precisao_media=("precisao", "mean"),
                sensibilidade_media=("sensibilidade", "mean"),
                auc_media=("auc", "mean"),
                ap_media=("ap", "mean"),
                n_folds=("fold", "count"),
                tempo_segundos=("tempo_segundos", "sum"),
            )
            .reset_index()
        )
        assert (resumo.n_folds == 25).all()
        resumo.to_csv(args.output / "cv_summary.csv", index=False, lineterminator="\n")
    else:
        meta["cv"] = {
            "status": (
                "nao executada nesta reexecucao; "
                "somente holdout A/B/C/D e controle A0"
            ),
            "historico": "CV 5x5 original preservada em historico_cv; nao reestimada",
        }
    meta["fim_utc"] = datetime.now(UTC).isoformat()
    meta["tempo_treinos_segundos"] = (
        float(res.tempo_segundos.sum())
        + sum(r["tempo_segundos"] for r in legacy_rows)
        + (sum(r["tempo_segundos"] for r in cv_rows) if not args.holdout_only else 0)
    )
    meta["tempo_processo_segundos"] = time.perf_counter() - inicio
    execucoes.append(
        {
            "fase": "holdout" if args.holdout_only else "cv",
            "tempo_processo_segundos": meta["tempo_processo_segundos"],
            "script_sha256": sha256(Path(__file__)),
        }
    )
    meta["execucoes"] = execucoes
    meta["tempo_total_segundos"] = sum(r["tempo_processo_segundos"] for r in execucoes)
    meta["intervalo_inicio_fim_segundos"] = (
        datetime.fromisoformat(meta["fim_utc"])
        - datetime.fromisoformat(meta["inicio_utc"])
    ).total_seconds()
    meta["csv_sha256"] = sha256(csv)
    meta["controle_csv_sha256"] = sha256(legado)
    meta["contabilidade_tempos"] = (
        "tempo_treinos_segundos soma ajustes completos A/B/C/D e A0; "
        "tempo_total_segundos soma apenas chamadas registradas em execucoes. "
        "O checkpoint recuperado nao registrou a duracao da retomada que completou "
        "o holdout; portanto tempo_total_segundos nao e o tempo integral "
        "do experimento. "
        "Os 900 s interrompidos incluem trabalho descartado do TabNet B."
    )
    salvar_json(meta_path, meta)
    gerar_tabela(res, meta)
    print("CONCLUIDO", meta["tempo_total_segundos"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv"
    )
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--holdout-only", action="store_true")
    executar(parser.parse_args())


if __name__ == "__main__":
    main()
