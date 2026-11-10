"""Faixas da nota final dos pares verdadeiros recuperados e já encontrados (agregado, sem identificadores).

Universo: conjunto de teste do RF+SMOTE histórico (18.509 potenciais pares, 74 pares verdadeiros),
limiar de probabilidade 0,5. Definições (as mesmas de scripts/banca_b_recuperados_por_par.py):
  tp           = probabilidade >= 0,5 e par verdadeiro (PAR em {1, 2});
  recuperados  = tp e nota final < 8 (aceitos pelo RF+SMOTE e não pela decisão de limiar da nota >= 8);
  ja_encontrados = tp e nota final >= 8 (comuns à nota e ao modelo).
Faixas semiabertas [min, max). Saída: data/perfil_recuperados_faixas.csv (faixa, limites, recuperados,
ja_encontrados, total) e data/perfil_recuperados_faixas_metadata.json. Nenhuma linha por par, nenhum
identificador; o extrato é lido como texto (dtype=str) e as predições ficam só em memória.

Uso: python -m scripts.perfil_recuperados_faixas [--input CAMINHO] [--predictions CSV | --retrain-historical]
Sem --predictions, retreina o RF+SMOTE histórico (padrão, seed 42) e falha se os controles
(teste 18.509; 74 positivos; 67 tp; 3 fp; 27 recuperados) não forem reproduzidos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV = ROOT / "data/perfil_recuperados_faixas.csv"
METADATA = ROOT / "data/perfil_recuperados_faixas_metadata.json"
CORTE_NOTA = 8.0
LIMIAR = 0.5
SEED = 42
CONTROLES = dict(test_n=18509, test_positivos=74, tp=67, fp=3, recuperados=27)
# (rótulo, mínimo incluído, máximo excluído); None = aberto.
FAIXAS = [("<5", None, 5.0), ("5-6", 5.0, 6.0), ("6-7", 6.0, 7.0), ("7-8", 7.0, 8.0),
          ("8-9", 8.0, 9.0), ("9-11", 9.0, 11.0), (">=11", 11.0, None)]
COLUNAS = ["faixa", "nota_min", "nota_max_exclusivo", "recuperados", "ja_encontrados", "total"]


def tabular(nota: pd.Series, recuperados: pd.Series, ja_encontrados: pd.Series) -> pd.DataFrame:
    """Conta recuperados e já encontrados por faixa da nota; as faixas cobrem toda a reta, sem lacuna."""
    if nota.isna().any():
        raise ValueError("Nota final ausente entre os potenciais pares do teste")
    if (recuperados & ja_encontrados).any():
        raise ValueError("Recuperados e já encontrados devem ser disjuntos")
    linhas = []
    for rotulo, minimo, maximo in FAIXAS:
        dentro = pd.Series(True, index=nota.index)
        if minimo is not None:
            dentro &= nota.ge(minimo)
        if maximo is not None:
            dentro &= nota.lt(maximo)
        rec, jae = int((dentro & recuperados).sum()), int((dentro & ja_encontrados).sum())
        linhas.append(dict(faixa=rotulo, nota_min=minimo, nota_max_exclusivo=maximo,
                           recuperados=rec, ja_encontrados=jae, total=rec + jae))
    frame = pd.DataFrame(linhas, columns=COLUNAS)
    if frame.recuperados.sum() != int(recuperados.sum()) or frame.ja_encontrados.sum() != int(ja_encontrados.sum()):
        raise ValueError("As faixas não cobrem todos os pares selecionados")
    return frame


def faixas_do_teste(raw: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Tabula o teste original a partir do extrato (texto) e das probabilidades fora da amostra."""
    from scripts.feature_engineering import engineer_features, split_data

    numeric = raw.apply(lambda col: pd.to_numeric(col.str.replace(",", ".", regex=False), errors="coerce"))
    X, y = engineer_features(numeric)
    _, test, _, y_test = split_data(X, y, random_state=SEED)
    if not predictions.index.is_unique or set(predictions.index) != set(test.index):
        raise ValueError("Predições devem cobrir exatamente o teste original, sem duplicatas")
    probability = pd.to_numeric(predictions.loc[test.index, "probabilidade"], errors="raise")
    if not probability.between(0, 1).all():
        raise ValueError("Probabilidades inválidas")
    positive = probability.ge(LIMIAR)
    tp = positive & y_test.eq(1)
    nota = test["nota final"]
    recuperados, ja_encontrados = tp & nota.lt(CORTE_NOTA), tp & nota.ge(CORTE_NOTA)
    controles = dict(test_n=len(test), test_positivos=int(y_test.sum()), tp=int(tp.sum()),
                     fp=int((positive & y_test.eq(0)).sum()), recuperados=int(recuperados.sum()))
    if controles != CONTROLES:
        raise ValueError(f"Controle histórico não reproduzido: {controles}")
    return tabular(nota, recuperados, ja_encontrados), controles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--predictions", type=Path)
    source.add_argument("--retrain-historical", action="store_true")
    args = parser.parse_args()
    pd.options.future.infer_string = False
    raw = pd.read_csv(args.input, sep=";", dtype=str)
    raw.columns = [c.split(",")[0] for c in raw.columns]
    if args.predictions:
        predictions = pd.read_csv(args.predictions, index_col="indice")
    else:
        from scripts.banca_b_encerramento_agregado import historical_predictions
        predictions, _ = historical_predictions(raw)
    frame, controles = faixas_do_teste(raw, predictions)
    frame.to_csv(CSV, index=False, lineterminator="\n")
    import imblearn
    import sklearn
    metadata = dict(
        fonte="data/COMPARADORSEMIDENT.csv (privado, não versionado), teste original do RF+SMOTE histórico",
        definicao="recuperados = RF+SMOTE >= 0,5, par verdadeiro (PAR 1 ou 2) e nota final < 8; "
                  "ja_encontrados = RF+SMOTE >= 0,5, par verdadeiro e nota final >= 8; faixas [min, max)",
        seed=SEED, limiar=LIMIAR, corte_nota=CORTE_NOTA, controles=controles,
        procedimento="predicoes_csv" if args.predictions else "retreino_reproducao_ablacao_historica",
        entrada_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        predicoes_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest() if args.predictions else None,
        fontes_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "scripts/perfil_recuperados_faixas.py", "scripts/banca_b_encerramento_agregado.py",
            "scripts/feature_engineering.py")},
        versoes=dict(python=platform.python_version(), pandas=pd.__version__,
                     scikit_learn=sklearn.__version__, imbalanced_learn=imblearn.__version__),
        csv_sha256=hashlib.sha256(CSV.read_bytes()).hexdigest())
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(frame.to_string(index=False))
    print(json.dumps(controles))


if __name__ == "__main__":
    main()
