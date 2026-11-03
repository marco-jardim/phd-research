"""Verificação dos artefatos M5b e, opcionalmente, do PDF compilado da tese."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    valid = pd.read_csv(ROOT / "data/ancoras_beta_validacao.csv")
    curves = pd.read_csv(ROOT / "data/ancoras_beta_confiabilidade.csv")
    decisions = pd.read_csv(ROOT / "data/ancoras_beta_decisao.csv")
    sensitivity = pd.read_csv(ROOT / "data/ancoras_beta_sensibilidade.csv")
    assert len(valid) == 10 and len(decisions) == 10 and len(sensitivity) == 8
    assert valid.calibracao.nunique() == 5
    assert np.isfinite(valid[["brier", "log_loss", "ece_10", "auc"]]).all().all()
    for _, row in valid.iterrows():
        bins = curves[curves.calibracao.eq(row.calibracao) & curves.escopo.eq(row.escopo)
                      & curves.tipo_bin.eq("uniforme")]
        assert len(bins) == 10 and bins.n.sum() == row.n
        assert bins.positivos.sum() == row.positivos
        assert abs(bins.contribuicao_ece.sum() - row.ece_10) < 1e-12
    assert (decisions[["revisao", "auto_match", "auto_nonmatch"]].sum(axis=1) == 21620).all()
    # Nenhum CSV novo deve conter identificadores ou campos dos registros.
    for path in (ROOT / "data").glob("ancoras_beta_*.csv"):
        cols = pd.read_csv(path, nrows=0).columns
        assert not any(c.startswith(("R_", "C_", "COMPREC", "REFREC", "NOME", "DTNASC")) for c in cols)
    print("ARTIFACTS PASS: five models, 10 metrics/decisions, 8 sensitivities, ECE/count consistency, no record fields.")
    if args.log:
        log = args.log.read_text(encoding="utf-8", errors="replace")
        assert not any(line.startswith("!") for line in log.splitlines()), "Erro LaTeX no log"
        assert "Citation" not in "\n".join(line for line in log.splitlines() if "undefined" in line), "Citação indefinida"
        assert "There were undefined citations" not in log
        assert not re.search(r"Citation.{0,240}?undefined", log, flags=re.DOTALL), "Citação indefinida em aviso quebrado em linhas"
        assert "tab_cap8_ancoras_beta.tex" in log
        print("LATEX LOG PASS: new table loaded, no LaTeX error or undefined citation.")
    if args.pdf:
        import fitz

        doc = fitz.open(args.pdf)
        texts = [page.get_text() for page in doc]
        headings = [i + 1 for i, text in enumerate(texts) if "Calibração Beta por momentos" in text]
        tables = [i + 1 for i, text in enumerate(texts)
                  if "Beta por âncoras" in text and "Beta + restrição cinza" in text]
        assert headings and tables, "Subseção/tabela não localizada no PDF"
        print(f"PDF PASS: {len(doc)} pages; subsection title pages={headings}; new table pages={tables}.")
        normalized = [re.sub(r"\s+", " ", text) for text in texts]
        body = [i + 1 for i, text in enumerate(normalized)
                if "Calibração Beta por momentos" in text and "Na margem OOF" in text]
        ending = [i + 1 for i, text in enumerate(normalized) if "não solução sem gabarito." in text]
        assert len(body) == 1 and body == ending and body == tables, "Subseção excede uma página ou tabela deslocada"
        print(f"ONE-PAGE SUBSECTION PASS: model, table and interpretation on PDF page {body[0]}.")


if __name__ == "__main__":
    main()
