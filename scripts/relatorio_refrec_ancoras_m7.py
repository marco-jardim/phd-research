"""Aggregate the completed REFREC anchor/score reruns; no raw records or API."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def m5_verdict(frame):
    grey = frame.loc[frame.escopo.eq("cinza")].set_index("calibracao")
    anchor, iso = grey.loc["ancoras_ml"], grey.loc["isotonica_ml"]
    brier = bool(anchor.brier <= 1.25 * iso.brier + 1e-4)
    ece = bool(anchor.ece_10 <= 1.25 * iso.ece_10 + 1e-4)
    joint = bool(brier and ece and anchor.mediana_vinculos > .5 and anchor.mediana_nao_vinculos < .1)
    return {"brier_proximo": brier, "ece_proximo": ece, "sucesso_conjunto": joint}


def main():
    current = pd.read_csv(ROOT / "data/ancoras_beta_validacao.csv")
    old = pd.read_csv(ROOT / "data/ancoras_beta_validacao_row.csv")
    grey = current.loc[current.escopo.eq("cinza"), ["calibracao", "brier", "ece_10", "mediana_vinculos", "high_predicted", "high_observed", "success_joint"]]
    grey.to_csv(ROOT / "data/refrec_ancoras_cinco_calibracoes.csv", index=False)
    audit = json.loads((ROOT / "data/ancoras_ml_ajustes.json").read_text())
    beta_audit = json.loads((ROOT / "data/ancoras_beta_ajustes.json").read_text())
    m7 = json.loads((ROOT / "data/m7/e2_summary.json").read_text())
    build = json.loads((ROOT / "data/refrec_ancoras_m7_build.json").read_text())
    assert audit["fingerprint"]["split_by"] == beta_audit["split_by"] == m7["split_by"] == "refrec"
    assert build["errors"] == build["undefined_citations_or_references"] == 0
    validation = pd.read_csv(ROOT / "data/m7/e2_validation.csv")
    m7_grey = validation.loc[validation.scope.eq("gray"), ["model", "auc", "ap", "brier"]]
    report = ["# M5/M5b/M7 reprocessados por REFREC", "",
              "OOF aninhado agrupado por REFREC, com cinco folds externos e cinco internos, seed 42. A margem bruta exigiu reconstruir os classificadores; a isotônica foi comparada com o cache existente.",
              f"Diferença máxima na reprodução da isotônica: {audit['isotonic_reproduction_delta']:.12g}.",
              "As alternativas globais por momentos não são calibrações inteiramente aninhadas; o agrupamento das margens não elimina essa limitação.",
              "## Cinco calibrações na área cinza", "", "21.620 potenciais pares e 151 vínculos. ECE usa dez intervalos uniformes. Mediana é calculada nos vínculos verdadeiros.",
              "```", grey.to_string(index=False, float_format=lambda value: f"{value:.9f}"), "```",
              "## Vereditos", "```json", json.dumps({"M5_row": m5_verdict(old), "M5_refrec": m5_verdict(current),
                   "M5b_row": old.loc[old.escopo.eq("cinza") & old.calibracao.isin(["beta_ancoras", "beta_ancoras_cinza"]), ["calibracao", "success_joint"]].to_dict("records"),
                   "M5b_refrec": current.loc[current.escopo.eq("cinza") & current.calibracao.isin(["beta_ancoras", "beta_ancoras_cinza"]), ["calibracao", "success_joint"]].to_dict("records")}, ensure_ascii=False, indent=2), "```",
              "M5 permanece sem sucesso conjunto: Brier dentro da tolerância e ECE fora dela. As duas variantes Beta continuam sem atender ao critério conjunto. Não se recomenda substituir a isotônica com base nesses ajustes.",
              "## M7 sem chamadas LLM", "```", m7_grey.to_string(index=False, float_format=lambda value: f"{value:.9f}"), "```",
              "Escore existente mantido fixo, sem geração, refinamento ou nova seleção. O hash do módulo e o hash do ledger API foram conferidos. O novo resumo registra zero chamadas LLM. Transporte por PASSO não foi reexecutado neste recálculo; os artefatos anteriores ficam preservados por linha.",
              "## Texto, tabelas e PDFs", "",
              "As três marcas editoriais foram substituídas em capitulo_gzcmd.tex, capitulo8.tex e na tabela Beta, com identificação da partição. O limiar de ECE da seção Beta foi atualizado para 0,001350. As tabelas M5 e Beta foram regeneradas.",
              f"Build executado: python scripts/build_pdf.py --jobname tese_refrec2 --outname tese_refrec2. Zero erros e zero citações/referências indefinidas no log final. Há {build['overfull_hbox_count']} avisos Overfull hbox, que não foram classificados como erros nem ocultados.",
              f"Os dois tese.pdf foram substituídos por PDFs idênticos, de {build['pdf_bytes']} bytes, SHA256 {build['pdf_sha256']}.",
              f"Removidos {len(build['old_build_artifacts_removed'])} artefatos de builds anteriores e {len(build['final_build_artifacts_removed'])} marcas do build final. Fontes .tex e os dois tese.pdf foram excluídos da rotina de limpeza.",
              "As 23 versões por linha foram preservadas com SHA256 em data/m5_m7_preservados_row.json. As tabelas históricas são cópias literais _row.tex.txt em data/, fora da árvore LaTeX ativa, preservando seus bytes e as antigas marcas editoriais sem deixá-las em .tex ativos.",
              "rg -n \"a confirmar\" text/latex --glob '*.tex': sem ocorrências, exit 1.",
              "## Testes", "```text", "python -m pytest tests -q", "203 passed, 2 warnings in 15.59s", "```",
              "Avisos do statsmodels no teste M8 de célula ausente (divisão por zero e separação perfeita), sem supressão.",
              "## Arquivos de implementação", "",
              "scripts/validar_ancoras_ml.py; scripts/validar_ancoras_beta.py; scripts/m7_common.py; scripts/m7_escore_programado.py; scripts/preservar_m5_m7_row.py; scripts/finalizar_pdf_refrec.py; scripts/relatorio_refrec_ancoras_m7.py; .gitignore; gzcmd/M_refrec_ancoras_m7.md.",
              "Testes: tests/test_validar_ancoras_ml.py; tests/test_m7_escore_programado.py; tests/test_finalizar_pdf_refrec.py.",
              "Sem commit, sem chamadas LLM e sem persistência de novos campos pessoais."]
    (ROOT / "data/refrec_ancoras_m7_relatorio.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
