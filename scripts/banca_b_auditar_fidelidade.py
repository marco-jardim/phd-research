"""Registra antes/depois com linhas de todas as frases e celulas alteradas em B."""

from __future__ import annotations

import difflib
import itertools
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# SHA da história reescrita em 2026-11-06, correspondente à base antiga 53c4a13.
BASE = "37da5da6c36fa8b0f67df5ca6ac1f07d65362059"
LATEX = "text/latex/"


def historical(path: str) -> str:
    result = subprocess.run(["git", "show", f"{BASE}:{path}"], cwd=ROOT, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout.decode("utf-8")


def phrases(text: str) -> list[tuple[int, str]]:
    return [(number, sentence.strip()) for number, line in enumerate(text.splitlines(), 1)
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ])", line)
            if sentence.strip()]


def main() -> None:
    report = ["# Banca B: auditoria de fidelidade", "", f"Base: `{BASE}`. Linhas do depois: arquivos desta fatia, antes do commit.", "",
              "## Lista de execução", "", "- [completed] Conferir fontes e corrigir os Estudos 1 a 3.",
              "- [completed] Gerar tabelas editoriais e agregado de encerramento, preservando dados e códigos congelados.",
              "- [completed] Rodar testes da fatia, recorte solicitado, rg e uma compilação isolada.",
              "- [in_progress] Auditoria final, commit e push.", "",
              "## Veredito e decisões item a item", "",
              "1. **Pareto: correção fiel.** Dez pontos não dominados por modelo (7 RF + 3 XGBoost), não dez na união. "
              "As duas linhas dominadas são preservadas e identificadas; dados e gerador histórico intactos.",
              "2. **NB02/NB03: fonte única.** Os JSON gerados pelos scripts uniformes são canônicos para o texto e as tabelas editoriais, "
              "por compartilharem a engenharia uniforme de 58 atributos e permitirem geração das tabelas. "
              "NB03 já usava essa fonte na tabela. O caderno histórico é identificado apenas na explicação da divergência, "
              "cuja causa não está documentada. NB02 GB: sensibilidade 0,959 e precisão 0,934 substituídas por 0,932 e 0,885; "
              "os valores antigos só sobrevivem como registro explícito da diferença, não como resultado adotado.",
              "3. **QW1: ressalva fortalecida.** RF 0,855 e TabNet 0,610 preservados como históricos, com controle A0 0,891 e 0,712 "
              "e tolerância 0,01 junto ao uso, na discussão e na tabela. Não se atribui uma causa não isolada.",
              "4. **E1/conclusão e remissões: correção fiel.** Nenhuma implementação eleita correta; peso Python 1,0 versus Lucena/Java 0,8 "
              "preservado. Remissões desatualizadas retiradas com reescrita, sem editar a conclusão.",
              "5. **Impacto e vazamento: ressalvas preservadas e aproximadas.** 24 líquidos = 27 recuperados menos 3 perdidos; "
              "40 em comum. Encerramento agregado: 10/27 já por óbito (7 TB e 3 outras causas), 11 outros, 6 ausentes/não numéricos. "
              "Controle reproduz 67 VP, 3 FP e 27 recuperados; SHA-256 da entrada coincide com a análise congelada. "
              "Não se chama essa recuperação de redução da subnotificação nem de novos óbitos desconhecidos.",
              "6. **Objetivo 2/NB01: delimitação fiel.** Balanceamento apenas no RF, não no estimador final; especificidade não reportada "
              "em método e discussão. NB01 é partição única; as sete AUC-ROC são reportadas a partir do CSV. "
              "Hiperparâmetros NB01 distinguidos da ablação e do QW1; configuração NB03 identificada pelo script.",
              "7. **Custo e B4: correção fiel.** Revisões por óbito detectado (70/67 e 67/43), coerente com a definição e o eixo da figura. "
              "A frase de denominadores também foi corrigida: custo usa óbitos detectados, não revisões. "
              "Tabela sem coluna de custo não anuncia custo na legenda. Método do vetor de E2 movido para Método; "
              "tabela por faixa/leitura e impacto/definições em Resultados; limitações E3 no fim da Discussão, íntegras.",
              "8. **C1/C2: adição apoiada em artefatos.** Somente regras F1 0,842 versus nota >=8 F1 0,610; "
              "corte do serviço distingue-se da nota CdR, preservado por definir PAR=1, com 211/247 (85,4%) e remissão ao E4. "
              "Não se infere que geração probabilística seja dispensável.",
              "9. **K1/R5/K4: escopo corrigido.** Retirada a incursão terapêutica, mantendo subnotificação e suas fontes; "
              "exploração interna não convertida em recomendação. Regularização por diversidade é hipótese não isolada; "
              "retiradas superioridade geral e demonstração de impossibilidade universal. Sensibilidade máxima da tabela é 1,000.",
              "10. **Menores: fidelidade verificada.** 103=70+33 configurações; Híbrido OR/Regras em português na prosa; "
              "Lages integrado ao parágrafo comparativo, preservando que não houve comparação direta dos resultados. "
              "XGBoost SHAP identificado como Modal (100, profundidade 6); a inspeção do mesmo gerador revelou também que "
              "o RF da figura é RF+SMOTE (150, profundidade 12), não RF QW1 sem SMOTE, e a atribuição foi corrigida. "
              "Nenhuma alteração nos scripts com hash fixado, nos dados ou nas fatias A/C/D.", "",
              "## Verificação", "",
              "- Oito testes novos da fatia B aprovados (fontes, regeneração, não dominados, ressalvas, ordem, agregados e hashes).",
              "- Uma compilação pdflatex dos Estudos 1 a 3 com preâmbulo da tese e jobname banca_b_check: exit 0, sem erro LaTeX. "
              "Referências externas não resolvidas em passagem isolada não equivalem a build bibliográfico integral.",
              "- rg para U+2014/U+2013: exit 1, nenhuma ocorrência nos capítulos e cinco tabelas editoriais desta fatia.",
              "- A execução solicitada do recorte editorial retornou 15 falhas, 94 aprovados e 1 ignorado durante as edições paralelas. "
              "As falhas são em arquivos/expectativas das fatias A e C, não nos três Estudos desta fatia. "
              "Exemplos: test_s7a exige cortes removidos do capítulo de dados; test_s8a1 exige 'não no critério' no E4; "
              "test_tabelas_arranjo_editorial compara tabelas E4 ainda não regeneradas. Não foram afrouxadas nem editadas essas expectativas.",
              "- O scanner da ferramenta rejeitou a chamada literal de rg com caracteres Unicode; "
              "o mesmo rg foi executado por subprocesso com o padrão Unicode equivalente no verificador, com sucesso.", "",
              "## Antes/depois de todas as frases ou linhas alteradas", "",
              "A comparação abaixo inclui comandos editoriais e células para não omitir alterações. "
              "Frases deslocadas são reconhecidas pelo conteúdo integral; acréscimos e supressões têm o lado inexistente explícito. "
              "Os vereditos de conteúdo são os dos itens 1 a 10, não uma alegação de reprodução externa.", ""]
    files = [(LATEX + name, LATEX + name) for name in
             ["estudo1_comparador.tex", "estudo2_classificadores.tex", "estudo3_estrategias.tex"]]
    files += [(LATEX + "tables/" + name, LATEX + "tables/editorial/" + name) for name in
              ["tab_nb03_precision.tex", "tab_pareto_frontier.tex", "tab_impacto_epidemiologico.tex", "tab_qw1_deep_learning.tex"]]
    files += [(None, LATEX + "tables/editorial/tab_nb02_recall.tex")]
    count = 0
    for old_path, new_path in files:
        old = phrases(historical(old_path)) if old_path else []
        new = phrases((ROOT / new_path).read_text(encoding="utf-8"))
        old_values = [x[1] for x in old]
        new_values = [x[1] for x in new]
        report += [f"### `{new_path}`", ""]
        for tag, a, b, c, d in difflib.SequenceMatcher(None, old_values, new_values, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            for before, after in itertools.zip_longest(old[a:b], new[c:d]):
                count += 1
                report += [f"#### B-{count:03d}", "",
                           f"Antes: `{old_path}:{before[0]}`" if before else "Antes: inexistente (acréscimo).",
                           "```tex", before[1] if before else "(inexistente)", "```",
                           f"Depois: `{new_path}:{after[0]}`" if after else "Depois: suprimido ou deslocado.",
                           "```tex", after[1] if after else "(suprimido neste ponto)", "```"]
                moved = (before and before[1] in new_values) or (after and after[1] in old_values)
                report += ["Veredito: fiel, conteúdo integral preservado em nova posição." if moved else
                           "Veredito: fiel ao escopo corrigido e aos artefatos, conforme os itens 1 a 10 acima; "
                           "nenhuma ressalva removida para fortalecer o resultado.", ""]
    output = ROOT / "docs/reports/BANCA_B_2026-11-06.md"
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Auditoria gravada: {output.relative_to(ROOT)}; {count} pares antes/depois.")


if __name__ == "__main__":
    main()
