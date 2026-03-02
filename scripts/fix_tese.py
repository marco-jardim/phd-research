"""Apply all remaining edits to tese.tex for the new thesis topic."""
import re

PATH = r"D:\git\phd-research\text\latex\tese.tex"

with open(PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

replacements = {
    # Cover: Orientador -> Orientadora (line ~158)
    r"{\large \bf Orientador: Rejane Sobrino Pinheiro}":
        r"{\large \bf Orientadora: Rejane Sobrino Pinheiro}",
    # Cover: comment out Coorientador (line ~162)
    r"{\large \bf Coorientador: Nome completo do coorientador}":
        r"%{\large \bf Coorientador: }",
    # Resumo: title placeholder (line ~255)
    r"{\Large\bf T\'{\i}tulo da Tese}":
        r"{\Large\bf Aprendizado de M\'aquina Aplicado ao P\'os-Processamento \\do Relacionamento Probabil\'{\i}stico de Bases de Dados de Sa\'ude}",
    # Resumo: author placeholder (line ~257)
    "Nome do candidato}\\\\\n":
        "Marco Elisio Oliveira Jardim}\\\\\n",
    # Resumo: keywords placeholder (line ~270)
    r"\textbf{Palavras-chave:} Palavra-chave 1,  Palavra-chave 2, etc.":
        r"\textbf{Palavras-chave:} Relacionamento de registros, Aprendizado de m\'aquina, Linkage probabil\'{\i}stico, Desbalanceamento de classes, Bases de dados de sa\'ude, Tuberculose, SIM, Sinan.",
    # Abstract: title placeholder (line ~286)
    r"{\Large\bf T\'{\i}tulo em ingl\^es}":
        r"{\Large\bf Machine Learning Applied to Post-Processing of \\Probabilistic Record Linkage of Health Databases}",
    # Abstract: Orientador (line ~290)
    r"{\bf Orientador: }\\":
        r"{\bf Advisor: Rejane Sobrino Pinheiro}\\",
    # Abstract: Coorientador (line ~291)
    r"{\bf Coorientador: }\\\\":
        r"%{\bf Coorientador: }\\\\",
    # Abstract: keywords placeholder (line ~301)
    r"\textbf{Keywords:} Palavras-chave em ingl\^es.":
        r"\textbf{Keywords:} Record linkage, Machine learning, Probabilistic linkage, Class imbalance, Health databases, Tuberculosis, SIM, Sinan.",
}

content = "".join(lines)

applied = 0
for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new, 1)
        applied += 1
        print(f"  OK: {old[:50]}...")
    else:
        print(f"  SKIP (not found): {old[:50]}...")

# Nomenclature: replace old entries with new ones
old_nomencl_start = r"\nomenclature{SUS}"
old_nomencl_end = r"\nomenclature{RNP}{Rede Nacional de Ensino e Pesquisa}"
if old_nomencl_start in content and old_nomencl_end in content:
    start_idx = content.index(old_nomencl_start)
    end_idx = content.index(old_nomencl_end) + len(old_nomencl_end)
    new_nomencl = r"""\nomenclature{SUS}{Sistema \'Unico de Sa\'ude}
\nomenclature{SIM}{Sistema de Informa\c{c}\~ao sobre Mortalidade}
\nomenclature{Sinan}{Sistema de Informa\c{c}\~ao de Agravos de Notifica\c{c}\~ao}
\nomenclature{SIH-SUS}{Sistema de Informa\c{c}\~oes Hospitalares do SUS}
\nomenclature{SIA-SUS}{Sistema de Informa\c{c}\~oes Ambulatoriais do SUS}
\nomenclature{Sinasc}{Sistema de Informa\c{c}\~ao sobre Nascidos Vivos}
\nomenclature{GAL}{Gerenciador de Ambiente Laboratorial}
\nomenclature{SITETB}{Sistema de Informa\c{c}\~ao de Tratamentos Especiais da Tuberculose}
\nomenclature{TB}{Tuberculose}
\nomenclature{ML}{Aprendizado de M\'aquina (\textit{Machine Learning})}
\nomenclature{RF}{Floresta Aleat\'oria (\textit{Random Forest})}
\nomenclature{XGBoost}{\textit{Extreme Gradient Boosting}}
\nomenclature{LightGBM}{\textit{Light Gradient Boosting Machine}}
\nomenclature{SVM}{M\'aquina de Vetores de Suporte (\textit{Support Vector Machine})}
\nomenclature{MLP}{Perceptron Multicamadas (\textit{Multilayer Perceptron})}
\nomenclature{SMOTE}{\textit{Synthetic Minority Over-sampling Technique}}
\nomenclature{AUC-ROC}{\'Area sob a Curva ROC (\textit{Area Under the ROC Curve})}
\nomenclature{AUC-PR}{\'Area sob a Curva Precis\~ao-Revoca\c{c}\~ao}
\nomenclature{KDD}{Descoberta de Conhecimento em Bases de Dados (\textit{Knowledge Discovery in Databases})}
\nomenclature{LGPD}{Lei Geral de Prote\c{c}\~ao de Dados}"""
    content = content[:start_idx] + new_nomencl + content[end_idx:]
    applied += 1
    print("  OK: Nomenclature entries replaced")
else:
    print("  SKIP: Nomenclature block not found")

with open(PATH, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nDone. Applied {applied} replacements.")
