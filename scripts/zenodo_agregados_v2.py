"""Monta a pasta local com os arquivos novos da versão 2 do depósito público de agregados.

Depósito: Zenodo 10.5281/zenodo.24034389 (conceito 10.5281/zenodo.24034388),
CC-BY-4.0, acesso aberto; título na entrada Jardim2026curvas de
text/latex/referencias.bib. Lista de arquivos aprovada pela autora em 2026-11-10.

O script só lê data/ e gzcmd/ e escreve apenas em .local/zenodo_agregados_v2/
(fora do git). Não acessa a rede nem o Zenodo. Para cada arquivo da lista:

1. usa a cópia do repositório, que é o caminho citado pela tese, e a compara byte
   a byte com a cópia em data/zenodo/ (pacote v2_gzcmd_v31 ou qw3); se diferirem,
   mantém a cópia citada e relata a diferença;
2. inclui o arquivo de metadados de cada CSV (<nome>_metadata.json ou
   <nome>_meta.json, ou o metadado compartilhado indicado na lista) quando existe e
   não contém caminho local absoluto; caso contrário, relata e exclui;
3. verifica a privacidade (nomes de colunas, datas completas e caminhos locais
   absolutos) e copia só os arquivos aprovados, com nomes planos (sem subpastas);
4. grava MANIFEST_agregados_v2.csv e README_agregados_v2.md e verifica de novo a
   pasta montada.

Nunca imprime valores de linhas: só cabeçalhos, contagens e números de linha.
Saída 1 se algum arquivo da lista faltar ou falhar na verificação.

Uso: python scripts/zenodo_agregados_v2.py [--root RAIZ] [--out .local/zenodo_agregados_v2]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ".local/zenodo_agregados_v2"
MANIFEST_NAME = "MANIFEST_agregados_v2.csv"
README_NAME = "README_agregados_v2.md"
MANIFEST_COLUMNS = ("arquivo", "origem", "bytes", "sha256", "n_linhas", "colunas")
V1_MANIFEST = "data/curvas_zenodo.json"  # arquivos da versão 1 (nomes reservados)
ZENODO_V2 = "data/zenodo/v2_gzcmd_v31"
ZENODO_QW3 = "data/zenodo/qw3"
SCRIPT_REL = "scripts/zenodo_agregados_v2.py"

DOI_VERSAO_1 = "10.5281/zenodo.24034389"
DOI_CONCEITO = "10.5281/zenodo.24034388"
DOI_RESTRITO = "10.5281/zenodo.23948459"
TITULO = ("Curvas agregadas de recuperação de vínculos abaixo do corte: ordenações, "
          "fusão e contexto por óbito (tese de doutorado, IESC/UFRJ)")

# ---------------------------------------------------------------------------
# Verificação de privacidade
# ---------------------------------------------------------------------------

SUBSTRINGS_PROIBIDAS = (
    "pair_id", "pairid", "comprec", "refrec", "row_index", "endereco", "logradouro",
    "dtnasc", "dt_nasc", "dt_obito", "dtobito", "data_nasc", "datanasc",
)
TOKENS_PROIBIDOS = frozenset({"index", "nome", "cpf", "cns"})
PREFIXO_NM = re.compile(r"(?:^|[^a-z0-9])nm_")
DATA_COMPLETA = re.compile(r"(?<!\d)(?:\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})(?!\d)")
CAMINHOS_ABSOLUTOS = (
    # C:\Users\, D:\git\, com barra normal ou barra invertida escapada (JSON).
    re.compile(r"(?i)(?<![a-z0-9])[a-z]:(?:\\{1,2}|/)+(?:users|git)(?:\\{1,2}|/)"),
    # Qualquer outro caminho absoluto com letra de unidade.
    re.compile(r"(?i)(?<![a-z0-9])[a-z]:(?:\\{1,2}|/)[^\s\\/\"',;]"),
    # Diretórios de usuário POSIX.
    re.compile(r"(?<![A-Za-z0-9._-])/(?:home|Users)/[^/\s]+"),
)
TRAVESSOES = ("\u2014", "\u2013")
csv.field_size_limit(2**31 - 1)


def _normalizar(nome: str) -> str:
    decomposto = unicodedata.normalize("NFKD", nome)
    return "".join(c for c in decomposto if not unicodedata.combining(c)).strip().lower()


def coluna_proibida(nome: str) -> bool:
    """True se o nome da coluna sugere identificador de par, de registro ou dado pessoal."""
    normal = _normalizar(nome)
    if any(trecho in normal for trecho in SUBSTRINGS_PROIBIDAS):
        return True
    if PREFIXO_NM.search(normal):
        return True
    return any(token in TOKENS_PROIBIDOS for token in re.split(r"[^a-z0-9]+", normal) if token)


def _tem_caminho_absoluto(texto: str) -> bool:
    return any(padrao.search(texto) for padrao in CAMINHOS_ABSOLUTOS)


def _contagem(contador: Counter) -> str:
    return ", ".join(f"{coluna} ({n} célula(s))" for coluna, n in sorted(contador.items()))


def _ler_texto(caminho: Path) -> str:
    return caminho.read_bytes().decode("utf-8-sig")


def checar_privacidade(caminho: Path) -> list[str]:
    """Problemas de privacidade do arquivo; lista vazia significa aprovado.

    CSV: nomes de colunas proibidos, células com data completa (dd/mm/aaaa ou
    aaaa-mm-dd) e células com caminho local absoluto. Demais arquivos de texto:
    caminho local absoluto em qualquer linha. As mensagens trazem só nomes de
    colunas, contagens e números de linha, nunca valores.
    """
    caminho = Path(caminho)
    try:
        texto = _ler_texto(caminho)
    except UnicodeDecodeError:
        return ["não decodificável como UTF-8; verificação impossível"]
    problemas: list[str] = []
    if caminho.suffix.lower() == ".csv":
        leitor = csv.reader(io.StringIO(texto, newline=""))
        cabecalho = next(leitor, None)
        if not cabecalho:
            return ["CSV sem cabeçalho"]
        proibidas = [coluna for coluna in cabecalho if coluna_proibida(coluna)]
        if proibidas:
            problemas.append("coluna(s) proibida(s): " + ", ".join(proibidas))
        if any(_tem_caminho_absoluto(c) or DATA_COMPLETA.search(c) for c in cabecalho):
            problemas.append("cabeçalho com data completa ou caminho absoluto")
        datas: Counter = Counter()
        caminhos: Counter = Counter()
        for registro in leitor:
            for j, celula in enumerate(registro):
                coluna = cabecalho[j] if j < len(cabecalho) else f"<coluna {j + 1}>"
                if DATA_COMPLETA.search(celula):
                    datas[coluna] += 1
                if _tem_caminho_absoluto(celula):
                    caminhos[coluna] += 1
        if datas:
            problemas.append("data completa em " + _contagem(datas))
        if caminhos:
            problemas.append("caminho local absoluto em " + _contagem(caminhos))
    else:
        linhas = [i for i, linha in enumerate(texto.splitlines(), 1) if _tem_caminho_absoluto(linha)]
        if linhas:
            amostra = ", ".join(str(i) for i in linhas[:10]) + (" ..." if len(linhas) > 10 else "")
            problemas.append(f"caminho local absoluto em {len(linhas)} linha(s): {amostra}")
    return problemas


def resumo_arquivo(caminho: Path) -> tuple[int, str]:
    """(n_linhas, colunas): registros sem o cabeçalho e cabeçalho para CSV; linhas para os demais."""
    texto = _ler_texto(caminho)
    if caminho.suffix.lower() == ".csv":
        leitor = csv.reader(io.StringIO(texto, newline=""))
        cabecalho = next(leitor, [])
        return sum(1 for _ in leitor), ";".join(cabecalho)
    return len(texto.splitlines()), ""


# ---------------------------------------------------------------------------
# Lista aprovada (2026-11-10)
# ---------------------------------------------------------------------------

E4_ORDENACAO_PT = "Estudo 4, seção Ordenação na mesma carga e curva de esforço"
E4_ORDENACAO_EN = "Study 4, section on queue ordering at equal workload and effort curve"
E4_OOF_PT = "Estudo 4, seção Representação e probabilidade calibrada fora da amostra"
E4_OOF_EN = "Study 4, section on representation and out-of-fold calibrated probability"
AP_REVISOR_PT = "Apêndice do Estudo 4, seção Seleção e avaliação do revisor LLM"
AP_REVISOR_EN = "Study 4 appendix, section on LLM reviewer selection and evaluation"
AP_CASCATA_PT = "Apêndice do Estudo 4, seção Cascata sobre a base inteira (tabela da cascata v3.1)"
AP_CASCATA_EN = "Study 4 appendix, section on the cascade over the full base (v3.1 cascade table)"
AP_PROMPTS_PT = "Apêndice Material Suplementar do Framework GZ-CMD, seção dos prompts"
AP_PROMPTS_EN = "GZ-CMD supplementary appendix, prompts section"


@dataclass(frozen=True)
class Item:
    repo: str            # caminho citado pela tese, relativo à raiz
    tag: str             # prefixo curto de origem, usado só em colisão de nome
    desc_pt: str
    desc_en: str
    uso_pt: str
    uso_en: str
    metadados: tuple[str, ...] = ()  # metadado compartilhado, além de <nome>_metadata.json e <nome>_meta.json


ITENS: tuple[Item, ...] = (
    Item("data/ordenacao_bootstrap_o3_o1.csv", "data",
         "Bootstrap agrupado por óbito da diferença de revisões O3 menos O1, por meta de pares "
         "verdadeiros recuperados abaixo do corte: mediana, IC 95% e proporção de réplicas com O3 menor.",
         "Death-record clustered bootstrap of the difference in reviews O3 minus O1, by target of true "
         "pairs recovered below the cut-off: median, 95% CI and share of replicates in which O3 is lower.",
         E4_ORDENACAO_PT, E4_ORDENACAO_EN),
    Item("data/ordenacao_empates_o3.csv", "data",
         "Sensibilidade das revisões de O3 ao desempate aleatório dos blocos empatados da probabilidade "
         "isotônica, com semente fixa: valor original, mínimo, percentis e máximo por modo.",
         "Sensitivity of O3 review counts to random tie-breaking within tied isotonic probability "
         "blocks, fixed seed: original value, minimum, percentiles and maximum by mode.",
         E4_ORDENACAO_PT, E4_ORDENACAO_EN),
    Item("data/fusao_robustez_faixa.csv", "data",
         "Revisões por fila, recalculadas dentro da faixa do escore do OpenRecLink que contém os pares "
         "verdadeiros abaixo do corte, com as diferenças frente a O1 e O3.",
         "Reviews per queue recomputed within the OpenRecLink score range that contains the true pairs "
         "below the cut-off, with differences against O1 and O3.",
         E4_ORDENACAO_PT + " (tabela de robustez da fusão) e seção Limitações",
         E4_ORDENACAO_EN + " (fusion robustness table) and limitations section",
         ("data/fusao_robustez_metadata.json",)),
    Item("data/fusao_robustez_bootstrap.csv", "data",
         "Bootstrap agrupado por óbito das diferenças de revisões das fusões F3 e F4 frente a O1 e O3, "
         "por meta.",
         "Death-record clustered bootstrap of the differences in reviews of fusions F3 and F4 against "
         "O1 and O3, by target.",
         E4_ORDENACAO_PT + " (tabela de robustez da fusão)",
         E4_ORDENACAO_EN + " (fusion robustness table)",
         ("data/fusao_robustez_metadata.json",)),
    Item("data/desempenho_faixas_oof_refrec.csv", "data",
         "Desempenho da probabilidade calibrada fora da amostra (XGBoost, partições agrupadas por "
         "óbito) por faixa da nota final: AUC, precisão média, Brier, VP, FP, FN, sensibilidade e VPP.",
         "Performance of the out-of-fold calibrated probability (XGBoost, folds grouped by death "
         "record) by final score band: AUC, average precision, Brier, TP, FP, FN, sensitivity and PPV.",
         E4_OOF_PT + " (tabela de desempenho por faixa)", E4_OOF_EN + " (performance by band table)"),
    Item("data/pcal_ml_oof_grupos_metricas.csv", "data",
         "Métricas agregadas da probabilidade fora da amostra por esquema de partição e recorte: AUC, "
         "precisão média, Brier e mediana da probabilidade nos pares verdadeiros.",
         "Aggregate metrics of the out-of-fold probability by fold scheme and subset: AUC, average "
         "precision, Brier and median probability among true pairs.",
         E4_OOF_PT + " (controle da tabela por faixa)", E4_OOF_EN + " (control for the band table)"),
    Item("data/xgboost_feature_importance_global.csv", "data",
         "Importância global dos atributos do XGBoost, medida pelo ganho.",
         "Global XGBoost feature importance, measured by gain.",
         "Estudo 1 (Comparador de registros), figura de importância dos atributos",
         "Study 1 (record comparator), feature importance figure"),
    Item("data/funil_gzcmd_final.csv", "data",
         "Funil da cascata GZ-CMD v3.1 sobre a base inteira por modo, faixa da nota final e etapa: "
         "contagens, VP, FP, FN e VN, encaminhamentos ao revisor humano e custo do LLM.",
         "Funnel of the GZ-CMD v3.1 cascade over the full base by mode, final score band and stage: "
         "counts, TP, FP, FN and TN, referrals to the human reviewer and LLM cost.",
         "Estudo 4, seção Política v3.1; " + AP_CASCATA_PT,
         "Study 4, v3.1 policy section; " + AP_CASCATA_EN,
         ("data/funil_gzcmd_final_metadata.json",)),
    Item("data/funil_gzcmd_final_metricas.csv", "data",
         "Métricas globais da cascata GZ-CMD v3.1 por modo e versão.",
         "Overall metrics of the GZ-CMD v3.1 cascade by mode and version.",
         AP_CASCATA_PT + "; tabela da referência operacional",
         AP_CASCATA_EN + "; operational reference table",
         ("data/funil_gzcmd_final_metadata.json",)),
    Item("data/taxas_erro_revisor_por_banda.csv", "data",
         "Proporções de falso positivo, falso negativo e inconclusivo do revisor LLM por modo, modelo "
         "e faixa da nota final, com IC de Wilson.",
         "False positive, false negative and inconclusive proportions of the LLM reviewer by mode, "
         "model and final score band, with Wilson CI.",
         "Estudo 4, seção Revisor de referência, ancoragem e cascata na base inteira",
         "Study 4, section on the reference reviewer, anchoring and full-base cascade"),
    Item("gzcmd/error_rates_by_band_v4.yaml", "gzcmd",
         "Proporções de erro por faixa da nota final carregadas pela política v3.1; a tese usa o "
         "bloco do Claude Haiku 5.5 com instruções v4 e dossiê sem as saídas do motor.",
         "Band-specific error proportions loaded by the v3.1 policy; the thesis uses the Claude Haiku "
         "5.5 block with v4 instructions and a dossier without engine outputs.",
         "Estudo 4, seção Política v3.1; " + AP_REVISOR_PT,
         "Study 4, v3.1 policy section; " + AP_REVISOR_EN),
    Item("data/exp2_reruns_resumo.csv", "data",
         "Decisões dos revisores LLM (VP, FP, FN, VN e inconclusivos) com IC de Wilson de 95% por "
         "modo, modelo, dossiê e versão das instruções.",
         "LLM reviewer decisions (TP, FP, FN, TN and inconclusive) with 95% Wilson CI by mode, model, "
         "dossier and instruction version.",
         AP_REVISOR_PT + " (tabela das reexecuções); Estudo 4, seção Revisor de referência",
         AP_REVISOR_EN + " (reruns table); Study 4, reference reviewer section"),
    Item("data/exp2_reruns_pareado.csv", "data",
         "Comparações pareadas entre configurações do mesmo revisor: pares em comum, mudanças de "
         "decisão, teste exato de McNemar, concordância e kappa.",
         "Paired comparisons between configurations of the same reviewer: common pairs, decision "
         "changes, exact McNemar test, agreement and kappa.",
         AP_REVISOR_PT + " (complementa a tabela das reexecuções)",
         AP_REVISOR_EN + " (complements the reruns table)"),
    Item("data/exp2_reruns_por_banda.csv", "data",
         "Mesmas métricas de exp2_reruns_resumo.csv por estrato e faixa da nota final.",
         "Same metrics as exp2_reruns_resumo.csv by stratum and final score band.",
         AP_REVISOR_PT + " (tabela dos revisores)", AP_REVISOR_EN + " (reviewers table)"),
    # qw3_group_metrics.csv fica só no depósito restrito (decisão da autora de 2026-11-10):
    # tem células de 1 a 4 por subgrupo demográfico.
    Item("data/m8/perfil_vinculos.csv", "m8",
         "Perfil agregado dos vínculos por grupo e campo de comparação: frações de subescore abaixo "
         "de 0,8, iguais a zero e ausentes (auditoria da blocagem, M8).",
         "Aggregate profile of links by group and comparison field: shares of subscores below 0.8, "
         "equal to zero and missing (blocking audit, M8).",
         "Capítulo Fontes de dados e linkage de origem; Apêndice Proveniência e procedimentos do "
         "linkage de origem (contagens de data/m8/)",
         "Data sources chapter; appendix on the provenance of the source linkage (data/m8/ counts)"),
    Item("gzcmd/gzcmd_v3_llm_prompt.md", "gzcmd",
         "Arquivo-base das instruções v3 do revisor LLM.",
         "Base file of the v3 LLM reviewer instructions.",
         AP_PROMPTS_PT, AP_PROMPTS_EN),
    Item("gzcmd/gzcmd_v4_llm_prompt.md", "gzcmd",
         "Arquivo-base das instruções v4 do revisor LLM, usadas pelo revisor de referência.",
         "Base file of the v4 LLM reviewer instructions, used by the reference reviewer.",
         AP_PROMPTS_PT + "; Estudo 4, seção Revisão assistida por LLM",
         AP_PROMPTS_EN + "; Study 4, LLM-assisted review section"),
)


# ---------------------------------------------------------------------------
# Resolução das fontes, metadados e nomes
# ---------------------------------------------------------------------------

def sha256_bytes(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def copia_zenodo(repo_rel: str) -> str:
    """Caminho da cópia correspondente em data/zenodo/."""
    if repo_rel.startswith("data/qw3/"):
        return f"{ZENODO_QW3}/{repo_rel.removeprefix('data/qw3/')}"
    return f"{ZENODO_V2}/{repo_rel}"


@dataclass
class Resolucao:
    item: Item
    origem: str | None
    situacao: str
    diferenca: str = ""


def resolver(item: Item, root: Path) -> Resolucao:
    """Escolhe a cópia citada pela tese (repositório) e a compara com a de data/zenodo/."""
    repo = root / item.repo
    zen_rel = copia_zenodo(item.repo)
    zen = root / zen_rel
    if repo.is_file() and zen.is_file():
        a, b = repo.read_bytes(), zen.read_bytes()
        if a == b:
            return Resolucao(item, item.repo, f"idêntico a {zen_rel}")
        detalhe = (f"{item.repo}: {len(a)} B, sha256 {sha256_bytes(a)[:12]}; "
                   f"{zen_rel}: {len(b)} B, sha256 {sha256_bytes(b)[:12]}")
        if a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n"):
            detalhe += "; só o fim de linha difere (CRLF/LF)"
        return Resolucao(item, item.repo, f"DIFERE de {zen_rel}; usada a cópia citada pela tese", detalhe)
    if repo.is_file():
        return Resolucao(item, item.repo, f"só no repositório (sem {zen_rel})")
    if zen.is_file():
        return Resolucao(item, zen_rel, f"ausente em {item.repo}; usada {zen_rel}")
    return Resolucao(item, None, f"AUSENTE em {item.repo} e em {zen_rel}")


def candidatos_metadados(item: Item) -> list[str]:
    caminho = PurePosixPath(item.repo)
    if caminho.suffix.lower() != ".csv":
        return list(item.metadados)
    proprios = [f"{caminho.stem}_metadata.json", f"{caminho.stem}_meta.json"]
    return [str(caminho.parent / nome) for nome in proprios] + list(item.metadados)


def nomes_reservados(root: Path) -> set[str]:
    """Nomes dos arquivos já publicados na versão 1 e dos arquivos gerados aqui."""
    manifesto = json.loads((root / V1_MANIFEST).read_text(encoding="utf-8"))
    nomes = {PurePosixPath(entrada["nome"]).name for entrada in manifesto["arquivos"]}
    return nomes | {MANIFEST_NAME, README_NAME}


def nomes_planos(origens: list[tuple[str, str]], reservados: set[str]) -> dict[str, str]:
    """origem -> nome plano; em colisão, prefixa a etiqueta curta de origem."""
    base = Counter(PurePosixPath(origem).name.lower() for origem, _ in origens)
    usados = {nome.lower() for nome in reservados}
    saida: dict[str, str] = {}
    for origem, tag in origens:
        nome = PurePosixPath(origem).name
        if base[nome.lower()] > 1 or nome.lower() in usados:
            nome = f"{tag}_{nome}"
        if nome.lower() in usados:
            raise SystemExit(f"colisão de nome não resolvida pelo prefixo: {nome}")
        usados.add(nome.lower())
        saida[origem] = nome
    return saida


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------

def preparar_pasta(out: Path, root: Path) -> None:
    local = (root / ".local").resolve()
    out = out.resolve()
    if local not in out.parents:
        raise SystemExit(f"a pasta de saída tem de ficar dentro de {local}")
    out.mkdir(parents=True, exist_ok=True)
    for filho in out.iterdir():
        if filho.is_dir():
            raise SystemExit(f"subpasta inesperada em {out}: {filho.name}")
        filho.unlink()


def _linhas_tabela(entradas: list[dict], ingles: bool) -> list[str]:
    cab = ("| File | Content | Use in the thesis |" if ingles
           else "| Arquivo | Conteúdo | Uso na tese |")
    linhas = [cab, "|---|---|---|"]
    for e in entradas:
        linhas.append(f"| `{e['arquivo']}` | {e['desc_en' if ingles else 'desc_pt']} | "
                      f"{e['uso_en' if ingles else 'uso_pt']} |")
    return linhas


def texto_readme(entradas: list[dict]) -> str:
    pt = [
        "# Agregados complementares da tese: versão 2 do depósito",
        "",
        "## Português",
        "",
        f"Arquivos novos da versão 2 do depósito \"{TITULO}\" no Zenodo (versão 1: {DOI_VERSAO_1}; "
        f"DOI conceito: {DOI_CONCEITO}), em acesso aberto, licença CC-BY-4.0. Os arquivos da "
        "versão 1 continuam no depósito.",
        "",
        "Todos os arquivos são agregados (contagens, proporções, métricas, intervalos e parâmetros), "
        "metadados de geração ou textos de instruções do revisor LLM. Nenhum traz linhas por par, "
        "`pair_id`, identificadores de registro ou dados pessoais. Os dados por par permanecem no "
        f"depósito de acesso restrito {DOI_RESTRITO}, disponível mediante solicitação e acordo de uso.",
        "",
        "Siglas: O1, fila ordenada pelo escore do OpenRecLink; O3, fila ordenada pela probabilidade "
        "calibrada fora da amostra; F3 e F4, fusões de filas descritas no Estudo 4; VPP, valor "
        "preditivo positivo.",
        "",
        f"Tamanhos e SHA-256 em `{MANIFEST_NAME}`. A pasta é montada por `{SCRIPT_REL}` "
        "(https://github.com/marco-jardim/phd-research), que verifica nomes de colunas, datas "
        "completas e caminhos locais antes da cópia.",
        "",
        *_linhas_tabela(entradas, ingles=False),
    ]
    en = [
        "",
        "## English",
        "",
        f"New files for version 2 of the Zenodo deposit \"{TITULO}\" (version 1: {DOI_VERSAO_1}; "
        f"concept DOI: {DOI_CONCEITO}), open access, CC-BY-4.0 license. The version 1 files remain "
        "in the deposit.",
        "",
        "All files are aggregates (counts, proportions, metrics, intervals and parameters), generation "
        "metadata or LLM reviewer instruction texts. None contains per-pair rows, `pair_id`, record "
        "identifiers or personal data. Per-pair data remain in the restricted-access deposit "
        f"{DOI_RESTRITO}, available on request under a data use agreement.",
        "",
        "Abbreviations: O1, queue ordered by the OpenRecLink score; O3, queue ordered by the out-of-fold "
        "calibrated probability; F3 and F4, queue fusions described in Study 4; PPV, positive "
        "predictive value. Section names are given in English; the thesis is written in Portuguese.",
        "",
        f"Sizes and SHA-256 in `{MANIFEST_NAME}`. The folder is assembled by `{SCRIPT_REL}` "
        "(https://github.com/marco-jardim/phd-research), which checks column names, full dates and "
        "local paths before copying.",
        "",
        *_linhas_tabela(entradas, ingles=True),
        "",
    ]
    return "\n".join(pt + en)


def _tabela(titulo: str, cabecalho: tuple[str, ...], linhas: list[tuple[str, ...]]) -> None:
    print(f"\n== {titulo} ==")
    larguras = [max(len(str(x)) for x in col) for col in zip(cabecalho, *linhas)] if linhas else [len(c) for c in cabecalho]
    larguras = [min(w, 60) for w in larguras]
    print(" | ".join(c.ljust(w) for c, w in zip(cabecalho, larguras)))
    print("-+-".join("-" * w for w in larguras))
    for linha in linhas:
        print(" | ".join(str(c).ljust(w) for c, w in zip(linha, larguras)))


def montar(root: Path, out: Path) -> int:
    root, out = root.resolve(), out.resolve()
    falhas = 0
    resolucoes = [resolver(item, root) for item in ITENS]
    _tabela("Resolução das fontes", ("arquivo", "origem usada", "situação"),
            [(PurePosixPath(r.item.repo).name, r.origem or "-", r.situacao) for r in resolucoes])
    for r in resolucoes:
        if r.diferenca:
            print(f"  diferença ({PurePosixPath(r.item.repo).name}): {r.diferenca}")

    # Verificação das fontes da lista aprovada.
    aprovados: list[tuple[Resolucao, str]] = []
    linhas_priv = []
    for r in resolucoes:
        nome = PurePosixPath(r.item.repo).name
        if r.origem is None:
            falhas += 1
            linhas_priv.append((nome, "FAIL", "arquivo ausente"))
            continue
        problemas = checar_privacidade(root / r.origem)
        if problemas:
            falhas += 1
            linhas_priv.append((nome, "FAIL", "; ".join(problemas)))
        else:
            n, _ = resumo_arquivo(root / r.origem)
            linhas_priv.append((nome, "PASS", f"{n} linha(s)"))
            aprovados.append((r, nome))
    _tabela("Verificação de privacidade das fontes aprovadas", ("arquivo", "resultado", "detalhe"),
            linhas_priv)

    # Metadados de cada CSV aprovado.
    metadados: dict[str, list[str]] = {}
    tags: dict[str, str] = {}
    for r, nome in aprovados:
        for candidato in candidatos_metadados(r.item):
            if (root / candidato).is_file():
                metadados.setdefault(candidato, []).append(nome)
                tags.setdefault(candidato, r.item.tag)
    linhas_meta = []
    meta_incluidos: list[str] = []
    for caminho, csvs in metadados.items():
        problemas = checar_privacidade(root / caminho)
        if problemas:
            linhas_meta.append((caminho, ", ".join(csvs), "EXCLUÍDO", "; ".join(problemas)))
        else:
            linhas_meta.append((caminho, ", ".join(csvs), "INCLUÍDO", ""))
            meta_incluidos.append(caminho)
    _tabela("Metadados encontrados", ("metadados", "de", "situação", "motivo"), linhas_meta)

    # Nomes planos e cópia.
    origens = [(r.origem, r.item.tag) for r, _ in aprovados] + [(m, tags[m]) for m in meta_incluidos]
    planos = nomes_planos(origens, nomes_reservados(root))
    preparar_pasta(out, root)
    entradas_readme: list[dict] = []
    manifesto: list[dict] = []
    for r, _ in aprovados:
        entradas_readme.append({"arquivo": planos[r.origem], "desc_pt": r.item.desc_pt,
                                "desc_en": r.item.desc_en, "uso_pt": r.item.uso_pt,
                                "uso_en": r.item.uso_en})
    for m in meta_incluidos:
        csvs = ", ".join(f"`{n}`" for n in metadados[m])
        entradas_readme.append({"arquivo": planos[m],
                                "desc_pt": f"Metadados de geração e proveniência de {csvs}.",
                                "desc_en": f"Generation and provenance metadata for {csvs}.",
                                "uso_pt": "Reprodutibilidade", "uso_en": "Reproducibility"})
    for origem, nome in planos.items():
        destino = out / nome
        shutil.copyfile(root / origem, destino)
        if sha256_bytes(destino.read_bytes()) != sha256_bytes((root / origem).read_bytes()):
            raise SystemExit(f"cópia não idêntica: {nome}")
        n, colunas = resumo_arquivo(destino)
        manifesto.append({"arquivo": nome, "origem": origem, "bytes": destino.stat().st_size,
                          "sha256": sha256_bytes(destino.read_bytes()), "n_linhas": n,
                          "colunas": colunas})

    readme = texto_readme(entradas_readme)
    (out / README_NAME).write_bytes(readme.encode("utf-8"))
    n, _ = resumo_arquivo(out / README_NAME)
    manifesto.append({"arquivo": README_NAME, "origem": f"gerado por {SCRIPT_REL}",
                      "bytes": (out / README_NAME).stat().st_size,
                      "sha256": sha256_bytes((out / README_NAME).read_bytes()), "n_linhas": n,
                      "colunas": ""})
    manifesto.sort(key=lambda linha: linha["arquivo"].lower())
    with (out / MANIFEST_NAME).open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(manifesto)

    # Verificação final da pasta montada.
    linhas_final = []
    for arquivo in sorted(out.iterdir(), key=lambda p: p.name.lower()):
        problemas = checar_privacidade(arquivo)
        if arquivo.name == README_NAME and any(t in readme for t in TRAVESSOES):
            problemas.append("travessão no README")
        if problemas:
            falhas += 1
        linhas_final.append((arquivo.name, "FAIL" if problemas else "PASS", "; ".join(problemas)))
    _tabela(f"Verificação da pasta montada ({out.relative_to(root).as_posix()})",
            ("arquivo", "resultado", "detalhe"), linhas_final)
    print(f"\nArquivos na pasta: {len(linhas_final)}; falhas: {falhas}; "
          f"metadados excluídos: {len(metadados) - len(meta_incluidos)}")
    return 1 if falhas else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=None,
                        help=f"pasta de saída (padrão: {DEFAULT_OUT}, relativa à raiz)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.out if args.out is not None else root / DEFAULT_OUT
    if not out.is_absolute():
        out = root / out
    return montar(root, out)


if __name__ == "__main__":
    sys.exit(main())
