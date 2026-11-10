"""Constantes e utilitários comuns do cruzamento com as âncoras (sem rede e sem API)."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

from scripts.atestado.common import resolve_path

DEFAULT_OUT = ".local/cruzamento"
DEFAULT_ANCORAS = ".local/ancoras/ancoras_candidatas.yaml"
DEFAULT_TESE = "text/latex/tese.tex"
DEFAULT_SLIDES = "text/slides/defesa.tex"
DEFAULT_ROTEIRO = "text/slides/roteiro_defesa.md"
DEFAULT_REGISTRO = "tests/data/registro_invariantes.yaml"
DEFAULT_PINOS_INVENTARIO = ".local/fase0/inventario_pinos.csv"
MAX_PALAVRAS_FATIA = 1200
FONTES = ("tese", "slides", "roteiro")
FONTES_PADRAO = "tese"

FATIAS_NOME = "fatias.jsonl"
MAPA_NOME = "mapa.jsonl"
MAPA_OPENCODE_NOME = "mapa_opencode.jsonl"
ETAPA_MAPEAR = "mapear"
ETAPA_MAPEAR_OPENCODE = "mapear_opencode"
PARES_NOME = "pares.jsonl"
ADJUDICACOES_NOME = "adjudicacoes.jsonl"
RELATORIO_NOME = "relatorio.md"
MATRIZ_NOME = "matriz_ancoras.csv"
PINOS_NOME = "pinos.jsonl"
PINOS_META_NOME = "pinos_meta.json"
ETAPAS = (ETAPA_MAPEAR, ETAPA_MAPEAR_OPENCODE, "cruzar", "adjudicar", "pinos")
CATEGORIAS_PINO = ("frase", "negacao", "ressalva")
CONFLITOS_PINO = ("nenhum", "forte_demais", "contradiz_ancora", "cautela_solta")

TOPICOS = (
    "corte_openreclink", "O1_escore", "O2_cdr", "O3_prob_calibrada", "O4_P_recalibrada", "F4_fusao",
    "CM_contexto_obito", "fila_adaptativa", "curva_esforco_parada", "P_vigilancia", "P_confirmacao",
    "calibracao_isotonica_platt", "LLM_aceitador", "LLM_filtro", "LLM_ancoragem", "LLM_modelos",
    "registro_interno", "criterio_25", "referencia_operacional", "vies_verificacao_incorporacao",
    "vazamento_particao", "classificadores_metricas", "balanceamento", "cdr_equivalencia_java", "shap",
    "equidade_passo11", "equidade_outros", "blocagem", "subnotificacao_nao_examinados", "custos",
    "revisor_perfeito_sensibilidade_humana", "dados_coorte", "outro",
)
TOPICO_OUTRO = "outro"
RELACOES = ("afirma", "contradiz", "enfraquece", "exagera", "omite_ressalva")
RELACOES_ADJUDICAVEIS = ("contradiz", "exagera")
TIPOS_AFIRMACAO = ("resultado", "cautela", "recomendacao", "metodo", "definicao")
POLARIDADES = ("afirma", "nega", "condicional")
TIPOS_INCONSISTENCIA = (
    "numero_divergente", "polaridade_oposta", "escopo_trocado", "definicao_divergente",
    "superlativo_sem_suporte",
)
VEREDITOS = ("CONFIRMADO", "FALSO_ALARME", "DEPENDE_DE_ESCOPO")
ORIGENS_ANCORA = ("atual", "proposta")

TERMINOLOGIA = """\
- "par verdadeiro" e "não par" são a classe do par segundo a referência operacional (a revisão manual de origem); "vínculo" é o resultado aceito pela decisão de linkage.
- "referência operacional" é o gabarito do estudo (só terceiros falam em "padrão-ouro").
- grey area (área cinza) é a região de escores abaixo do corte do OpenRecLink.
- P é o arranjo sequencial: o corte do OpenRecLink é preservado e o GZ-CMD atua abaixo dele.
- O1, O2, O3, O4 e F4 são ordenações (filas de revisão); O1 é o escore, O2 a nota final do CdR, O3 a probabilidade calibrada, O4 a probabilidade recalibrada e F4 a fusão.
- PAR=1 é o par aceito pelo corte; PAR=2 é o par recuperado na revisão manual abaixo do corte.
- VPP é o valor preditivo positivo; "revisão manual" é a conferência por revisor humano; "escore" é a pontuação do OpenRecLink; "limiar" é o corte de uma ordenação; "ponto de corte" é o corte único do OpenRecLink.\
"""


# ---------------------------------------------------------------------------
# Identificadores, arquivos e texto
# ---------------------------------------------------------------------------
def id_curto(*partes: str, n: int = 16) -> str:
    """Identificador estável: início do sha256 das partes unidas por ``|``."""
    return hashlib.sha256("|".join(partes).encode("utf-8")).hexdigest()[:n]


def colapsar_espacos(texto: str) -> str:
    return " ".join(unicodedata.normalize("NFC", texto).split())


def pasta_saida(out: str | Path) -> Path:
    path = resolve_path(out)
    path.mkdir(parents=True, exist_ok=True)
    return path


def extrair_objeto(texto: str) -> dict:
    """Primeiro objeto JSON da resposta (tolera cercas de código e texto ao redor)."""
    inicio = texto.find("{")
    if inicio < 0:
        raise ValueError("sem objeto JSON")
    valor, _fim = json.JSONDecoder().raw_decode(texto[inicio:])
    if not isinstance(valor, dict):
        raise ValueError("objeto JSON inválido")
    return valor


def inteiro_ou_none(valor) -> int | None:
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def lista_de_textos(valor) -> list[str]:
    if isinstance(valor, (list, tuple)):
        return [str(item) for item in valor if str(item).strip()]
    if valor in (None, ""):
        return []
    return [str(valor)]


# ---------------------------------------------------------------------------
# Âncoras
# ---------------------------------------------------------------------------
def carregar_ancoras(caminho: Path) -> list[dict]:
    """Âncoras do YAML: lista de {id, origem, texto, ressalva?} (ou ``{ancoras: [...]}``)."""
    import yaml

    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"{caminho} não existe")
    dados = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    if isinstance(dados, dict):
        dados = dados.get("ancoras")
    if not isinstance(dados, list) or not dados:
        raise ValueError(f"{caminho}: esperada uma lista não vazia de âncoras")
    ancoras: list[dict] = []
    vistos: set[str] = set()
    for item in dados:
        if not isinstance(item, dict) or not item.get("id") or not item.get("texto"):
            raise ValueError(f"{caminho}: âncora sem id ou texto: {item!r}")
        ident = str(item["id"]).strip()
        if ident in vistos:
            raise ValueError(f"{caminho}: id de âncora repetido: {ident}")
        vistos.add(ident)
        origem = str(item.get("origem") or "atual").strip()
        if origem not in ORIGENS_ANCORA:
            raise ValueError(f"{caminho}: origem inválida em {ident}: {origem!r}")
        ancoras.append({
            "id": ident,
            "origem": origem,
            "texto": " ".join(str(item["texto"]).split()),
            "ressalva": " ".join(str(item.get("ressalva") or "").split()),
        })
    return ancoras


def bloco_ancoras(ancoras: list[dict]) -> str:
    linhas = []
    for a in ancoras:
        linha = f'- {a["id"]} ({a["origem"]}): {a["texto"]}'
        if a["ressalva"]:
            linha += f' RESSALVA: {a["ressalva"]}'
        linhas.append(linha)
    return "\n".join(linhas)


def _fonte_da_ficha(fonte) -> str:
    if not isinstance(fonte, dict):
        return "-"
    if fonte.get("derivado"):
        return f"derivado: {fonte['derivado']}"
    arquivo = fonte.get("arquivo") or "-"
    partes = [f"arquivo {arquivo}"]
    if fonte.get("chave"):
        partes.append(f"chave {fonte['chave']}")
    if fonte.get("coluna"):
        partes.append(f"coluna {fonte['coluna']}")
    filtro = fonte.get("filtro")
    if isinstance(filtro, dict) and filtro:
        partes.append("filtro " + ", ".join(f"{k}={v}" for k, v in filtro.items()))
    return "; ".join(partes)


def carregar_ficha(caminho: Path) -> str:
    """Ficha de valores conferidos: uma linha por valor das entradas numéricas (N##, tipo != afirmacao).

    Devolve ``""`` se o registro não existe. Valores sem ``texto_pt`` (auxiliares) ficam de fora.
    """
    import yaml

    caminho = Path(caminho)
    if not caminho.is_file():
        return ""
    dados = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    if not isinstance(dados, list):
        raise ValueError(f"{caminho}: esperada uma lista de entradas")
    linhas: list[str] = []
    for entrada in dados:
        if not isinstance(entrada, dict):
            continue
        ident = str(entrada.get("id") or "").strip()
        if not ident.startswith("N") or entrada.get("tipo") == "afirmacao":
            continue
        descricao = " ".join(str(entrada.get("descricao") or "").split())
        for valor in entrada.get("valores") or []:
            if not isinstance(valor, dict) or valor.get("texto_pt") in (None, ""):
                continue
            linhas.append(
                f'- {ident} ({descricao}) | {valor.get("nome")} = {valor["texto_pt"]} | '
                f'{_fonte_da_ficha(valor.get("fonte"))}'
            )
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Fatias: numeração de linhas e verificação literal de trechos
# ---------------------------------------------------------------------------
def numerar(fatia: dict) -> str:
    """Texto da fatia com o número da linha do arquivo em cada linha não vazia."""
    saida = []
    for numero, linha in zip(fatia["linhas"], fatia["texto"].split("\n")):
        saida.append(f"{numero:>5}| {linha}" if linha.strip() else "")
    return "\n".join(saida)


def indexar_fatia(fatia: dict) -> tuple[str, list[int]]:
    """Texto com espaços colapsados e, para cada caractere, a linha do arquivo de origem."""
    texto = unicodedata.normalize("NFC", fatia["texto"])
    numeros = fatia["linhas"]
    saida: list[str] = []
    linhas: list[int] = []
    espaco_antes = True
    for i, linha in enumerate(texto.split("\n")):
        numero = numeros[i] if i < len(numeros) else 0
        for ch in linha + "\n":
            if ch.isspace():
                if not espaco_antes:
                    saida.append(" ")
                    linhas.append(numero)
                espaco_antes = True
            else:
                saida.append(ch)
                linhas.append(numero)
                espaco_antes = False
    return "".join(saida), linhas


def localizar_trecho(indice: tuple[str, list[int]], trecho: str) -> int | None:
    """Linha em que ``trecho`` começa se existir literalmente na fatia (espaços normalizados)."""
    alvo = colapsar_espacos(trecho)
    if not alvo:
        return None
    pos = indice[0].find(alvo)
    if pos < 0:
        return None
    return indice[1][pos] or None


def ler_jsonl(caminho: Path) -> list[dict]:
    from scripts.atestado.common import read_jsonl

    return read_jsonl(caminho)


def formatar_milhar(n: int) -> str:
    return f"{n:,}".replace(",", ".")
