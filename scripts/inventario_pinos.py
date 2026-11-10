#!/usr/bin/env python
"""Inventário determinístico dos testes que fixam (pinam) prosa da tese.

Lê ``tests/*.py`` com ``ast`` (somente biblioteca padrão) e lista cada asserção
cujo operando é um literal de texto (ou tupla/lista de literais, ou f-string só
com partes literais) comparado a texto lido de arquivos da tese
(``text/latex/**/*.tex``, ``text/slides/*``, ``roteiro_defesa.md``).

Saídas (em ``.local/fase0/``):
  - ``inventario_pinos.csv``: UTF-8 com BOM, separador ``;``, uma linha por pino.
  - ``inventario_pinos.md``: resumo por arquivo, tipo, categoria e proposta.

Uso::

    python scripts/inventario_pinos.py [--tests DIR] [--out DIR]

A saída não contém data/hora: duas execuções sobre o mesmo código geram arquivos
idênticos. Nenhum teste nem outro arquivo é modificado.

Como a origem do texto é determinada (melhor esforço, por módulo):
  1. constantes de caminho (``Path(...)``, ``ROOT / "text/latex"`` etc.) e
     chamadas ``read_text``/``read_bytes``/``open``/``read`` sobre caminhos da
     tese marcam variáveis como "texto da tese" (propagado por atribuições,
     laços, ``with`` e compreensões);
  2. funções auxiliares do módulo que leem a tese (por exemplo ``text(name)``)
     passam a valer como leitores, inclusive quando importadas de outro teste;
  3. fixtures leitoras (módulo ou ``conftest.py``) marcam parâmetros de teste;
  4. quando nada disso se aplica, mas o módulo cita arquivos da tese e o
     operando tem nome típico de texto (``texto``, ``t``, ``source``...), a
     linha entra com ``deteccao = inferida``.

Colunas extras além das pedidas: ``linha_assert``, ``detalhe`` e ``deteccao``.
Para comparações sem literal (igualdade com saída regenerada, bytes de tabela),
``literal`` traz ``[expr] ...`` e ``categoria`` vale ``estrutural``.
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTS = ROOT / "tests"
DEFAULT_OUT = ROOT / ".local" / "fase0"

# Indícios de que uma string aponta para arquivos da tese.
PATH_IND = re.compile(
    r"(?:^|[/\\])(?:latex|slides)(?:[/\\]|$)|(?:^|[/\\])text[/\\]|\.tex\b|roteiro"
)
FILE_RE = re.compile(r"^[\w.*/\\-]+\.(?:tex|md)$")

READ_ATTRS = {"read_text", "read_bytes", "read", "readlines", "readline"}
RE_FUNCS = {"search", "match", "fullmatch", "findall", "finditer"}
STR_METHODS = {"startswith", "endswith", "find", "rfind", "index", "rindex"}
SKIP_TAINTED_CALLS = {"len", "sum", "int", "float", "bool", "round", "max", "min"}
TEXTISH = {
    "texto", "text", "tex", "t", "source", "src", "conteudo", "content", "tese",
    "body", "latex", "slides", "roteiro", "tabela", "table", "doc", "corpus",
    "normalized", "normalizado", "txt",
}

TERMOS = (
    "área cinza", "vínculo", "vínculos", "não vínculo", "revisão clerical",
    "sinan", "sih-sus", "precisão", "relacionamento de registros",
    "vinculação", "zona cinzenta",
)
HEDGE = ("não se demonstrou", "post hoc", "exploratóri", "hipótese", "a validar")
NEG_RE = re.compile(r"\b(?:não|nem|sem|nunca|nenhum\w*)\b")
BIBKEY_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:19|20)\d{2}[A-Za-z]*\b")
LATEX_ARG_RE = re.compile(r"\\(?:ref|label|input|path|eqref|url|texttt)\{[^}]*\}")
ANCHOR_SKIP_RE = re.compile(r"^[\w-]{1,12}$")

OPSYM = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">",
    ast.GtE: ">=",
}
OPFLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}

COLUMNS = [
    "arquivo_teste", "funcao", "linha", "tipo", "literal", "alvo", "categoria",
    "afetado_pela_terminologia", "proposta_migracao", "linha_assert", "detalhe",
    "deteccao",
]


def unp(node):
    """Texto do nó; ``ast.unparse`` só existe a partir do Python 3.9."""
    fn = getattr(ast, "unparse", None)
    return fn(node) if fn else ast.dump(node)


def target_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out = []
        for elt in target.elts:
            out.extend(target_names(elt))
        return out
    if isinstance(target, ast.Starred):
        return target_names(target.value)
    return []


def unwrap_param(node):
    """``pytest.param(a, b)`` vira a tupla ``(a, b)``."""
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "param" and node.args:
            if len(node.args) == 1:
                return node.args[0]
            return ast.Tuple(elts=list(node.args), ctx=ast.Load())
    return node


# --------------------------------------------------------------------------
# Classificação das linhas
# --------------------------------------------------------------------------

def terminologia_afetada(literal):
    low = unicodedata.normalize("NFC", literal).lower()
    return [term for term in TERMOS if term in low]


def classificar_categoria(literal, tipo):
    text = unicodedata.normalize("NFC", literal)
    if text.startswith("[expr] "):
        return "estrutural"
    low = text.lower()
    if re.search(r"\\cite\w*", text) or BIBKEY_RE.search(text):
        return "citacao"
    if any(h in low for h in HEDGE):
        return "ressalva"
    if tipo == "regex":
        body = re.sub(r"\\[A-Za-z]|\{\d+(?:,\d*)?\}", "", text)
    else:
        body = LATEX_ARG_RE.sub("", text)
    if re.search(r"\d|%", body):
        return "numero"
    if NEG_RE.search(low):
        return "negacao"
    if len(text.split()) <= 3:
        return "termo"
    return "frase"


def classificar_proposta(tipo, categoria, detalhe):
    if tipo == "bytes_tabela":
        return "regenerar_tabela"
    if categoria == "estrutural" or detalhe.startswith("âncora"):
        return "manter"
    if tipo in ("ausencia", "regex"):
        return "regra_linter"
    if categoria == "numero":
        return "registro_numero"
    if tipo == "contagem" or categoria == "termo":
        return "regra_linter"
    return "registro_afirmacao"


def formatar_literal(value):
    value = value.replace("\r", "\\r").replace("\n", "\\n")
    if len(value) > 300:
        value = value[:297] + "..."
    return value


def formatar_alvo(targets, deteccao, module_targets):
    if deteccao == "inferida":
        base = sorted(module_targets)
        prefix = "(inferido) "
    else:
        base = sorted(targets)
        prefix = ""
    files = [t for t in base if FILE_RE.match(t)]
    chosen = files or base
    if not chosen:
        return prefix.strip() if prefix else "(dinâmico)"
    text = "|".join(chosen[:6])
    if len(chosen) > 6:
        text += "|+%d" % (len(chosen) - 6)
    return prefix + text


class Scope:
    """Estado de uma função (ou do módulo): taint, ligações e constantes."""

    def __init__(self, func="<module>"):
        self.func = func
        self.taint = {}      # nome -> (tipo, alvos); tipo em {"text", "path"}
        self.env = {}        # nome -> lista de nós possíveis (laço, parametrize)
        self.consts = {}     # nome -> nó de valor literal
        self.compiled = {}   # nome -> nó do padrão de re.compile

    def copy(self, func=None):
        other = Scope(func or self.func)
        other.taint = dict(self.taint)
        other.env = dict(self.env)
        other.consts = dict(self.consts)
        other.compiled = dict(self.compiled)
        return other


class Analyzer:
    def __init__(self, path, tree, global_fixtures):
        self.path = path
        self.name = path.stem
        self.fname = path.name
        self.tree = tree
        self.consts = {}
        self.imports = {}
        for st in tree.body:
            if isinstance(st, ast.Assign):
                for tgt in st.targets:
                    if isinstance(tgt, ast.Name):
                        self.consts[tgt.id] = st.value
            elif isinstance(st, ast.AnnAssign) and st.value is not None:
                if isinstance(st.target, ast.Name):
                    self.consts[st.target.id] = st.value
            elif isinstance(st, ast.ImportFrom):
                mod = (st.module or "").split(".")[-1]
                for alias in st.names:
                    self.imports[alias.asname or alias.name] = (mod, alias.name)
        strings = [
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        self.has_indicator = any(PATH_IND.search(s) for s in strings)
        self.module_targets = sorted({
            s for s in strings if FILE_RE.match(s) and PATH_IND.search(s)
        })
        self.readers = {}          # nome -> (n_parametros, alvos)
        self.local_readers = set()
        self.fixtures = dict(global_fixtures)
        self.local_fixtures = {}
        self.rows = []
        self.seen = set()
        self.dry = False
        self._flag = False
        self._dry_targets = set()
        self._handled = set()
        self.cur_line = 0

    # ------------------------------------------------------------------
    # Resolução de literais
    # ------------------------------------------------------------------
    def resolve(self, name, sc):
        if name in sc.env:
            return sc.env[name]
        node = sc.consts.get(name)
        return [node] if node is not None else []

    def lits(self, node, sc, depth=0):
        """Lista de ``(texto, linha)`` se o nó é literal de texto; senão ``[]``."""
        if depth > 6 or node is None:
            return []
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return [(node.value, node.lineno)]
            return []
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                elif isinstance(value, ast.FormattedValue) and value.format_spec is None:
                    inner = self.lits(value.value, sc, depth + 1)
                    if len(inner) != 1:
                        return []
                    parts.append(inner[0][0])
                else:
                    return []
            return [("".join(parts), node.lineno)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.lits(node.left, sc, depth + 1)
            right = self.lits(node.right, sc, depth + 1)
            if len(left) == 1 and len(right) == 1:
                return [(left[0][0] + right[0][0], node.lineno)]
            return []
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            out = []
            for elt in node.elts:
                out.extend(self.lits(elt, sc, depth + 1))
            return out
        if isinstance(node, ast.Starred):
            return self.lits(node.value, sc, depth + 1)
        if isinstance(node, ast.Name):
            out = []
            for sub in self.resolve(node.id, sc):
                out.extend(self.lits(sub, sc, depth + 1))
            return out
        return []

    def dict_node(self, node, sc):
        for _ in range(5):
            if isinstance(node, ast.Dict):
                return node
            if isinstance(node, ast.Name):
                nodes = self.resolve(node.id, sc)
                if len(nodes) == 1:
                    node = nodes[0]
                    continue
            break
        return None

    def iter_items(self, node, sc, depth=0):
        """Nós-elemento de um iterável literal, ou ``None`` se não resolvível."""
        if depth > 6 or node is None:
            return None
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            out = []
            for elt in node.elts:
                if isinstance(elt, ast.Starred):
                    sub = self.iter_items(elt.value, sc, depth + 1)
                    if sub:
                        out.extend(sub)
                else:
                    out.append(elt)
            return out
        if isinstance(node, ast.Dict):
            return [k for k in node.keys if k is not None]
        if isinstance(node, ast.Name):
            out, ok = [], False
            for sub in self.resolve(node.id, sc):
                items = self.iter_items(sub, sc, depth + 1)
                if items is not None:
                    out.extend(items)
                    ok = True
            return out if ok else None
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and node.args and func.id in (
                "sorted", "list", "tuple", "set", "reversed", "frozenset"
            ):
                return self.iter_items(node.args[0], sc, depth + 1)
            if isinstance(func, ast.Attribute) and not node.args:
                if func.attr in ("keys", "values"):
                    d = self.dict_node(func.value, sc)
                    if d is not None:
                        if func.attr == "keys":
                            return [k for k in d.keys if k is not None]
                        return list(d.values)
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.iter_items(node.left, sc, depth + 1)
            right = self.iter_items(node.right, sc, depth + 1)
            if left is not None and right is not None:
                return left + right
        return None

    def bind_pattern(self, target, nodes, env):
        if isinstance(target, ast.Name):
            if nodes:
                env[target.id] = list(nodes)
            else:
                env.pop(target.id, None)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, sub_target in enumerate(target.elts):
                if isinstance(sub_target, ast.Starred):
                    continue
                sub = [
                    e.elts[i] for e in nodes
                    if isinstance(e, (ast.Tuple, ast.List)) and len(e.elts) > i
                ]
                self.bind_pattern(sub_target, sub, env)

    def bind_for(self, target, it, sc):
        for name in target_names(target):
            sc.env.pop(name, None)
        if (isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute)
                and it.func.attr == "items" and not it.args):
            d = self.dict_node(it.func.value, sc)
            if (d is not None and isinstance(target, (ast.Tuple, ast.List))
                    and len(target.elts) == 2):
                pairs = [(k, v) for k, v in zip(d.keys, d.values) if k is not None]
                self.bind_pattern(target.elts[0], [k for k, _ in pairs], sc.env)
                self.bind_pattern(target.elts[1], [v for _, v in pairs], sc.env)
            return
        if (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                and it.func.id == "enumerate" and it.args
                and isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2):
            items = self.iter_items(it.args[0], sc)
            if items:
                self.bind_pattern(target.elts[1], [unwrap_param(i) for i in items], sc.env)
            return
        items = self.iter_items(it, sc)
        if items:
            self.bind_pattern(target, [unwrap_param(i) for i in items], sc.env)

    def bind_parametrize(self, dec, sc):
        if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                and dec.func.attr == "parametrize"):
            return
        kw = {k.arg: k.value for k in dec.keywords}
        names_node = dec.args[0] if dec.args else kw.get("argnames")
        vals_node = dec.args[1] if len(dec.args) > 1 else kw.get("argvalues")
        if names_node is None or vals_node is None:
            return
        names = []
        if isinstance(names_node, ast.Constant) and isinstance(names_node.value, str):
            names = [s.strip() for s in names_node.value.split(",") if s.strip()]
        elif isinstance(names_node, (ast.Tuple, ast.List)):
            names = [
                c.value for c in names_node.elts
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            ]
        items = self.iter_items(vals_node, sc)
        if not items or not names:
            return
        items = [unwrap_param(i) for i in items]
        if len(names) == 1:
            sc.env[names[0]] = items
        else:
            pattern = ast.Tuple(
                elts=[ast.Name(id=n, ctx=ast.Store()) for n in names], ctx=ast.Store()
            )
            self.bind_pattern(pattern, items, sc.env)

    def collection_literal(self, node, sc, depth=0):
        """Nó de coleção literal não vazia a que ``node`` se refere, se houver."""
        if depth > 5 or node is None:
            return None
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return node if node.elts else None
        if isinstance(node, ast.Dict):
            return node if node.keys else None
        if isinstance(node, ast.Name):
            nodes = self.resolve(node.id, sc)
            if len(nodes) == 1:
                return self.collection_literal(nodes[0], sc, depth + 1)
            return None
        if isinstance(node, ast.Subscript):
            return self.collection_literal(node.value, sc, depth + 1)
        return None

    # ------------------------------------------------------------------
    # Origem do texto (taint)
    # ------------------------------------------------------------------
    def mark(self, targets):
        self._flag = True
        self._dry_targets |= set(targets)

    def call_hit(self, call, sc):
        """Alvos se a chamada lê a tese (leitor, read_*/open); senão ``None``."""
        func = call.func
        if isinstance(func, ast.Name) and func.id in self.readers:
            targets = set(self.readers[func.id][1])
            for arg in list(call.args) + [k.value for k in call.keywords]:
                res = self.scan(arg, sc)
                if res is not None:
                    targets |= res[1]
            self.mark(targets)
            return targets
        if isinstance(func, ast.Name) and func.id == "open":
            subjects = list(call.args[:1])
        elif isinstance(func, ast.Attribute) and func.attr in READ_ATTRS:
            subjects = [func.value]
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            subjects = [func.value] + list(call.args[:1])
        else:
            return None
        targets, hit = set(), False
        for subject in subjects:
            res = self.scan(subject, sc)
            if res is not None:
                hit = True
                targets |= res[1]
        if not hit:
            return None
        self.mark(targets)
        return targets

    def scan(self, node, sc):
        """``("text"|"path", alvos)`` ou ``None``."""
        tmp = any(
            isinstance(n, ast.Name) and n.id.lower().startswith("tmp")
            for n in ast.walk(node)
        )
        text_hit = path_hit = False
        text_t, path_t = set(), set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                info = sc.taint.get(sub.id)
                if info:
                    if info[0] == "text":
                        text_hit = True
                        text_t |= info[1]
                    else:
                        path_hit = True
                        path_t |= info[1]
                elif not tmp:
                    for nd in sc.env.get(sub.id, []):
                        for val, _ in self.lits(nd, sc):
                            if FILE_RE.match(val):
                                path_hit = True
                                path_t.add(val)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                if not tmp and PATH_IND.search(sub.value):
                    path_hit = True
                    path_t.add(sub.value)
            elif isinstance(sub, ast.Call):
                hit = self.call_hit(sub, sc)
                if hit is not None:
                    text_hit = True
                    text_t |= hit
        if text_hit:
            return ("text", text_t)
        if path_hit:
            return ("path", path_t)
        return None

    def text_of(self, node, sc):
        """``(alvos, deteccao)`` se ``node`` é texto da tese; senão ``None``."""
        res = self.scan(node, sc)
        if res is not None and res[0] == "text":
            return (res[1], "direta")
        if self.has_indicator:
            base = node
            while True:
                if isinstance(base, ast.Attribute):
                    base = base.value
                elif isinstance(base, ast.Call):
                    base = base.func
                elif isinstance(base, ast.Subscript):
                    base = base.value
                else:
                    break
            if (isinstance(base, ast.Name) and base.id in TEXTISH
                    and base.id not in sc.taint):
                return (set(), "inferida")
        return None

    # ------------------------------------------------------------------
    # Emissão
    # ------------------------------------------------------------------
    def emit(self, sc, tipo, literal, lit_line, tinfo, detalhe=""):
        if self.dry:
            return
        targets, deteccao = tinfo
        key = (sc.func, self.cur_line, lit_line, literal, tipo, detalhe)
        if key in self.seen:
            return
        self.seen.add(key)
        categoria = classificar_categoria(literal, tipo)
        termos = terminologia_afetada(literal)
        self.rows.append({
            "_termos": termos,
            "arquivo_teste": self.fname,
            "funcao": sc.func,
            "linha": lit_line,
            "tipo": tipo,
            "literal": formatar_literal(literal),
            "alvo": formatar_alvo(targets, deteccao, self.module_targets),
            "categoria": categoria,
            "afetado_pela_terminologia": "sim" if termos else "nao",
            "proposta_migracao": classificar_proposta(tipo, categoria, detalhe),
            "linha_assert": self.cur_line,
            "detalhe": detalhe,
            "deteccao": deteccao,
        })

    # ------------------------------------------------------------------
    # Percurso de instruções
    # ------------------------------------------------------------------
    def assign(self, targets, value, sc):
        info = self.scan(value, sc)
        is_literal = isinstance(
            value, (ast.Constant, ast.Tuple, ast.List, ast.Set, ast.Dict,
                    ast.JoinedStr, ast.BinOp)
        )
        compiled = None
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and value.func.attr == "compile" and isinstance(value.func.value, ast.Name)
                and value.func.value.id == "re" and value.args):
            compiled = value.args[0]
        for target in targets:
            for name in target_names(target):
                sc.env.pop(name, None)
                if info is not None:
                    sc.taint[name] = info
                else:
                    sc.taint.pop(name, None)
                if isinstance(target, ast.Name) and is_literal:
                    sc.consts[name] = value
                else:
                    sc.consts.pop(name, None)
                if isinstance(target, ast.Name) and compiled is not None:
                    sc.compiled[name] = compiled
                else:
                    sc.compiled.pop(name, None)

    def run_body(self, body, sc):
        for st in body:
            self.run_stmt(st, sc)

    def run_stmt(self, st, sc):
        self.cur_line = st.lineno
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.run_function(st, sc)
        elif isinstance(st, ast.ClassDef):
            self.run_body(st.body, sc.copy())
        elif isinstance(st, ast.Assert):
            self.visit(st.test, sc, False, True)
        elif isinstance(st, ast.Assign):
            self.visit(st.value, sc, False, False)
            self.assign(st.targets, st.value, sc)
        elif isinstance(st, ast.AnnAssign):
            if st.value is not None:
                self.visit(st.value, sc, False, False)
                self.assign([st.target], st.value, sc)
        elif isinstance(st, ast.AugAssign):
            self.visit(st.value, sc, False, False)
            info = self.scan(st.value, sc)
            if info is not None and isinstance(st.target, ast.Name):
                old = sc.taint.get(st.target.id)
                kind = "text" if "text" in (info[0], old[0] if old else "") else info[0]
                sc.taint[st.target.id] = (kind, info[1] | (old[1] if old else set()))
        elif isinstance(st, (ast.For, ast.AsyncFor)):
            self.visit(st.iter, sc, False, False)
            info = self.scan(st.iter, sc)
            for name in target_names(st.target):
                if info is not None:
                    sc.taint[name] = info
                else:
                    sc.taint.pop(name, None)
            self.bind_for(st.target, st.iter, sc)
            self.run_body(st.body, sc)
            self.run_body(st.orelse, sc)
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            for item in st.items:
                self.visit(item.context_expr, sc, False, False)
                if item.optional_vars is not None:
                    info = self.scan(item.context_expr, sc)
                    for name in target_names(item.optional_vars):
                        if info is not None:
                            sc.taint[name] = info
                        else:
                            sc.taint.pop(name, None)
            self.run_body(st.body, sc)
        elif isinstance(st, (ast.If, ast.While)):
            self.visit(st.test, sc, False, False)
            self.run_body(st.body, sc)
            self.run_body(st.orelse, sc)
        elif isinstance(st, ast.Try):
            self.run_body(st.body, sc)
            for handler in st.handlers:
                self.run_body(handler.body, sc)
            self.run_body(st.orelse, sc)
            self.run_body(st.finalbody, sc)
        elif isinstance(st, (ast.Expr, ast.Return)):
            if st.value is not None:
                self.visit(st.value, sc, False, False)

    def run_function(self, fn, parent):
        sc = parent.copy(fn.name)
        args = fn.args
        for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            if arg.arg in self.fixtures:
                sc.taint[arg.arg] = ("text", set(self.fixtures[arg.arg]))
            sc.env.pop(arg.arg, None)
            sc.consts.pop(arg.arg, None)
        for dec in fn.decorator_list:
            self.bind_parametrize(dec, sc)
        self.run_body(fn.body, sc)

    # ------------------------------------------------------------------
    # Leitores e fixtures
    # ------------------------------------------------------------------
    def module_scope(self):
        sc = Scope()
        sc.consts = dict(self.consts)
        for st in self.tree.body:
            if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.run_stmt(st, sc)
        return sc

    def detect_readers(self, registry):
        for alias, (mod, orig) in self.imports.items():
            info = registry.get(mod, {}).get(orig)
            if info is not None and alias not in self.readers:
                self.readers[alias] = info
        self.dry = True
        try:
            base = self.module_scope()
            changed, rounds = True, 0
            while changed and rounds < 4:
                changed, rounds = False, rounds + 1
                for st in self.tree.body:
                    if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if st.name.startswith("test") or st.name in self.readers:
                        continue
                    self._flag = False
                    self._dry_targets = set()
                    self.run_function(st, base)
                    if self._flag:
                        n_params = len(st.args.posonlyargs) + len(st.args.args)
                        self.readers[st.name] = (n_params, frozenset(self._dry_targets))
                        self.local_readers.add(st.name)
                        changed = True
                        if any("fixture" in unp(d) for d in st.decorator_list):
                            self.local_fixtures[st.name] = set(self._dry_targets)
        finally:
            self.dry = False
        self.fixtures.update(self.local_fixtures)
        registry[self.name] = {
            k: v for k, v in self.readers.items() if k in self.local_readers
        }

    def run(self):
        self.rows, self.seen = [], set()
        self._handled = set()
        sc = Scope()
        sc.consts = dict(self.consts)
        for st in self.tree.body:
            if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.run_stmt(st, sc)
        for st in self.tree.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.run_stmt(st, sc)
        return self.rows

    # ------------------------------------------------------------------
    # Detecção de pinos em expressões
    # ------------------------------------------------------------------
    def visit(self, node, sc, neg, full):
        """Percorre ``node``; ``full`` liga comparações (só em ``assert``)."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            self.visit(node.operand, sc, not neg, full)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            inner = sc.copy()
            for gen in node.generators:
                self.visit(gen.iter, inner, neg, full)
                info = self.scan(gen.iter, inner)
                for name in target_names(gen.target):
                    if info is not None:
                        inner.taint[name] = info
                    else:
                        inner.taint.pop(name, None)
                self.bind_for(gen.target, gen.iter, inner)
                for cond in gen.ifs:
                    self.visit(cond, inner, neg, full)
            elts = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
            for elt in elts:
                self.visit(elt, inner, neg, full)
            return
        if isinstance(node, ast.Compare) and full:
            self.on_compare(node, sc, neg)
        if isinstance(node, ast.Call):
            self.on_call(node, sc, neg, full)
        for child in ast.iter_child_nodes(node):
            self.visit(child, sc, neg, full)

    def count_call(self, node, sc):
        """``(tinfo, literais)`` se ``node`` é ``<texto>.count(<literal>)``."""
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "count" and node.args):
            tinfo = self.text_of(node.func.value, sc)
            lits = self.lits(node.args[0], sc)
            if tinfo is not None and lits:
                return tinfo, lits
        return None

    def on_compare(self, node, sc, neg):
        operands = [node.left] + list(node.comparators)
        for i, op in enumerate(node.ops):
            left, right = operands[i], operands[i + 1]
            if isinstance(op, (ast.In, ast.NotIn)):
                tr = self.text_of(right, sc)
                ll = self.lits(left, sc)
                if tr is not None and ll:
                    presence = isinstance(op, ast.In) != neg
                    tipo = "presenca" if presence else "ausencia"
                    for value, line in ll:
                        self.emit(sc, tipo, value, line, tr)
                    continue
                tl = self.text_of(left, sc)
                rl = self.lits(right, sc)
                if tl is not None and rl and not isinstance(right, ast.Constant):
                    for value, line in rl:
                        self.emit(sc, "outro", value, line, tl, "texto pertence a conjunto literal")
                continue
            sym = OPSYM.get(type(op))
            if sym is None:
                continue
            done = False
            for call_side, other, flipped in ((left, right, False), (right, left, True)):
                found = self.count_call(call_side, sc)
                if found is not None:
                    tinfo, lits = found
                    self._handled.add(id(call_side))
                    shown = OPFLIP[sym] if flipped else sym
                    detalhe = "count %s %s" % (shown, unp(other)[:60])
                    for value, line in lits:
                        self.emit(sc, "contagem", value, line, tinfo, detalhe)
                    done = True
                    break
            if done or not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            self.on_equality(node, left, right, sc)

    def on_equality(self, node, left, right, sc):
        tl, tr = self.text_of(left, sc), self.text_of(right, sc)
        ll, rl = self.lits(left, sc), self.lits(right, sc)
        if tl is not None and rl:
            for value, line in rl:
                self.emit(sc, "outro", value, line, tl, "igualdade com literal")
            return
        if tr is not None and ll:
            for value, line in ll:
                self.emit(sc, "outro", value, line, tr, "igualdade com literal")
            return
        if tl is None and tr is None:
            return
        tinfo = tl if tl is not None else tr
        side = left if tl is not None else right
        other = right if tl is not None else left
        for expr in (side, other):
            if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
                    and expr.func.id in SKIP_TAINTED_CALLS):
                return
        coll = self.collection_literal(other, sc)
        if coll is not None:
            self.emit(sc, "outro", unp(coll), getattr(coll, "lineno", node.lineno), tinfo,
                      "igualdade com coleção literal")
            return
        if isinstance(other, ast.Constant):
            return
        targets = set(tinfo[0])
        for expr in (left, right):
            res = self.scan(expr, sc)
            if res is not None:
                targets |= res[1]
        tables = any(
            "tables" in t.lower() or re.search(r"(?:^|[/\\])tab_", t) for t in targets
        )
        tipo = "bytes_tabela" if tables else "outro"
        self.emit(sc, tipo, "[expr] " + unp(node), node.lineno, (targets, tinfo[1]),
                  "igualdade com conteúdo regenerado")

    def on_call(self, node, sc, neg, full):
        func = node.func
        self.re_call(node, sc, neg)
        if isinstance(func, ast.Name) and func.id in self.readers:
            n_params = self.readers[func.id][0]
            if n_params >= 2:
                targets = set(self.readers[func.id][1])
                for extra in node.args[1:]:
                    for value, line in self.lits(extra, sc):
                        if PATH_IND.search(value) or FILE_RE.match(value):
                            continue
                        if ANCHOR_SKIP_RE.match(value):
                            continue
                        self.emit(sc, "outro", value, line, (targets, "direta"),
                                  "âncora via helper %s" % func.id)
            return
        if not full or id(node) in self._handled:
            return
        if isinstance(func, ast.Attribute) and node.args:
            if func.attr == "count" or func.attr in STR_METHODS:
                tinfo = self.text_of(func.value, sc)
                lits = self.lits(node.args[0], sc)
                if tinfo is not None and lits:
                    tipo = "contagem" if func.attr == "count" else "outro"
                    for value, line in lits:
                        self.emit(sc, tipo, value, line, tinfo, func.attr)

    def compiled_pattern(self, node, sc):
        if isinstance(node, ast.Name):
            return sc.compiled.get(node.id)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "compile" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "re" and node.args):
            return node.args[0]
        return None

    def re_call(self, node, sc, neg):
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in RE_FUNCS:
            return
        if isinstance(func.value, ast.Name) and func.value.id == "re":
            if len(node.args) < 2:
                return
            pattern, subject = node.args[0], node.args[1]
        else:
            pattern = self.compiled_pattern(func.value, sc)
            if pattern is None or not node.args:
                return
            subject = node.args[0]
        tinfo = self.text_of(subject, sc)
        lits = self.lits(pattern, sc)
        if tinfo is None or not lits:
            return
        detalhe = "re.%s%s" % (func.attr, " (negado)" if neg else "")
        for value, line in lits:
            self.emit(sc, "regex", value, line, tinfo, detalhe)



# --------------------------------------------------------------------------
# Execução e saídas
# --------------------------------------------------------------------------

def coletar(tests_dir):
    """Analisa todos os ``*.py`` de ``tests_dir``; devolve linhas e avisos."""
    analyzers, avisos = [], []
    global_fixtures = {}
    for path in sorted(tests_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            avisos.append("%s: não analisado (%s)" % (path.name, exc.__class__.__name__))
            continue
        analyzers.append(Analyzer(path, tree, global_fixtures))
    registry = {}
    for _ in range(3):
        for analyzer in analyzers:
            analyzer.readers = {
                k: v for k, v in analyzer.readers.items() if k in analyzer.local_readers
            }
            analyzer.detect_readers(registry)
            if analyzer.name == "conftest":
                global_fixtures.update(analyzer.local_fixtures)
        for analyzer in analyzers:
            analyzer.fixtures.update(global_fixtures)
    rows = []
    for analyzer in analyzers:
        rows.extend(analyzer.run())
    rows.sort(key=lambda r: (
        r["arquivo_teste"], r["linha"], r["funcao"], r["tipo"], r["literal"],
        r["linha_assert"], r["detalhe"],
    ))
    return rows, avisos, len(analyzers)


def escrever_csv(rows, path):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=COLUMNS, delimiter=";", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def tabela_md(titulo, counter, cabecalho, ordem=None):
    linhas = ["### " + titulo, "", "| %s | pinos |" % cabecalho, "|---|---:|"]
    chaves = ordem if ordem is not None else sorted(counter, key=lambda k: (-counter[k], k))
    for chave in chaves:
        linhas.append("| %s | %d |" % (chave, counter.get(chave, 0)))
    linhas.append("| **total** | **%d** |" % sum(counter.values()))
    linhas.append("")
    return linhas


def escrever_md(rows, avisos, n_modulos, path):
    total = len(rows)
    por_arquivo = Counter(r["arquivo_teste"] for r in rows)
    afetados = [r for r in rows if r["afetado_pela_terminologia"] == "sim"]
    afet_arquivo = Counter(r["arquivo_teste"] for r in afetados)
    termos = Counter()
    for r in afetados:
        for termo in r["_termos"]:
            termos[termo] += 1
    cruzado = Counter((r["tipo"], r["proposta_migracao"]) for r in rows)
    out = [
        "# Inventário de testes que fixam prosa da tese",
        "",
        "Gerado por `scripts/inventario_pinos.py` (AST, biblioteca padrão). "
        "Detalhe linha a linha em `inventario_pinos.csv` (UTF-8 com BOM, `;`).",
        "",
        "- Módulos de `tests/` analisados: %d" % n_modulos,
        "- Módulos com ao menos um pino: %d" % len(por_arquivo),
        "- Pinos (linhas do CSV): %d" % total,
        "- Literais distintos: %d" % len({r["literal"] for r in rows}),
        "- Pinos afetados pelas decisões de terminologia: %d" % len(afetados),
        "",
    ]
    if avisos:
        out += ["Avisos:", ""] + ["- " + a for a in avisos] + [""]
    out += ["## Contagens", ""]
    out += tabela_md("Por arquivo de teste", por_arquivo, "arquivo")
    out += tabela_md("Por tipo", Counter(r["tipo"] for r in rows), "tipo")
    out += tabela_md("Por categoria", Counter(r["categoria"] for r in rows), "categoria")
    out += tabela_md("Por proposta de migração",
                     Counter(r["proposta_migracao"] for r in rows), "proposta")
    out += tabela_md("Por detecção", Counter(r["deteccao"] for r in rows), "detecção")
    out += ["### Tipo por proposta", "", "| tipo | proposta | pinos |", "|---|---|---:|"]
    for (tipo, proposta), n in sorted(cruzado.items()):
        out.append("| %s | %s | %d |" % (tipo, proposta, n))
    out.append("")
    out += ["## Pinos afetados pela terminologia", "",
            "Pinos cujo literal contém termo que as decisões de terminologia "
            "substituem; quebrarão quando o texto adotar os termos novos.", ""]
    out += tabela_md("Por arquivo de teste (afetados)", afet_arquivo, "arquivo")
    out += tabela_md("Por termo presente no literal (um pino pode ter vários termos)",
                     termos, "termo")
    out += tabela_md("Afetados por proposta de migração",
                     Counter(r["proposta_migracao"] for r in afetados), "proposta")
    out += [
        "## Limitações",
        "",
        "- Detecção de melhor esforço: a origem do texto é inferida por rastreio de "
        "atribuições dentro do módulo; fluxos por funções de outros pacotes (por "
        "exemplo `scripts/`) e por fixtures não declaradas em `conftest.py` não são vistos.",
        "- Linhas com `deteccao = inferida` usam só o nome do operando (`texto`, `t`, "
        "`source`...) em módulos que citam arquivos da tese; conferir à mão.",
        "- Só asserções (e chamadas `re.*` sobre texto da tese em qualquer instrução) "
        "entram; verificações por `if ...: falhas.append(...)` não são capturadas.",
        "- `linha` é a linha do literal (na constante ou no `parametrize`, quando "
        "o literal vem de lá); `linha_assert` é a da instrução.",
        "- `tipo = outro` com `detalhe` iniciado por `âncora` são fragmentos passados "
        "a leitores de parágrafo/seção; a proposta é `manter` por padrão.",
        "- `categoria = estrutural` marca igualdades sem literal (`[expr] ...`), como "
        "bytes de tabelas regeneradas; não é uma das categorias de conteúdo.",
        "- A heurística de categoria dá precedência a citação, ressalva, número, "
        "negação, termo e frase, nessa ordem; a de proposta trata ausência e regex "
        "como `regra_linter`.",
        "",
    ]
    path.write_text("\n".join(out), encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tests", type=Path, default=DEFAULT_TESTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    rows, avisos, n_modulos = coletar(args.tests)
    args.out.mkdir(parents=True, exist_ok=True)
    escrever_csv(rows, args.out / "inventario_pinos.csv")
    escrever_md(rows, avisos, n_modulos, args.out / "inventario_pinos.md")
    afetados = sum(1 for r in rows if r["afetado_pela_terminologia"] == "sim")
    print("modulos=%d pinos=%d afetados_terminologia=%d saida=%s"
          % (n_modulos, len(rows), afetados, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
