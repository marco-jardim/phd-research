"""Audit explicit checkout EOLs for byte-sensitive versioned references.

Conservative discovery: literals (including output stems/globs) in byte/hash
test modules, generated editorial tables, and files fingerprinted in tracked
tests/manifests. No source or reference bytes are rewritten. Binary files are
reported but do not need text EOL conversion. --suggest prints missing rules.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = re.compile(r'read_bytes\(|sha256\(|assert_frozen_csv_hash\(|filecmp\.')


def discover(root=ROOT):
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    names = [n for n in names if n and (root / n).is_file()]
    reasons = {}

    def add(name, reason):
        reasons.setdefault(name, set()).add(reason)

    hash_sources = {}
    literals = {}
    for name in names:
        path = root / name
        if path.suffix not in {'.py', '.json', '.yaml', '.yml', '.md'}:
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        # Pin values in tests and metadata also cover indirect/dynamic hash loops.
        for digest in set(re.findall(r'\b[0-9a-f]{64}\b', text)):
            hash_sources.setdefault(digest, set()).add(name)
        if name.startswith('tests/') and path.suffix == '.py' and SIGNALS.search(text):
            tree = ast.parse(text)
            # Only byte/hash test FUNCTIONS, not unrelated prose/style tests in
            # the same module. Generic suffix globs are handled by the explicit
            # editorial family, not expanded over unrelated repository caches.
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                         and SIGNALS.search(ast.get_source_segment(text, node) or '')]
            literals[name] = {node.value for function in functions for node in ast.walk(function)
                              if isinstance(node, ast.Constant) and isinstance(node.value, str)
                              and len(node.value) >= 8 and '\n' not in node.value}
            # Resolve imported generator constants too (e.g. tabela.TEX,
            # sens.OUT). These references need not contain a filename literal
            # in the test itself. Expand only local scripts/gzcmd modules.
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
                    imports.update(node.module + '.' + alias.name for alias in node.names)
            for module in imports:
                if not module.startswith(('scripts.', 'gzcmd.')):
                    continue
                source = root / (module.replace('.', '/') + '.py')
                if source.is_file():
                    imported = ast.parse(source.read_text(encoding='utf-8'))
                    literals[name].update(node.value for node in ast.walk(imported)
                                          if isinstance(node, ast.Constant) and isinstance(node.value, str)
                                          and len(node.value) >= 8 and '\n' not in node.value)
    for name in names:
        raw = (root / name).read_bytes()
        normalized = raw.replace(b'\r\n', b'\n')
        for digest in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(normalized).hexdigest(),
                       hashlib.sha256(normalized.replace(b'\n', b'\r\n')).hexdigest()}:
            for source in hash_sources.get(digest, ()):
                add(name, 'fingerprint:' + source)
        if name.startswith('text/latex/tables/editorial/'):
            add(name, 'editorial generator references (conservative superset)')
        for test, tokens in literals.items():
            for token in tokens:
                # Full paths, filenames, stems, and literal glob expressions;
                # directory-only tokens are deliberately excluded.
                token = token.replace('\\', '/')
                if token.endswith('/') or token in {'text/latex', 'text/slides', 'data/sprint3b'}:
                    continue
                if token == name or token == Path(name).name or token == Path(name).stem or (
                    '*' in token and token not in {'*.json', '*.csv', '*.tex', '*.py', '*.md'}
                    and (fnmatch.fnmatchcase(name, token) or fnmatch.fnmatchcase(Path(name).name, token))):
                    add(name, 'byte/hash test:' + test)
    return {name: sorted(why) for name, why in sorted(reasons.items())}


def audit(root=ROOT):
    references = discover(root)
    if not references:
        raise ValueError('No byte-sensitive references found')
    result = subprocess.run(['git', 'check-attr', '-z', '--stdin', 'text', 'eol'], cwd=root,
                            input='\0'.join(references).encode() + b'\0', capture_output=True, check=True)
    fields = result.stdout.decode().split('\0')
    attributes = {}
    for i in range(0, len(fields) - 1, 3):
        name, key, value = fields[i:i + 3]
        attributes.setdefault(name, {})[key] = value
    rows = []
    for name, evidence in references.items():
        raw = (root / name).read_bytes()
        try:
            raw.decode('utf-8')
            binary = b'\0' in raw
        except UnicodeDecodeError:
            binary = True
        crlf = raw.count(b'\r\n')
        lf = raw.count(b'\n') - crlf
        mixed_checkout = False
        if crlf and lf:
            # A historical worktree may contain mixed EOLs although Git stores
            # an LF blob. Preserve the frozen worktree, require LF on future
            # checkouts, and report the discrepancy instead of rewriting it.
            blob = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=root)
            mixed_checkout = blob == raw.replace(b'\r\n', b'\n') and b'\r\n' not in blob
        expected = 'crlf' if crlf and not lf else 'lf'
        attrs = attributes[name]
        ok = binary or (attrs['text'] in {'set', 'auto'} and attrs['eol'] == expected
                        and (not (crlf and lf) or mixed_checkout))
        rows.append({'path': name, 'evidence': evidence, 'binary': binary, 'crlf': crlf, 'lf': lf,
                     'mixed_worktree_matches_LF_git_blob': mixed_checkout,
                     'expected_eol': expected, 'attributes': attrs, 'ok': ok})
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suggest', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    missing = [row for row in rows if not row['ok']]
    for row in missing:
        print(row['path'] + ' text eol=' + row['expected_eol'] if args.suggest else json.dumps(row, ensure_ascii=False))
    print(f'Audited {len(rows)} references; {len(missing)} incoherent/missing EOL rules')
    if missing and not args.suggest:
        raise SystemExit(1)
