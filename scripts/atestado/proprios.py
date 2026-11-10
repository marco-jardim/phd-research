"""Obras próprias: em vez de recuperar texto, confere os artefatos citados na frase (\\path{...})."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ARTEFATO_OK = "ARTEFATO_OK"
ARTEFATO_AUSENTE = "ARTEFATO_AUSENTE"
SEM_ARTEFATO = "SEM_ARTEFATO_NA_FRASE"

_PATH_RE = re.compile(r"\\path\s*\{([^}]*)\}")


def extract_path_targets(*texts: str) -> list[str]:
    """Alvos de ``\\path{...}`` nos textos, sem repetição e na ordem de aparição."""
    seen: list[str] = []
    for text in texts:
        for match in _PATH_RE.finditer(text or ""):
            target = match.group(1).replace("\\_", "_").strip()
            if target and target not in seen:
                seen.append(target)
    return seen


def git_tracked_files(root: Path) -> set[str]:
    """Arquivos versionados (``git ls-files``); vazio se o git não estiver disponível."""
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        result = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, encoding="utf-8",
            check=True, creationflags=flags,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def target_exists(target: str, root: Path, tracked: set[str]) -> bool:
    posix = target.replace("\\", "/")
    if posix.startswith("./"):
        posix = posix[2:]
    if any(ch in posix for ch in "*?["):
        return any(root.glob(posix)) or any(
            Path(name).match(posix) for name in tracked
        )
    clean = posix.rstrip("/")
    if clean in tracked or any(name.startswith(clean + "/") for name in tracked):
        return True
    return (root / clean).exists()


def check_artifacts(contexts: list[dict], root: Path, tracked: set[str] | None = None) -> list[dict]:
    """Veredito de artefato por contexto de obra própria."""
    files = git_tracked_files(root) if tracked is None else tracked
    rows = []
    for ctx in contexts:
        targets = extract_path_targets(ctx.get("sentence_latex", ""), ctx.get("paragraph_text", ""))
        missing = [t for t in targets if not target_exists(t, root, files)]
        if not targets:
            verdict = SEM_ARTEFATO
        elif missing:
            verdict = ARTEFATO_AUSENTE
        else:
            verdict = ARTEFATO_OK
        rows.append(
            {
                "id": ctx["id"],
                "key": ctx["key"],
                "file": ctx.get("file", ""),
                "line": ctx.get("line", 0),
                "veredito": verdict,
                "alvos": targets,
                "ausentes": missing,
            }
        )
    return rows
