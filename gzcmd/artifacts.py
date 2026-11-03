"""Idempotent preservation of this pipeline's pre-v4 generated artifacts only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


def preserve_m3v3_artifacts(root: Path) -> dict:
    manifest = root / "data/arquivos_preservados_m3v3.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        for item in saved["files"]:
            path = root / item["archive"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Preserved artifact changed: {path}")
        return saved
    paths = set()
    for pattern in ("funil_gzcmd*.csv", "funil_gzcmd_relatorio.md", "perda_total_por_modelo.csv",
                    "lote_llm_novos_*.csv", "varredura_custos*.csv", "varredura_custos*.json",
                    "varredura_custos_relatorio.md", "varredura_np*.csv", "varredura_np*.json",
                    "comparacao_np_confirmacao.csv", "variante_guarda_confirmacao.csv", "np_confirmacao_relatorio.md"):
        paths.update((root / "data").glob(pattern))
    for name in ("fig_cap8_varredura_custos", "fig_cap8_np_confirmacao"):
        for extension in ("pdf", "png"):
            path = root / "text/latex/figures" / f"{name}.{extension}"
            if path.exists():
                paths.add(path)
    saved = {"suffix": "_m3v3", "files": []}
    for source in sorted(paths):
        if "_m3v3" in source.stem:
            continue
        target = source.with_name(source.stem + "_m3v3" + source.suffix)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Refusing to overwrite a different archive: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        saved["files"].append({"source": source.relative_to(root).as_posix(),
                               "archive": target.relative_to(root).as_posix(), "sha256": digest})
    manifest.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return saved
