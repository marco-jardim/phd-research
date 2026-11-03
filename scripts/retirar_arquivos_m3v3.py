"""Untrack unused pre-v4 snapshots, retaining and checking every local byte."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    references = subprocess.run(["rg", "-l", "_m3v3", "text/latex", "--glob", "*.tex"],
                                cwd=ROOT, capture_output=True, text=True)
    if references.returncode not in (0, 1):
        raise RuntimeError("Unable to audit chapter/table references")
    if references.stdout.strip():
        raise ValueError("Snapshots are cited by TeX; refusing to untrack")
    tracked = subprocess.run(["git", "ls-files", "*_m3v3*"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.splitlines()
    before = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in tracked}
    if tracked:
        subprocess.run(["git", "rm", "--cached", "--", *tracked], cwd=ROOT, check=True)
    for path, digest in before.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    output = ROOT / "data/arquivos_m3v3_desversionados.json"
    if tracked or not output.exists():
        output.write_text(json.dumps({"tex_references": [], "local_files_preserved": before}, indent=2), encoding="utf-8")
    print("M3V3_UNTRACKED_LOCAL_BYTES_UNCHANGED", len(tracked))


if __name__ == "__main__":
    main()
