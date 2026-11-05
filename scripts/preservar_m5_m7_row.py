"""Preserve row-split M5/M5b/M7 outputs before group reruns."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / "data/m5_m7_preservados_row.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        for entry in entries:
            archived = ROOT / entry["archive"]
            assert hashlib.sha256(archived.read_bytes()).hexdigest() == entry["sha256"]
            if archived.suffix == ".tex":
                # Keep literal historical placeholders out of the active TeX tree.
                target = ROOT / "data" / (archived.name + ".txt")
                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError("Different archived table text exists")
                shutil.move(str(archived), target)
                entry["archive"] = target.relative_to(ROOT).as_posix()
        manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        print("M5_M7_ROW_PRESERVED", len(entries))
        return
    paths = set()
    for pattern in ("data/ancoras_ml_*", "data/ancoras_beta_*", "data/m7/e2_*"):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file() and "_row" not in p.stem)
    for name in ("tab_cap8_ancoras_ml.tex", "tab_cap8_ancoras_beta.tex"):
        paths.add(ROOT / "text/latex/tables" / name)
    entries = []
    for source in sorted(paths):
        target = source.with_name(source.stem + "_row" + source.suffix)
        if source.suffix == ".tex":
            target = ROOT / "data" / (target.name + ".txt")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Different row archive: {target.name}")
        shutil.copy2(source, target)
        entries.append({"source": source.relative_to(ROOT).as_posix(), "archive": target.relative_to(ROOT).as_posix(), "sha256": digest})
    manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print("M5_M7_ROW_PRESERVED", len(entries))


if __name__ == "__main__":
    main()
