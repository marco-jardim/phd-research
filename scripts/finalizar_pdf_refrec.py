"""Remove only named build outputs; publish approved PDF after log validation."""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ("tese_refrec.*", "tese_*_check.*", "tese_b?_*", "tese_final*",
            "tese_m5b_*", "tese_m8_*", "tese_rev_*", "tese_abl_*")
EXTENSIONS = (".pdf", ".aux", ".log", ".toc", ".lof", ".lot", ".bbl", ".blg", ".out",
              ".nlo", ".nls", ".ilg", ".idx", ".ind", ".synctex.gz", ".nav", ".snm",
              ".vrb", ".fls", ".fdb_latexmk", ".bcf", ".run.xml", ".brf", ".loa", ".lol",
              ".thm", ".hd", ".dvi", ".xdv", ".synctex", ".synctex.gz(busy)")


def is_build_artifact(path, patterns=PATTERNS):
    return path.is_file() and path.name != "tese.pdf" and any(fnmatch.fnmatchcase(path.name, p) for p in patterns) and path.name.endswith(EXTENSIONS)


def clean(patterns):
    removed = []
    for directory in (ROOT / "text/latex", ROOT / "text/pdf"):
        for path in directory.iterdir():
            if is_build_artifact(path, patterns):
                path.unlink()
                removed.append(path.relative_to(ROOT).as_posix())
    return removed


def audit_log(text):
    errors = re.findall(r"^!.*$", text, flags=re.MULTILINE)
    undefined = re.findall(r"^.*(?:Citation|Reference|undefined citations|undefined references).*undefined.*$|^.*There were undefined (?:references|citations).*$", text, flags=re.MULTILINE | re.IGNORECASE)
    return errors, undefined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-old", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    metadata_path = ROOT / "data/refrec_ancoras_m7_build.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if args.clean_old:
        metadata.setdefault("old_build_artifacts_removed", []).extend(clean(PATTERNS))
        print("OLD_BUILD_ARTIFACTS_REMOVED", len(metadata["old_build_artifacts_removed"]))
    if args.publish:
        source = ROOT / "text/latex/tese_refrec2.pdf"
        log = ROOT / "text/latex/tese_refrec2.log"
        errors, undefined = audit_log(log.read_text(encoding="latin1"))
        if errors or undefined:
            raise ValueError(f"Build not publishable: errors={len(errors)}, undefined={len(undefined)}")
        if not source.read_bytes().startswith(b"%PDF"):
            raise ValueError("Invalid PDF")
        destinations = [ROOT / "text/latex/tese.pdf", ROOT / "text/pdf/tese.pdf"]
        for destination in destinations:
            if destination.exists():
                with destination.open("ab"):
                    pass
        for destination in destinations:
            shutil.copy2(source, destination)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path in destinations)
        metadata.update({"jobname": "tese_refrec2", "errors": 0, "undefined_citations_or_references": 0,
                         "pdf_sha256": digest, "pdf_bytes": source.stat().st_size,
                         "destinations": [p.relative_to(ROOT).as_posix() for p in destinations],
                         "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                         "overfull_hbox_count": log.read_text(encoding="latin1").count("Overfull \\hbox")})
        metadata["final_build_artifacts_removed"] = clean(("tese_refrec2.*",))
        print("PDF_PUBLISHED", digest, metadata["pdf_bytes"], "ERRORS=0 UNDEFINED=0")
        print("FINAL_BUILD_ARTIFACTS_REMOVED", len(metadata["final_build_artifacts_removed"]))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
