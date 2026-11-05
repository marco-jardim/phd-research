"""Portable provenance paths for the aggregate arrangement generators."""
from pathlib import Path


def relative_path(path, root, data_root=None):
    """Map local input clones to the same repository-relative logical path."""
    path = Path(path).resolve()
    for base in (root, data_root):
        if base is not None and path.is_relative_to(Path(base).resolve()):
            return path.relative_to(Path(base).resolve()).as_posix()
    raise ValueError(f"Provenance source outside repository roots: {path.name}")


def portable(value, root, data_root=None):
    """Normalize nested journal provenance, including absolute dictionary keys."""
    if isinstance(value, dict):
        return {portable(k, root, data_root): portable(v, root, data_root)
                for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v, root, data_root) for v in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return relative_path(value, root, data_root)
    return value


def verify_replay_prereg(data_root, expected_hash):
    """Verify frozen S1 inputs without requiring today's audit tests to be old.

    Training/execution retains verify_prereg's strict current-code check. Offline
    replay alone allows changed tests, checking their registered bytes in git.
    All numerical code and data fingerprints still have to match exactly.
    """
    import hashlib
    import subprocess

    import yaml

    from gzcmd.arranjo_corte import sha256
    from scripts.arranjo_corte_s1_simular import PREREG, ROOT, fingerprints

    if sha256(PREREG) != expected_hash:
        raise ValueError("Preregistration SHA256 mismatch")
    spec = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", "gzcmd/prereg_arranjo_corte.yaml"],
        cwd=ROOT, text=True).strip()
    recorded = subprocess.check_output(["git", "show", commit + ":gzcmd/prereg_arranjo_corte.yaml"], cwd=ROOT)
    if yaml.safe_load(recorded) != spec:
        raise ValueError("Preregistration differs from committed version")
    actual = fingerprints(data_root)
    for name in actual:
        if name.startswith("tests/"):
            original = subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
            actual[name] = hashlib.sha256(original).hexdigest()
    if actual != spec["sha256"]:
        raise ValueError("Frozen replay input/code fingerprint mismatch")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/pq/arranjo-corte"],
                   cwd=ROOT, check=True)
    return spec, commit
