"""Explicit opt-in boundaries between portable aggregates and private inputs."""
import os
import hashlib
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PHD_RESEARCH_DATA_ROOT", str(
    Path("D:/git/phd-research") if Path("D:/git/phd-research").exists() else ROOT)))


def requires_files(*paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    return pytest.mark.skipif(bool(missing), reason=(
        "Private/local inputs unavailable; set PHD_RESEARCH_DATA_ROOT: " + ", ".join(missing)))


def requires_history(ref="b09e979"):
    present = subprocess.run(["git", "cat-file", "-e", ref + "^{commit}"], cwd=ROOT,
                             capture_output=True).returncode == 0
    return pytest.mark.skipif(not present, reason=f"Historical commit {ref} unavailable (shallow clone)")


def requires_reviewer_sources():
    import yaml
    spec = yaml.safe_load((ROOT / "gzcmd/error_rates_by_band_v4.yaml").read_text(encoding="utf-8"))
    return requires_files(*(DATA_ROOT / source["path"] for source in spec["sources"]
                            if source["model"] == "claude-haiku-5-5"))


def assert_frozen_csv_hash(path, expected):
    """Git normalizes CSV EOLs; accept only equivalent LF/CRLF byte forms."""
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    assert expected in {hashlib.sha256(content).hexdigest(),
                        hashlib.sha256(content.replace(b"\n", b"\r\n")).hexdigest()}
