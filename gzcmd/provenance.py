"""Hash de proveniência independente de plataforma.

Os artefatos JSON/CSV são normalizados pelo git para LF no commit, mas no
Windows podem estar em CRLF no disco. O hash registrado nos YAML de taxas deve
ser o mesmo nos dois casos, por isso calcula-se sobre o conteúdo com os
terminadores de linha normalizados para LF.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_normalized(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def sha256_file_normalized(path: Path) -> str:
    return sha256_normalized(Path(path).read_bytes())
