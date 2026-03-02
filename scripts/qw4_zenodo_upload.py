from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import io
import zipfile

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
ZENODO_DIR = ROOT / "data" / "zenodo"
METADATA_PATH = ZENODO_DIR / "zenodo_metadata.json"

# Arquivos/pastas que nunca sobem (artefatos locais, nao dados de pesquisa)
_SKIP = {"deposit_id.txt", "deposit_id_sandbox.txt"}


def _load_token(sandbox: bool) -> str:
    load_dotenv(ROOT / ".env")
    key = "ZENODO_SANDBOX_TOKEN" if sandbox else "ZENODO_ACCESS_TOKEN"
    token = os.getenv(key)
    if not token:
        raise RuntimeError(f"Token ausente: {key}")
    return token


def _api_base(sandbox: bool) -> str:
    return "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"


def _deposit_id_file(sandbox: bool) -> Path:
    name = "deposit_id_sandbox.txt" if sandbox else "deposit_id.txt"
    return ZENODO_DIR / name


def _request(method: str, url: str, token: str, **kwargs) -> requests.Response:
    params = kwargs.pop("params", {})
    params["access_token"] = token
    response = requests.request(method, url, params=params, timeout=120, **kwargs)
    response.raise_for_status()
    return response


def _create_or_get_deposit(token: str, sandbox: bool) -> dict:
    """Retorna rascunho existente ou cria um novo deposito (primeira vez)."""
    base = _api_base(sandbox)
    id_file = _deposit_id_file(sandbox)

    if id_file.exists():
        dep_id = id_file.read_text(encoding="utf-8").strip()
        if dep_id:
            resp = requests.get(
                f"{base}/deposit/depositions/{dep_id}",
                params={"access_token": token},
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()

    created = _request("POST", f"{base}/deposit/depositions", token, json={}).json()
    id_file.write_text(str(created["id"]), encoding="utf-8")
    return created


def _new_version(token: str, sandbox: bool) -> dict:
    """Cria nova versao a partir do deposito publicado existente."""
    base = _api_base(sandbox)
    id_file = _deposit_id_file(sandbox)

    if not id_file.exists() or not id_file.read_text(encoding="utf-8").strip():
        raise RuntimeError(
            "deposit_id nao encontrado — execute sem --new-version primeiro "
            "para criar o deposito inicial."
        )

    dep_id = id_file.read_text(encoding="utf-8").strip()
    print(f"Criando nova versao a partir do deposito {dep_id} ...")
    resp = _request(
        "POST",
        f"{base}/deposit/depositions/{dep_id}/actions/newversion",
        token,
    ).json()

    # A nova versao fica em resp["links"]["latest_draft"]
    draft_url = resp["links"]["latest_draft"]
    draft = _request("GET", draft_url, token).json()
    new_id = draft["id"]
    print(f"  Novo rascunho criado: {new_id}")
    id_file.write_text(str(new_id), encoding="utf-8")
    return draft


def _delete_existing_files(deposition: dict, token: str) -> None:
    files = deposition.get("files", [])
    if not files:
        return
    for item in files:
        file_id = item.get("id")
        if not file_id:
            continue
        links = item.get("links", {})
        self_link = links.get("self")
        if self_link:
            _request("DELETE", self_link, token)
    print(f"  {len(files)} arquivo(s) antigo(s) removido(s) do rascunho.")


def _upload_files(deposition: dict, token: str) -> None:
    """Upload arquivos para o bucket do Zenodo.

    Arquivos planos na raiz de ZENODO_DIR sobem diretamente.
    Subdiretórios são compactados em ZIPs (API legacy não aceita '/' no nome).
    """
    bucket = deposition.get("links", {}).get("bucket")
    if not bucket:
        raise RuntimeError("Deposicao sem bucket de upload")

    # --- arquivos planos na raiz ---
    flat_files = [
        p for p in ZENODO_DIR.iterdir()
        if p.is_file() and p.name not in _SKIP
    ]
    for path in sorted(flat_files):
        with path.open("rb") as fh:
            _request(
                "PUT",
                f"{bucket}/{path.name}",
                token,
                data=fh,
                headers={"Content-Type": "application/octet-stream"},
            )
        print(f"    OK  {path.name}")

    # --- subdiretórios → ZIP em memória ---
    subdirs = [p for p in ZENODO_DIR.iterdir() if p.is_dir()]
    for subdir in sorted(subdirs):
        zip_name = f"{subdir.name}.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(subdir.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(ZENODO_DIR))
        buf.seek(0)
        size_kb = len(buf.getvalue()) // 1024
        _request(
            "PUT",
            f"{bucket}/{zip_name}",
            token,
            data=buf,
            headers={"Content-Type": "application/octet-stream"},
        )
        print(f"    OK  {zip_name} ({size_kb} KB)")

    print(f"  Upload concluido: {len(flat_files)} arquivos + {len(subdirs)} ZIPs.")


def _update_metadata(deposition: dict, token: str, sandbox: bool) -> dict:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    dep_id = deposition["id"]
    base = _api_base(sandbox)
    return _request(
        "PUT",
        f"{base}/deposit/depositions/{dep_id}",
        token,
        json=metadata,
    ).json()


def _publish(deposition: dict, token: str, sandbox: bool) -> dict:
    dep_id = deposition["id"]
    base = _api_base(sandbox)
    return _request(
        "POST",
        f"{base}/deposit/depositions/{dep_id}/actions/publish",
        token,
    ).json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload idempotente dos artefatos QW-4 para Zenodo",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Usa sandbox.zenodo.org em vez de zenodo.org",
    )
    parser.add_argument(
        "--new-version",
        action="store_true",
        dest="new_version",
        help=(
            "Cria nova versao a partir do deposito publicado existente. "
            "Use quando a v1 ja estiver publicada e voce quiser publicar a v2."
        ),
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publica o deposit ao final (irreversivel — use com cuidado)",
    )
    args = parser.parse_args()

    token = _load_token(args.sandbox)

    if args.new_version:
        deposition = _new_version(token, args.sandbox)
    else:
        deposition = _create_or_get_deposit(token, args.sandbox)

    _delete_existing_files(deposition, token)

    # Recarrega deposicao para manter links atualizados
    dep_id = deposition["id"]
    base = _api_base(args.sandbox)
    deposition = _request(
        "GET",
        f"{base}/deposit/depositions/{dep_id}",
        token,
    ).json()

    _upload_files(deposition, token)
    deposition = _update_metadata(deposition, token, args.sandbox)

    if args.publish:
        print("Publicando deposito (IRREVERSIVEL)...")
        deposition = _publish(deposition, token, args.sandbox)

    print(f"\nDeposit ID: {deposition['id']}")
    if deposition.get("doi"):
        print(f"DOI: {deposition['doi']}")
    if deposition.get("links", {}).get("html"):
        print(f"URL: {deposition['links']['html']}")


if __name__ == "__main__":
    main()
