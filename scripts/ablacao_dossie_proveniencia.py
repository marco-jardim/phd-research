"""Exporta somente proveniência de arquivos, nunca resultados ou campos por par.

Ausência de metadados não é suprida pelo prompt do checkout atual. O histórico
do runner documenta a intenção, mas não certifica o prompt de uma execução.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COORTES = ("anthropic", "anthropic_sem_mo", "anthropic_sem_mo_v4",
           "opencode_go", "opencode_go_sem_mo", "opencode_go_sem_mo_v4")
PARAMS = ("temperature", "max_tokens", "top_p", "top_k", "seed", "thinking",
          "temperature_sent", "extended_thinking", "reasoning_effort")


def metadata(path, relative_to):
    raw = path.read_bytes()
    source = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(source, dict):
        source = {}
    nested = source.get("metadata", {})
    if not isinstance(nested, dict):
        nested = {}
    def recorded(*keys):
        return next((container[key] for container in (source, nested)
                     for key in keys if key in container), None)
    prompt = source.get("system_prompt")
    params = source.get("params", source.get("generation_params", source.get("metadata", {})))
    if not isinstance(params, dict):
        params = {}
    return {
        "source": path.relative_to(relative_to).as_posix(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "model": recorded("model", "model_exact"),
        "prompt_version": recorded("prompt_version"),
        "prompt_sha": recorded("prompt_sha", "prompt_sha256", "prompt_file_sha256"),
        "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if isinstance(prompt, str) else None,
        "date": recorded("created_at", "timestamp", "date"),
        "params": {k: params.get(k, source.get(k)) for k in PARAMS
                   if k in params or k in source},
    }


def generate(source):
    arms = sorted(set((source / "ablacao").glob("*_controle.json"))
                  | set((source / "ablacao").glob("*_sem_pcal.json")))
    if not arms:
        raise FileNotFoundError("Originais privados ausentes; informe --source com a raiz data/sprint3b")
    cohorts = {name: [metadata(p, source) for p in sorted((source / name).glob("*.json"))]
               for name in COORTES}
    history = subprocess.run(["git", "log", "-p", "--follow", "--",
                              "scripts/run_exp2_ablacao_dossie.py"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    return {"schema_version": 1, "arms": [metadata(p, source) for p in arms],
            "cohorts": cohorts,
            "runner_history": {"command": "git log -p --follow -- scripts/run_exp2_ablacao_dossie.py",
                               "sha256": hashlib.sha256(history).hexdigest(),
                               "finding": "O runner importa os revisores e declara prompt original, sem congelar sua versão no checkpoint. O histórico não certifica a instrução de cada execução."},
            "missing_metadata": "null indica campo não registrado; params vazio não significa defaults conhecidos",
            "scope": "hashes de arquivos inteiros e metadados permitidos, sem identificadores por par"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/sprint3b")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/sprint3b/ablacao/ablacao_dossie_proveniencia.json")
    args = parser.parse_args()
    result = generate(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    print(f"Proveniência: {len(result['arms'])} braços; {sum(map(len, result['cohorts'].values()))} arquivos de coortes")


if __name__ == "__main__":
    main()
