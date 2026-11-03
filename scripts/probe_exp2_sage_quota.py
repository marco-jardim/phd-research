"""Make exactly one minimal decision request before resuming an Exp2 run."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main():
    key = dotenv_values(ROOT / ".env").get("SAGE_API_KEY", "")
    if not key:
        raise SystemExit("SAGE_API_KEY missing (value suppressed)")
    body = {"content": "The light is green.", "question": {
        "id": "quota_probe", "kind": "yesno", "instructions": "Is the light green?"}}
    response = requests.post("https://sage.levanto.ai/decide", json=body,
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                             timeout=(15, 90), allow_redirects=False)
    literal = re.sub(r"lv_live_[A-Za-z0-9_.-]+", "[REDACTED]", response.text.replace(key, "[REDACTED]"))
    folder = ROOT / "data/sprint3b/sage_sem_mo"
    counts = {}
    for mode in ("vigilancia", "confirmacao"):
        path = folder / f"exp2_llm_sync_{mode}.json"
        if path.exists():
            run = json.loads(path.read_text(encoding="utf-8"))
            counts[mode] = {"completed": len(run["results"]), "expected": run["n_review_pairs"], "status": run["status"]}
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "method": "POST",
              "final_url": response.url, "status": response.status_code, "requests": 1,
              "body_literal": literal, "checkpoint_counts": counts}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "quota_probe.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if response.status_code != 200:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
