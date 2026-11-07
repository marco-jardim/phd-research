"""Small synthetic Sage endpoint probe; never prints authorization values."""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://sage.levanto.ai"
OUTPUT = ROOT / "data/sprint3b/sage_sem_mo/diagnostico_http.json"
BODY = {"content": "The light is green.", "question": {
    "id": "synthetic_ready_probe", "kind": "yesno", "instructions": "Is the light green?"}}


def main():
    key = dotenv_values(ROOT / ".env").get("SAGE_API_KEY", "")
    if not key:
        raise SystemExit("SAGE_API_KEY missing in .env (value suppressed)")
    choice_probe = "--choice" in sys.argv
    if choice_probe:
        BODY["question"] = {"id": "synthetic_choice_probe", "kind": "choice", "instructions": "What color is the light?",
                            "options": [{"option": "match", "description": "The light is green."},
                                        {"option": "nonmatch", "description": "The light is not green."}]}
        BODY.update({"reasoning": "auto", "grounding": {"trigger": "never"}})
    metadata = {"timestamp": datetime.now(timezone.utc).isoformat(),
                "quickstart": "https://docs.levanto.ai/decision-model/quickstart.md",
                "documented_base": BASE, "documented_path": "/decide",
                "header_template": "Authorization: Bearer <SAGE_API_KEY>",
                "key_source": ".env", "key_prefix_valid": key.startswith("lv_live_"),
                "prefix8": "lv_live_" if key.startswith("lv_live_") else "INVALID_PREFIX_SUPPRESSED",
                "request": BODY, "responses": []}
    if choice_probe and OUTPUT.exists():
        prior = json.loads(OUTPUT.read_text(encoding="utf-8"))
        metadata["previous_probe"] = prior
    print("CONFIG " + json.dumps({k: v for k, v in metadata.items() if k != "responses"}), flush=True)

    def redact(text):
        text = text.replace(key, "[REDACTED]")
        text = re.sub(r"lv_live_[A-Za-z0-9_.-]+", "[REDACTED]", text)
        return re.sub(r"(?i)Bearer\s+[^\s\"<>]+", "Bearer [REDACTED]", text)

    def probe(path, method="POST", attempt=1):
        headers = {"Content-Type": "application/json"}
        if method == "POST":
            headers["Authorization"] = "Bearer " + key
        try:
            response = requests.request(method, BASE + path, headers=headers,
                                        json=BODY if method == "POST" else None,
                                        timeout=(15, 90), allow_redirects=True)
            record = {"method": method, "path": path, "attempt": attempt,
                      "status": response.status_code, "final_url": redact(response.url),
                      "body_literal": redact(response.text),
                      "redirect_statuses": [r.status_code for r in response.history]}
        except requests.RequestException as exc:
            record = {"method": method, "path": path, "attempt": attempt,
                      "status": "connection_error", "type": type(exc).__name__}
        metadata["responses"].append(record)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print("HTTP " + json.dumps(record, ensure_ascii=False), flush=True)
        return record

    if not choice_probe:
        probe("/ready", "GET")
    first = probe("/decide")
    if first["status"] == 402 or first["status"] == 200:
        return
    if first["status"] == 404:
        alternate = probe("/v1/decide")
        if alternate["status"] in (200, 402):
            return
    for attempt, delay in ((2, 2), (3, 4)):
        time.sleep(delay)
        result = probe("/decide", attempt=attempt)
        if result["status"] in (200, 402):
            break


if __name__ == "__main__":
    main()
