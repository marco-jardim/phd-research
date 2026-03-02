"""
QW-1 Modal Runner — trains TabNet + XGBoost on cloud GPU (L4).

Runs qw1_deep_learning_comparison.py on Modal serverless GPU.
Checkpoint persists in a Modal Volume so --resume works across runs.

Usage
-----
# 1. Upload dataset to Modal volume (one-time):
modal volume create qw1-data
modal volume put qw1-data data/COMPARADORSEMIDENT.csv COMPARADORSEMIDENT.csv

# Optional: upload existing checkpoint for resume
modal volume put qw1-data data/qw1/ qw1/

# 2a. Run (foreground — streams logs):
modal run scripts/qw1_modal_runner.py

# 2b. Run (background — returns app-id):
modal run --detach scripts/qw1_modal_runner.py

# 3. Check progress (background):
modal app logs <app-id>

# 4. Download results:
modal volume get qw1-data qw1/ data/qw1/
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/root/project"
VOLUME_MOUNT = "/vol"

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------
app = modal.App("qw1-training")

vol = modal.Volume.from_name("qw1-data", create_if_missing=True)

# Image: deps + local source code baked in (Modal 1.3+ API)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "pytorch-tabnet==4.1.0",
        "scikit-learn",
        "pandas>=2.0,<3.0",
        "numpy<2.0",
        "xgboost",
    )
    .add_local_dir("gzcmd", remote_path=f"{PROJECT_ROOT}/gzcmd")
    .add_local_file(
        "scripts/qw1_deep_learning_comparison.py",
        remote_path=f"{PROJECT_ROOT}/scripts/qw1_deep_learning_comparison.py",
    )
    .add_local_file(
        "scripts/feature_engineering.py",
        remote_path=f"{PROJECT_ROOT}/scripts/feature_engineering.py",
    )
)

# ---------------------------------------------------------------------------
# Training function — runs on cloud GPU
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="L4",
    timeout=10800,  # 3 h safety margin
    volumes={VOLUME_MOUNT: vol},
)
def train(resume: bool = True) -> int:
    import os
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path

    project = Path(PROJECT_ROOT)
    vol_root = Path(VOLUME_MOUNT)
    data_dir = project / "data"
    qw1_dir = data_dir / "qw1"

    data_dir.mkdir(parents=True, exist_ok=True)
    qw1_dir.mkdir(exist_ok=True)

    # ---- copy CSV from volume → project data/ ----
    csv_src = vol_root / "COMPARADORSEMIDENT.csv"
    csv_dst = data_dir / "COMPARADORSEMIDENT.csv"
    if not csv_src.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_src}. "
            "Upload first:\n"
            "  modal volume put qw1-data data/COMPARADORSEMIDENT.csv "
            "COMPARADORSEMIDENT.csv"
        )
    shutil.copy2(csv_src, csv_dst)
    print(f"[modal] CSV copied ({csv_src.stat().st_size / 1e6:.1f} MB)", flush=True)

    # ---- copy checkpoint from volume if resuming ----
    vol_qw1 = vol_root / "qw1"
    if resume and vol_qw1.exists():
        n = 0
        for f in vol_qw1.iterdir():
            if f.is_file():
                shutil.copy2(f, qw1_dir / f.name)
                n += 1
        if n:
            print(f"[modal] restored {n} checkpoint file(s) from volume", flush=True)

    # ---- show GPU info ----
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
        print(f"[modal] GPU: {gpu_info}", flush=True)
    except Exception:
        print("[modal] GPU info unavailable", flush=True)

    # ---- run the training script ----
    cmd = [
        sys.executable,
        str(project / "scripts" / "qw1_deep_learning_comparison.py"),
        "--data",
        str(csv_dst),
    ]
    if resume:
        cmd.append("--resume")

    print(f"[modal] cmd: {' '.join(cmd)}", flush=True)
    print(f"[modal] cwd: {project}", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(project),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    last_sync = time.time()
    SYNC_INTERVAL = 300  # sync checkpoint to volume every 5 min

    for line in proc.stdout:  # type: ignore[union-attr]
        print(line, end="", flush=True)

        now = time.time()
        if now - last_sync > SYNC_INTERVAL:
            _sync_to_volume(qw1_dir, vol_qw1)
            last_sync = now

    proc.wait()

    # ---- final sync ----
    _sync_to_volume(qw1_dir, vol_qw1)

    print(f"\n[modal] exit code: {proc.returncode}", flush=True)
    return proc.returncode


def _sync_to_volume(local_dir, vol_dir):
    """Persist checkpoint / results from container disk to Modal Volume."""
    import shutil
    from pathlib import Path

    vol_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in Path(local_dir).iterdir():
        if f.is_file():
            shutil.copy2(f, vol_dir / f.name)
            n += 1
    vol.commit()
    print(f"[modal] synced {n} file(s) to volume", flush=True)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    rc = train.remote(resume=True)
    if rc == 0:
        print("\n=== Training complete! ===")
        print("Download results:")
        print("  modal volume get qw1-data qw1/ data/qw1/")
    else:
        print(f"\n=== Training exited with code {rc} ===")
        print("Check logs, then resume:")
        print("  modal run scripts/qw1_modal_runner.py")
