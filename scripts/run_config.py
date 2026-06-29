"""Run a single ablation/sweep config from configs/ablation_configs.json.

Usage (called by SLURM array job):
    python scripts/run_config.py $SLURM_ARRAY_TASK_ID

The script selects config[idx] from the JSON file, builds the train_dr.py
command, and runs it in a subprocess inheriting the current environment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CONFIG_FILE = REPO_ROOT / "configs" / "ablation_configs.json"

# Boolean flags that map directly to argparse store_true arguments
BOOL_FLAGS = {
    "use_multi_tile",
    "use_image_text",
    "use_concept_spine",
    "grad_checkpoint",
    "no_cache",
    "no_pretrained",
}

# Keys that are metadata, not CLI args
META_KEYS = {"name"}


def build_cmd(cfg: dict) -> list[str]:
    cmd = ["python", str(REPO_ROOT / "train_dr.py"), "--dr_root", "Datasets/DR"]
    for key, val in cfg.items():
        if key in META_KEYS:
            continue
        if key in BOOL_FLAGS:
            if val:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(val)])
    return cmd


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_config.py <index>", file=sys.stderr)
        sys.exit(1)

    idx = int(sys.argv[1])
    configs = json.loads(CONFIG_FILE.read_text())

    if idx >= len(configs):
        print(f"Index {idx} out of range (total {len(configs)} configs)", file=sys.stderr)
        sys.exit(1)

    cfg = configs[idx]
    name = cfg.get("name", f"config_{idx}")

    print(f"[run_config] task={idx}  name={name}")

    cmd = build_cmd(cfg)
    print(f"[run_config] command: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
