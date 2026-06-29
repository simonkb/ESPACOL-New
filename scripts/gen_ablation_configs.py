"""Generate configs/ablation_configs.json for SLURM array ablation + sweep jobs.

Run once locally before submitting to HPC:
    python scripts/gen_ablation_configs.py

Produces configs/ablation_configs.json (array of dicts, each = one CLI invocation).
Prints the exact sbatch command with the correct --array=0-N range.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Best baseline hyperparameters (from previous wandb sweep)
# ---------------------------------------------------------------------------
BEST_ALPHA = 0.00662474091401746
BEST_BETA = 0.05516050165777829
BEST_GAMMA = 0.05
BEST_DELTA = 0.1
BEST_ETA = 0.1
BEST_NU = 0.1

COMMON = {
    "folds": "0",
    "use_multi_tile": True,
    "batch_size": 24,
    "grad_checkpoint": True,
    "epochs": 75,
}

# ---------------------------------------------------------------------------
# Phase 1: Module-by-module ablation (7 fixed configs)
# ---------------------------------------------------------------------------
ablation_configs = [
    {
        "name": "01_rmse_only",
        "run_dir": "runs/ablation/01_rmse_only",
        "alpha": 0.0,
        "beta": 0.0,
        "use_image_text": False,
        "use_concept_spine": False,
    },
    {
        "name": "02_plus_scolw",
        "run_dir": "runs/ablation/02_plus_scolw",
        "alpha": 0.0,
        "beta": BEST_BETA,
        "use_image_text": False,
        "use_concept_spine": False,
    },
    {
        "name": "03_plus_pcol",
        "run_dir": "runs/ablation/03_plus_pcol",
        "alpha": BEST_ALPHA,
        "beta": BEST_BETA,
        "use_image_text": False,
        "use_concept_spine": False,
    },
    {
        "name": "04_plus_it",
        "run_dir": "runs/ablation/04_plus_it",
        "alpha": BEST_ALPHA,
        "beta": BEST_BETA,
        "gamma": BEST_GAMMA,
        "use_image_text": True,
        "use_concept_spine": False,
    },
    {
        "name": "05_plus_pic",
        "run_dir": "runs/ablation/05_plus_pic",
        "alpha": BEST_ALPHA,
        "beta": BEST_BETA,
        "gamma": BEST_GAMMA,
        "delta": BEST_DELTA,
        "eta": 0.0,
        "nu": 0.0,
        "use_image_text": True,
        "use_concept_spine": True,
    },
    {
        "name": "06_plus_cons",
        "run_dir": "runs/ablation/06_plus_cons",
        "alpha": BEST_ALPHA,
        "beta": BEST_BETA,
        "gamma": BEST_GAMMA,
        "delta": BEST_DELTA,
        "eta": BEST_ETA,
        "nu": 0.0,
        "use_image_text": True,
        "use_concept_spine": True,
    },
    {
        "name": "07_full_arch",
        "run_dir": "runs/ablation/07_full_arch",
        "alpha": BEST_ALPHA,
        "beta": BEST_BETA,
        "gamma": BEST_GAMMA,
        "delta": BEST_DELTA,
        "eta": BEST_ETA,
        "nu": BEST_NU,
        "use_image_text": True,
        "use_concept_spine": True,
    },
]

# ---------------------------------------------------------------------------
# Phase 2: Random hyperparameter sweep (50 configs over the full arch)
# ---------------------------------------------------------------------------
SWEEP_SPACE = {
    "alpha": [0.001, 0.003, 0.006, 0.01, 0.02],
    "beta":  [0.01, 0.03, 0.055, 0.1, 0.15],
    "gamma": [0.01, 0.02, 0.05, 0.1, 0.2],
    "delta": [0.05, 0.1, 0.2],
    "eta":   [0.02, 0.05, 0.1, 0.2],
    "nu":    [0.0, 0.05, 0.1, 0.2],
}
N_SWEEP = 50

random.seed(42)
sweep_configs: list[dict] = []
seen: set[tuple] = set()

while len(sweep_configs) < N_SWEEP:
    sample = {k: random.choice(v) for k, v in SWEEP_SPACE.items()}
    key = tuple(sample[k] for k in sorted(SWEEP_SPACE))
    if key in seen:
        continue
    seen.add(key)
    idx = len(sweep_configs) + 1
    name = f"sweep_{idx:03d}"
    sweep_configs.append(
        {
            "name": name,
            "run_dir": f"runs/sweep/{name}",
            "use_image_text": True,
            "use_concept_spine": True,
            **sample,
        }
    )

# ---------------------------------------------------------------------------
# Merge and apply COMMON fields, then write output
# ---------------------------------------------------------------------------
all_configs: list[dict] = []
for cfg in ablation_configs + sweep_configs:
    merged = {**COMMON, **cfg}
    all_configs.append(merged)

out_path = Path(__file__).parent.parent / "configs" / "ablation_configs.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(all_configs, indent=2))

n = len(all_configs)
print(f"Written {n} configs to {out_path}")
print(f"  Ablation configs: {len(ablation_configs)} (idx 0–{len(ablation_configs)-1})")
print(f"  Sweep configs:    {len(sweep_configs)} (idx {len(ablation_configs)}–{n-1})")
print()
print("Submit command:")
print(f"  sbatch --array=0-{n-1} submit_ablation_array.sh")
