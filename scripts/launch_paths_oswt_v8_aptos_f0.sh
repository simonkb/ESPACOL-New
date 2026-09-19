#!/bin/bash
# Submit the complete prospective PATHS-V8 experiment graph.
set -euo pipefail

: "${PATHS_OSWT_APTOS_V3_SHA256:?export the audited APTOS V3 SHA-256}"
: "${PATHS_OSWT_DR_V3_SHA256:?export the audited EyePACS V3 SHA-256}"

IMPLEMENTATION_FILES=(
  configs/paths_oswt_config.py configs/paths_config.py configs/origin_config.py
  Datasets/origin_data.py Datasets/mosaic_data.py Datasets/dataloaders.py
  models/origin_encoder.py models/origin.py models/paths.py models/paths_oswt.py
  losses/origin.py losses/paths.py
  training/origin_trainer.py training/paths_trainer.py training/paths_oswt_trainer.py
  train_paths_oswt.py train_origin.py utils/spatial_mask.py
  scripts/submit_paths_oswt_preflight.sh scripts/submit_paths_oswt_aptos_f0.sh
  scripts/submit_paths_oswt_aptos_ungated_f0.sh
  scripts/submit_paths_oswt_promotion_gate.sh scripts/submit_paths_oswt_dr_f0.sh
  scripts/launch_paths_oswt_v8_aptos_f0.sh
)
for path in "${IMPLEMENTATION_FILES[@]}"; do
  [[ "$(git rev-parse "HEAD:${path}")" == "$(git hash-object "${path}")" ]] || {
    echo "Tracked implementation differs from HEAD: ${path}" >&2; exit 2;
  }
done

PRE="$(sbatch --parsable --export=ALL scripts/submit_paths_oswt_preflight.sh | cut -d';' -f1)"
TREAT="$(sbatch --parsable --export=ALL --dependency="afterok:${PRE}" scripts/submit_paths_oswt_aptos_f0.sh | cut -d';' -f1)"
CONTROL="$(sbatch --parsable --export=ALL --dependency="afterok:${PRE}" scripts/submit_paths_oswt_aptos_ungated_f0.sh | cut -d';' -f1)"
GATE="$(sbatch --parsable --export=ALL --dependency="afterok:${TREAT}:${CONTROL}" scripts/submit_paths_oswt_promotion_gate.sh | cut -d';' -f1)"
DR="$(sbatch --parsable --export=ALL --dependency="afterok:${GATE}" scripts/submit_paths_oswt_dr_f0.sh | cut -d';' -f1)"

printf 'preflight=%s\ntreatment=%s\nungated_control=%s\npromotion_gate=%s\neyepacs=%s\n' \
  "${PRE}" "${TREAT}" "${CONTROL}" "${GATE}" "${DR}"
