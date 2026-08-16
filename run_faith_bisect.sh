#!/bin/bash
# Four-configuration faithfulness bisect on a single checkpoint, then one
# markdown table. Evaluation only — no training.
#
#   mode  x  split   isolates whether the trainer's low drop is explained by
#                    eval-vs-train conditions or by the train/val gap.
#
# Every configuration uses the same --per_grade sample and the same seed, so the
# four rows differ only in the two factors under test. augment=False throughout.
#
# Usage:
#   bash run_faith_bisect.sh [CKPT] [DR_ROOT] [PER_GRADE]
set -euo pipefail

CKPT="${1:-runs/faith_on/fold0/fold0_best.pth}"
DR_ROOT="${2:-Datasets/DR}"
PER_GRADE="${3:-100}"

FOLD_DIR="$(dirname "$CKPT")"
FOLD=0
HISTORY="$FOLD_DIR/fold${FOLD}_history.csv"
OUT_MD="$FOLD_DIR/bisect.md"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

# Preflight: fail before four long runs rather than after the first
python eval_faithfulness.py --ckpt "$CKPT" --inspect

run_one () {
    local mode="$1" split="$2" bs="$3" tag="$4"
    echo ""
    echo "──────── mode=$mode split=$split batch_size=$bs ────────"
    python eval_faithfulness.py \
        --ckpt "$CKPT" \
        --dr_root "$DR_ROOT" \
        --folds "$FOLD" \
        --per_grade "$PER_GRADE" \
        --mode "$mode" \
        --split "$split" \
        --batch_size "$bs" \
        --out_csv "$FOLD_DIR/bisect_${tag}.csv"
}

run_one eval  val    8  eval_val
run_one train train 24  train_train
run_one eval  train  8  eval_train
run_one train val   24  train_val

# The checkpoint's stored epoch, so the trainer reference row is like-for-like
EPOCH=$(python -c "
import torch,sys
s=torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(s.get('epoch',''))
" "$CKPT")

echo ""
echo "──────── combined table ────────"
python make_bisect_table.py \
    --entry "eval / val=$FOLD_DIR/bisect_eval_val.csv" \
    --entry "train / train=$FOLD_DIR/bisect_train_train.csv" \
    --entry "eval / train=$FOLD_DIR/bisect_eval_train.csv" \
    --entry "train / val=$FOLD_DIR/bisect_train_val.csv" \
    ${EPOCH:+--history "$HISTORY" --epoch "$EPOCH"} \
    --title "Faithfulness bisect — $(basename "$(dirname "$FOLD_DIR")")/$(basename "$FOLD_DIR")" \
    --out "$OUT_MD"
