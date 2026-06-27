# ESPAOCL fixes integrated into the training repo

The three method fixes are now in the modular pipeline, gated so the base
PCOL+SCOLw+RMSE behaviour is byte-identical when the flags are off.

## Files changed / added

Changed: `models/backbone.py`, `models/heads.py`, `models/framework.py`,
`models/__init__.py`, `losses/combined.py`, `losses/__init__.py`,
`training/trainer.py`, `train_dr.py`, `train_busi.py`.

New: `losses/pic.py` (per-image concept loss), `losses/segmentation.py`
(BCE+Dice), `models/concept_bank.py` (BioMedCLIP text concepts + phrase/grade
lists), `Datasets/busi_seg.py` (mask-loading BUSI dataset).

Your `configs/config.py`, `utils/`, `losses/pcol.py`, `losses/scolw.py`,
`Datasets/dataloaders.py`, `training/cross_val.py` are unchanged and are NOT in
this package.

## One required config change

Add these fields to your `TrainConfig` (or BUSI/DRConfig) in `configs/config.py`.
The code reads them with `getattr(..., default)`, so missing fields fall back to
"off", but adding them makes the knobs explicit and sweepable:

```python
    # ESPAOCL fixes
    backbone: str = "efficientnet"   # "efficientnet" | "biomedclip"  (Fix 3)
    use_concept: bool = False        # per-image concept loss          (Fix 2)
    use_seg: bool = False            # auxiliary segmentation loss      (Fix 1)
    delta: float = 0.0               # weight for L_pic
    eps: float = 0.0                 # weight for L_seg
    pic_temperature: float = 0.1
    pic_lambda: float = 0.5
    biomedclip_model: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
```

## How to run

Baseline (unchanged):
```
python train_dr.py --dr_root Datasets/DR --folds 0
```

Fix 2, the DR Moderate fix (per-image concept loss). Validate on fold 0 first:
```
python train_dr.py --dr_root Datasets/DR --folds 0 --use_concept --delta 0.05
```

Fix 1, BUSI heatmap sharpening (segmentation loss, backbone trains with it):
```
python train_busi.py --busi_root Datasets/BUSI --folds 0 --use_seg --eps 0.5
```
Both BUSI fixes together:
```
python train_busi.py --busi_root Datasets/BUSI --folds 0 --use_concept --use_seg
```

Fix 3, the BioMedCLIP vision tower (input becomes 224; heads rebuild at 512;
seg head uses the 768-d token grid automatically):
```
python train_dr.py --dr_root Datasets/DR --folds 0 --backbone biomedclip --use_concept
```

## Success criterion (Fix 2)

DR Moderate own-grade top-1 should rise off 0.31 (chance 0.27) toward 0.45+ while
the anchored grades hold. Measure with your notebook's `_dr_top1_by_grade` on the
retrained checkpoint.

## What is tested vs needs your environment

Tested here with torch 2.12 (CPU, random EfficientNet weights):
- `build_model` with `use_concept`/`use_seg`, both forward paths, correct shapes.
- The extended six-term loss: all terms compute, gradient reaches the backbone,
  the seg term auto-drops on mask-less (DR) batches.
- `ExtendedTrainer.fit` runs a full epoch end to end.

Not executed here (need your data and network):
- The `train_*.py` runs with the real BioMedCLIP download and your datasets.
- The `biomedclip` backbone itself (needs the open_clip weights). Its dims and
  resolution handling are written to the standard open_clip API; verify on first
  load.

## One thing to check

The concept phrases in `models/concept_bank.py` follow the deck's ordering. If
your notebook's `DR_CONCEPTS` / `BUSI_CONCEPTS` used different exact wording, paste
those strings in so the bank matches the vectors your analysis was built on. The
grades must stay in the same row order as the phrases.
