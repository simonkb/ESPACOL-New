# ESPACOL — PyTorch Reproduction

> **A Hybrid Contrastive Ordinal Regression Method for Advancing Disease Severity Assessment in Imbalanced Medical Datasets**
> Afsah Saleem, Joshua R. Lewis, Syed Zulqarnain Gilani — MICCAI 2024

This README is structured to be read **in parallel with the paper**. Each section below corresponds to a paper section and points to the exact file and line where it is implemented.

## Repository Structure

```
ESPACOL-New/
│
├── configs/
│   └── config.py               # All hyperparameters (Paper §3)
│
├── models/
│   ├── backbone.py             # EfficientNet-V2S encoder (Paper §2.1, §3)
│   ├── heads.py                # Projection heads + Regression head (Paper §2.1, §2.4)
│   └── framework.py            # Full model wiring — Fig. 1 in paper
│
├── losses/
│   ├── pcol.py                 # Eq. (1): PCOL loss (Paper §2.2)
│   ├── scolw.py                # Eq. (2): SCOLw loss (Paper §2.3)
│   └── combined.py             # Eq. (3): L_total + class weight helper (Paper §2.4)
│
├── Datasets/
│   └── dataloaders.py          # Image loading, transforms, stratified sampler (Paper §3)
│
├── training/
│   ├── cross_val.py            # Subject-independent CV splits (Paper §3)
│   └── trainer.py              # Training loop, early stopping, LR scheduler (Paper §3)
│
├── utils/
│   ├── metrics.py              # Accuracy and MAE (Paper §3 - Evaluation Metrics)
│   └── checkpoint.py           # Save / load best model checkpoint
│
├── train_dr.py                 # Entry point: DR 10-fold CV
├── train_busi.py               # Entry point: BUSI 5-fold CV
├── train_dr_sweep.py           # WandB hyperparameter sweep (DR proxy subset)
├── train_busi_sweep.py         # WandB hyperparameter sweep (BUSI)
└── create_dr_subset.py         # Creates 4K balanced DR proxy for sweeps
```

---

## Paper §2.1 — Preliminaries: Model Architecture

**Paper quote:** *"The feature map ψ from the encoder is passed through global average pooling (grey layer) to convert into feature-embeddings, then to two separate MLP layers (blue box), each consisting of two dense layers with 1280 and 128 neurons."*

### Backbone — `models/backbone.py`

```
Input image (300×300×3)
    └─► EfficientNet-V2S feature extractor
    └─► AdaptiveAvgPool2d  (Global Average Pooling)
    └─► Flatten
    └─► ψ ∈ ℝ¹²⁸⁰   (1280-dim feature vector)
```

- Uses `torchvision.models.efficientnet_v2_s` with ImageNet pre-trained weights.
- The classifier head is discarded; only `features` + `avgpool` are kept.
- Output dimension is always 1280 (`EfficientNetV2SBackbone.OUT_DIM = 1280`).

### Projection Heads — `models/heads.py`

Two **identical** MLP heads (one for PCOL, one for SCOLw):

```
ψ (1280)  →  Linear(1280→1280)  →  BatchNorm  →  ReLU
          →  Linear(1280→128)
          →  L2-normalize        →  z ∈ ℝ¹²⁸ (unit sphere)
```

`MLPProjectionHead` produces L2-normalized 128-dim embeddings used by both contrastive losses.

### Regression Head — `models/heads.py`

```
ψ (1280)  →  Linear(1280→1)  →  scalar prediction ŷ ∈ ℝ
```

A single linear layer. At inference, `ŷ` is rounded to the nearest integer class.

**Paper quote:** *"In the testing phase, only the regression head is used for inference."*

### Full Model Wiring — `models/framework.py`

`HybridContrastiveOrdinalModel.forward()` returns three outputs per forward pass:

| Output   | Shape   | Used by     |
|----------|---------|-------------|
| `z_pcol` | (N, 128)| PCOL loss   |
| `z_scolw`| (N, 128)| SCOLw loss  |
| `pred`   | (N,)    | RMSE loss + accuracy |

`build_model()` constructs the full model from the three sub-modules.

---

## Paper §2.2 — PCOL: Prototype-based Contrastive Ordinal Loss

**Equation (1):**

$$L_{PCOL} = -\sum_{i \in \mathcal{I}} \log \frac{\exp(f_a^\top \cdot c_p \;/\; \tau)}{\sum_{c_n \in N(i)} \exp\!\bigl((f_a^\top \cdot c_n + r_{a,n})\;/\;\tau\bigr)}$$

**Implemented in:** `losses/pcol.py` — `PCOLLoss.forward()`

### Notation mapped to code

| Paper symbol | Meaning | Code variable |
|---|---|---|
| $f_a$ | L2-normalized anchor embedding | `embeddings[i]` |
| $c_p$ | Prototype of anchor's class (mean of same-class embeddings, L2-normed) | `protos[is_pos]` |
| $c_n$ | Prototype of a negative class | `protos[~is_pos]` |
| $N(i)$ | Set of all negative class prototypes | `~is_pos` mask |
| $r_{a,n}$ | Ordinal distance: $\|y_a - y_n\|$ | `ord_dist` |
| $\tau$ | Temperature | `self.temperature` |

### Step-by-step in code (`losses/pcol.py`)

1. **Build prototypes** (lines 61–71): for each class present in the batch, compute mean of embeddings → L2-normalize → one prototype vector per class.
2. **Similarity matrix** (line 76): `sim = embeddings @ protos.T` — shape (N, C).
3. **Ordinal distance matrix** (lines 79–82): `ord_dist[i,j] = |y_i - y_j|` — shape (N, C).
4. **Positive mask** (line 86): `is_pos[i,j] = (label_i == label_j)`.
5. **Numerator** (lines 89–91): `pos_sim / τ` for the one matching prototype per anchor.
6. **Denominator** (lines 94–99): `logsumexp((sim + ord_dist) / τ)` over negative prototypes only.
7. **Loss** (line 102): `-(numerator - denominator).mean()`.

> **Key design choice:** `r_{a,n}` is added to the **denominator** logits only (negative prototypes). This makes the model pay a larger penalty for being close to far-away classes, enforcing ordinal consistency.

---

## Paper §2.3 — SCOLw: Weighted Supervised Contrastive Ordinal Loss

**Equation (2):**

$$L_{SCOLw} = \sum_{i \in B} \frac{-w_i}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(f_a^\top \cdot f_p \;/\; \tau)}{\sum_{n \in N(i)} \exp\!\bigl((f_a^\top \cdot f_n + r_{a,n})\;/\;\tau\bigr)}$$

**Implemented in:** `losses/scolw.py` — `SCOLwLoss.forward()`

### Notation mapped to code

| Paper symbol | Meaning | Code variable |
|---|---|---|
| $f_a, f_p, f_n$ | L2-normalized anchor / positive / negative embeddings | `embeddings[i]`, `[p]`, `[n]` |
| $P(i)$ | Same-class samples (excluding self) | `is_pos = same_cls & ~self_mask` |
| $N(i)$ | Different-class samples | `is_neg = ~same_cls` |
| $w_i$ | Per-sample class weight | `w = class_weights[labels]` |
| $r_{a,n}$ | Ordinal distance: $\|y_a - y_n\|$ | `ord_dist` |
| $\tau$ | Temperature | `self.temperature` |

### Step-by-step in code (`losses/scolw.py`)

1. **Per-sample weights** (line 54): `w[i] = class_weights[label_i]` — look up pre-computed inverse-frequency weight.
2. **Pairwise similarity** (line 57): `sim = embeddings @ embeddings.T` — (N, N).
3. **Ordinal distances** (lines 60–63): `ord_dist[i,j] = |y_i - y_j|` — (N, N).
4. **Masks** (lines 65–68): separate `is_pos` (same class, not self) and `is_neg` (different class).
5. **Denominator** (lines 71–75): `logsumexp((sim + ord_dist) / τ)` over `is_neg` entries per anchor.
6. **Log-probability** (line 79): `log_prob[i,p] = sim[i,p]/τ - log_denom[i]`.
7. **Per-anchor loss** (line 88): `(-w_i / |P(i)|) * sum_p log_prob[i,p]`.
8. **Anchors with no positives** (lines 92–94): masked out (no contrastive signal possible).

### Class weights

**Paper quote:** *"weights are dynamically assigned to each sample in the training batch based on the inverse frequency of its class in the dataset."*

`compute_class_weights()` in `losses/combined.py`:
```
w[c] = N_total / (n_classes × n_c)
```
In practice this is computed **per mini-batch** from the current batch's labels (not the full dataset), so with stratified sampling (4 samples per class) all weights ≈ 1.0.

---

## Paper §2.4 — Total Loss

**Equation (3):**

$$L_{total} = \alpha \cdot L_{PCOL} + \beta \cdot L_{SCOLw} + L_{RMSE}$$

**Implemented in:** `losses/combined.py` — `HybridContrastiveOrdinalLoss.forward()`

```python
total = alpha * l_pcol + beta * l_scolw + l_rmse
```

| Term | Loss | Head | Weight |
|------|------|------|--------|
| $\alpha \cdot L_{PCOL}$ | PCOLLoss | PCOL projection head | `alpha` |
| $\beta \cdot L_{SCOLw}$ | SCOLwLoss | SCOLw projection head | `beta` |
| $L_{RMSE}$ | `sqrt(MSE(pred, labels))` | Regression head | 1.0 (fixed) |

**Paper quote:** *"The projection heads and regression head are optimized jointly in a single-stage supervised contrastive learning framework."*

All three heads share the same backbone and are updated together in each training step.

---

## Paper §3 — Experimental Setting

### Hyperparameters — `configs/config.py`

`TrainConfig` holds all base hyperparameters. Dataset-specific subclasses override what differs.

| Paper specification | Config field | Value (DR) | Value (BUSI) |
|---|---|---|---|
| Image size 300×300 | `img_size` | 300 | 300 |
| Normalize to [0,1] | `build_transform()` in dataloaders | ToTensor only | ToTensor only |
| Batch size 24 | `batch_size` | 24 | 24 |
| Max epochs 75 | `epochs` | 75 | 75 |
| Initial LR 1×10⁻³ | `lr` | 1×10⁻⁴ † | 5×10⁻⁴ |
| LR factor 0.2 | `lr_factor` | 0.2 | 0.2 |
| LR patience 5 epochs | `lr_patience` | 8 † | 5 |
| Early stopping 13 epochs | `early_stop_patience` | 20 † | 13 |
| Projection: 1280→128 | `proj_hidden_dim`, `proj_out_dim` | 1280, 128 | 1280, 128 |
| α, β loss weights | `alpha`, `beta` | 0.20, 0.09 | 0.00337, 0.0929 |
| Temperature τ | `temperature` | 1.0 | 0.1 |
| 10-fold CV | `n_folds` | 10 | — |
| 5-fold CV | `n_folds` | — | 5 |

† Empirically determined values that differ from the paper's stated hyperparameters. See notes in `configs/config.py`.

### Datasets

#### DR (Diabetic Retinopathy) — `Datasets/dataloaders.py` → `DRDataset`

- 35,126 fundus images, 5 severity grades (0–4)
- Class distribution: Grade 0: 73.5%, Grade 1: 7%, Grade 2: 15%, Grade 3: 2.5%, Grade 4: 2%
- Labels loaded from `trainLabels.csv` (columns: `image`, `level`)
- Images in `Datasets/DR/train/*.jpeg`

#### BUSI (Breast Ultrasound Images) — `Datasets/dataloaders.py` → `BUSIDataset`

- 780 ultrasound images, 3 classes: Normal (0), Benign (1), Malignant (2)
- Images organized in `Datasets/BUSI/normal/`, `benign/`, `malignant/`
- Mask files (`*_mask.png`) are automatically excluded

### Data Preprocessing — `Datasets/dataloaders.py`

#### Evaluation transform (`build_transform`)
```
Resize(300×300) → ToTensor()   [scales to [0, 1]]
```

#### Training transform (`build_train_transform`)
```
Resize(300×300) → RandomHorizontalFlip(p=0.5)
               → RandomVerticalFlip(p=0.5)
               → RandomRotation(±10°)
               → ColorJitter(brightness=0.1, contrast=0.1)
               → ToTensor()
```

### Class-Stratified Batch Sampling — `Datasets/dataloaders.py` → `StratifiedBatchSampler`

**Paper quote:** *"We use class-stratified batch sampling to ensure the stability of class prototypes even in small batch sizes."*

- `per_class = batch_size // n_classes` samples drawn from each class per batch
- With batch_size=24, n_classes=5: 4 samples per class per batch
- Minority classes are oversampled (reshuffled when exhausted) to ensure every batch contains all classes
- Seed varies per epoch to prevent identical batch sequences across epochs

### Image Caching — `Datasets/dataloaders.py` → `preload_dr_images()`

For DR (35K large images): pre-decodes all JPEG images to 300×300 uint8 tensors in RAM (~9.5 GB) using 16 parallel threads. Optional disk cache stores decoded `.pt` tensors so subsequent runs skip JPEG decoding. Controlled by `--no_cache` and `--cache_dir` flags.

---

## Paper §3 — Cross-Validation Protocol — `training/cross_val.py`

### DR: 10-fold subject-independent CV — `DRCrossValidator`

**Paper quote:** *"Following [18,3,12], we use 10-fold subject-independent cross-validation for evaluation."*

Patient IDs are extracted from filenames (e.g. `10_left.jpeg` → patient `10`). Left and right eye images of the same patient are always assigned to the **same fold**, guaranteeing subject independence. Patients are stratified by their majority class before fold assignment.

### BUSI: 5-fold stratified CV — `BUSICrossValidator`

**Paper quote:** *"Following [12,9], we perform 5-fold subject-independent cross-validation."*

Stratified at the image level (no explicit patient IDs). Each class is split independently into 5 chunks and assigned round-robin to maintain class balance across folds.

### Fold strategy used in `train_dr.py` / `train_busi.py`

```
All items
  └─► DRCrossValidator / BUSICrossValidator
        └─► get_fold(i)  returns (train_raw, val_held_out, test)
              └─► val=test fold is used as both validation and test set
                  (held-out val is merged back into training)
```

This mirrors the paper's evaluation: the test fold accuracy IS the reported metric.

---

## Paper §3 — Training Loop — `training/trainer.py`

### `Trainer` class

Handles one fold of cross-validation. Key responsibilities:

| Component | Implementation |
|---|---|
| Optimizer | `torch.optim.Adam(lr=cfg.lr, weight_decay=cfg.weight_decay)` |
| LR scheduler | `ReduceLROnPlateau(mode='min', factor=0.2, patience=lr_patience)` — monitors val RMSE |
| Early stopping | `EarlyStopping(patience=early_stop_patience)` — monitors val RMSE |
| Checkpoint selection | Saves best checkpoint by **val_acc** (not val loss) |
| AMP | Automatic mixed precision on CUDA only (Tensor Cores) |

### Training step — `_train_epoch()`

```python
for x, y in train_loader:
    # Compute per-batch class weights (inverse frequency of batch labels)
    batch_weights = compute_class_weights(y, n_classes)

    # Forward: all three heads
    z_pcol, z_scolw, pred = model(x)

    # Combined loss Eq.(3)
    loss, components = criterion(z_pcol, z_scolw, pred, y, batch_weights)

    # Backward with gradient clipping (max_norm=1.0)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### Evaluation step — `_eval_epoch()`

**Paper quote:** *"In the testing phase, only the regression head is used for inference."*

At validation/test time, contrastive losses are **not computed** (they require class-stratified batches which are not guaranteed in evaluation loaders). Only the regression head is called via `model.predict(x)`, and RMSE is computed for the LR scheduler and early stopping signal.

### Epoch logging

Each epoch writes to a CSV (`fold{k}_history.csv`) with: epoch, elapsed time, LR, all loss components, val_loss, val_acc, val_mae.

---

## Paper §3 — Evaluation Metrics — `utils/metrics.py`

**Paper quote:** *"we evaluate the performance of our framework in terms of Accuracy (Acc.) and Mean Absolute Error (MAE)."*

Both metrics operate on **rounded** regression outputs:

```python
pred_class = round(pred).clamp(0, n_classes - 1)

Accuracy = mean(pred_class == true_label) × 100
MAE      = mean(|pred_class - true_label|)
```

`confusion_stats()` also computes the class-wise breakdown used in Fig. 2 of the paper (correct / adjacent-class error / non-adjacent error proportions).

---

## Entry Points

### `train_dr.py` — DR 10-fold CV

```bash
# Full 10-fold run (Lightning AI / CUDA recommended)
python train_dr.py --dr_root Datasets/DR --run_dir runs/dr_v6

# Single fold (validate config before committing to all 10 folds)
python train_dr.py --dr_root Datasets/DR --run_dir runs/dr_v6 --folds 0

# Proxy run on 4K subset (local Mac / fast iteration)
python train_dr.py --dr_root Datasets/DR \
    --train_csv Datasets/DR/trainLabels_4k.csv \
    --run_dir runs/dr_proxy --folds 0 --no_cache
```

### `train_busi.py` — BUSI 5-fold CV

```bash
python train_busi.py --busi_root Datasets/BUSI --run_dir runs/busi
```

### `train_dr_sweep.py` / `train_busi_sweep.py` — Hyperparameter Search

WandB Bayesian sweeps defined in `sweep_config_dr.yaml` / `sweep_config_busi.yaml`.

```bash
# Launch sweep agent (uses 4K proxy for fast iteration)
python train_dr_sweep.py --dr_root Datasets/DR \
    --train_csv Datasets/DR/trainLabels_4k.csv \
    --sweep_id <entity/project/sweep_id>
```

`SweepTrainer` (in `train_dr_sweep.py`) extends `Trainer` with:
- Per-epoch WandB metric logging (`running_best_val_acc`)
- Two-stage early abandonment (kills clearly failing trials at epoch 10 and epoch 20)

---

## Implementation Notes vs. Paper

| Topic | Paper states | This implementation |
|---|---|---|
| LR | 1×10⁻³ | 1×10⁻⁴ for DR (lr=1e-3 unstable with batch-level weights) |
| LR patience | 5 epochs | 8 for DR (5-class contrastive landscape needs more time) |
| Early stopping | 13 epochs | 20 (gives model time to recover after LR drops) |
| Class weights ($w_i$) | "inverse frequency of class in dataset" | Computed per **mini-batch** (per author response); with stratified sampling gives ≈ uniform weights |
| Temperature τ | Not specified per-dataset | τ=1.0 for DR (τ<1 causes exp overflow with 5-class ordinal distances); τ=0.1 for BUSI |
| α, β | Not specified | DR: α=0.20, β=0.09 (found via proxy sweep); BUSI: α=0.00337, β=0.0929 |

---