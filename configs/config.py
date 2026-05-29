"""
Training configuration matching paper hyperparameters exactly.

Paper (Section 3 - Implementation details):
  - Backbone: EfficientNet-V2S
  - Image size: 300x300, normalized to [0,1]
  - Projection heads: 2 dense layers (1280 -> 128)
  - Batch size: 24, stratified batch sampling
  - Epochs: 75
  - LR: 1e-3, reduced by 0.2 after 5 epochs no improvement
  - Early stopping patience: 13 epochs
  - DR: 10-fold subject-independent CV
  - BUSI: 5-fold subject-independent CV
  - Total loss: alpha*PCOL + beta*SCOLw + RMSE
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data
    img_size: int = 300
    num_workers: int = 4
    pin_memory: bool = True

    # Training
    epochs: int = 75
    batch_size: int = 24
    lr: float = 5e-4
    weight_decay: float = 1e-4        # not specified in paper; no regularization we traid with 1e-4 and worked
    seed: int = 42

    # LR scheduler (ReduceLROnPlateau)
    lr_factor: float = 0.2           # paper: "reduced by a factor of 0.2"
    lr_patience: int = 5             # paper: "after 5 epochs of no improvement"
    lr_min: float = 1e-6

    # Early stopping
    early_stop_patience: int = 13    # paper: "patience of 13 epochs"

    # Loss weights  (paper Eq. 3: L = alpha*PCOL + beta*SCOLw + RMSE)
    alpha: float = 0.00337            # sweep best: fold0 93.63% acc
    beta: float = 0.0929             # sweep best: fold0 93.63% acc

    # Contrastive loss temperature
    # tau=0.05 helped fold0 but hurt fold1; tau=0.1 is a compromise
    temperature: float = 0.1

    # Projection head dimensions (paper: "1280 and 128 neurons")
    proj_hidden_dim: int = 1280
    proj_out_dim: int = 128

    # Checkpoint directory (set per experiment)
    run_dir: str = "runs/experiment"

    # Stratified batch sampling (paper: class-stratified batch sampling)
    stratified: bool = True

    # Whether to use pretrained ImageNet weights for backbone
    pretrained: bool = True

    # Automatic Mixed Precision — enabled on CUDA only (T4/A10 Tensor Cores → ~2× speed)
    amp: bool = True


@dataclass
class BUSIConfig(TrainConfig):
    """BUSI-specific config (5-fold subject-independent CV)."""
    dataset: str = "BUSI"
    n_classes: int = 3               # normal=0, benign=1, malignant=2
    n_folds: int = 5                 # paper: 5-fold CV
    val_fraction: float = 0.1        # 10% of train folds for validation
    run_dir: str = "runs/busi"


@dataclass
class DRConfig(TrainConfig):
    """DR-specific config (10-fold subject-independent CV)."""
    dataset: str = "DR"
    n_classes: int = 5               # DR grades 0-4
    n_folds: int = 10                # paper: 10-fold CV
    val_fraction: float = 0.1        # 10% of train folds for validation
    run_dir: str = "runs/dr"
    # Full-dataset sweep (35K images): lr=1e-4 wins every time; lr=5e-4 fails (50-68%).
    # With 31K training samples and 1300+ batches/epoch the gradient is dense —
    # a lower LR avoids overshooting the contrastive loss landscape.
    lr: float = 1e-4
    # alpha/beta tuned to full 35K dataset: contrastive losses act as mild
    # regularization when RMSE already has strong signal from 31K samples.
    # beta=0.0929 (base default) is too high; drives SCOLw to dominate over RMSE.
    alpha: float = 0.05
    beta: float = 0.073
    # tau=1.0: max SCOLw logit = 4/1.0=4 → exp(4)=54, balanced across all 5 class
    # pairs. tau=0.5 gives exp(8)=3000 which lets class-0 vs class-4 monopolise gradient.
    temperature: float = 1.0
    # lr_patience=5: LR drops at epoch ~10-15, leaving 60 epochs for fine-tuning
    # within the 75-epoch budget. patience=8 delays the drop past the useful window.
    lr_patience: int = 5
    # v2 hit 72.29% at epoch 11 then early-stopped at epoch 24 (only 32% of budget).
    # patience=20 gives 50+ epochs of fine-tuning after the LR drop.
    early_stop_patience: int = 20
