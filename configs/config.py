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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    # Data
    img_size: int = 300
    num_workers: int = 4
    pin_memory: bool = True

    # Training
    epochs: int = 75
    batch_size: int = 24
    lr: float = 1e-3
    weight_decay: float = 1e-4       # not specified; standard choice
    seed: int = 42

    # LR scheduler (ReduceLROnPlateau)
    lr_factor: float = 0.2           # paper: "reduced by a factor of 0.2"
    lr_patience: int = 5             # paper: "after 5 epochs of no improvement"
    lr_min: float = 1e-6

    # Early stopping
    early_stop_patience: int = 13    # paper: "patience of 13 epochs"

    # Loss weights  (paper Eq. 3: L = alpha*PCOL + beta*SCOLw + RMSE)
    alpha: float = 1.0               # not specified; equal weighting
    beta: float = 1.0                # not specified; equal weighting

    # Contrastive loss temperature
    # tau=0.07 is SupCon standard; ordinal distances are raw |y_a - y_n|
    # If ordinal terms dominate, increase tau or set normalize_ordinal=True
    temperature: float = 0.07

    # Projection head dimensions (paper: "1280 and 128 neurons")
    proj_hidden_dim: int = 1280
    proj_out_dim: int = 128

    # Checkpoint directory (set per experiment)
    run_dir: str = "runs/experiment"

    # Stratified batch sampling (paper: class-stratified batch sampling)
    stratified: bool = True

    # Whether to use pretrained ImageNet weights for backbone
    pretrained: bool = True


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
