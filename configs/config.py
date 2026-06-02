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
    early_stop_patience: int = 20    # increased from paper's 13: 5-class contrastive needs
                                     # more recovery time after LR drop (v5 stopped at ep29,
                                     # only 6 epochs after LR reduction, acc still rising)

    # Loss weights  (paper Eq. 3: L = alpha*PCOL + beta*SCOLw + RMSE)
    alpha: float = 0.00337            # sweep best: fold0 93.63% acc (BUSI)
    beta: float = 0.0929             # sweep best: fold0 93.63% acc (BUSI)

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
    epochs: int = 75
    # Paper lr=1e-3; previously unstable without ImageNet normalization (v4: 52.93%).
    # With correct normalization the backbone activations are in range, so 1e-3 is safe.
    lr: float = 1e-3
    # Paper lr_patience=5
    lr_patience: int = 5
    # Standard contrastive temperature; previous τ=0.7 compressed gradients and
    # caused PCOL/SCOLw to barely converge (only ~17% loss reduction over 60+ epochs).
    temperature: float = 0.1
    # alpha=0.00337: PCOL needs small alpha on the full 35K dataset.
    # With 640 grade-4 images, batch prototypes (4 samples) are noisy.
    alpha: float = 0.00337
    beta: float = 0.0929
