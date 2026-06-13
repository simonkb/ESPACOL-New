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
    img_size: int = 224              # BiomedCLIP ViT-B/16 native resolution
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

    # Projection head dimensions.  proj_hidden_dim=0 means auto (= backbone.OUT_DIM).
    proj_hidden_dim: int = 0
    proj_out_dim: int = 128

    # Checkpoint directory (set per experiment)
    run_dir: str = "runs/experiment"

    # Stratified batch sampling (paper: class-stratified batch sampling)
    stratified: bool = True

    # Whether to use pretrained weights for backbone
    pretrained: bool = True

    # Automatic Mixed Precision — enabled on CUDA only (T4/A10 Tensor Cores → ~2× speed)
    amp: bool = True

    # Image backbone — BiomedCLIP ViT-B/16 (same model as text encoder)
    image_encoder_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # Use CLIP normalization (mean/std from open_clip) instead of ImageNet normalization
    use_clip_normalization: bool = True
    # Separate LR for the pretrained ViT backbone — much lower than head LR.
    # ViT features are fragile at high LR; 1/20 of head LR is a safe default.
    image_encoder_lr: float = 1e-5

    use_image_text: bool = True
    gamma: float = 0.0929
    lambda_ord_it: float = 1.0
    text_encoder_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    finetune_text_encoder: bool = False
    text_finetune_layers: int = 0
    text_encoder_lr: float = 1e-6
    text_finetune_start_epoch: int = 20


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
    epochs: int = 120            # extended: 10 frozen warmup + 110 fine-tuning
    batch_size: int = 24
    lr: float = 2e-4
    weight_decay: float = 1e-6
    lr_patience: int = 15            # longer patience: reset after backbone unfreeze
    lr_min: float = 1e-7             # lower floor: 3-4 backbone LR drops instead of 2
    early_stop_patience: int = 30
    temperature: float = 0.7
    alpha: float = 0.00662474091401746
    beta: float = 0.05516050165777829
    use_image_text: bool = True
    gamma: float = 0.05
    lambda_ord_it: float = 2.0
    finetune_text_encoder: bool = True
    text_finetune_layers: int = 2
    text_encoder_lr: float = 1e-6
    text_finetune_start_epoch: int = 20
    image_encoder_lr: float = 2e-5   # stable range; no oscillation at this LR
    freeze_backbone_epochs: int = 10 # heads-only warmup before backbone fine-tuning

    # ── TAMO (Text-Anchored Metric Ordinality) ────────────────────────────────
    # use_tamo: enable the TAMO loss and upgrade all heads to DeepProjectionHead.
    # gamma_tamo: weight for L_TAMO in total loss (L_PMD + lambda_orc * L_ORC).
    # lambda_orc: relative weight of ORC term vs PMD within TAMO.
    # Set to True via --use_tamo flag in train_dr.py to activate TAMO.
    # Default False so existing sweep scripts are unaffected.
    use_tamo: bool = False
    gamma_tamo: float = 0.15        # start conservative; scale up after ablation
    lambda_orc: float = 0.5         # ORC is noisier than PMD; down-weight initially
    tamo_huber_delta: float = 0.1   # Huber threshold in normalized distance units
