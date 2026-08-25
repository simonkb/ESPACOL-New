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

    # Resume training from existing best checkpoint (e.g. after SLURM preemption).
    # Default False — new jobs always start from epoch 1.
    resume: bool = False

    # LR multiplier for randomly-initialised OPTIC components (CTOT, GPA, ODH)
    # relative to the pretrained EfficientNet backbone. 1.0 = same LR for all.
    # Recommended: 2.5 when use_tile_transformer=True so the Transformer gets
    # enough gradient signal before ReduceLROnPlateau cuts the LR.
    new_component_lr_mult: float = 1.0

    # Stratified batch sampling (paper: class-stratified batch sampling)
    stratified: bool = True

    # Whether to use pretrained ImageNet weights for backbone
    pretrained: bool = True

    # Automatic Mixed Precision — enabled on CUDA only (T4/A10 Tensor Cores → ~2× speed)
    amp: bool = True

    gamma: float = 0.0929
    text_encoder_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    finetune_text_encoder: bool = False
    text_finetune_layers: int = 0
    text_encoder_lr: float = 1e-6
    text_finetune_start_epoch: int = 20

    # ── OPTIC architecture flags (all default off → identical to baseline) ──
    # CrossTileOrdinalTransformer: replaces AttentionPool with tile-aware Transformer
    use_tile_transformer: bool = False
    tile_transformer_dim: int = 512
    tile_transformer_nhead: int = 8
    tile_transformer_layers: int = 2
    tile_transformer_dropout: float = 0.1

    # GradePrototypeAttention: per-tile grade evidence for intrinsic explainability
    use_grade_prototypes: bool = False

    # OrdinalDistributionHead: CORAL-style distributional prediction (replaces RMSE)
    use_ordinal_head: bool = False

    # OrdinalStochasticDominanceLoss: pairwise CDF-level ranking loss
    use_osd_loss: bool = False
    lambda_osd: float = 0.5
    osd_margin: float = 0.0

    # TileConsistencyLoss: penalise tile evidence conflicting with image prediction
    use_tile_consistency: bool = False
    lambda_tcl: float = 0.1
    tcl_margin: float = 0.0   # 0 = penalise any tile-image disagreement (recommended)

    # GradePrototypeCELoss: direct supervision on GPA tile evidence
    # Forces grade prototypes to be discriminative: mean tile evidence across
    # tiles should predict the correct grade. Without this, GPA receives <0.001
    # weighted gradient and its prototypes are essentially untrained.
    lambda_gpa: float = 0.0

    # Backbone freeze schedule: freeze EfficientNet for the first N epochs so
    # randomly-initialised components (CTOT, GPA) can stabilise on fixed pretrained
    # features before joint fine-tuning begins. 0 = no freeze (default).
    backbone_freeze_epochs: int = 0

    # ── OPTIC-C: Concept-Grounded Grade Prototype flags ──
    # ConceptGradePrototypeModule replaces SCOLw as the dominant gradient signal.
    # Set alpha=0, beta=0 when use_concept_prototype=True.
    use_concept_prototype: bool = False

    # L_proto_CE: cosine prototype CrossEntropy — dominant novel loss
    lambda_proto_ce: float = 0.0

    # L_tile_concept: per-tile concept BCE vs clinical grade-concept soft targets
    lambda_tile_concept: float = 0.0

    # Number of DR concepts (must match len(DR_CONCEPTS) in clinical_text.py)
    n_concepts: int = 9

    # Temperature for cosine prototype similarity (lower = sharper prototype boundaries)
    proto_temperature: float = 0.1

    # Label smoothing for proto_CE cross-entropy — regularises the dominant loss
    # and prevents it from collapsing to near-zero on training data (overfitting).
    proto_label_smoothing: float = 0.0

    # Switch to CosineAnnealingLR at the backbone unfreeze point instead of
    # continuing with ReduceLROnPlateau. Deterministic decay from base LR to
    # lr_min over the remaining (epochs - backbone_freeze_epochs) epochs.
    # Eliminates the patience-based timing luck that causes some folds to miss
    # their second LR drop when val_acc oscillates at the first reduced LR.
    use_cosine_lr: bool = False


@dataclass
class BUSIConfig(TrainConfig):
    """BUSI-specific config (5-fold subject-independent CV)."""
    dataset: str = "BUSI"
    n_classes: int = 3               # normal=0, benign=1, malignant=2
    n_folds: int = 5                 # paper: 5-fold CV
    val_fraction: float = 0.1        # 10% of train folds for validation
    run_dir: str = "runs/busi"
    use_multi_tile: bool = False
    tile_grid: int = 3


@dataclass
class DRConfig(TrainConfig):
    """DR-specific config (10-fold subject-independent CV)."""
    dataset: str = "DR"
    n_classes: int = 5               # DR grades 0-4
    n_folds: int = 10                # paper: 10-fold CV
    val_fraction: float = 0.1        # 10% of train folds for validation
    run_dir: str = "runs/dr"
    epochs: int = 75
    batch_size: int = 24
    # Paper lr=1e-3; previously unstable without ImageNet normalization (v4: 52.93%).
    # With correct normalization the backbone activations are in range, so 1e-3 is safe.
    lr: float = 2e-4
    weight_decay: float = 1e-6
    # Paper lr_patience=5
    lr_patience: int = 8
    early_stop_patience: int = 20
    # Standard contrastive temperature; previous τ=0.7 compressed gradients and
    # caused PCOL/SCOLw to barely converge (only ~17% loss reduction over 60+ epochs).
    temperature: float = 0.7
    # alpha=0.00337: PCOL needs small alpha on the full 35K dataset.
    # With 640 grade-4 images, batch prototypes (4 samples) are noisy.
    alpha: float = 0.00662474091401746
    beta: float = 0.05516050165777829
    gamma: float = 0.05
    finetune_text_encoder: bool = True
    text_finetune_layers: int = 2
    text_encoder_lr: float = 1e-6
    text_finetune_start_epoch: int = 20
    use_multi_tile: bool = False
    tile_grid: int = 3
