"""
Concept bank for the per-image concept loss (Fix 2).

The frozen BioMedCLIP text tower encodes each clinical phrase to a 512-d vector.
At training time the trainer projects these through the model's (trainable)
concept_proj each batch, so the concept vectors share the image concept space
and the gradient reaches the projection. This mirrors the notebook's
V_concept_builder closure.

The phrase lists below follow the deck's concept ordering. If your notebook used
slightly different wording for DR_CONCEPTS / BUSI_CONCEPTS, paste those exact
strings here so the bank matches the vectors your analysis was built on; the
grades must stay in the same row order as the phrases.
"""

import torch
import torch.nn.functional as F

# DR: 5 ICDR grades, 11 concepts
DR_CONCEPTS = [
    "normal retina", "healthy fundus",                                  # No-DR = 0
    "microaneurysms",                                                   # Mild  = 1
    "dot and blot hemorrhages", "hard exudates", "cotton wool spots",   # Moderate = 2
    "venous beading", "intraretinal microvascular abnormalities",       # Severe = 3
    "neovascularization", "preretinal hemorrhage", "vitreous hemorrhage",  # PDR = 4
]
DR_CONCEPT_GRADES = torch.tensor([0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 4])

# BUSI: 3 classes, 10 concepts
BUSI_CONCEPTS = [
    "no mass", "no architectural distortion",                          # Normal = 0
    "oval shape", "circumscribed margin",
    "parallel orientation", "posterior acoustic enhancement",          # Benign = 1
    "irregular shape", "spiculated margins",
    "nonparallel orientation", "posterior acoustic shadowing",         # Malignant = 2
]
BUSI_CONCEPT_GRADES = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])


def load_biomedclip(model_str="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                    device="cuda"):
    """Load BioMedCLIP and its tokenizer; the text tower is used and frozen."""
    import open_clip
    model = open_clip.create_model_and_transforms(model_str)[0].to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(model_str)
    return model, tokenizer


@torch.no_grad()
def encode_concept_texts(model, tokenizer, phrases, device="cuda") -> torch.Tensor:
    """(C,) phrases -> (C, 512) L2-normalized frozen text features."""
    tokens = tokenizer(phrases).to(device)
    feats = model.encode_text(tokens).float()
    return F.normalize(feats, dim=-1)


def make_concept_builder(model, tokenizer, phrases, project_fn, device="cuda"):
    """Return a closure V_concept_builder() -> (C, concept_dim) with live gradient
    through the model's concept_proj (project_fn = model.project_concepts).
    The raw text features are encoded once; only the projection is recomputed."""
    raw = encode_concept_texts(model, tokenizer, phrases, device)   # (C, 512) frozen
    def builder():
        return project_fn(raw)
    return builder
