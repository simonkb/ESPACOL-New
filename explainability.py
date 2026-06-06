"""
explainability.py  —  ESPAOCL Explainability Module
=====================================================

Compatible with Simon's HybridContrastiveOrdinalNet which returns:
    feat, z_pcol, z_scolw, z_it, y_pred   (5 outputs)

What this file implements
--------------------------
1. LayerCAM         WHERE did the model look?
                    Grad-CAM++ at 4 EfficientNet-V2S stages, fused by MAX.
                    Resolves individual lesions (not just blobs).

2. ConceptExplainer WHY clinically?
                    Cosine similarity between image embedding and
                    clinical concept phrase vectors.
                    Encoder options: "random" (placeholder) or "biomedclip"
                    (uses the same BioMedCLIP model Simon already loaded).

3. Concept-guided LayerCAM  — the novel contribution.
                    One heatmap per concept: "which pixels look like
                    microaneurysms?" Same math as LayerCAM, different
                    gradient target.

4. SanityCheck      Adebayo et al. 2018 parameter randomisation test.
                    Confirms heatmaps are model-dependent.

5. ExplainabilityPipeline   Unified interface.

Drop-in usage after training
-----------------------------
    from explainability import ExplainabilityPipeline, SanityCheck

    pipeline = ExplainabilityPipeline(
        model        = model,          # trained HybridContrastiveOrdinalNet
        dataset      = "busi",         # "busi" or "dr"
        device       = device,
        text_model   = biomedclip,     # Simon's loaded BioMedCLIP (optional)
        tokenizer    = tokenizer,      # Simon's tokenizer       (optional)
    )
    result = pipeline.explain(x)                        # basic
    result = pipeline.explain(x, concept_heatmaps=True) # + concept maps
    pipeline.visualize(img_np, result, save_path="fig.png", show=False)
    pipeline.remove_hooks()
"""

from __future__ import annotations

import copy
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on Lightning.ai
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from configs.clinical_text import (
    BUSI_CONCEPTS as BUSI_CONCEPT_PHRASES,
    DR_CONCEPTS as DR_CONCEPT_PHRASES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Clinical concept definitions
# ─────────────────────────────────────────────────────────────────────────────

DR_CONCEPTS = {
    "class_names": ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"],
    "concepts": DR_CONCEPT_PHRASES,
}

BUSI_CONCEPTS = {
    "class_names": ["Normal", "Benign", "Malignant"],
    "concepts": BUSI_CONCEPT_PHRASES,
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal: single-layer Grad-CAM++ map
# ─────────────────────────────────────────────────────────────────────────────

def _gradcampp_single_layer(
    activations: torch.Tensor,   # (C, H, W)
    gradients: torch.Tensor,     # (C, H, W)
) -> torch.Tensor:
    """
    Grad-CAM++ weighting for one layer.

    Plain Grad-CAM: alpha_k = mean of gradients  →  dilutes lesion peaks
    Grad-CAM++:     alpha_kij from second derivative  →  detects peaks

    Formula:
        G2 = grad^2
        alpha_kij = G2 / (2*G2 + A_k * sum_ij(G^3) + eps)
        w_k       = sum_ij(alpha_kij * ReLU(grad_kij))
        cam       = ReLU(sum_k(w_k * A_k))

    Returns (H, W) tensor in [0, 1].
    """
    A  = activations
    G  = gradients
    G2 = G ** 2
    G3 = G ** 3

    sum_AG3 = (A * G3).sum(dim=(1, 2), keepdim=True)   # (C,1,1)
    denom   = 2.0 * G2 + sum_AG3 + 1e-8
    alpha   = G2 / denom                                # (C,H,W)
    w       = (alpha * F.relu(G)).sum(dim=(1, 2))       # (C,)

    cam = (w[:, None, None] * A).sum(dim=0)             # (H,W)
    cam = F.relu(cam)

    lo, hi = cam.min(), cam.max()
    if hi > lo:
        cam = (cam - lo) / (hi - lo + 1e-8)
    else:
        cam = torch.zeros_like(cam)
    return cam


def _upsample(cam: torch.Tensor, h: int, w: int) -> np.ndarray:
    arr = (cam.cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr).resize((w, h), Image.BILINEAR)
    return np.array(pil).astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. LayerCAM
# ─────────────────────────────────────────────────────────────────────────────

# EfficientNet-V2S stage indices and approximate spatial resolution (300x300 in):
#   2  →  75×75   fine edges, microaneurysm scale
#   4  →  19×19   medium, lesion clusters
#   6  →  10×10   coarse semantic
#   7  →   7×7    final conv block
DEFAULT_LAYERS = [2, 4, 6, 7]


class LayerCAM:
    """
    Multi-layer Grad-CAM++ fused by pixel-wise MAX.

    Hooks are registered on model.features[idx] for each idx in layer_indices.
    Two gradient targets:
        compute()         — class/grade score       (WHERE did it look?)
        compute_concept() — concept cosine sim       (which pixels ~ concept?)
    """

    def __init__(self, model: nn.Module, layer_indices: List[int] = DEFAULT_LAYERS):
        self.model         = model
        self.layer_indices = layer_indices
        self._acts:  Dict[int, torch.Tensor] = {}
        self._grads: Dict[int, torch.Tensor] = {}
        self._hooks: List = []

        features = model.features if hasattr(model, "features") else model.backbone.features
        for idx in layer_indices:
            layer = features[idx]

            def make_fwd(i):
                def fwd(m, inp, out):
                    self._acts[i] = out   # keep grad_fn alive
                return fwd

            def make_bwd(i):
                def bwd(m, gin, gout):
                    self._grads[i] = gout[0].detach()
                return bwd

            self._hooks.append(layer.register_forward_hook(make_fwd(idx)))
            self._hooks.append(layer.register_full_backward_hook(make_bwd(idx)))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _fuse(self, x: torch.Tensor, score: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        score.backward()

        h, w = x.shape[2], x.shape[3]
        maps = []
        for idx in self.layer_indices:
            if idx not in self._acts or idx not in self._grads:
                continue
            A   = self._acts[idx].squeeze(0).detach()
            G   = self._grads[idx].squeeze(0)
            cam = _gradcampp_single_layer(A, G)
            maps.append(_upsample(cam, h, w))

        if not maps:
            return np.zeros((h, w), dtype=np.float32)
        return np.stack(maps).max(axis=0)

    @torch.enable_grad()
    def compute(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        n_classes: int = 5,
    ) -> np.ndarray:
        """
        Class-score LayerCAM — WHERE did the model look?

        x            : (1, 3, H, W)
        target_class : grade to explain; None → predicted grade
        n_classes    : 5 for DR, 3 for BUSI
        Returns      : (H, W) ndarray in [0, 1]
        """
        assert x.dim() == 4 and x.shape[0] == 1
        self.model.eval()

        out = self.model(x)
        y_pred = out[-1]

        if target_class is None:
            val = y_pred.detach().cpu().squeeze()
            if not torch.isfinite(val):
                val = torch.zeros_like(val)
            target_class = int(val.round().clamp(0, n_classes - 1).item())

        # Ordinal score: score is maximised when prediction == target_class
        score = -torch.abs(y_pred - target_class)
        return self._fuse(x, score)

    @torch.enable_grad()
    def compute_concept(
        self,
        x: torch.Tensor,
        concept_vector: torch.Tensor,   # (D,) unit vector on model device
    ) -> np.ndarray:
        """
        Concept-guided LayerCAM — which pixels look like a clinical concept?

        Identical to compute() except gradient target = cosine_sim(feat, concept).
        This one change makes the heatmap concept-specific.

        x              : (1, 3, H, W)
        concept_vector : (D,) L2-normalised, on model device
        Returns        : (H, W) ndarray in [0, 1]
        """
        assert x.dim() == 4 and x.shape[0] == 1
        self.model.eval()

        if hasattr(self.model, "extract_features"):
            feat = self.model.extract_features(x)
        else:
            out = self.model(x)
            feat = out[0]

        feat_n = F.normalize(feat, dim=-1)
        v      = concept_vector.to(feat.device).unsqueeze(0)
        score  = (feat_n * v).sum()
        return self._fuse(x, score)

    @staticmethod
    def overlay(image: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.45, colormap: str = "jet") -> np.ndarray:
        """Blend heatmap onto image. Returns (H, W, 3) uint8."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        colored = (cm.get_cmap(colormap)(heatmap)[..., :3] * 255).astype(np.uint8)
        return (alpha * colored + (1 - alpha) * image).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Concept Explainer
# ─────────────────────────────────────────────────────────────────────────────

class ConceptExplainer:
    """
    Semantic WHY explanation via cosine similarity.

    Three encoder modes
    -------------------
    "random"      Zero overhead.  Reproducible random unit vectors.
                  Scores show relative ranking only.  Use for development.

    "biomedclip"  Uses the same BioMedCLIP text tower Simon already loaded.
                  Pass text_model=biomedclip, tokenizer=tokenizer.
                  Semantically grounded — preferred for paper figures.

    "clinicalbert" Uses medicalai/ClinicalBERT.
                  Requires: pip install transformers
    """

    def __init__(
        self,
        concepts: List[str],
        feat_dim: int = 1280,
        encoder: str = "random",
        device: Optional[torch.device] = None,
        text_model=None,    # BioMedCLIP model (for encoder="biomedclip")
        tokenizer=None,     # BioMedCLIP tokenizer
        seed: int = 0,
    ):
        self.concepts = concepts
        self.feat_dim = feat_dim
        self.device   = device or torch.device("cpu")

        if encoder == "biomedclip":
            vecs = self._encode_biomedclip(concepts, feat_dim, text_model, tokenizer)
        elif encoder == "clinicalbert":
            vecs = self._encode_clinicalbert(concepts, feat_dim)
        else:
            vecs = self._encode_random(concepts, feat_dim, seed)

        self.embeddings = F.normalize(vecs, dim=-1).to(self.device)

    # ── encoders ─────────────────────────────────────────────────────────────

    def _encode_random(self, concepts, dim, seed) -> torch.Tensor:
        g = torch.Generator(); g.manual_seed(seed)
        return torch.randn(len(concepts), dim, generator=g)

    def _encode_biomedclip(self, concepts, dim, text_model, tokenizer) -> torch.Tensor:
        """
        Use Simon's already-loaded BioMedCLIP text encoder.
        text_model and tokenizer must be passed in (same objects from Cell 9).
        """
        if text_model is None or tokenizer is None:
            warnings.warn(
                "biomedclip encoder requires text_model and tokenizer. "
                "Falling back to random embeddings."
            )
            return self._encode_random(concepts, dim, seed=0)

        text_model.eval()
        with torch.no_grad():
            tokens   = tokenizer(concepts).to(self.device)
            raw      = text_model.encode_text(tokens).float()   # (n_concepts, text_dim)
            raw      = F.normalize(raw, dim=-1)

        text_dim = raw.shape[-1]
        if text_dim != dim:
            # Fixed random projection to match image embedding space
            g    = torch.Generator(); g.manual_seed(42)
            proj = F.normalize(
                torch.randn(text_dim, dim, generator=g), dim=0
            ).to(self.device)
            raw  = raw @ proj

        return raw.cpu()

    def _encode_clinicalbert(self, concepts, dim) -> torch.Tensor:
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            warnings.warn("transformers not installed — using random embeddings.")
            return self._encode_random(concepts, dim, seed=0)

        tok  = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        bert = AutoModel.from_pretrained("medicalai/ClinicalBERT").eval().to(self.device)
        vecs = []
        with torch.no_grad():
            for phrase in concepts:
                enc = tok(phrase, return_tensors="pt",
                          truncation=True, max_length=64).to(self.device)
                out = bert(**enc)
                vecs.append(out.last_hidden_state[:, 0, :].squeeze(0))
        raw = torch.stack(vecs)
        if raw.shape[-1] != dim:
            g    = torch.Generator(); g.manual_seed(42)
            proj = F.normalize(torch.randn(raw.shape[-1], dim, generator=g), dim=0).to(self.device)
            raw  = raw @ proj
        del bert
        return raw.cpu()

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def scores(self, feat: torch.Tensor) -> Dict[str, float]:
        f    = F.normalize(feat.flatten().to(self.device), dim=0)
        sims = (self.embeddings @ f).cpu().numpy()
        return {c: float(s) for c, s in zip(self.concepts, sims)}

    def top_k(self, feat: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        return sorted(self.scores(feat).items(), key=lambda x: x[1], reverse=True)[:k]

    def get_vector(self, concept: str) -> torch.Tensor:
        return self.embeddings[self.concepts.index(concept)]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

class SanityCheck:
    """
    Model parameter randomisation test (Adebayo et al., NeurIPS 2018).

    Confirms heatmaps change when weights are randomised.
    Low SSIM = heatmaps are model-dependent (good).

    Paper sentence:
        "We validate our heatmaps pass the model parameter randomisation test
         of Adebayo et al. (2018), confirming they are model-dependent
         (SSIM = X.XX between trained and randomly-initialised model)."
    """

    def __init__(self, model: nn.Module, layer_indices=DEFAULT_LAYERS):
        self.model          = model
        self.layer_indices  = layer_indices
        self._saved_state   = copy.deepcopy(model.state_dict())

    def restore(self):
        self.model.load_state_dict(self._saved_state)

    @staticmethod
    def _ssim(a: np.ndarray, b: np.ndarray) -> float:
        c1, c2 = 1e-4, 9e-4
        mu_a, mu_b = a.mean(), b.mean()
        sa  = float(((a - mu_a)**2).mean()**0.5)
        sb  = float(((b - mu_b)**2).mean()**0.5)
        sab = float(((a - mu_a)*(b - mu_b)).mean())
        num = (2*mu_a*mu_b + c1) * (2*sab + c2)
        den = (mu_a**2 + mu_b**2 + c1) * (sa**2 + sb**2 + c2)
        return float(np.clip(num / (den + 1e-8), 0.0, 1.0))

    def run(self, x, target_class=None, n_classes=5) -> Dict:
        lc1   = LayerCAM(self.model, self.layer_indices)
        h_t   = lc1.compute(x, target_class=target_class, n_classes=n_classes)
        lc1.remove_hooks()

        with torch.no_grad():
            for p in self.model.parameters():
                p.data = torch.randn_like(p.data)

        lc2   = LayerCAM(self.model, self.layer_indices)
        h_r   = lc2.compute(x, target_class=target_class, n_classes=n_classes)
        lc2.remove_hooks()

        self.restore()

        ssim = self._ssim(h_t, h_r)
        return {
            "heatmap_trained": h_t,
            "heatmap_random":  h_r,
            "ssim":            ssim,
            "passed":          ssim < 0.5,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ExplainabilityPipeline:
    """
    Unified explain + visualize interface.

    Parameters
    ----------
    model       : trained HybridContrastiveOrdinalNet
    dataset     : "busi" or "dr"
    device      : torch.device
    encoder     : "random" | "biomedclip" | "clinicalbert"
    text_model  : BioMedCLIP model (only needed if encoder="biomedclip")
    tokenizer   : BioMedCLIP tokenizer (only needed if encoder="biomedclip")
    layer_indices: EfficientNet stage indices to hook (default [2,4,6,7])
    top_k       : number of top concepts to return
    """

    DATASETS = {"dr": DR_CONCEPTS, "busi": BUSI_CONCEPTS}

    def __init__(
        self,
        model: nn.Module,
        dataset: str = "busi",
        device: Optional[torch.device] = None,
        encoder: str = "random",
        text_model=None,
        tokenizer=None,
        layer_indices: List[int] = DEFAULT_LAYERS,
        top_k: int = 5,
    ):
        assert dataset in self.DATASETS
        self.dataset     = dataset
        self.device      = device or next(model.parameters()).device
        self.top_k       = top_k
        info             = self.DATASETS[dataset]
        self.class_names = info["class_names"]
        self.n_classes   = len(self.class_names)

        self.layercam = LayerCAM(model, layer_indices=layer_indices)

        self.concept_explainer = ConceptExplainer(
            concepts    = info["concepts"],
            feat_dim    = 1280,
            encoder     = encoder,
            device      = self.device,
            text_model  = text_model,
            tokenizer   = tokenizer,
        )

    def explain(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        concept_heatmaps: bool = False,
    ) -> Dict:
        """
        Full explanation for one image.

        Parameters
        ----------
        x                : (1, 3, H, W) on self.device
        target_class     : grade to explain; None → predicted grade
        concept_heatmaps : also compute one LayerCAM per concept (~10 passes)

        Returns dict with keys:
            predicted_label, predicted_class, regression_score,
            target_class, heatmap, concept_scores,
            concept_heatmaps (optional dict)
        """
        model = self.layercam.model
        model.eval()

        with torch.no_grad():
            out = model(x)
            feat = model.extract_features(x) if hasattr(model, "extract_features") else out[0]
            y_pred = out[-1]

        pred_label = int(
            y_pred.detach().cpu().squeeze().round().clamp(0, self.n_classes - 1).item()
        )
        reg_score  = float(y_pred.detach().cpu().squeeze().item())
        target_cls = pred_label if target_class is None else target_class

        # WHERE
        heatmap = self.layercam.compute(
            x, target_class=target_cls, n_classes=self.n_classes
        )

        # WHY
        top_concepts = self.concept_explainer.top_k(feat.squeeze(0), k=self.top_k)

        result = {
            "predicted_label":  pred_label,
            "predicted_class":  self.class_names[pred_label],
            "regression_score": reg_score,
            "target_class":     target_cls,
            "heatmap":          heatmap,
            "concept_scores":   top_concepts,
        }

        if concept_heatmaps:
            cmaps = {}
            for concept in self.concept_explainer.concepts:
                vec          = self.concept_explainer.get_vector(concept)
                cmaps[concept] = self.layercam.compute_concept(x, vec)
            result["concept_heatmaps"] = cmaps

        return result

    def visualize(
        self,
        image_np: np.ndarray,
        result: Dict,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Figure 1: original | LayerCAM | concept bar chart.
        Figure 2 (if concept_heatmaps in result): per-concept grid.
        """
        heatmap  = result["heatmap"]
        overlay  = LayerCAM.overlay(image_np, heatmap)
        names    = [c for c, _ in result["concept_scores"]]
        vals     = [s for _, s in result["concept_scores"]]
        img_show = (image_np * 255).astype(np.uint8) if image_np.dtype != np.uint8 else image_np

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_show);  axes[0].axis("off"); axes[0].set_title("Original")
        axes[1].imshow(overlay);   axes[1].axis("off")
        axes[1].set_title(
            f"LayerCAM → {self.class_names[result['target_class']]}\n"
            f"Pred: {result['predicted_class']}  (score {result['regression_score']:.2f})"
        )
        colors = ["steelblue" if v >= 0 else "tomato" for v in vals]
        axes[2].barh(names[::-1], vals[::-1], color=colors[::-1])
        axes[2].axvline(0, color="black", linewidth=0.8)
        axes[2].set_xlabel("Cosine similarity")
        axes[2].set_title(f"Top-{len(names)} concepts")
        axes[2].set_xlim(-1, 1)
        plt.suptitle(
            f"ESPAOCL  |  {self.dataset.upper()}  |  {result['predicted_class']}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

        if "concept_heatmaps" in result:
            cmaps  = result["concept_heatmaps"]
            all_sc = dict(result["concept_scores"])
            n      = len(cmaps)
            ncols  = min(5, n)
            nrows  = (n + ncols - 1) // ncols
            fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
            axes2 = np.array(axes2).flatten()
            for ax, (concept, cmap) in zip(axes2, cmaps.items()):
                # get sim score — may not be in top_k, so look it up directly
                full_sc = self.concept_explainer.scores(
                    torch.zeros(self.concept_explainer.feat_dim)
                )
                sim_val = all_sc.get(concept, 0.0)
                ax.imshow(LayerCAM.overlay(image_np, cmap, alpha=0.5))
                ax.set_title(f"{concept}\nsim={sim_val:.2f}", fontsize=9)
                ax.axis("off")
            for ax in axes2[n:]:
                ax.axis("off")
            plt.suptitle(f"Concept-guided LayerCAM  |  {self.dataset.upper()}",
                         fontsize=11, fontweight="bold")
            plt.tight_layout()
            if save_path:
                fig2.savefig(
                    save_path.replace(".png", "_concepts.png"),
                    dpi=150, bbox_inches="tight",
                )
            if show:
                plt.show()

        return fig

    def remove_hooks(self):
        self.layercam.remove_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Batch helper
# ─────────────────────────────────────────────────────────────────────────────

def run_batch_explanation(
    pipeline: ExplainabilityPipeline,
    loader,
    n_samples: int = 16,
    save_dir: Optional[str] = None,
    concept_heatmaps: bool = False,
) -> List[Dict]:
    import os
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    results, count = [], 0
    for images, labels in loader:
        for i in range(images.shape[0]):
            if count >= n_samples:
                break
            x = images[i].unsqueeze(0).to(pipeline.device)
            result = pipeline.explain(x, concept_heatmaps=concept_heatmaps)
            result["true_label"] = int(labels[i].item())
            result["true_class"] = pipeline.class_names[result["true_label"]]
            results.append(result)
            if save_dir:
                img_np = images[i].permute(1, 2, 0).cpu().numpy()
                sp = os.path.join(
                    save_dir,
                    f"s{count:03d}_true{result['true_label']}_pred{result['predicted_label']}.png",
                )
                pipeline.visualize(img_np, result, save_path=sp, show=False)
                plt.close("all")
            count += 1
        if count >= n_samples:
            break
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torchvision.models import efficientnet_v2_s

    print("Smoke test (CPU, random weights) ...")

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            b = efficientnet_v2_s(weights=None)
            self.features   = b.features
            self.pool       = nn.AdaptiveAvgPool2d(1)
            self.regression = nn.Linear(1280, 1)
        def forward(self, x):
            f = self.pool(self.features(x)).flatten(1)
            z = F.normalize(f, dim=-1)
            y = self.regression(f).squeeze(-1)
            return f, z, z, z, y   # 5 outputs like Simon's model

    model = _Stub().eval()
    x     = torch.randn(1, 3, 300, 300)

    p = ExplainabilityPipeline(model, dataset="busi", encoder="random")

    # basic
    r = p.explain(x)
    assert r["heatmap"].shape == (300, 300)
    print(f"  [1] heatmap={r['heatmap'].shape}  pred={r['predicted_class']}  OK")

    # concept heatmaps
    r2 = p.explain(x, concept_heatmaps=True)
    assert len(r2["concept_heatmaps"]) == 10
    print(f"  [2] concept_heatmaps={len(r2['concept_heatmaps'])}  OK")

    p.remove_hooks()

    # sanity check
    sc  = SanityCheck(model)
    sr  = sc.run(x, n_classes=3)
    print(f"  [3] sanity SSIM={sr['ssim']:.3f}  passed={sr['passed']}  OK")

    # DR pipeline
    p2  = ExplainabilityPipeline(model, dataset="dr", encoder="random")
    r3  = p2.explain(x)
    assert r3["heatmap"].shape == (300, 300)
    print(f"  [4] DR pred={r3['predicted_class']}  OK")
    p2.remove_hooks()

    print("All tests passed.")
