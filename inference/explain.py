"""
Physician-readable concept explanations for DR grade predictions.

Usage:
    from inference.explain import ConceptExplainer

    explainer = ConceptExplainer(model, concept_text_emb, DR_CONCEPTS)
    report = explainer.explain(image_tensor)   # single image (1, T, C, H, W) or (1, C, H, W)
    print(report["text"])

    # Batch inference
    reports = explainer.explain_batch(images)  # list of dicts
"""

from __future__ import annotations

import json
from typing import Optional

import torch
import torch.nn as nn

DR_GRADE_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


class ConceptExplainer:
    """
    Wraps a trained HybridContrastiveOrdinalModel to produce physician-readable
    grade predictions with top-k concept justifications.

    Args:
        model          : trained model with concept_spine attached
        concept_text_emb: (M, 128) concept text embeddings from ClinicalTextEncoder
        concept_names  : list of M concept label strings (e.g. DR_CONCEPTS)
        top_k          : number of top concepts to report per image
        device         : inference device (defaults to model's current device)
    """

    def __init__(
        self,
        model: nn.Module,
        concept_text_emb: torch.Tensor,
        concept_names: list[str],
        top_k: int = 3,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.concept_text_emb = concept_text_emb
        self.concept_names = concept_names
        self.top_k = top_k
        self.device = device or next(model.parameters()).device

        self.model.eval()
        self.concept_text_emb = self.concept_text_emb.to(self.device)

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> dict:
        """
        Single image or batch inference. Returns a list of per-image dicts.

        Each dict contains:
            grade       : int   — predicted DR grade (0-4)
            grade_name  : str   — e.g. "Moderate DR"
            confidence  : float — probability of predicted grade from ordinal head
                                  (or None if regression head used)
            concepts    : list of {"name": str, "score": float}  — top-k concepts
            text        : str   — formatted physician-readable report
            json        : str   — JSON-serializable summary
        """
        x = x.to(self.device)
        if x.dim() == 4:
            x = x.unsqueeze(0)  # (C, H, W) → (1, C, H, W)

        out = self.model.forward(x, concept_text_emb=self.concept_text_emb)

        pred_continuous = out["pred"]                         # (N,)
        pred_grades = pred_continuous.round().clamp(0, 4).long()  # (N,)

        # Confidence from ordinal probs P(Y > k)
        confidence = None
        if "ordinal_probs" in out:
            probs = out["ordinal_probs"]   # (N, K-1)
            # P(Y = k) = P(Y > k-1) - P(Y > k)   (with boundary conditions)
            K = probs.shape[1] + 1
            p_exceed = torch.cat(
                [torch.ones(probs.shape[0], 1, device=probs.device), probs,
                 torch.zeros(probs.shape[0], 1, device=probs.device)], dim=1
            )  # (N, K+1)
            p_class = p_exceed[:, :-1] - p_exceed[:, 1:]  # (N, K)
            p_class = p_class.clamp(min=0.0)
            confidence = p_class.gather(1, pred_grades.unsqueeze(1)).squeeze(1)  # (N,)

        # Concept scores
        results = []
        N = pred_grades.shape[0]
        has_concepts = "c" in out and "w" in out

        for i in range(N):
            grade = pred_grades[i].item()
            grade_name = DR_GRADE_NAMES.get(grade, f"Grade {grade}")
            conf = confidence[i].item() if confidence is not None else None

            concepts = []
            if has_concepts:
                weighted_scores = out["w"] * out["c"][i]           # (M,)
                topk_scores, topk_idx = weighted_scores.topk(self.top_k)
                for rank, (idx, score) in enumerate(
                    zip(topk_idx.tolist(), topk_scores.tolist()), start=1
                ):
                    concepts.append({
                        "rank": rank,
                        "name": self.concept_names[idx],
                        "score": round(float(score), 4),
                    })

            text = _format_report(grade, grade_name, conf, concepts)
            results.append({
                "grade": grade,
                "grade_name": grade_name,
                "confidence": round(conf, 4) if conf is not None else None,
                "concepts": concepts,
                "text": text,
                "json": json.dumps({
                    "grade": grade,
                    "grade_name": grade_name,
                    "confidence": round(conf, 4) if conf is not None else None,
                    "top_concepts": [
                        {"name": c["name"], "score": c["score"]} for c in concepts
                    ],
                }),
            })

        return results[0] if len(results) == 1 else results

    def explain_batch(self, x: torch.Tensor) -> list[dict]:
        """Convenience wrapper — always returns a list."""
        result = self.explain(x)
        return result if isinstance(result, list) else [result]


def _format_report(
    grade: int,
    grade_name: str,
    confidence: Optional[float],
    concepts: list[dict],
) -> str:
    conf_str = f"  —  confidence: {confidence * 100:.1f}%" if confidence is not None else ""
    lines = [f"{grade_name} (Grade {grade}){conf_str}"]

    if concepts:
        lines.append("Clinical indicators:")
        for c in concepts:
            marker = "●" if c["rank"] == 1 else " "
            primary = "  ← primary driver" if c["rank"] == 1 else ""
            lines.append(f"  {marker} {c['name']:<28s} [score: {c['score']:.2f}]{primary}")
    else:
        lines.append("(Concept spine not active — enable --use_concept_spine for indicators)")

    return "\n".join(lines)
