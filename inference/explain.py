"""
ConceptExplainer: dual explainability for OPTIC-C (OPTICConceptModel).

Given a single fundus image, produces:
  1. Spatial heatmap — GPA tile_weights for the predicted grade overlaid on image
  2. Concept report  — per-concept presence score + top contributing tile locations

Usage:
    explainer = ConceptExplainer(model, text_encoder, device, cfg)
    result = explainer.explain(pil_image)
    # result["grade"]           — int, predicted DR grade 0–4
    # result["grade_label"]     — str, e.g. "Moderate NPDR"
    # result["confidence"]      — float, softmax confidence of predicted grade
    # result["heatmap"]         — np.ndarray (H, W) spatial attention for predicted grade
    # result["concept_scores"]  — dict {concept_name: score} sorted by score desc
    # result["concept_tiles"]   — dict {concept_name: [(tile_row, tile_col, score), ...]}
    # result["proto_logits"]    — np.ndarray (K,) raw prototype logits
"""

from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from configs.clinical_text import DR_CONCEPTS, DR_CLASS_DESCRIPTIONS

GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


class ConceptExplainer:
    """
    Args:
        model:        OPTICConceptModel (on device, in eval mode after loading best ckpt)
        text_encoder: ClinicalTextEncoder (on device, with concept_descriptions loaded)
        device:       torch.device
        tile_grid:    grid size (default 3 → 3×3 tiles + 1 global = 10 total)
        tile_size:    pixel size of each tile (default 300)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        text_encoder: torch.nn.Module | None,
        device: torch.device,
        tile_grid: int = 3,
        tile_size: int = 300,
    ):
        self.model = model
        self.text_encoder = text_encoder
        self.device = device
        self.tile_grid = tile_grid
        self.tile_size = tile_size
        self.concepts = DR_CONCEPTS

    @torch.no_grad()
    def explain(self, image: Image.Image) -> dict:
        """
        Args:
            image: PIL Image (RGB), any size — will be resized to tile mosaic
        Returns:
            dict with keys: grade, grade_label, confidence, heatmap,
                            concept_scores, concept_tiles, proto_logits
        """
        from Datasets.dataloaders import build_tile_transform

        tfm = build_tile_transform(
            tile_size=self.tile_size,
            tile_grid=self.tile_grid,
            augment=False,
        )
        x = tfm(image).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        # Concept + grade text embeddings
        concept_embeds = None
        grade_text_embeds = None
        if self.text_encoder is not None:
            self.text_encoder.eval()
            grade_text_embeds = self.text_encoder()
            concept_embeds = self.text_encoder.get_concept_embeds()

        self.model.eval()
        out = self.model(
            x,
            labels=torch.zeros(1, dtype=torch.long, device=self.device),
            concept_embeds=concept_embeds,
            grade_text_embeds=grade_text_embeds,
        )

        pred = out["pred"]   # (1,) expected grade scalar
        grade = int(pred.round().clamp(0, 4).item())

        # Prototype confidence via softmax over prototype logits
        proto_logits = out.get("proto_logits")
        if proto_logits is not None:
            proto_probs = F.softmax(proto_logits[0].float(), dim=-1)
            confidence = proto_probs[grade].item()
            proto_logits_np = proto_logits[0].float().cpu().numpy()
        else:
            confidence = 0.0
            proto_logits_np = np.zeros(5)

        # ── Spatial heatmap from GPA ──────────────────────────────────────────
        heatmap = None
        tile_weights = out.get("tile_weights")   # (1, K, T)
        if tile_weights is not None:
            # tile_weights[:, grade, :] → attention of predicted grade over T tiles
            grade_weights = tile_weights[0, grade, :].float().cpu().numpy()  # (T,)
            T = len(grade_weights)
            # Tile layout: positions 0..T-2 are 3×3 spatial tiles; T-1 is global
            n_local = self.tile_grid * self.tile_grid
            local_weights = grade_weights[:n_local]   # exclude global tile
            heatmap = local_weights.reshape(self.tile_grid, self.tile_grid)
            # Normalize to [0, 1]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # ── Concept scores ────────────────────────────────────────────────────
        concept_scores = {}
        concept_tiles = {}
        tile_concept_scores = out.get("tile_concept_scores")   # (1, T, C)
        if tile_concept_scores is not None:
            scores = torch.sigmoid(tile_concept_scores[0].float() * 10.0)  # (T, C)
            n_local = self.tile_grid * self.tile_grid
            local_scores = scores[:n_local]   # (T_local, C) — skip global tile

            for ci, concept_name in enumerate(self.concepts):
                per_tile = local_scores[:, ci].cpu().numpy()   # (T_local,)
                mean_score = float(per_tile.mean())
                concept_scores[concept_name] = round(mean_score, 4)

                # Top-2 contributing tiles for this concept
                top_tile_idxs = np.argsort(per_tile)[::-1][:2]
                tile_locs = []
                for ti in top_tile_idxs:
                    row = int(ti) // self.tile_grid
                    col = int(ti) % self.tile_grid
                    tile_locs.append((row, col, round(float(per_tile[ti]), 4)))
                concept_tiles[concept_name] = tile_locs

            # Sort by score descending
            concept_scores = dict(
                sorted(concept_scores.items(), key=lambda kv: kv[1], reverse=True)
            )

        return {
            "grade": grade,
            "grade_label": GRADE_LABELS.get(grade, str(grade)),
            "confidence": round(confidence, 4),
            "heatmap": heatmap,
            "concept_scores": concept_scores,
            "concept_tiles": concept_tiles,
            "proto_logits": proto_logits_np,
        }

    def format_report(self, result: dict) -> str:
        """Human-readable text report for a physician."""
        lines = [
            f"DR Grade: {result['grade']} — {result['grade_label']}",
            f"Prototype confidence: {result['confidence'] * 100:.1f}%",
            "",
            "Clinical indicators (by concept presence score):",
        ]
        for concept, score in result["concept_scores"].items():
            bar = "█" * int(score * 20)
            lines.append(f"  {concept:<45} [{score:.2f}] {bar}")

        return "\n".join(lines)

    def format_json(self, result: dict) -> str:
        """JSON report — heatmap as nested list, concepts as dict."""
        out = {
            "grade": result["grade"],
            "grade_label": result["grade_label"],
            "confidence": result["confidence"],
            "heatmap": result["heatmap"].tolist() if result["heatmap"] is not None else None,
            "concept_scores": result["concept_scores"],
            "concept_tiles": result["concept_tiles"],
            "proto_logits": result["proto_logits"].tolist(),
        }
        return json.dumps(out, indent=2)
