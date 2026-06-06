from __future__ import annotations

import torch
import torch.nn as nn


class ImageTextOrdinalLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, lambda_ord: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_ord = lambda_ord

    def forward(
        self,
        image_embeddings: torch.Tensor,
        labels: torch.Tensor,
        text_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        sim = image_embeddings @ text_prototypes.t()
        n_classes = text_prototypes.shape[0]
        class_ids = torch.arange(n_classes, device=image_embeddings.device)
        ord_dist = (labels.long().unsqueeze(1) - class_ids.unsqueeze(0)).abs().float()
        ord_dist = ord_dist / max(n_classes - 1, 1)
        logits = (sim + self.lambda_ord * ord_dist) / self.temperature
        pos_logits = sim.gather(1, labels.long().unsqueeze(1)).squeeze(1) / self.temperature
        log_denom = torch.logsumexp(logits, dim=1)
        return -(pos_logits - log_denom).mean()
