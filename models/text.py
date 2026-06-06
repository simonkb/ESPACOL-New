from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ClinicalTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        class_descriptions: Dict[int, str],
        proj_out_dim: int,
        device: torch.device,
    ):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "Image-text extension requires open_clip_torch. Install it with: "
                "pip install open_clip_torch"
            ) from exc

        self.class_ids = sorted(class_descriptions)
        texts = [class_descriptions[k] for k in self.class_ids]
        self.text_model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.text_model = self.text_model.to(device).eval()
        for p in self.text_model.parameters():
            p.requires_grad = False

        with torch.no_grad():
            tokens = self.tokenizer(["dimension probe"]).to(device)
            text_dim = self.text_model.encode_text(tokens).shape[-1]
            text_tokens = self.tokenizer(texts).to(device)
            raw = self.text_model.encode_text(text_tokens).float()
            raw = F.normalize(raw, dim=-1)

        self.register_buffer("raw_text_embeddings", raw)
        self.projection = TextProjectionHead(text_dim, proj_out_dim)

    def forward(self) -> torch.Tensor:
        return self.projection(self.raw_text_embeddings)
