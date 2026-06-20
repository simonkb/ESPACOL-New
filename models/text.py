from __future__ import annotations

from typing import Dict, Iterable

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
        finetune_text_encoder: bool = False,
        finetune_layers: int = 0,
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
        self.finetune_text_encoder = finetune_text_encoder
        self.finetune_layers = finetune_layers

        for p in self.text_model.parameters():
            p.requires_grad = False

        with torch.no_grad():
            tokens = self.tokenizer(["dimension probe"]).to(device)
            text_dim = self.text_model.encode_text(tokens).shape[-1]

            text_tokens = self.tokenizer(texts).to(device)
            raw = self.text_model.encode_text(text_tokens).float()
            raw = F.normalize(raw, dim=-1)

        self.register_buffer("text_tokens", text_tokens)
        self.register_buffer("raw_text_embeddings", raw)

        self.projection = TextProjectionHead(text_dim, proj_out_dim)

        if self.finetune_text_encoder:
            self.set_text_finetune(False)

    def _text_transformer_blocks(self):
        candidates = [
            # BiomedCLIP / PubMedBERT path found in your model:
            # text.transformer.encoder.layer.0 ... layer.11
            ("text", "transformer", "encoder", "layer"),

            # Other possible OpenCLIP / HF paths
            ("transformer", "resblocks"),
            ("text", "transformer", "resblocks"),
            ("text_model", "encoder", "layers"),
            ("text", "encoder", "layers"),
        ]

        for path in candidates:
            obj = self.text_model
            for name in path:
                obj = getattr(obj, name, None)
                if obj is None:
                    break
            if obj is not None:
                return obj

        return None

    def _set_requires_grad(self, modules: Iterable[nn.Module], value: bool) -> None:
        for module in modules:
            for p in module.parameters():
                p.requires_grad = value

    def _unfreeze_if_exists(self, path: tuple[str, ...]) -> None:
        obj = self.text_model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                return

        if isinstance(obj, nn.Module):
            for p in obj.parameters():
                p.requires_grad = True
        elif isinstance(obj, nn.Parameter):
            obj.requires_grad = True

    def set_text_finetune(self, enabled: bool) -> int:
        for p in self.text_model.parameters():
            p.requires_grad = False

        if not enabled or self.finetune_layers <= 0:
            self.text_model.eval()
            return 0

        blocks = self._text_transformer_blocks()
        if blocks is None:
            print("Available text model modules:")
            for name, _ in self.text_model.named_modules():
                print(name)
            raise RuntimeError(
                "Could not locate BiomedCLIP text transformer blocks for fine-tuning"
            )

        selected = list(blocks)[-self.finetune_layers:]
        self._set_requires_grad(selected, True)

        # OpenCLIP-style projection/norm
        self._unfreeze_if_exists(("ln_final",))
        self._unfreeze_if_exists(("text_projection",))

        # BiomedCLIP path from your printed model:
        # text.pooler and text.proj
        self._unfreeze_if_exists(("text", "pooler"))
        self._unfreeze_if_exists(("text", "proj"))

        self.text_model.train()

        return sum(
            p.numel()
            for p in self.text_model.parameters()
            if p.requires_grad
        )

    def trainable_text_parameters(self):
        return [p for p in self.text_model.parameters() if p.requires_grad]

    def forward(self) -> torch.Tensor:
        text_is_trainable = any(
            p.requires_grad for p in self.text_model.parameters()
        )

        if self.finetune_text_encoder and text_is_trainable:
            raw = self.text_model.encode_text(self.text_tokens).float()
            raw = F.normalize(raw, dim=-1)
            return F.normalize(self.projection(raw), dim=-1)

        return F.normalize(self.projection(self.raw_text_embeddings), dim=-1)