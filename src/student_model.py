#!/usr/bin/env python3
"""
student_model.py

TinyCLIP-style student for Florence-2 distillation.

Architecture:
  - Vision encoder:  ResNet-18 trunk → linear projection (< 12 M params total)
  - Text encoder:    Florence-2 language encoder (frozen, shared with teacher)
  - Decoder:         2-layer causal transformer decoder conditioned on image token

The frozen text encoder follows the thesis requirement of keeping the Florence-2
text encoder fixed while the rest of the architecture is scaled down.

Forward modes
  • generation:  (images, input_ids) -> logits  [TFLite-export path]
  • distillation train:  also exposes .encode_image(), .encode_text()
    for loss computation in distill_train.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class StudentConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 64
    vision_out_dim: int = 256
    text_enc_hidden: int = 768   # Florence-2 language encoder hidden size


# ---------------------------------------------------------------------------
# Vision encoder
# ---------------------------------------------------------------------------

class TinyVisionEncoder(nn.Module):
    """ResNet-18 trunk with a linear projection head.

    images [B, 3, H, W] → image_embed [B, D]
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        backbone = resnet18(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # → [B, 512, 1, 1]
        self.proj = nn.Linear(512, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.cnn(images).flatten(1)   # [B, 512]
        return self.proj(x)               # [B, D]


# ---------------------------------------------------------------------------
# Frozen Florence-2 text encoder wrapper (Step 3)
# ---------------------------------------------------------------------------

class FrozenTextEncoder(nn.Module):
    """
    Wraps the Florence-2 language model's encoder half (encoder-side of
    the seq2seq language model) and keeps all weights frozen.

    Input:  token ids [B, T]
    Output: pooled text embedding [B, H]  (mean-pooled last hidden state)
    """

    def __init__(self, florence_language_model, hidden_size: int = 768):
        super().__init__()
        self.encoder = florence_language_model.model.encoder
        self.embed_tokens = florence_language_model.model.shared
        self.hidden_size = hidden_size

        # Freeze every parameter
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        token_embeds = self.embed_tokens(input_ids)
        enc_out = self.encoder(
            inputs_embeds=token_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = enc_out.last_hidden_state  # [B, T, H]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled  # [B, H]


# ---------------------------------------------------------------------------
# Mini decoder
# ---------------------------------------------------------------------------

class MiniDecoder(nn.Module):
    """
    2-layer causal transformer decoder conditioned on a single image vector.

    token_embedding may be the Florence-2 embedding matrix (frozen).
    """

    def __init__(
        self,
        config: StudentConfig,
        token_embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.config = config

        self.token_emb = token_embedding if token_embedding is not None else \
            nn.Embedding(config.vocab_size, config.d_model)

        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)

        self.vis_proj = nn.Linear(config.vision_out_dim, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, vis_feat: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        tok = self.token_emb(input_ids)                               # [B, T, D]
        pos = self.pos_emb(torch.arange(T, device=device).unsqueeze(0).expand(B, T))
        tgt = tok + pos                                               # [B, T, D]

        mem = self.vis_proj(vis_feat).unsqueeze(1)                    # [B, 1, D]
        causal = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        dec_out = self.decoder(tgt=tgt, memory=mem, tgt_mask=causal)  # [B, T, D]
        return self.out_proj(dec_out)                                 # [B, T, V]


# ---------------------------------------------------------------------------
# Full student
# ---------------------------------------------------------------------------

class TinyCLIPStudent(nn.Module):
    """
    Full student model:

        images               → TinyVisionEncoder → image_emb  [B, D]
        (image_emb, tokens)  → MiniDecoder       → logits     [B, T, V]

    Optionally holds a frozen FrozenTextEncoder for computing text embeddings
    during distillation training (not needed for TFLite export).
    """

    def __init__(
        self,
        config: Optional[StudentConfig] = None,
        token_embedding: Optional[nn.Embedding] = None,
        text_encoder: Optional[FrozenTextEncoder] = None,
    ):
        super().__init__()
        self.config = config or StudentConfig()
        self.encoder = TinyVisionEncoder(out_dim=self.config.vision_out_dim)
        self.decoder = MiniDecoder(self.config, token_embedding=token_embedding)
        # text_encoder is kept as a module only during training; not exported to TFLite
        if text_encoder is not None:
            self.text_encoder: Optional[FrozenTextEncoder] = text_encoder
        else:
            self.text_encoder = None

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] → [B, D] (normalised)."""
        emb = self.encoder(images)
        return F.normalize(emb, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, T] → [B, H] (normalised). Requires text_encoder to be set."""
        if self.text_encoder is None:
            raise RuntimeError("text_encoder not set; call build_student_with_florence_encoders.")
        emb = self.text_encoder(input_ids, attention_mask)
        return F.normalize(emb, dim=-1)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        images:    [B, 3, H, W]
        input_ids: [B, T]
        → logits:  [B, T, vocab_size]
        """
        vis_feat = self.encoder(images)
        return self.decoder(vis_feat, input_ids)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_student_with_florence_embeddings(
    florence_model,
    config: Optional[StudentConfig] = None,
) -> TinyCLIPStudent:
    """Build student reusing Florence-2's token embedding (frozen)."""
    text_emb = florence_model.get_input_embeddings()
    for p in text_emb.parameters():
        p.requires_grad = False
    return TinyCLIPStudent(config=config, token_embedding=text_emb)


def build_student_with_florence_encoders(
    florence_model,
    config: Optional[StudentConfig] = None,
) -> TinyCLIPStudent:
    """
    Build student with:
      - Frozen Florence-2 token embedding (shared vocabulary)
      - Frozen Florence-2 language encoder (for text embedding during training)

    This is the thesis-aligned construction: the text encoder is fixed and
    reused from Florence-2 while the vision encoder + decoder are trained.
    """
    language_model = florence_model.language_model

    # Frozen token embedding
    text_emb = florence_model.get_input_embeddings()
    for p in text_emb.parameters():
        p.requires_grad = False

    hidden_size = language_model.config.hidden_size if hasattr(language_model.config, "hidden_size") \
        else language_model.config.d_model if hasattr(language_model.config, "d_model") \
        else 768

    cfg = config or StudentConfig(text_enc_hidden=hidden_size)

    text_encoder = FrozenTextEncoder(language_model, hidden_size=hidden_size)

    return TinyCLIPStudent(
        config=cfg,
        token_embedding=text_emb,
        text_encoder=text_encoder,
    )
