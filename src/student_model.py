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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, mobilenet_v2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _freeze_module_parameters(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _resolve_hidden_size(language_model) -> int:
    config = language_model.config
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "d_model"):
        return config.d_model
    return 768

@dataclass
class StudentConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 64
    vision_out_dim: int = 256
    text_enc_hidden: int = 768   # Florence-2 language encoder hidden size
    backbone: str = "resnet18"   # "resnet18" | "mobilenetv2" | "custom_tiny"
    reduced_vocab_size: Optional[int] = None  # None = use full vocab; int = reduced vocab
    vocab_mapping_path: Optional[str] = None  # path to vocab_mapping_*.json


# ---------------------------------------------------------------------------
# Vision encoder
# ---------------------------------------------------------------------------

class TinyVisionEncoder(nn.Module):
    """Vision encoder with configurable backbone.

    images [B, 3, H, W] → image_embed [B, D]
    LayerNorm after proj keeps activations bounded and avoids NaN after first step.

    Supported backbones:
      - "resnet18":    ResNet-18 (~11.2M params, 512-d features)
      - "mobilenetv2": MobileNetV2 (~3.4M params, 1280-d features)
      - "custom_tiny": Minimal 5-layer CNN (~0.3M params, 128-d features)
    """

    def __init__(self, out_dim: int = 256, backbone: str = "resnet18"):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            net = resnet18(weights=None)
            self.cnn = nn.Sequential(*list(net.children())[:-1])  # → [B, 512, 1, 1]
            feat_dim = 512
        elif backbone == "mobilenetv2":
            net = mobilenet_v2(weights=None)
            self.cnn = nn.Sequential(
                net.features,                          # → [B, 1280, 7, 7]
                nn.AdaptiveAvgPool2d((1, 1)),          # → [B, 1280, 1, 1]
            )
            feat_dim = 1280
        elif backbone == "custom_tiny":
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 112x112
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 56x56
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), # 28x28
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, stride=2, padding=1),# 14x14
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, stride=2, padding=1),# 7x7
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),                # [B, 128, 1, 1]
            )
            feat_dim = 128
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.proj = nn.Linear(feat_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.cnn(images).flatten(1)   # [B, feat_dim]
        x = self.proj(x)                   # [B, D]
        return self.ln(x)                  # [B, D] bounded


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
        _freeze_module_parameters(self)

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

    token_embedding may be the Florence-2 embedding matrix (frozen); if its
    embedding_dim != config.d_model, we project down to d_model.
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

        embed_dim = self.token_emb.embedding_dim
        if embed_dim != config.d_model:
            self.embed_proj = nn.Linear(embed_dim, config.d_model)
        else:
            self.embed_proj = None

        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.register_buffer(
            "_position_ids",
            torch.arange(config.max_seq_len).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "_causal_mask",
            torch.triu(
                torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

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
        dtype = self.vis_proj.weight.dtype   # match trainable params (float32 when student not half)

        tok = self.token_emb(input_ids)                               # [B, T, embed_dim]
        tok = tok.to(dtype)   # token_emb may be frozen from teacher (e.g. float16); decoder is float32
        if self.embed_proj is not None:
            tok = self.embed_proj(tok)                                # [B, T, d_model]
        pos_ids = self._position_ids[:, :T].expand(B, T)
        pos = self.pos_emb(pos_ids)
        tgt = tok + pos                                               # [B, T, d_model]

        mem = self.vis_proj(vis_feat.to(dtype)).unsqueeze(1)          # [B, 1, D]
        causal = self._causal_mask[:T, :T]

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
        self.encoder = TinyVisionEncoder(
            out_dim=self.config.vision_out_dim,
            backbone=self.config.backbone,
        )
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
    _freeze_module_parameters(text_emb)
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
    _freeze_module_parameters(text_emb)

    hidden_size = _resolve_hidden_size(language_model)

    cfg = config or StudentConfig(text_enc_hidden=hidden_size)

    text_encoder = FrozenTextEncoder(language_model, hidden_size=hidden_size)

    return TinyCLIPStudent(
        config=cfg,
        token_embedding=text_emb,
        text_encoder=text_encoder,
    )


def build_student_reduced_vocab(
    config: StudentConfig,
    vocab_mapping_path: str,
    florence_model=None,
) -> TinyCLIPStudent:
    """
    Build student with a reduced vocabulary.

    The token embedding and output projection are sized to the reduced vocab.
    If florence_model is provided, initialise the reduced embedding from the
    corresponding rows of the Florence-2 embedding (warm-start).

    Args:
        config: StudentConfig with reduced_vocab_size set.
        vocab_mapping_path: Path to vocab_mapping_*.json
            (keys = reduced_id str, values = florence_id int).
        florence_model: Optional Florence-2 model for warm-starting embeddings.
    """
    with open(vocab_mapping_path, "r") as f:
        mapping = json.load(f)  # {"0": florence_id, "1": florence_id, ...}

    reduced_size = len(mapping)
    cfg = StudentConfig(
        vocab_size=reduced_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.max_seq_len,
        vision_out_dim=config.vision_out_dim,
        text_enc_hidden=config.text_enc_hidden,
        backbone=config.backbone,
        reduced_vocab_size=reduced_size,
        vocab_mapping_path=vocab_mapping_path,
    )

    # Learnable embedding at reduced size (no frozen Florence-2 embedding)
    token_emb = nn.Embedding(reduced_size, cfg.d_model)

    if florence_model is not None:
        # Warm-start: copy rows from Florence-2 embedding
        teacher_emb = florence_model.get_input_embeddings()
        teacher_weight = teacher_emb.weight.data  # [full_vocab, embed_dim]
        proj_needed = teacher_weight.shape[1] != cfg.d_model

        with torch.no_grad():
            for reduced_str, florence_id in mapping.items():
                reduced_id = int(reduced_str)
                if florence_id < teacher_weight.shape[0]:
                    row = teacher_weight[florence_id].float()
                    if proj_needed:
                        # Truncate or average down to d_model
                        row = row[:cfg.d_model] if row.shape[0] >= cfg.d_model else \
                              F.pad(row, (0, cfg.d_model - row.shape[0]))
                    token_emb.weight[reduced_id] = row

    student = TinyCLIPStudent(config=cfg, token_embedding=token_emb)
    return student


# ---------------------------------------------------------------------------
# Deployment model extraction
# ---------------------------------------------------------------------------

def extract_deployment_model(
    student: TinyCLIPStudent,
) -> TinyCLIPStudent:
    """
    Extract a deployment-only student by stripping the frozen text encoder.
    The returned model has only the vision encoder + decoder (no text_encoder).

    This is the model that gets exported to ONNX/TFLite.
    """
    deploy = TinyCLIPStudent(
        config=student.config,
        token_embedding=student.decoder.token_emb,
        text_encoder=None,  # strip text encoder
    )
    # Copy vision encoder weights
    deploy.encoder.load_state_dict(student.encoder.state_dict())
    # Copy decoder weights (token_emb is already shared via reference)
    deploy.decoder.load_state_dict(student.decoder.state_dict())
    return deploy


def count_deployment_params(student: TinyCLIPStudent) -> dict:
    """Count parameters for the deployment path (no text encoder)."""
    enc_params = sum(p.numel() for p in student.encoder.parameters())
    dec_params = sum(p.numel() for p in student.decoder.parameters())
    total = enc_params + dec_params
    text_enc_params = sum(p.numel() for p in student.text_encoder.parameters()) \
        if student.text_encoder is not None else 0

    return {
        "vision_encoder": enc_params,
        "decoder": dec_params,
        "deployment_total": total,
        "text_encoder_stripped": text_enc_params,
        "full_model_total": total + text_enc_params,
        "deployment_size_fp32_mb": round(total * 4 / 1024**2, 2),
        "deployment_size_fp16_mb": round(total * 2 / 1024**2, 2),
        "deployment_size_int8_mb": round(total / 1024**2, 2),
    }
