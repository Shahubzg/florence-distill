#!/usr/bin/env python3
"""
distill_train.py

DIME-FM-style distillation loop — thesis-aligned implementation.

Losses (from DIME-FM paper, adapted for Florence-2 teacher):

  1. Uni-Modal Distance Preserving Regularizer  (L_IM)
     Encourages the student to preserve the teacher's intra-modal geometry:
         L_IM = || S_student - S_teacher ||_F^2
     where S[i,j] = cosine_sim(emb_i, emb_j) for image embeddings.
     Teacher image embeddings are taken from pre-computed embeddings.npz.

  2. VL Score Distillation Loss  (L_VL)
     Matches the full within-batch B×B teacher VL score matrix:
         L_VL = KL( softmax(T_teacher/τ) || log_softmax(T_student/τ) )  [row-wise]
     where T[i,j] = log p(response_j | image_i, prompt).

     Teacher VL matrices are computed ON-THE-FLY per training batch using the
     frozen Florence-2 model, for BOTH caption prompts AND (when all samples in
     the batch have them) class-level bbox prompts.  This follows the thesis
     requirement: "for each minibatch, compute the teacher outputs".

Thesis requirements explicitly met:
  - Text encoder fixed: FrozenTextEncoder wraps Florence-2 language encoder.
  - No Pseudo-VL Score Distillation Loss.
  - COCO captions + bbox/class structured prompts both used to drive the teacher.
  - Teacher is kept frozen throughout.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import BaseModelOutput

from student_model import (
    TinyCLIPStudent,
    StudentConfig,
    build_student_with_florence_encoders,
    build_student_reduced_vocab,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DistillDataset(Dataset):
    """
    Pairs COCO images/captions with pre-computed teacher image embeddings.

    pairs_csv columns expected:
        image_path, caption, image_id, file_name, bbox_prompts (JSON list)
        Optional: all_captions (JSON list of all captions for multi-caption mode)
    embeddings_npz keys:
        image_embeddings [N, D]   (L2-normalised, from teacher_baseline.py)
    """

    def __init__(self, pairs_csv: Path, embeddings_npz: Path, multi_caption: bool = False):
        import pandas as pd
        self.df = pd.read_csv(pairs_csv)
        data = np.load(embeddings_npz)
        self.teacher_img_emb: np.ndarray = data["image_embeddings"]   # [N, D]
        self.multi_caption = multi_caption

        assert len(self.df) == self.teacher_img_emb.shape[0], (
            "Mismatch between pairs_csv rows and embeddings.npz size."
        )

        # Deserialise bbox_prompts (JSON list stored as string) if present
        if "bbox_prompts" in self.df.columns:
            self.df["bbox_prompts"] = self.df["bbox_prompts"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else []
            )
        else:
            self.df["bbox_prompts"] = [[] for _ in range(len(self.df))]

        # Deserialise all_captions if present and multi-caption mode enabled
        if multi_caption and "all_captions" in self.df.columns:
            self.df["all_captions"] = self.df["all_captions"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else []
            )
        else:
            self.df["all_captions"] = [[] for _ in range(len(self.df))]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        bbox_prompts = row["bbox_prompts"] if isinstance(row["bbox_prompts"], list) else []

        # Multi-caption: randomly pick one of the available captions
        caption = str(row["caption"])
        if self.multi_caption:
            all_caps = row["all_captions"]
            if isinstance(all_caps, list) and len(all_caps) > 1:
                caption = random.choice(all_caps)

        return {
            "image_path": str(row["image_path"]),
            "caption": caption,
            "bbox_prompts": bbox_prompts,
            "teacher_img_emb": torch.from_numpy(self.teacher_img_emb[idx]).float(),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "image_paths": [b["image_path"] for b in batch],
        "captions": [b["caption"] for b in batch],
        "bbox_prompts": [b["bbox_prompts"] for b in batch],
        "teacher_img_emb": torch.stack([b["teacher_img_emb"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Teacher on-the-fly VL scoring
# ---------------------------------------------------------------------------

def _shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
) -> torch.Tensor:
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted[shifted == -100] = pad_token_id
    return shifted


@torch.no_grad()
def compute_teacher_vl_matrix(
    teacher,
    processor,
    pixel_values: torch.Tensor,   # [B, 3, H, W]  already on device
    task_prompt: str,
    responses: List[str],
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the teacher's B×B VL score matrix for the current training batch.

    score[i, j] = avg log p_teacher(responses[j] | image_i, task_prompt)

    This is the correct on-the-fly teacher soft-target computation required by
    the thesis ("for each minibatch, compute the teacher outputs").
    """
    B = pixel_values.shape[0]
    teacher_dtype = next(teacher.parameters()).dtype
    pixel_values = pixel_values.to(device=device, dtype=teacher_dtype)

    # Tokenize task prompt (Florence2Processor requires images when called, so use tokenizer only)
    prompt_inputs = processor.tokenizer(
        [task_prompt] * B,
        return_tensors="pt",
        padding=True,
    )
    input_ids = prompt_inputs["input_ids"].to(device)

    # Merge image features + prompt tokens → encoder hidden states
    inputs_embeds = teacher.get_input_embeddings()(input_ids)
    image_features = teacher._encode_image(pixel_values)
    merged_embeds, attn_mask = teacher._merge_input_ids_with_image_features(
        image_features, inputs_embeds
    )
    encoder_outputs = teacher.language_model.model.encoder(
        inputs_embeds=merged_embeds,
        attention_mask=attn_mask.to(merged_embeds.dtype),
        return_dict=True,
    )

    # Tokenise all responses
    resp_tokens = processor.tokenizer(
        responses, return_tensors="pt", padding=True,
        truncation=True, max_length=max_len,
    )
    labels = resp_tokens["input_ids"].to(device)

    pad_id = getattr(processor.tokenizer, "pad_token_id", None) or teacher.config.pad_token_id
    if pad_id is None:
        pad_id = 0
    decoder_start_id = teacher.config.text_config.decoder_start_token_id

    labels_masked = labels.clone()
    labels_masked[labels_masked == pad_id] = -100

    decoder_input_ids = _shift_tokens_right(labels, pad_id, decoder_start_id)
    decoder_attn_mask = (decoder_input_ids != pad_id).long().to(device)

    score_mat = torch.zeros(B, B, device=device)
    for i in range(B):
        enc_out = BaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state[i].unsqueeze(0).expand(B, -1, -1)
        )
        enc_attn = attn_mask[i].unsqueeze(0).expand(B, -1)

        out = teacher.language_model(
            encoder_outputs=enc_out,
            attention_mask=enc_attn.to(enc_out.last_hidden_state.dtype),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attn_mask,
            return_dict=True,
        )
        logits = out.logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        mask = (labels_masked != -100).float()
        score_mat[i] = (token_lp * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)

    return score_mat   # [B, B]


# ---------------------------------------------------------------------------
# Student on-the-fly VL scoring
# ---------------------------------------------------------------------------

def compute_student_vl_matrix(
    student: TinyCLIPStudent,
    processor,
    pixel_values: torch.Tensor,   # [B, 3, H, W]
    vis_feats: torch.Tensor,      # [B, D]  pre-computed by encoder
    responses: List[str],
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the student's B×B VL score matrix.
    score[i, j] = avg log p_student(responses[j] | image_i).

    vis_feats are passed in to avoid re-running the encoder.
    """
    B = len(responses)

    tok = processor.tokenizer(
        responses, return_tensors="pt", padding=True,
        truncation=True, max_length=max_len,
    )
    labels = tok["input_ids"].to(device)
    pad_id = processor.tokenizer.pad_token_id or 0

    labels_masked = labels.clone()
    labels_masked[labels_masked == pad_id] = -100

    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0] = pad_id

    score_mat = torch.zeros(B, B, device=device)
    for i in range(B):
        vis_i = vis_feats[i].unsqueeze(0).expand(B, -1)  # [B, D]
        logits = student.decoder(vis_i, shifted)          # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        mask = (labels_masked != -100).float()
        score_mat[i] = (token_lp * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)

    return score_mat   # [B, B]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def unimodal_distance_loss(
    student_emb: torch.Tensor,   # [B, D]  L2-normalised
    teacher_emb: torch.Tensor,   # [B, D]  L2-normalised
) -> torch.Tensor:
    """
    L_IM: Uni-Modal Distance Preserving Regularizer.
    Match the within-batch image-image similarity matrices (Frobenius MSE).
    Computed in float32 to avoid fp16 NaN.
    """
    student_emb = student_emb.float()
    teacher_emb = teacher_emb.float()
    sim_s = student_emb @ student_emb.T   # [B, B]
    sim_t = teacher_emb @ teacher_emb.T   # [B, B]
    return F.mse_loss(sim_s, sim_t)


def vl_score_distillation_loss(
    score_s: torch.Tensor,   # [B, B] student log-prob scores
    score_t: torch.Tensor,   # [B, B] teacher log-prob scores
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    L_VL: VL Score Distillation Loss.
    Row-wise KL divergence between student and teacher score distributions.
    Computed in float32; scores clamped to avoid -inf and fp16 NaN.
    """
    score_s = score_s.float().clamp(min=-50.0, max=0.0)   # log-probs
    score_t = score_t.float().clamp(min=-50.0, max=0.0)
    target = F.softmax(score_t / temperature, dim=-1)        # [B, B]
    log_pred = F.log_softmax(score_s / temperature, dim=-1)  # [B, B]
    return F.kl_div(log_pred, target, reduction="batchmean")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--teacher_model_id", type=str,
                   default="/leonardo_work/IscrC_DEMOLLM/florence_distill/models/florence-2-base")
    # Data
    p.add_argument("--pairs_csv", type=str,
                   default="/leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/results_baseline/pairs_preview.csv")
    p.add_argument("--embeddings_npz", type=str,
                   default="/leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/results_baseline/embeddings.npz")
    p.add_argument("--val_pairs_csv", type=str, default=None,
                   help="Validation pairs CSV. If set, run validation each epoch.")
    p.add_argument("--val_embeddings_npz", type=str, default=None,
                   help="Validation embeddings NPZ.")
    p.add_argument("--multi_caption", action="store_true",
                   help="Randomly sample from all_captions per image each epoch.")
    # Output
    p.add_argument("--output_dir", type=str,
                   default="/leonardo_work/IscrC_DEMOLLM/florence_distill/ckpts/distill_student")
    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="Linear warmup steps before cosine annealing.")
    p.add_argument("--max_response_len", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature for VL score KL divergence.")
    p.add_argument("--lambda_im", type=float, default=1.0,
                   help="Weight for L_IM.")
    p.add_argument("--lambda_vl", type=float, default=1.0,
                   help="Weight for L_VL (caption).")
    p.add_argument("--lambda_vl_bbox", type=float, default=0.5,
                   help="Weight for L_VL (bbox/class prompt). 0 to disable.")
    # Architecture
    p.add_argument("--backbone", type=str, default="resnet18",
                   choices=["resnet18", "mobilenetv2", "custom_tiny"],
                   help="Vision encoder backbone.")
    p.add_argument("--vocab_mapping", type=str, default=None,
                   help="Path to vocab_mapping_*.json for reduced vocab mode.")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    # Efficiency
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (increase for large datasets).")
    p.add_argument("--use_amp", action="store_true",
                   help="Use automatic mixed precision for student forward.")
    # Checkpointing
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to checkpoint to resume training from.")
    p.add_argument("--save_every", type=int, default=1,
                   help="Save checkpoint every N epochs.")
    p.add_argument("--keep_last_n", type=int, default=3,
                   help="Keep only the last N checkpoints.")
    # Logging
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--tensorboard", action="store_true",
                   help="Enable TensorBoard logging.")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--debug_nan", type=int, default=3,
                   help="When loss is non-finite, print diagnostics for first N occurrences (0=disable).")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_pixel_values(
    processor,
    image_paths: List[str],
    device: torch.device,
) -> torch.Tensor:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        inputs = processor(images=images, return_tensors="pt")
    finally:
        for image in images:
            image.close()
    return inputs["pixel_values"].to(device=device, dtype=torch.float32)


def _save_student_checkpoint(
    ckpt_path: Path,
    epoch: int,
    student: TinyCLIPStudent,
    cfg: StudentConfig,
    optimizer: torch.optim.Optimizer,
    global_step: int = 0,
) -> None:
    # Save student weights in fp16 to reduce checkpoint size (~half).
    state_dict = student.state_dict()
    state_dict_fp16 = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "student_state_dict": state_dict_fp16,
        "config": cfg.__dict__,
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)


def _log_nan_diagnostic(
    loss_im: torch.Tensor,
    loss_vl: torch.Tensor,
    loss_vl_bbox: torch.Tensor,
    student_img_emb: torch.Tensor,
    teacher_img_emb_n: torch.Tensor,
    score_s_cap: torch.Tensor,
    score_t_cap: torch.Tensor,
    batch_idx: int,
    epoch: int,
) -> None:
    """Print where non-finite values first appear (NaN vs Inf, which term/tensor)."""
    lines = [
        "",
        "[NaN/Inf diagnostic]",
        f"  epoch={epoch} batch_idx={batch_idx}",
        "  --- Loss terms ---",
        f"    loss_im:     value={loss_im.item():.6g}  is_nan={torch.isnan(loss_im).item()}  is_inf={torch.isinf(loss_im).item()}",
        f"    loss_vl:     value={loss_vl.item():.6g}  is_nan={torch.isnan(loss_vl).item()}  is_inf={torch.isinf(loss_vl).item()}",
        f"    loss_vl_bbox: value={loss_vl_bbox.item():.6g}  is_nan={torch.isnan(loss_vl_bbox).item()}  is_inf={torch.isinf(loss_vl_bbox).item()}",
        "  --- L_IM inputs (embeddings) ---",
    ]
    for name, emb in [("student_img_emb", student_img_emb), ("teacher_img_emb_n", teacher_img_emb_n)]:
        emb = emb.detach().float()
        n_el = emb.numel()
        n_finite = torch.isfinite(emb).sum().item()
        n_nan = torch.isnan(emb).sum().item()
        n_inf = torch.isinf(emb).sum().item()
        norms = torch.linalg.norm(emb, dim=-1)
        zero_norms = (norms < 1e-8).sum().item()
        lines.append(
            f"    {name}: shape={tuple(emb.shape)}  finite={n_finite}/{n_el}  nan={n_nan}  inf={n_inf}  "
            f"zero_norms={zero_norms}/{emb.shape[0]}  norm_min={norms.min().item():.6g}  norm_max={norms.max().item():.6g}"
        )
    lines.append("  --- VL score matrices (caption) ---")
    for name, sc in [("score_s_cap", score_s_cap), ("score_t_cap", score_t_cap)]:
        sc = sc.detach().float()
        n_el = sc.numel()
        n_finite = torch.isfinite(sc).sum().item()
        n_nan = torch.isnan(sc).sum().item()
        n_inf = torch.isinf(sc).sum().item()
        finite_vals = sc[torch.isfinite(sc)]
        min_s = finite_vals.min().item() if finite_vals.numel() > 0 else float("nan")
        max_s = finite_vals.max().item() if finite_vals.numel() > 0 else float("nan")
        lines.append(
            f"    {name}: shape={tuple(sc.shape)}  finite={n_finite}/{n_el}  nan={n_nan}  inf={n_inf}  "
            f"min={min_s:.4g}  max={max_s:.4g}"
        )
    lines.append("")
    tqdm.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _get_lr_scheduler(optimizer, args, steps_per_epoch):
    """Build LR scheduler with optional warmup."""
    total_steps = args.epochs * steps_per_epoch

    if args.warmup_steps > 0:
        from torch.optim.lr_scheduler import LambdaLR
        import math

        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / max(1, args.warmup_steps)
            progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


def _cleanup_old_checkpoints(output_dir: Path, keep_last_n: int) -> None:
    """Remove old checkpoints, keeping only the last N."""
    ckpts = sorted(output_dir.glob("student_epoch*.pt"),
                   key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep_last_n]:
        old.unlink()
        print(f"  Removed old checkpoint: {old.name}")


@torch.no_grad()
def _run_validation(
    student: TinyCLIPStudent,
    teacher,
    processor,
    val_loader: DataLoader,
    args,
    device: torch.device,
    teacher_dtype: torch.dtype,
) -> Dict:
    """Run validation and return average losses."""
    student.eval()
    total_im, total_vl, total_loss, n_batches = 0.0, 0.0, 0.0, 0

    for batch in val_loader:
        image_paths = batch["image_paths"]
        captions = batch["captions"]
        teacher_img_emb = batch["teacher_img_emb"].to(device=device)

        pixel_values = _load_pixel_values(processor, image_paths, device)
        teacher_pixel_values = pixel_values.to(dtype=teacher_dtype)

        raw_emb = student.encoder(pixel_values)
        student_img_emb = F.normalize(raw_emb.float(), dim=-1, eps=1e-6)
        teacher_img_emb_n = F.normalize(teacher_img_emb.float(), dim=-1, eps=1e-6)

        loss_im = unimodal_distance_loss(student_img_emb, teacher_img_emb_n)

        vis_feats = raw_emb.clone()
        score_s = compute_student_vl_matrix(
            student, processor, pixel_values, vis_feats,
            captions, args.max_response_len, device,
        )
        score_t = compute_teacher_vl_matrix(
            teacher, processor, teacher_pixel_values, "<CAPTION>",
            captions, args.max_response_len, device,
        )
        loss_vl = vl_score_distillation_loss(score_s, score_t, temperature=args.temperature)

        loss = args.lambda_im * loss_im + args.lambda_vl * loss_vl
        if torch.isfinite(loss):
            total_im += loss_im.item()
            total_vl += loss_vl.item()
            total_loss += loss.item()
            n_batches += 1

    student.train()
    if n_batches == 0:
        return {"val_loss": float("nan"), "val_loss_im": float("nan"), "val_loss_vl": float("nan")}
    return {
        "val_loss": total_loss / n_batches,
        "val_loss_im": total_im / n_batches,
        "val_loss_vl": total_vl / n_batches,
    }


def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    teacher_dtype = torch.float16 if device.type == "cuda" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- TensorBoard -----
    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))
            print(f"TensorBoard logging to {output_dir / 'tb_logs'}")
        except ImportError:
            print("Warning: tensorboard not installed, skipping TB logging.")

    # ----- Load teacher model (frozen) -----
    print(f"Loading Florence-2 teacher from {args.teacher_model_id} ...")
    processor = AutoProcessor.from_pretrained(
        args.teacher_model_id, trust_remote_code=True
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_id, torch_dtype=teacher_dtype, trust_remote_code=True,
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # ----- Build student -----
    vocab_size = processor.tokenizer.vocab_size or len(processor.tokenizer)
    cfg = StudentConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_response_len,
        vision_out_dim=args.d_model,
        backbone=args.backbone,
    )

    if args.vocab_mapping:
        print(f"Building student with reduced vocabulary from {args.vocab_mapping} ...")
        student = build_student_reduced_vocab(cfg, args.vocab_mapping, florence_model=teacher)
    else:
        print("Building student with frozen Florence-2 text encoder ...")
        student = build_student_with_florence_encoders(teacher, config=cfg)

    student.to(device)

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in trainable_params)
    print(f"  Total params: {total_params:,}  |  Trainable: {trainable:,}")
    print(f"  Backbone: {args.backbone}  |  Reduced vocab: {args.vocab_mapping is not None}")

    # ----- Dataset & DataLoader -----
    dataset = DistillDataset(
        Path(args.pairs_csv), Path(args.embeddings_npz),
        multi_caption=args.multi_caption,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch.")

    # ----- Validation DataLoader -----
    val_loader = None
    if args.val_pairs_csv and args.val_embeddings_npz:
        val_dataset = DistillDataset(Path(args.val_pairs_csv), Path(args.val_embeddings_npz))
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn,
        )
        print(f"Validation: {len(val_dataset)} samples, {len(val_loader)} batches.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _get_lr_scheduler(optimizer, args, len(loader))

    # ----- AMP scaler -----
    scaler = None
    if args.use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        print("Using automatic mixed precision (AMP).")

    # ----- Resume from checkpoint -----
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        print(f"Resuming from {args.resume_from} ...")
        ckpt_data = torch.load(args.resume_from, map_location=device)
        student.load_state_dict(ckpt_data["student_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt_data:
            optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        start_epoch = ckpt_data.get("epoch", 0)
        global_step = ckpt_data.get("global_step", start_epoch * len(loader))
        # Advance scheduler to correct step
        for _ in range(global_step):
            scheduler.step()
        print(f"  Resumed at epoch {start_epoch}, global_step {global_step}")

    history: List[Dict] = []
    avg = 0.0
    nan_diagnostic_count = 0

    for epoch in range(start_epoch, args.epochs):
        student.train()
        epoch_loss = 0.0
        n_skipped = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            image_paths: List[str] = batch["image_paths"]
            captions: List[str] = batch["captions"]
            bbox_prompts_batch: List[List[str]] = batch["bbox_prompts"]
            teacher_img_emb = batch["teacher_img_emb"].to(device=device)   # [B, D]

            # Load images once
            pixel_values = _load_pixel_values(processor, image_paths, device)
            teacher_pixel_values = pixel_values.to(dtype=teacher_dtype)

            # ---- Student image embeddings (for L_IM) ----
            amp_ctx = torch.cuda.amp.autocast() if scaler else torch.no_grad.__class__()
            if scaler:
                amp_ctx = torch.cuda.amp.autocast()

            raw_emb = student.encoder(pixel_values)   # [B, D]
            if not torch.isfinite(raw_emb).all():
                raw_emb = torch.where(torch.isfinite(raw_emb), raw_emb, torch.zeros_like(raw_emb))
            raw_emb_f32 = raw_emb.float()
            student_img_emb = F.normalize(raw_emb_f32, dim=-1, eps=1e-6)
            teacher_img_emb_n = F.normalize(teacher_img_emb.float(), dim=-1, eps=1e-6)

            loss_im = unimodal_distance_loss(student_img_emb, teacher_img_emb_n)

            # ---- Student VL scores for captions (reuse vis_feats) ----
            vis_feats = raw_emb.clone()

            score_s_cap = compute_student_vl_matrix(
                student, processor, pixel_values, vis_feats,
                captions, args.max_response_len, device,
            )

            # ---- Teacher VL scores for captions (ON-THE-FLY, per thesis) ----
            score_t_cap = compute_teacher_vl_matrix(
                teacher, processor, teacher_pixel_values, "<CAPTION>",
                captions, args.max_response_len, device,
            )

            loss_vl = vl_score_distillation_loss(
                score_s_cap, score_t_cap, temperature=args.temperature
            )

            # ---- Additional VL signal: bbox/class prompts (thesis requirement) ----
            loss_vl_bbox = torch.tensor(0.0, device=device)
            if args.lambda_vl_bbox > 0:
                class_prompts = [bp[-1] for bp in bbox_prompts_batch if bp]
                if len(class_prompts) == len(captions):
                    score_s_cls = compute_student_vl_matrix(
                        student, processor, pixel_values, vis_feats,
                        class_prompts, args.max_response_len, device,
                    )
                    score_t_cls = compute_teacher_vl_matrix(
                        teacher, processor, teacher_pixel_values, "<CAPTION>",
                        class_prompts, args.max_response_len, device,
                    )
                    loss_vl_bbox = vl_score_distillation_loss(
                        score_s_cls, score_t_cls, temperature=args.temperature
                    )

            # ---- Total loss ----
            loss = (args.lambda_im * loss_im
                    + args.lambda_vl * loss_vl
                    + args.lambda_vl_bbox * loss_vl_bbox)

            if not torch.isfinite(loss).item():
                n_skipped += 1
                if args.debug_nan > 0 and nan_diagnostic_count < args.debug_nan:
                    nan_diagnostic_count += 1
                    _log_nan_diagnostic(
                        loss_im=loss_im, loss_vl=loss_vl, loss_vl_bbox=loss_vl_bbox,
                        student_img_emb=student_img_emb, teacher_img_emb_n=teacher_img_emb_n,
                        score_s_cap=score_s_cap, score_t_cap=score_t_cap,
                        batch_idx=global_step, epoch=epoch + 1,
                    )
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_every == 0:
                    pbar.set_postfix(
                        loss="nan", L_IM="nan", L_VL="nan", L_VL_cls="nan",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                continue

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    L_IM=f"{loss_im.item():.4f}",
                    L_VL=f"{loss_vl.item():.4f}",
                    L_VL_cls=f"{loss_vl_bbox.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            # TensorBoard step logging
            if tb_writer and global_step % args.log_every == 0:
                tb_writer.add_scalar("train/loss", loss.item(), global_step)
                tb_writer.add_scalar("train/loss_im", loss_im.item(), global_step)
                tb_writer.add_scalar("train/loss_vl_cap", loss_vl.item(), global_step)
                tb_writer.add_scalar("train/loss_vl_cls", loss_vl_bbox.item(), global_step)
                tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            history.append({
                "step": global_step,
                "epoch": epoch + 1,
                "loss": loss.item(),
                "loss_im": loss_im.item(),
                "loss_vl_cap": loss_vl.item(),
                "loss_vl_cls": loss_vl_bbox.item(),
            })

        n_stepped = len(loader) - n_skipped
        avg = epoch_loss / n_stepped if n_stepped > 0 else float("nan")
        print(f"Epoch {epoch + 1} avg loss: {avg:.4f}  (stepped {n_stepped}/{len(loader)}, skipped {n_skipped})")

        # TensorBoard epoch logging
        if tb_writer:
            tb_writer.add_scalar("epoch/avg_loss", avg, epoch + 1)

        # ----- Validation -----
        if val_loader is not None:
            val_metrics = _run_validation(student, teacher, processor, val_loader, args, device, teacher_dtype)
            print(f"  Val: loss={val_metrics['val_loss']:.4f}  "
                  f"L_IM={val_metrics['val_loss_im']:.4f}  L_VL={val_metrics['val_loss_vl']:.4f}")
            if tb_writer:
                for k, v in val_metrics.items():
                    tb_writer.add_scalar(f"val/{k}", v, epoch + 1)

        # ----- Checkpoint -----
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt = output_dir / f"student_epoch{epoch + 1}.pt"
            _save_student_checkpoint(ckpt, epoch + 1, student, cfg, optimizer, global_step)
            print(f"  Saved checkpoint: {ckpt}")
            _cleanup_old_checkpoints(output_dir, args.keep_last_n)

    if tb_writer:
        tb_writer.close()

    with (output_dir / "train_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "epochs": args.epochs,
        "start_epoch": start_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lambda_im": args.lambda_im,
        "lambda_vl": args.lambda_vl,
        "lambda_vl_bbox": args.lambda_vl_bbox,
        "temperature": args.temperature,
        "backbone": args.backbone,
        "reduced_vocab": args.vocab_mapping is not None,
        "teacher_model_id": args.teacher_model_id,
        "total_params": total_params,
        "trainable_params": trainable,
        "final_avg_loss": avg,
        "global_step": global_step,
    }
    with (output_dir / "train_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\nTraining complete. Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    train()
