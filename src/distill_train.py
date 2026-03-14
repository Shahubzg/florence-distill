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
from typing import Dict, List, Optional

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
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DistillDataset(Dataset):
    """
    Pairs COCO images/captions with pre-computed teacher image embeddings.

    pairs_csv columns expected:
        image_path, caption, image_id, file_name, bbox_prompts (JSON list)
    embeddings_npz keys:
        image_embeddings [N, D]   (L2-normalised, from teacher_baseline.py)
    """

    def __init__(self, pairs_csv: Path, embeddings_npz: Path):
        import pandas as pd
        self.df = pd.read_csv(pairs_csv)
        data = np.load(embeddings_npz)
        self.teacher_img_emb: np.ndarray = data["image_embeddings"]   # [N, D]

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

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        bbox_prompts = row["bbox_prompts"] if isinstance(row["bbox_prompts"], list) else []
        return {
            "image_path": str(row["image_path"]),
            "caption": str(row["caption"]),
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
    dtype = pixel_values.dtype

    # Encode prompt text for all images
    prompt_inputs = processor(
        text=[task_prompt] * B,
        return_tensors="pt",
        padding=True,
    )
    input_ids = prompt_inputs["input_ids"].to(device)

    # Merge image features + prompt tokens → encoder hidden states
    inputs_embeds = teacher.get_input_embeddings()(input_ids)
    image_features = teacher._encode_image(pixel_values.to(dtype))
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
    """
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
    KL( softmax(score_t/τ) || log_softmax(score_s/τ) ) averaged over rows.
    """
    target = F.softmax(score_t / temperature, dim=-1)        # [B, B]
    log_pred = F.log_softmax(score_s / temperature, dim=-1)  # [B, B]
    return F.kl_div(log_pred, target, reduction="batchmean")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model_id", type=str,
                   default="/leonardo_work/IscrC_DEMOLLM/florence_distill/models/florence-2-base")
    p.add_argument("--pairs_csv", type=str,
                   default="/leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/results_baseline/pairs_preview.csv")
    p.add_argument("--embeddings_npz", type=str,
                   default="/leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/results_baseline/embeddings.npz")
    p.add_argument("--output_dir", type=str,
                   default="/leonardo_work/IscrC_DEMOLLM/florence_distill/ckpts/distill_student")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_response_len", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature for VL score KL divergence.")
    p.add_argument("--lambda_im", type=float, default=1.0,
                   help="Weight for L_IM.")
    p.add_argument("--lambda_vl", type=float, default=1.0,
                   help="Weight for L_VL (caption).")
    p.add_argument("--lambda_vl_bbox", type=float, default=0.5,
                   help="Weight for L_VL (bbox/class prompt). 0 to disable.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load teacher model (frozen) -----
    print(f"Loading Florence-2 teacher from {args.teacher_model_id} ...")
    processor = AutoProcessor.from_pretrained(
        args.teacher_model_id, trust_remote_code=True
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_id, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # ----- Build student (frozen Florence-2 text encoder per thesis) -----
    print("Building student with frozen Florence-2 text encoder ...")
    vocab_size = processor.tokenizer.vocab_size or len(processor.tokenizer)
    cfg = StudentConfig(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=2,
        max_seq_len=args.max_response_len,
        vision_out_dim=256,
    )
    student = build_student_with_florence_encoders(teacher, config=cfg)
    student.to(device)

    total_params = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}  |  Trainable: {trainable:,}")

    # ----- Dataset & DataLoader -----
    dataset = DistillDataset(Path(args.pairs_csv), Path(args.embeddings_npz))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch.")

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    history: List[Dict] = []
    global_step = 0
    avg = 0.0

    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            image_paths: List[str] = batch["image_paths"]
            captions: List[str] = batch["captions"]
            bbox_prompts_batch: List[List[str]] = batch["bbox_prompts"]
            teacher_img_emb = batch["teacher_img_emb"].to(device)   # [B, D]

            # Load images once
            images = [Image.open(p).convert("RGB") for p in image_paths]
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
            for img in images:
                img.close()

            # ---- Student image embeddings (for L_IM) ----
            student_img_emb = F.normalize(student.encoder(pixel_values), dim=-1)  # [B, D]
            teacher_img_emb_n = F.normalize(teacher_img_emb, dim=-1)

            loss_im = unimodal_distance_loss(student_img_emb, teacher_img_emb_n)

            # ---- Student VL scores for captions (reuse vis_feats) ----
            vis_feats = student.encoder(pixel_values)   # [B, D]  (un-normalised, for decoder)

            score_s_cap = compute_student_vl_matrix(
                student, processor, pixel_values, vis_feats,
                captions, args.max_response_len, device,
            )

            # ---- Teacher VL scores for captions (ON-THE-FLY, per thesis) ----
            score_t_cap = compute_teacher_vl_matrix(
                teacher, processor, pixel_values, "<CAPTION>",
                captions, args.max_response_len, device,
            )

            loss_vl = vl_score_distillation_loss(
                score_s_cap, score_t_cap, temperature=args.temperature
            )

            # ---- Additional VL signal: bbox/class prompts (thesis requirement) ----
            loss_vl_bbox = torch.tensor(0.0, device=device)
            if args.lambda_vl_bbox > 0:
                # Use the class-level prompt (last entry) if all samples have bbox prompts
                class_prompts = [bp[-1] for bp in bbox_prompts_batch if bp]
                if len(class_prompts) == len(captions):
                    score_s_cls = compute_student_vl_matrix(
                        student, processor, pixel_values, vis_feats,
                        class_prompts, args.max_response_len, device,
                    )
                    score_t_cls = compute_teacher_vl_matrix(
                        teacher, processor, pixel_values, "<CAPTION>",
                        class_prompts, args.max_response_len, device,
                    )
                    loss_vl_bbox = vl_score_distillation_loss(
                        score_s_cls, score_t_cls, temperature=args.temperature
                    )

            # ---- Total loss ----
            loss = (args.lambda_im * loss_im
                    + args.lambda_vl * loss_vl
                    + args.lambda_vl_bbox * loss_vl_bbox)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
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

            history.append({
                "step": global_step,
                "epoch": epoch + 1,
                "loss": loss.item(),
                "loss_im": loss_im.item(),
                "loss_vl_cap": loss_vl.item(),
                "loss_vl_cls": loss_vl_bbox.item(),
            })

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1} avg loss: {avg:.4f}")

        ckpt = output_dir / f"student_epoch{epoch + 1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "student_state_dict": student.state_dict(),
            "config": cfg.__dict__,
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt)
        print(f"  Saved checkpoint: {ckpt}")

    with (output_dir / "train_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_im": args.lambda_im,
        "lambda_vl": args.lambda_vl,
        "lambda_vl_bbox": args.lambda_vl_bbox,
        "temperature": args.temperature,
        "teacher_model_id": args.teacher_model_id,
        "total_params": total_params,
        "trainable_params": trainable,
        "final_avg_loss": avg,
    }
    with (output_dir / "train_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\nTraining complete. Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    train()
