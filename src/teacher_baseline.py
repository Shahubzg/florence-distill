#!/usr/bin/env python3
"""
teacher_baseline.py

Florence-2-base teacher baseline for DIME-FM distillation prep.

This script:
  - loads a COCO caption subset
  - extracts image embeddings (mean-pooled projected visual tokens)
    for the Uni-Modal Distance Preserving Regularizer
  - computes VL scores using Florence-2's actual pipeline:
    image + <CAPTION> task prompt → decoder log-prob of caption
    for the VL Score Distillation Loss
  - builds within-batch VL score matrices
  - logs retrieval-style sanity metrics
  - saves figures and artifacts

Florence-2 is a seq2seq multimodal model, not a CLIP-style dual encoder.
The VL score is the conditional log-likelihood: log p(caption | image, <CAPTION>).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import BaseModelOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/leonardo_work/IscrC_DEMOLLM/florence_distill/models/florence-2-base")
    parser.add_argument("--image_root", type=str, required=True, help="COCO image directory")
    parser.add_argument("--captions_json", type=str, required=True, help="COCO captions annotation JSON")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for image embedding extraction")
    parser.add_argument("--vl_batch_size", type=int, default=16, help="Batch size for VL score matrix (B×B decoder passes)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_caption_len", type=int, default=128)
    parser.add_argument("--task_prompt", type=str, default="<CAPTION>", help="Florence-2 task token")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_coco_subset(captions_json: Path, image_root: Path, num_samples: int, seed: int) -> List[Dict]:
    with captions_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    seen = set()
    pairs = []
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        if image_id in seen:
            continue
        file_name = image_id_to_file.get(image_id)
        if file_name is None:
            continue
        image_path = image_root / file_name
        if not image_path.exists():
            continue
        pairs.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "image_path": str(image_path),
                "caption": ann["caption"].strip(),
            }
        )
        seen.add(image_id)

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:num_samples]


# ---------------------------------------------------------------------------
# Image embeddings (for Uni-Modal Distance Preserving Regularizer)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_image_embeddings(
    model, processor, image_paths: List[str], device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(img)

    proc = processor(images=images, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device=device, dtype=dtype)

    for img in images:
        img.close()

    image_tokens = model._encode_image(pixel_values)
    image_emb = image_tokens.mean(dim=1).float()
    image_norms = torch.linalg.vector_norm(image_emb, dim=-1)
    image_emb = F.normalize(image_emb, dim=-1)
    return image_emb, image_norms


# ---------------------------------------------------------------------------
# VL score computation (for VL Score Distillation Loss)
# Uses the actual Florence-2 pipeline: image + task prompt → decoder scores
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_images_with_prompt(
    model, processor, image_paths: List[str], task_prompt: str,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[BaseModelOutput, torch.Tensor]:
    """
    Run the Florence-2 encoder: image pixels + task prompt → encoder hidden states.
    Returns cached encoder_outputs and attention_mask for decoder reuse.
    """
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(img)

    inputs = processor(
        text=[task_prompt] * len(images),
        images=images,
        return_tensors="pt",
        padding=True,
    )

    for img in images:
        img.close()

    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)

    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_features = model._encode_image(pixel_values)
    merged_embeds, attention_mask = model._merge_input_ids_with_image_features(
        image_features, inputs_embeds
    )

    encoder_outputs = model.language_model.model.encoder(
        inputs_embeds=merged_embeds,
        attention_mask=attention_mask.to(merged_embeds.dtype),
        return_dict=True,
    )

    return encoder_outputs, attention_mask


def _shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    """Shift labels right to create decoder input: prepend decoder_start_token_id."""
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted[shifted == -100] = pad_token_id
    return shifted


@torch.no_grad()
def score_captions_against_encoder(
    model, processor, encoder_hidden: torch.Tensor, attention_mask: torch.Tensor,
    captions: List[str], max_caption_len: int, device: torch.device,
) -> torch.Tensor:
    """
    Given ONE image's encoder output, score B captions via decoder log-prob.
    Returns tensor of shape [B] with average log-prob per caption.
    """
    B = len(captions)

    cap_tokens = processor.tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_caption_len,
    )
    labels = cap_tokens["input_ids"].to(device)

    pad_id = processor.tokenizer.pad_token_id
    decoder_start_id = model.config.text_config.decoder_start_token_id

    labels_masked = labels.clone()
    labels_masked[labels_masked == pad_id] = -100

    decoder_input_ids = _shift_tokens_right(labels, pad_id, decoder_start_id)

    enc_hidden_expanded = encoder_hidden.unsqueeze(0).expand(B, -1, -1)
    attn_mask_expanded = attention_mask.unsqueeze(0).expand(B, -1)

    enc_out = BaseModelOutput(last_hidden_state=enc_hidden_expanded)

    outputs = model.language_model(
        encoder_outputs=enc_out,
        attention_mask=attn_mask_expanded.to(enc_hidden_expanded.dtype),
        decoder_input_ids=decoder_input_ids.to(device),
        return_dict=True,
    )

    logits = outputs.logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (labels_masked != -100).float()
    per_sample_score = (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

    return per_sample_score


@torch.no_grad()
def compute_vl_score_matrix(
    model, processor, image_paths: List[str], captions: List[str],
    task_prompt: str, max_caption_len: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute B×B VL score matrix.
    score[i, j] = avg log p(caption_j | image_i, task_prompt).
    Caches encoder outputs (one per image), runs decoder B times.
    """
    B = len(image_paths)
    assert len(captions) == B

    encoder_outputs, attention_mask = encode_images_with_prompt(
        model, processor, image_paths, task_prompt, device, dtype
    )

    score_matrix = torch.zeros(B, B)
    for i in range(B):
        scores = score_captions_against_encoder(
            model, processor,
            encoder_hidden=encoder_outputs.last_hidden_state[i],
            attention_mask=attention_mask[i],
            captions=captions,
            max_caption_len=max_caption_len,
            device=device,
        )
        score_matrix[i] = scores.cpu()

    return score_matrix


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(sim: np.ndarray) -> Dict[str, float]:
    n = sim.shape[0]
    assert sim.shape[0] == sim.shape[1]

    ranks_i2t = []
    ranks_t2i = []

    for i in range(n):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks_i2t.append(rank)

    for j in range(n):
        order = np.argsort(-sim[:, j])
        rank = int(np.where(order == j)[0][0]) + 1
        ranks_t2i.append(rank)

    def recall_at_k(ranks: List[int], k: int) -> float:
        return float(np.mean(np.array(ranks) <= k))

    diag = np.diag(sim)
    off_diag = sim[~np.eye(n, dtype=bool)]

    return {
        "matrix_size": int(n),
        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "off_diag_mean": float(off_diag.mean()),
        "off_diag_std": float(off_diag.std()),
        "diag_minus_off_diag": float(diag.mean() - off_diag.mean()),
        "i2t_r1": recall_at_k(ranks_i2t, 1),
        "i2t_r5": recall_at_k(ranks_i2t, 5),
        "t2i_r1": recall_at_k(ranks_t2i, 1),
        "t2i_r5": recall_at_k(ranks_t2i, 5),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_score_matrix_figure(sim: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(sim, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Caption index")
    plt.ylabel("Image index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_norm_hist(image_norms: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(image_norms, bins=20, alpha=0.7, label="image embedding norms")
    plt.legend()
    plt.title("Image embedding norms (before L2 normalization)")
    plt.xlabel("Norm")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_vl_score_hist(matched: np.ndarray, unmatched: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(matched, bins=30, alpha=0.7, label=f"matched (mean={matched.mean():.2f})")
    plt.hist(unmatched, bins=30, alpha=0.7, label=f"unmatched (mean={unmatched.mean():.2f})")
    plt.legend()
    plt.title("VL scores: log p(caption | image, <CAPTION>)")
    plt.xlabel("Avg log-prob")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_root = Path(args.image_root)
    captions_json = Path(args.captions_json)

    device = torch.device(args.device)
    use_fp16 = device.type == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32

    pairs = load_coco_subset(captions_json, image_root, args.num_samples, args.seed)
    if len(pairs) == 0:
        raise RuntimeError("No valid image-caption pairs found.")

    preview_df = pd.DataFrame(pairs)
    preview_df.to_csv(output_dir / "pairs_preview.csv", index=False)

    print(f"Loaded {len(pairs)} image-caption pairs.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval()

    # ---- Phase 1: Image embeddings (for uni-modal regularizer analysis) ----
    print("\n=== Phase 1: Image embedding extraction ===")
    all_image_emb = []
    all_image_norms = []

    skipped_emb = 0
    for start in tqdm(range(0, len(pairs), args.batch_size), desc="Image embeddings"):
        batch = pairs[start : start + args.batch_size]
        image_paths = [x["image_path"] for x in batch]

        try:
            image_emb, image_norms = extract_image_embeddings(
                model, processor, image_paths, device, dtype
            )
        except Exception as e:
            skipped_emb += len(batch)
            print(f"  Skipping batch at {start}: {e}")
            continue

        all_image_emb.append(image_emb.cpu())
        all_image_norms.append(image_norms.cpu())

    if len(all_image_emb) == 0:
        raise RuntimeError(f"All image batches failed ({skipped_emb} skipped).")
    if skipped_emb > 0:
        print(f"  Warning: {skipped_emb} samples skipped during image embedding extraction.")

    image_emb_all = torch.cat(all_image_emb, dim=0).numpy()
    image_norms_all = torch.cat(all_image_norms, dim=0).numpy()

    image_sim = image_emb_all @ image_emb_all.T
    print(f"  Image-image cosine sim: diag mean={np.diag(image_sim).mean():.4f}, "
          f"off-diag mean={image_sim[~np.eye(len(image_sim), dtype=bool)].mean():.4f}")

    # ---- Phase 2: VL score matrices (for VL Score Distillation Loss) ----
    print(f"\n=== Phase 2: VL score computation (task_prompt={args.task_prompt}) ===")

    all_vl_matrices = []
    all_matched_scores = []
    all_unmatched_scores = []

    vl_bs = args.vl_batch_size
    skipped_vl = 0
    for start in tqdm(range(0, len(pairs), vl_bs), desc="VL score batches"):
        batch = pairs[start : start + vl_bs]
        image_paths = [x["image_path"] for x in batch]
        captions = [x["caption"] for x in batch]

        try:
            score_matrix = compute_vl_score_matrix(
                model, processor, image_paths, captions,
                task_prompt=args.task_prompt,
                max_caption_len=args.max_caption_len,
                device=device, dtype=dtype,
            )
        except Exception as e:
            skipped_vl += len(batch)
            print(f"  Skipping VL batch at {start}: {e}")
            continue

        bsz = score_matrix.shape[0]
        diag = score_matrix.diag()
        off_mask = ~torch.eye(bsz, dtype=torch.bool)
        off_diag = score_matrix[off_mask]

        all_matched_scores.append(diag)
        all_unmatched_scores.append(off_diag)
        all_vl_matrices.append(score_matrix.numpy())

    if len(all_matched_scores) == 0:
        raise RuntimeError(f"All VL batches failed ({skipped_vl} skipped).")
    if skipped_vl > 0:
        print(f"  Warning: {skipped_vl} samples skipped during VL scoring.")

    matched_scores = torch.cat(all_matched_scores).numpy()
    unmatched_scores = torch.cat(all_unmatched_scores).numpy()

    print(f"  Matched VL scores:   mean={matched_scores.mean():.4f} std={matched_scores.std():.4f}")
    print(f"  Unmatched VL scores: mean={unmatched_scores.mean():.4f} std={unmatched_scores.std():.4f}")
    print(f"  Gap (matched - unmatched): {matched_scores.mean() - unmatched_scores.mean():.4f}")

    # Aggregate within-batch retrieval metrics
    vl_retrieval_metrics = {}
    if len(all_vl_matrices) > 0:
        per_batch_metrics = []
        for mat in all_vl_matrices:
            m = compute_retrieval_metrics(mat)
            per_batch_metrics.append(m)

        for key in per_batch_metrics[0]:
            vals = [m[key] for m in per_batch_metrics]
            vl_retrieval_metrics[f"vl_{key}"] = float(np.mean(vals))

    # ---- Save everything ----
    print("\n=== Saving artifacts ===")

    metrics = {
        "model_id": args.model_id,
        "num_samples": int(len(pairs)),
        "batch_size": int(args.batch_size),
        "vl_batch_size": int(vl_bs),
        "task_prompt": args.task_prompt,
        "device": str(device),
        "dtype": str(dtype),
        "image_norm_mean": float(image_norms_all.mean()),
        "image_norm_std": float(image_norms_all.std()),
        "matched_vl_score_mean": float(matched_scores.mean()),
        "matched_vl_score_std": float(matched_scores.std()),
        "unmatched_vl_score_mean": float(unmatched_scores.mean()),
        "unmatched_vl_score_std": float(unmatched_scores.std()),
        "vl_score_gap": float(matched_scores.mean() - unmatched_scores.mean()),
    }
    metrics.update(vl_retrieval_metrics)

    if torch.cuda.is_available() and device.type == "cuda":
        metrics["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / 1024**2)

    np.savez_compressed(
        output_dir / "embeddings.npz",
        image_embeddings=image_emb_all,
        image_norms=image_norms_all,
        image_similarity=image_sim,
        matched_vl_scores=matched_scores,
        unmatched_vl_scores=unmatched_scores,
    )

    if len(all_vl_matrices) > 0:
        np.savez_compressed(
            output_dir / "vl_score_matrices.npz",
            **{f"batch_{i}": mat for i, mat in enumerate(all_vl_matrices)},
        )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_score_matrix_figure(
        image_sim, output_dir / "image_image_similarity.png",
        "Image-image cosine similarity (for uni-modal regularizer)"
    )
    if len(all_vl_matrices) > 0:
        save_score_matrix_figure(
            all_vl_matrices[0], output_dir / "vl_score_matrix_batch0.png",
            f"VL scores: log p(caption | image, {args.task_prompt}) — batch 0"
        )
    save_norm_hist(image_norms_all, output_dir / "embedding_norm_hist.png")
    save_vl_score_hist(matched_scores, unmatched_scores, output_dir / "vl_score_hist.png")

    print(f"\nAll artifacts saved to {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
