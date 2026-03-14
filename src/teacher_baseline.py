#!/usr/bin/env python3
"""
teacher_baseline.py

Florence-2-base teacher baseline for DIME-FM distillation prep.

This script:
  - loads a COCO caption + instance (bbox/class) subset
  - extracts image embeddings (mean-pooled projected visual tokens)
    for the Uni-Modal Distance Preserving Regularizer
  - computes VL score matrices using Florence-2's actual pipeline:
      * caption prompt    : <CAPTION>
      * bbox prompt       : "What is the object at [x, y, w, h]?"
      * class prompt      : "Can you find the <classes> in this image?"
  - builds within-batch B×B VL score matrices per prompt type
  - logs retrieval-style sanity metrics
  - saves embeddings, VL score matrices, metrics, and figures

The VL score is the conditional log-likelihood:
    log p(response | image, prompt)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import BaseModelOutput


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str,
                        default="/leonardo_work/IscrC_DEMOLLM/florence_distill/models/florence-2-base")
    parser.add_argument("--image_root", type=str, required=True,
                        help="COCO image directory")
    parser.add_argument("--captions_json", type=str, required=True,
                        help="COCO captions annotation JSON (captions_*.json)")
    parser.add_argument("--instances_json", type=str, default=None,
                        help="COCO instances annotation JSON (instances_*.json). "
                             "If provided, bbox/class prompts are added.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for image embedding extraction")
    parser.add_argument("--vl_batch_size", type=int, default=16,
                        help="Batch size for VL score matrix (B×B decoder passes)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_response_len", type=int, default=64,
                        help="Max token length for scoring any response")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# COCO data loading
# ---------------------------------------------------------------------------

def load_coco_categories(instances: Dict) -> Dict[int, str]:
    """Returns {category_id: category_name}."""
    return {c["id"]: c["name"] for c in instances.get("categories", [])}


def load_coco_instances(instances_json: Path) -> Dict[int, Dict]:
    """
    Returns { image_id: {"bboxes": [...], "classes": [...]} }.
    bbox format: [x, y, w, h] (COCO format, absolute pixels).
    """
    with instances_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cat_map = load_coco_categories(data)
    result: Dict[int, Dict] = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in result:
            result[iid] = {"bboxes": [], "classes": []}
        result[iid]["bboxes"].append(ann["bbox"])
        result[iid]["classes"].append(cat_map.get(ann["category_id"], "object"))
    return result


def build_bbox_prompts(bboxes: List[List[float]], classes: List[str]) -> List[str]:
    """
    Build structured prompts from COCO bbox/class annotations.
    Returns a list of prompt strings for one image.
    """
    prompts = []
    # Per-object bbox prompt
    for bbox, cls in zip(bboxes[:3], classes[:3]):  # limit to 3 to keep VL batches manageable
        x, y, w, h = bbox
        prompts.append(f"What is the object at [{x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}]?")
    # Whole-image class prompt
    unique_cls = list(dict.fromkeys(classes))[:5]
    prompts.append("Can you find the " + ", ".join(unique_cls) + " in this image?")
    return prompts


def load_coco_subset(
    captions_json: Path,
    image_root: Path,
    num_samples: int,
    seed: int,
    instances_data: Optional[Dict[int, Dict]] = None,
) -> List[Dict]:
    with captions_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    seen: set = set()
    pairs: List[Dict] = []
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

        entry: Dict = {
            "image_id": image_id,
            "file_name": file_name,
            "image_path": str(image_path),
            "caption": ann["caption"].strip(),
            "bbox_prompts": [],
        }

        if instances_data is not None and image_id in instances_data:
            inst = instances_data[image_id]
            entry["bbox_prompts"] = build_bbox_prompts(inst["bboxes"], inst["classes"])

        pairs.append(entry)
        seen.add(image_id)

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:num_samples]


# ---------------------------------------------------------------------------
# Image embeddings (for Uni-Modal Distance Preserving Regularizer)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_image_embeddings(
    model, processor,
    image_paths: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    images = [Image.open(p).convert("RGB") for p in image_paths]
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
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_images_with_prompt(
    model, processor,
    image_paths: List[str],
    task_prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[BaseModelOutput, torch.Tensor]:
    """Run Florence-2 encoder on a batch with the given task prompt.
    Returns (encoder_outputs, encoder_attention_mask)."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
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


def _shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
) -> torch.Tensor:
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted[shifted == -100] = pad_token_id
    return shifted


@torch.no_grad()
def score_responses_against_encoder(
    model, processor,
    encoder_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    responses: List[str],
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Score a list of responses against ONE image's encoder output.
    Returns [B] average log-prob per response.
    """
    B = len(responses)
    cap_tokens = processor.tokenizer(
        responses,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    labels = cap_tokens["input_ids"].to(device)

    pad_id = getattr(processor.tokenizer, "pad_token_id", None) or model.config.pad_token_id
    if pad_id is None:
        raise ValueError("No pad_token_id found.")
    decoder_start_id = model.config.text_config.decoder_start_token_id

    labels_masked = labels.clone()
    labels_masked[labels_masked == pad_id] = -100

    decoder_input_ids = _shift_tokens_right(labels, pad_id, decoder_start_id)
    decoder_attention_mask = (decoder_input_ids != pad_id).long().to(device)

    enc_out = BaseModelOutput(
        last_hidden_state=encoder_hidden.unsqueeze(0).expand(B, -1, -1)
    )
    attn = attention_mask.unsqueeze(0).expand(B, -1)

    outputs = model.language_model(
        encoder_outputs=enc_out,
        attention_mask=attn.to(enc_out.last_hidden_state.dtype),
        decoder_input_ids=decoder_input_ids.to(device),
        decoder_attention_mask=decoder_attention_mask,
        return_dict=True,
    )
    logits = outputs.logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (labels_masked != -100).float()
    return (token_lp * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)


@torch.no_grad()
def compute_vl_score_matrix(
    model, processor,
    image_paths: List[str],
    responses: List[str],
    task_prompt: str,
    max_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    B×B VL score matrix.
    score[i, j] = avg log p(responses[j] | image_i, task_prompt).
    Encoder is cached; decoder runs B times per image.
    """
    B = len(image_paths)
    assert len(responses) == B
    encoder_outputs, attention_mask = encode_images_with_prompt(
        model, processor, image_paths, task_prompt, device, dtype
    )
    matrix = torch.zeros(B, B)
    for i in range(B):
        matrix[i] = score_responses_against_encoder(
            model, processor,
            encoder_hidden=encoder_outputs.last_hidden_state[i],
            attention_mask=attention_mask[i],
            responses=responses,
            max_len=max_len,
            device=device,
        ).cpu()
    return matrix


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(sim: np.ndarray) -> Dict:
    n = sim.shape[0]
    assert sim.shape[0] == sim.shape[1]
    ranks_i2t, ranks_t2i = [], []
    for i in range(n):
        ranks_i2t.append(int(np.where(np.argsort(-sim[i]) == i)[0][0]) + 1)
    for j in range(n):
        ranks_t2i.append(int(np.where(np.argsort(-sim[:, j]) == j)[0][0]) + 1)

    def r_at_k(r, k):
        return float(np.mean(np.array(r) <= k))

    diag = np.diag(sim)
    off = sim[~np.eye(n, dtype=bool)]
    return {
        "matrix_size": int(n),
        "diag_mean": float(diag.mean()),
        "off_diag_mean": float(off.mean()),
        "gap": float(diag.mean() - off.mean()),
        "i2t_r1": r_at_k(ranks_i2t, 1),
        "i2t_r5": r_at_k(ranks_i2t, 5),
        "t2i_r1": r_at_k(ranks_t2i, 1),
        "t2i_r5": r_at_k(ranks_t2i, 5),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_matrix_figure(mat: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Response index")
    plt.ylabel("Image index")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_vl_hist(matched: np.ndarray, unmatched: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(matched, bins=30, alpha=0.7, label=f"matched (μ={matched.mean():.2f})")
    plt.hist(unmatched, bins=30, alpha=0.7, label=f"unmatched (μ={unmatched.mean():.2f})")
    plt.legend()
    plt.title(title)
    plt.xlabel("Avg log-prob")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Helper: run VL phase for one prompt type
# ---------------------------------------------------------------------------

def run_vl_phase(
    tag: str,
    pairs: List[Dict],
    response_key: str,
    task_prompt: str,
    model, processor,
    vl_bs: int,
    max_len: int,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Dict]:
    """
    Run VL scoring for `response_key` field in each pair.
    `response_key` = "caption" for caption prompts,
                   = one of the bbox_prompts for bbox/class prompts.
    Returns (matched_scores, unmatched_scores, all_matrices, retrieval_metrics_avg).
    """
    # Filter pairs that have a valid response for this key
    valid = [p for p in pairs if p.get(response_key) and str(p[response_key]).strip()]
    if not valid:
        print(f"  [{tag}] No valid samples — skipping.")
        return np.array([]), np.array([]), [], {}

    print(f"\n=== VL phase: {tag} ({len(valid)} samples, prompt='{task_prompt}') ===")
    all_matrices, matched, unmatched = [], [], []
    skipped = 0

    for start in tqdm(range(0, len(valid), vl_bs), desc=f"VL {tag}"):
        batch = valid[start : start + vl_bs]
        image_paths = [b["image_path"] for b in batch]
        responses = [str(b[response_key]) for b in batch]

        try:
            mat = compute_vl_score_matrix(
                model, processor, image_paths, responses, task_prompt, max_len, device, dtype
            )
        except Exception as e:
            skipped += len(batch)
            print(f"  Skipping {tag} batch at {start}: {e}")
            continue

        bsz = mat.shape[0]
        diag = mat.diag()
        off_diag = mat[~torch.eye(bsz, dtype=torch.bool)]
        matched.append(diag)
        unmatched.append(off_diag)
        all_matrices.append(mat.numpy())

    if not matched:
        print(f"  [{tag}] All batches failed ({skipped} skipped).")
        return np.array([]), np.array([]), [], {}

    matched_all = torch.cat(matched).numpy()
    unmatched_all = torch.cat(unmatched).numpy()

    per_batch = [compute_retrieval_metrics(m) for m in all_matrices]
    avg_metrics = {f"{tag}_{k}": float(np.mean([m[k] for m in per_batch]))
                   for k in per_batch[0]}

    print(f"  Matched:   mean={matched_all.mean():.4f}  std={matched_all.std():.4f}")
    print(f"  Unmatched: mean={unmatched_all.mean():.4f}  std={unmatched_all.std():.4f}")
    print(f"  Gap: {matched_all.mean() - unmatched_all.mean():.4f}")

    # Save matrices
    np.savez_compressed(
        output_dir / f"vl_score_matrices_{tag}.npz",
        **{f"batch_{i}": m for i, m in enumerate(all_matrices)},
    )
    if all_matrices:
        save_matrix_figure(
            all_matrices[0],
            output_dir / f"vl_score_matrix_{tag}_batch0.png",
            f"VL scores [{tag}] — batch 0",
        )
    save_vl_hist(
        matched_all, unmatched_all,
        output_dir / f"vl_score_hist_{tag}.png",
        f"VL scores [{tag}]",
    )

    return matched_all, unmatched_all, all_matrices, avg_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load instances if provided
    instances_data = None
    if args.instances_json:
        print(f"Loading instances from {args.instances_json} ...")
        instances_data = load_coco_instances(Path(args.instances_json))
        print(f"  Found instance data for {len(instances_data)} images.")

    pairs = load_coco_subset(
        Path(args.captions_json), Path(args.image_root),
        args.num_samples, args.seed, instances_data,
    )
    if not pairs:
        raise RuntimeError("No valid image-caption pairs found.")

    # How many have bbox prompts?
    n_bbox = sum(1 for p in pairs if p["bbox_prompts"])
    print(f"Loaded {len(pairs)} pairs ({n_bbox} with bbox/class prompts).")

    df_save = pd.DataFrame(pairs).copy()
    df_save["bbox_prompts"] = df_save["bbox_prompts"].apply(json.dumps)
    df_save.to_csv(output_dir / "pairs_preview.csv", index=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval()

    # ------------------------------------------------------------------ #
    # Phase 1: Image embeddings (uni-modal regularizer)
    # ------------------------------------------------------------------ #
    print("\n=== Phase 1: Image embedding extraction ===")
    all_emb, all_norms = [], []
    skipped_emb = 0
    for start in tqdm(range(0, len(pairs), args.batch_size), desc="Image embeddings"):
        batch = pairs[start : start + args.batch_size]
        try:
            emb, norms = extract_image_embeddings(
                model, processor,
                [b["image_path"] for b in batch],
                device, dtype,
            )
        except Exception as e:
            skipped_emb += len(batch)
            print(f"  Skipping at {start}: {e}")
            continue
        all_emb.append(emb.cpu())
        all_norms.append(norms.cpu())

    if not all_emb:
        raise RuntimeError(f"All image batches failed ({skipped_emb} skipped).")

    image_emb_all = torch.cat(all_emb).numpy()
    image_norms_all = torch.cat(all_norms).numpy()
    image_sim = image_emb_all @ image_emb_all.T
    print(f"  Image-image sim: diag={np.diag(image_sim).mean():.4f}  "
          f"off-diag={image_sim[~np.eye(len(image_sim), dtype=bool)].mean():.4f}")

    # ------------------------------------------------------------------ #
    # Phase 2: VL score matrices — caption prompt
    # ------------------------------------------------------------------ #
    cap_matched, cap_unmatched, cap_matrices, cap_metrics = run_vl_phase(
        tag="caption",
        pairs=pairs,
        response_key="caption",
        task_prompt="<CAPTION>",
        model=model, processor=processor,
        vl_bs=args.vl_batch_size,
        max_len=args.max_response_len,
        device=device, dtype=dtype,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------ #
    # Phase 3: VL score matrices — bbox/class prompts (thesis requirement)
    # ------------------------------------------------------------------ #
    # "bbox_prompts" is a list of prompt strings; use the first one per image
    # (the per-object bbox prompt) and also the class-level prompt (last entry).
    bbox_pairs_first = [
        {**p, "bbox_prompt_0": p["bbox_prompts"][0]}
        for p in pairs if p["bbox_prompts"]
    ]
    bbox_matched_0, bbox_unmatched_0, bbox_matrices_0, bbox_metrics_0 = run_vl_phase(
        tag="bbox_prompt_0",
        pairs=bbox_pairs_first,
        response_key="bbox_prompt_0",
        task_prompt="<OD>",
        model=model, processor=processor,
        vl_bs=args.vl_batch_size,
        max_len=args.max_response_len,
        device=device, dtype=dtype,
        output_dir=output_dir,
    )

    # Class-level prompt: last entry in bbox_prompts
    class_pairs = [
        {**p, "class_prompt": p["bbox_prompts"][-1]}
        for p in pairs if len(p["bbox_prompts"]) > 1
    ]
    cls_matched, cls_unmatched, cls_matrices, cls_metrics = run_vl_phase(
        tag="class_prompt",
        pairs=class_pairs,
        response_key="class_prompt",
        task_prompt="<CAPTION>",
        model=model, processor=processor,
        vl_bs=args.vl_batch_size,
        max_len=args.max_response_len,
        device=device, dtype=dtype,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------ #
    # Save all artifacts
    # ------------------------------------------------------------------ #
    print("\n=== Saving artifacts ===")

    np.savez_compressed(
        output_dir / "embeddings.npz",
        image_embeddings=image_emb_all,
        image_norms=image_norms_all,
        image_similarity=image_sim,
        matched_vl_scores=cap_matched if cap_matched.size else np.array([]),
        unmatched_vl_scores=cap_unmatched if cap_unmatched.size else np.array([]),
    )

    save_matrix_figure(
        image_sim,
        output_dir / "image_image_similarity.png",
        "Image-image cosine similarity (uni-modal regularizer)",
    )

    metrics: Dict = {
        "model_id": args.model_id,
        "num_samples": len(pairs),
        "num_bbox_samples": n_bbox,
        "vl_batch_size": args.vl_batch_size,
        "device": str(device),
        "dtype": str(dtype),
        "image_norm_mean": float(image_norms_all.mean()),
        "image_norm_std": float(image_norms_all.std()),
    }

    for tag, matched, unmatched in [
        ("caption", cap_matched, cap_unmatched),
        ("bbox_prompt_0", bbox_matched_0, bbox_unmatched_0),
        ("class_prompt", cls_matched, cls_unmatched),
    ]:
        if matched.size:
            metrics[f"{tag}_matched_mean"] = float(matched.mean())
            metrics[f"{tag}_unmatched_mean"] = float(unmatched.mean())
            metrics[f"{tag}_gap"] = float(matched.mean() - unmatched.mean())

    metrics.update(cap_metrics)
    metrics.update(bbox_metrics_0)
    metrics.update(cls_metrics)

    if torch.cuda.is_available() and device.type == "cuda":
        metrics["peak_gpu_memory_mb"] = float(
            torch.cuda.max_memory_allocated(device) / 1024 ** 2
        )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll artifacts saved to {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
