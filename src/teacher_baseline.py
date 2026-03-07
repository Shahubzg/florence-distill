#!/usr/bin/env python3
"""
teacher_baseline.py

Clean Florence-2-base teacher baseline
This script:
  - loads a small COCO caption subset
  - extracts proxy image embeddings from Florence-2 projected visual tokens
  - extracts proxy text embeddings from the Florence language encoder
  - builds cosine similarity matrices
  - logs retrieval-style sanity metrics
  - saves figures and artifacts

Important:
Florence-2 is a seq2seq multimodal model, not a CLIP-style dual encoder.
So the embeddings here are engineering proxies for analysis and distillation prep:
  * image embedding = mean pooled projected image tokens
  * text embedding  = mean pooled language-encoder hidden states
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/Florence-2-base")
    parser.add_argument("--image_root", type=str, required=True, help="COCO image directory")
    parser.add_argument("--captions_json", type=str, required=True, help="COCO captions annotation JSON")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="caption",
        choices=["caption", "detailed_caption", "custom"],
        help="How to wrap captions/prompts.",
    )
    parser.add_argument(
        "--custom_prompt_prefix",
        type=str,
        default="",
        help='Used only when prompt_mode=custom. Example: "Describe: "',
    )
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


def build_text_inputs(processor, captions: List[str], prompt_mode: str, max_text_len: int, custom_prompt_prefix: str = ""):
    if prompt_mode == "caption":
        texts = captions
    elif prompt_mode == "detailed_caption":
        texts = [f"Describe with a paragraph: {c}" for c in captions]
    elif prompt_mode == "custom":
        texts = [f"{custom_prompt_prefix}{c}" for c in captions]
    else:
        raise ValueError(f"Unsupported prompt_mode={prompt_mode}")

    text_inputs = processor.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_len,
    )
    return texts, text_inputs


@torch.no_grad()
def extract_image_embeddings(model, processor, image_paths: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    proc = processor(images=images, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device)

    image_tokens = model._encode_image(pixel_values)
    image_emb = image_tokens.mean(dim=1)
    image_norms = torch.linalg.vector_norm(image_emb, dim=-1)
    image_emb = F.normalize(image_emb, dim=-1)
    return image_emb, image_norms


@torch.no_grad()
def extract_text_embeddings(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    token_embeds = model.get_input_embeddings()(input_ids)
    encoder_outputs = model.language_model.model.encoder(
        input_ids=None,
        attention_mask=attention_mask,
        inputs_embeds=token_embeds,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = encoder_outputs.last_hidden_state

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    text_norms = torch.linalg.vector_norm(pooled, dim=-1)
    pooled = F.normalize(pooled, dim=-1)
    return pooled, text_norms


def compute_retrieval_metrics(sim: np.ndarray) -> Dict[str, float]:
    n = sim.shape[0]
    assert sim.shape[0] == sim.shape[1], "Expecting square similarity matrix."

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


def save_similarity_figure(sim: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(sim, aspect="auto")
    plt.colorbar()
    plt.title("Florence-2 proxy image-text cosine similarity")
    plt.xlabel("Text index")
    plt.ylabel("Image index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_norm_hist(image_norms: np.ndarray, text_norms: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(image_norms, bins=20, alpha=0.7, label="image norms")
    plt.hist(text_norms, bins=20, alpha=0.7, label="text norms")
    plt.legend()
    plt.title("Embedding norms before L2 normalization")
    plt.xlabel("Norm")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval()

    all_image_emb = []
    all_text_emb = []
    all_image_norms = []
    all_text_norms = []

    for start in tqdm(range(0, len(pairs), args.batch_size), desc="Batches"):
        batch = pairs[start : start + args.batch_size]
        image_paths = [x["image_path"] for x in batch]
        captions = [x["caption"] for x in batch]

        _, text_inputs = build_text_inputs(
            processor=processor,
            captions=captions,
            prompt_mode=args.prompt_mode,
            max_text_len=args.max_text_len,
            custom_prompt_prefix=args.custom_prompt_prefix,
        )

        image_emb, image_norms = extract_image_embeddings(model, processor, image_paths, device)
        text_emb, text_norms = extract_text_embeddings(
            model=model,
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            device=device,
        )

        all_image_emb.append(image_emb.cpu())
        all_text_emb.append(text_emb.cpu())
        all_image_norms.append(image_norms.cpu())
        all_text_norms.append(text_norms.cpu())

    image_emb = torch.cat(all_image_emb, dim=0).numpy()
    text_emb = torch.cat(all_text_emb, dim=0).numpy()
    image_norms = torch.cat(all_image_norms, dim=0).numpy()
    text_norms = torch.cat(all_text_norms, dim=0).numpy()

    sim = image_emb @ text_emb.T
    metrics = compute_retrieval_metrics(sim)
    metrics.update(
        {
            "model_id": args.model_id,
            "num_samples": int(len(pairs)),
            "batch_size": int(args.batch_size),
            "device": str(device),
            "dtype": str(dtype),
            "prompt_mode": args.prompt_mode,
            "image_norm_mean_before_l2": float(image_norms.mean()),
            "image_norm_std_before_l2": float(image_norms.std()),
            "text_norm_mean_before_l2": float(text_norms.mean()),
            "text_norm_std_before_l2": float(text_norms.std()),
        }
    )

    if torch.cuda.is_available() and device.type == "cuda":
        metrics["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / 1024**2)

    np.savez_compressed(
        output_dir / "embeddings.npz",
        image_embeddings=image_emb,
        text_embeddings=text_emb,
        image_norms=image_norms,
        text_norms=text_norms,
        similarity=sim,
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_similarity_figure(sim, output_dir / "similarity_matrix.png")
    save_norm_hist(image_norms, text_norms, output_dir / "embedding_norm_hist.png")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
