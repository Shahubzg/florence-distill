#!/usr/bin/env python3
"""
Compare a distilled student against the Florence-2 teacher on the same COCO split.

The script reports three groups of metrics:
  1. Retrieval-style VL quality for teacher and student separately
  2. Teacher-matching metrics (KL / MAE / matrix correlation)
  3. Efficiency metadata such as parameter counts, checkpoint size, and timing

The retrieval metrics are computed batch-wise from BxB VL score matrices, matching
the benchmark style already used in teacher_baseline.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from distill_train import (
    DistillDataset,
    compute_student_vl_matrix,
    compute_teacher_vl_matrix,
    set_seed,
    vl_score_distillation_loss,
)
from student_model import StudentConfig, TinyCLIPStudent, build_student_with_florence_encoders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher_model_id",
        type=str,
        default="/leonardo_work/IscrC_DEMOLLM/florence_distill/models/florence-2-base",
    )
    parser.add_argument(
        "--pairs_csv",
        type=str,
        default="/leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/results_baseline/pairs_preview.csv",
    )
    parser.add_argument(
        "--embeddings_npz",
        type=str,
        default="/leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/results_baseline/embeddings.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Student checkpoint from distill_train.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/eval_student_vs_teacher",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_response_len", type=int, default=64)
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=0,
        help="Optional: evaluate only the first N samples after loading (0 = use all).",
    )
    parser.add_argument(
        "--include_bbox_prompt_0",
        action="store_true",
        help="Also benchmark the first per-object bbox prompt when available.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


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


def _batched(records: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _limit_dataset(dataset: DistillDataset, limit_samples: int) -> None:
    if limit_samples <= 0 or limit_samples >= len(dataset):
        return
    dataset.df = dataset.df.iloc[:limit_samples].reset_index(drop=True)
    dataset.teacher_img_emb = dataset.teacher_img_emb[:limit_samples]


def _load_records(pairs_csv: Path, embeddings_npz: Path, limit_samples: int) -> List[Dict]:
    dataset = DistillDataset(pairs_csv, embeddings_npz)
    _limit_dataset(dataset, limit_samples)
    return [dataset[idx] for idx in range(len(dataset))]


def _timed_call(fn, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return result, elapsed


def _matrix_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if a_flat.size < 2:
        return float("nan")
    if np.allclose(a_flat.std(), 0.0) or np.allclose(b_flat.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _compute_retrieval_metrics(sim: np.ndarray) -> Dict[str, float]:
    n = sim.shape[0]
    assert sim.shape[0] == sim.shape[1]
    ranks_i2t, ranks_t2i = [], []
    for i in range(n):
        ranks_i2t.append(int(np.where(np.argsort(-sim[i]) == i)[0][0]) + 1)
    for j in range(n):
        ranks_t2i.append(int(np.where(np.argsort(-sim[:, j]) == j)[0][0]) + 1)

    def r_at_k(ranks: List[int], k: int) -> float:
        return float(np.mean(np.array(ranks) <= k))

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


def _aggregate_metric_dicts(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {
        key: float(np.mean([metrics[key] for metrics in metric_dicts]))
        for key in keys
    }


def _extract_prompt_records(
    records: List[Dict],
    tag: str,
) -> tuple[List[Dict], str]:
    if tag == "caption":
        return [r for r in records if r["caption"].strip()], "<CAPTION>"
    if tag == "class_prompt":
        valid = [r for r in records if r["bbox_prompts"]]
        extracted = [
            {
                **r,
                "response": str(r["bbox_prompts"][-1]),
            }
            for r in valid
        ]
        return extracted, "<CAPTION>"
    if tag == "bbox_prompt_0":
        valid = [r for r in records if r["bbox_prompts"]]
        extracted = [
            {
                **r,
                "response": str(r["bbox_prompts"][0]),
            }
            for r in valid
        ]
        return extracted, "<CAPTION>"
    raise ValueError(f"Unknown prompt tag: {tag}")


def _response_for_record(record: Dict, tag: str) -> str:
    if tag == "caption":
        return str(record["caption"])
    return str(record["response"])


def _build_student(
    checkpoint_path: Path,
    teacher,
) -> TinyCLIPStudent:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = ckpt.get("config")
    if cfg_dict is None:
        raise KeyError("Checkpoint does not contain a 'config' entry.")
    cfg = StudentConfig(**cfg_dict)
    student = build_student_with_florence_encoders(teacher, config=cfg)
    state = ckpt.get("student_state_dict", ckpt)
    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    return student


@torch.no_grad()
def evaluate_image_geometry(
    records: List[Dict],
    student: TinyCLIPStudent,
    processor,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    sim_mse, sim_corr = [], []
    for batch in tqdm(list(_batched(records, batch_size)), desc="Image geometry"):
        image_paths = [record["image_path"] for record in batch]
        teacher_img_emb = torch.stack([record["teacher_img_emb"] for record in batch]).to(device)
        pixel_values = _load_pixel_values(processor, image_paths, device)
        raw_emb = student.encoder(pixel_values)
        student_img_emb = F.normalize(raw_emb.float(), dim=-1, eps=1e-6)
        teacher_img_emb_n = F.normalize(teacher_img_emb.float(), dim=-1, eps=1e-6)
        sim_s = student_img_emb @ student_img_emb.T
        sim_t = teacher_img_emb_n @ teacher_img_emb_n.T
        sim_mse.append(F.mse_loss(sim_s, sim_t).item())
        sim_corr.append(_matrix_corr(sim_s.cpu().numpy(), sim_t.cpu().numpy()))
    return {
        "image_sim_mse_mean": float(np.mean(sim_mse)),
        "image_sim_mse_std": float(np.std(sim_mse)),
        "image_sim_corr_mean": float(np.nanmean(sim_corr)),
    }


@torch.no_grad()
def evaluate_prompt_type(
    tag: str,
    records: List[Dict],
    teacher,
    student: TinyCLIPStudent,
    processor,
    batch_size: int,
    max_len: int,
    device: torch.device,
    teacher_dtype: torch.dtype,
    output_dir: Path,
) -> Dict:
    valid_records, task_prompt = _extract_prompt_records(records, tag)
    if not valid_records:
        return {
            "num_samples": 0,
            "num_batches": 0,
            "teacher": {},
            "student": {},
            "student_vs_teacher": {},
        }

    teacher_metrics_all: List[Dict[str, float]] = []
    student_metrics_all: List[Dict[str, float]] = []
    teacher_diag, teacher_off = [], []
    student_diag, student_off = [], []
    kl_values, mae_values, corr_values = [], [], []
    teacher_time_total = 0.0
    student_time_total = 0.0
    saved_batch0 = False

    for batch_idx, batch in enumerate(tqdm(list(_batched(valid_records, batch_size)), desc=f"VL {tag}")):
        image_paths = [record["image_path"] for record in batch]
        responses = [_response_for_record(record, tag) for record in batch]
        pixel_values = _load_pixel_values(processor, image_paths, device)
        teacher_pixel_values = pixel_values.to(dtype=teacher_dtype)
        vis_feats = student.encoder(pixel_values).detach()

        score_t, teacher_elapsed = _timed_call(
            lambda: compute_teacher_vl_matrix(
                teacher,
                processor,
                teacher_pixel_values,
                task_prompt,
                responses,
                max_len,
                device,
            ),
            device,
        )
        score_s, student_elapsed = _timed_call(
            lambda: compute_student_vl_matrix(
                student,
                processor,
                pixel_values,
                vis_feats,
                responses,
                max_len,
                device,
            ),
            device,
        )

        score_t_np = score_t.detach().float().cpu().numpy()
        score_s_np = score_s.detach().float().cpu().numpy()
        teacher_metrics_all.append(_compute_retrieval_metrics(score_t_np))
        student_metrics_all.append(_compute_retrieval_metrics(score_s_np))

        teacher_diag.append(np.diag(score_t_np))
        teacher_off.append(score_t_np[~np.eye(score_t_np.shape[0], dtype=bool)])
        student_diag.append(np.diag(score_s_np))
        student_off.append(score_s_np[~np.eye(score_s_np.shape[0], dtype=bool)])

        kl_values.append(vl_score_distillation_loss(score_s, score_t).item())
        mae_values.append(float(np.mean(np.abs(score_s_np - score_t_np))))
        corr_values.append(_matrix_corr(score_s_np, score_t_np))
        teacher_time_total += teacher_elapsed
        student_time_total += student_elapsed

        if not saved_batch0:
            np.savez_compressed(
                output_dir / f"{tag}_batch0_matrices.npz",
                teacher=score_t_np,
                student=score_s_np,
                abs_diff=np.abs(score_s_np - score_t_np),
            )
            saved_batch0 = True

    teacher_diag_all = np.concatenate(teacher_diag)
    teacher_off_all = np.concatenate(teacher_off)
    student_diag_all = np.concatenate(student_diag)
    student_off_all = np.concatenate(student_off)

    return {
        "num_samples": len(valid_records),
        "num_batches": len(teacher_metrics_all),
        "teacher": {
            **_aggregate_metric_dicts(teacher_metrics_all),
            "matched_mean": float(teacher_diag_all.mean()),
            "unmatched_mean": float(teacher_off_all.mean()),
            "gap": float(teacher_diag_all.mean() - teacher_off_all.mean()),
            "vl_seconds_total": teacher_time_total,
            "vl_seconds_per_sample": teacher_time_total / len(valid_records),
        },
        "student": {
            **_aggregate_metric_dicts(student_metrics_all),
            "matched_mean": float(student_diag_all.mean()),
            "unmatched_mean": float(student_off_all.mean()),
            "gap": float(student_diag_all.mean() - student_off_all.mean()),
            "vl_seconds_total": student_time_total,
            "vl_seconds_per_sample": student_time_total / len(valid_records),
        },
        "student_vs_teacher": {
            "vl_kl_mean": float(np.mean(kl_values)),
            "vl_score_mae_mean": float(np.mean(mae_values)),
            "vl_score_corr_mean": float(np.nanmean(corr_values)),
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    teacher_dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading evaluation records from {args.pairs_csv} ...")
    records = _load_records(Path(args.pairs_csv), Path(args.embeddings_npz), args.limit_samples)
    print(f"Loaded {len(records)} records.")

    print(f"Loading Florence-2 teacher from {args.teacher_model_id} ...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_id,
        torch_dtype=teacher_dtype,
        trust_remote_code=True,
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    processor = AutoProcessor.from_pretrained(args.teacher_model_id, trust_remote_code=True)

    print(f"Loading student checkpoint from {args.checkpoint} ...")
    student = _build_student(Path(args.checkpoint), teacher)
    student.to(device)
    student.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    image_geometry = evaluate_image_geometry(records, student, processor, args.batch_size, device)

    prompt_tags = ["caption", "class_prompt"]
    if args.include_bbox_prompt_0:
        prompt_tags.append("bbox_prompt_0")

    prompt_results = {}
    for tag in prompt_tags:
        prompt_results[tag] = evaluate_prompt_type(
            tag=tag,
            records=records,
            teacher=teacher,
            student=student,
            processor=processor,
            batch_size=args.batch_size,
            max_len=args.max_response_len,
            device=device,
            teacher_dtype=teacher_dtype,
            output_dir=output_dir,
        )

    teacher_params = sum(param.numel() for param in teacher.parameters())
    student_total_params = sum(param.numel() for param in student.parameters())
    student_trainable_params = sum(param.numel() for param in student.parameters() if param.requires_grad)

    summary = {
        "teacher_model_id": args.teacher_model_id,
        "checkpoint": args.checkpoint,
        "pairs_csv": args.pairs_csv,
        "embeddings_npz": args.embeddings_npz,
        "num_records": len(records),
        "batch_size": args.batch_size,
        "max_response_len": args.max_response_len,
        "device": str(device),
        "teacher_dtype": str(teacher_dtype),
        "teacher_params": teacher_params,
        "student_total_params": student_total_params,
        "student_trainable_params": student_trainable_params,
        "student_checkpoint_size_mb": Path(args.checkpoint).stat().st_size / (1024 ** 2),
        "image_geometry": image_geometry,
        "prompts": prompt_results,
    }

    if device.type == "cuda":
        summary["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    summary_path = output_dir / "student_vs_teacher_metrics.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nSaved evaluation summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
