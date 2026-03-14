#!/usr/bin/env python3
"""
export_student_tflite.py

Export the TinyCLIP-style student to ONNX and (optionally) TFLite.

Pipeline:
  1. Build student (vision encoder + frozen Florence-2 token embedding)
  2. Export to ONNX using torch.onnx.export  (always runs)
  3. Verify ONNX output with onnxruntime  (always runs if onnxruntime is available)
  4. Convert ONNX → TFLite via onnx-tf + tensorflow  (runs if both installed)

Usage on Leonardo:
    python export_student_tflite.py --teacher_model_id /path/to/florence-2-base

The script is designed to run without GPU; CPU is sufficient for export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoProcessor

# Add src/ to path so student_model is importable when called from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from student_model import (
    TinyCLIPStudent,
    StudentConfig,
    build_student_with_florence_embeddings,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model_id", type=str,
                   default="/leonardo_work/IscrC_DEMOLLM/florence_distill/models/florence-2-base")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional: path to student .pt checkpoint from distill_train.py")
    p.add_argument("--output_dir", type=str,
                   default="/leonardo_work/IscrC_DEMOLLM/florence_distill/ckpts/export")
    p.add_argument("--onnx_name", type=str, default="tinyclip_student.onnx")
    p.add_argument("--tflite_name", type=str, default="tinyclip_student.tflite")
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--opset", type=int, default=14,
                   help="ONNX opset version (>=12 required; 14 recommended).")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--skip_tflite", action="store_true",
                   help="Skip TFLite conversion even if tensorflow/onnx-tf are available.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build student
# ---------------------------------------------------------------------------

def build_student(
    teacher_model_id: str,
    seq_len: int,
    processor,
    checkpoint: str | None,
    device: torch.device,
) -> TinyCLIPStudent:
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_id, trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    vocab_size = processor.tokenizer.vocab_size or len(processor.tokenizer)
    cfg = StudentConfig(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=2,
        max_seq_len=seq_len,
        vision_out_dim=256,
    )
    student = build_student_with_florence_embeddings(teacher, config=cfg)

    if checkpoint:
        ckpt_data = torch.load(checkpoint, map_location="cpu")
        state = ckpt_data.get("student_state_dict", ckpt_data)
        student.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")

    student.to(device)
    student.eval()
    return student


# ---------------------------------------------------------------------------
# Step 2: ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    student: TinyCLIPStudent,
    onnx_path: Path,
    height: int,
    width: int,
    seq_len: int,
    opset: int,
) -> None:
    device = next(student.parameters()).device
    dummy_images = torch.randn(1, 3, height, width, device=device)
    dummy_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            student,
            (dummy_images, dummy_ids),
            str(onnx_path),
            input_names=["images", "input_ids"],
            output_names=["logits"],
            opset_version=opset,
            dynamic_axes={
                "images":    {0: "batch"},
                "input_ids": {0: "batch", 1: "seq_len"},
                "logits":    {0: "batch", 1: "seq_len"},
            },
            do_constant_folding=True,
        )
    print(f"[ONNX] Exported to {onnx_path}")


# ---------------------------------------------------------------------------
# Step 3: Verify ONNX with onnxruntime
# ---------------------------------------------------------------------------

def verify_onnx(
    onnx_path: Path,
    height: int,
    width: int,
    seq_len: int,
    student: TinyCLIPStudent,
) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ONNX verify] onnxruntime not installed — skipping verification.")
        return False

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    dummy_images = np.random.randn(1, 3, height, width).astype(np.float32)
    dummy_ids = np.zeros((1, seq_len), dtype=np.int64)

    ort_out = sess.run(["logits"], {"images": dummy_images, "input_ids": dummy_ids})[0]

    # Compare with PyTorch
    with torch.no_grad():
        pt_out = student(
            torch.from_numpy(dummy_images),
            torch.from_numpy(dummy_ids),
        ).numpy()

    max_diff = float(np.abs(ort_out - pt_out).max())
    print(f"[ONNX verify] Max abs diff (ORT vs PyTorch): {max_diff:.6f}")
    if max_diff < 1e-4:
        print("[ONNX verify] PASSED — outputs match within tolerance.")
        return True
    else:
        print("[ONNX verify] WARNING — outputs differ more than expected.")
        return False


# ---------------------------------------------------------------------------
# Step 4: ONNX → TFLite conversion
# ---------------------------------------------------------------------------

def convert_onnx_to_tflite(
    onnx_path: Path,
    tflite_path: Path,
    output_dir: Path,
) -> bool:
    """Returns True if conversion succeeded."""
    # Check dependencies
    missing = []
    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")
    try:
        import onnx_tf  # noqa: F401
    except ImportError:
        missing.append("onnx-tf")
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        missing.append("tensorflow")

    if missing:
        print(f"[TFLite] Skipping conversion: missing packages: {missing}")
        print("  Install with: pip install onnx onnx-tf tensorflow")
        return False

    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    print("[TFLite] Loading ONNX model ...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    tf_saved_path = output_dir / "tinyclip_student_tf_savedmodel"
    print(f"[TFLite] Converting ONNX → TF SavedModel at {tf_saved_path} ...")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(tf_saved_path))

    print("[TFLite] Converting TF SavedModel → TFLite ...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_saved_path))

    # FP16 quantization: reduces model size by ~2× with minimal accuracy loss
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    size_mb = tflite_path.stat().st_size / (1024 ** 2)
    print(f"[TFLite] Saved to {tflite_path}  ({size_mb:.2f} MB)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load processor first (needed to get vocab_size)
    processor = AutoProcessor.from_pretrained(
        args.teacher_model_id, trust_remote_code=True
    )

    print("Building student model ...")
    student = build_student(
        args.teacher_model_id, args.seq_len, processor, args.checkpoint, device
    )

    n_total = sum(p.numel() for p in student.parameters())
    print(f"  Student parameters: {n_total:,} ({n_total * 4 / 1024**2:.1f} MB FP32)")

    # Step 2: ONNX export
    onnx_path = out_dir / args.onnx_name
    export_onnx(student, onnx_path, args.height, args.width, args.seq_len, args.opset)

    # Step 3: ONNX verification
    onnx_ok = verify_onnx(onnx_path, args.height, args.width, args.seq_len, student)

    # Step 4: TFLite
    tflite_ok = False
    if not args.skip_tflite:
        tflite_path = out_dir / args.tflite_name
        tflite_ok = convert_onnx_to_tflite(onnx_path, tflite_path, out_dir)
    else:
        print("[TFLite] Skipped (--skip_tflite).")

    # Summary
    summary = {
        "onnx_path": str(onnx_path),
        "onnx_exported": True,
        "onnx_verified": onnx_ok,
        "tflite_path": str(out_dir / args.tflite_name) if tflite_ok else None,
        "tflite_converted": tflite_ok,
        "student_params": n_total,
        "student_size_fp32_mb": round(n_total * 4 / 1024 ** 2, 2),
        "height": args.height,
        "width": args.width,
        "seq_len": args.seq_len,
        "opset": args.opset,
    }
    summary_path = out_dir / "export_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nExport summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
