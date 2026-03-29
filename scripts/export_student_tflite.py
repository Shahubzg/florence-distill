#!/usr/bin/env python3
"""
export_student_tflite.py

Thesis-aligned PyTorch -> TFLite exporter for the distilled student model.

Required conversion strategy:
  1) Try direct conversion with litert-torch.
  2) If direct path fails, fallback to:
       PyTorch -> ONNX -> TensorFlow SavedModel -> TFLite

Verification:
  - Always verifies ONNX output against PyTorch (if onnxruntime is installed).
  - Optionally runs a smoke test on the produced .tflite file.

This script is intended to run locally on your laptop.
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
    build_student_reduced_vocab,
    extract_deployment_model,
    count_deployment_params,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _freeze_module_parameters(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _make_example_inputs(
    height: int,
    width: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(1, 3, height, width, device=device),
        torch.zeros(1, seq_len, dtype=torch.long, device=device),
    )

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
                   help="Skip TFLite conversion even if conversion deps are available.")
    p.add_argument("--skip_direct_litert", action="store_true",
                   help="Skip direct litert-torch path and use fallback pipeline.")
    # Architecture options
    p.add_argument("--backbone", type=str, default="resnet18",
                   choices=["resnet18", "mobilenetv2", "custom_tiny"])
    p.add_argument("--vocab_mapping", type=str, default=None,
                   help="Path to vocab_mapping_*.json for reduced vocab export.")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    # Quantization
    p.add_argument("--quantize_int8", action="store_true",
                   help="Apply INT8 post-training quantization to TFLite model.")
    p.add_argument("--calibration_images", type=str, default=None,
                   help="Directory of calibration images for INT8 quantization.")
    p.add_argument("--n_calibration", type=int, default=100,
                   help="Number of calibration samples for INT8 quantization.")
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
    backbone: str = "resnet18",
    vocab_mapping: str | None = None,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
) -> TinyCLIPStudent:
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_id, trust_remote_code=True,
    )
    teacher.eval()
    _freeze_module_parameters(teacher)

    vocab_size = processor.tokenizer.vocab_size or len(processor.tokenizer)
    cfg = StudentConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=seq_len,
        vision_out_dim=d_model,
        backbone=backbone,
    )

    if vocab_mapping:
        print(f"Building student with reduced vocabulary from {vocab_mapping} ...")
        student = build_student_reduced_vocab(cfg, vocab_mapping, florence_model=teacher)
    else:
        student = build_student_with_florence_embeddings(teacher, config=cfg)

    if checkpoint:
        ckpt_data = torch.load(checkpoint, map_location="cpu")
        state = ckpt_data.get("student_state_dict", ckpt_data)
        student.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")

    # Extract deployment-only model (strip text encoder if present)
    student = extract_deployment_model(student)
    print(f"Deployment model params: {count_deployment_params(student)}")

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
    dummy_images, dummy_ids = _make_example_inputs(height, width, seq_len, device)

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


def convert_direct_litert(
    student: TinyCLIPStudent,
    tflite_path: Path,
    height: int,
    width: int,
    seq_len: int,
) -> tuple[bool, str]:
    """
    Direct PyTorch -> TFLite conversion using litert-torch.
    Returns (success, message).
    """
    try:
        import litert_torch  # type: ignore
    except Exception as exc:
        # Some environments fail during import due to torch/torchao ABI/version
        # mismatch (e.g. missing torch.int1). We treat any import-time failure as
        # a non-fatal direct-path miss and fall back to ONNX->TF->TFLite.
        return False, f"litert-torch import failed: {exc}"

    device = next(student.parameters()).device
    example_inputs = _make_example_inputs(height, width, seq_len, device)

    try:
        exported = torch.export.export(student, example_inputs)
    except Exception as exc:
        return False, f"torch.export failed: {exc}"

    try:
        if hasattr(litert_torch, "convert") and hasattr(litert_torch.convert, "to_tflite"):
            tflite_bytes = litert_torch.convert.to_tflite(exported)
        elif hasattr(litert_torch, "to_tflite"):
            tflite_bytes = litert_torch.to_tflite(exported)
        else:
            return False, "Unsupported litert-torch API (no to_tflite entrypoint found)"

        if not isinstance(tflite_bytes, (bytes, bytearray)):
            return False, "litert-torch returned non-bytes payload"

        tflite_path.write_bytes(tflite_bytes)
        size_mb = tflite_path.stat().st_size / (1024 ** 2)
        return True, f"direct conversion succeeded ({size_mb:.2f} MB)"
    except Exception as exc:
        return False, f"litert-torch conversion failed: {exc}"


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
    device = next(student.parameters()).device
    with torch.no_grad():
        pt_out = student(
            torch.from_numpy(dummy_images).to(device),
            torch.from_numpy(dummy_ids).to(device),
        ).cpu().numpy()

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

def _make_representative_dataset(
    calibration_dir: str | None,
    height: int,
    width: int,
    seq_len: int,
    n_samples: int = 100,
):
    """Generator for INT8 calibration data."""
    import glob

    def representative_dataset():
        if calibration_dir:
            from PIL import Image as PILImage
            image_files = sorted(glob.glob(f"{calibration_dir}/*.jpg"))[:n_samples]
            for img_path in image_files:
                img = PILImage.open(img_path).convert("RGB").resize((width, height))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]
                ids = np.zeros((1, seq_len), dtype=np.int64)
                yield [img_tensor.astype(np.float32), ids]
        else:
            for _ in range(min(n_samples, 50)):
                img = np.random.randn(1, 3, height, width).astype(np.float32)
                ids = np.zeros((1, seq_len), dtype=np.int64)
                yield [img, ids]

    return representative_dataset


def convert_onnx_to_tflite(
    onnx_path: Path,
    tflite_path: Path,
    output_dir: Path,
    quantize_int8: bool = False,
    calibration_dir: str | None = None,
    height: int = 224,
    width: int = 224,
    seq_len: int = 64,
    n_calibration: int = 100,
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

    if quantize_int8:
        print("[TFLite] Applying INT8 post-training quantization ...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_representative_dataset(
            calibration_dir, height, width, seq_len, n_calibration
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        # FP16 quantization: reduces model size by ~2× with minimal accuracy loss
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    size_mb = tflite_path.stat().st_size / (1024 ** 2)
    quant_type = "INT8" if quantize_int8 else "FP16"
    print(f"[TFLite] Saved to {tflite_path}  ({size_mb:.2f} MB, {quant_type})")
    return True


def smoke_test_tflite(
    tflite_path: Path,
    height: int,
    width: int,
    seq_len: int,
) -> tuple[bool, str]:
    """
    Basic execution smoke test of a produced TFLite model.
    Returns (success, message).
    """
    try:
        import tensorflow as tf
    except ImportError:
        return False, "tensorflow not installed (cannot run TFLite smoke test)"

    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if len(input_details) < 2:
            return False, f"unexpected input count: {len(input_details)}"

        # Build dummy inputs matching interpreter dtypes
        img = np.random.randn(1, 3, height, width).astype(input_details[0]["dtype"])
        ids = np.zeros((1, seq_len), dtype=input_details[1]["dtype"])

        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.set_tensor(input_details[1]["index"], ids)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])
        return True, f"invoke OK, output shape={tuple(out.shape)}"
    except Exception as exc:
        return False, f"TFLite invoke failed: {exc}"


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
        args.teacher_model_id, args.seq_len, processor, args.checkpoint, device,
        backbone=args.backbone,
        vocab_mapping=args.vocab_mapping,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )

    n_total = sum(p.numel() for p in student.parameters())
    print(f"  Student parameters: {n_total:,} ({n_total * 4 / 1024**2:.1f} MB FP32)")

    # Step 2: ONNX export
    onnx_path = out_dir / args.onnx_name
    export_onnx(student, onnx_path, args.height, args.width, args.seq_len, args.opset)

    # Step 3: ONNX verification
    onnx_ok = verify_onnx(onnx_path, args.height, args.width, args.seq_len, student)

    # Step 4: TFLite conversion according to thesis:
    #         direct litert-torch first, fallback ONNX->TF->TFLite.
    tflite_ok = False
    tflite_path = out_dir / args.tflite_name
    tflite_route = None
    tflite_message = ""
    tflite_smoke_ok = False
    tflite_smoke_msg = ""

    if not args.skip_tflite:
        if not args.skip_direct_litert:
            ok_direct, msg_direct = convert_direct_litert(
                student, tflite_path, args.height, args.width, args.seq_len
            )
            print(f"[TFLite][direct] {msg_direct}")
            if ok_direct:
                tflite_ok = True
                tflite_route = "litert-torch"
                tflite_message = msg_direct
            else:
                print("[TFLite] Falling back to ONNX -> TF -> TFLite ...")

        if not tflite_ok:
            ok_fallback = convert_onnx_to_tflite(
                onnx_path, tflite_path, out_dir,
                quantize_int8=args.quantize_int8,
                calibration_dir=args.calibration_images,
                height=args.height,
                width=args.width,
                seq_len=args.seq_len,
                n_calibration=args.n_calibration,
            )
            tflite_ok = ok_fallback
            if ok_fallback:
                tflite_route = "onnx-tf-tflite"
                tflite_message = "fallback conversion succeeded"
            else:
                tflite_message = "fallback conversion failed"

        if tflite_ok:
            tflite_smoke_ok, tflite_smoke_msg = smoke_test_tflite(
                tflite_path, args.height, args.width, args.seq_len
            )
            print(f"[TFLite][smoke] {tflite_smoke_msg}")
    else:
        print("[TFLite] Skipped (--skip_tflite).")

    # Summary
    summary = {
        "onnx_path": str(onnx_path),
        "onnx_exported": True,
        "onnx_verified": onnx_ok,
        "tflite_path": str(out_dir / args.tflite_name) if tflite_ok else None,
        "tflite_converted": tflite_ok,
        "tflite_route": tflite_route,
        "tflite_message": tflite_message,
        "tflite_smoke_test_ok": tflite_smoke_ok,
        "tflite_smoke_test_message": tflite_smoke_msg,
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
