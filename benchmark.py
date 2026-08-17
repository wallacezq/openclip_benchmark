"""
benchmark.py
============
End-to-end latency benchmark for zero-shot image classification using
apple/DFN5B-CLIP-ViT-H-14-378 accelerated with OpenVINO.

Stages timed
------------
1. Model export      — Convert the HuggingFace model to OpenVINO IR (FP16, INT8)
2. Label init        — Build CLIP text embeddings for all labels in labels.json
3. Per-image inference breakdown
     a. Preprocess   — AutoImageProcessor → pixel_values tensor
     b. Visual enc   — OpenVINO visual encoder forward pass (image → embedding)
     c. Matching     — Logit computation + softmax + top-K selection

Usage
-----
    python benchmark.py [--device CPU|GPU] [--runs 20] [--warmup 3]
                        [--image PATH_TO_IMAGE] [--precision fp16|int8|fp16+int8]

If no image is supplied a synthetic 378×378 RGB image is used.
Results are printed to stdout and written to benchmark_results.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Monkey-patch: fix "multiple values for argument 'allow_new'" on Python 3.14+
# (same patch used in classifier.py – must run before any optimum import)
# ---------------------------------------------------------------------------
import functools as _functools
from optimum.utils.normalized_config import NormalizedConfig as _NormalizedConfig


class _ConfigFactory:
    """Non-descriptor callable that replaces functools.partial for NormalizedConfig."""
    __slots__ = ("_func", "_kw")

    def __init__(self, func, **kw):
        self._func = func
        self._kw = kw

    def __call__(self, config):
        return self._func(config, **self._kw)


@classmethod
def _safe_with_args(cls, allow_new=False, **kwargs):
    return _ConfigFactory(cls, allow_new=allow_new, **kwargs)

_NormalizedConfig.with_args = _safe_with_args

for _mod_path in (
    "optimum.exporters.onnx.model_configs",
    "optimum.exporters.openvino.model_configs",
):
    try:
        import importlib
        _mod = importlib.import_module(_mod_path)
    except ImportError:
        continue
    for _name in dir(_mod):
        _obj = getattr(_mod, _name, None)
        if isinstance(_obj, type):
            _ncc = _obj.__dict__.get("NORMALIZED_CONFIG_CLASS")
            if isinstance(_ncc, _functools.partial):
                setattr(_obj, "NORMALIZED_CONFIG_CLASS",
                        _ConfigFactory(_ncc.func, **_ncc.keywords))

# ---------------------------------------------------------------------------
# Main imports (after patch)
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import open_clip
from transformers import AutoImageProcessor
from optimum.intel.openvino import (
    OVModelOpenCLIPForZeroShotImageClassification,
    OVModelOpenCLIPVisual,
    OVWeightQuantizationConfig,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID          = "apple/DFN5B-CLIP-ViT-H-14-378"
OPENCLIP_MODEL_ID = "ViT-H-14-378-quickgelu"
PRETRAINED        = "dfn5b"
LABELS_PATH       = Path(__file__).parent / "labels.json"
OV_BASE_DIR       = Path("DFN5B-CLIP-ViT-H-14-378-openclip")
OV_FP16_DIR       = OV_BASE_DIR / "FP16"
OV_INT8_DIR       = OV_BASE_DIR / "INT8"
RESULTS_PATH      = Path("benchmark_results.json")

TOP_K = 5

CLASS_TEMPLATES: list[str] = [
    "a photo of a {label} in bounding box.",
    "a product photo of a {label} in bounding box.",
    "a retail image of a {label} in bounding box.",
    "a picture of a {label} in bounding box.",
    "an image of a {label} in bounding box.",
    "a photo of {label} in red bounding box.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class InferenceTimes(NamedTuple):
    preprocess_s:  float  # AutoImageProcessor
    visual_enc_s:  float  # OV visual forward pass
    matching_s:    float  # logits + softmax + topk
    total_s:       float  # end-to-end


def _hr(label: str, char: str = "─", width: int = 68) -> None:
    pad = width - len(label) - 2
    print(f"\n{'─' * 2} {label} {'─' * max(pad, 0)}")


def _resolve_ov_device(requested: str) -> str:
    """Return the requested OpenVINO device if available, else CPU."""
    if requested.upper() == "CPU":
        return "CPU"
    try:
        import openvino as ov
        available = ov.Core().available_devices
        if requested.upper() in [d.upper() for d in available]:
            return requested.upper()
        logger.warning(
            "Device '%s' not available %s — falling back to CPU.", requested, available
        )
    except Exception as exc:
        logger.warning("Could not query OV devices (%s) — falling back to CPU.", exc)
    return "CPU"


def _load_labels() -> list[str]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"labels.json not found at {LABELS_PATH}")
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    logger.info("Loaded %d labels from %s", len(labels), LABELS_PATH)
    return labels


def _make_sample_image() -> np.ndarray:
    """Return a synthetic 378×378 RGB image (gradient pattern)."""
    h, w = 378, 378
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(50, 200, w, dtype=np.uint8)          # R gradient
    img[:, :, 1] = np.linspace(100, 50, w, dtype=np.uint8)          # G gradient
    img[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = [180, 80, 40]  # centre block
    return img


def _load_image(path: str) -> np.ndarray:
    """Load an image from disk as an RGB numpy array."""
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _ov_runtime_config(device: str) -> dict[str, str]:
    """Return a conservative OpenVINO runtime config for the target device."""
    config: dict[str, str] = {"PERFORMANCE_HINT": "LATENCY"}
    if device.upper() == "GPU":
        cache_dir = Path(__file__).parent / ".openvino_cache" / "GPU"
        cache_dir.mkdir(parents=True, exist_ok=True)
        config.update(
            {
                "NUM_STREAMS": "1",
                "CACHE_DIR": str(cache_dir.resolve()),
            }
        )
    return config


def _load_ov_visual_model(model_dir: Path, requested_device: str):
    """Load the compiled visual model, retrying on CPU if GPU compilation fails."""
    ov_config = _ov_runtime_config(requested_device)
    try:
        return OVModelOpenCLIPVisual.from_pretrained(
            model_dir,
            device=requested_device,
            ov_config=ov_config,
        )
    except RuntimeError as exc:
        if requested_device.upper() == "CPU":
            raise
        logger.warning(
            "OpenVINO model load failed on %s (%s). Retrying on CPU.",
            requested_device,
            exc,
        )
        return OVModelOpenCLIPVisual.from_pretrained(
            model_dir,
            device="CPU",
            ov_config=_ov_runtime_config("CPU"),
        )


def _parse_precision_mode(value: str) -> list[tuple[str, Path]]:
    """Map the CLI precision selector to the model directories to benchmark."""
    normalized = value.strip().lower()
    if normalized == "fp16":
        return [("FP16", OV_FP16_DIR)]
    if normalized == "int8":
        return [("INT8", OV_INT8_DIR)]
    if normalized in {"fp16+int8", "both"}:
        return [("FP16", OV_FP16_DIR), ("INT8", OV_INT8_DIR)]
    raise ValueError("precision must be one of: fp16, int8, fp16+int8")


# ---------------------------------------------------------------------------
# Phase 1 — Model export
# ---------------------------------------------------------------------------

def phase_export_models() -> dict[str, float]:
    """Export the HuggingFace model to OpenVINO IR FP16 and INT8.

    Returns timing in seconds for each precision.  Skips export if the
    directory already exists (cached from a previous run).
    """
    timings: dict[str, float] = {}

    # ── FP16 ──────────────────────────────────────────────────────────────
    _hr("Phase 1a  │  Model export → FP16")
    if OV_FP16_DIR.exists():
        logger.info("FP16 model already present at %s — skipping export.", OV_FP16_DIR)
        timings["export_fp16_s"] = 0.0
        timings["export_fp16_cached"] = True
    else:
        logger.info("Exporting FP16 OV model from %s …", MODEL_ID)
        t0 = time.perf_counter()
        OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(
            MODEL_ID
        ).save_pretrained(OV_FP16_DIR)
        timings["export_fp16_s"] = time.perf_counter() - t0
        timings["export_fp16_cached"] = False
        logger.info("FP16 export done in %.1f s → %s", timings["export_fp16_s"], OV_FP16_DIR)

    # ── INT8 ──────────────────────────────────────────────────────────────
    _hr("Phase 1b  │  Model export → INT8")
    if OV_INT8_DIR.exists():
        logger.info("INT8 model already present at %s — skipping export.", OV_INT8_DIR)
        timings["export_int8_s"] = 0.0
        timings["export_int8_cached"] = True
    else:
        logger.info("Exporting INT8 quantised OV model from %s …", MODEL_ID)
        t0 = time.perf_counter()
        OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(
            MODEL_ID,
            quantization_config=OVWeightQuantizationConfig(bits=8),
        ).save_pretrained(OV_INT8_DIR)
        timings["export_int8_s"] = time.perf_counter() - t0
        timings["export_int8_cached"] = False
        logger.info("INT8 export done in %.1f s → %s", timings["export_int8_s"], OV_INT8_DIR)

    return timings


# ---------------------------------------------------------------------------
# Phase 2 — Label embedding initialisation
# ---------------------------------------------------------------------------

def phase_init_label_embeddings(
    labels: list[str],
    clip_model,
    tokenizer,
) -> tuple[torch.Tensor, float]:
    """Build the zero-shot weight matrix from text prompts.

    Uses the cached .pth file if present and the label list matches;
    otherwise performs a full rebuild.  Returns (weight_matrix, elapsed_s).
    """
    _hr("Phase 2  │  Label embedding initialisation")

    weights_path = Path("clip_zeroshot_cls.pth")
    labels_path  = Path("clip_zeroshot_cls_labels.json")

    # Check for a usable cache
    if weights_path.exists() and labels_path.exists():
        try:
            with open(labels_path) as f:
                cached_labels = json.load(f)
            if cached_labels == labels:
                logger.info(
                    "All %d label embeddings are cached — loading from disk.", len(labels)
                )
                t0 = time.perf_counter()
                weights = torch.load(weights_path, map_location="cpu")
                elapsed = time.perf_counter() - t0
                logger.info("Loaded cached weights in %.3f s (shape %s)", elapsed, list(weights.shape))
                return weights, elapsed
        except Exception as exc:
            logger.warning("Cache unusable (%s) — rebuilding.", exc)

    # Full rebuild
    logger.info("Building embeddings for %d labels …", len(labels))
    t0 = time.perf_counter()
    weights = []
    clip_model.eval()
    for i, label in enumerate(labels):
        texts = [t.format(label=label) for t in CLASS_TEMPLATES]
        with torch.no_grad():
            emb = clip_model.encode_text(tokenizer(texts))
        emb = F.normalize(emb, dim=-1).mean(dim=0)
        emb = emb / emb.norm()
        weights.append(emb)
        if (i + 1) % 10 == 0 or (i + 1) == len(labels):
            logger.info("  encoded %d / %d labels", i + 1, len(labels))

    weight_matrix = torch.stack(weights, dim=1)
    elapsed = time.perf_counter() - t0

    # Persist
    torch.save(weight_matrix, weights_path)
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    logger.info("Built and saved label embeddings in %.1f s (shape %s)", elapsed, list(weight_matrix.shape))
    return weight_matrix, elapsed


# ---------------------------------------------------------------------------
# Phase 3 — Per-image inference with timing breakdown
# ---------------------------------------------------------------------------

def _run_single(
    img_array: np.ndarray,
    processor,
    ov_vision,
    zeroshot_weights: torch.Tensor,
    labels: list[str],
) -> tuple[InferenceTimes, list[dict]]:
    """Run one inference pass and return detailed timings + top-K predictions."""

    # ── a. Preprocess ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    img_inputs = processor(images=[img_array], return_tensors="pt")
    t_preprocess = time.perf_counter() - t0

    # ── b. Visual encoding (OV) ─────────────────────────────────────────
    t1 = time.perf_counter()
    visual_out   = ov_vision(**img_inputs)
    image_features = visual_out["image_features"]  # (1, D)
    t_visual = time.perf_counter() - t1

    # ── c. Matching (logits + softmax + top-K) ──────────────────────────
    t2 = time.perf_counter()
    logits = 100.0 * image_features @ zeroshot_weights   # (1, N)
    probs  = torch.softmax(logits, dim=-1).squeeze()      # (N,)
    top_values, top_indices = probs.topk(TOP_K)
    t_match = time.perf_counter() - t2

    total = t_preprocess + t_visual + t_match
    times = InferenceTimes(
        preprocess_s=t_preprocess,
        visual_enc_s=t_visual,
        matching_s=t_match,
        total_s=total,
    )
    predictions = [
        {
            "rank": rank + 1,
            "label": labels[idx.item()],
            "short_name": labels[idx.item()].split("/")[-1],
            "confidence_pct": round(val.item() * 100, 4),
        }
        for rank, (val, idx) in enumerate(zip(top_values, top_indices))
    ]
    return times, predictions


def phase_inference_benchmark(
    img_array: np.ndarray,
    processor,
    ov_vision,
    zeroshot_weights: torch.Tensor,
    labels: list[str],
    precision: str,
    warmup_runs: int,
    bench_runs: int,
) -> dict:
    """Warm up, then run bench_runs measured passes.  Returns timing statistics."""

    _hr(f"Phase 3  │  Inference benchmark  [{precision}]  warmup={warmup_runs}  runs={bench_runs}")

    # Warmup
    logger.info("Running %d warmup iterations …", warmup_runs)
    for _ in range(warmup_runs):
        _run_single(img_array, processor, ov_vision, zeroshot_weights, labels)

    # Measured runs
    logger.info("Running %d measured iterations …", bench_runs)
    all_times: list[InferenceTimes] = []
    last_predictions: list[dict] = []

    for i in range(bench_runs):
        t, preds = _run_single(img_array, processor, ov_vision, zeroshot_weights, labels)
        all_times.append(t)
        last_predictions = preds
        if (i + 1) % max(1, bench_runs // 5) == 0:
            logger.info(
                "  [%d/%d] pre=%.1fms  vis=%.1fms  match=%.2fms  total=%.1fms",
                i + 1, bench_runs,
                t.preprocess_s * 1000,
                t.visual_enc_s * 1000,
                t.matching_s   * 1000,
                t.total_s      * 1000,
            )

    def _stats(vals: list[float]) -> dict:
        return {
            "mean_ms":   round(statistics.mean(vals) * 1000, 3),
            "min_ms":    round(min(vals) * 1000, 3),
            "max_ms":    round(max(vals) * 1000, 3),
            "stdev_ms":  round(statistics.stdev(vals) * 1000, 3) if len(vals) > 1 else 0.0,
        }

    result = {
        "precision": precision,
        "runs": bench_runs,
        "warmup_runs": warmup_runs,
        "preprocess":  _stats([t.preprocess_s  for t in all_times]),
        "visual_enc":  _stats([t.visual_enc_s  for t in all_times]),
        "matching":    _stats([t.matching_s    for t in all_times]),
        "total":       _stats([t.total_s       for t in all_times]),
        "top_predictions": last_predictions,
    }
    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _print_export_summary(export_timings: dict) -> None:
    _hr("Export summary")
    rows = [
        ("FP16", export_timings.get("export_fp16_s", 0), export_timings.get("export_fp16_cached", True)),
        ("INT8", export_timings.get("export_int8_s", 0), export_timings.get("export_int8_cached", True)),
    ]
    print(f"  {'Precision':<8}  {'Time (s)':>10}  {'Notes'}")
    print(f"  {'─' * 8}  {'─' * 10}  {'─' * 20}")
    for prec, secs, cached in rows:
        note = "cached (skipped)" if cached else "freshly exported"
        print(f"  {prec:<8}  {secs:>10.1f}  {note}")


def _print_label_init_summary(elapsed_s: float, n_labels: int) -> None:
    _hr("Label init summary")
    print(f"  Labels:        {n_labels}")
    print(f"  Elapsed:       {elapsed_s * 1000:.1f} ms" if elapsed_s < 1 else
          f"  Elapsed:       {elapsed_s:.2f} s")
    print(f"  Per label:     {elapsed_s / n_labels * 1000:.2f} ms/label")


def _print_inference_summary(results: list[dict]) -> None:
    _hr("Inference benchmark summary")
    fields = [
        ("Stage",        12),
    ]
    col_fields = ["mean_ms", "min_ms", "max_ms", "stdev_ms"]

    # Header
    hdr  = f"  {'Stage':<18}"
    hdr += "".join(f"  {p['precision']:>8}" for p in results)
    print(hdr)
    print("  " + "─" * (18 + 10 * len(results)))

    stage_keys = [
        ("preprocess",  "Preprocess"),
        ("visual_enc",  "Visual enc (OV)"),
        ("matching",    "Matching"),
        ("total",       "Total"),
    ]
    for key, label in stage_keys:
        row = f"  {label:<18}"
        for r in results:
            row += f"  {r[key]['mean_ms']:>7.2f}ms"
        print(row)

    print()
    for r in results:
        print(f"  [{r['precision']}] std-dev breakdown:")
        for key, label in stage_keys:
            s = r[key]
            print(f"    {label:<18}  mean={s['mean_ms']:>7.2f}ms  "
                  f"min={s['min_ms']:>7.2f}ms  max={s['max_ms']:>7.2f}ms  "
                  f"σ={s['stdev_ms']:>6.2f}ms")
        print()


def _print_top_predictions(results: list[dict]) -> None:
    _hr("Top predictions (last run)")
    for r in results:
        print(f"  [{r['precision']}]")
        for p in r["top_predictions"]:
            bar = "█" * int(p["confidence_pct"] / 2)
            print(f"    #{p['rank']}  {p['label']:<55} {p['confidence_pct']:>6.2f}%  {bar}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCLIP + OpenVINO latency benchmark")
    parser.add_argument("--device",  default="GPU",  help="OV device (default: GPU, falls back to CPU)")
    parser.add_argument("--runs",    type=int, default=20,  help="Number of measured inference runs (default: 20)")
    parser.add_argument("--warmup",  type=int, default=3,   help="Number of warmup runs (default: 3)")
    parser.add_argument("--image",   default=None,          help="Path to a sample image (default: synthetic)")
    parser.add_argument(
        "--precision",
        default="fp16+int8",
        help="Inference precision to run: fp16, int8, or fp16+int8 (default: fp16+int8)",
    )
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip model export if OV directories already exist (implied automatically)")
    args = parser.parse_args()

    try:
        precision_plan = _parse_precision_mode(args.precision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print()
    print("=" * 68)
    print("  OpenCLIP + OpenVINO  ─  Latency Benchmark")
    print(f"  Model  : {MODEL_ID}")
    print(f"  Device : {args.device}  (auto-resolved)")
    print(f"  Precision mode : {args.precision}")
    print(f"  Runs   : {args.warmup} warmup + {args.runs} measured")
    print("=" * 68)

    ov_device = _resolve_ov_device(args.device)
    logger.info("Using OpenVINO device: %s", ov_device)

    # ── Load labels ────────────────────────────────────────────────────────
    labels = _load_labels()

    # ── Prepare sample image ───────────────────────────────────────────────
    if args.image:
        logger.info("Loading sample image from %s", args.image)
        img_array = _load_image(args.image)
    else:
        logger.info("No image supplied — using synthetic 378×378 test image")
        img_array = _make_sample_image()

    all_results: dict = {
        "model_id": MODEL_ID,
        "ov_device": ov_device,
        "runs": args.runs,
        "warmup_runs": args.warmup,
        "image_source": args.image or "synthetic",
        "n_labels": len(labels),
    }

    # ── Phase 1: Model export ──────────────────────────────────────────────
    export_timings = phase_export_models()
    all_results["export"] = export_timings
    _print_export_summary(export_timings)

    # ── Shared: load OpenCLIP model for text encoding ─────────────────────
    logger.info("Loading OpenCLIP model (%s / %s) for text encoding …", OPENCLIP_MODEL_ID, PRETRAINED)
    clip_model, _, _ = open_clip.create_model_and_transforms(OPENCLIP_MODEL_ID, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL_ID)
    clip_model.eval()

    # ── Shared: image processor ───────────────────────────────────────────
    logger.info("Loading AutoImageProcessor for %s …", MODEL_ID)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    # ── Phase 2: Label embedding initialisation ───────────────────────────
    zeroshot_weights, label_init_s = phase_init_label_embeddings(labels, clip_model, tokenizer)
    all_results["label_init"] = {
        "n_labels": len(labels),
        "elapsed_s": round(label_init_s, 4),
        "elapsed_ms": round(label_init_s * 1000, 2),
        "per_label_ms": round(label_init_s / len(labels) * 1000, 4),
    }
    _print_label_init_summary(label_init_s, len(labels))

    # ── Phase 3: Inference benchmarks (selected precision(s)) ──────────────
    inference_results: list[dict] = []

    for precision, model_dir in precision_plan:
        logger.info("Loading OV visual model [%s] from %s on %s …", precision, model_dir, ov_device)
        ov_vision = _load_ov_visual_model(model_dir, ov_device)

        result = phase_inference_benchmark(
            img_array=img_array,
            processor=processor,
            ov_vision=ov_vision,
            zeroshot_weights=zeroshot_weights,
            labels=labels,
            precision=precision,
            warmup_runs=args.warmup,
            bench_runs=args.runs,
        )
        inference_results.append(result)
        del ov_vision  # free device memory before loading next precision

    all_results["inference"] = inference_results

    # ── Print full report ──────────────────────────────────────────────────
    _print_inference_summary(inference_results)
    _print_top_predictions(inference_results)

    # ── Speedup (INT8 vs FP16) ─────────────────────────────────────────────
    if len(inference_results) == 2:
        fp16_total = inference_results[0]["total"]["mean_ms"]
        int8_total = inference_results[1]["total"]["mean_ms"]
        if int8_total > 0:
            speedup = fp16_total / int8_total
            _hr("INT8 vs FP16 speedup")
            print(f"  FP16 mean total : {fp16_total:.2f} ms")
            print(f"  INT8 mean total : {int8_total:.2f} ms")
            print(f"  Speedup         : {speedup:.2f}x  ({'INT8 faster' if speedup > 1 else 'FP16 faster'})")
            all_results["int8_vs_fp16_speedup"] = round(speedup, 4)
    else:
        _hr("Speedup")
        print("  Skipped because only one precision mode was selected.")

    # ── Save JSON report ───────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print()
    print(f"  Full results saved to: {RESULTS_PATH.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()
