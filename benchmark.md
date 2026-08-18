# Benchmark Guide

This repository includes a latency benchmark for zero-shot classification using the Apple DFN5B CLIP model with OpenVINO acceleration.

The benchmark script measures the end-to-end latency for a sample image and breaks the result down by inference stage.

## Benchmark script

- Script: `benchmark.py`
- Purpose: export Hugging Face CLIP model to OpenVINO IR in FP16 and INT8 formats, initialize label embeddings from `labels.json`, and benchmark image-to-embedding and label matching latency.

## Phases measured

### 1. Model export

The benchmark first converts the Hugging Face model to OpenVINO IR equivalents:

- FP16 export: `DFN5B-CLIP-ViT-H-14-378-openclip/FP16`
- INT8 export: `DFN5B-CLIP-ViT-H-14-378-openclip/INT8`

This phase measures how long it takes to export each precision variant. If the exported model already exists, the benchmark skips the export and marks it as cached.

### 2. Label embedding initialization

The benchmark initializes label embeddings using the labels in `labels.json`.

It:

- loads the label list from `labels.json`
- builds zero-shot text embeddings using CLIP templates
- normalizes the embeddings
- saves a cached weights file named `clip_zeroshot_cls.pth`
- writes the current label manifest to `clip_zeroshot_cls_labels.json`

This captures the cost of converting label text to embedding vectors before classification.

### 3. Inference benchmark per precision

The benchmark then measures latency for a sample image for both FP16 and INT8 models.

Each inference pass is broken into:

1. Preprocess
   - image to processor input tensor
2. Visual encoding
   - OpenVINO visual encoder forward pass
   - image → embedding
3. Matching
   - logits computation
   - softmax
   - top-5 label selection

The script reports:

- mean latency
- minimum latency
- maximum latency
- standard deviation
- end-to-end total latency

## Example usage

Run the benchmark with default settings:

```bash
python benchmark.py
```

Run only FP16 inference:

```bash
python benchmark.py --precision fp16
```

Run only INT8 inference:

```bash
python benchmark.py --precision int8
```

Run both FP16 and INT8 inference:

```bash
python benchmark.py --precision fp16+int8
```

Run with CPU and custom iterations:

```bash
python benchmark.py --device CPU --runs 50 --warmup 5
```

Run with a real sample image:

```bash
python benchmark.py --image ./sample.jpg
```

Export static shape models for NPU inference:

```bash
python benchmark.py --export-npu-static --precision fp16
```

This exports separate image and text encoder models with fixed input shapes (378×378 images, 77-token text context) required for Intel NPU compilation.

Export both FP16 and INT8 NPU static variants:

```bash
python benchmark.py --export-npu-static --precision fp16+int8
```

## Options

```bash
python benchmark.py --help
```

Available options:

- `--device` : OpenVINO device, such as `GPU` or `CPU`
- `--runs` : number of measured inference runs
- `--warmup` : number of warmup runs before measuring
- `--image` : path to an image; otherwise a synthetic test image is used
- `--precision` : select `fp16`, `int8`, or `fp16+int8`
- `--export-npu-static` : export static shape models for Intel NPU (separate image/text encoders with fixed shapes)
- `--skip-export` : flag to skip model export when directories already exist

## Output

The script prints a summary to the terminal and also writes a JSON report to:

- `benchmark_results.json`

The report includes model details, export timings, label init timing, and per-precision inference breakdowns.

## Notes

- The first run can take a while because the OpenVINO export and label embedding initialization are one-time costs.
- If GPU is unavailable, the benchmark falls back to CPU automatically.
- The benchmark uses the sample class labels already defined in the repo under `labels.json`.
- On some integrated GPUs, OpenVINO may fail during kernel compilation with `CL_OUT_OF_HOST_MEMORY`. The benchmark loader now retries on CPU in that case, and the `--device CPU` option is the most reliable fallback when the GPU compiler path is unstable.

## NPU static shape export

Intel NPU requires models with static (fixed) input shapes. The `--export-npu-static` option generates separate encoder models:

- **Image encoder**: `image_encoder.xml` with static shape (1, 3, 378, 378)
- **Text encoder**: `text_encoder.xml` with static shape (1, 77)

Precision selection is controlled by `--precision`:

- `--precision fp16` exports and runs NPU static FP16 models
- `--precision int8` exports and runs NPU static INT8 models
- `--precision fp16+int8` exports/runs both and reports both results

Models are stored per precision:

- `DFN5B-CLIP-ViT-H-14-378-openclip-npu-static/FP16/`
- `DFN5B-CLIP-ViT-H-14-378-openclip-npu-static/INT8/`

When running with `--device NPU`, the benchmark now auto-checks the selected precision variant(s) and auto-exports missing static models before inference.

Note: INT8 static export uses NNCF weight compression. If `nncf` is not installed, install it first or run `--precision fp16`.

This approach follows the pattern from [wallacezq/zero-shot-image-classification-npu](https://github.com/wallacezq/zero-shot-image-classification-npu), which demonstrates NPU deployment with static shapes.
