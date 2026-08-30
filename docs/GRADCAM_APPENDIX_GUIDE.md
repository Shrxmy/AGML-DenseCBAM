# Grad-CAM Appendix Implementation Guide

## 1. Current feasibility

Grad-CAM is directly implementable with the existing final V2 checkpoints. Both model types expose spatial feature maps that remain connected to the TMD output:

| Model | Grad-CAM target layer | Reason |
|---|---|---|
| Reconstructed benchmark | `benchmark_fusion_conv` | Last spatial fused-attention representation before global pooling |
| Proposed AGML-DenseCBAM | `cbam_attention` | Post-CBAM spatial representation used by the primary TMD branch |

Gradients are calculated for the model's predicted TMD class. Each appendix panel contains:

1. the exact model input, including deterministic synthetic corruption when applicable;
2. the Grad-CAM heatmap;
3. the heatmap overlay;
4. predicted class and Normal/Subluxation probabilities.

## 2. Recommended final V2 appendix generation

Run from the repository root in the WSL `thesis-env` environment:

```bash
python scripts/generate_gradcam_appendix.py \
  --results_root chapter4_results/final_v2 \
  --output_dir chapter4_results/final_v2/gradcam_appendix \
  --samples_per_class 1 \
  --seed 42
```

With five folds, two true classes, two scenarios, and two models, this creates 40 individual model/sample panels plus 20 paired benchmark-versus-proposed comparison panels:

```text
5 folds × 2 true classes × 2 scenarios × 2 models = 40 panels
```

Outputs include:

```text
chapter4_results/final_v2/gradcam_appendix/
├── clean/fold_1/...fold_5/
├── artifact_mix/fold_1/...fold_5/
├── comparisons/clean/fold_1/...fold_5/
├── comparisons/artifact_mix/fold_1/...fold_5/
└── gradcam_appendix_metadata.csv
```

The metadata CSV records the true class, predicted class, correctness, probabilities, synthetic-artifact label, target layer, selection seed/key, image size, overlay alpha, original-image SHA-256, model-input SHA-256, checkpoint SHA-256, prediction-table SHA-256, and resolved/recorded paths. `gradcam_appendix_config.json` fingerprints the generation script and metadata. The generator requires five folds by default, verifies test-folder paths, rejects explicit validation results, and requires the seed/image size to match the recorded training configuration.

## 3. Anti-cherry-picking selection rule

Appendix samples are selected using a stable SHA-256 key within each fold, scenario, and true class. Selection does not use model correctness, confidence, or heatmap appearance. The same selected image is then supplied to the benchmark and proposed model.

This permits paired visual comparison while preventing manual selection of only favorable examples. Misclassified examples remain in the appendix if the deterministic selection rule chooses them. The selection seed is locked to the recorded evaluation seed and stored in the appendix metadata; it must not be searched repeatedly to obtain more visually favorable heatmaps.

## 4. Generate one specified image

For a manually documented case, use:

```bash
python scripts/generate_gradcam.py \
  --checkpoint chapter4_results/final_v2/proposed_artifact_mix/fold_1_proposed_artifact_mix_best.keras \
  --image data_5_fold/fold_1/test/normal/<FILENAME>.jpg \
  --model_type proposed \
  --scenario artifact_mix \
  --precision_policy auto \
  --output_dir chapter4_results/final_v2/gradcam_manual
```

`auto` restores the precision policy from the checkpoint's adjacent `run_config.json`. For a legacy checkpoint without that file, specify `--precision_policy float32` or `--precision_policy mixed_float16` explicitly; precision is never inferred from current hardware.

The script saves:

- raw model input;
- raw grayscale heatmap;
- color overlay;
- three-panel appendix figure;
- JSON prediction and provenance metadata.

## 5. ROI metrics when expert annotations become available

Grad-CAM is qualitative by default. Localization Energy and intersection-over-union must not be reported without independently prepared expert ROI annotations.

The ROI CSV must contain:

```csv
sample_id,x_min,y_min,x_max,y_max
<filename>.jpg,100,80,420,360
```

Then run:

```bash
python scripts/generate_gradcam.py \
  --checkpoint <CHECKPOINT.keras> \
  --image <IMAGE.jpg> \
  --model_type proposed \
  --scenario artifact_mix \
  --roi_csv expert_rois.csv \
  --heatmap_threshold 0.5 \
  --output_dir chapter4_results/final_v2/gradcam_with_roi
```

The ROI annotator's credentials, annotation procedure, coordinate system, and the heatmap threshold must be prespecified and documented. ROI annotations created after viewing model heatmaps would introduce confirmation bias.

## 6. Suggested appendix presentation

A defensible appendix organization is:

| Appendix subsection | Content |
|---|---|
| Appendix A | Benchmark and proposed clean-condition panels by fold |
| Appendix B | Benchmark and proposed artifact-condition panels by fold |
| Appendix C | Grad-CAM metadata table with true/predicted classes and confidence |
| Appendix D | ROI localization metrics only if independent expert annotations exist |

Suggested caption:

> **Grad-CAM comparison for a deterministically selected held-out image.** The left panel shows the exact model input, the center panel shows the class-targeted Grad-CAM heatmap, and the right panel shows the heatmap overlay. Samples were selected by a prespecified stable-hash rule independent of model correctness or heatmap appearance. Grad-CAM visualizations are qualitative and do not establish anatomical localization accuracy without expert region-of-interest annotations.

## 7. Interpretation limitations

- Grad-CAM indicates sensitivity of the predicted score to coarse feature-map regions; it does not prove causal reasoning.
- A visually plausible heatmap does not establish clinical validity.
- Benchmark and proposed heatmaps use different architecture-appropriate target layers.
- Heatmap resolution is constrained by the final spatial feature-map resolution.
- Artifact-mix panels visualize the deterministic synthetic input seen by the model, not a naturally acquired clinical artifact.
- V3 Grad-CAM should be generated only after a V3 checkpoint is formally accepted; the completed V2 checkpoints remain the appropriate source for the current final-result appendix.
