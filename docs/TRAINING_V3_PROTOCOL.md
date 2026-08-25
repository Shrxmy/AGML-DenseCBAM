# Training Protocol V3: Two-Stage Fine-Tuning and Conservative Augmentation

## Status

V3 is a prespecified exploratory extension created **after** completion and inspection of V2. V1 and V2 code, checkpoints, results, and conclusions remain preserved. V3 must never overwrite `chapter4_results/final_3002_seed42/` or `chapter4_results/final_v2/`.

Because the V2 outer-fold results have already been inspected, V3 development decisions must use training/validation behavior only. A development smoke run therefore uses `--evaluation_split validation`; it does not generate predictions from an outer test split. The Chapter IV analyzer explicitly rejects validation-only result files. Any later outer-fold V3 results must be described as a separate post-V2 experiment, not as untouched confirmatory validation. Independent external validation remains preferable.

## Fixed V3 changes

V3 changes only the training schedule and training-only augmentation. The V2 model heads, CBAM placement, artifact branch, synthetic-artifact protocol, folds, losses, class weighting, image size, and batch size remain unchanged.

### Stage 1: frozen-backbone warm-up

- DenseNet201 backbone: fully frozen
- Task-specific attention and classification heads: trainable
- Epochs: 5
- Learning rate: `1e-4`
- Best checkpoint selected by primary validation TMD loss

### Stage 2: selective fine-tuning

- Unfreeze from DenseNet layer `conv5_block1_0_bn` through the final backbone activation
- Earlier DenseNet blocks remain frozen
- DenseNet BatchNormalization layers remain frozen because the batch size is 8
- Task-specific heads remain trainable
- Learning rate: `1e-5`
- Maximum combined Stage 1 + Stage 2 duration: 50 epochs
- Early stopping: primary validation TMD loss, patience 5
- ReduceLROnPlateau: factor 0.1, patience 3, minimum learning rate `1e-6`
- The globally better Stage 1 or Stage 2 checkpoint is retained according to primary validation TMD loss

### Conservative training-only augmentation

After resize to 224 × 224 and before synthetic-artifact application, each training image receives one seeded affine transformation containing:

| Transformation | Locked range |
|---|---:|
| Horizontal flip | Probability 0.5 |
| Rotation | Uniform from −5° to +5° |
| Horizontal translation | Uniform from −3% to +3% of width |
| Vertical translation | Uniform from −3% to +3% of height |
| Zoom | Uniform scale from 0.95 to 1.05 |
| Contrast | Uniform factor from 0.90 to 1.10 |
| Border handling | Reflection padding |

No vertical flip, elastic deformation, aggressive crop, or unrestricted geometric transformation is permitted. Validation and test images receive no V3 geometric or contrast augmentation. The locked V2 synthetic-artifact condition remains deterministic for validation/test and stochastic by epoch for training.

## Unchanged settings

- Input: 224 × 224 RGB with DenseNet ImageNet preprocessing
- Batch size: 8
- TMD loss weight: 1.0
- Synthetic-artifact loss weight: 0.3 for the proposed model
- Optimizer: Adam
- TMD class weighting: enabled
- Mixed precision: enabled on GPU
- Maximum epochs: 50 total
- Synthetic-artifact protocol: `v2_moderate_pre_cbam_aux`
- Random seed: 42
- Exact same folds and conditions for benchmark and proposed models
- Canonical run-configuration SHA-256 stored in every fold result
- Cross-fold Python, TensorFlow, precision-policy, script, and configuration consistency checks

## Development smoke command

This command evaluates only Fold 1 validation data and must be written outside final V1/V2 directories:

```bash
python scripts/run_case_5fold_isolated.py \
  --folds_root data_5_fold \
  --model_type proposed \
  --scenario artifact_mix \
  --epochs 10 \
  --warmup_epochs 3 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --fine_tune_learning_rate 1e-5 \
  --artifact_loss_weight 0.3 \
  --two_stage_fine_tuning \
  --conservative_augmentation \
  --evaluation_split validation \
  --fold_limit 1 \
  --output_dir chapter4_results/development_v3/proposed_artifact_mix
```

The shortened 3+7 epoch schedule above is a functionality smoke test only. It does not redefine the final locked schedule of 5 warm-up epochs and up to 45 selective fine-tuning epochs.

## Advancement criteria

Before any V3 outer-fold evaluation:

1. Unit tests and syntax checks pass.
2. Both stages execute and all histories remain finite.
3. Only the intended final DenseNet block is unfrozen in Stage 2, with backbone BatchNormalization frozen.
4. Validation behavior does not show catastrophic divergence.
5. No decision is based on Fold 1 test predictions or the preserved V2 outer-fold results.
6. All four C1–C4 cases use the same V3 schedule and conservative augmentation in any eventual comparison.

V3 is not permitted to be repeatedly altered merely to exceed the V2 score. Negative or neutral V3 results remain valid findings.
