# Technical notes

The Word chapters in `paper/chapters/` are the authoritative account. These notes are a short reference.

## Data

The task is image-level classification against inherited Normal and Subluxation labels. It is not a full diagnosis of temporomandibular disorders.

- 3,425 source files contain 3,010 unique images, with 415 extra copies across 361 duplicate groups.
- 8 images appeared under both labels. These 17 files were excluded, leaving 3,002 images: 1,359 Normal and 1,643 Subluxation.
- Each of the five folds has 2,058 training images, 343 or 344 validation images, and 600 or 601 test images. Each image is tested exactly once.

## Models and shared settings

- **Backbone:** trainable DenseNet201 with ImageNet weights.
- **Benchmark:** pool3 self-attention, downsampling and projection, fused with late DenseNet features, then a two-class output. It is a reconstruction of Sancar et al. (2026), not a copy of the released code.
- **Proposed:** CBAM, then global average pooling, Dense 1024, dropout .5, batch normalization, Dense 128 and a two-class output.
- **Auxiliary head:** branches before CBAM with average and max pooling, Dense 256, dropout .3 and a four-class output (none, motion blur, Gaussian noise, metal streak). The loss weights are 1.0 and 0.3. In C3, every auxiliary target is "none".
- **Training:** Adam at 1e-4, at most 50 epochs, L2 of .01 on the first dense layers, mixed precision, balanced class weights, seed 42. Early stopping uses patience 5, and learning-rate reduction uses factor .1 with patience 3.

## Versions

| | Final V2 (primary) | V3 (disclosed follow-up) |
|---|---|---|
| Input | 224 × 224 | 448 × 224, closer to the original proportions |
| Scaling | `x/127.5 - 1` | ImageNet mean and standard deviation, which the pretrained weights expect |
| Augmentation | Horizontal flip | Flip plus mild rotation, scaling, shifting, brightness and contrast |
| Prediction | Single pass | Average of the image and its mirror |
| Batch size | 8 | 6 |

The first V3 screen also froze backbone batch normalization and collapsed to chance. The repeated screen, v3b, removed that change and met its adoption rule. All four cases then ran once.

V4 was validation-only and needed a mean of at least 93.20% with at least 4 of 5 folds improved, against a V3 baseline of 92.20%.

| Candidate | Change | Mean validation accuracy | Folds improved |
|---|---|---:|---:|
| A | pool3 route and artifact-driven gate | 54.52 (Fold 1 only; did not learn) | Not completed |
| B | DenseNet121 backbone | 92.03 | 3 of 5 |
| C | A without pool3, chosen after seeing Fold 1 validation results | 92.26 | 2 of 5, plus 1 tied |

An earlier seven-candidate Fold 1 screen selected a learning rate of 3e-5. Its five-fold follow-up did not improve on Final V2. Its records are in `results/hyperparameter_tests/` and `results/hyperparameter_confirmatory/`.

## Reading the results

- The manuscript uses .005 across ten primary comparisons. V2's C4 versus C2 values (p = .0112 for accuracy, .0137 for F1) do not reach it, and the V3 gap narrows to 0.20 points of accuracy.
- The V3 gain appears in both models, so it is a recipe effect and not evidence for the proposed model.
- These are complete-model comparisons, not tests of CBAM or the auxiliary head alone.
- Grad-CAM is qualitative and has no expert anatomical validation.
- Recorded speeds include image loading and output handling. V3 speeds are not comparable with V2 because V3 adds an image cache and mirror averaging.

## Known limitations

- **Patient independence is not established.** Folds group images by exact file content because patient IDs are unavailable.
- **The test folds have been seen more than once.** Earlier development work also leaves lifetime test exposure unresolved.
- **Stopping rules differ.** The benchmark monitors `val_loss`, which includes L2. The proposed model monitors `val_tmd_output_loss`. In V2 the benchmark ran 20 to 25 epochs and the proposed model 10 to 20.
- **Label quality is unverified.** The 8 conflicting images show that some labels disagree. In V3, 114 images are wrong in all four cases, which is 45 to 47% of each case's errors. Whether they are mislabeled or just hard needs an expert.
- **Generated artifacts are synthetic.** Gaussian noise differs per color channel on grayscale images, while blur and streaks stay gray, so noise may be easier to recognize.
- **Near-duplicates remain.** Two near-identical pairs (4 images, same class within each pair) passed the exact-file check and cross splits in some folds.

## Verification (November 1, 2026)

For V2, all checksums match, metrics and paired tests recompute from the saved predictions, and C2 and C4 received identical test artifacts. Fold 1 checkpoints of all four cases regenerate their saved predicted classes exactly (601 of 601 each). Folds 2 to 5 were not regenerated. The historical trainer is revision `9249331`, and the current modular code is a later refactor.
