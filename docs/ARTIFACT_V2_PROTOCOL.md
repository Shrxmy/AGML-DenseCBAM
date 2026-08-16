# Synthetic Artifact Protocol V2

## Status

V2 is a prespecified corrective protocol following the completed V1 pilot. It is not designed to guarantee that the proposed model outperforms the benchmark. Both models receive the same artifact distribution, folds, preprocessing, optimizer settings, and evaluation procedure.

## Reason for revision

The training-only calibration audit found that the V1 maximum-overlay metal transform produced no pixel change in 50 of 200 sampled images (25%). The V1 auxiliary classifier consequently collapsed the clean and metal categories, with 0% clean recall. V1 remains preserved under Git tag `pilot-v1` and `chapter4_results/final_3002_seed42/`.

## Locked V2 corruption ranges

| Category | V2 parameters | Rationale |
|---|---|---|
| None | Unmodified image | Negative/control category |
| Motion blur | Horizontal kernels sampled from 5, 7, or 9 | Moderate range; removes the V1 kernel-11 extreme |
| Gaussian noise | Sigma sampled uniformly from 8 to 12 | Moderate visible noise; removes the V1 sigma-18 extreme |
| Metal streak | 1–2 full-width additive blurred streaks; thickness 2–4 px; intensity increment 80–110; blur sigma 1.5–2.5 | Localized and guaranteed to change pixels; avoids V1's silent no-op |

Artifact categories remain sampled uniformly, including the clean category. Training corruption remains seeded and stochastic by epoch; validation/test corruption remains deterministic by sample and seed.

## Multi-task branch and weighting

- Primary TMD loss weight: `1.0`
- Auxiliary synthetic-artifact loss weight: `0.3`
- TMD branch input: post-CBAM DenseNet features with global average pooling
- Artifact branch input: pre-CBAM DenseNet features with concatenated global average and global max pooling

A development-only Fold 1 smoke test showed that weight `0.1` left artifact loss at random-chance cross-entropy (`≈ ln(4)`) for five epochs. Weight `0.3` restored auxiliary learning without reducing smoke-test TMD accuracy, so it is retained. A second smoke test showed that post-CBAM global-average features still collapsed clean and localized metal into one category. The artifact branch therefore uses pre-CBAM features and adds max pooling to preserve sparse corruption evidence. These decisions were made from development training/validation behavior before the final V2 five-fold run.

## Interpretation safeguards

- These transformations are controlled synthetic stress tests, not physically validated reconstructions of acquisition artifacts.
- Higher artifact-classification accuracy does not by itself establish clinical realism.
- V2 may still produce a negative result; all C1–C4 outcomes must be reported honestly.
- V2 outputs must be written to a new directory and must not overwrite V1.
