# AGML-DenseCBAM

Undergraduate thesis code and results for classifying temporomandibular joint (TMJ) images as Normal or Subluxation. The proposed model, AGML-DenseCBAM, is a DenseNet201 with CBAM attention and an auxiliary head that recognizes generated image artifacts. It is compared with a reconstructed attention-based DenseNet201 benchmark.

This is a research prototype and not a clinically validated diagnostic tool.

## Results

Each case ran over the same five folds. Values are mean accuracy / F1, in percent.

| Case | Model | Images | Final V2 (primary) | V3 (disclosed follow-up) |
|---|---|---|---:|---:|
| C1 | Benchmark | Clean | 89.21 / 90.07 | 91.84 / 92.36 |
| C2 | Benchmark | Artifact mix | 88.94 / 89.89 | 91.67 / 92.12 |
| C3 | Proposed | Clean | 89.44 / 90.29 | 91.51 / 92.16 |
| C4 | Proposed | Artifact mix | 90.27 / 91.15 | 91.87 / 92.54 |

No primary comparison between the models reaches the manuscript's .005 threshold in either version, so neither version shows that the proposed model is better. V3 changed the training recipe for all four cases and is a second look at the same test folds, so it does not replace V2.

V4 tested three changes to the proposed model on validation data only. None met its adoption rule, so V4 has no test results.

## Repository layout

```text
scripts/                  Training, analysis and Grad-CAM tools
scripts/pipeline/         Data, model, training and evaluation code
scripts/development/      Validation-only screening runner (V3 and V4 screens)
tests/                    Unit tests
base/                     Source study's released code, kept for reference
docs/                     Documentation and approved architecture figures
data/, data_5_fold/       Source images and folds (not in Git)
results/                  Saved results and checkpoints (only checksums are in Git)
paper/                    Word chapters (not in Git)
output/                   Supporting paper material (not in Git)
```

## Documentation

- [Guide](docs/GUIDE.md): how to run everything and how to restore archived files
- [Technical notes](docs/TECHNICAL_NOTES.md): data, models, results and limitations
- [System architecture](docs/SYSTEM_ARCHITECTURE.md): approved diagrams
- [Paper rules](paper/README.md): the Word chapters are the authoritative version of the manuscript
- [Groupmates' guide](output/groupmates_guide/README.md): plain-language study guide

Data, papers and models are excluded from Git. Their presence here does not establish permission to reuse or publish them.
