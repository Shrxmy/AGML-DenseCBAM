# Guide

Run everything from the repository root in WSL:

```bash
cd /home/solyvie/workspace/thesis-projects/working-run/AGML-DenseCBAM
source /home/solyvie/environments/miniforge3/etc/profile.d/conda.sh
conda activate thesis-env
python -m unittest discover -s tests
```

## Scripts

| Script | Purpose |
|---|---|
| `make_5fold_dataset.py` | Audits the source images and builds the folds |
| `run_case_5fold_isolated.py` | Runs one case over all folds, with a fresh process per fold (use this for real runs) |
| `train_one_case_5fold.py` | The training entry point the runner calls |
| `analyze_chapter4.py` | Builds the summary tables, paired tests and charts from saved results |
| `generate_learning_curves.py` | Redraws learning curves from saved histories |
| `generate_gradcam.py` | Grad-CAM for one image |
| `generate_gradcam_appendix.py` | Paired Grad-CAM panels for a stable sample of images |
| `run_controlled_development.py` | Validation-only screening runner, which never predicts on test images |

Use `--help` on any script for its full options.

## Run a case

Always use a new output folder. This command reproduces the V3 settings for C4:

```bash
python scripts/run_case_5fold_isolated.py \
  --folds_root data_5_fold --output_dir results/new_run/proposed_artifact_mix \
  --model_type proposed --scenario artifact_mix \
  --recipe v3 --image_size 224 --image_height 448 --batch_size 6
```

Use `--model_type benchmark` or `--scenario clean` for the other cases. Leave out the recipe, height and batch options to get the V2 settings. Analyze all four finished cases with:

```bash
python scripts/analyze_chapter4.py --results_root results/new_run
```

The analyzer's own metadata still uses .05. The manuscript's threshold is .005, so read the p-values against .005.

## Screen a model change

The runner only accepts a new folder under `results/development_controlled/`. Add `--plan-only` to check a command without training.

```bash
python -B scripts/run_controlled_development.py \
  --folds-root data_5_fold --output-dir results/development_controlled/new_screen_fold_1 \
  --scenario artifact_mix --attention cbam --auxiliary-head on --artifact-loss-weight 0.3 \
  --image-size 224 --image-height 448 --recipe v3 --candidate artifact_gated --batch-size 6 \
  --single-fold fold_1 --plan-only
```

See [`scripts/development/README.md`](../scripts/development/README.md) for what the runner guards against.

## Rules

- Never overwrite `results/final_v2/`, `results/final_v3/`, `data/` or `data_5_fold/`.
- Choose models and settings on validation data, never on test results.
- Keep failed runs. The paper discloses them.
- Edit the paper in `paper/chapters/` only. Windows and WSL copies do not sync automatically, so compare them first.

## Check saved results

```bash
sha256sum --quiet -c results/final_v2.sha256
sha256sum --quiet -c results/final_v3.sha256
```

These checks show that the files are unchanged. They do not show that the results are correct.

## Archives

Older material is kept outside the repository in `/home/solyvie/workspace/thesis-projects/repository-archives/`.

| Archive | Contents |
|---|---|
| `AGML-DenseCBAM_cleanup_20261005_002454/` | Old checkpoints and result trees, including everything needed for `results/final_3002_seed42.sha256`, old figures and the earlier source |
| `AGML-DenseCBAM_cleanup_20261101/` | Retired one-off scripts and their tests, the notebook, configs, smoke runs, Zone.Identifier files (`zone_identifier_files.tgz`) and the previous docs (`docs_before/`) |

To restore a file, move it back to the same relative path. `moved_manifest.sha256` lists the moved files with their hashes. The small records in `results/` from older runs do not include checkpoints, so they cannot be resumed.
