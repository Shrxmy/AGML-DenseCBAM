#!/usr/bin/env bash
set -eo pipefail

source /home/solyvie/environments/miniforge3/etc/profile.d/conda.sh
conda activate thesis-env
cd /home/solyvie/workspace/thesis-projects/working-run/AGML-DenseCBAM

mkdir -p chapter4_results/development_v3
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
  --output_dir chapter4_results/development_v3/proposed_artifact_mix \
  2>&1 | tee chapter4_results/development_v3_smoke.log
