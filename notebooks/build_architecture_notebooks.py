"""Builds the two system-architecture notebooks from the real code.

Run from the repository root:  python notebooks/build_architecture_notebooks.py
Notebook 1 shows the current module files (V2 settings). Notebook 2 contains the exact Final V2 trainer
from revision 9249331, split into cells by architecture module.
"""
import ast
import hashlib
import subprocess
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
REFERENCE = NOTEBOOKS / "reference_9249331"
V2_REVISION = "9249331"
V2_FILES = {"trainer": "scripts/train_one_case_5fold.py", "folds": "scripts/make_5fold_dataset.py", "gradcam": "scripts/generate_gradcam.py"}


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def save_reference_files():
    """Writes the exact V2 files, byte for byte, so the notebook can fingerprint them."""
    REFERENCE.mkdir(parents=True, exist_ok=True)
    sources = {}
    for key, path in V2_FILES.items():
        data = subprocess.run(["git", "show", f"{V2_REVISION}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
        target = REFERENCE / Path(path).name
        target.write_bytes(data)
        sources[key] = data.decode("utf-8")
    return sources


def blocks(source):
    """Maps each top-level function, class and constant to its exact source text."""
    tree = ast.parse(source)
    lines = source.splitlines()
    out = {}
    for node in tree.body:
        start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
        text = "\n".join(lines[start - 1:node.end_lineno])
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = text
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = text
    return out


def header(source, stop_marker):
    """The import block at the top of a file, without the shebang line."""
    text = source.split(stop_marker)[0]
    return "\n".join(line for line in text.splitlines() if not line.startswith("#!")).strip()


def import_lines(source):
    """The import statements of a file, without __future__ (the notebook already has it)."""
    tree = ast.parse(source)
    lines = source.splitlines()
    return "\n".join("\n".join(lines[n.lineno - 1:n.end_lineno]) for n in tree.body
                     if isinstance(n, (ast.Import, ast.ImportFrom)) and getattr(n, "module", None) != "__future__")


# ---------------------------------------------------------------- shared text and demo cells

TITLE = """
# AGML-DenseCBAM system architecture, module by module

{intro}

The system has six modules, in the order the code runs them:

![Figure 4.1: system architecture](../docs/figures/architecture.png)

 Each section below explains one module in plain words, shows its code, and then runs a small demonstration on the real data.

| Module | What it does | Main code |
|---|---|---|
| 1. Data Audit | Finds duplicate images and removes images that carry both labels | `scan_source_images`, `prepare_unique_items` |
| 2. Five-Fold Partitioning | Splits the 3,002 kept images into five folds, each with training, validation and test parts | `grouped_stratified_folds`, `create_folds`, `index_split_dataset` |
| 3. Input Preparation | Loads, resizes and scales each image, and adds an artifact in the mixed cases | `apply_artifact`, `TMJSequence` |
| 4. Model Building | Builds one model for the case: the benchmark (C1, C2) or AGML-DenseCBAM (C3, C4) | `build_benchmark_model`, `build_proposed_model` |
| 5. Training and Selection | Trains that model and keeps the epoch with the lowest validation loss | `compile_model`, callbacks, `model.fit` |
| 6. Evaluation and Interpretation | Scores the selected model on the test fold and draws Grad-CAM heatmaps | `collect_predictions`, metrics, `gradcam` |

**The four cases** are two models times two kinds of training images:

| | Trained on clean images | Trained with artifacts added |
|---|---|---|
| Benchmark (reconstructed attention-based DenseNet201) | C1 | C2 |
| AGML-DenseCBAM (proposed) | C3 | C4 |

Every case runs through the same six modules on the same five folds. Only Module 3 (artifacts on or off) and Module 4 (which model) change between cases.
"""

SETTINGS = """
# Settings for this notebook. Only CASE and FOLD need changing to look at another case.
import os
import sys
from pathlib import Path

# The project folder holds scripts/, data/, data_5_fold/, results/final_v2/ and notebooks/. It is found automatically:
# in Colab, the Drive folder below; elsewhere, the repository this notebook sits in. AGML_PROJECT_ROOT overrides both.
COLAB_PROJECT_FOLDER = "/content/drive/MyDrive/AGML-DenseCBAM"
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_ROOT = Path(COLAB_PROJECT_FOLDER)
except ImportError:
    PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROJECT_ROOT = Path(os.environ.get("AGML_PROJECT_ROOT", PROJECT_ROOT)).resolve()
assert (PROJECT_ROOT / "data_5_fold").is_dir(), f"No data_5_fold/ in {PROJECT_ROOT}. Set AGML_PROJECT_ROOT to the project folder."
print("Project folder:", PROJECT_ROOT)
SOURCE_IMAGES = PROJECT_ROOT / "data"                # the original extracted images (3,425 files)
FOLDS_ROOT = PROJECT_ROOT / "data_5_fold"            # the five saved folds used by every case
SAVED_V2 = PROJECT_ROOT / "results" / "final_v2"     # the saved Final V2 models and result files

CASE = "C4"        # C1, C2, C3 or C4
FOLD = "fold_1"    # fold_1 to fold_5

RUN_EVALUATION = True   # Module 6 demo: load the saved Final V2 model and score its test fold (about 2 minutes)
RUN_TRAINING = False    # Module 5: train this case and fold from scratch (about an hour); not needed for the discussion

CASES = {"C1": ("benchmark", "clean"), "C2": ("benchmark", "artifact_mix"), "C3": ("proposed", "clean"), "C4": ("proposed", "artifact_mix")}
CASE_FOLDERS = {"C1": "benchmark_clean", "C2": "benchmark_artifact_mix", "C3": "proposed_clean", "C4": "proposed_artifact_mix"}
"""

CONFIG_DEMO = """
# The Final V2 settings, as recorded in results/final_v2/<case>/run_config.json.
def make_config(case):
    model_type, scenario = CASES[case]
    return RunConfig(
        folds_root=FOLDS_ROOT,
        output_dir=PROJECT_ROOT / "results" / "notebook_runs" / CASE_FOLDERS[case],  # never results/final_v2
        model_type=model_type,
        scenario=scenario,
        image_size=(224, 224),
        batch_size=8,
        epochs=50,
        learning_rate=1e-4,
        l2_strength=1e-2,
        random_state=42,
        freeze_backbone=False,
        tmd_loss_weight=1.0,
        artifact_loss_weight=0.3,
        fold_limit=None,
        single_fold=FOLD,
        verify_integrity=True,
        mixed_precision=True,
        class_weighting=True,
    )

config = make_config(CASE)
model_type, scenario = CASES[CASE]
multi_task = model_type == "proposed"
configure_tensorflow(config)  # mixed precision on the GPU, as in the real runs
print(CASE, "=", model_type, "model,", scenario, "images")
"""

M1_TEXT = """
## Module 1. Data Audit

**What it does.** The source study gave us 3,425 image files labeled Normal or Subluxation. Many files are exact copies of each other. The audit reads every file, computes a fingerprint of its content (SHA-256), and groups identical files together.

- Copies with the same label are kept once.
- Eight images appear under **both** labels. They are removed, because we cannot tell which label is right. No label is ever changed.

**Input:** the folder of source images. **Output:** 3,002 unique images (1,359 Normal, 1,643 Subluxation), plus an audit file listing every duplicate group.

In Figure 4.1 this is Module 1.
"""

M1_DEMO = """
# Run the audit on the real source images. The audit file is written to a temporary folder,
# so the saved folds are not touched.
import tempfile
import pandas as pd

images = scan_source_images(SOURCE_IMAGES)
print(f"Source files: {len(images)}")
audit_folder = Path(tempfile.mkdtemp())
items = prepare_unique_items(images, output_root=audit_folder, input_root=SOURCE_IMAGES, conflict_policy="exclude", group_mapping={})
print(f"Kept: {len(items)} unique images")
print(pd.Series([item.class_name for item in items]).value_counts().to_string())
"""

M2_TEXT = """
## Module 2. Five-Fold Partitioning

**What it does.** The 3,002 images are split into five folds. In each fold:

- about one fifth of the images (600 or 601) are the **test** part,
- about one seventh of the rest (343 or 344) are the **validation** part, used to pick the best epoch in Module 5,
- the remaining 2,058 are the **training** part.

Every image is in the test part of exactly one fold. The split keeps the Normal/Subluxation ratio about the same in every part, and it uses a fixed seed (42), so the same folds come out every time. All four cases use these same saved folds, which is what makes the comparisons fair.

`create_folds` copies the images into `data_5_fold/fold_k/{train,validation,test}/` and writes a `manifest.csv`. `index_split_dataset` reads a fold back for training and checks that no image appears in two parts.

In Figure 4.1 this is Module 2.
"""

M2_DEMO = """
# Recompute the five test folds from the audited images and compare them with the saved folds.
# create_folds is not run here, because it would rewrite data_5_fold.
test_folds = grouped_stratified_folds(items, n_splits=5, random_state=42)
for number, indices in enumerate(test_folds, start=1):
    manifest = pd.read_csv(FOLDS_ROOT / f"fold_{number}" / "manifest.csv")
    saved = set(manifest.loc[manifest.fold_split == "test", "content_sha256"])
    recomputed = {items[i].content_sha256 for i in indices}
    print(f"fold_{number}: {len(recomputed)} test images, identical to the saved fold: {recomputed == saved}")
"""

M2_DEMO_READ = """
# Read one saved fold the way training does, with the leakage checks switched on.
fold_df = index_split_dataset(FOLDS_ROOT / FOLD, verify_integrity=True)
fold_df.groupby(["split", "class_name"]).size().unstack()
"""

M3_TEXT = """
## Module 3. Input Preparation

**What it does.** Turns each image file into the numbers the model reads. Every image is loaded, converted to RGB, resized to 224 × 224 pixels and scaled to the range −1 to 1.

The case decides one thing here: whether artifacts are added.

- **Clean cases (C1, C3):** no artifact is ever added.
- **Mixed cases (C2, C4):** each image gets one of four conditions with equal chance: none, motion blur, Gaussian noise or a metal streak. The program records which one it chose; that record is the answer key for the auxiliary task.

There are three lanes, one per part of the fold:

| Lane | Flip | Artifact | Why |
|---|---|---|---|
| Training | Random left–right flip, half the time | A new random artifact every epoch | More variety for learning |
| Validation | None | Fixed per image | Every epoch, and every model, is judged on the same inputs |
| Test | None | Fixed per image | Every model is tested on exactly the same inputs |

The fixed artifact comes from the image's file name and the seed, so C2 and C4 see identical test images.

In Figure 4.1 this is Module 3.
"""

M3_DEMO_ARTIFACTS = """
# One test image with each of the four generated conditions.
import cv2
import numpy as np
import matplotlib.pyplot as plt

sample_path = fold_df[fold_df.split == "test"].filepath.iloc[0]
image = cv2.resize(cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA)
figure, axes = plt.subplots(1, 4, figsize=(14, 4))
for label, axis in enumerate(axes):
    axis.imshow(apply_artifact(image, label, np.random.default_rng(7)))
    axis.set_title(ARTIFACT_LABELS[label])
    axis.axis("off")
plt.show()
"""

M3_DEMO_LANES = """
# The three lanes for a mixed case. Each panel shows the image the model receives, with its label and artifact.
def show_batch(sequence, title, count=4):
    x, y = sequence[0][:2]
    figure, axes = plt.subplots(1, count, figsize=(14, 4))
    for i, axis in enumerate(axes):
        tmd = DISPLAY_LABELS[int(np.argmax(y["tmd_output"][i]))]
        artifact = ARTIFACT_LABELS[int(np.argmax(y["artifact_output"][i]))]
        axis.imshow(np.clip((x[i] + 1.0) * 127.5, 0, 255).astype(np.uint8))
        axis.set_title(f"{tmd}\\n{artifact}", fontsize=9)
        axis.axis("off")
    figure.suptitle(title)
    plt.show()

lanes = {split: TMJSequence(fold_df[fold_df.split == split], (224, 224), 8, True, "artifact_mix", split == "train", 42)
         for split in ("train", "validation", "test")}
show_batch(lanes["train"], "Training lane, epoch 1")
lanes["train"].on_epoch_end()
show_batch(lanes["train"], "Training lane, epoch 2: new order, new artifacts")
show_batch(lanes["test"], "Test lane: no flip, fixed artifacts")
"""

M4A_TEXT = """
## Module 4. Model Building

**What it does.** Builds the one model the case needs. C1 and C2 build the benchmark; C3 and C4 build AGML-DenseCBAM. A case never builds or combines both. Both models start from the same DenseNet201 backbone with ImageNet weights, and the whole backbone is trained.

**Benchmark (reconstructed from Sancar et al.):**

1. Take the earlier DenseNet feature map `pool3_relu` and apply **self-attention** to it.
2. Shrink it (4 × 4 average pooling) and project it to match the last feature map.
3. Join it with the last DenseNet feature map, `conv5_block32_concat`, and mix them with a 1 × 1 convolution.
4. Classifier head: global average pooling, Dense 1,024, dropout, batch normalization, Dense 128, then **Normal / Subluxation**.

**AGML-DenseCBAM (proposed):** one shared backbone, two heads.

1. **Main head:** apply **CBAM** to the last feature map (channel attention, then spatial attention), then the same style of classifier head, ending in **Normal / Subluxation** (`tmd_output`).
2. **Auxiliary head:** taken from the last feature map **before** CBAM. Average and max pooling are joined, then Dense 256 and dropout, ending in the **four artifact conditions** (`artifact_output`).

The auxiliary head sits before CBAM so that attention cannot hide the artifact evidence from it. Both heads train the shared backbone together; that sharing is the multi-task part.

In Figure 4.1 this is Module 4.
"""

M4A_DEMO = """
# Build both models and list only the layers added on top of DenseNet201.
import tensorflow as tf

def added_layers(model):
    added = [layer for layer in model.layers if layer.name.split("_")[0] in ("benchmark", "cbam", "tmd", "artifact")]
    return pd.DataFrame({"layer": [l.name for l in added], "type": [type(l).__name__ for l in added],
                         "parameters": [l.count_params() for l in added]})

for case_name in ("C2", "C4"):
    case_config = make_config(case_name)
    builder = build_proposed_model if case_config.model_type == "proposed" else build_benchmark_model
    model = builder(case_config)
    print(f"\\n{case_name}: {model.name}, {model.count_params():,} parameters, outputs: {list(model.output_names)}")
    display(added_layers(model))
    del model
    tf.keras.backend.clear_session()
configure_tensorflow(config)
"""

M4B_TEXT = """
## Module 5. Training and Selection

**What it does.** Trains the model built in Module 4 and keeps its best version. Training is a loop: the training images pass through the model again and again while it corrects itself.

![Training loop](../docs/figures/training_loop.png)

In words: the 2,058 training images are cut into 258 batches of 8. For each batch, the model guesses, the guesses are compared with the true labels (the loss), and Adam nudges the weights to lower the loss. After all 258 batches, one epoch is done. The model is then checked on the validation images; if that is its best validation loss so far, it is saved. This repeats until 5 epochs pass without improvement or 50 epochs are done. One `model.fit` call runs this whole loop.

The two lanes have different jobs:

- **Training lane** batches update the weights.
- **Validation lane** batches are only scored after each epoch. That score decides which epoch is kept.

The settings:

- **Optimizer:** Adam, learning rate 0.0001, batch size 8, up to 50 epochs. L2 regularization (0.01) is applied to the first dense layers.
- **Loss:** categorical cross-entropy. The proposed model adds its two losses: main loss + 0.3 × auxiliary loss.
- **Class weights:** Normal and Subluxation are weighted so that both classes count equally in the training loss.
- **Selection:** after each epoch, three callbacks watch the validation loss. The benchmark watches `val_loss`; the proposed model watches its main-head loss, `val_tmd_output_loss`.
  - `ModelCheckpoint` saves the model whenever the watched loss reaches a new low (the `_best.keras` file).
  - `EarlyStopping` stops after 5 epochs without improvement and restores the best weights.
  - `ReduceLROnPlateau` divides the learning rate by 10 after 3 epochs without improvement.

**Output:** the selected model, kept in memory for Module 6, and its saved `_best.keras` checkpoint, used later for Grad-CAM.

In Figure 4.1 this is Module 5.
"""

M4B_DEMO = """
# Class weights for this fold, and the compile settings for this case.
train_df = fold_df[fold_df.split == "train"]
class_weights = balanced_class_weights(train_df.tmd_label)
print("Class weights:", {DISPLAY_LABELS[k]: round(v, 3) for k, v in class_weights.items()})
print("Loss weights: main", config.tmd_loss_weight, "| auxiliary", config.artifact_loss_weight if multi_task else "not used")
print("Watched validation loss:", "val_tmd_output_loss" if multi_task else "val_loss")
"""

M4B_HISTORY = """
# What fitting produced in the real Final V2 run of this case and fold: the kept epoch is the one
# with the lowest watched validation loss.
history = pd.read_csv(SAVED_V2 / CASE_FOLDERS[CASE] / f"{FOLD}_history.csv")
watched = "val_tmd_output_loss" if multi_task else "val_loss"
best = int(history[watched].idxmin())
figure, axis = plt.subplots(figsize=(8, 4))
axis.plot(history.index + 1, history[watched], marker="o", label=watched)
axis.axvline(best + 1, color="black", linestyle="--", label=f"kept epoch {best + 1}")
axis.set_xlabel("epoch"); axis.set_ylabel("validation loss"); axis.legend(); axis.set_title(f"{CASE}, {FOLD}: {len(history)} epochs ran")
plt.show()
"""

M4B_TRAIN = """
# Train this case and fold from scratch (Modules 2 to 6 in one call). Off by default; about an hour on the RTX 4050.
# Results go to results/notebook_runs/, never to results/final_v2.
if RUN_TRAINING:
    run_one_fold(FOLDS_ROOT / FOLD, config)
else:
    print("Training is switched off. Set RUN_TRAINING = True in the settings cell to run it.")
"""

M5_TEXT = """
## Module 6. Evaluation and Interpretation

**What it does.** Runs the selected model once over the test lane and records, for every test image, the true label, the predicted label and the confidence. The proposed model also records its predicted artifact. From these come:

- the confusion matrix (true/false Normal and Subluxation counts),
- accuracy, precision, recall, specificity and F1-score,
- the calibration error (ECE), and the speed in images per second,
- for the proposed model, how well it recognized each artifact.

Grad-CAM is a separate branch: it needs the **saved checkpoint** from Module 5 and one processed image, and draws where the predicted class score is most sensitive.

The demo below loads the saved Final V2 checkpoint for this case and fold, scores its test fold again, and compares the numbers with the saved Final V2 result file.

In Figure 4.1 this is Module 6.
"""

M5_DEMO = """
# Score the saved Final V2 model on its test fold and compare with the saved results.
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

if RUN_EVALUATION:
    checkpoint = SAVED_V2 / CASE_FOLDERS[CASE] / f"{FOLD}_{CASE_FOLDERS[CASE]}_best.keras"
    model = tf.keras.models.load_model(checkpoint, custom_objects={"AttentionBlock": AttentionBlock}, compile=False)
    test_gen = TMJSequence(fold_df[fold_df.split == "test"], (224, 224), 8, multi_task, scenario, False, 42)
    y_true, y_pred, y_conf, artifact_true, artifact_pred, artifact_conf, paths = collect_predictions(model, test_gen, multi_task)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recomputed = {
        "accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred), "specificity": tn / (tn + fp), "f1": f1_score(y_true, y_pred),
        "ece_10_bins": expected_calibration_error(y_true, y_pred, y_conf, n_bins=10),
    }
    saved = pd.read_csv(SAVED_V2 / CASE_FOLDERS[CASE] / f"{FOLD}_results.csv").iloc[0]
    comparison = pd.DataFrame({"recomputed now": recomputed, "saved Final V2": {k: saved[k] for k in recomputed}}).round(4)
    print(f"{CASE}, {FOLD}: {len(y_true)} test images | TN {tn}, FP {fp}, FN {fn}, TP {tp}")
    display(comparison)
else:
    print("Evaluation is switched off. Set RUN_EVALUATION = True in the settings cell.")
"""

M5_GRADCAM = """
# Grad-CAM for the first four test images, from the saved checkpoint.
if RUN_EVALUATION:
    x, _ = test_gen[0][:2]
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    for i, axis in enumerate(axes):
        heatmap, probabilities, class_index = gradcam(model, x[i:i + 1], TARGET_LAYERS[model_type], None)
        # Under mixed precision the V2 gradcam returns float16, which OpenCV cannot resize; later code converts inside gradcam.
        heatmap = heatmap.astype(np.float32)
        display_rgb = np.clip((x[i] + 1.0) * 127.5, 0, 255).astype(np.uint8)
        axis.imshow(overlay_heatmap(display_rgb, heatmap, 0.4))
        axis.set_title(f"true {DISPLAY_LABELS[y_true[i]]}\\npredicted {DISPLAY_LABELS[class_index]} ({probabilities[class_index]:.0%})", fontsize=9)
        axis.axis("off")
    figure.suptitle(f"Grad-CAM at layer {TARGET_LAYERS[model_type]}")
    plt.show()
"""

FLOW_TEXT = """
## One case and fold from start to end

`run_one_fold` is the function that runs one case on one fold. It calls the modules in order:

{flow}

`run_case` repeats `run_one_fold` for each fold and then averages the results. The final runs used `run_case_5fold_isolated.py`, which starts a fresh process for each fold so that no memory is shared between folds.
"""

FLOW_STEPS = [
    ("index_split_dataset(", "Module 2: read the fold and check it"),
    ("balanced_class_weights(", "Module 5: compute class weights"),
    ("TMJSequence(", "Module 3: build the three lanes"),
    ("build_proposed_model(config) if multi_task", "Module 4: build the benchmark or the proposed model"),
    ("compile_model(", "Module 5: set loss, weights and optimizer"),
    ("fit", "Module 5: train with the three callbacks and keep the best epoch"),
    ("collect_predictions(", "Module 6: predict the test lane"),
    ("result_df = pd.DataFrame(", "Module 6: compute the metrics"),
    ("to_csv(", "Module 6: save results, predictions and history"),
]


def flow_table(source):
    rows = ["| Line in `run_one_fold` | Module and job |", "|---|---|"]
    lines = source.splitlines()
    for marker, job in FLOW_STEPS:
        found = next((line.strip() for line in lines if (marker in line if marker != "fit" else ("model.fit(" in line or "fit_model(" in line))), None)
        if found:
            rows.append(f"| `{found}` | {job} |")
    return "\n".join(rows)


def module_cells(m1_code, m2_code, m3_code, m4a_code, m4b_code, m5_code):
    return [
        md(M1_TEXT), *m1_code, code(M1_DEMO),
        md(M2_TEXT), *m2_code, code(M2_DEMO), code(M2_DEMO_READ),
        md(M3_TEXT), *m3_code, code(M3_DEMO_ARTIFACTS), code(M3_DEMO_LANES),
        md(M4A_TEXT), *m4a_code, code(M4A_DEMO),
        md(M4B_TEXT), *m4b_code, code(M4B_DEMO), code(M4B_HISTORY), code(M4B_TRAIN),
        md(M5_TEXT), *m5_code, code(M5_DEMO), code(M5_GRADCAM),
    ]


def write(notebook_cells, name):
    notebook = nbf.v4.new_notebook()
    notebook.cells = notebook_cells
    notebook.metadata["kernelspec"] = {"display_name": "Python 3 (thesis-env)", "language": "python", "name": "python3"}
    nbf.write(notebook, NOTEBOOKS / name)
    print("wrote", name, len(notebook_cells), "cells")


# ---------------------------------------------------------------- notebook 1: current module files

def build_modules_notebook():
    show = lambda *names: code("\n".join(f"show({n})" for n in names))
    pipeline = (ROOT / "scripts/pipeline/experiment.py").read_text(encoding="utf-8")
    cells = [
        md(TITLE.format(intro="This notebook follows the **current code files**, where each module of the system lives in its own file. It uses the Final V2 settings. The Final V2 results themselves were produced by an earlier single-file version of the trainer with the same design; notebook 2 contains that exact file.")),
        code(SETTINGS),
        md("## Code files\n\nThe code is imported from the repository. `show(...)` prints a function's code, so each section shows exactly what runs.\n\n| Module | File |\n|---|---|\n| 1, 2 | `scripts/make_5fold_dataset.py` (fold creation), `scripts/pipeline/data.py` (reading folds) |\n| 3 | `scripts/pipeline/artifacts.py`, `scripts/pipeline/data.py` |\n| 4, 5 | `scripts/pipeline/models.py` |\n| 6 | `scripts/pipeline/evaluation.py`, `scripts/generate_gradcam.py` |\n| All, in order | `scripts/pipeline/experiment.py` |"),
        code("""
import inspect
import sys
from IPython.display import Markdown, display

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from make_5fold_dataset import (scan_source_images, prepare_unique_items, write_duplicate_audit, grouped_stratified_folds,
                                grouped_train_val_split, assert_no_group_overlap, create_folds)
from pipeline.config import RunConfig, configure_tensorflow, ARTIFACT_LABELS, DISPLAY_LABELS
from pipeline.data import index_split_dataset, validate_split_integrity, validate_manifest_integrity, TMJSequence, balanced_class_weights
from pipeline.artifacts import add_motion_blur, add_gaussian_noise, add_metal_streak, apply_artifact, densenet_preprocess
from pipeline.models import AttentionBlock, make_backbone, build_benchmark_model, build_proposed_model, compile_model, _training_callbacks, fit_model
from pipeline.evaluation import collect_predictions, expected_calibration_error
from pipeline.experiment import run_one_fold, run_case
from generate_gradcam import gradcam, overlay_heatmap, TARGET_LAYERS


def show(obj):
    \"\"\"Prints the code of a function or class as a code block that follows the editor's light or dark theme.\"\"\"
    display(Markdown(f"```python\\n{inspect.getsource(obj)}\\n```"))
"""),
        code(CONFIG_DEMO),
    ]
    cells += module_cells(
        [show("scan_source_images", "prepare_unique_items")],
        [show("grouped_stratified_folds", "grouped_train_val_split", "create_folds", "index_split_dataset")],
        [show("apply_artifact", "add_metal_streak", "densenet_preprocess", "TMJSequence")],
        [show("AttentionBlock", "build_benchmark_model", "build_proposed_model")],
        [show("compile_model", "_training_callbacks", "fit_model", "balanced_class_weights")],
        [show("collect_predictions", "expected_calibration_error", "gradcam")],
    )
    cells += [md(FLOW_TEXT.format(flow=flow_table(blocks(pipeline)["run_one_fold"]))), show("run_one_fold", "run_case")]
    write(cells, "01_System_Architecture_Modules.ipynb")


# ---------------------------------------------------------------- notebook 2: exact Final V2 trainer

def build_v2_notebook(sources):
    trainer, folds, cam = (blocks(sources[k]) for k in ("trainer", "folds", "gradcam"))
    take = lambda table, *names: code("\n\n\n".join(table[n] for n in names))
    trainer_path = REFERENCE / "train_one_case_5fold.py"
    digest = hashlib.sha256(trainer_path.read_bytes()).hexdigest()
    imports = (header(sources["trainer"], "TMD_LABELS =") + "\n\n# From make_5fold_dataset.py\n" + import_lines(sources["folds"])
               + "\n\n# From generate_gradcam.py\n" + "\n".join(l for l in import_lines(sources["gradcam"]).splitlines() if "train_one_case_5fold" not in l))
    cells = [
        md(TITLE.format(intro=f"This notebook contains the **exact code that produced the Final V2 results**: `scripts/train_one_case_5fold.py`, `scripts/make_5fold_dataset.py` and `scripts/generate_gradcam.py` at revision `{V2_REVISION}`. The code cells below are copied from those files without changes, only reordered by module. Byte-for-byte copies are in `notebooks/reference_{V2_REVISION}/`.")),
        code(SETTINGS),
        md("## Imports and shared settings\n\nThe import lines of the three files, then the constants and the run settings (`RunConfig`)."),
        code(imports),
        take(trainer, "TMD_LABELS", "DISPLAY_LABELS", "ARTIFACT_LABELS", "ARTIFACT_PROTOCOL", "IMAGE_EXTENSIONS"),
        take(folds, "IMAGE_EXTS", "CLASS_NAMES", "SOURCE_SPLITS"),
        take(cam, "TARGET_LAYERS"),
        take(trainer, "RunConfig", "seed_everything", "configure_tensorflow", "sha256_file"),
        md(f"**Fingerprint check.** The Final V2 result files record the SHA-256 of the trainer that produced them. The cell below compares it with the copy in `reference_{V2_REVISION}/`. The training code reads `__file__` for this fingerprint, so it is set to that copy."),
        code(f"""
import json
REFERENCE_DIR = PROJECT_ROOT / "notebooks" / "reference_{V2_REVISION}"
__file__ = str(REFERENCE_DIR / "train_one_case_5fold.py")
recorded = json.loads((SAVED_V2 / CASE_FOLDERS[CASE] / "run_config.json").read_text())["training_script_sha256"]
print("Recorded in Final V2:", recorded)
print("This copy:           ", sha256_file(Path(__file__)))
print("Same code:", recorded == sha256_file(Path(__file__)))
"""),
        code(CONFIG_DEMO),
    ]
    cells += module_cells(
        [take(folds, "SourceImage", "Item", "normalized_path", "scan_source_images"), take(folds, "load_group_mapping", "resolve_group_id"),
         take(folds, "write_duplicate_audit", "prepare_unique_items")],
        [take(folds, "safe_unique_name", "reset_fold_dirs", "ensure_fold_dirs"), take(folds, "grouped_stratified_folds"),
         take(folds, "grouped_train_val_split", "assert_no_group_overlap"), take(folds, "create_folds"),
         take(trainer, "validate_split_integrity", "validate_manifest_integrity", "index_split_dataset")],
        [take(trainer, "ensure_uint8", "add_motion_blur", "add_gaussian_noise"), take(trainer, "add_metal_streak_v1"),
         take(trainer, "add_metal_streak", "apply_artifact", "densenet_preprocess"), take(trainer, "TMJSequence")],
        [take(trainer, "AttentionBlock"), take(trainer, "make_backbone", "build_benchmark_model"), take(trainer, "build_proposed_model")],
        [take(trainer, "compile_model", "balanced_class_weights"),
         md("The three callbacks and `model.fit` are inside `run_one_fold` (shown in full at the end):\n\n```python\n" + _excerpt(trainer["run_one_fold"], "checkpoint_path = ", "    )\n\n    # Warm up") + "\n```")],
        [take(trainer, "unpack_batch", "expected_calibration_error", "collect_predictions"), take(cam, "gradcam", "overlay_heatmap")],
    )
    cells += [md(FLOW_TEXT.format(flow=flow_table(trainer["run_one_fold"]))), take(trainer, "run_one_fold"), take(trainer, "run_case"),
              md("`parse_args` turns the command line into a `RunConfig`; its defaults are the Final V2 settings used in `make_config` above."),
              take(trainer, "parse_args")]
    write(cells, "02_Final_V2_Trainer_By_Module.ipynb")
    print("V2 trainer sha256:", digest)


def _excerpt(text, start_marker, end_marker):
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return "\n".join(line[4:] if line.startswith("    ") else line for line in text[start - 4:end + 6].splitlines()).strip()


if __name__ == "__main__":
    sources = save_reference_files()
    build_modules_notebook()
    build_v2_notebook(sources)
