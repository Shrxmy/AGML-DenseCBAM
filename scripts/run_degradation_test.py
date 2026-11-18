"""Scores one saved checkpoint on its held-out fold under fixed test-time conditions. Never trains.

Reads the saved checkpoints and fold images only. Writes one CSV per case and fold under --out.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_gradcam import load_checkpoint  # noqa: E402
from pipeline.artifacts import add_gaussian_noise, add_motion_blur, densenet_preprocess, densenet_torch_preprocess, ensure_uint8  # noqa: E402
from pipeline.data import index_split_dataset  # noqa: E402

# Levels 1 and 2 sit at the ends of the trained range. Level 3 is beyond anything used in training.
CONDITIONS = {
    "clean": ("none", 0),
    "blur_1": ("blur", 5), "blur_2": ("blur", 9), "blur_3": ("blur", 15),
    "noise_1": ("noise", 8.0), "noise_2": ("noise", 12.0), "noise_3": ("noise", 24.0),
    "streak_1": ("streak", (1, 80.0)), "streak_2": ("streak", (2, 110.0)), "streak_3": ("streak", (2, 160.0)),
}
VERSIONS = {
    "v2": {"size": (224, 224), "preprocess": densenet_preprocess, "mirror": False, "batch": 32},
    "v3": {"size": (448, 224), "preprocess": densenet_torch_preprocess, "mirror": True, "batch": 12},
}


def family_rng(family, filename, seed=42):
    # The same file gets the same noise pattern and streak path at every level and in every model.
    digest = hashlib.sha256(f"degradation:{seed}:{family}:{filename}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def add_streak(image, rng, count, strength):
    # Same geometry as pipeline.artifacts.add_metal_streak, with a fixed count and intensity.
    h, w = image.shape[:2]
    combined = np.zeros((h, w), dtype=np.float32)
    for _ in range(2):
        mask = np.zeros((h, w), dtype=np.float32)
        y_start = int(rng.integers(0, h))
        y_end = int(np.clip(y_start + rng.integers(-h // 3, h // 3 + 1), 0, h - 1))
        thickness = int(rng.integers(2, 5))
        sigma = float(rng.uniform(1.5, 2.5))
        if _ >= count:
            continue
        cv2.line(mask, (0, y_start), (w - 1, y_end), 1.0, thickness)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
        combined = np.maximum(combined, mask / float(mask.max()) * strength)
    return ensure_uint8(image.astype(np.float32) + combined[..., None])


def degrade(image, condition, filename):
    family, level = CONDITIONS[condition]
    if family == "none":
        return image
    rng = family_rng(family, filename)
    if family == "blur":
        return add_motion_blur(image, kernel_size=level)
    if family == "noise":
        return add_gaussian_noise(image, rng, sigma=level)
    return add_streak(image, rng, *level)


def probabilities(model, batch):
    output = model.predict_on_batch(batch)
    if isinstance(output, dict):
        return np.asarray(output["tmd_output"])
    if isinstance(output, (list, tuple)):
        return np.asarray(output[model.output_names.index("tmd_output")])
    return np.asarray(output)


def self_check():
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    image[:, 32:] = 160
    for name in CONDITIONS:
        first = degrade(image, name, "a.jpg")
        assert np.array_equal(first, degrade(image, name, "a.jpg")), name
        assert first.shape == image.shape and first.dtype == np.uint8, name
        assert np.array_equal(first, image) == (name == "clean"), name
    # Higher levels must change the image more.
    for family in ("blur", "noise", "streak"):
        change = [np.abs(degrade(image, f"{family}_{k}", "a.jpg").astype(int) - image).sum() for k in (1, 2, 3)]
        assert change[0] < change[1] < change[2], (family, change)
    print("self-check passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=VERSIONS)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--folds-root", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    target = args.out / args.version / f"{args.case}_fold_{args.fold}.csv"
    if target.exists():
        print("already done", target)
        return
    settings = VERSIONS[args.version]
    checkpoint = args.results_root / args.case / f"fold_{args.fold}_{args.case}_best.keras"
    model, policy = load_checkpoint(checkpoint)
    frame = index_split_dataset(args.folds_root / f"fold_{args.fold}", verify_integrity=False)
    frame = frame[frame["split"] == "test"].reset_index(drop=True)
    height, width = settings["size"]
    images = []
    for path in frame["filepath"]:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        images.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    rows = []
    for condition in args.conditions.split(","):
        for start in range(0, len(images), settings["batch"]):
            part = frame.iloc[start:start + settings["batch"]]
            batch = np.stack([settings["preprocess"](degrade(images[i], condition, Path(p).name)) for i, p in zip(part.index, part["filepath"])])
            probs = probabilities(model, batch)
            if settings["mirror"]:
                probs = (probs + probabilities(model, batch[:, :, ::-1])) / 2.0
            rows.extend((condition, Path(p).name, int(t), float(q)) for p, t, q in zip(part["filepath"], part["tmd_label"], probs[:, 1]))
        print(args.case, args.fold, condition, "done", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    pd.DataFrame(rows, columns=["condition", "filename", "y_true", "p_subluxation"]).to_csv(temporary, index=False)
    temporary.rename(target)
    print("saved", target, "policy", policy, flush=True)


if __name__ == "__main__":
    main()
