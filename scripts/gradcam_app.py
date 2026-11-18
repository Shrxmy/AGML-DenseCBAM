#!/usr/bin/env python
"""Desktop demo: pick a saved model, open a TMJ image, see its prediction and Grad-CAM.

Research prototype only. Uses the saved Final V2 and V3 checkpoints; trains nothing and writes nothing.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Dict, NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VERSIONS = {"V3 follow-up": ("final_v3", "v3", (448, 224)), "Final V2 (primary)": ("final_v2", "v2", (224, 224))}
CASES = {
    "C1 Benchmark, clean": ("benchmark", "clean"),
    "C2 Benchmark, artifact mix": ("benchmark", "artifact_mix"),
    "C3 Proposed, clean": ("proposed", "clean"),
    "C4 Proposed, artifact mix": ("proposed", "artifact_mix"),
}
AUTO_FOLD = "Auto"
FOLDS = [AUTO_FOLD, "fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
DISPLAY_HEIGHT, DISPLAY_WIDTH = 440, 220
DISCLAIMER = (
    "Research prototype, not for diagnosis. Grad-CAM is qualitative and has not been validated by experts. "
    "Scores reflect agreement with inherited image labels."
)


class Record(NamedTuple):
    label: str
    splits: Dict[str, str]  # fold name -> train, validation or test


def checkpoint_path(root: Path, version: str, case: str, fold: str) -> Path:
    model_type, scenario = CASES[case]
    return root / "results" / VERSIONS[version][0] / f"{model_type}_{scenario}" / f"{fold}_{model_type}_{scenario}_best.keras"


def load_records(folds_root: Path) -> Dict[str, Record]:
    records: Dict[str, Record] = {}
    for manifest in sorted(folds_root.glob("fold_*/manifest.csv")):
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                record = records.setdefault(row["content_sha256"], Record(row["class_name"], {}))
                record.splits[manifest.parent.name] = row["fold_split"]
    return records


def choose_fold(requested: str, record: Record | None) -> str:
    if requested != AUTO_FOLD:
        return requested
    if record is None:
        return "fold_1"
    return next(fold for fold, split in sorted(record.splits.items()) if split == "test")


def data_note(fold: str, record: Record | None) -> str:
    if record is None:
        return f"Image is not in the study folds. Using the {fold} model."
    role = record.splits.get(fold)
    label = f"Recorded label: {record.label.title()}."
    if role == "test":
        return f"{label} Held out from the {fold} model, so this is a fair test."
    if role == "validation":
        return f"{label} Warning: the {fold} model used this image to choose its best epoch."
    return f"{label} Warning: the {fold} model trained on this image, so its score is not a fair test."


def analyze(model, model_type: str, recipe: str, size: tuple[int, int], path: Path) -> dict:
    """Score the image as it is, with no generated artifact. V3 averages with the mirror image, as its results do."""
    import pandas as pd

    from scripts.generate_gradcam import TARGET_LAYERS, gradcam, overlay_heatmap
    from scripts.train_one_case_5fold import ARTIFACT_LABELS, DISPLAY_LABELS, TMJSequence

    frame = pd.DataFrame([{"filepath": str(path), "tmd_label": 0, "class_name": "unknown", "split": "test"}])
    x = TMJSequence(frame, size, 1, model_type == "proposed", "clean", False, 42, recipe=recipe)[0][0]
    layer = TARGET_LAYERS[model_type]
    heatmap, probabilities, target = gradcam(model, x, layer, None)
    views = [x]
    if recipe == "v3":
        views.append(x[:, :, ::-1])
        probabilities = (probabilities + gradcam(model, views[1], layer, None)[1]) / 2
        if int(probabilities.argmax()) != target:
            target = int(probabilities.argmax())
            heatmap = gradcam(model, x, layer, target)[0]

    artifact = None
    if model_type == "proposed":
        scores = np.mean([np.asarray(model(view, training=False)["artifact_output"])[0] for view in views], axis=0)
        artifact = (ARTIFACT_LABELS[int(scores.argmax())].replace("_", " "), float(scores.max()))

    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    shown = cv2.resize(image, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
    return {
        "shown": shown,
        "overlay": overlay_heatmap(shown, heatmap, 0.4),
        "prediction": DISPLAY_LABELS[target],
        "confidence": float(probabilities[target]),
        "artifact": artifact,
        "layer": layer,
    }


class Engine:
    """Keeps TensorFlow and one loaded model on a single background thread."""

    def __init__(self) -> None:
        self.model = None
        self.key = None
        self.records = None

    def run(self, path: Path, version: str, case: str, requested_fold: str) -> dict:
        if cv2.imread(str(path)) is None:
            raise ValueError(f"Could not read this file as an image: {path.name}")
        if self.records is None:
            self.records = load_records(ROOT / "data_5_fold")
        record = self.records.get(hashlib.sha256(path.read_bytes()).hexdigest())
        fold = choose_fold(requested_fold, record)
        self._load(version, case, fold)
        _, recipe, size = VERSIONS[version]
        result = analyze(self.model, CASES[case][0], recipe, size, path)
        result["note"] = data_note(fold, record)
        return result

    def _load(self, version: str, case: str, fold: str) -> None:
        key = (version, case, fold)
        if key == self.key:
            return
        import tensorflow as tf

        from scripts.generate_gradcam import load_checkpoint

        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:  # Already set once the GPU is initialized.
                pass
        self.model, self.key = None, None
        tf.keras.backend.clear_session()
        gc.collect()
        checkpoint = checkpoint_path(ROOT, version, case, fold)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Saved model not found: {checkpoint.relative_to(ROOT)}")
        self.model = load_checkpoint(checkpoint)[0]
        self.key = key


class App:
    def __init__(self, root: tk.Tk, image: Path | None) -> None:
        self.root = root
        self.engine = Engine()
        self.jobs: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.path = image
        self.photos: list[ImageTk.PhotoImage] = []
        threading.Thread(target=self._worker, daemon=True).start()

        root.title("AGML-DenseCBAM Grad-CAM")
        frame = ttk.Frame(root, padding=14)
        frame.grid()
        ttk.Label(frame, text="AGML-DenseCBAM: TMJ Normal vs Subluxation", font=("TkDefaultFont", 14, "bold")).grid(columnspan=4)

        controls = ttk.Frame(frame)
        controls.grid(columnspan=4, pady=10)
        self.version = self._choice(controls, "Results", list(VERSIONS), 0)
        self.case = self._choice(controls, "Model", list(CASES), 3)
        self.fold = self._choice(controls, "Fold", FOLDS, 0)
        self.button = ttk.Button(controls, text="Open image", command=self.open_image)
        self.button.pack(side="left", padx=(12, 0))

        blank = ImageTk.PhotoImage(Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), "#222222"))
        self.photos.append(blank)
        ttk.Label(frame, text="Image").grid(row=2, column=0)
        ttk.Label(frame, text="Grad-CAM").grid(row=2, column=1)
        self.left = ttk.Label(frame, image=blank)
        self.left.grid(row=3, column=0, padx=6)
        self.right = ttk.Label(frame, image=blank)
        self.right.grid(row=3, column=1, padx=6)

        ramp = np.linspace(255, 0, DISPLAY_HEIGHT - 40).astype(np.uint8).reshape(-1, 1)
        colors = cv2.cvtColor(cv2.applyColorMap(ramp, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        self.ramp = ImageTk.PhotoImage(Image.fromarray(colors).resize((18, DISPLAY_HEIGHT - 40)))
        scale = ttk.Frame(frame)
        scale.grid(row=3, column=2, padx=(4, 0))
        ttk.Label(scale, text="High").pack()
        ttk.Label(scale, image=self.ramp).pack()
        ttk.Label(scale, text="Low").pack()

        self.prediction = ttk.Label(frame, text="Open an image to begin.", font=("TkDefaultFont", 12, "bold"))
        self.prediction.grid(columnspan=4, pady=(10, 0))
        self.details = ttk.Label(frame, text="", justify="center", wraplength=540)
        self.details.grid(columnspan=4, pady=4)
        ttk.Label(frame, text=DISCLAIMER, justify="center", wraplength=540, foreground="#555555").grid(columnspan=4, pady=(6, 0))

        for widget in (self.version, self.case, self.fold):
            widget.bind("<<ComboboxSelected>>", lambda _event: self.start())
        root.after(100, self._poll)
        if image is not None:
            self.start()

    def _choice(self, parent: ttk.Frame, label: str, values: list[str], index: int) -> ttk.Combobox:
        ttk.Label(parent, text=label).pack(side="left", padx=(8, 3))
        box = ttk.Combobox(parent, values=values, state="readonly", width=max(map(len, values)) + 1)
        box.current(index)
        box.pack(side="left")
        return box

    def open_image(self) -> None:
        chosen = filedialog.askopenfilename(
            initialdir=next((p for p in (ROOT / "output/defense_demo", ROOT / "data_5_fold") if p.is_dir()), ROOT),
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*")],
        )
        if chosen:
            self.path = Path(chosen)
            self.start()

    def start(self) -> None:
        if self.path is None:
            return
        self._set_busy(True)
        self.prediction.config(text="Working...")
        self.details.config(text="Loading a model can take a minute the first time.")
        self.jobs.put((self.path, self.version.get(), self.case.get(), self.fold.get()))

    def _set_busy(self, busy: bool) -> None:
        self.button.config(state="disabled" if busy else "normal")
        for box in (self.version, self.case, self.fold):
            box.config(state="disabled" if busy else "readonly")

    def _worker(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                self.results.put(self.engine.run(*job))
            except Exception as error:  # Show any failure in the window instead of losing it on a thread.
                self.results.put(error)

    def _poll(self) -> None:
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            self._set_busy(False)
            self._show(result)
        self.root.after(100, self._poll)

    def _show(self, result) -> None:
        if isinstance(result, Exception):
            self.prediction.config(text="Could not analyze this image.")
            self.details.config(text=str(result))
            return
        self.photos = [ImageTk.PhotoImage(Image.fromarray(result[key])) for key in ("shown", "overlay")]
        self.left.config(image=self.photos[0])
        self.right.config(image=self.photos[1])
        self.prediction.config(text=f"{result['prediction']}: {result['confidence'] * 100:.1f}% confidence")
        lines = [result["note"], f"Grad-CAM layer: {result['layer']}."]
        if result["artifact"]:
            lines.insert(1, f"Artifact head: {result['artifact'][0]} ({result['artifact'][1] * 100:.1f}%).")
        self.details.config(text="\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", help="Optional image to open at startup.")
    args = parser.parse_args()
    root = tk.Tk()
    App(root, args.image)
    root.mainloop()


if __name__ == "__main__":
    main()
