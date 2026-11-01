"""Pure/static controlled-development checks: no TF, images, weights or inference.

Actual TF model structure checks are explicitly skipped pending separate parent
approval. This file alone can run on standard-library Python in WSL.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts.development.config import (
    EVALUATION_BOUNDARY, DevelopmentConfig, discover_folds, guard_output,
    make_plan, parse_args, reserve_output, source_provenance, validate_config,
    validate_index_rows, validate_manifest_split_counts,
)
from scripts.development.runner import (
    collect_validation_predictions, common_protocol_payload, execute, make_sequences,
    normalize_primary_monitor, select_checkpoint,
)
from scripts.pipeline.provenance import training_source_sha256

ROOT = Path(__file__).resolve().parents[1]


class FakeColumn(list):
    def tolist(self):
        return list(self)


class FakeFrame:
    def __init__(self, split, count=2):
        self.columns = {"split": FakeColumn([split] * count),
                        "filepath": FakeColumn([f"/{split}/{index}.png" for index in range(count)])}

    def __len__(self):
        return len(self.columns["split"])

    def __getitem__(self, name):
        return self.columns[name]


class FakeIndexed:
    """Minimal dataframe stand-in; no pandas, arrays, or image files."""
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def to_dict(self, orient):
        assert orient == "records"
        return list(self.rows)

    def reset_index(self, drop):
        assert drop
        return self

    def __getitem__(self, key):
        if isinstance(key, str):
            class MaskColumn(FakeColumn):
                def __eq__(self, value):
                    return [item == value for item in self]
            return MaskColumn(row[key] for row in self.rows)
        return FakeIndexed([row for row, selected in zip(self.rows, key) if selected])


class ControlledDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folds_root = self.root / "input_folds"
        self.config = DevelopmentConfig(self.folds_root, self.root / "results/development_controlled/smoke")
        self.fixture_fold("fold_1")

    def fixture_fold(self, name):
        fold = self.folds_root / name
        for split in ("train", "validation", "test"):
            for label in ("normal", "subluxation"):
                (fold / split / label).mkdir(parents=True)
        # Intentionally invalid contents: plan must not read/parse this.
        (fold / "manifest.csv").write_text("do not parse or hash dataset manifests in plan-only\n")
        return fold

    def test_cli_historical_proposed_defaults(self):
        config, plan_only = parse_args(["--output-dir", str(self.config.output_dir)])
        self.assertFalse(plan_only)
        self.assertEqual((config.attention, config.auxiliary_head, config.scenario), ("cbam", "on", "artifact_mix"))
        self.assertEqual((config.image_size, config.batch_size, config.epochs), (224, 8, 50))
        self.assertEqual((config.learning_rate, config.l2_strength, config.artifact_loss_weight, config.seed),
                         (1e-4, 0.01, 0.3, 42))
        payload = config.payload()
        self.assertEqual(payload["tmd_loss_weight"], 1.0)
        self.assertTrue(payload["class_weighting"])
        self.assertTrue(payload["verify_integrity"])
        self.assertFalse(payload["freeze_backbone"])

    def test_cli_all_flags_and_four_variants(self):
        for attention in ("cbam", "none"):
            for auxiliary in ("on", "off"):
                config, plan_only = parse_args([
                    "--folds-root", str(self.folds_root), "--output-dir", str(self.config.output_dir),
                    "--attention", attention, "--auxiliary-head", auxiliary, "--scenario", "clean",
                    "--artifact-loss-weight", "0.7", "--image-size", "33", "--batch-size", "2",
                    "--epochs", "3", "--learning-rate", "0.00003", "--l2-strength", "0", "--seed", "7",
                    "--single-fold", "fold_1", "--plan-only",
                ])
                self.assertTrue(plan_only)
                self.assertEqual((config.attention, config.auxiliary_head), (attention, auxiliary))
                self.assertEqual(config.multi_task, auxiliary == "on")
                self.assertEqual(config.monitor, "val_primary_ce")
                self.assertEqual(config.image_size, 33)  # Proposed-only: no pool alignment constraint.
                self.assertEqual(config.payload()["effective_artifact_loss_weight"], 0.7 if auxiliary == "on" else None)
                self.assertIn("excluding regularization", config.payload()["checkpoint_objective_note"])

    def test_shared_primary_monitor_excludes_total_and_auxiliary_loss(self):
        for attention in ("cbam", "none"):
            for auxiliary in ("on", "off"):
                config = dataclasses.replace(self.config, attention=attention, auxiliary_head=auxiliary)
                key = "val_tmd_output_primary_ce" if auxiliary == "on" else "val_primary_ce"
                logs = {key: 0.2, "val_loss": 9.0, "val_tmd_output_loss": 4.0}
                self.assertEqual(normalize_primary_monitor(logs), 0.2)
                self.assertEqual(logs[config.monitor], 0.2)
        for logs in ({"val_loss": 0.1}, {"val_primary_ce": float("nan")},
                     {"val_primary_ce": 0.1, "val_tmd_output_primary_ce": 0.2}):
            with self.assertRaises(RuntimeError):
                normalize_primary_monitor(logs)

    def test_common_protocol_excludes_only_cell_toggles_and_output(self):
        original = common_protocol_payload(self.config)
        for attention in ("cbam", "none"):
            for auxiliary in ("on", "off"):
                other = dataclasses.replace(self.config, attention=attention, auxiliary_head=auxiliary,
                                            output_dir=self.config.output_dir.parent / f"{attention}_{auxiliary}")
                self.assertEqual(original, common_protocol_payload(other))
        self.assertNotEqual(original, common_protocol_payload(dataclasses.replace(self.config, seed=7)))

    def test_plan_guards_each_emitted_sibling_destination(self):
        occupied = self.config.output_dir.parent / "smoke_cbam_aux_off"
        occupied.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            make_plan(self.config, self.root)

    def test_config_rejects_invalid_values(self):
        for name, values in {
            "scenario": ["test"], "attention": ["self"], "auxiliary_head": ["disabled"],
            "image_size": [31, 32.5], "batch_size": [0, True], "epochs": [0, 1.5], "seed": [-1, 2**32, 4.2],
            "learning_rate": [0, -1, float("nan"), float("inf")],
            "l2_strength": [-1, float("nan"), float("inf")],
            "artifact_loss_weight": [-1, float("nan"), float("inf")],
            "folds": [(), ("fold_1", "fold_1"), ("../fold_1",), ("fold_0",)],
            "single_fold": ["", "../test", "fold_01"],
        }.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_config(dataclasses.replace(self.config, **{name: value}))
        with self.assertRaises(ValueError):
            validate_config(dataclasses.replace(self.config, single_fold="fold_1", folds=("fold_1",)))

    def test_fold_selection_numeric_order_and_missing_metadata(self):
        self.fixture_fold("fold_10")
        self.fixture_fold("fold_2")
        self.assertEqual([p.name for p in discover_folds(self.config)], ["fold_1", "fold_2", "fold_10"])
        selected = dataclasses.replace(self.config, folds=("fold_10", "fold_2"))
        self.assertEqual([p.name for p in discover_folds(selected)], ["fold_2", "fold_10"])
        self.assertEqual([p.name for p in discover_folds(dataclasses.replace(self.config, single_fold="fold_2"))], ["fold_2"])
        with self.assertRaises(ValueError):
            discover_folds(dataclasses.replace(self.config, single_fold="fold_3"))
        (self.folds_root / "fold_1/manifest.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            discover_folds(self.config)

    def test_guard_rejects_protected_paths_and_allowed_root(self):
        for path in ("results", "results/development_controlled", "results/final_v2/new",
                     "results/historical/new", "chapter4_results/new", "data/new", "base/new", "paper/new"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                guard_output(dataclasses.replace(self.config, output_dir=self.root / path), self.root)
        self.assertEqual(guard_output(self.config, self.root), self.config.output_dir)

    def test_input_overlap_both_directions_is_forbidden(self):
        for folds in (self.config.output_dir, self.config.output_dir / "folds",
                      self.config.output_dir.parent, self.root):
            with self.subTest(folds=folds), self.assertRaises(ValueError):
                guard_output(dataclasses.replace(self.config, folds_root=folds), self.root)

    def test_existing_empty_or_nonempty_output_is_never_reused(self):
        self.config.output_dir.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            reserve_output(self.config, self.root)
        evidence = self.config.output_dir / "evidence.txt"
        evidence.write_text("unchanged")
        with self.assertRaises(FileExistsError):
            reserve_output(self.config, self.root)
        self.assertEqual(evidence.read_text(), "unchanged")
        self.assertEqual(list(self.config.output_dir.iterdir()), [evidence])

    def test_atomic_reservation_collision(self):
        output = reserve_output(self.config, self.root)
        self.assertTrue((output / "RUN_RESERVED").is_file())
        with self.assertRaises(FileExistsError):
            reserve_output(self.config, self.root)
        self.assertEqual(len(list(output.iterdir())), 1)

    def test_atomic_leaf_collision_between_guard_and_mkdir(self):
        original = Path.mkdir
        def race(path, *args, **kwargs):
            if path == self.config.output_dir:
                original(path)  # Another runner wins reservation.
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "mkdir", race), self.assertRaises(FileExistsError):
            reserve_output(self.config, self.root)
        self.assertFalse((self.config.output_dir / "RUN_RESERVED").exists())

    def test_symlink_output_ancestor_is_forbidden(self):
        allowed = self.root / "results/development_controlled"
        allowed.mkdir(parents=True)
        target = self.root / "other"
        target.mkdir()
        link = allowed / "link"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Platform cannot create symlinks")
        with self.assertRaises(ValueError):
            guard_output(dataclasses.replace(self.config, output_dir=link / "new"), self.root)

    def test_plan_is_deterministic_read_only_and_source_covered(self):
        # Sources are real code, dataset fixture contains no images or valid CSV.
        with mock.patch("scripts.development.config.source_provenance", return_value=source_provenance(ROOT)):
            first = make_plan(self.config, self.root)
            second = make_plan(self.config, self.root)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertFalse((self.root / "results").exists())
        self.assertFalse(first["evaluation_boundary"]["generalization_estimate"])
        self.assertEqual(first["evaluation_boundary"]["evaluation_split"], "validation")
        self.assertFalse(first["folds"][0]["manifest_contents_checked"])
        self.assertFalse(first["folds"][0]["cardinality_checked"])
        self.assertEqual(len(first["commands_argv"]), 4)
        sources = first["sources"]
        self.assertEqual(sources["training_source_sha256"], training_source_sha256(ROOT / "scripts"))
        for name in ("scripts/run_controlled_development.py", "scripts/development/config.py",
                     "scripts/development/runner.py", "scripts/development/models.py", "scripts/pipeline/data.py"):
            self.assertIn(name, sources["source_file_sha256"])
        self.assertNotEqual(sources["training_source_sha256"], sources["development_source_sha256"])

    def test_plan_commands_preserve_flags_and_selection(self):
        with mock.patch("scripts.development.config.source_provenance", return_value={}):
            plan = make_plan(dataclasses.replace(self.config, scenario="clean", single_fold="fold_1", epochs=2), self.root)
        variants = set()
        for command in plan["commands_argv"]:
            config, plan_only = parse_args(command[2:])
            self.assertFalse(plan_only)
            self.assertEqual((config.scenario, config.epochs, config.folds), ("clean", 2, ("fold_1",)))
            variants.add((config.attention, config.auxiliary_head))
        self.assertEqual(variants, {("cbam", "on"), ("cbam", "off"), ("none", "on"), ("none", "off")})

    def test_actual_cli_plan_blocks_heavy_imports_and_dataset_reads(self):
        # Use actual entry point, both direct-file and module entry contracts.
        code = '''
import importlib.abc, pathlib, runpy, sys
class NoHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tensorflow', 'keras', 'numpy', 'pandas', 'cv2', 'sklearn'}:
            raise AssertionError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, NoHeavy())
original_open = pathlib.Path.open
def no_dataset_open(path, *args, **kwargs):
    if path.suffix != '.py':
        raise AssertionError('Forbidden dataset/weight/output read: ' + str(path))
    return original_open(path, *args, **kwargs)
pathlib.Path.open = no_dataset_open
sys.argv = ['scripts/run_controlled_development.py', '--folds-root', sys.argv[1], '--output-dir', sys.argv[2], '--plan-only']
runpy.run_path('scripts/run_controlled_development.py', run_name='__main__')
'''
        output = ROOT / "results/development_controlled" / ("pure_test_" + self.root.name)
        self.assertFalse(output.exists())
        command = [sys.executable, "-B", "-c", code, str(self.folds_root), str(output)]
        first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(json.loads(first.stdout)["mode"], "plan_only")
        self.assertFalse(output.exists())
        module = subprocess.run([sys.executable, "-B", "-m", "scripts.run_controlled_development",
                                 "--folds-root", str(self.folds_root), "--output-dir", str(output), "--plan-only"],
                                cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(first.stdout, module.stdout)

    def test_split_cardinality_is_not_equal_split_assumption(self):
        rows = []
        fold = self.folds_root / "fold_1"
        for split, total_per_class in (("train", 5), ("validation", 2), ("test", 3)):
            for index, label in enumerate(("normal", "subluxation")):
                for item in range(total_per_class):
                    rows.append({"split": split, "class_name": label, "tmd_label": index,
                                 "filepath": str(fold / split / label / f"{item}.png")})
        counts = validate_index_rows(rows, fold)
        self.assertEqual([counts[split]["count"] for split in ("train", "validation", "test")], [10, 4, 6])
        with self.assertRaises(ValueError):
            validate_index_rows([row for row in rows if row["split"] != "test"], fold)
        with self.assertRaises(ValueError):
            validate_index_rows([*rows, {**rows[0], "filepath": str(self.root / "elsewhere.png")}], fold)
        with self.assertRaises(ValueError):
            validate_index_rows([*rows, {**rows[0], "tmd_label": 1}], fold)

    def test_manifest_split_cardinality_rejects_same_total_wrong_distribution(self):
        metadata = {split: {"count": count} for split, count in (("train", 4), ("validation", 2), ("test", 2))}
        rows = [{"fold_split": split} for split, data in metadata.items() for _ in range(data["count"])]
        self.assertEqual(validate_manifest_split_counts(rows, metadata), {"train": 4, "validation": 2, "test": 2})
        with self.assertRaises(ValueError):
            validate_manifest_split_counts([{**rows[0], "fold_split": "validation"}, *rows[1:]], metadata)
        with self.assertRaises(ValueError):
            validate_manifest_split_counts([*rows, {"fold_split": "unknown"}], metadata)

    def test_sequences_only_accept_train_and_validation_and_adjust_targets(self):
        for auxiliary in ("on", "off"):
            factory = mock.Mock(side_effect=[object(), object()])
            config = dataclasses.replace(self.config, auxiliary_head=auxiliary)
            make_sequences(config, FakeFrame("train"), FakeFrame("validation"), {0: 1.2, 1: 0.9}, factory)
            train, validation = factory.call_args_list
            self.assertEqual(train.kwargs["multi_task"], auxiliary == "on")
            self.assertEqual(validation.kwargs["multi_task"], auxiliary == "on")
            self.assertTrue(train.kwargs["training"])
            self.assertFalse(validation.kwargs["training"])
            self.assertIn("tmd_class_weights", train.kwargs)
            self.assertNotIn("tmd_class_weights", validation.kwargs)
        factory = mock.Mock()
        for train, validation in ((FakeFrame("test"), FakeFrame("validation")),
                                  (FakeFrame("train"), FakeFrame("test")),
                                  (FakeFrame("train"), FakeFrame("validation", 0))):
            with self.assertRaises(ValueError):
                make_sequences(self.config, train, validation, {}, factory)
        factory.assert_not_called()

    def test_post_checkpoint_collection_refuses_test_or_misaligned_predictions(self):
        collector = mock.Mock()
        for split, training in (("test", False), ("validation", True)):
            generator = mock.Mock(training=training, df=FakeFrame(split))
            with self.assertRaises(ValueError):
                collect_validation_predictions(object(), generator, True, collector)
        collector.assert_not_called()
        generator = mock.Mock(training=False, df=FakeFrame("validation"))
        collector.return_value = ([], [], [], [], [], [], generator.df["filepath"].tolist())
        collect_validation_predictions(object(), generator, False, collector)
        collector.return_value = ([], [], [], [], [], [], ["/test/0.png", "/test/1.png"])
        with self.assertRaises(ValueError):
            collect_validation_predictions(object(), generator, False, collector)

    def test_checkpoint_first_best_epoch_and_finite_guard(self):
        self.assertEqual(select_checkpoint([0.5, 0.2, 0.2, 0.3]), (2, 0.2))
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                select_checkpoint(values)

    def test_failed_reserved_run_reports_nonzero_without_importing_tf(self):
        with mock.patch("scripts.development.runner.source_provenance", side_effect=RuntimeError("preflight failure")):
            with self.assertRaisesRegex(RuntimeError, "preflight failure"):
                execute(self.config, self.root)
        status = json.loads((self.config.output_dir / "run_status.json").read_text())
        self.assertEqual((status["state"], status["exit_code"]), ("failed", 1))
        self.assertTrue((self.config.output_dir / "RUN_RESERVED").exists())
        with self.assertRaises(FileExistsError):
            execute(self.config, self.root)

    def test_mocked_execution_audits_all_folds_first_and_records_completion(self):
        self.fixture_fold("fold_2")
        audited = []
        def indexer(fold):
            self.assertTrue((self.config.output_dir / "RUN_RESERVED").is_file())
            audited.append(fold.name)
            rows = [{"split": split, "class_name": label, "tmd_label": index,
                     "filepath": str(fold / split / label / "fixture.png")}
                    for split in ("train", "validation", "test")
                    for index, label in enumerate(("normal", "subluxation"))]
            counts = validate_index_rows(rows, fold)
            return FakeIndexed(rows), {"manifest_sha256": "fixture_manifest_hash",
                                       "split_metadata": counts,
                                       "manifest_split_counts": {s: v["count"] for s, v in counts.items()},
                                       "byte_identity_only": True}
        for fold in discover_folds(self.config):
            (fold / "manifest.csv").write_text("fold_split\ntrain\ntrain\nvalidation\nvalidation\ntest\ntest\n")
        def mocked_fold(config, fold, train, validation, output):
            self.assertEqual(audited, ["fold_1", "fold_2"])
            self.assertEqual(set(train["split"]), {"train"})
            self.assertEqual(set(validation["split"]), {"validation"})
            return {"fold": fold.name, "evaluation_boundary": EVALUATION_BOUNDARY}
        configure = mock.Mock()
        fake_config = types.SimpleNamespace(ARTIFACT_PROTOCOL="artifact_fixture", TRAINING_PROTOCOL="training_fixture",
                                            configure_tensorflow=configure)
        fake_tf = types.SimpleNamespace(__version__="mock_only", keras=types.SimpleNamespace(
            mixed_precision=types.SimpleNamespace(global_policy=lambda: types.SimpleNamespace(name="mock_float32"))))
        with mock.patch.dict(sys.modules, {"scripts.pipeline.config": fake_config,
                                           "tensorflow": fake_tf}), \
             mock.patch("scripts.development.runner.audit_indexed_fold", side_effect=indexer), \
             mock.patch("scripts.development.runner.source_provenance", return_value=source_provenance(ROOT)), \
             mock.patch("scripts.development.runner.environment_versions", return_value={"python": "mock"}), \
             mock.patch("scripts.development.runner.pipeline_config", return_value="runtime_fixture"), \
             mock.patch("scripts.development.runner.run_fold", side_effect=mocked_fold) as fold_runner:
            summary = execute(self.config, self.root)
        configure.assert_called_once_with("runtime_fixture")
        self.assertEqual(fold_runner.call_count, 2)
        self.assertEqual(len(summary["fold_results"]), 2)
        status = json.loads((self.config.output_dir / "run_status.json").read_text())
        self.assertEqual((status["state"], status["exit_code"], status["completed_folds"]), ("completed", 0, ["fold_1", "fold_2"]))
        provenance = json.loads((self.config.output_dir / "provenance.json").read_text())
        self.assertEqual(set(provenance["fold_manifest_sha256"]), {"fold_1", "fold_2"})
        self.assertEqual(provenance["split_metadata"]["fold_1"]["test"]["count"], 2)
        self.assertEqual(provenance["manifest_split_counts"]["fold_1"]["validation"], 2)

    def test_audit_failure_never_reaches_model_training_or_configuration(self):
        configure = mock.Mock()
        indexer = mock.Mock(side_effect=ValueError("leakage detected"))
        fake_config = types.SimpleNamespace(ARTIFACT_PROTOCOL="fixture", TRAINING_PROTOCOL="fixture",
                                            configure_tensorflow=configure)
        with mock.patch.dict(sys.modules, {"scripts.pipeline.config": fake_config,
                                           }), \
             mock.patch("scripts.development.runner.audit_indexed_fold", side_effect=indexer), \
             mock.patch("scripts.development.runner.source_provenance", return_value={}), \
             mock.patch("scripts.development.runner.environment_versions", return_value={}), \
             mock.patch("scripts.development.runner.run_fold") as fold_runner:
            with self.assertRaisesRegex(ValueError, "leakage detected"):
                execute(self.config, self.root)
        configure.assert_not_called()
        fold_runner.assert_not_called()
        status = json.loads((self.config.output_dir / "run_status.json").read_text())
        self.assertEqual((status["state"], status["exit_code"]), ("failed", 1))
        self.assertEqual(status["active_fold"], "fold_1")

    def test_interrupted_run_records_failure_exit_130(self):
        with mock.patch("scripts.development.runner.source_provenance", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                execute(self.config, self.root)
        status = json.loads((self.config.output_dir / "run_status.json").read_text())
        self.assertEqual((status["state"], status["exit_code"]), ("failed", 130))

    def test_cli_failure_has_nonzero_status_and_no_execution(self):
        from scripts.run_controlled_development import main
        with mock.patch("scripts.run_controlled_development.make_plan", side_effect=ValueError("refused")), \
             mock.patch("sys.stderr"), mock.patch("scripts.development.runner.execute") as runtime:
            self.assertEqual(main(["--output-dir", str(self.config.output_dir), "--plan-only"]), 1)
        runtime.assert_not_called()

    def test_static_training_boundary_callbacks_and_restore(self):
        source = (ROOT / "scripts/development/runner.py").read_text()
        self.assertIn("indexed, audit = audit_indexed_fold(fold)", source)
        self.assertIn("rows, audit = audit_fold(fold)", source)
        self.assertIn('indexed["split"] == "validation"', source)
        self.assertNotIn('indexed["split"] == "test"', source)
        self.assertNotIn("test_gen", source)
        self.assertNotIn("run_one_fold", source)
        self.assertIn("restore_best_weights=True", source)
        self.assertLess(source.index("PrimaryMonitorAlias(),"), source.index("MonitorGuard(),"))
        self.assertIn("tf.keras.utils.set_random_seed(config.seed)", source)
        self.assertIn("sources_unchanged_during_run", source)
        self.assertIn("patience=5", source)
        self.assertIn("patience=3", source)
        self.assertLess(source.index("model.load_weights(str(checkpoint))"), source.index("y_true, y_pred, confidence"))
        self.assertLess(source.index("reserve_output(config, project_root)"), source.index("from scripts.pipeline.config import ARTIFACT_PROTOCOL"))
        self.assertFalse(EVALUATION_BOUNDARY["test_generator_or_prediction"])

    def test_static_primary_and_auxiliary_layers_match_historical_proposed(self):
        # Compare AST layer expressions, not a TF instance or weight file.
        historical = ast.parse((ROOT / "scripts/pipeline/models.py").read_text())
        development = ast.parse((ROOT / "scripts/development/models.py").read_text())
        old = next(node for node in historical.body if isinstance(node, ast.FunctionDef) and node.name == "build_proposed_model")
        new = next(node for node in development.body if isinstance(node, ast.FunctionDef) and node.name == "build_model")
        def named_layers(function):
            layers = {}
            for node in ast.walk(function):
                if isinstance(node, ast.Call):
                    for keyword in node.keywords:
                        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                            layers[keyword.value.value] = ast.dump(node, include_attributes=False)
            return layers
        old_layers, new_layers = named_layers(old), named_layers(new)
        for name in ("tmd_gap", "tmd_fc1", "tmd_dropout", "tmd_bn", "tmd_fc2", "tmd_output",
                     "artifact_gap", "artifact_gmp", "artifact_pooled_features", "artifact_fc1", "artifact_dropout", "artifact_output"):
            self.assertEqual(old_layers[name], new_layers[name], name)
        # Whole auxiliary assignments also verify pooling really branches from conv5.
        def auxiliary_assignments(function):
            return [ast.dump(node, include_attributes=False) for node in function.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in {"artifact_average", "artifact_maximum", "artifact_features", "auxiliary", "artifact_output"}]
        self.assertEqual(auxiliary_assignments(old), auxiliary_assignments(new))
        self.assertIn('if not config.multi_task:', (ROOT / "scripts/development/models.py").read_text())


@unittest.skip("UNRUN: TensorFlow model structure requires separate parent authorization; weights=None only")
class DeferredTensorFlowStructureTests(unittest.TestCase):
    def test_all_variants_have_identical_primary_head_and_correct_output_contract(self):
        # Skipped unconditionally. No TF import or model construction occurs now.
        from scripts.development.models import build_model
        import tensorflow as tf

        for attention in ("cbam", "none"):
            for auxiliary in ("on", "off"):
                config = DevelopmentConfig(Path("unused"), Path("unused"), attention=attention, auxiliary_head=auxiliary)
                model = build_model(config, weights=None)
                self.assertEqual(model.get_layer("tmd_fc1").units, 1024)
                self.assertEqual(model.get_layer("tmd_dropout").rate, 0.5)
                self.assertEqual(model.get_layer("tmd_fc2").units, 128)
                self.assertEqual(model.get_layer("tmd_output").units, 2)
                self.assertEqual("cbam_attention" in {layer.name for layer in model.layers}, attention == "cbam")
                self.assertEqual(set(model.output_names), {"tmd_output", "artifact_output"} if auxiliary == "on" else {"tmd_output"})
                tf.keras.backend.clear_session()


if __name__ == "__main__":
    unittest.main()
