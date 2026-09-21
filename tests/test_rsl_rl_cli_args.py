"""Check that training CLI flags cannot accidentally enable checkpoint loading."""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


_PATH = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/cli_args.py"
_SPEC = importlib.util.spec_from_file_location("training_cli_args", _PATH)
cli_args = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli_args)


class TrainingCliArgsTest(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        cli_args.add_rsl_rl_args(parser)
        return parser

    def test_false_overrides_a_resuming_config(self):
        for value in ("False", "false", "NO", "0"):
            args = self.parser().parse_args(["--resume", value])
            cfg = SimpleNamespace(resume=True, logger="tensorboard")
            cli_args.update_rsl_rl_cfg(cfg, args)
            self.assertIs(cfg.resume, False)

    def test_true_preserves_explicit_resume_and_checkpoint(self):
        for value in ("True", "true", "YES", "1"):
            args = self.parser().parse_args(["--resume", value, "--checkpoint", "model_500.pt"])
            cfg = SimpleNamespace(resume=False, logger="tensorboard")
            cli_args.update_rsl_rl_cfg(cfg, args)
            self.assertIs(cfg.resume, True)
            self.assertEqual(cfg.load_checkpoint, "model_500.pt")

    def test_omitting_resume_preserves_config(self):
        args = self.parser().parse_args([])
        self.assertIsNone(args.resume)
        for initial in (False, True):
            cfg = SimpleNamespace(resume=initial, logger="tensorboard")
            cli_args.update_rsl_rl_cfg(cfg, args)
            self.assertIs(cfg.resume, initial)

    def test_invalid_resume_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.parser().parse_args(["--resume", "maybe"])
        self.assertEqual(error.exception.code, 2)

    def test_finetune_overrides_both_optimizers_and_disables_annealing(self):
        args = self.parser().parse_args([
            "--resume", "true", "--log_root", "logs/foot_finetune",
            "--learning_rate", "1e-4", "--encoder_learning_rate", "1e-4",
            "--learning_rate_schedule", "fixed", "--experiment_name", "foot_finetune",
        ])
        cfg = SimpleNamespace(resume=False, logger="tensorboard", algorithm=SimpleNamespace(
            learning_rate=1e-3, est_learning_rate=1e-3, schedule="adaptive", anneal_lr=True,
        ))
        cli_args.update_rsl_rl_cfg(cfg, args)
        self.assertTrue(cfg.resume)
        self.assertEqual(cfg.algorithm.learning_rate, 1e-4)
        self.assertEqual(cfg.algorithm.est_learning_rate, 1e-4)
        self.assertEqual(cfg.algorithm.schedule, "fixed")
        self.assertFalse(cfg.algorithm.anneal_lr)
        self.assertEqual(cfg.experiment_name, "foot_finetune")
        self.assertEqual(args.log_root, "logs/foot_finetune")

    def test_new_training_keeps_default_optimizer_settings(self):
        args = self.parser().parse_args(["--resume", "false"])
        cfg = SimpleNamespace(resume=True, logger="tensorboard", experiment_name="foot",
                              algorithm=SimpleNamespace(learning_rate=.001, est_learning_rate=.001,
                                                        schedule="adaptive", anneal_lr=False))
        expected = vars(cfg.algorithm).copy()
        cli_args.update_rsl_rl_cfg(cfg, args)
        self.assertEqual(vars(cfg.algorithm), expected)
        self.assertEqual(cfg.experiment_name, "foot")
        self.assertIsNone(args.log_root)

    def test_train_log_root_is_explicit_or_uses_experiment_directory(self):
        tree = ast.parse((_PATH.parent / "train.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        assignments = [node for node in main.body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "log_root_path"]
        cfg = SimpleNamespace(experiment_name="wf_tron_1a_foot_all_terrain")
        for root, expected in [(None, "logs/rsl_rl/wf_tron_1a_foot_all_terrain"),
                               ("logs/foot_finetune", "logs/foot_finetune")]:
            args = self.parser().parse_args([] if root is None else ["--log_root", root])
            namespace = dict(os=os, args_cli=args, agent_cfg=cfg)
            exec(compile(ast.Module(body=assignments, type_ignores=[]), "train_log_root", "exec"), namespace)
            self.assertEqual(namespace["log_root_path"], os.path.abspath(expected))

    def test_invalid_learning_rates_and_schedule_are_rejected(self):
        for name in ("--learning_rate", "--encoder_learning_rate"):
            for value in ("0", "-1", "nan", "inf", "bad"):
                with self.subTest(name=name, value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.parser().parse_args([name, value])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parser().parse_args(["--learning_rate_schedule", "unknown"])


if __name__ == "__main__":
    unittest.main()
