"""Kiểm tra các module starter đã triển khai, hợp đồng với eval.py và notebook hợp lệ.

Chạy từ thư mục gốc repo:
    python -m unittest discover -s tests -v
Các kiểm tra model/loss không cần GPU; cần cài các dependency huấn luyện.
"""
import ast
import json
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "starter"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STARTER))

import eval as ev  # noqa: E402
import train  # noqa: E402
import losses  # noqa: E402

K = ev.NUM_CLASSES


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class TestPredictionContract(unittest.TestCase):
    def test_saved_predictions_are_accepted_by_eval(self):
        rng = np.random.default_rng(0)
        n = 200
        y = rng.integers(0, K, n)
        probs = softmax(rng.normal(size=(n, K)))
        names = [f"img{i}.jpg" for i in range(n)]
        with tempfile.TemporaryDirectory() as d:
            path = ev.save_predictions(Path(d) / "sub" / "F01_seed0_test.csv", names, y, probs)
            pred = ev.read_pred(str(path))
            self.assertEqual(pred.seed, 0)
            np.testing.assert_array_equal(pred.y_pred, probs.argmax(1))
            np.testing.assert_allclose(pred.probs, probs, atol=1e-6)

    def test_save_predictions_rejects_logits(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(10, K))
        with self.assertRaisesRegex(ValueError, "chuẩn hoá"):
            ev.save_predictions("unused.csv", [f"{i}.jpg" for i in range(10)], np.zeros(10, int), logits)

    def test_save_predictions_rejects_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "dạng"):
            ev.save_predictions("unused.csv", ["a.jpg"], [0], np.ones((1, 5)) / 5)


class TestTrainHelpers(unittest.TestCase):
    def test_pred_path_follows_eval_naming(self):
        cfg = train.Config(exp_id="F01", seed=2, pred_dir="predictions")
        self.assertEqual(train.pred_path(cfg, "test"), Path("predictions/F01_seed2_test.csv"))
        self.assertEqual(ev.parse_seed(str(train.pred_path(cfg, "test"))), 2)

    def test_run_dir(self):
        cfg = train.Config(exp_id="T03", seed=1, out_dir="runs")
        self.assertEqual(train.run_dir(cfg), Path("runs/T03/seed1"))

    def test_test_predictions_off_by_default(self):
        self.assertFalse(train.Config().save_test_predictions)

    def test_baseline_defaults_match_guide(self):
        c = train.Config()
        self.assertEqual((c.epochs, c.batch_size, c.lr_backbone, c.lr_head, c.weight_decay),
                         (12, 64, 1e-4, 1e-3, 0.05))

    def test_parse_overrides(self):
        result = train.parse_overrides(["seed=2", "loss=focal", "ema_decay=none", "amp=false"])
        self.assertEqual(result, {"seed": 2, "loss": "focal", "ema_decay": None, "amp": False})


class TestStarterFiles(unittest.TestCase):
    def test_all_python_files_compile(self):
        for f in sorted(STARTER.glob("*.py")) + [ROOT / "eval.py"]:
            py_compile.compile(str(f), doraise=True)

    def test_runtime_modules_have_no_unimplemented_stubs(self):
        for name in ("dataset.py", "model.py", "losses.py", "train.py", "inference.py", "benchmark.py"):
            tree = ast.parse((STARTER / name).read_text(encoding="utf-8"))
            n = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Raise)
                    and isinstance(node.exc, ast.Call) and getattr(node.exc.func, "id", "") == "NotImplementedError")
            self.assertEqual(n, 0, f"{name}: còn {n} NotImplementedError stub")

    def test_starter_has_no_complete_helper_modules(self):
        """Mọi file trong starter/ đều là pseudo-code: không còn module hoàn chỉnh kiểu records.py."""
        self.assertFalse((STARTER / "records.py").exists())
        for f in STARTER.glob("*.py"):
            self.assertNotIn("import records", f.read_text(encoding="utf-8"), f.name)

    def test_label_smoothing_zero_matches_cross_entropy(self):
        logits = torch.randn(8, K)
        targets = torch.randint(0, K, (8,))
        actual = losses.LabelSmoothingCE(smoothing=0.0)(logits, targets)
        expected = torch.nn.functional.cross_entropy(logits, targets)
        torch.testing.assert_close(actual, expected)

    def test_focal_gamma_zero_matches_cross_entropy(self):
        logits = torch.randn(8, K)
        targets = torch.randint(0, K, (8,))
        actual = losses.FocalLoss(gamma=0.0)(logits, targets)
        expected = torch.nn.functional.cross_entropy(logits, targets)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)

    def test_mixup_and_cutmix_contract(self):
        x = torch.rand(4, 3, 16, 16)
        y = torch.arange(4)
        for mode in ("mixup", "cutmix"):
            mixed, targets = losses.mix_batch(x, y, alpha=1.0, mode=mode)
            self.assertEqual(tuple(mixed.shape), tuple(x.shape))
            self.assertEqual(tuple(targets[0].shape), tuple(y.shape))
            self.assertEqual(tuple(targets[1].shape), tuple(y.shape))
            self.assertGreaterEqual(targets[2], 0.0)
            self.assertLessEqual(targets[2], 1.0)
            loss = losses.mixed_loss(torch.nn.CrossEntropyLoss(), torch.randn(4, K), targets)
            self.assertEqual(loss.ndim, 0)

    def test_notebook_is_valid_and_clean(self):
        nb = json.loads((STARTER / "lab_day2.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("eval.py score", text)
        self.assertIn("eval.py grade", text)
        self.assertIn("b7b30f96d466fba86016aa5a26606e0f", text)  # MD5 của images.zip


if __name__ == "__main__":
    unittest.main()
