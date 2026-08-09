from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO_ROOT / "evolution" / "bitlinear_training_hopper"


def load_evaluator():
    path = EXPERIMENT / "evaluator.py"
    spec = importlib.util.spec_from_file_location("bitlinear_system_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import system evolution evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BitLinearSystemEvolutionStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.evaluator = load_evaluator()
        except ImportError as error:
            raise unittest.SkipTest(str(error)) from error

    def test_initial_patch_is_valid_and_empty(self) -> None:
        patch = self.evaluator._extract_patch(EXPERIMENT / "initial_program.py")
        self.assertEqual(self.evaluator._validate_patch(patch), ())

    def test_allowlist_covers_complete_bitlinear_compute_path(self) -> None:
        self.assertEqual(
            self.evaluator.MUTABLE_PATHS,
            {
                "src/ternary_training_kernels.py",
                "src/ternary_training.py",
                "src/deepgemm_integration.py",
                "src/packed_ternary.py",
                "src/model.py",
                "src/kernels.py",
            },
        )

    def test_protected_or_secret_patch_is_rejected(self) -> None:
        protected = """diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-x\n+y\n"""
        with self.assertRaisesRegex(ValueError, "protected"):
            self.evaluator._validate_patch(protected)
        secret = """diff --git a/src/ternary_training.py b/src/ternary_training.py\n--- a/src/ternary_training.py\n+++ b/src/ternary_training.py\n@@ -1 +1 @@\n-x\n+api_key = 'not-allowed'\n"""
        with self.assertRaisesRegex(ValueError, "prohibited"):
            self.evaluator._validate_patch(secret)

    def test_config_requires_fp8_wgrad_and_secret_hygiene(self) -> None:
        config = (EXPERIMENT / "config.yaml").read_text()
        self.assertIn("FP8 Wgrad", config)
        self.assertIn("Azure NVIDIA H100 (SM90)", config)
        self.assertIn("never read, print, request", config)
        self.assertNotIn("sk-", config)


if __name__ == "__main__":
    unittest.main()
