from __future__ import annotations

import importlib.util
import sys
import types
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
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
            is_available=lambda: False,
            get_device_capability=lambda: (0, 0),
            get_device_name=lambda *_args, **_kwargs: "stub-cuda",
        )
        sys.modules["torch"] = torch_stub
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
        self.assertIn("terminus_400m_64k_h100_v1", config)
        self.assertIn("64,000-token", config)
        self.assertIn("never read, print, request", config)
        self.assertNotIn("sk-", config)

    def test_evaluator_locks_the_400m_64k_target_and_forwards_full_model_dims(self) -> None:
        source = (EXPERIMENT / "evaluator.py").read_text()
        self.assertIn('TARGET_PROFILE = "terminus_400m_64k_h100_v1"', source)
        self.assertIn('"vocab_size": 64_000', source)
        self.assertIn('"layers": 20', source)
        self.assertIn('"ffn_dim": 4_608', source)
        self.assertIn('"n_heads": 16', source)
        self.assertIn("TARGET_SHAPE_ENV_VARS", source)
        self.assertIn("the 400M target must not silently change", source)
        self.assertIn('"TERMINUS_EVOLVE_LAYERS"', source)
        self.assertIn('"--vocab-size", str(target_shape["vocab_size"])', source)
        self.assertIn('"--layers", str(target_shape["layers"])', source)
        self.assertIn('"--dim", str(target_shape["dim"])', source)
        self.assertIn('"--ffn-dim", str(target_shape["ffn_dim"])', source)
        self.assertIn('"--heads", str(target_shape["n_heads"])', source)
        self.assertIn('"--kv-heads", str(target_shape["n_kv_heads"])', source)

    def test_azure_operator_waits_for_vm_start_and_remote_gate_uses_full_stack(self) -> None:
        operator = (REPO_ROOT / "ops" / "azure_h100_bitlinear_operator.sh").read_text()
        gate = (REPO_ROOT / "ops" / "azure_h100_bitlinear_remote_gate.sh").read_text()
        self.assertIn('AZURE_VM_START_TIMEOUT_S="${AZURE_VM_START_TIMEOUT_S:-1800}"', operator)
        self.assertIn('sleep "$AZURE_VM_POLL_INTERVAL_S"', operator)
        self.assertIn('current_power_state="$(power_state)"', operator)
        self.assertIn("TERMINUS_EVOLVE_LAYERS", gate)
        self.assertIn("terminus_400m_64k_h100_v1", gate)
        self.assertIn('[TERMINUS_EVOLVE_VOCAB_SIZE]=64000', gate)
        self.assertIn('--vocab-size "$TERMINUS_EVOLVE_VOCAB_SIZE"', gate)
        self.assertIn('--layers "$TERMINUS_EVOLVE_LAYERS"', gate)
        self.assertIn("fla-core==0.5.0", gate)
        self.assertIn("flash-linear-attention==0.5.0", gate)
        self.assertIn("tilelang==0.1.10", gate)
        self.assertIn("muon-optimizer @ git+https://github.com/KellerJordan/Muon.git@", gate)
        self.assertIn("flash_attn_interface", gate)


if __name__ == "__main__":
    unittest.main()
