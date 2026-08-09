from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO_ROOT / "evolution" / "packed_ternary_hopper"


def load_evaluator():
    path = EXPERIMENT / "evaluator.py"
    spec = importlib.util.spec_from_file_location("packed_seed_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import packed ternary evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvolutionSeedStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.evaluator = load_evaluator()
        except ImportError as error:
            raise unittest.SkipTest(str(error)) from error

    def test_initial_candidate_passes_static_policy(self) -> None:
        self.evaluator._validate_candidate_source(EXPERIMENT / "initial_program.py")

    def test_library_gemm_fallback_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            candidate.write_text("import torch\ndef packed_ternary_matmul(a, *args):\n    return torch.matmul(a, a)\n")
            with self.assertRaisesRegex(ValueError, "torch.matmul"):
                self.evaluator._validate_candidate_source(candidate)

    def test_benchmark_covers_all_flagship_matrix_roles(self) -> None:
        names = {case[0] for case in self.evaluator.BENCHMARK_CASES}
        for role in ("q_decode", "kv_decode", "ffn_up_decode", "ffn_down_decode"):
            self.assertIn(role, names)

    def test_seed_uses_k_major_codes_and_a_dedicated_decode_kernel(self) -> None:
        source = (EXPERIMENT / "initial_program.py").read_text()
        self.assertIn("_packed_ternary_decode_kernel", source)
        self.assertIn("[packed_K, N]", source)
        self.assertIn("packed >> 6", source)

    def test_evaluator_can_measure_deepgemm_as_dense_baseline(self) -> None:
        source = (EXPERIMENT / "evaluator.py").read_text()
        self.assertIn("deep_gemm", source)
        self.assertIn("bf16_gemm_nt", source)

    def test_config_contains_no_inline_secret(self) -> None:
        config = (EXPERIMENT / "config.yaml").read_text()
        self.assertNotIn("API_KEY", config)
        self.assertNotIn("sk-", config)


if __name__ == "__main__":
    unittest.main()
