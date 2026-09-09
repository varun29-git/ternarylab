import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from terminus.budget import budget_report, TARGET_TOKENS
from terminus.cluster_benchmark import benchmark
from terminus.experiments import candidates
from terminus.loss import streaming_cross_entropy_sum
from terminus.model import BitNetLlama
from terminus.preflight import count_model
from terminus.ternary_export import pack_trits, unpack_trits, export_model, load_export
from terminus.train import load_config, run_training
from tests.helpers import tiny_config, fixture


class ExactDesignTests(unittest.TestCase):
    def test_ablation_matrix_preserves_all_invariants(self):
        base = json.loads(Path("configs/1.5b_exact_h100_fp8.json").read_text())
        self.assertEqual(len(candidates(base)), 17)

    def test_fused_qkv_gate_up_preserve_quantization_and_gradients(self):
        config = tiny_config(learned_norms=False, ternary_embeddings=True, loss_backend="streaming",
                             weight_scale_layout="blockwise", alpha_group_size=8)
        reference = BitNetLlama(config)
        actual = BitNetLlama(replace(config, fused_projections=True))

        def combined(name, values):
            if "qkv_proj" in name:
                parts = [values[name.replace("qkv_proj", f"{kind}_proj")] for kind in ("q", "k", "v")]
            elif "gate_up_proj" in name:
                parts = [values[name.replace("gate_up_proj", f"{kind}_proj")] for kind in ("gate", "up")]
            else:
                return values[name]
            return torch.cat(parts, 0) if name.endswith("weight") else parts[0]

        state = reference.state_dict()
        actual.load_state_dict({name: combined(name, state) for name in actual.state_dict()})
        self.assertEqual(actual.count_parameters(), reference.count_parameters())
        inputs, targets = torch.randint(32, (2, 13)), torch.randint(32, (2, 13))
        output, expected = actual(inputs, targets), reference(inputs, targets)
        torch.testing.assert_close(output["logits"], expected["logits"], rtol=2e-5, atol=2e-6)
        output["loss"].backward()
        expected["loss"].backward()
        gradients = {name: parameter.grad for name, parameter in reference.named_parameters()}
        for name, parameter in actual.named_parameters():
            torch.testing.assert_close(parameter.grad, combined(name, gradients), rtol=2e-4, atol=2e-6)

    def test_count_and_budget_are_exact_for_both_precisions(self):
        for mode in ("bf16", "fp8"):
            config, training, _ = load_config(f"configs/1.5b_exact_h100_{mode}.json")
            counts = count_model(config)
            self.assertEqual(counts["total"], 1_500_000_000)
            self.assertEqual(counts["unquantized"], 0)
            report = budget_report(config, training)
            self.assertGreater(TARGET_TOKENS, 1_400_000_000_000)
            self.assertEqual(report["optimizer_updates"] * report["tokens_per_update"], TARGET_TOKENS)
            self.assertEqual(sum(report["allocation_cluster_hours"].values()) * 32 * 96, 600_000)
            self.assertEqual(report["flops"]["checkpoint_recompute_flops_per_token"], 0)
            required = report["required_cluster_goodput_tokens_per_second"]
            self.assertTrue(budget_report(config, training, required)["projection_from_supplied_goodput"]["within_production_allocation"])
            self.assertFalse(budget_report(config, training, required * .9)["projection_from_supplied_goodput"]["within_production_allocation"])
            with self.assertRaises(ValueError):
                budget_report(replace(config, vocab_size=config.vocab_size + 1), training)

    def test_streaming_loss_matches_full_loss_and_gradients(self):
        torch.manual_seed(27)
        for ignore_all in (False, True):
            for chunk in (1, 7, 40):
                hidden = torch.randn(2, 13, 12, requires_grad=True)
                weight = torch.randn(37, 12, requires_grad=True)
                targets = torch.randint(37, (2, 13))
                targets[:, :13 if ignore_all else 2] = -100
                expected = F.cross_entropy(F.linear(hidden, weight).flatten(0, 1), targets.flatten(), reduction="sum", ignore_index=-100)
                gradients = torch.autograd.grad(expected * .37, (hidden, weight))
                actual = streaming_cross_entropy_sum(hidden, weight, targets, chunk)
                actual_gradients = torch.autograd.grad(actual * .37, (hidden, weight))
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-6)
                for value, reference in zip(actual_gradients, gradients):
                    torch.testing.assert_close(value, reference, atol=3e-6, rtol=2e-5)

    def test_streaming_bf16_autocast_and_inference(self):
        x = torch.randn(9, 16, requires_grad=True)
        w = torch.randn(41, 16, requires_grad=True)
        labels = torch.randint(41, (9,))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = streaming_cross_entropy_sum(x, w, labels, 4)
            expected = F.cross_entropy(F.linear(x, w), labels, reduction="sum")
        gradients = torch.autograd.grad(actual, (x, w))
        references = torch.autograd.grad(expected, (x, w))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
        for value, reference in zip(gradients, references):
            torch.testing.assert_close(value, reference, rtol=.02, atol=.02)
        with torch.inference_mode():
            self.assertTrue(torch.isfinite(streaming_cross_entropy_sum(x, w, labels, 4)))
        for require_x, require_w in ((True, False), (False, True)):
            xx, ww = x.detach().requires_grad_(require_x), w.detach().requires_grad_(require_w)
            streaming_cross_entropy_sum(xx, ww, labels, 4).backward()
            self.assertEqual(xx.grad is not None, require_x)
            self.assertEqual(ww.grad is not None, require_w)

    def test_all_ternary_tied_weight_receives_lookup_and_loss_gradients(self):
        torch.manual_seed(32)
        config = tiny_config(learned_norms=False, ternary_embeddings=True, loss_backend="streaming")
        actual = BitNetLlama(config)
        reference = copy.deepcopy(actual)
        inputs, targets = torch.randint(32, (2, 11)), torch.randint(32, (2, 11))
        loss = actual(inputs, targets, return_logits=False)["loss"]
        expected = reference(inputs, targets, return_logits=True)["loss"]
        loss.backward()
        expected.backward()
        torch.testing.assert_close(loss, expected)
        for value, target in zip(actual.parameters(), reference.parameters()):
            self.assertIsNotNone(value.grad)
            self.assertGreater(float(value.grad.abs().sum()), 0)
            torch.testing.assert_close(value.grad, target.grad, atol=2e-6, rtol=1e-4)
        self.assertEqual(actual.count_parameters()["unquantized"], 0)

    def test_packing_and_cached_inference_export_roundtrip(self):
        for length in (1, 4, 5, 17, 25):
            codes = torch.randint(-1, 2, (length, 7), dtype=torch.int8)
            packed = pack_trits(codes)
            self.assertEqual(packed.numel(), (codes.numel() + 4) // 5)
            torch.testing.assert_close(unpack_trits(packed, codes.shape), codes, atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as temporary:
            model = BitNetLlama(tiny_config(learned_norms=False, ternary_embeddings=True,
                                           weight_scale_layout="blockwise", alpha_group_size=8,
                                           fused_projections=True)).eval()
            path = Path(temporary) / "weights.pt"
            receipt = export_model(model, path)
            self.assertFalse(receipt["latent_weights_included"])
            restored = load_export(path)
            self.assertEqual(sum(p.numel() for p in restored.parameters()), 0)
            tokens = torch.randint(32, (1, 11))
            with torch.inference_mode():
                expected = model(tokens)["logits"]
                actual = restored(tokens)["logits"]
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                cache, chunks = None, []
                for start in range(0, 11, 3):
                    result, cache = restored.forward_cached(tokens[:, start:start + 3], cache=cache)
                    chunks.append(result["logits"])
                torch.testing.assert_close(torch.cat(chunks, 1), expected, rtol=2e-5, atol=2e-6)
            with path.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_export(path)

    def test_fixed_invariants_refuse_wrong_run_before_cuda(self):
        with self.assertRaisesRegex(ValueError, "world size"):
            run_training("configs/1.5b_exact_h100_fp8.json", "missing", ".", "unused")

    @unittest.skipUnless(torch.distributed.is_available(), "distributed unavailable")
    def test_real_data_ddp_benchmark_matches_single_and_certifies_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, batch in (("single", 2), ("ddp", 1)):
                fixture(root / name, examples=64, batch_size=batch)
                path = root / name / "config.json"
                config = json.loads(path.read_text())
                config["model"].update(ternary_embeddings=True, learned_norms=False, loss_backend="streaming", fused_projections=True)
                config["training"]["validate_manifest_once_per_node"] = True
                path.write_text(json.dumps(config))
            single = benchmark(root / "single/config.json", root / "single.json", manifest_path=root / "single/manifest.json",
                               data_root=root / "single", batch_size=2, warmup=1, steps=2)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
                       "--master_addr=127.0.0.1", f"--master_port={port}", "--module", "terminus.cluster_benchmark",
                       "--config", str(root / "ddp/config.json"), "--manifest", str(root / "ddp/manifest.json"),
                       "--data-root", str(root / "ddp"), "--output", str(root / "ddp.json"),
                       "--batch-size", "1", "--warmup", "1", "--steps", "2"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180,
                        env={**os.environ, "OMP_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            actual = json.loads((root / "ddp.json").read_text())
            self.assertFalse(actual["synthetic"])
            self.assertEqual(actual["world_size"], 2)
            self.assertEqual(actual["tokens_per_update"], single["tokens_per_update"])
            for step in range(2):
                loss = sum(rank["losses"][step] for rank in actual["rank_results"]) / 2
                self.assertAlmostEqual(loss, single["rank_results"][0]["losses"][step], places=5)
                for rank in actual["rank_results"]:
                    self.assertAlmostEqual(rank["gradient_norms"][step], single["rank_results"][0]["gradient_norms"][step], places=5)


if __name__ == "__main__":
    unittest.main()
