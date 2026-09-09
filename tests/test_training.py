import contextlib
import io
import json
import math
import os
import subprocess
import socket
import signal
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from terminus.checkpoint import load_checkpoint, restore_rng_state, rng_state, save_checkpoint
from terminus.integrity import fingerprint, source_hashes
from terminus.model import BitNetLlama
from terminus.optim import OptimizerConfig, build_optimizer, muon_update_scale_for_param
from terminus.train import TrainConfig, run_training, schedule_scale
from tests.helpers import checkpoint_payload, fixture, tiny_config


class TrainingTests(unittest.TestCase):
    def test_schedule_exact_endpoint(self):
        self.assertEqual(schedule_scale(10, 100, 20, .1), .5)
        self.assertEqual(schedule_scale(20, 100, 20, .1), 1.)
        self.assertEqual(schedule_scale(100, 100, 20, .1), .1)
        with self.assertRaises(ValueError):
            schedule_scale(1, 10, 10, .1)

    def test_preflight_refuses_cpu_stale_or_incomplete_evidence(self):
        from dataclasses import asdict
        from terminus.preflight import run_preflight, validate_preflight_report
        from terminus.train import runtime_contract

        config, device = tiny_config(), torch.device("cpu")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            run_preflight(config, device)
        with self.assertRaises(ValueError):
            validate_preflight_report(None, config, device)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            path.write_text(json.dumps({"status": "passed", "identity": {}}))
            with self.assertRaisesRegex(ValueError, "stale"):
                validate_preflight_report(path, config, device)
            identity = {"model_sha256": fingerprint(asdict(config)), "source": source_hashes(), "runtime": runtime_contract(device)}
            path.write_text(json.dumps({"status": "passed", "identity": identity, "attention": [True]}))
            with self.assertRaisesRegex(ValueError, "evidence"):
                validate_preflight_report(path, config, device)

    def test_bad_training_configs(self):
        for options in ({"dtype": "fp16"}, {"grad_clip": math.nan}, {"gradient_accumulation_steps": 0},
                        {"log_interval": 0}, {"final_lr_fraction": math.nan}, {"warmup_tokens": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TrainConfig(**options)

    def test_optimizer_group_coverage_and_frozen_parameter(self):
        model = BitNetLlama(tiny_config())
        model.layers[0].ffn.up_proj.weight.requires_grad_(False)
        optimizer = build_optimizer(model)
        identifiers = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(set(identifiers), {id(parameter) for parameter in model.parameters() if parameter.requires_grad})
        norm_ids = {id(parameter) for parameter in model.parameters() if parameter.ndim < 2}
        for group in optimizer.param_groups:
            if any(id(parameter) in norm_ids for parameter in group["params"]):
                self.assertEqual(group["weight_decay"], 0)

    def test_muon_scale_accounts_for_upstream_aspect_ratio(self):
        for outputs, inputs in ((16, 4), (4, 16), (8, 8)):
            parameter = torch.nn.Parameter(torch.empty(outputs, inputs))
            upstream_scale = math.sqrt(max(1, outputs / inputs))
            combined = muon_update_scale_for_param(parameter) * upstream_scale
            self.assertAlmostEqual(combined, .2 * math.sqrt(max(outputs, inputs)))

    def test_rng_roundtrip(self):
        state = rng_state()
        expected = (torch.rand(3), np.random.rand(3))
        restore_rng_state(state)
        torch.testing.assert_close(torch.rand(3), expected[0])
        np.testing.assert_array_equal(np.random.rand(3), expected[1])

    def test_checkpoint_checks_integrity_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            model = BitNetLlama(tiny_config())
            optimizer = build_optimizer(model)
            state = {"updates": 1, "tokens_seen": 16, "next_micro_step": 1}
            path = save_checkpoint(directory, model, optimizer, state, {"test": True}, [rng_state()])
            self.assertEqual(load_checkpoint(directory, {"test": True})["state"], state)
            with self.assertRaises(FileExistsError):
                save_checkpoint(directory, model, optimizer, state, {"test": True}, [rng_state()])
            with self.assertRaisesRegex(ValueError, "contract"):
                load_checkpoint(path, {"test": False})
            with path.open("r+b") as handle:
                handle.seek(128)
                handle.write(b"BAD!")
            with patch("terminus.checkpoint.torch.load") as loader:
                with self.assertRaisesRegex(ValueError, "integrity"):
                    load_checkpoint(path, {"test": True})
                loader.assert_not_called()

    def test_incomplete_checkpoint_does_not_replace_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            model = BitNetLlama(tiny_config())
            optimizer = build_optimizer(model)
            first = {"updates": 1, "tokens_seen": 16, "next_micro_step": 1}
            save_checkpoint(directory, model, optimizer, first, {}, [rng_state()])
            with patch("terminus.checkpoint.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_checkpoint(directory, model, optimizer, {**first, "updates": 2}, {}, [rng_state()])
            self.assertEqual(load_checkpoint(directory, {})["state"], first)

    def test_nonfinite_checkpoint_state_preserves_previous_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            model = BitNetLlama(tiny_config())
            optimizer = build_optimizer(model)
            state = {"updates": 1, "tokens_seen": 16, "next_micro_step": 1}
            save_checkpoint(directory, model, optimizer, state, {}, [rng_state()])
            parameter = next(model.parameters())
            with torch.no_grad():
                parameter.view(-1)[0] = float("inf")
            with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
                save_checkpoint(directory, model, optimizer, {**state, "updates": 2}, {}, [rng_state()])
            self.assertEqual(load_checkpoint(directory, {})["state"], state)
            with torch.no_grad():
                parameter.zero_()
            optimizer.state[parameter]["exp_avg"] = torch.full_like(parameter, float("nan"))
            with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
                save_checkpoint(directory, model, optimizer, {**state, "updates": 2}, {}, [rng_state()])

    def test_quantization_curriculum_reaches_endpoint(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            fixture(root / "dataset")
            config_path = root / "dataset/config.json"
            config = json.loads(config_path.read_text())
            config["training"]["quantization_warmup_tokens"] = 72
            config["training"]["quantization_start_strength"] = 0.0
            config_path.write_text(json.dumps(config))
            run_training(config_path, root / "dataset/manifest.json", root / "dataset", root / "run")
            state = checkpoint_payload(root / "run")["model"]
            self.assertTrue(all(float(value) == 1 for name, value in state.items() if name.endswith("quantization_strength")))
            config["training"]["quantization_warmup_tokens"] = 73
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "quantization warmup"):
                run_training(config_path, root / "dataset/manifest.json", root / "dataset", root / "bad")

    def test_exact_resume_with_dropout_and_partial_accumulation(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            fixture(root / "dataset", dropout=.1)
            arguments = (root / "dataset/config.json", root / "dataset/manifest.json", root / "dataset")
            complete = run_training(*arguments, root / "uninterrupted")
            stopped = run_training(*arguments, root / "resumed", stop_after_updates=1)
            self.assertEqual(stopped["updates"], 1)
            actual = run_training(*arguments, root / "resumed", resume=root / "resumed/checkpoints")
            self.assertEqual(complete, actual)
            self.assertEqual(actual["tokens_seen"], 72)
            expected_payload, actual_payload = checkpoint_payload(root / "uninterrupted"), checkpoint_payload(root / "resumed")
            for name, expected in expected_payload["model"].items():
                torch.testing.assert_close(actual_payload["model"][name], expected, rtol=0, atol=0)
            for index, expected_state in expected_payload["optimizer"]["state"].items():
                for name, expected in expected_state.items():
                    torch.testing.assert_close(actual_payload["optimizer"]["state"][index][name], expected, rtol=0, atol=0)

    def test_changed_data_contract_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            fixture(root / "dataset")
            arguments = (root / "dataset/config.json", root / "dataset/manifest.json", root / "dataset")
            run_training(*arguments, root / "run", stop_after_updates=1)
            config_path = root / "dataset/config.json"
            config = json.loads(config_path.read_text())
            config["optimizer"]["adamw_lr"] *= 2
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex((RuntimeError, ValueError), "incompatible|contract"):
                run_training(*arguments, root / "run", resume=root / "run/checkpoints")

    def test_nonfinite_loss_does_not_publish_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            fixture(root / "dataset")
            original = BitNetLlama.forward

            def broken_forward(model, *args, **kwargs):
                output = original(model, *args, **kwargs)
                output["loss_sum"] = output["loss_sum"] * float("nan")
                return output

            with patch.object(BitNetLlama, "forward", broken_forward), self.assertRaises(RuntimeError):
                run_training(root / "dataset/config.json", root / "dataset/manifest.json", root / "dataset", root / "run")
            self.assertFalse((root / "run/checkpoints/latest.json").exists())

    @unittest.skipUnless(torch.distributed.is_available(), "distributed unavailable")
    def test_two_process_ddp_matches_single_process_with_empty_tail_rank(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            fixture(root / "single_data", batch_size=4)
            fixture(root / "ddp_data", batch_size=2)
            run_training(root / "single_data/config.json", root / "single_data/manifest.json", root / "single_data", root / "single")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--master_addr=127.0.0.1", f"--master_port={port}", "--nproc_per_node=2", "--module", "terminus.train",
                       "--config", str(root / "ddp_data/config.json"), "--manifest", str(root / "ddp_data/manifest.json"),
                       "--data-root", str(root / "ddp_data"), "--output-dir", str(root / "distributed")]
            environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo"}
            def launch(arguments):
                process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment, start_new_session=True)
                try:
                    stdout, stderr = process.communicate(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate()
                    self.fail("distributed smoke timed out: " + stdout + stderr)
                self.assertEqual(process.returncode, 0, stdout + stderr)

            launch(command)
            recovered_command = [str(root / "recovered") if argument == str(root / "distributed") else argument for argument in command]
            launch(recovered_command + ["--stop-after-updates", "1"])
            launch(recovered_command + ["--resume", str(root / "recovered/checkpoints")])
            expected, actual = checkpoint_payload(root / "single"), checkpoint_payload(root / "distributed")
            self.assertEqual(expected["state"]["tokens_seen"], actual["state"]["tokens_seen"])
            recovered = checkpoint_payload(root / "recovered")
            self.assertEqual(actual["state"], recovered["state"])
            self.assertEqual(len(recovered["rank_rng"]), 2)
            for name, weights in actual["model"].items():
                self.assertTrue(torch.equal(recovered["model"][name], weights), name)
            for parameter_id, optimizer_state in actual["optimizer"]["state"].items():
                for key, value in optimizer_state.items():
                    restored = recovered["optimizer"]["state"][parameter_id][key]
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(restored, value), key)
                    else:
                        self.assertEqual(restored, value)
            for name, weights in expected["model"].items():
                torch.testing.assert_close(actual["model"][name], weights, rtol=2e-4, atol=2e-6)

    def test_real_token_benchmark_stream(self):
        from terminus.benchmark import training_benchmark

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(80, dtype=np.uint16).tofile(path)
            report = training_benchmark("configs/development.json", batch_size=1, warmup_steps=0,
                                        measured_steps=1, tokens_path=path)
            self.assertEqual(report["token_source"], str(path))
            self.assertGreater(report["tokens_per_second"], 0)
