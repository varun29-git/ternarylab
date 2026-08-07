from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AttentionContractStaticTest(unittest.TestCase):
    def test_production_backend_surface_has_no_auto(self) -> None:
        kernels = (ROOT / "kernels.py").read_text()
        self.assertIn('VALID_ATTENTION_BACKENDS = {"fa3", "flex", "sdpa_reference"}', kernels)
        self.assertIn('"silent_fallback": False', kernels)
        self.assertNotIn('VALID_ATTENTION_BACKENDS = {"auto"', kernels)
        self.assertIn('"allow_1_58_bit_cost_or_memory_claim": False', kernels)
        self.assertIn('"distributed_alpha_grouping_validated": False', kernels)

    def test_exact_1024_token_fa3_window(self) -> None:
        model = (ROOT / "model.py").read_text()
        self.assertIn("(self.local_window - 1, 0)", model)
        self.assertIn("(-1, -1) if self.is_global_layer", model)

    def test_launcher_runs_hard_gate_before_training(self) -> None:
        launcher = (ROOT / "run_h100_e2e.sh").read_text()
        quantization_gate = launcher.index("python -m unittest discover")
        gate = launcher.index("python attention_preflight.py")
        training = launcher.index('python "train.py"')
        self.assertLess(quantization_gate, gate)
        self.assertLess(gate, training)
        self.assertIn("--run-scaling", launcher)

    def test_norm_parameters_cannot_fall_back_to_general_adamw(self) -> None:
        launcher = (ROOT / "run_h100_e2e.sh").read_text()
        train = (ROOT / "train.py").read_text()
        optimizer = (ROOT / "optim.py").read_text()
        self.assertIn('SPLIT_NORM_PARAMETERS="${SPLIT_NORM_PARAMETERS:-1}"', launcher)
        self.assertIn("SPLIT_NORM_PARAMETERS=1 is mandatory", launcher)
        self.assertIn("split_norm_parameters: bool = True", train)
        self.assertIn("split_norm_parameters is mandatory", train)
        self.assertIn("split_norm_parameters: bool = True", optimizer)

    def test_subln_is_default_and_covers_both_residual_branches(self) -> None:
        model = (ROOT / "model.py").read_text()
        train = (ROOT / "train.py").read_text()
        launcher = (ROOT / "run_h100_e2e.sh").read_text()
        self.assertIn("use_subln: bool = True", model)
        self.assertIn("SingleScaleRMSNorm(config.dim, eps=config.rms_norm_eps)", model)
        self.assertIn("SingleScaleRMSNorm(config.ffn_dim, eps=config.rms_norm_eps)", model)
        self.assertNotIn("self.attn_sub_norm = RMSNorm", model)
        self.assertNotIn("self.ffn_sub_norm = RMSNorm", model)
        self.assertIn("self.o_proj(self.attn_sub_norm(out))", model)
        self.assertIn("hidden = self.ffn_sub_norm(hidden)", model)
        self.assertIn("use_subln: bool = True", train)
        self.assertIn('USE_SUBLN="${USE_SUBLN:-1}"', launcher)

    def test_training_uses_no_hard_gradient_or_latent_clipping(self) -> None:
        model = (ROOT / "model.py").read_text()
        train = (ROOT / "train.py").read_text()
        launcher = (ROOT / "run_h100_e2e.sh").read_text()
        self.assertIn("grad_clip: float = 0.0", train)
        self.assertIn("alpha_grad_clip: float = 0.0", train)
        self.assertNotIn("clip_grad_norm_", train)
        self.assertNotIn("clamp_alpha_", train)
        self.assertIn("hard_gradient_clipping=disabled", launcher)
        self.assertIn('ENABLE_ACTIVATION_QUANTIZATION="${ENABLE_ACTIVATION_QUANTIZATION:-0}"', launcher)
        self.assertIn("activation quantization uses integer saturation and is forbidden", launcher)

    def test_residual_and_final_norms_are_affine_but_qk_stays_non_affine(self) -> None:
        model = (ROOT / "model.py").read_text()
        self.assertIn("self.attn_norm = RMSNorm", model)
        self.assertIn("self.ffn_norm = RMSNorm", model)
        self.assertIn("self.norm = RMSNorm", model)
        self.assertIn("self.q_norm = UnitRMSNorm", model)
        self.assertIn("self.k_norm = UnitRMSNorm", model)
        self.assertNotIn("residual_norm_mode", model)
        self.assertNotIn("subln_mode", model)

    def test_launcher_only_requires_the_selected_attention_backend(self) -> None:
        launcher = (ROOT / "run_h100_e2e.sh").read_text()
        self.assertIn(
            'ATTENTION_PREFLIGHT_BACKENDS="${ATTENTION_PREFLIGHT_BACKENDS:-${ATTENTION_BACKEND}}"',
            launcher,
        )
        self.assertIn("Fixed causal local attention must use FA3", launcher)

    def test_gradient_preflight_has_a_stable_near_zero_relative_denominator(self) -> None:
        preflight = (ROOT / "attention_preflight.py").read_text()
        self.assertIn("gradient-relative-rms-floor", preflight)
        self.assertIn("reference_rms", preflight)
        self.assertIn("relative_rms_floor", preflight)

    def test_launcher_does_not_shadow_the_installed_pinned_fa3_extension(self) -> None:
        launcher = (ROOT / "run_h100_e2e.sh").read_text()
        self.assertIn("Do not put the Hopper source tree on PYTHONPATH", launcher)
        self.assertNotIn('export PYTHONPATH="${FA3_HOPPER_PATH}/hopper', launcher)

    def test_optimized_model_fails_before_dense_mask_preparation(self) -> None:
        model = (ROOT / "model.py").read_text()
        early_failure = model.index(
            'attention_mask is not None and self.config.attention_backend != "sdpa_reference"'
        )
        conditional_preparation = model.index(
            'if self.config.attention_backend == "sdpa_reference"',
            early_failure,
        )
        self.assertLess(early_failure, conditional_preparation)

    def test_cached_decode_has_explicit_local_global_policy(self) -> None:
        model = (ROOT / "model.py").read_text()
        self.assertIn("class InferenceKVCache", model)
        self.assertIn("if attention.is_global_layer", model)
        self.assertIn("self.config.local_attention_window", model)
        self.assertIn("offset=cache.position", model)
        sample = (ROOT / "sample.py").read_text()
        self.assertIn("model.forward_cached(input_ids", sample)
        self.assertIn("model.forward_cached(next_token", sample)

    def test_flex_cached_decode_offsets_query_positions(self) -> None:
        kernels = (ROOT / "kernels.py").read_text()
        self.assertIn("query_position_offset", kernels)
        self.assertIn("q_idx + query_position_offset - kv_idx", kernels)

    def test_requested_20_layer_4096_1024_defaults(self) -> None:
        model = (ROOT / "model.py").read_text()
        self.assertIn("n_layers: int = 20", model)
        self.assertIn("max_seq_len: int = 4096", model)
        self.assertIn("local_attention_window: int = 1024", model)
        self.assertIn("global_attention_every: int = 5", model)
        launcher = (ROOT / "run_h100_e2e.sh").read_text()
        self.assertIn('SEQ_LEN="${SEQ_LEN:-4096}"', launcher)
        self.assertIn('BATCH_SIZE="${BATCH_SIZE:-14}"', launcher)
        self.assertIn('N_LAYERS="${N_LAYERS:-20}"', launcher)
        self.assertIn('GLOBAL_ATTENTION_EVERY="${GLOBAL_ATTENTION_EVERY:-5}"', launcher)

    def test_qknorm_is_all_layer_and_qkclip_is_absent(self) -> None:
        model = (ROOT / "model.py").read_text().lower()
        train = (ROOT / "train.py").read_text().lower()
        launcher = (ROOT / "run_h100_e2e.sh").read_text().lower()
        self.assertIn('qk_norm_mode: str = "unit_rms"', model)
        self.assertIn("self.q_norm = unitrmsnorm", model)
        self.assertIn("self.k_norm = unitrmsnorm", model)
        self.assertIn('qk_norm_mode: str = "unit_rms"', train)
        self.assertIn('qk_norm_mode="${qk_norm_mode:-unit_rms}"', launcher)
        self.assertIn("affine q/k gains are forbidden", launcher)
        self.assertNotIn("qk_clip", model)
        self.assertNotIn("qk_clip", train)

    def test_launcher_defaults_to_bf16_and_observation_only_health_gates(self) -> None:
        launcher = (ROOT / "run_h100_e2e.sh").read_text()
        self.assertIn('DTYPE="${DTYPE:-bf16}"', launcher)
        self.assertIn('HEALTH_GATE_MODE="${HEALTH_GATE_MODE:-observe}"', launcher)

    def test_weight_decay_is_constant_through_training(self) -> None:
        train = (ROOT / "train.py").read_text()
        launcher = (ROOT / "run_h100_e2e.sh").read_text()
        self.assertNotIn("weight_decay_taper", train)
        self.assertNotIn("weight_decay_off", train)
        self.assertNotIn('group["weight_decay"] =', train)
        self.assertIn('"weight_decay_schedule": "constant_through_100_percent"', train)
        self.assertIn("weight_decay_schedule=constant_through_100_percent", launcher)

    def test_deterministic_quantizer_uses_full_target_identity_ste(self) -> None:
        model = (ROOT / "model.py").read_text()
        self.assertIn("quantized_target = alpha_scale * ternary", model)
        self.assertIn(
            "bitnet_weight = latent + (quantized_target - latent).detach()",
            model,
        )
        self.assertNotIn(
            "bitnet_weight = self.alpha_scale() * ste_weight",
            model,
        )
        self.assertIn(
            'if self.alpha_mode in {"learned_log_gain", "learned_grouped_log_gain"}',
            model,
        )

    def test_activation_quantized_path_reuses_corrected_weight_ste(self) -> None:
        model = (ROOT / "model.py").read_text()
        self.assertIn("x_dequant = x_int * scale", model)
        self.assertIn("x_ste = x_float + (x_dequant - x_float).detach()", model)
        self.assertNotIn("out = out.float() * activation_scale", model)

    def test_cache_attends_before_local_compaction(self) -> None:
        model = (ROOT / "model.py").read_text()
        attention_inputs = model.index("cache.attention_inputs(k_new, v_new)")
        attention_call = model.index("flash_attention3(", attention_inputs)
        commit = model.index("cache.commit(k_all, v_all)", attention_call)
        self.assertLess(attention_inputs, attention_call)
        self.assertLess(attention_call, commit)
        self.assertNotIn("cache.append(k_new, v_new)", model)

    def test_checkpoint_contract_is_atomic_and_next_step_based(self) -> None:
        train = (ROOT / "train.py").read_text()
        self.assertIn('"next_step": next_step', train)
        self.assertIn("os.replace(temp_path, path)", train)
        self.assertIn('"resume_contract": resume_contract', train)
        self.assertIn("validate_resume_contract", train)
        sample = (ROOT / "sample.py").read_text()
        self.assertIn("strict=True", sample)
        self.assertNotIn("strict=False", sample)

    def test_data_and_probe_semantics_are_explicit(self) -> None:
        data = (ROOT / "data.py").read_text()
        launcher = (ROOT / "run_h100_e2e.sh").read_text()
        probe = (ROOT / "throughput_probe.py").read_text()
        self.assertIn("source_mix", data)
        self.assertIn("_weighted_schedule", data)
        self.assertIn('token_buffer[self.loader.config.seq_len :]', data)
        self.assertIn('"source_mix": source_mix', launcher)
        self.assertIn("input_ids = token_stream[:, :-1]", probe)
        self.assertIn("targets = token_stream[:, 1:]", probe)
        self.assertIn('parser.add_argument("--disable-cuda-amp"', probe)
        self.assertIn('parser.add_argument("--qk-norm-mode"', probe)
        self.assertIn('parser.add_argument("--attention-softcap"', probe)

    def test_flex_decode_does_not_grow_block_mask_cache(self) -> None:
        model = (ROOT / "model.py").read_text()
        kernels = (ROOT / "kernels.py").read_text()
        preflight = (ROOT / "inference_cache_preflight.py").read_text()
        self.assertIn("if seq_len == 1", model)
        self.assertIn("causal=False", model)
        self.assertIn("_FLEX_BLOCK_MASK_CACHE_MAX_SIZE = 32", kernels)
        self.assertIn("cache_entries_after_decode != cache_entries_before_decode", preflight)

    def test_non_hopper_flex_path_is_allowed_while_fa3_remains_hopper_gated(self) -> None:
        attention_preflight = (ROOT / "attention_preflight.py").read_text()
        cache_preflight = (ROOT / "inference_cache_preflight.py").read_text()
        kernels = (ROOT / "kernels.py").read_text()
        self.assertIn('if "fa3" in backends and major < 9', attention_preflight)
        self.assertIn("use --backends flex on non-Hopper GPUs", attention_preflight)
        self.assertIn('if "fa3" in backends and major < 9', cache_preflight)
        self.assertIn("use --backends flex on non-Hopper GPUs", cache_preflight)
        self.assertIn("if requested == \"fa3\":", kernels)
        self.assertIn("requires Hopper-or-newer CUDA hardware", kernels)

    def test_flex_scaling_gate_measures_compiled_warm_path_not_compile_overhead(self) -> None:
        preflight = (ROOT / "attention_preflight.py").read_text()
        self.assertIn("def _compiled_flex_runner(", preflight)
        self.assertIn("runner = _compiled_flex_runner(window, float(softcap))", preflight)
        self.assertIn('memory_ratio = upper["peak_workspace_bytes"]', preflight)
        self.assertNotIn('memory_ratio = upper["cold_peak_bytes"]', preflight)

    def test_run_artifacts_name_the_weight_ste_contract(self) -> None:
        train = (ROOT / "train.py").read_text()
        self.assertIn(
            '"weight_ste_contract": "full_dequantized_target_identity"',
            train,
        )
        self.assertIn(
            '"- Weight STE: `full_dequantized_target_identity`"',
            train,
        )

    def test_production_research_telemetry_cannot_repeat_test16_defaults(self) -> None:
        launcher = (ROOT / "run_h100_e2e.sh").read_text()
        train = (ROOT / "train.py").read_text()
        self.assertIn('RESEARCH_MODE="${RESEARCH_MODE:-1}"', launcher)
        self.assertIn('UPDATE_SAMPLE_SIZE="${UPDATE_SAMPLE_SIZE:-4096}"', launcher)
        self.assertNotIn('RESEARCH_MODE="${RESEARCH_MODE:-0}"', launcher)
        self.assertNotIn('UPDATE_SAMPLE_SIZE="${UPDATE_SAMPLE_SIZE:-0}"', launcher)
        self.assertIn("Terminus Flagship 30B production requires RESEARCH_MODE=1", launcher)
        self.assertIn("UPDATE_SAMPLE_SIZE must be a positive integer", launcher)
        self.assertIn("wants_activation_stats = train_config.research_mode and should_log_interval(", train)

    def test_test16_scale_diagnostics_are_preserved_in_payloads(self) -> None:
        model = (ROOT / "model.py").read_text()
        train = (ROOT / "train.py").read_text()
        for field in (
            '"p999_logit"',
            '"near_512_mass"',
            '"distant_mass"',
            '"mean_head_attention_cosine_similarity"',
            '"post_rope_q_vector_rms_max"',
            '"p999_abs"',
            '"per_channel_rms_p99"',
            '"max_abs_over_alpha"',
            '"normalized_abs_kurtosis"',
            '"p01_boundary_distance_over_alpha"',
        ):
            self.assertIn(field, model)
        self.assertIn('"state_by_key"', train)
        self.assertIn('"effective_bitlinear"', train)
        self.assertIn("bitlinear_effective_update_stats", train)


if __name__ == "__main__":
    unittest.main()
