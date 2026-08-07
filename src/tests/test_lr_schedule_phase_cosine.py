from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
except ImportError:
    np = None

if np is not None:
    sys.path.insert(0, str(ROOT))
    from train import lr_schedule_scale  # noqa: E402


@unittest.skipIf(np is None, "NumPy/PyTorch are unavailable in this local environment")
class PhaseCosineLRScheduleTest(unittest.TestCase):
    def test_phase_cosine_hits_requested_waypoints(self) -> None:
        total_steps = 1000
        phase1_fraction = 0.25
        phase2_fraction = 0.50
        phase2_ramp_fraction = 0.20

        self.assertTrue(
            math.isclose(
                lr_schedule_scale(
                    step=0,
                    total_steps=total_steps,
                    warmup_steps=0,
                    decay_steps=0,
                    schedule="phase_cosine",
                    final_lr_fraction=0.10,
                    phase1_fraction_of_total=phase1_fraction,
                    phase2_fraction_of_total=phase2_fraction,
                    phase_lr_start_fraction=0.25,
                    phase1_end_lr_fraction=0.75,
                    phase2_peak_lr_fraction=1.00,
                    phase2_peak_ramp_fraction=phase2_ramp_fraction,
                    linear_lr_halfway_fraction=0.50,
                ),
                0.25,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

        phase1_end_step = int(total_steps * phase1_fraction)
        self.assertTrue(
            math.isclose(
                lr_schedule_scale(
                    step=phase1_end_step,
                    total_steps=total_steps,
                    warmup_steps=0,
                    decay_steps=0,
                    schedule="phase_cosine",
                    final_lr_fraction=0.10,
                    phase1_fraction_of_total=phase1_fraction,
                    phase2_fraction_of_total=phase2_fraction,
                    phase_lr_start_fraction=0.25,
                    phase1_end_lr_fraction=0.75,
                    phase2_peak_lr_fraction=1.00,
                    phase2_peak_ramp_fraction=phase2_ramp_fraction,
                    linear_lr_halfway_fraction=0.50,
                ),
                0.75,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

        phase2_peak_step = int(total_steps * (phase1_fraction + phase2_fraction * phase2_ramp_fraction))
        self.assertTrue(
            math.isclose(
                lr_schedule_scale(
                    step=phase2_peak_step,
                    total_steps=total_steps,
                    warmup_steps=0,
                    decay_steps=0,
                    schedule="phase_cosine",
                    final_lr_fraction=0.10,
                    phase1_fraction_of_total=phase1_fraction,
                    phase2_fraction_of_total=phase2_fraction,
                    phase_lr_start_fraction=0.25,
                    phase1_end_lr_fraction=0.75,
                    phase2_peak_lr_fraction=1.00,
                    phase2_peak_ramp_fraction=phase2_ramp_fraction,
                    linear_lr_halfway_fraction=0.50,
                ),
                1.00,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

        self.assertLess(
            lr_schedule_scale(
                step=total_steps - 1,
                total_steps=total_steps,
                warmup_steps=0,
                decay_steps=0,
                schedule="phase_cosine",
                final_lr_fraction=0.10,
                phase1_fraction_of_total=phase1_fraction,
                phase2_fraction_of_total=phase2_fraction,
                phase_lr_start_fraction=0.25,
                phase1_end_lr_fraction=0.75,
                phase2_peak_lr_fraction=1.00,
                phase2_peak_ramp_fraction=phase2_ramp_fraction,
                linear_lr_halfway_fraction=0.50,
            ),
            1.00,
        )


if __name__ == "__main__":
    unittest.main()
