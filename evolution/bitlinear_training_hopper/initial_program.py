"""OpenEvolve candidate envelope for the complete packed-BitLinear train path.

The only mutable value is ``SYSTEM_PATCH``.  It is a unified ``git diff``
applied to a disposable full-repository copy by the trusted evaluator.  This
lets an evolution candidate jointly change packing, Fprop, Dgrad, FP8 casting,
DeepGEMM Wgrad integration, autograd, and model wiring without ever modifying
the committed baseline.
"""

SYSTEM_PATCH = r'''# EVOLVE-BLOCK-START
# Start from the exact baseline. Replace this comment with a unified git patch
# whose paths are limited to the evaluator's BitLinear-kernel allowlist.
# EVOLVE-BLOCK-END'''
