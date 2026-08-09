#!/usr/bin/env python3
"""
Self-evolving BitLinear GEMM optimization.

This script runs an iterative optimization loop where Zoo (the AI agent)
acts as the evolutionary optimizer, generating improved kernel variants
based on benchmark feedback.
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add the experiment directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
import triton.language as tl

from evaluator import evaluate


# Evolution history
evolution_history: List[Dict[str, Any]] = []
best_score = 0.0
best_program = ""


def load_seed() -> str:
    """Load the seed program."""
    seed_path = os.path.join(os.path.dirname(__file__), "seed.py")
    with open(seed_path, "r") as f:
        return f.read()


def save_program(program: str, generation: int, score: float) -> str:
    """Save a program variant."""
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"gen_{generation:03d}_score_{score:.2f}.py"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        f.write(program)
    
    return filepath


def save_best_program(program: str, score: float) -> None:
    """Save the best program."""
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, "best_kernel.py")
    with open(filepath, "w") as f:
        f.write(program)
    
    # Also save metadata
    metadata = {
        "score": score,
        "timestamp": datetime.now().isoformat(),
        "generations": len(evolution_history),
    }
    with open(os.path.join(output_dir, "best_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def apply_optimization(code: str, opt_id: int) -> str:
    """Apply a specific optimization to the code."""
    
    optimizations = [
        # 0: Add torch.compile
        lambda c: c.replace("def bitlinear_gemm", "@torch.compile\ndef bitlinear_gemm"),
        
        # 1: Use contiguous memory
        lambda c: c.replace(
            "b_scaled = b_ternary.float() * alpha[:, None]",
            "b_scaled = (b_ternary.float() * alpha[:, None]).contiguous()"
        ),
        
        # 2: Pre-transpose B
        lambda c: c.replace(
            "c = torch.matmul(a.float(), b_scaled.T)",
            "b_scaled_t = b_scaled.T.contiguous()\n    c = torch.matmul(a.float(), b_scaled_t)"
        ),
        
        # 3: Enable TF32
        lambda c: "import torch\ntorch.set_float32_matmul_precision('high')\n" + c,
        
        # 4: Use float32 for accumulation
        lambda c: c.replace(
            "return c.to(a.dtype)",
            "return c.to(a.dtype)"
        ),
        
        # 5: Add CUDA synchronization
        lambda c: c.replace(
            "c = torch.matmul(a.float(), b_scaled.T)",
            "c = torch.matmul(a.float(), b_scaled.T)\n    torch.cuda.synchronize()"
        ),
        
        # 6: Use bfloat16 for computation
        lambda c: c.replace(
            "c = torch.matmul(a.float(), b_scaled.T)",
            "c = torch.matmul(a.to(torch.bfloat16).float(), b_scaled.T)"
        ),
        
        # 7: Flatten and reshape for better memory access
        lambda c: c.replace(
            "b_scaled = b_ternary.float() * alpha[:, None]",
            "b_scaled = (b_ternary.float() * alpha[:, None]).view(N, K)"
        ),
        
        # 8: Use einsum instead of matmul
        lambda c: c.replace(
            "c = torch.matmul(a.float(), b_scaled.T)",
            "c = torch.einsum('mk,nk->mn', a.float(), b_scaled)"
        ),
        
        # 9: Add in-place operations
        lambda c: c.replace(
            "b_scaled = b_ternary.float() * alpha[:, None]",
            "b_scaled = b_ternary.float()\n    b_scaled.mul_(alpha[:, None])"
        ),
        
        # 10: Use torch.addmm for fused operation
        lambda c: c.replace(
            "c = torch.matmul(a.float(), b_scaled.T)",
            "c = torch.addmm(torch.zeros(M, N, device=a.device, dtype=torch.float32), a.float(), b_scaled.T)"
        ),
        
        # 11: Enable cudnn benchmark
        lambda c: "import torch\ntorch.backends.cudnn.benchmark = True\n" + c,
        
        # 12: Use torch.no_grad
        lambda c: c.replace(
            "def bitlinear_gemm",
            "@torch.no_grad()\ndef bitlinear_gemm"
        ),
        
        # 13: Combine torch.compile and contiguous
        lambda c: c.replace(
            "def bitlinear_gemm",
            "@torch.compile\ndef bitlinear_gemm"
        ).replace(
            "b_scaled = b_ternary.float() * alpha[:, None]",
            "b_scaled = (b_ternary.float() * alpha[:, None]).contiguous()"
        ),
        
        # 14: Full optimization: compile + contiguous + TF32
        lambda c: (
            "import torch\n"
            "torch.set_float32_matmul_precision('high')\n"
            + c.replace(
                "def bitlinear_gemm",
                "@torch.compile\ndef bitlinear_gemm"
            ).replace(
                "b_scaled = b_ternary.float() * alpha[:, None]",
                "b_scaled = (b_ternary.float() * alpha[:, None]).contiguous()"
            )
        ),
    ]
    
    if opt_id < len(optimizations):
        return optimizations[opt_id](code)
    return code


def generate_variant(seed_code: str, generation: int) -> str:
    """Generate a variant by applying random optimizations."""
    # Apply 1-3 random optimizations
    num_opts = random.randint(1, 3)
    result = seed_code
    
    applied = set()
    for _ in range(num_opts):
        opt_id = random.randint(0, 14)
        if opt_id not in applied:
            result = apply_optimization(result, opt_id)
            applied.add(opt_id)
    
    return result


def run_evolution(num_generations: int = 20) -> Dict[str, Any]:
    """Run the evolutionary optimization loop."""
    global best_score, best_program, evolution_history
    
    print("=" * 70)
    print("Self-Evolving BitLinear GEMM Optimization")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Generations: {num_generations}")
    print()
    
    # Enable TF32 for better performance
    torch.set_float32_matmul_precision('high')
    
    # Load seed
    seed_code = load_seed()
    
    # Evaluate seed
    print("Generation 0: Evaluating seed kernel...")
    seed_path = os.path.join(os.path.dirname(__file__), "seed.py")
    seed_results = evaluate(seed_path)
    
    best_score = seed_results["score"]
    best_program = seed_code
    
    evolution_history.append({
        "generation": 0,
        "score": seed_results["score"],
        "tflops": seed_results["tflops"],
        "correctness": seed_results["correctness"],
        "variant": "seed",
    })
    
    print(f"  Score: {seed_results['score']:.2f} TFLOPS")
    print(f"  Correctness: {seed_results['correctness']}")
    print()
    
    # Try all optimizations systematically first
    print("Phase 1: Systematic optimization search...")
    for opt_id in range(15):
        variant_code = apply_optimization(seed_code, opt_id)
        variant_path = save_program(variant_code, opt_id, 0.0)
        
        results = evaluate(variant_path)
        
        if results["correctness"] and results["score"] > best_score:
            best_score = results["score"]
            best_program = variant_code
            save_best_program(best_program, best_score)
            print(f"  Opt {opt_id:2d}: NEW BEST! {results['score']:.2f} TFLOPS")
        else:
            status = "PASS" if results["correctness"] else "FAIL"
            print(f"  Opt {opt_id:2d}: {results['score']:.2f} TFLOPS ({status})")
        
        evolution_history.append({
            "generation": opt_id + 1,
            "score": results["score"],
            "tflops": results["tflops"],
            "correctness": results["correctness"],
            "variant": f"opt_{opt_id}",
        })
        
        torch.cuda.empty_cache()
    
    print()
    print("Phase 2: Random combination search...")
    
    # Now do random combinations
    for gen in range(16, num_generations + 16):
        print(f"Generation {gen}: Generating variant...")
        
        # Generate variant
        variant_code = generate_variant(seed_code, gen)
        variant_path = save_program(variant_code, gen, 0.0)
        
        # Evaluate variant
        results = evaluate(variant_path)
        
        # Update best if improved
        if results["correctness"] and results["score"] > best_score:
            best_score = results["score"]
            best_program = variant_code
            save_best_program(best_program, best_score)
            print(f"  NEW BEST! Score: {results['score']:.2f} TFLOPS")
        else:
            status = "PASS" if results["correctness"] else "FAIL"
            print(f"  Score: {results['score']:.2f} TFLOPS ({status}) (best: {best_score:.2f})")
        
        # Record history
        evolution_history.append({
            "generation": gen,
            "score": results["score"],
            "tflops": results["tflops"],
            "correctness": results["correctness"],
            "variant": f"gen_{gen}",
        })
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        
        print()
    
    # Save final results
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    with open(os.path.join(output_dir, "evolution_history.json"), "w") as f:
        json.dump(evolution_history, f, indent=2)
    
    # Print summary
    print("=" * 70)
    print("Evolution Complete!")
    print("=" * 70)
    print(f"Best score: {best_score:.2f} TFLOPS")
    print(f"Improvement: {((best_score / seed_results['score']) - 1) * 100:.1f}%")
    print(f"Generations: {len(evolution_history)}")
    print(f"Best kernel saved to: {os.path.join(output_dir, 'best_kernel.py')}")
    print()
    
    return {
        "best_score": best_score,
        "best_program": best_program,
        "history": evolution_history,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run BitLinear GEMM evolution")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations")
    
    args = parser.parse_args()
    run_evolution(args.generations)
