"""
Evaluator for Trainium NKI kernel optimization.

This evaluator tests evolved NKI kernels for:
1. Correctness: Output matches reference implementation within tolerance
2. Performance: Latency measured via NKI benchmark

The evaluator returns metrics compatible with OpenEvolve's evolution framework.
"""

import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

# Import OpenEvolve's EvaluationResult for structured returns
try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:
    EvaluationResult = None


# Baseline latency for the reference GEMM kernel (in milliseconds)
# Measured on trn1 instance with K=8192, M=4096, N=8192
BASELINE_LATENCY_MS = 21.0

# Test configuration
TEST_SIZES = {
    "small": {"K": 1024, "M": 512, "N": 1024},
    "medium": {"K": 4096, "M": 2048, "N": 4096},
    "large": {"K": 8192, "M": 4096, "N": 8192},
}
DEFAULT_TEST_SIZE = "large"

# Path to test runner script
EVALUATOR_DIR = Path(__file__).parent
TEST_RUNNER = EVALUATOR_DIR / "test_runner.py"


def _extract_latency(stdout: str) -> Optional[float]:
    """Extract latency from stdout using pattern 'Latency: <latency> ms'."""
    for line in stdout.split('\n'):
        if 'Latency:' in line and 'ms' in line:
            try:
                return float(line.split('Latency:')[1].split('ms')[0].strip())
            except ValueError:
                continue
    return None


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Evaluate an evolved NKI kernel for correctness and performance.
    
    Args:
        program_path: Path to the evolved program file
        
    Returns:
        Dictionary with metrics including combined_score (speedup ratio)
    """
    try:
        sizes = TEST_SIZES[DEFAULT_TEST_SIZE]
        
        # Run the test runner script
        result = subprocess.run(
            [sys.executable, str(TEST_RUNNER), program_path, 
             str(sizes["K"]), str(sizes["M"]), str(sizes["N"])],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        # Check correctness
        correct = "CORRECTNESS: PASSED" in stdout
        
        if not correct:
            metrics = {
                "combined_score": 0.0,
                "correctness": 0.0,
                "latency_ms": float('inf'),
                "speedup": 0.0,
            }
            if EvaluationResult:
                return EvaluationResult(
                    metrics=metrics,
                    artifacts={"stdout": stdout, "stderr": stderr}
                )
            return metrics
        
        # Extract latency
        latency = _extract_latency(stdout)
        
        if latency is None:
            metrics = {
                "combined_score": 0.5,
                "correctness": 1.0,
                "latency_ms": float('inf'),
                "speedup": 0.0,
            }
            if EvaluationResult:
                return EvaluationResult(
                    metrics=metrics,
                    artifacts={"stdout": stdout, "stderr": stderr, "note": "No latency extracted"}
                )
            return metrics
        
        # Calculate speedup (higher is better)
        speedup = BASELINE_LATENCY_MS / latency if latency > 0 else 0.0
        
        metrics = {
            "combined_score": speedup,
            "correctness": 1.0,
            "latency_ms": latency,
            "speedup": speedup,
        }
        
        if EvaluationResult:
            return EvaluationResult(
                metrics=metrics,
                artifacts={
                    "stdout": stdout,
                    "stderr": stderr,
                    "baseline_latency_ms": BASELINE_LATENCY_MS,
                }
            )
        return metrics
        
    except subprocess.TimeoutExpired:
        metrics = {
            "combined_score": 0.0,
            "correctness": 0.0,
            "latency_ms": float('inf'),
            "speedup": 0.0,
            "timeout": True,
        }
        if EvaluationResult:
            return EvaluationResult(metrics=metrics, artifacts={"error": "Evaluation timeout"})
        return metrics
        
    except Exception as e:
        metrics = {
            "combined_score": 0.0,
            "correctness": 0.0,
            "latency_ms": float('inf'),
            "speedup": 0.0,
            "error": str(e),
        }
        if EvaluationResult:
            return EvaluationResult(
                metrics=metrics,
                artifacts={"error": str(e), "traceback": traceback.format_exc()}
            )
        return metrics


def measure_baseline():
    """Measure baseline latency of the initial kernel."""
    initial_program = EVALUATOR_DIR / "initial_program.py"
    sizes = TEST_SIZES[DEFAULT_TEST_SIZE]
    
    result = subprocess.run(
        [sys.executable, str(TEST_RUNNER), str(initial_program),
         str(sizes["K"]), str(sizes["M"]), str(sizes["N"])],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])
    
    latency = _extract_latency(result.stdout)
    if latency:
        print(f"\nUpdate BASELINE_LATENCY_MS in evaluator.py to: {latency:.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--baseline":
            measure_baseline()
        else:
            result = evaluate(sys.argv[1])
            print("\nEvaluation Result:")
            metrics = result.metrics if hasattr(result, 'metrics') else result
            for k, v in metrics.items():
                print(f"  {k}: {v}")
    else:
        print("Usage:")
        print("  python evaluator.py <program_path>  - Evaluate a program")
        print("  python evaluator.py --baseline      - Measure baseline latency")
