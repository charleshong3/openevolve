# Trainium NKI Kernel Optimization

This OpenEvolve example optimizes NKI (Neuron Kernel Interface) kernels for AWS Trainium hardware using evolutionary code optimization.

## Overview

The example evolves a GEMM (General Matrix Multiplication) kernel to improve performance while maintaining correctness. The optimization targets:

- **Throughput**: Minimize kernel latency (P99)
- **Correctness**: Output must match reference within numerical tolerance

## Prerequisites

### Hardware Requirements

- AWS Trainium instance (trn1.2xlarge or larger)
- Or AWS Inferentia2 instance (inf2.xlarge or larger)

### Software Requirements

```bash
# Install Neuron SDK (on Trainium/Inferentia instance)
pip install neuronxcc

# Install OpenEvolve
pip install -e /path/to/openevolve

# Or install from requirements
pip install -r requirements.txt
```

### Environment Setup

1. **Launch a Trainium instance** on AWS (e.g., trn1.2xlarge)

2. **Install the Neuron SDK**:
   ```bash
   # Add Neuron repository
   sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
   [neuron]
   name=Neuron YUM Repository
   baseurl=https://yum.repos.neuron.amazonaws.com
   enabled=1
   metadata_expire=0
   EOF
   
   # Install Neuron tools
   sudo yum install aws-neuronx-tools aws-neuronx-runtime-lib -y
   
   # Install Python packages
   pip install neuronxcc torch-neuronx
   ```

3. **Verify installation**:
   ```bash
   neuron-ls  # Should show available NeuronCores
   ```

## Usage

### 1. Measure Baseline Performance

First, measure the baseline latency of the initial kernel on your hardware:

```bash
cd examples/trainium_kernel_opt
python evaluator.py --baseline
```

Update `BASELINE_LATENCY_MS` in `evaluator.py` with the measured value.

### 2. Run Evolution (Recommended: Using Wrapper Script)

The wrapper script dynamically injects kernel-specific ISA documentation:

```bash
cd examples/trainium_kernel_opt

# Run with default settings (GEMM kernel)
python run.py

# Specify kernel type and output directory
python run.py --kernel gemm --output-dir outputs/trainium_gemm

# Preview the enhanced config without running
python run.py --dry-run
```

Available kernel types: `gemm`, `softmax`, `attention`, `layernorm`, `rmsnorm`, `mamba`, `transpose`

### Alternative: Run OpenEvolve Directly

```bash
# From the openevolve root directory
python -m openevolve \
    examples/trainium_kernel_opt/initial_program.py \
    examples/trainium_kernel_opt/evaluator.py \
    --config examples/trainium_kernel_opt/config.yaml \
    --output-dir outputs/trainium_gemm
```

Note: Running directly uses the static config without dynamic ISA injection.

### 3. Monitor Progress

The evolution will output:
- `combined_score`: Speedup ratio (>1.0 means faster than baseline)
- `latency_ms`: Measured P99 latency
- `correctness`: 1.0 if output matches reference

### 4. View Results

Best programs are saved in the output directory:
```bash
ls outputs/trainium_gemm/programs/
```

## Files

| File | Description |
|------|-------------|
| `run.py` | **Wrapper script** - injects dynamic ISA docs and runs OpenEvolve |
| `initial_program.py` | Starting GEMM kernel with EVOLVE-BLOCK markers |
| `evaluator.py` | Correctness testing and benchmarking |
| `config.yaml` | OpenEvolve configuration with NKI optimization strategies |
| `nki_isa.py` | NKI ISA documentation generator (used by run.py) |
| `requirements.txt` | Python dependencies |

## Customization

### Different Kernel Types

To optimize a different kernel type (e.g., attention, softmax):

1. Replace `initial_program.py` with your kernel
2. Update `evaluator.py` test sizes and reference implementation
3. Modify `config.yaml` system message for kernel-specific optimizations

### Dynamic ISA Documentation

The `run.py` wrapper automatically injects kernel-specific ISA documentation into the system message. This provides the LLM with detailed API references for the relevant NKI operations.

To preview the injected documentation:

```bash
python run.py --kernel gemm --dry-run
```

For programmatic use:

```python
from nki_isa import NkiIsaGenerator

generator = NkiIsaGenerator()
gemm_docs = generator.generate_isa("gemm")
attention_docs = generator.generate_isa("attention")
softmax_docs = generator.generate_isa("softmax")
```

## Optimization Strategies

The config includes 42 optimization strategies from NKI best practices:

1. **Memory Optimization**: Minimize HBM access, maximize SBUF/PSUM reuse
2. **Compute Optimization**: Maximize Tensor Engine utilization
3. **Data Layout**: Optimize for sequential access patterns
4. **Loop Transformations**: Tiling, fusion, reordering
5. **Precision**: Use bfloat16 where possible
6. **Pipelining**: Overlap data movement and compute

## Scoring

The evaluator uses a speedup-based scoring:

```
combined_score = baseline_latency / evolved_latency
```

- Score > 1.0: Faster than baseline (improvement)
- Score = 1.0: Same as baseline
- Score < 1.0: Slower than baseline
- Score = 0.0: Failed correctness test

## Troubleshooting

### Compilation Errors

NKI kernels may fail to compile due to constraint violations. Common issues:
- Tile dimensions exceed limits (P > 128, F > 512)
- Invalid indexing with affine_range variables
- Loop-carried dependencies in affine_range

### Timeout Issues

Kernel compilation can be slow. Increase timeout in `config.yaml`:
```yaml
evaluator:
  timeout: 600  # 10 minutes
```

### Memory Issues

For large kernels, reduce test sizes in `evaluator.py`:
```python
TEST_SIZES = {
    "small": {"K": 512, "M": 256, "N": 512},
    ...
}
```

## References

- [NKI Programming Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
- [Trainium Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium.html)
- [OpenEvolve Documentation](../../README.md)
