#!/usr/bin/env python3
"""
Wrapper script to run OpenEvolve with dynamic NKI ISA documentation.

This script:
1. Loads the base config.yaml
2. Injects kernel-specific ISA documentation from nki_isa.py
3. Runs OpenEvolve with the enhanced config

Usage:
    python run.py [--kernel gemm] [--output-dir outputs/trainium_gemm]
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import yaml

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from nki_isa import get_isa_docs


def load_config(config_path: Path) -> dict:
    """Load the base YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def inject_isa_docs(config: dict, kernel_name: str) -> dict:
    """Inject dynamic ISA documentation into the system message."""
    isa_docs = get_isa_docs(kernel_name)
    
    # Append detailed ISA docs to the system message
    current_message = config.get("prompt", {}).get("system_message", "")
    
    enhanced_message = f"""

# Detailed NKI API Reference for {kernel_name.upper()} Kernels

The following is detailed documentation for NKI APIs relevant to {kernel_name} optimization:

{isa_docs}
""" + current_message
    
    if "prompt" not in config:
        config["prompt"] = {}
    config["prompt"]["system_message"] = enhanced_message
    
    # Ensure API key uses environment variable syntax (preserved through yaml dump/load)
    if "llm" in config:
        # Keep ${OPENAI_API_KEY} syntax so it gets resolved by OpenEvolve's config loader
        if "api_key" not in config["llm"] or config["llm"]["api_key"] is None:
            config["llm"]["api_key"] = "${OPENAI_API_KEY}"
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenEvolve for Trainium NKI kernel optimization"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="gemm",
        choices=["gemm", "softmax", "attention", "layernorm", "rmsnorm", "mamba", "transpose"],
        help="Kernel type for ISA documentation (default: gemm)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evolution results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to base config file (default: config.yaml in script directory)"
    )
    parser.add_argument(
        "--initial-program",
        type=str,
        default=None,
        help="Path to initial program (default: initial_program.py in script directory)"
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default=None,
        help="Path to evaluator (default: evaluator.py in script directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the enhanced config instead of running OpenEvolve"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    config_path = Path(args.config) if args.config else SCRIPT_DIR / "config.yaml"
    initial_program = Path(args.initial_program) if args.initial_program else SCRIPT_DIR / "initial_program.py"
    evaluator = Path(args.evaluator) if args.evaluator else SCRIPT_DIR / "evaluator.py"
    output_dir = args.output_dir or f"outputs/trainium_{args.kernel}"
    
    # Load and enhance config
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    print(f"Injecting ISA documentation for kernel type: {args.kernel}")
    config = inject_isa_docs(config, args.kernel)
    
    if args.dry_run:
        print("\n" + "=" * 60)
        print("Enhanced config (dry run):")
        print("=" * 60)
        print(yaml.dump(config, default_flow_style=False))
        return
    
    # Save the prompt/system message for debugging
    prompt_save_path = Path(output_dir) / "system_prompt.txt"
    prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_save_path, 'w') as f:
        f.write("# System Prompt for OpenEvolve Trainium Kernel Optimization\n")
        f.write(f"# Kernel type: {args.kernel}\n")
        f.write(f"# Generated at: {__import__('datetime').datetime.now().isoformat()}\n")
        f.write("#" + "=" * 70 + "\n\n")
        f.write(config.get("prompt", {}).get("system_message", ""))
    print(f"System prompt saved to: {prompt_save_path}")
    
    # Write enhanced config to temp file
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.yaml', 
        delete=False,
        prefix='trainium_config_'
    ) as f:
        yaml.dump(config, f, default_flow_style=False)
        temp_config_path = f.name
    
    try:
        print(f"Enhanced config written to: {temp_config_path}")
        print(f"Initial program: {initial_program}")
        print(f"Evaluator: {evaluator}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Import and run OpenEvolve
        try:
            from openevolve import OpenEvolve
            from openevolve.config import load_config as load_openevolve_config
        except ImportError:
            print("Error: OpenEvolve not found. Install it with:")
            print("  pip install -e /path/to/openevolve")
            sys.exit(1)
        
        # Load the enhanced config
        oe_config = load_openevolve_config(temp_config_path)
        
        # Create and run OpenEvolve
        print("Starting OpenEvolve evolution...")
        print("=" * 60)
        
        import asyncio
        
        async def run_evolution():
            evolve = OpenEvolve(
                initial_program_path=str(initial_program),
                evaluation_file=str(evaluator),
                config=oe_config,
                output_dir=output_dir,
            )
            await evolve.run()
        
        asyncio.run(run_evolution())
        
    finally:
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
            print(f"\nCleaned up temp config: {temp_config_path}")


if __name__ == "__main__":
    main()
