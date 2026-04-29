# Convert merged HuggingFace model to GGUF for Ollama
# Run on Mac Mini (ARM) or any machine with llama.cpp

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to merged HF model dir")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--quant", default="Q4_K_M", choices=["Q4_K_M", "Q8_0", "TQ1_0", "TQ2_0"])
    parser.add_argument("--ctx", type=int, default=32768, help="Context length")
    args = parser.parse_args()

    # Find convert_hf_to_gguf.py
    llama_cpp_dir = Path.home() / "llama.cpp"
    if not llama_cpp_dir.exists():
        print("Cloning llama.cpp...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(llama_cpp_dir)], check=True)

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print("ERROR: convert_hf_to_gguf.py not found")
        sys.exit(1)

    # Install deps if needed
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(llama_cpp_dir / "requirements.txt")], check=False)

    # Run conversion
    # NOTE: Gemma 4 requires transformers>=5.0.0. If not available, you may need to patch.
    cmd = [
        sys.executable, str(convert_script),
        args.input,
        "--outfile", args.output,
        "--outtype", args.quant.lower(),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"GGUF saved to: {args.output}")
    print(f"Import into Ollama with: ollama create <name> -f Modelfile")


if __name__ == "__main__":
    main()
