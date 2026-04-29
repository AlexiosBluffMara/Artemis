#!/bin/bash
set -euo pipefail

echo "=== Gemma 4 Mac Mini Setup ==="

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    brew install ollama
fi

# Pull the models we need
# Primary: gemma4:e4b (Q4_K_M, ~9.6GB, 4B active params)
echo "Pulling gemma4:e4b (primary)..."
ollama pull gemma4:e4b

# Fallback: gemma4:e2b (Q4_K_M, ~7.2GB, 2.5B active params)
echo "Pulling gemma4:e2b (fallback)..."
ollama pull gemma4:e2b

# Optional: Q8 for high-quality tasks
echo "Pulling gemma4:e4b-it-q8_0 (optional quality)..."
ollama pull gemma4:e4b-it-q8_0 || echo "(skipped)"

echo ""
echo "Models available:"
ollama list | grep gemma

echo ""
echo "Testing inference..."
ollama run gemma4:e4b --verbose << 'EOF'
Answer in one word: What is the capital of France?
EOF

echo "=== Setup complete ==="
