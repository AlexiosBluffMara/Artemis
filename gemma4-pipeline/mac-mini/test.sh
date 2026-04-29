#!/bin/bash
set -euo pipefail

echo "=== Gemma 4 Mac Mini Inference Test ==="

echo "--- Test 1: Factual accuracy ---"
ollama run gemma4:e4b --verbose << 'EOF'
What is the capital of France? Answer in one word.
EOF

echo ""
echo "--- Test 2: Reasoning ---"
ollama run gemma4:e4b --verbose << 'EOF'
If a train travels 60 miles per hour for 2.5 hours, how far does it travel? Show your work.
EOF

echo ""
echo "--- Test 3: Multilingual ---"
ollama run gemma4:e4b --verbose << 'EOF'
Translate 'Good morning' to Spanish.
EOF

echo ""
echo "--- Test 4: Coherence (longer context) ---"
ollama run gemma4:e4b --verbose << 'EOF'
Summarize the plot of Romeo and Juliet in three sentences.
EOF

echo "=== Tests complete ==="
