# RTX 5090 Training Pipeline

## Hardware Specs
- GPU: NVIDIA RTX 5090 (32GB VRAM)
- Required: torch 2.11.0+cu129 (Blackwell/Gemma 4 fix)
- OS: Ubuntu 22.04+ recommended

## Pre-Flight Checklist

```bash
# 1. Verify GPU
nvidia-smi | grep "RTX 5090"

# 2. Clear stale Triton cache (Unsloth Blackwell bug)
rm -rf ~/.triton/cache

# 3. Install correct PyTorch for Blackwell
pip install torch==2.11.0+cu129 --index-url https://download.pytorch.org/whl/cu129

# 4. Install Unsloth
pip install unsloth

# 5. Verify Gemma 4 loads
python -c "from unsloth import FastModel; FastModel.from_pretrained('unsloth/gemma-4-E4B-it')"
```

## Training Script

See `train.py` for full implementation. Quick run:

```bash
python train.py \
  --base_model unsloth/gemma-4-E4B-it \
  --dataset data-prep/facebook-chat.jsonl \
  --output_dir ./outputs/gemma4-facebook-persona \
  --epochs 3 \
  --batch_size 2 \
  --lr 2e-4 \
  --lora_r 16 \
  --lora_alpha 16
```

## Post-Training Pipeline

```bash
# 1. Export merged model (LoRA + base -> single weights)
python export_merged.py --checkpoint ./outputs/gemma4-facebook-persona/final --output ./merged

# 2. Convert to GGUF (CPU, run on Mac Mini or server)
cd mac-mini
python convert_to_gguf.py --input ./merged --output ./gemma4-custom.gguf --quant Q4_K_M

# 3. Create Ollama Modelfile
FROM ./gemma4-custom.gguf
SYSTEM """
You are Alexios Bluff Mara's AI persona, trained on Facebook messages.
You speak with warmth, humor, and precision.
"""

# 4. Import and serve
ollama create alexios-gemma4:q4km -f Modelfile.custom
ollama run alexios-gemma4:q4km
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUBLAS_STATUS_EXECUTION_FAILED` | `pip install torch==2.11.0+cu129; rm -rf ~/.triton/cache` |
| OOM during training | Reduce `max_seq_length` or `batch_size` |
| Slow training | Ensure `fast_inference=True` in Unsloth |
| GGUF corruption on ARM | Build llama.cpp from source; don't use TQ2_0 on Mac Mini |
