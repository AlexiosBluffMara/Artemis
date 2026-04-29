# Gemma 4 Custom Pipeline (Alexios Bluff Mara)

Fork of NousResearch/hermes-agent, customized for Alexios Bluff Mara's dual-brain AI setup:
- **Mac Mini (M4 Pro, 24GB)**: Fast, accurate local inference via Ollama + llama.cpp
- **RTX 5090 (Chicago)**: Unsloth fine-tuning, custom quant conversion, LoRA merging

## Architecture

```
+----------------------------------+     +----------------------------------+
|  Mac Mini (M4 Pro, 24GB)         |<--->|  RTX 5090 (Unsloth Server)      |
|  Ollama + llama.cpp              |VPN  |  Fine-tuning + LoRA merging    |
|  Gemma 4 E4B Q4_K_M (~9.6GB)     |     |  Custom GGUF export pipeline   |
+----------------------------------+     +----------------------------------+
         |                                          |
         v                                          v
  +------------------+                    +------------------+
  | Nous Portal      |                    | Facebook Export  |
  | Initial Responder|                    | Custom Persona   |
  +------------------+                    +------------------+
```

## Quick Start

### Mac Mini (Inference)
```bash
cd gemma4-pipeline/mac-mini
./setup.sh          # Install Ollama, pull models
./test.sh           # Validate inference speed & coherence
```

### RTX 5090 (Training)
```bash
cd gemma4-pipeline/rtx-5090
pip install -r requirements.txt
python train.py --config configs/facebook-persona.yaml
```

## Model Targets

| Model | Size | Quant | Use |
|-------|------|-------|-----|
| gemma-4-E2B-it | 2.5B active | Q4_K_M (~6.2GB) | Fast fallback |
| gemma-4-E4B-it | 4.0B active | Q4_K_M (~9.6GB) | **Primary Mac Mini** |
| gemma-4-E4B-it | 4.0B active | Q8_0 (~11GB) | High-quality nightly |
| Custom Gemma | 4.0B active | Q4_K_M (~9.6GB) | Facebook-trained persona |

## Directory Layout

```
mac-mini/           Ollama configs, Modelfiles, inference scripts
rtx-5090/           Unsloth training scripts, conversion pipeline
data-prep/          Facebook message export, preprocessing, dataset curation
configs/            YAML configs for training, inference, system prompts
docs/               Architecture decisions, benchmark results
```

## Status

- [x] Ollama Q4_K_M inference on M4 Pro working (27 tok/s)
- [ ] Custom TQ2_0 quant (blocked by llama.cpp ARM stability)
- [ ] Unsloth fine-tuning pipeline on RTX 5090
- [ ] Facebook message persona dataset
- [ ] Custom GGUF export + Ollama import

## Maintaining Upstream Sync

```bash
# Keep upstream main in sync
git fetch upstream
git merge upstream/main
# Our files in gemma4-pipeline/ never conflict with upstream
```

## License

MIT (same as upstream Hermes Agent)
