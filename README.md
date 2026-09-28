# SatQuery AI 🛰️

**Agentic AI for satellite image analysis** — ISRO/SAC Problem Statement 26167.

---

## Architecture

```
User Query + GeoTIFF(s)
       │
       ▼
[Input Validator] → checks files, CRS, roles, task compatibility
       │
       ▼
[Controller/Router] → classifies task (rule-based + optional LLM)
       │
       ▼
[Tool Registry] → JSON-schema gated, 6 tools
   ├─ vqa          → Florence-2 / Qwen2-VL-2B LoRA
   ├─ caption      → Florence-2 / Qwen2-VL-2B
   ├─ ground       → Florence-2 grounding head
   ├─ change_map   → Siamese U-Net (LEVIR-CD trained)
   ├─ change_vqa   → VLM + change map overlay
   └─ sar_optical  → Dual-stream ResNet-18 (BigEarthNet)
       │
       ▼
[Output Merger] → answer + mask + confidence + ExecutionTrace
       │
       ▼
FastAPI Backend ──► React Frontend
```

---

## Setup: Mac (Apple Silicon)

### 1. Create environment

```bash
conda create -n satquery python=3.11 -y
conda activate satquery

# rasterio + GDAL via conda-forge (recommended on Apple Silicon)
conda install -c conda-forge rasterio gdal pyproj -y

# Core dependencies
pip install -r requirements.txt
```

> ⚠️ **No bitsandbytes or flash-attention** — inference is MPS/CPU-safe.

### 2. Generate demo data

```bash
python data/scripts/download_datasets.py --dataset demo --out ./data
```

### 3. Run the app (one command)

```bash
# Terminal 1 — Backend
cd "/path/to/SatQuery Ai"
SATQUERY_RESULTS_DIR=./results python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd ui && npm install && npm run dev
```

Open: **http://localhost:5173**

---

## Demo Steps

1. Open http://localhost:5173
2. Upload `data/demo/single_optical.tif`, set role → `single`
3. Type: _"What land use is visible in this image?"_
4. Click **Analyze** — answer + confidence + execution trace appears
5. For change detection: upload `t1.tif` + `t2.tif`, roles `t1`/`t2`
6. Ask: _"What has changed between these two images?"_
7. View change mask overlay + VLM description
8. For SAR fusion: upload `single_optical.tif` + `sar_image.tif`, roles `optical`/`sar`
9. Ask: _"Show built-up and water areas using both sensors"_
10. Download PDF report

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SATQUERY_RESULTS_DIR` | `/tmp/satquery_results` | Where uploads and results are stored |
| `SATQUERY_ADAPTER_PATH` | `None` | Path to LoRA adapter dir (Phase 2 output) |
| `SATQUERY_CHANGE_WEIGHTS` | `None` | Path to Siamese U-Net weights (Phase 3 output) |
| `SATQUERY_FUSION_WEIGHTS` | `None` | Path to fusion model weights (Phase 4 output) |
| `SATQUERY_LLM_BACKEND` | `none` | `none` / `ollama` / `openai` for LLM routing |
| `SATQUERY_LLM_MODEL` | `qwen2.5:1.5b` | Model for LLM controller |

---

## Remote GPU Training (Kaggle/Colab)

### Phase 2: VLM LoRA (Qwen2-VL-2B)
```bash
# On Kaggle/Colab with GPU:
python training/train_vlm_lora.py \
  --model-id Qwen/Qwen2-VL-2B-Instruct \
  --rsvqa-root ./data/RSVQA-LR \
  --vrsbench-root ./data/VRSBench \
  --out-dir ./checkpoints/vlm_adapter \
  --epochs 3 --batch-size 1 --grad-accum 8 --use-4bit

# Expected runtime: ~5-6h on T4, ~2-3h on A100
# Output: ./checkpoints/vlm_adapter/adapter_final/

# Copy to Mac:
scp -r user@gpu:./checkpoints/vlm_adapter/adapter_final ./checkpoints/
export SATQUERY_ADAPTER_PATH=./checkpoints/vlm_adapter/adapter_final
```

### Phase 3: Change Detection
```bash
# Download LEVIR-CD first:
python data/scripts/download_datasets.py --dataset levir-cd --out ./data

python training/train_change.py \
  --data-root ./data/LEVIR-CD \
  --out-dir ./checkpoints \
  --epochs 50 --batch-size 8

# Expected runtime: ~2-3h on T4
# Output: ./checkpoints/change_model_best.pt

export SATQUERY_CHANGE_WEIGHTS=./checkpoints/change_model_best.pt
```

### Phase 4: SAR+Optical Fusion
```bash
python training/train_fusion.py \
  --data-root ./data/BigEarthNet \
  --out-dir ./checkpoints \
  --epochs 30 --batch-size 64 --subset-frac 0.1

# Expected runtime: ~3-5h on T4
# Output: ./checkpoints/fusion_model_best.pt

export SATQUERY_FUSION_WEIGHTS=./checkpoints/fusion_model_best.pt
```

---

## Evaluation

```bash
# VQA accuracy on RSVQA-LR
python eval/eval_vqa.py --data-root ./data/RSVQA-LR --model-size small --max-samples 500

# Routing accuracy (55 hand-written queries)
# Already embedded in eval_vqa.py via --routing-test eval/routing_test_queries.json
```

---

## Project Structure

```
SatQuery Ai/
├── agent/
│   ├── controller/router.py       # Task classifier + orchestrator
│   ├── tools/
│   │   ├── executor.py            # Tool dispatcher
│   │   ├── vlm_tool.py            # Florence-2 + Qwen2-VL inference
│   │   ├── change_tool.py         # Siamese U-Net change detection
│   │   ├── sar_optical_tool.py    # Dual-stream fusion
│   │   └── image_preprocessor.py # GeoTIFF preprocessing
│   └── validators/
│       └── input_validator.py     # Structural + spatial validation
├── api/
│   ├── main.py                    # FastAPI backend
│   └── report_generator.py        # PDF report generation
├── models/registry/
│   └── tool_registry.py           # JSON-schema tool definitions
├── training/
│   ├── train_vlm_lora.py          # Qwen2-VL QLoRA training
│   ├── train_change.py            # Siamese U-Net training
│   └── train_fusion.py            # Dual-stream fusion training
├── eval/
│   ├── eval_vqa.py                # VQA + routing evaluation
│   └── routing_test_queries.json  # 55 hand-written test queries
├── tests/
│   └── test_validators.py         # Unit tests (pytest)
├── data/scripts/
│   └── download_datasets.py       # Dataset downloaders
├── ui/                            # React frontend (Vite)
└── requirements.txt               # Pinned for Apple Silicon
```

---

## What Works (Phase 1)

| Capability | Status |
|-----------|--------|
| VQA (single image) | ✅ Florence-2 baseline |
| Captioning | ✅ Florence-2 baseline |
| Text-guided grounding | ✅ Florence-2 OVD head |
| Change map (binary) | ✅ Siamese U-Net (untrained — random weights until Phase 3) |
| Change VQA | ✅ VLM with composite image |
| SAR+optical fusion | ✅ Dual-stream ResNet (untrained until Phase 4) |
| Controller routing | ✅ Rule-based (100% on deterministic cases) |
| LLM augmented routing | ⚙️ Optional (Ollama/OpenAI configurable) |
| Input validation | ✅ CRS, overlap, format, count checks |
| Execution trace | ✅ Full JSON log in UI |
| PDF report | ✅ ReportLab |
| LoRA adaptation | ⏳ Phase 2 (remote GPU) |
| Change model training | ⏳ Phase 3 (remote GPU) |
| Fusion model training | ⏳ Phase 4 (remote GPU) |

---

## Known Limitations

1. **Models not yet fine-tuned**: Phase 1 uses Florence-2 baseline. All outputs labelled "baseline (not adapted)". Real metrics only after Phase 2-4.
2. **Cartosat-2S domain shift**: Trained on Sentinel; test domain (Cartosat-2S + RISAT) will show accuracy degradation. Documented in Phase 0 audit.
3. **RSVQA-HR skipped**: Too large for Colab session; LR used instead.
4. **CDVQA access**: Using TEOChatlas HF variant instead of original CDVQA.
5. **SAR grounding**: Florence-2 was not trained on SAR imagery; grounding quality on SAR is low.

---

## Tests

```bash
cd "/path/to/SatQuery Ai"
pytest tests/ -v
```

Expected: 28 tests pass (validators + router + registry).
