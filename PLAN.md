# SatQuery AI — Full Execution Plan
> ISRO/SAC Problem Statement 26167  
> Last updated: 2026-09-28

---

## Overview

| Phase | Title | Machine | Status | Est. Time |
|-------|-------|---------|--------|-----------|
| 0 | Audit | Mac | ✅ Done | — |
| 1 | Skeleton + Baseline VLM + UI | Mac | ✅ Done | — |
| 2 | VLM Fine-tuning (LoRA) | GPU Laptop | ⏳ Next | 5–6 h |
| 3 | Change Detection Pipeline | GPU Laptop | ⏳ | 2–3 h |
| 4 | SAR + Optical Fusion Pipeline | GPU Laptop | ⏳ | 3–4 h |
| 5 | Full Integration Test | Mac | ⏳ | 1 h |
| 6 | Evaluation + Polish | Mac + GPU | ⏳ | 1–2 h |

**Total GPU time: ~12–15 h** (run overnight)

---

## Phase 0 — Audit ✅ Done

Decisions recorded in `phase0_audit.md`.

| Dataset | Decision |
|---------|----------|
| BigEarthNet S1+S2 v2.0 | 10% subset via `torchgeo` |
| RSVQA-LR | Full dataset (770k QA pairs) |
| RSVQA-HR | **Skipped** — too large for session |
| VRSBench | 10k sample cap from HuggingFace |
| LEVIR-CD+ | Full dataset (~985 pairs) via HuggingFace |
| OSCD | Secondary eval only |
| CDVQA | TEOChatlas HF variant |

| Model | Role | Reason |
|-------|------|--------|
| Florence-2-base | Local VLM (Mac dev) | Fast MPS, native grounding |
| Qwen2-VL-2B-Instruct | Full VLM (GPU) | Dynamic resolution, best RS perf |
| Siamese U-Net (EfficientNet-B0) | Change map | ~8M params, 2–3h on GPU |
| Dual-stream ResNet-18 | SAR+Optical fusion | ~11M params, BigEarthNet |

---

## Phase 1 — Skeleton + Baseline ✅ Done

**Measured metrics:**

| Metric | Value |
|--------|-------|
| Unit tests | **29 / 29 pass** |
| Routing accuracy | **98.2%** (55 / 56 queries) |
| Frontend | Loads with 0 errors at localhost:5173 |

**Key files built:**

```
agent/validators/input_validator.py
agent/controller/router.py
models/registry/tool_registry.py
agent/tools/vlm_tool.py
agent/tools/change_tool.py
agent/tools/sar_optical_tool.py
agent/tools/image_preprocessor.py
agent/tools/executor.py
api/main.py
api/report_generator.py
training/train_vlm_lora.py
training/train_change.py
training/train_fusion.py
eval/eval_vqa.py
eval/routing_test_queries.json
tests/test_validators.py
ui/  (React + Vite)
```

---

## Phase 2 — VLM Fine-tuning (GPU Laptop)

**Goal:** Fine-tune Qwen2-VL-2B-Instruct with QLoRA on RSVQA-LR + VRSBench.

### Prerequisites
- NVIDIA GPU, ≥ 12 GB VRAM (RTX 3080 / 4070 or better)
- CUDA 11.8 or 12.1
- Linux or Windows with WSL2

### 2.1 — Environment

```bash
conda create -n satquery-gpu python=3.11 -y
conda activate satquery-gpu

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.0 peft==0.12.0 datasets accelerate trl bitsandbytes
conda install -c conda-forge rasterio gdal pyproj -y
pip install timm einops pillow numpy scikit-learn pandas reportlab fastapi uvicorn
```

### 2.2 — Get data

**RSVQA-LR (manual):**
1. Register free at https://rsvqa.sylvainlobry.com/
2. Download "Low Resolution" package
3. Extract to `./data/RSVQA-LR/`

```
data/RSVQA-LR/
  Images_LR/
  QA_LR_train.json
  QA_LR_val.json
  QA_LR_test.json
```

**VRSBench (automatic):**
```bash
python data/scripts/download_datasets.py --dataset vrsbench --out ./data
```

### 2.3 — Train

```bash
python training/train_vlm_lora.py \
  --model-id      Qwen/Qwen2-VL-2B-Instruct \
  --rsvqa-root    ./data/RSVQA-LR \
  --vrsbench-root ./data/VRSBench \
  --out-dir       ./checkpoints/vlm_adapter \
  --epochs        3 \
  --batch-size    1 \
  --grad-accum    8 \
  --lr            2e-4 \
  --lora-r        16 \
  --use-4bit
```

| Setting | Value |
|---------|-------|
| LoRA rank | 16 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Effective batch size | 8 (1 × accum 8) |
| Gradient checkpointing | On |
| Attention | eager (no flash_attn) |

**Expected runtime:**

| GPU | Time |
|-----|------|
| RTX 3080 (10 GB) | ~6 h |
| RTX 4090 (24 GB) | ~3 h |
| A100 (40 GB) | ~2 h |

**Output:**
```
./checkpoints/vlm_adapter/
  adapter_epoch0/
  adapter_epoch1/
  adapter_epoch2/
  adapter_final/    ← use this
  training_log.json
```

### 2.4 — Copy to Mac

```bash
scp -r user@gpu-laptop:/path/to/checkpoints/vlm_adapter/adapter_final \
    ./checkpoints/vlm_adapter/

export SATQUERY_ADAPTER_PATH=./checkpoints/vlm_adapter/adapter_final
```

---

## Phase 3 — Change Detection (GPU Laptop)

**Goal:** Train Siamese U-Net on LEVIR-CD for binary change masks.

### 3.1 — Get data

```bash
python data/scripts/download_datasets.py --dataset levir-cd --out ./data
# Auto-downloads from HuggingFace blanchon/LEVIR_CDPlus (~2.5 GB)
```

```
data/LEVIR-CD/
  train/ A/ B/ label/   ← 636 pairs
  val/   A/ B/ label/   ← 99 pairs
  test/  A/ B/ label/   ← 250 pairs
```

### 3.2 — Train

```bash
python training/train_change.py \
  --data-root  ./data/LEVIR-CD \
  --out-dir    ./checkpoints \
  --img-size   256 \
  --batch-size 8 \
  --epochs     50 \
  --lr         1e-4 \
  --seed       42
```

**Expected metrics (published range for this architecture):**

| Metric | Expected |
|--------|----------|
| val F1 | 0.75 – 0.83 |
| val IoU | 0.64 – 0.72 |

**Expected runtime:**

| GPU | Time |
|-----|------|
| RTX 3080 | ~2.5 h |
| RTX 4090 | ~1.5 h |

**Output:**
```
./checkpoints/
  change_model_best.pt
  change_model_epoch010.pt  (periodic)
  change_training_log.json
```

### 3.3 — Copy to Mac

```bash
scp user@gpu-laptop:/path/to/checkpoints/change_model_best.pt ./checkpoints/
export SATQUERY_CHANGE_WEIGHTS=./checkpoints/change_model_best.pt

# Verify
python -c "
import torch
ckpt = torch.load('./checkpoints/change_model_best.pt', map_location='cpu')
print('best F1:', ckpt.get('best_f1'))
"
```

---

## Phase 4 — SAR + Optical Fusion (GPU Laptop)

**Goal:** Train Dual-stream ResNet-18 on 10% of BigEarthNet S1+S2.

### 4.1 — Install torchgeo

```bash
pip install torchgeo==0.6.0
# torchgeo auto-downloads BigEarthNet on first run (~6 GB for 10% subset)
```

### 4.2 — Train

```bash
python training/train_fusion.py \
  --data-root   ./data/BigEarthNet \
  --out-dir     ./checkpoints \
  --img-size    120 \
  --batch-size  64 \
  --epochs      30 \
  --lr          3e-4 \
  --subset-frac 0.10 \
  --seed        42
```

**Expected metrics (10% subset, 19-class multilabel):**

| Model | Expected macro-F1 |
|-------|------------------|
| Optical-only baseline | 0.48 – 0.58 |
| SAR-only baseline | 0.38 – 0.46 |
| Fusion (optical+SAR) | 0.55 – 0.65 |

**Expected runtime:**

| GPU | Time |
|-----|------|
| RTX 3080 | ~4 h |
| RTX 4090 | ~2.5 h |

**Output:**
```
./checkpoints/
  fusion_model_best.pt
  fusion_training_log.json
```

### 4.3 — Copy to Mac

```bash
scp user@gpu-laptop:/path/to/checkpoints/fusion_model_best.pt ./checkpoints/
export SATQUERY_FUSION_WEIGHTS=./checkpoints/fusion_model_best.pt
```

---

## Phase 5 — Full Integration Test (Mac)

### 5.1 — Start backend with all adapters

```bash
export SATQUERY_RESULTS_DIR=./results
export SATQUERY_ADAPTER_PATH=./checkpoints/vlm_adapter/adapter_final
export SATQUERY_CHANGE_WEIGHTS=./checkpoints/change_model_best.pt
export SATQUERY_FUSION_WEIGHTS=./checkpoints/fusion_model_best.pt

cd "/Users/tholkappian/Projects/SatQuery Ai"
conda activate satquery
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5.2 — Start frontend

```bash
cd ui && npm run dev
# Open: http://localhost:5173
```

### 5.3 — Demo checklist

**Pipeline 1: Single-image VQA**
- Upload `data/demo/single_optical.png`, role = `single`
- Query: *"What land use is visible in this image?"*
- ✅ Verify: text answer + confidence + execution trace
- ✅ Verify: model_label = "LoRA-adapted (VRSBench+RSVQA-LR)"

**Pipeline 2: Change Detection**
- Upload `data/demo/t1.png` (role=`t1`) + `data/demo/t2.png` (role=`t2`)
- Query: *"What has changed between these two images?"*
- ✅ Verify: change mask overlay + VLM description + changed area %

**Pipeline 3: SAR + Optical Fusion**
- Upload `data/demo/single_optical.png` (role=`optical`) + `data/demo/sar_image.png` (role=`sar`)
- Query: *"Show built-up and water areas using both sensors"*
- ✅ Verify: colour overlay + class probabilities + comparison (fusion vs optical-only vs SAR-only)

**PDF Report**
- ✅ Download and verify it contains: query, answer, trace, images, confidence note

---

## Phase 6 — Evaluation + Polish (Mac + GPU)

### 6.1 — VQA accuracy (RSVQA-LR)

```bash
python eval/eval_vqa.py \
  --data-root  ./data/RSVQA-LR \
  --model-size full \
  --max-samples 1000 \
  --out ./eval/results_vqa.json
```

### 6.2 — Change metrics (from training log)

```bash
python3 -c "
import json
log = json.load(open('checkpoints/change_training_log.json'))
best = max(log, key=lambda x: x['f1'])
print(f\"Epoch {best['epoch']} | F1={best['f1']:.4f} | IoU={best['iou']:.4f}\")
"
```

### 6.3 — Fusion metrics (from training log)

```bash
python3 -c "
import json
log = json.load(open('checkpoints/fusion_training_log.json'))
best = max(log, key=lambda x: x.get('macro_f1', 0))
print(f\"Epoch {best['epoch']} | macro-F1={best['macro_f1']:.4f}\")
"
```

### 6.4 — Final results table (fill in real numbers)

| Benchmark | Metric | Value | Notes |
|-----------|--------|-------|-------|
| RSVQA-LR | Exact match | TBD | 1000-sample subset |
| RSVQA-LR | BLEU-1 | TBD | 1000-sample subset |
| LEVIR-CD | val F1 | TBD | Full val split (99 pairs) |
| LEVIR-CD | val IoU | TBD | Full val split |
| BigEarthNet 10% | macro-F1 (fusion) | TBD | ~54k samples |
| BigEarthNet 10% | macro-F1 (optical-only) | TBD | baseline comparison |
| BigEarthNet 10% | macro-F1 (SAR-only) | TBD | baseline comparison |
| Routing | Accuracy | **98.2%** | 56 hand-written queries ✅ |

---

## Overnight Training Script (GPU Laptop)

Save as `run_all_training.sh`, run before sleeping:

```bash
#!/bin/bash
set -e
cd /path/to/SatQuery\ Ai
conda activate satquery-gpu

echo "=== PHASE 2: VLM LoRA ==="
python training/train_vlm_lora.py \
  --rsvqa-root ./data/RSVQA-LR \
  --vrsbench-root ./data/VRSBench \
  --out-dir ./checkpoints/vlm_adapter \
  --epochs 3 --batch-size 1 --grad-accum 8 --use-4bit

echo "=== PHASE 3: Change Detection ==="
python data/scripts/download_datasets.py --dataset levir-cd --out ./data
python training/train_change.py \
  --data-root ./data/LEVIR-CD \
  --out-dir ./checkpoints \
  --epochs 50 --batch-size 8

echo "=== PHASE 4: SAR+Optical Fusion ==="
python training/train_fusion.py \
  --data-root ./data/BigEarthNet \
  --out-dir ./checkpoints \
  --epochs 30 --batch-size 64 --subset-frac 0.10

echo "=== ALL DONE ==="
ls -lh checkpoints/*.pt checkpoints/vlm_adapter/adapter_final/
```

```bash
chmod +x run_all_training.sh
nohup ./run_all_training.sh > training_run.log 2>&1 &
tail -f training_run.log    # watch progress
```

---

## Environment Variables Reference

| Variable | Set after | Value |
|----------|-----------|-------|
| `SATQUERY_RESULTS_DIR` | Phase 1 | `./results` |
| `SATQUERY_ADAPTER_PATH` | Phase 2 | `./checkpoints/vlm_adapter/adapter_final` |
| `SATQUERY_CHANGE_WEIGHTS` | Phase 3 | `./checkpoints/change_model_best.pt` |
| `SATQUERY_FUSION_WEIGHTS` | Phase 4 | `./checkpoints/fusion_model_best.pt` |
| `SATQUERY_LLM_BACKEND` | Optional | `none` / `ollama` / `openai` |

---

## Deliverables Checklist

```
✅  phase0_audit.md                     — dataset + model audit
✅  agent/ api/ models/ training/ eval/ — full codebase
✅  ui/                                 — React frontend (verified in browser)
✅  tests/test_validators.py            — 29/29 tests pass
✅  eval/routing_test_queries.json      — 56 queries, 98.2% accuracy
✅  data/demo/                          — demo images for testing
✅  README.md                           — setup + demo guide
✅  PLAN.md                             — this file
⏳  checkpoints/vlm_adapter/            — after Phase 2
⏳  checkpoints/change_model_best.pt    — after Phase 3
⏳  checkpoints/fusion_model_best.pt    — after Phase 4
⏳  eval/results_final.json             — after Phase 6
```

---

## Known Limitations

1. **Domain shift** — Trained on Sentinel-2/S1; test domain is Cartosat-2S + RISAT. Expect 10–20% accuracy drop. Mitigated by per-image normalization.
2. **RSVQA-HR skipped** — LR used (770k QA pairs; sufficient for adaptation).
3. **BigEarthNet 10% subset** — Full dataset fusion F1 ~0.75; 10% gives ~0.55–0.65. Documented.
4. **SAR grounding** — Florence-2 not trained on SAR; grounding quality on SAR is unreliable.
5. **Multi-task queries** — "Caption and ground this image" routes to one task (single-task router limitation; the 1 routing failure in 56 queries).
