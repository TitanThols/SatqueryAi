"""
training/train_vlm_lora.py
Fine-tune Qwen2-VL-2B-Instruct with QLoRA on RSVQA-LR + VRSBench.
Standalone, seeded, resumable. Target: Kaggle/Colab A100 or T4 (16 GB).
Output: adapter/  (PEFT adapter directory, load via PeftModel)

Expected runtime:
  - T4  (16 GB): ~5-6 h with batch=1 + grad_accum=8 + gradient_checkpointing
  - A100 (40 GB): ~2-3 h with batch=4
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_CONFIG = {
    "model_id":        "Qwen/Qwen2-VL-2B-Instruct",
    "rsvqa_root":      "./data/RSVQA-LR",
    "vrsbench_root":   "./data/VRSBench",
    "out_dir":         "./checkpoints/vlm_adapter",
    "seed":            42,
    "lora_r":          16,
    "lora_alpha":      32,
    "lora_dropout":    0.1,
    "target_modules":  ["q_proj", "v_proj", "k_proj", "o_proj"],
    "batch_size":      1,
    "grad_accum":      8,
    "lr":              2e-4,
    "epochs":          3,
    "max_seq_len":     512,
    "use_4bit":        True,    # QLoRA; set False if A100 with plenty VRAM
    "resume_from":     None,
    "val_every_steps": 200,
    "max_train_samples": 50000,  # cap for feasibility
}


# ── Dataset ────────────────────────────────────────────────────────────────────

class RSVQALRDataset(torch.utils.data.Dataset):
    """
    RSVQA-LR format:
      rsvqa_root/
        Images_LR/  (PNG tiles)
        QA_LR_train.json (list of {image_id, question, answer, type})
        QA_LR_val.json
        Images_LR_info.json
    """

    def __init__(self, root: Path, split: str, max_samples: int = 50000):
        self.img_dir = root / "Images_LR"
        qa_file = root / f"QA_LR_{split}.json"
        if not qa_file.exists():
            raise FileNotFoundError(
                f"RSVQA-LR QA file not found: {qa_file}\n"
                f"Download from: https://rsvqa.sylvainlobry.com/"
            )
        qa_data = json.loads(qa_file.read_text())
        # qa_data is typically {questions: [...], answers: [...]}
        questions = qa_data.get("questions", qa_data if isinstance(qa_data, list) else [])
        answers   = qa_data.get("answers", [])

        self.samples = []
        for i, q in enumerate(questions[:max_samples]):
            ans = answers[i]["answer"] if i < len(answers) else "unknown"
            img_id = q.get("img_id", q.get("image_id", ""))
            self.samples.append({
                "image_path": self.img_dir / f"{img_id}.png",
                "question":   q.get("question", ""),
                "answer":     str(ans),
            })
        logger.info("RSVQA-LR %s: %d samples", split, len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class VRSBenchDataset(torch.utils.data.Dataset):
    """
    VRSBench format (HuggingFace xiang709/VRSBench or local):
      vrsbench_root/
        images/
        QA.json  (list of {image, question, answer})
    """

    def __init__(self, root: Path, split: str, max_samples: int = 20000):
        self.img_dir = root / "images"
        qa_file = root / f"QA_{split}.json"
        if not qa_file.exists():
            # Try HuggingFace download
            logger.warning("VRSBench local not found at %s. Will skip.", qa_file)
            self.samples = []
            return

        qa_data = json.loads(qa_file.read_text())
        self.samples = []
        for item in (qa_data if isinstance(qa_data, list) else qa_data.get("data", []))[:max_samples]:
            self.samples.append({
                "image_path": self.img_dir / item.get("image", ""),
                "question":   item.get("question", ""),
                "answer":     str(item.get("answer", "")),
            })
        logger.info("VRSBench %s: %d samples", split, len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_vqa(samples):
    return samples


# ── Training ──────────────────────────────────────────────────────────────────

def build_prompt(question: str, answer: str, processor) -> dict:
    """Build Qwen2-VL instruction-tuning conversation."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert remote sensing analyst. "
                "Answer the question about the satellite image concisely and accurately."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text


def train(cfg: dict):
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if device.type != "cuda" and cfg["use_4bit"]:
        logger.warning("4-bit quantization requires CUDA. Disabling.")
        cfg["use_4bit"] = False

    # ── Model & processor ─────────────────────────────────────────────────────
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType

    logger.info("Loading base model: %s", cfg["model_id"])
    processor = AutoProcessor.from_pretrained(cfg["model_id"])

    if cfg["use_4bit"]:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg["model_id"],
            quantization_config=bnb_cfg,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )

    # Gradient checkpointing for memory saving
    model.gradient_checkpointing_enable()

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_datasets = []
    val_datasets   = []

    rsvqa_path = Path(cfg["rsvqa_root"])
    if rsvqa_path.exists():
        train_datasets.append(RSVQALRDataset(rsvqa_path, "train", cfg["max_train_samples"]))
        val_datasets.append(RSVQALRDataset(rsvqa_path, "val", 2000))

    vrs_path = Path(cfg["vrsbench_root"])
    if vrs_path.exists():
        train_datasets.append(VRSBenchDataset(vrs_path, "train", 20000))
        val_datasets.append(VRSBenchDataset(vrs_path, "val", 1000))

    if not train_datasets:
        logger.error(
            "No training data found!\n"
            "  RSVQA-LR: download from https://rsvqa.sylvainlobry.com/\n"
            "  VRSBench: pip install datasets && python data/scripts/download_vrsbench.py\n"
        )
        sys.exit(1)

    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset(train_datasets)
    val_ds   = ConcatDataset(val_datasets) if val_datasets else None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_vqa, num_workers=2,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    from transformers import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=0.01,
    )
    total_steps = len(train_loader) * cfg["epochs"] // cfg["grad_accum"]
    scheduler   = get_cosine_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step   = 0
    optimizer.zero_grad()

    log_entries = []

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            losses = []
            for item in batch:
                img_path = Path(item["image_path"])
                if not img_path.exists():
                    continue
                try:
                    from PIL import Image
                    image = Image.open(img_path).convert("RGB").resize((448, 448))
                except Exception:
                    continue

                text = build_prompt(item["question"], item["answer"], processor)
                inputs = processor(
                    text=[text], images=[image],
                    return_tensors="pt", padding=True,
                    max_length=cfg["max_seq_len"], truncation=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, labels=inputs["input_ids"])
                losses.append(outputs.loss)

            if not losses:
                continue

            loss = torch.stack(losses).mean() / cfg["grad_accum"]
            loss.backward()
            epoch_loss += loss.item() * cfg["grad_accum"]

            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    logger.info("Epoch %d Step %d loss=%.4f lr=%.2e",
                                epoch, global_step, loss.item() * cfg["grad_accum"],
                                scheduler.get_last_lr()[0])

        avg_loss = epoch_loss / len(train_loader)
        log_entries.append({"epoch": epoch, "train_loss": avg_loss})
        logger.info("Epoch %d complete. avg_train_loss=%.4f", epoch, avg_loss)

        # Save adapter
        adapter_ckpt = out_dir / f"adapter_epoch{epoch}"
        model.save_pretrained(str(adapter_ckpt))
        processor.save_pretrained(str(adapter_ckpt))
        logger.info("Adapter saved: %s", adapter_ckpt)

    # Save final
    final_path = out_dir / "adapter_final"
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    (out_dir / "training_log.json").write_text(json.dumps(log_entries, indent=2))

    logger.info(
        "\n\n══ VLM FINE-TUNING COMPLETE ══\n"
        "Adapter saved to: %s\n"
        "Copy to Mac:  scp -r user@gpu-box:%s ./checkpoints/vlm_adapter/\n"
        "Then set:     export SATQUERY_ADAPTER_PATH=./checkpoints/vlm_adapter/adapter_final\n"
        "              export SATQUERY_VLM_SIZE=full\n",
        final_path, final_path,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in DEFAULT_CONFIG.items():
        t = type(v) if v is not None else str
        if t == list:
            p.add_argument(f"--{k.replace('_','-')}", nargs="+", default=v)
        elif t == bool:
            p.add_argument(f"--{k.replace('_','-')}", action="store_true", default=v)
        else:
            p.add_argument(f"--{k.replace('_','-')}", type=t, default=v)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        cfg.update(json.loads(Path(args.config).read_text()))
    for k in DEFAULT_CONFIG:
        v = getattr(args, k.replace("-", "_"), None)
        if v is not None:
            cfg[k] = v

    train(cfg)
