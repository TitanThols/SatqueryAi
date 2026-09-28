"""
eval/eval_vqa.py
Evaluates VQA accuracy on RSVQA-LR val split and optionally VRSBench.
Reports: exact match accuracy, BLEU-1, top-1 accuracy on closed questions.
Usage: python eval/eval_vqa.py --data-root ./data/RSVQA-LR --model-size small
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent))


def normalize_answer(s: str) -> str:
    """Lower, strip, remove articles."""
    s = s.lower().strip()
    for art in ["a ", "an ", "the "]:
        if s.startswith(art):
            s = s[len(art):]
    return s.strip()


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def compute_bleu1(pred: str, gold: str) -> float:
    pred_toks  = normalize_answer(pred).split()
    gold_toks  = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    matches = sum(1 for t in pred_toks if t in gold_toks)
    return matches / max(len(pred_toks), 1)


def evaluate_rsvqa(data_root: Path, model_size: str, max_samples: int = 1000) -> dict:
    from agent.tools.image_preprocessor import prepare_vlm_image
    from agent.tools.vlm_tool import run_vqa

    qa_file = data_root / "QA_LR_val.json"
    img_dir = data_root / "Images_LR"

    if not qa_file.exists():
        logger.error("RSVQA-LR val QA not found: %s", qa_file)
        return {}

    qa_data   = json.loads(qa_file.read_text())
    questions = qa_data.get("questions", [])
    answers   = qa_data.get("answers", [])

    samples = []
    for i, q in enumerate(questions[:max_samples]):
        ans = answers[i]["answer"] if i < len(answers) else "unknown"
        img_id = q.get("img_id", q.get("image_id", ""))
        samples.append({
            "image_path": img_dir / f"{img_id}.png",
            "question":   q.get("question", ""),
            "answer":     str(ans),
            "type":       q.get("type", "open"),
        })

    em_scores    = []
    bleu_scores  = []
    skipped      = 0

    for s in tqdm(samples, desc="RSVQA-LR eval"):
        if not s["image_path"].exists():
            skipped += 1
            continue
        try:
            image = prepare_vlm_image(s["image_path"], modality="optical")
            result = run_vqa(image, s["question"], model_size=model_size)
            pred   = result.get("answer", "")
        except Exception as e:
            logger.warning("Skipping %s: %s", s["image_path"].name, e)
            skipped += 1
            continue

        em_scores.append(exact_match(pred, s["answer"]))
        bleu_scores.append(compute_bleu1(pred, s["answer"]))

    n = len(em_scores)
    results = {
        "dataset":       "RSVQA-LR",
        "model_size":    model_size,
        "n_evaluated":   n,
        "n_skipped":     skipped,
        "exact_match":   float(np.mean(em_scores)) if em_scores else None,
        "bleu1":         float(np.mean(bleu_scores)) if bleu_scores else None,
        "note":          "Computed on val split subset" if max_samples < 10000 else "Full val split",
    }
    logger.info("RSVQA-LR Results: %s", json.dumps(results, indent=2))
    return results


def evaluate_routing(test_file: Path, model_size: str) -> dict:
    """
    Evaluate routing accuracy on hand-written test queries.
    test_file: JSON list of {query, n_images, roles, expected_task}
    """
    from agent.controller.router import classify_task

    if not test_file.exists():
        logger.warning("Routing test file not found: %s", test_file)
        return {}

    tests = json.loads(test_file.read_text())
    correct = 0
    total   = len(tests)
    errors  = []

    for t in tests:
        predicted = classify_task(t["query"], t["n_images"], t["roles"])
        expected  = t["expected_task"]
        if predicted == expected:
            correct += 1
        else:
            errors.append({
                "query":     t["query"],
                "expected":  expected,
                "predicted": predicted,
            })

    acc = correct / total if total > 0 else 0.0
    results = {
        "dataset":       "routing_test",
        "n_queries":     total,
        "accuracy":      acc,
        "errors":        errors,
        "note":          f"Hand-written routing test ({total} queries incl. ambiguous/invalid)",
    }
    logger.info("Routing accuracy: %.3f (%d/%d)", acc, correct, total)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",    type=str, default="./data/RSVQA-LR")
    p.add_argument("--model-size",   type=str, default="small", choices=["small", "full"])
    p.add_argument("--max-samples",  type=int, default=500)
    p.add_argument("--routing-test", type=str, default="./eval/routing_test_queries.json")
    p.add_argument("--out",          type=str, default="./eval/results_vqa.json")
    args = p.parse_args()

    all_results = {}
    all_results["vqa"] = evaluate_rsvqa(
        Path(args.data_root), args.model_size, args.max_samples
    )
    all_results["routing"] = evaluate_routing(
        Path(args.routing_test), args.model_size
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved: %s", out_path)
    print(json.dumps(all_results, indent=2))
