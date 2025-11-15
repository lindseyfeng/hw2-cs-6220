#!/usr/bin/env python3
# train_dpo.py

import argparse
import os
import csv
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from bert_score import score as bertscore
from detoxify import Detoxify


# -------------------------
# Data split (90% / 10%)
# -------------------------
def split_train_eval(jsonl_path):
    """
    Load JSONL file and split 90% / 10% into train / eval.
    Assumes each row has: prompt, chosen, rejected.
    """
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    n = len(ds)
    if n < 11:
        raise ValueError(f"Need ≥ 11 rows in {jsonl_path}, got {n}.")

    for k in ("prompt", "chosen", "rejected"):
        if k not in ds.column_names:
            raise ValueError(f"Missing column '{k}' in JSONL.")

    n_test = int(0.1 * n)
    if n_train <= 0 or n_train >= n:
        raise ValueError(f"Bad split: n={n}, n_train={n_train}")

    test_ds = ds.select(range(0, n_test))
    eval_ds = ds.select(range(n_test, n))

    print(f"Total rows: {n} | Train: {len(train_ds)} | Eval: {len(eval_ds)}")
    return train_ds, eval_ds

def generate(model, tok, prompts, max_new_tokens=128, batch_size=8, device="cuda"):
    """
    Batched deterministic generation with prompt stripping + progress countdown.
    """
    model.eval()
    model.to(device)

    outs = []
    n = len(prompts)
    num_batches = (n + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)

            batch_prompts = prompts[start:end]

            # Progress log
            print(
                f"[Generate] Batch {batch_idx+1}/{num_batches}  "
                f"(processed {end}/{n})"
            )

            # Tokenize batch
            batch_inp = tok(batch_prompts, return_tensors="pt", padding=True).to(device)

            gen = model.generate(
                **batch_inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

            # For each sequence in the batch, remove prompt prefix
            input_ids = batch_inp.input_ids
            for j in range(len(batch_prompts)):
                prompt_len = (input_ids[j] != tok.pad_token_id).sum().item()

                full = tok.decode(gen[j], skip_special_tokens=True)
                pref = tok.decode(gen[j][:prompt_len], skip_special_tokens=True)

                completion = full[len(pref):].strip() or full.strip()
                outs.append(completion)

    print(f"[Generate] Done! Generated {len(outs)} sequences.\n")
    return outs


# -------------------------
# BERTScore eval
# -------------------------
def eval_bertscore(gens, chosens, rejecteds, lang="en"):
    _, _, f_ch = bertscore(gens, chosens, lang=lang, rescale_with_baseline=False)
    _, _, f_re = bertscore(gens, rejecteds, lang=lang, rescale_with_baseline=False)

    f_ch = [float(x) for x in f_ch]
    f_re = [float(x) for x in f_re]
    delta = [c - r for c, r in zip(f_ch, f_re)]

    summary = {
        "mean_F1_vs_chosen": sum(f_ch) / len(f_ch),
        "mean_F1_vs_rejected": sum(f_re) / len(f_re),
        "mean_delta": sum(delta) / len(delta),
        "chosen_better_count": sum(d > 0 for d in delta),
        "rejected_better_count": sum(d < 0 for d in delta),
        "ties": sum(abs(d) < 1e-6 for d in delta),
    }
    return f_ch, f_re, delta, summary


# -------------------------
# Detoxify eval
# -------------------------
def eval_toxicity(original_texts, generated_texts, model_name="unbiased"):
    """
    Compute average toxicity before/after using Detoxify.
    Returns:
      avg_before, avg_after, tox_before_list, tox_after_list
    """
    toxic_detector = Detoxify(model_name)  # uses GPU if available

    toxic_scores_before = toxic_detector.predict(original_texts)
    toxic_scores_after = toxic_detector.predict(generated_texts)

    tox_before = toxic_scores_before["toxicity"]
    tox_after = toxic_scores_after["toxicity"]

    avg_before = float(sum(tox_before) / len(tox_before))
    avg_after = float(sum(tox_after) / len(tox_after))

    return avg_before, avg_after, tox_before, tox_after


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prefs_jsonl",
        required=True,
        help="JSONL with fields: prompt, chosen, rejected (and optionally toxicity_field).",
    )
    ap.add_argument("--model_name", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--eval_max_new_tokens", type=int, default=128)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument(
        "--toxicity_field",
        type=str,
        default="prompt",
        help="Column containing the original toxic text for Detoxify evaluation.",
    )
    args = ap.parse_args()

    train_ds, eval_ds = split_train_eval(args.prefs_jsonl)
    base = args.model_name.split("/")[-1]
    out_dir = base + "-DPO"
    

    os.makedirs(out_dir, exist_ok=True)

    
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else None,
    )
    if torch.cuda.is_available():
        model.to("cuda")

    # DPO training config
    dpo_args = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=args.bf16 and torch.cuda.is_available(),
        max_length=args.max_len,
        max_prompt_length=args.max_len,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        beta=0.05,
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        processing_class=tok,
        train_dataset=train_ds,
        ref_model=None,
    )

    # -------- Train --------
    trainer.train()

    # -------- Save model + tokenizer (HF-style) --------
    trainer.save_model(args.out_dir)       # saves config + pytorch_model.bin
    tok.save_pretrained(args.out_dir)

        # -------- Evaluation on held-out 10% --------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [r["prompt"] for r in eval_ds]
    gens = generate(trainer.model, tok, prompts, args.eval_max_new_tokens, device=device)
    chosens = [r["chosen"] for r in eval_ds]
    rejecteds = [r["rejected"] for r in eval_ds]

    # ---- BERTScore ----
    f_ch, f_re, delta, summary = eval_bertscore(gens, chosens, rejecteds, lang="en")
    print("\n=== BERTScore (held-out 10%) ===")
    for k, v in summary.items():
        # summary has means and counts
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # ---- Detoxify ----
    if args.toxicity_field in eval_ds.column_names:
        original_toxic = [r[args.toxicity_field] for r in eval_ds]
    else:
        original_toxic = prompts

    avg_before, avg_after, tox_before, tox_after = eval_toxicity(original_toxic, gens)

    print("\n=== Detoxify toxicity (held-out 10%) ===")
    print(f"Average toxicity BEFORE (original): {avg_before:.4f}")
    print(f"Average toxicity AFTER  (generated): {avg_after:.4f}")

    # -------- Save eval examples + per-example metrics to CSV --------
    csv_path = os.path.join(args.out_dir, "eval_inference_examples.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "prompt",
                "chosen",
                "rejected",
                "generated",
                "F1_vs_chosen",
                "F1_vs_rejected",
                "delta_F1",
                "tox_before",
                "tox_after",
            ]
        )

        for i, (row, gen, fc, fr, d, tb, ta) in enumerate(
            zip(eval_ds, gens, f_ch, f_re, delta, tox_before, tox_after)
        ):
            writer.writerow(
                [
                    i,
                    row["prompt"],
                    row["chosen"],
                    row["rejected"],
                    gen,
                    float(fc),
                    float(fr),
                    float(d),
                    float(tb),
                    float(ta),
                ]
            )

    print(f"\nSaved eval inference examples to: {csv_path}")

    # -------- Save metrics dict as JSON (averages only) --------
    metrics = {
        "model_name": args.model_name,
        "prefs_jsonl": args.prefs_jsonl,
        "num_train_samples": len(train_ds),
        "num_eval_samples": len(eval_ds),
        "bertscore_summary": summary,  # only aggregate stats
        "toxicity": {
            "avg_before": avg_before,
            "avg_after": avg_after,
        },
        "training_args": {
            "epochs": args.epochs,
            "per_device_batch_size": args.per_device_batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "max_len": args.max_len,
            "eval_max_new_tokens": args.eval_max_new_tokens,
            "bf16": args.bf16,
        },
    }

    metrics_path = os.path.join(out_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved eval metrics dict to: {metrics_path}")



if __name__ == "__main__":
    main()
