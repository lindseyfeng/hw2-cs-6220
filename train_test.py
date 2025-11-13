#!/usr/bin/env python3
# train_dpo.py
import argparse, os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from bert_score import score as bertscore
from detoxify import Detoxify   # <<< NEW


def split_train_eval(jsonl_path):
    # HF "json" loader supports JSON Lines files directly
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    n = len(ds)
    if n < 11:
        raise ValueError(f"Need ≥11 rows in {jsonl_path}, got {n}.")
    for k in ("prompt", "chosen", "rejected"):
        if k not in ds.column_names:
            raise ValueError(f"Missing column '{k}' in JSONL.")
    return ds.select(range(0, n - 10)), ds.select(range(n - 10, n))


def generate(model, tok, prompts, max_new_tokens=128, device="cuda"):
    model.eval()
    outs = []
    with torch.no_grad():
        for p in prompts:
            inp = tok(p, return_tensors="pt").to(device)
            gen = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )[0]
            # strip the prompt from the decoded text to get completion
            prompt_len = inp.input_ids.shape[1]
            full = tok.decode(gen, skip_special_tokens=True)
            pref = tok.decode(gen[:prompt_len], skip_special_tokens=True)
            outs.append(full[len(pref) :].strip() or full.strip())
    return outs


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


def eval_toxicity(original_texts, generated_texts, model_name="unbiased"):
    """Compute average toxicity before/after using Detoxify."""  # <<< NEW
    toxic_detector = Detoxify(model_name)  # uses GPU if available

    toxic_scores_before = toxic_detector.predict(original_texts)
    toxic_scores_after = toxic_detector.predict(generated_texts)

    tox_before = toxic_scores_before["toxicity"]
    tox_after = toxic_scores_after["toxicity"]

    avg_before = float(sum(tox_before) / len(tox_before))
    avg_after = float(sum(tox_after) / len(tox_after))

    return avg_before, avg_after, tox_before, tox_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefs_jsonl", required=True, help="JSONL with fields: prompt, chosen, rejected")
    ap.add_argument("--model_name", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--out_dir", default="Qwen2-0.5B-DPO")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--eval_max_new_tokens", type=int, default=128)
    ap.add_argument("--bf16", action="store_true")
    # optional: which field to treat as "original toxic comment"
    ap.add_argument(
        "--toxicity_field",
        type=str,
        default="prompt",   # <<< NEW: change to 'en_toxic_comment' if your JSONL has that
        help="Column containing the original toxic text for Detoxify evaluation.",
    )
    args = ap.parse_args()

    train_ds, eval_ds = split_train_eval(args.prefs_jsonl)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    dpo_args = DPOConfig(
        output_dir=args.out_dir,
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

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        processing_class=tok,
        train_dataset=train_ds,
        ref_model=None,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = [r["prompt"] for r in eval_ds]
    gens = generate(trainer.model, tok, prompts, args.eval_max_new_tokens, device=device)
    chosens = [r["chosen"] for r in eval_ds]
    rejecteds = [r["rejected"] for r in eval_ds]

    # ---------------- BERTScore eval ----------------
    f_ch, f_re, delta, summary = eval_bertscore(gens, chosens, rejecteds, lang="en")
    print("\n=== BERTScore (10 held-out) ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("\nidx | F1(chosen)  F1(rejected)  Δ")
    for i, (c, r, d) in enumerate(zip(f_ch, f_re, delta)):
        print(f"{i:>3} | {c:.4f}       {r:.4f}        {d:.4f}")

    # ---------------- Detoxify eval ----------------
    # Use either the 'prompt' column or a dedicated 'en_toxic_comment' field
    if args.toxicity_field in eval_ds.column_names:       # <<< NEW
        original_toxic = [r[args.toxicity_field] for r in eval_ds]
    else:
        # Fallback: treat prompt as original toxic comment
        original_toxic = prompts

    avg_before, avg_after, tox_before, tox_after = eval_toxicity(original_toxic, gens)

    print("\n=== Detoxify toxicity (held-out) ===")
    print(f"Average toxicity BEFORE (original): {avg_before:.4f}")
    print(f"Average toxicity AFTER  (generated): {avg_after:.4f}")

    # Optional: per-example dump
    print("\nidx | tox_before  tox_after")
    for i, (tb, ta) in enumerate(zip(tox_before, tox_after)):
        print(f"{i:>3} | {tb:.4f}      {ta:.4f}")


if __name__ == "__main__":
    main()
