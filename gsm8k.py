import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# -----------------------------
# CoT PROMPT
# -----------------------------
def cot_prompt(query, few_shot_examples=None):
    """
    Build a chain-of-thought prompt for GSM8K.
    """
    prompt = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += f"Q: {ex['question']}\nReasoning: {ex['reasoning']}\nAnswer: {ex['answer']}\n\n"
    prompt += f"Q: {query}\nLet's think step by step:"
    return prompt

# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def infer(model, tokenizer, prompt, max_tokens=200, temperature=0.7, top_p = 1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# -----------------------------
# MAIN
# -----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    # Load queries
    with open(args.query_file, "r") as f:
        queries = json.load(f)  # Expecting list of dicts: {"question":..., "answer":...}

    results = []

    # Inference loop
    for q in queries:
        prompt = cot_prompt(q["question"])
        pred_text = infer(model, tokenizer, prompt, max_tokens=args.max_length, temperature=args.temperature)
        pred_answer = pred_text.split("Answer:")[-1].strip() if "Answer:" in pred_text else pred_text.strip()
        results.append({
            "question": q["question"],
            "ground_truth": q["answer"],
            "prediction": pred_answer
        })

    # Evaluation
    for r in results:
        correct = r["prediction"].lower() == r["ground_truth"].lower()
        print(f"Q: {r['question']}")
        print(f"GT: {r['ground_truth']}")
        print(f"Prediction: {r['prediction']} -> {'Correct' if correct else 'Wrong'}")
        print("-"*50)

    # Save results
    with open("gsm8k_cot_results.json", "w") as f:
        json.dump(results, f, indent=2)


# -----------------------------
# ARGPARSE
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference for GSM8K using CoT.")
    parser.add_argument("--model_name", type=str, required=True, help="Huggingface model name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    main(args)
